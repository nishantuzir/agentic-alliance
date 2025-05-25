import json
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse as LangfuseClient
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph

from agents import DEFAULT_AGENT, get_agent, get_all_agent_info
from core import settings
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)
from utils.agentic_alliance_logger import setup_logger

logger = setup_logger(__name__)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

# Initialize Langfuse client if tracing is enabled
langfuse_client = None
langfuse_handler = None

if settings.TRACING and settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
    try:
        logger.info("Initializing Langfuse tracing")
        langfuse_client = LangfuseClient(
            secret_key=settings.LANGFUSE_SECRET_KEY,
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            host=settings.LANGFUSE_HOST
        )
        langfuse_handler = LangfuseCallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST
        )
        logger.info("Langfuse tracing initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse tracing: {e}")
        langfuse_client = None
        langfuse_handler = None

def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    logger.debug("Verifying bearer token")
    if not settings.AUTH_SECRET:
        logger.debug("No AUTH_SECRET configured, skipping verification")
        return
    auth_secret = settings.AUTH_SECRET.get_secret_value()
    if not http_auth or http_auth.credentials != auth_secret:
        logger.warning("Invalid authentication attempt")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    logger.debug("Bearer token verified successfully")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting application lifespan")
    # Construct agent with Sqlite checkpointer
    # TODO: It's probably dangerous to share the same checkpointer on multiple agents
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        logger.debug("Initializing agents with Sqlite checkpointer")
        agents = get_all_agent_info()
        for a in agents:
            logger.debug(f"Setting up agent: {a.key}")
            agent = get_agent(a.key)
            agent.checkpointer = saver
        logger.info("All agents initialized successfully")
        yield
    logger.info("Application lifespan ended")

app = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=[Depends(verify_bearer)])

@router.get("/info")
async def info() -> ServiceMetadata:
    logger.info("Retrieving service metadata")
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    metadata = ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=settings.DEFAULT_AGENT,
        default_agent_intro=settings.DEFAULT_AGENT_INTRO,
        default_model=settings.DEFAULT_MODEL,
    )
    logger.debug(f"Service metadata: {metadata}")
    return metadata

def _parse_input(user_input: UserInput) -> tuple[dict[str, Any], UUID]:
    logger.debug(f"Parsing user input: {user_input}")
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())
    logger.debug(f"Generated run_id: {run_id}, thread_id: {thread_id}")

    configurable = {"thread_id": thread_id, "model": user_input.model}

    if user_input.agent_config:
        if overlap := configurable.keys() & user_input.agent_config.keys():
            logger.error(f"Reserved keys found in agent_config: {overlap}")
            raise HTTPException(
                status_code=422, detail=f"agent_config contains reserved keys: {overlap}"
            )
        configurable.update(user_input.agent_config)
        logger.debug(f"Updated configurable with agent_config: {configurable}")

    callbacks = []
    if langfuse_handler:
        logger.debug("Adding Langfuse callback handler")
        callbacks.append(langfuse_handler)

    kwargs = {
        "input": {"messages": [HumanMessage(content=user_input.message)]},
        "config": RunnableConfig(
            configurable=configurable,
            run_id=run_id,
            callbacks=callbacks,
        ),
    }
    logger.debug(f"Parsed input kwargs: {kwargs}")
    return kwargs, run_id

@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    logger.info(f"Invoking agent {agent_id}")
    agent: CompiledStateGraph = get_agent(agent_id)
    kwargs, run_id = _parse_input(user_input)
    try:
        logger.debug(f"Invoking agent with run_id: {run_id}")
        response = await agent.ainvoke(config=RunnableConfig(
            run_id=run_id,
        ),**kwargs)
        output = langchain_to_chat_message(response["messages"][-1])
        output.run_id = str(run_id)
        logger.info(f"Agent invocation successful, run_id: {run_id}")
        return output
    except Exception as e:
        logger.error(f"Agent invocation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Unexpected error")

async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    logger.info(f"Starting message stream for agent {agent_id}")
    agent: CompiledStateGraph = get_agent(agent_id)
    kwargs, run_id = _parse_input(user_input)
    logger.debug(f"Stream initialized with run_id: {run_id}")

    # Process streamed events from the graph and yield messages over the SSE stream.
    async for event in agent.astream_events(**kwargs, version="v2"):
        if not event:
            continue

        new_messages = []
        # Yield messages written to the graph state after node execution finishes.
        if (
            event["event"] == "on_chain_end"
            # on_chain_end gets called a bunch of times in a graph execution
            # This filters out everything except for "graph node finished"
            and any(t.startswith("graph:step:") for t in event.get("tags", []))
            and "messages" in event["data"]["output"]
        ):
            logger.debug("Processing on_chain_end event")
            new_messages = event["data"]["output"]["messages"]

        # Also yield intermediate messages from agents.utils.CustomData.adispatch().
        if event["event"] == "on_custom_event" and "custom_data_dispatch" in event.get("tags", []):
            logger.debug("Processing custom event")
            new_messages = [event["data"]]

        for message in new_messages:
            try:
                chat_message = langchain_to_chat_message(message)
                chat_message.run_id = str(run_id)
                logger.debug(f"Processed message: {chat_message.type}")
            except Exception as e:
                logger.error(f"Error parsing message: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                continue
            # LangGraph re-sends the input message, which feels weird, so drop it
            if chat_message.type == "human" and chat_message.content == user_input.message:
                logger.debug("Skipping duplicate human message")
                continue
            yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

        # Yield tokens streamed from LLMs.
        if (
            event["event"] == "on_chat_model_stream"
            and user_input.stream_tokens
            and "llama_guard" not in event.get("tags", [])
        ):
            content = remove_tool_calls(event["data"]["chunk"].content)
            if content:
                # Empty content in the context of OpenAI usually means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content.
                yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
            continue

    yield "data: [DONE]\n\n"


def _sse_response_example() -> dict[int, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post("/{agent_id}/stream", response_class=StreamingResponse, responses=_sse_response_example())
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    logger.info(f"Processing feedback for run_id: {feedback.run_id}")
    if not langfuse_client:
        logger.warning("Langfuse client not initialized, skipping feedback")
        return FeedbackResponse(success=False, error="Tracing not enabled")
    
    try:
        logger.debug(f"Creating feedback with score: {feedback.score}")
        langfuse_client.score(
            run_id=feedback.run_id,
            key=feedback.key,
            score=feedback.score,
            **feedback.kwargs,
        )
        logger.info("Feedback recorded successfully")
        return FeedbackResponse(success=True)
    except Exception as e:
        logger.error(f"Failed to record feedback: {str(e)}")
        return FeedbackResponse(success=False, error=str(e))


@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:
    logger.info(f"Retrieving chat history for thread_id: {input.thread_id}")
    try:
        # TODO: Hard-coding DEFAULT_AGENT here is wonky
        agent: CompiledStateGraph = get_agent(DEFAULT_AGENT)
        state_snapshot = agent.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        logger.debug("Chat history retrieved successfully")
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"Failed to retrieve chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    logger.debug("Health check requested")
    return {"status": "healthy"}


app.include_router(router)
