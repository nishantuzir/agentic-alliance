import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    ServiceMetadata,
    StreamInput,
    UserInput,
)
from utils.agentic_alliance_logger import setup_logger

logger = setup_logger(__name__)

class AgentClientError(Exception):
    pass

class AgentClient:
    """Client for interacting with the agent service."""

    def __init__(
        self,
        base_url: str = "http://localhost",
        agent: str = None,
        timeout: float | None = None,
        get_info: bool = True,
    ) -> None:
        logger.info(f"Initializing AgentClient with base_url: {base_url}")
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        self.agent_intro: str | None = None
        if get_info:
            logger.debug("Fetching initial service info")
            self.retrieve_info()
        if agent:
            logger.debug(f"Setting initial agent: {agent}")
            self.update_agent(agent)

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def retrieve_info(self) -> None:
        logger.info("Retrieving service information")
        try:
            response = httpx.get(
                f"{self.base_url}/info",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.debug("Successfully retrieved service info")
        except httpx.HTTPError as e:
            logger.error(f"Failed to retrieve service info: {str(e)}")
            raise AgentClientError(f"Error getting service info: {e}")

        self.info: ServiceMetadata = ServiceMetadata.model_validate(response.json())
        if not self.agent or self.agent not in [a.key for a in self.info.agents]:
            logger.debug(f"Setting default agent: {self.info.default_agent}")
            self.agent = self.info.default_agent
            self.agent_intro = self.info.default_agent_intro

    def update_agent(self, agent: str, verify: bool = True) -> None:
        logger.info(f"Updating agent to: {agent}")
        if verify:
            if not self.info:
                logger.debug("Service info not available, retrieving...")
                self.retrieve_info()
            agent_keys = [a.key for a in self.info.agents]
            if agent not in agent_keys:
                logger.error(f"Agent {agent} not found in available agents: {', '.join(agent_keys)}")
                raise AgentClientError(
                    f"Agent {agent} not found in available agents: {', '.join(agent_keys)}"
                )
        self.agent = agent
        self.agent_intro == next(a.description for a in self.info.agents if a.key == agent)
        logger.debug(f"Agent updated successfully. Intro: {self.agent_intro}")

    async def ainvoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        logger.info(f"Invoking agent asynchronously with message: {message[:50]}...")
        if not self.agent:
            logger.error("No agent selected")
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
            logger.debug(f"Using thread_id: {thread_id}")
        if model:
            request.model = model
            logger.debug(f"Using model: {model}")
        if agent_config:
            request.agent_config = agent_config
            logger.debug(f"Using agent config: {agent_config}")

        async with httpx.AsyncClient() as client:
            try:
                logger.debug(f"Sending request to {self.base_url}/{self.agent}/invoke")
                response = await client.post(
                    f"{self.base_url}/{self.agent}/invoke",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                logger.debug("Request successful")
            except httpx.HTTPError as e:
                logger.error(f"Request failed: {str(e)}")
                raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        logger.info(f"Invoking agent synchronously with message: {message[:50]}...")
        if not self.agent:
            logger.error("No agent selected")
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
            logger.debug(f"Using thread_id: {thread_id}")
        if model:
            request.model = model
            logger.debug(f"Using model: {model}")
        if agent_config:
            request.agent_config = agent_config
            logger.debug(f"Using agent config: {agent_config}")

        try:
            logger.debug(f"Sending request to {self.base_url}/{self.agent}/invoke")
            response = httpx.post(
                f"{self.base_url}/{self.agent}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.debug("Request successful")
        except httpx.HTTPError as e:
            logger.error(f"Request failed: {str(e)}")
            raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                logger.debug("Received stream completion signal")
                return None
            try:
                parsed = json.loads(data)
                logger.debug(f"Parsed stream data: {parsed['type']}")
            except Exception as e:
                logger.error(f"Failed to parse stream data: {str(e)}")
                raise Exception(f"Error JSON parsing message from server: {e}")
            
            match parsed["type"]:
                case "message":
                    try:
                        message = ChatMessage.model_validate(parsed["content"])
                        logger.debug(f"Parsed message: {message.type}")
                        return message
                    except Exception as e:
                        logger.error(f"Failed to validate message: {str(e)}")
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    logger.debug("Received token")
                    return parsed["content"]
                case "error":
                    logger.error(f"Stream error: {parsed['content']}")
                    raise Exception(parsed["content"])
        return None

    def stream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str, None, None]:
        """
        Stream the agent's response synchronously.

        Each intermediate message of the agent process is yielded as a ChatMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming models as they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            Generator[ChatMessage | str, None, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model
        if agent_config:
            request.agent_config = agent_config
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

    async def astream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """
        Stream the agent's response asynchronously.

        Each intermediate message of the agent process is yielded as an AnyMessage.
        If stream_tokens is True (the default value), the response will also yield
        content tokens from streaming modelsas they are generated.

        Args:
            message (str): The message to send to the agent
            model (str, optional): LLM model to use for the agent
            thread_id (str, optional): Thread ID for continuing a conversation
            agent_config (dict[str, Any], optional): Additional configuration to pass through to the agent
            stream_tokens (bool, optional): Stream tokens as they are generated
                Default: True

        Returns:
            AsyncGenerator[ChatMessage | str, None]: The response from the agent
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model
        if agent_config:
            request.agent_config = agent_config
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/{self.agent}/stream",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            parsed = self._parse_stream_line(line)
                            if parsed is None:
                                break
                            yield parsed
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, kwargs: dict[str, Any] = {}
    ) -> None:
        logger.info(f"Creating feedback for run_id: {run_id}")
        feedback = Feedback(
            run_id=run_id,
            key=key,
            score=score,
            kwargs=kwargs,
        )
        async with httpx.AsyncClient() as client:
            try:
                logger.debug("Sending feedback request")
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=feedback.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                logger.info("Feedback created successfully")
            except httpx.HTTPError as e:
                logger.error(f"Failed to create feedback: {str(e)}")
                raise AgentClientError(f"Error creating feedback: {e}")

    def get_history(
        self,
        thread_id: str | None = None,
    ) -> ChatHistory:
        logger.info(f"Getting chat history for thread_id: {thread_id}")
        try:
            logger.debug("Sending history request")
            response = httpx.get(
                f"{self.base_url}/history",
                params={"thread_id": thread_id} if thread_id else None,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.debug("History retrieved successfully")
        except httpx.HTTPError as e:
            logger.error(f"Failed to get history: {str(e)}")
            raise AgentClientError(f"Error getting history: {e}")

        return ChatHistory.model_validate(response.json())
