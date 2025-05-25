from datetime import datetime
from typing import Literal

from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import calculator
from core import get_model, settings
from utils.agentic_alliance_logger import setup_logger

logger = setup_logger(__name__)

class AgentState(MessagesState, total=False):
    """`total=False` is PEP589 specs.

    documentation: https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """

    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps

logger.info("Initializing research assistant tools")
web_search = DuckDuckGoSearchResults(name="WebSearch")
tools = [web_search, calculator]
logger.debug("Added web search and calculator tools")

# Add weather tool if API key is set
# Register for an API key at https://openweathermap.org/api/
if settings.OPENWEATHERMAP_API_KEY:
    logger.info("Adding weather tool")
    wrapper = OpenWeatherMapAPIWrapper(
        openweathermap_api_key=settings.OPENWEATHERMAP_API_KEY.get_secret_value()
    )
    tools.append(OpenWeatherMapQueryRun(name="Weather", api_wrapper=wrapper))
    logger.debug("Weather tool added successfully")

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful research assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use calculator tool with numexpr to answer math questions. The user does not understand numexpr,
      so for the final response, use human readable format - e.g. "300 * 200", not "(300 \\times 200)".
    """

def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    logger.debug("Wrapping model with tools and instructions")
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

def format_safety_message(safety: LlamaGuardOutput) -> AIMessage:
    logger.warning(f"Formatting safety message for unsafe content: {safety.unsafe_categories}")
    content = (
        f"This conversation was flagged for unsafe content: {', '.join(safety.unsafe_categories)}"
    )
    return AIMessage(content=content)

async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info("Calling model with current state")
    model_name = config["configurable"].get("model", settings.DEFAULT_MODEL)
    logger.debug(f"Using model: {model_name}")
    m = get_model(model_name)
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)
    logger.debug("Model response received")

    # Run llama guard check here to avoid returning the message if it's unsafe
    logger.debug("Running LlamaGuard safety check")
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("Agent", state["messages"] + [response])
    if safety_output.safety_assessment == SafetyAssessment.UNSAFE:
        logger.warning("Unsafe content detected in model response")
        return {"messages": [format_safety_message(safety_output)], "safety": safety_output}

    if state["remaining_steps"] < 2 and response.tool_calls:
        logger.warning("Insufficient steps remaining for tool calls")
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, need more steps to process this request.",
                )
            ]
        }
    logger.debug("Returning model response")
    return {"messages": [response]}

async def llama_guard_input(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.info("Running LlamaGuard input check")
    llama_guard = LlamaGuard()
    safety_output = await llama_guard.ainvoke("User", state["messages"])
    logger.debug(f"LlamaGuard assessment: {safety_output.safety_assessment}")
    return {"safety": safety_output}

async def block_unsafe_content(state: AgentState, config: RunnableConfig) -> AgentState:
    logger.warning("Blocking unsafe content")
    safety: LlamaGuardOutput = state["safety"]
    return {"messages": [format_safety_message(safety)]}

# Define the graph
logger.info("Building research assistant state graph")
agent = StateGraph(AgentState)
agent.add_node("model", acall_model)
agent.add_node("tools", ToolNode(tools))
agent.add_node("guard_input", llama_guard_input)
agent.add_node("block_unsafe_content", block_unsafe_content)
agent.set_entry_point("guard_input")
logger.debug("Basic graph structure created")

# Check for unsafe input and block further processing if found
def check_safety(state: AgentState) -> Literal["unsafe", "safe"]:
    safety: LlamaGuardOutput = state["safety"]
    logger.debug(f"Checking safety: {safety.safety_assessment}")
    match safety.safety_assessment:
        case SafetyAssessment.UNSAFE:
            return "unsafe"
        case _:
            return "safe"

logger.debug("Adding conditional edges for safety checks")
agent.add_conditional_edges(
    "guard_input", check_safety, {"unsafe": "block_unsafe_content", "safe": "model"}
)

# Always END after blocking unsafe content
agent.add_edge("block_unsafe_content", END)

# Always run "model" after "tools"
agent.add_edge("tools", "model")

# After "model", if there are tool calls, run "tools". Otherwise END.
def pending_tool_calls(state: AgentState) -> Literal["tools", "done"]:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        logger.error(f"Expected AIMessage, got {type(last_message)}")
        raise TypeError(f"Expected AIMessage, got {type(last_message)}")
    if last_message.tool_calls:
        logger.debug("Tool calls pending")
        return "tools"
    logger.debug("No tool calls pending")
    return "done"

logger.debug("Adding conditional edges for tool calls")
agent.add_conditional_edges("model", pending_tool_calls, {"tools": "tools", "done": END})

logger.info("Compiling research assistant graph")
research_assistant = agent.compile(checkpointer=MemorySaver())
logger.info("Research assistant initialization complete")
