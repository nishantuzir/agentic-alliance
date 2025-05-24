from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from agents.background_task_agent.background_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.command_agent import command_agent
from agents.email_classification_agent import email_classification_agent
from agents.research_assistant import research_assistant
from schema import AgentInfo

DEFAULT_AGENT = "Chat"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "Chat": Agent(description="Hello! I'm an AI-powered chat application.", graph=chatbot),
    "Research Assistant": Agent(description="Hello! I'm a AI-powered research assistant. I have access to web search tool and calculator tool.", graph=research_assistant),
    "Command Agent": Agent(description="Hello! I'm a command agent.", graph=command_agent),
    "Background Task Agent": Agent(description="Hello! I am an AI-powered agent that can run tasks in the backgrond.", graph=bg_task_agent),
    "Email Classification Agent": Agent(description="Hello! I'm an AI-powered assistant. I can convert your emails written, in natural language, to json structures.", graph=email_classification_agent),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
