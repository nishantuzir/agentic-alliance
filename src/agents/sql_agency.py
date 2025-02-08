# Implementation
import warnings
warnings.filterwarnings("ignore") 
# import json
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.graph import END, StateGraph, START, MessagesState
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import tool
from langchain_ollama import ChatOllama
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from typing import Annotated, Dict, Literal, List, Tuple, Union, Any
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableWithFallbacks, RunnableLambda
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.prebuilt import ToolNode, tools_condition
# from IPython.display import Image, display
import uuid
# from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.types import interrupt, Command
# from langgraph.errors import NodeInterrupt
from core import get_model, settings
from langchain_core.language_models.chat_models import BaseChatModel


###############################################################################################
# CUSTOM CLASSES
###############################################################################################

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


###############################################################################################
# LLM CONFIGS
###############################################################################################


config = getattr(RunnableConfig, "configurable", {})
sql_llm =  get_model(config.get("model", settings.DEFAULT_MODEL))

###############################################################################################
# PROMPTS
###############################################################################################

query_generation_prompt_template = """
ROLE: Assume the role of an intelligent tool calling agent.\n
GOAL: Given a user query in natural language as input, call the appropriate tools available to you to 
generate the correct SQL query to address the user query.\n
INSTRUCTIONS:\n
1. Try to get all table names, schema and constraints from database. 
If possible, also get descriptions of tables and columns. Do this first before doing anything else.\n
2. Do not generate and add any text of your own in the output.\n
3. Use only the information returned by the available tools.\n

To generate the correct SQL query, follow the structured process below:

1. Understand the User Query: [Node 1 - Input Analysis]
   - Read the user's natural language query carefully and identify its core requirements.
   - Break down the user query into key components:
     - Operations requested (e.g., counting, filtering, joining, sorting).
     - Target table(s) and columns involved.
     - Specific filters, constraints, or output format requirements (e.g., ranges, distinct values, aggregates).
   - Clarify any ambiguities by hypothesizing logical assumptions based on common database practices.
     - For example, 'If the user requests a 'count of records' without grouping, assume a simple count of rows.'

2. Construct the Graph of Thought: [Node 2 - Structure Mapping]
   - Map out interconnected reasoning paths that lead to constructing the SQL query:
     - Each node represents a decision point (e.g., table selection, column selection, filter criteria, aggregation, row limits, etc.).
     - Edges represent logical dependencies between these decisions (e.g., selecting columns depends on the target table).
   - Explore different paths to achieve the query's goal, considering multiple ways to write the SQL query.
   - Ensure all components, such as joins, groupings, and conditions, are represented in the graph.

3. Evaluate and Refine the Graph: [Node 3 - Evaluation & Refinement]
   - Assess the correctness of each node, its alignment with the user query, and SQL best practices.
   - Discard invalid nodes or paths that do not meet the criteria (e.g., incorrect join types, unnecessary subqueries, etc.).
   - Refine the paths, focusing on reducing redundancy and ensuring the query is optimal (e.g., no unnecessary `DISTINCT` or excessive subqueries).

4. Identify the Optimal Path in the Graph: [Node 4 - Optimal Path Selection]
   - After evaluating the graph, identify the most suitable reasoning path that fully satisfies the user query.
   - The optimal path should:
     - Address all operations required by the user.
     - Maintain logical integrity (e.g., correct SQL syntax, proper use of aggregate functions, etc.).
     - Avoid unnecessary complexity (e.g., extraneous joins, invalid filters).

5. Generate the Output: [Node 5 - Final Query Generation]
   - Enclose the generate SQL query within '''sql '''.
   - Ensure the following in your output:
     - Do not add additional text in the output.
     - Add comments where applicable.
    
Important Notes:
   - Never fabricate or assume details about query if information provided is insufficient. 
   - If information insufficient, say information is insufficient to generate SQL but also strictly mention what information is required.
   - Ensure the SQL query is well-formed and fully aligned with the user's intent.
   - The SQL query generated must be correct and free of redundant operations.
   - Return only the final SQL query as output.
"""

query_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", query_generation_prompt_template), 
    ("placeholder", "{messages}")
])

#----------------------------------------------------------------------------------------------

query_validation_prompt_template = (
    "ROLE: Assume the role of an intelligent tool calling agent.\n"
    "GOAL: Given a SQL query as input, call the appropriate tools available to you to check and validate the SQL query.\n"

    "To achieve your goal, follow the structured process below:\n"

    "\nValidation Steps:\n"
    "1. Verify clause order (SELECT → FROM → WHERE, etc.), balanced parentheses, "
    "and valid column/table names and aliases (Syntax Check).\n"
    "2. Verify that the query logic matches the user intent, including joins, filters, grouping, and sorting. "
    "Flag missing or extraneous elements (Semantic Match Check).\n"
    "3. Validate join conditions, GROUP BY rules, subqueries, and set operations for logical consistency (Structural Check).\n"
    "4. Check for NULL handling, type compatibility, and proper use of expressions like IS NULL and CAST (Data Handling Check).\n"
    "5. Verify that the query includes only the requested fields and operations, matches aggregation levels, and respects table references (Context Alignment Check).\n"
    "6. Confirm no prohibited operations (e.g., DROP, DELETE) are present and unnecessary deduplication is avoided (Restrictions Check).\n"
    #"7. Based on information from tools, ensure all column names and table names used are correct (Spelling and Existence Check).\n"

    "\nExpected Output:\n"
    "- ONLY output those mistakes/errors which you are sure about. DO NOT include suggestive texts."
    "- If mistakes/errors are found, do the following:\n"
    "    1. Provide a numbered list containing only the failed checks with concise descriptions of mistakes/errors.\n" 
    "    2. Output the sql query within '''sql ''' specifiers. Strictly, DO NOT include any other text.\n"
    "- Else if no mistakes/errors are found, do the following:\n"
    "        1. Provide a simple text saying that there were no mistakes found in the generated SQL query.\n"
    "        2. Output the sql query, within '''sql ''' specifiers. Strictly, DO NOT include any other text.\n"
    "- In general, do not include any special characters in the output, unless it is required for the SQL query.\n"
    "- Include only descriptions of failed checks in the mistakes.\n"
    
    "\nGuidelines:\n"
    "- Systematic Exploration: Assess all reasoning branches thoroughly.\n"
    "- Justifications: Document reasoning for each validation step.\n"
    "- No Assumptions: Avoid assumptions; Use only the information returned by the available tools.\n"
)
query_validation_prompt = ChatPromptTemplate.from_messages(
    [("system", query_validation_prompt_template), 
     ("placeholder", "{messages}")]
)


###############################################################################################
# DB CONFIGS
###############################################################################################

def get_engine_for_postgres_db():
    return create_engine(settings.CDM_POSTGRES_DB_URL)

###############################################################################################
# DB INIT
###############################################################################################

postgres_engine = get_engine_for_postgres_db()
postgres_db = SQLDatabase(postgres_engine)

###############################################################################################

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

#----------------------------------------------------------------------------------------------

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tool_call["id"],
            )
            for tool_call in tool_calls
        ]
    }

###############################################################################################
# TOOLS
###############################################################################################

toolkit = SQLDatabaseToolkit(db=postgres_db, llm=sql_llm)
tools = toolkit.get_tools()

###############################################################################################

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

@tool
def db_query_tool(sql_query: str) -> str:
    """
    Execute a SQL query, provided as input, against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, display the full error, without modifying the error
    """
    result = postgres_db.run_no_throw(sql_query)
    if not result:
        return "Error: Query failed. Please try again."
    return result


###############################################################################################

table_retrieving_tool_node = create_tool_node_with_fallback([list_tables_tool])

schema_retrieving_tool_node = create_tool_node_with_fallback([get_schema_tool])

query_execution_tool_node = create_tool_node_with_fallback([db_query_tool])

###############################################################################################

schema_retrieving_callable = sql_llm.bind_tools([get_schema_tool])

#----------------------------------------------------------------------------------------------

query_generation_callable = query_generation_prompt | sql_llm

#----------------------------------------------------------------------------------------------

query_validation_callable = query_validation_prompt | sql_llm

#----------------------------------------------------------------------------------------------

query_execution_callable = sql_llm.bind_tools([db_query_tool])

#----------------------------------------------------------------------------------------------

###############################################################################################
# AGENTS
###############################################################################################

# accepts AIMessage, hence the input to the callable should have been {"messages": [state["messages"]][-1]};
# instead the input to callable seems to be {"messages": [state["messages"]]}, because there is only 1 message in the state
# additionally, there is no invokation of callable at all, simply because it is just a cue, albeit forceful,
# for the tool executor to execute the tool in the next step.
def table_retrieving_agent(state: State) -> dict[str, list[AIMessage]]:
    messages = AIMessage(content="",tool_calls=[{"name": "sql_db_list_tables", "args": {}, "id": str(uuid.uuid4()),}],)
    return {"messages": [messages]}

#----------------------------------------------------------------------------------------------

# accepts ToolMessage, hence the input is state["messages"]
def schema_retrieving_agent(state: State) -> dict[str, list[AIMessage]]:
    messages = schema_retrieving_callable.invoke(state["messages"])
    return {"messages": [messages]}

#----------------------------------------------------------------------------------------------

# accepts AIMessage, hence the input to callable is {"messages": [state["messages"]][-1]}; 
# the entire dictionary is passed as input because the prompt has {messages} as "placeholder";
# also [state["messages"]][-1] denotes the last or the latest message added to the state i.e., previous message
def query_generation_agent(state: State) -> dict[str, list[AIMessage]]:
    messages = query_generation_callable.invoke({"messages": [state["messages"]][-1]})
    return {"messages": [messages]}

#----------------------------------------------------------------------------------------------

# accepts AIMessage, hence the input to callable is {"messages": [state["messages"]][-1]}; 
# the entire dictionary is passed as input because the prompt has {messages} as "placeholder";
# also [state["messages"]][-1] denotes the last or the latest message added to the state i.e., previous message
def query_validation_agent(state: State) -> dict[str, list[AIMessage]]:
    messages = query_validation_callable.invoke({"messages": [state["messages"]][-1]})
    return {"messages": [messages]}

#----------------------------------------------------------------------------------------------

# accepts ToolMessage, hence the input to callable is state["messages"]
def query_execution_agent(state: State) -> dict[str, list[AIMessage]]:
    messages = query_execution_callable.invoke(state["messages"])
    return {"messages": [messages]}

###############################################################################################

sql_agency_workflow = StateGraph(State)

sql_agency_workflow.add_node("context_retrieval_agent", table_retrieving_agent)
sql_agency_workflow.add_node("table_retrieval_tool_executor", table_retrieving_tool_node)
sql_agency_workflow.add_node("schema_retrieval_tool_executor", schema_retrieving_tool_node)
sql_agency_workflow.add_node("schema_retrieving_agent", schema_retrieving_agent)
sql_agency_workflow.add_node("query_generation_agent", query_generation_agent)
sql_agency_workflow.add_node("query_validation_agent", query_validation_agent)
sql_agency_workflow.add_node("query_execution_agent", query_execution_agent)
sql_agency_workflow.add_node("query_execution_tool_executor", query_execution_tool_node)


#workflow.add_edge(START, "context_retrieval_agent")
sql_agency_workflow.add_edge("context_retrieval_agent", "table_retrieval_tool_executor")
sql_agency_workflow.add_edge("table_retrieval_tool_executor", "schema_retrieving_agent")
sql_agency_workflow.add_edge("schema_retrieving_agent", "schema_retrieval_tool_executor")
sql_agency_workflow.add_edge("schema_retrieval_tool_executor", "query_generation_agent")
sql_agency_workflow.add_edge("query_generation_agent", "query_validation_agent")
sql_agency_workflow.add_edge("query_validation_agent", "query_execution_agent")
sql_agency_workflow.add_edge("query_execution_agent", "query_execution_tool_executor")
sql_agency_workflow.add_edge("query_execution_tool_executor", END)

sql_agency_workflow.set_entry_point("context_retrieval_agent")

sql_agency_memory =  MemorySaver()

sql_agency = sql_agency_workflow.compile(checkpointer = sql_agency_memory)