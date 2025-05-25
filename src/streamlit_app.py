import asyncio
import os
import urllib.parse
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner import get_script_run_ctx

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage
from utils.agentic_alliance_logger import setup_logger

logger = setup_logger(__name__)

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('logs/agentic_alliance.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# A Streamlit app for interacting with the langgraph agent via a simple chat interface.
# The app has three main functions which are all run async:

# - main() - sets up the streamlit app and high level structure
# - draw_messages() - draws a set of chat messages - either replaying existing messages
#   or streaming new ones.
# - handle_feedback() - Draws a feedback widget and records feedback from the user.

# The app heavily uses AgentClient to interact with the agent's FastAPI endpoints.


APP_TITLE = "Agentic Alliance"
APP_ICON = "🛠️"
SIDEBAR_APP_ICON = "🛠️"



async def main() -> None:
    logger.info("Starting Streamlit application")
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={
        # 'Get Help': 'https://www.extremelycoolapp.com/help',
        # 'Report a bug': "https://www.extremelycoolapp.com/bug",
        # 'About': "# This is a header. This is an *extremely* cool app!"
    },
        layout="centered",
        initial_sidebar_state="auto",
    )
    logger.debug("Page configuration set")

    # Hide the streamlit upper-right chrome
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        
            
        </style>
        """,
    )

    # hide_streamlit_style = """
    #         <style>
    #         [data-testid="stToolbar"] {visibility: hidden !important;}
    #         footer {visibility: hidden !important;}
    #         </style>
    #         """
    # st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    if st.get_option("client.toolbarMode") != "viewer":
        logger.debug("Setting toolbar mode to viewer")
        st.set_option("client.toolbarMode", "viewer")
        await asyncio.sleep(0.1)
        st.rerun()

    if "agent_client" not in st.session_state:
        logger.info("Initializing agent client")
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 80)
            agent_url = f"http://{host}:{port}"
            logger.debug(f"Using default agent URL: {agent_url}")
        try:
            with st.spinner("Connecting to agent service..."):
                logger.info(f"Connecting to agent service at {agent_url}")
                st.session_state.agent_client = AgentClient(base_url=agent_url)
                logger.info("Successfully connected to agent service")
        except AgentClientError as e:
            logger.error(f"Failed to connect to agent service: {str(e)}")
            st.error(f"Error connecting to agent service: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    if "thread_id" not in st.session_state:
        logger.info("Initializing thread")
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = get_script_run_ctx().session_id
            logger.debug(f"Generated new thread ID: {thread_id}")
            messages = []
        else:
            logger.info(f"Resuming existing thread: {thread_id}")
            try:
                messages: ChatHistory = agent_client.get_history(thread_id=thread_id).messages
                logger.info(f"Retrieved {len(messages)} messages from history")
            except AgentClientError as e:
                logger.error(f"Failed to retrieve message history: {str(e)}")
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id
        st.session_state.welcome_message = agent_client.agent_intro
        st.session_state.last_message = None
        st.session_state.feedback = None
        logger.debug("Thread initialization complete")

    # Config options
    with st.sidebar:
        logger.debug("Rendering sidebar")
        logo_path = os.path.join(os.path.dirname(__file__), "..", "media", "agentic_alliance_logo_with_title.png")
        
        st.image(logo_path)
        
        #st.header(f"{APP_TITLE}", anchor=False)
        # st.header(APP_TITLE, anchor=False, divider=False)
        # st.markdown("<style>div[data-testid='stHeadingContainer'] {text-align: center;}</style>", unsafe_allow_html=True)
        # st.markdown(f"<h2 style='text-align: center; margin-top: 0;'>{APP_TITLE}</h2>", unsafe_allow_html=True)
        ""
        # st.caption(
        #         "Toolkit for running AI Agent services built with LangGraph, FastAPI and Streamlit"
        #     )
        with st.popover(":material/settings: Settings", use_container_width=True):
            logger.debug("Rendering settings popover")
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("Language Model", options=agent_client.info.models, index=model_idx)
            logger.info(f"Selected model: {model}")
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            agent_client.agent = st.selectbox(
                "Agent",
                options=agent_list,
                index=agent_idx,
            )
            logger.info(f"Selected agent: {agent_client.agent}")
            use_streaming = st.toggle("Streaming", value=True)
            logger.debug(f"Streaming enabled: {use_streaming}")

        @st.dialog("Architecture")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/nishantuzir/agentic-alliance/blob/main/media/agent_architecture.png?raw=true"
            )
            st.caption(
                "Reference: Check out the full sized image in [Full Architecture Diagram](https://github.com/nishantuzir/agentic-alliance/blob/main/media/agent_architecture.png)"
            )

            st.caption(
                "Source Code Reference: Check out the source code in [Agentic Alliance Github](https://github.com/nishantuzir/agentic-alliance)"
            )

        @st.dialog("Agentic Alliance")
        def about_dialog() -> None:

            st.caption(
                "About"
            )


        @st.dialog("Privacy Information")
        def privacy_dialog() -> None:

            st.caption(
                "All Prompts, responses and feedbacks in the application are recorded anonymously and displayed in Langfuse for the purpose of product evaluation and improvement only."
            )


        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        # with st.popover(":material/policy: Privacy", use_container_width=True):
        #     st.write(
        #         "All Prompts, responses and feedbacks in the application are recorded anonymously and displayed in Langfuse for the purpose of product evaluation and improvement only."
        #     )

        if st.button(":material/password: Privacy Information", use_container_width=True):
            privacy_dialog()

        if st.button(":material/info: About Agentic Alliance", use_container_width=True):
            about_dialog()
        

        @st.dialog("Share/Resume Chat")
        def share_chat_dialog() -> None:
            session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
            st_base_url = urllib.parse.urlunparse(
                [session.client.request.protocol, session.client.request.host, "", "", "", ""]
            )
            # if it's not localhost, switch to https by default
            if not st_base_url.startswith("https") and "localhost" not in st_base_url:
                st_base_url = st_base_url.replace("http", "https")
            chat_url = f"{st_base_url}?thread_id={st.session_state.thread_id}"
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share / Resume Chat", use_container_width=True):
            share_chat_dialog()
        
        #st.caption("")

    # Draw existing messages
    messages: list[ChatMessage] = st.session_state.messages

    # Add introduction message if chat is empty
    if not messages:
        intro_msg = ChatMessage(type="ai", content=st.session_state.welcome_message)
        st.session_state.messages.append(intro_msg)
        messages = st.session_state.messages

    # Always display all messages in order
    for m in messages:
        if m.type == "human":
            st.chat_message("human").write(m.content)
        elif m.type == "ai":
            st.chat_message("ai").write(m.content)
        # Optionally handle other types if needed

    # draw_messages() expects an async iterator over messages
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    # Only call draw_messages for initial history (not for new input)
    # await draw_messages(amessage_iter())

    # Generate new message if the user provided new input
    if user_input := st.chat_input("Type your message here...press Enter to send. Press Shift + Enter for a new line."):
        user_msg = ChatMessage(type="human", content=user_input)
        st.session_state.messages.append(user_msg)
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                )
                # Draw and append AI message via streaming
                await draw_messages(stream, is_new=True)
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                )
                st.session_state.messages.append(response)
                st.session_state.feedback = None  # Reset feedback for new AI message
                st.chat_message("ai").write(response.content)
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # Show feedback widget only after the latest AI message (not the intro)
    if (
        st.session_state.messages
        and st.session_state.messages[-1].type == "ai"
        and st.session_state.messages[-1].content != st.session_state.welcome_message
    ):
        if st.session_state.feedback:
            st.success("Thank you for your feedback!")
        else:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    logger.info("Starting to draw messages")
    message_placeholder = st.empty()
    full_response = ""
    ai_message_obj = None
    try:
        async for message in messages_agen:
            if isinstance(message, str):
                logger.debug(f"Received streaming message chunk: {message[:50]}...")
                full_response += message
                message_placeholder.markdown(full_response + "▌")
            else:
                logger.info(f"Received complete message: {message.type}")
                full_response = message.content
                message_placeholder.markdown(full_response)
                if is_new and message.type == "ai":
                    ai_message_obj = message
    except Exception as e:
        logger.error(f"Error while drawing messages: {str(e)}")
        raise
    finally:
        # Ensure the final message is displayed without the cursor
        if full_response:
            message_placeholder.markdown(full_response)
        # Append the AI message to history after streaming is done
        if is_new and ai_message_obj:
            st.session_state.messages.append(ai_message_obj)
            st.session_state.feedback = None  # Reset feedback for new AI message
    logger.info("Finished drawing messages")


async def handle_feedback() -> None:
    logger.info("Handling user feedback")
    try:
        feedback_given = False
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👍", use_container_width=True, key="feedback_pos"):
                st.session_state.feedback = "positive"
                feedback_given = True
                logger.debug("Received positive feedback")
        with col2:
            if st.button("👎", use_container_width=True, key="feedback_neg"):
                st.session_state.feedback = "negative"
                feedback_given = True
                logger.debug("Received negative feedback")
        if feedback_given or st.session_state.feedback:
            st.success("Thank you for your feedback!")
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
