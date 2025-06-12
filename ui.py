"""Render the chatbot user interface for the MissionExplore Demo application.

This module defines the Streamlit-based UI.
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


@st.dialog("Login")
def login_modal():
    """Renders the login modal using Streamlit dialog."""
    with st.form("login_form", enter_to_submit=True):
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter admin password"
        )
        if st.form_submit_button("Login"):
            if password == st.secrets.admin_password:
                st.session_state.admin_logged_in = True
                st.toast("Logged in as admin", icon=":material/check_circle:")
                st.success("Logged in as admin")
                st.rerun()
            else:
                st.error("Incorrect password")


def render_initial_ui() -> None:
    """Renders the initial UI components (sidebar, app title, popover, disabled chat input) before setup."""
    # Apply custom CSS for layout and styling
    st.markdown(
        """
        <style>
        .stAppHeader {
            background-color: rgba(255, 255, 255, 0.0);  /* Transparent background */
            visibility: visible;  /* Ensure the header is visible */
        }
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 0rem;
            padding-right: 0rem;
        }
        .chat-messages {
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 60px;
            margin-top: 60px; /* Space for app bar */
        }
        .stChatInput {
            position: fixed;
            bottom: 10px;
            width: 100%;
            max-width: 720px;
            margin: 0 auto;
            background-color: white;
            z-index: 1000;
        }
        .inline-image {
            margin: 10px 0;
            max-width: 100%;
        }
        .sidebar-title {
            font-size: 14px;
            font-weight: 600;
            color: #666;
            margin-bottom: 20px;
            text-transform: uppercase;
        }
        .app-title {
            font-size: 14px;
            font-weight: 600;
            color: #666;
            margin-top: 20px;
            text-transform: uppercase;
        }
        [data-testid="stPopover"] {
            width: auto !important;
            max-width: 48px !important;
            padding: 4px !important;
        }
        [data-testid="stPopover"] > div {
            width: 48px !important;
        }
        .popover-buttons {
            display: flex;
            flex-direction: column;
            gap: 4px;
            width: 40px;
        }
        .popover-container {
            display: flex;
            justify-content: flex-end;
        }
        </style>
        <script>
        function scrollToBottom() {
            if (window.newMessageAdded) {
                setTimeout(() => {
                    const chatMessages = document.querySelector('.chat-messages');
                    if (chatMessages) {
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                        window.newMessageAdded = false;
                    }
                }, 100); // Delay to ensure DOM is updated
            }
        }
        document.addEventListener('streamlit:render', scrollToBottom);
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Streamlit logo (renders in sidebar and top-left corner)
    st.logo(
        image="mgs-full-logo.svg",
        size="small",
        link="https://www.maxwellgeosystems.com/",
        icon_image="mgs-small-logo.svg"
    )

    # Render sidebar
    with st.sidebar:
        # App title and popover button in two columns
        st.divider()
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown('<div class="app-title">MISSION REFERENCE</div>', unsafe_allow_html=True)

    # Reserve space for chat input
    with st.empty():
        st.chat_input(
            placeholder="Setting up, please wait...",
            disabled=True,
            key="initial_chat_input"
        )


def render_chat_content() -> None:
    """Renders chat messages, history, and enabled chat input after setup is complete."""
    if not st.session_state.setup_complete:
        return
    

    def render_message_content(content):
        st.markdown(content, unsafe_allow_html=True)


    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for msg in st.session_state.get('messages', []):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                render_message_content(msg.content)
    st.markdown("</div>", unsafe_allow_html=True)

    if question := st.chat_input(
        placeholder="Ask a query about project data:",
        key="active_chat_input"
    ):
        user_message = HumanMessage(content=question)
        st.session_state.messages.append(user_message)
        st.session_state.new_message_added = True
        with st.chat_message("user"):
            st.markdown(question)

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id,
                "user_id": st.session_state.user_id
            }
        }
        initial_state = {
            "messages": [user_message],
            "timings": []
        }

        with st.spinner("Generating..."):
            with st.chat_message("assistant"):
                try:
                    def stream_tokens():
                        accumulated_content = ""
                        for chunk, metadata in st.session_state.graph.stream(initial_state, stream_mode="messages", config=config):
                            logging.debug(f"Streamed state: {chunk}")
                            if isinstance(chunk, dict) and chunk.get("type") == "ai" and not chunk.get("additional_kwargs", {}).get("chunk"):
                                content = chunk.get("content", "")
                                if content:
                                    accumulated_content += content
                                    yield content
                            elif hasattr(chunk, "content") and chunk.content:
                                yield chunk.content
                            # state_messages = [
                            #     {
                            #         "type": msg.type,
                            #         "content": msg.content,
                            #         "additional_kwargs": msg.additional_kwargs
                            #     }
                            #     for msg in st.session_state.graph.get_state(config).values.get("messages", [])
                            #     if msg.type in ("human", "ai")
                            # ]
                            # st.session_state.messages = initial_state["messages"] + state_messages
                        final_state = st.session_state.graph.get_state(config).values
                        if final_state:
                            ai_messages = [msg for msg in final_state.get("messages", []) if msg.type == "ai" and not msg.additional_kwargs.get("chunk")]
                            if ai_messages:
                                final_message = ai_messages[-1]
                                if final_message not in st.session_state.messages:
                                    st.session_state.messages.append(final_message)
                        return accumulated_content

                    stream_container = st.empty()
                    with stream_container:
                        stream_container.write_stream(stream_tokens())

                    final_state = st.session_state.graph.get_state(config).values
                    if final_state:
                        ai_messages = [msg for msg in final_state.get("messages", []) if isinstance(msg, AIMessage) and not getattr(msg, "type", "") == "tool_call"]
                        if ai_messages:
                            final_message = ai_messages[-1]

                            stream_container.empty()
                            render_message_content(final_message.content)

                except Exception as e:
                    st.error(f"Error streaming response: {e}")