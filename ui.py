"""Render the chatbot user interface for the MissionExplore Demo application.

This module defines the Streamlit-based UI.
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from parameters import users
import setup
import graph
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


@st.dialog("Change User")
def user_modal():
    """Renders the user selection modal using Streamlit dialog."""
    with st.form("user_form", enter_to_submit=False):
        display_names = [user['display_name'] for user in users]
        
        selected_user = next((user for user in users 
                            if user['id'] == st.session_state.get('selected_user_id', None)), None)
        default_index = display_names.index(selected_user['display_name']) if selected_user else 0
        
        selected_display_name = st.selectbox(
            "Change User",
            display_names,
            key="user_select",
            label_visibility="visible",
            index=default_index
        )
        
        if st.form_submit_button("Select"):
            if selected_display_name:
                selected_user = next(user for user in users 
                                   if user['display_name'] == selected_display_name)
                st.session_state.selected_user_id = selected_user['id']
                st.session_state.user_permissions = setup.get_user_permissions()
                st.session_state.graph = graph.build_graph(
                    llm=st.session_state.llm,
                    db=st.session_state.db,
                    user_permissions=st.session_state.user_permissions,
                    table_relationship_graph=st.session_state.table_relationship_graph,
                    user_id=st.session_state.selected_user_id
                )
                logging.debug(f'Changed user permissions to {st.session_state.user_permissions}')
                st.rerun()
            else:
                st.error("Please select a user")


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
        .intermediate-step {
            background-color: #f5f5f5;
            border-left: 4px solid #ccc;
            padding: 10px;
            margin: 5px 0;
            font-size: 0.9em;
            color: #666;
        }
        .intermediate-step code {
            background-color: #e8e8e8;
            padding: 2px 4px;
            border-radius: 3px;
        }
        [data-testid="stExpander"] > details,
        [data-testid="stExpander"] > details > summary,
        [data-testid="stExpander"] > details > div {
            background-color: #FFE4CC;
            border: none !important;
            box-shadow: none !important;
            border-radius: 5px;
            padding: 10px;
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
            st.markdown('<div class="app-title">MISSION EXPLORE</div>', unsafe_allow_html=True)
        with cols[1]:
            with st.popover(
                label="",
                icon=":material/menu:",
                use_container_width=False
            ):
                st.button(label="Switch User", icon=":material/account_circle:", key="user_button", help="Change user", on_click=user_modal, use_container_width=True)

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
    

    def render_message_content(content, msg_type=None):
        if msg_type in ["action", "step"]:
            st.markdown(f'<div class="intermediate-step">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(content, unsafe_allow_html=True)


    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    current_query_steps = []
    query_index = 0

    for msg in st.session_state.get('messages', []):
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
            if query_index < len(st.session_state.intermediate_steps_history):
                with st.expander(f"Intermediate Steps for Query {query_index + 1}", expanded=False):
                    for step in st.session_state.intermediate_steps_history[query_index]:
                        st.markdown(f'<div class="intermediate-step">{step}</div>', unsafe_allow_html=True)
            query_index += 1
        elif isinstance(msg, AIMessage) and msg.additional_kwargs.get("type") == "output":
            with st.chat_message("assistant"):
                render_message_content(msg.content, msg.additional_kwargs.get("type"))
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

        # Truncate history to prevent excessive memory usage
        MAX_HISTORY = 1000
        if len(st.session_state.messages) > MAX_HISTORY:
            st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]

        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id,
                "user_id": st.session_state.user_id
            }
        }
        initial_state = {
            "messages": st.session_state.messages,
            "timings": []
        }

        with st.spinner("Generating..."):
            with st.chat_message("assistant"):
                with st.expander(f"Intermediate Steps for Query {query_index + 1}", expanded=True):
                    thinking_container = st.empty()
                response_container = st.empty()

                thinking_text = ""
                thinking_buffer = ""
                final_response_chunks = []
                last_non_output_index = -1
                all_chunks = []
                last_msg_type = None

                for i, chunk in enumerate(st.session_state.graph.stream(initial_state, stream_mode="messages", config=config)):
                    message, metadata = chunk if isinstance(chunk, tuple) else (chunk, {})
                    content = message.content if hasattr(message, 'content') else str(message)
                    msg_type = message.additional_kwargs.get("type", "undefined") if hasattr(message, 'additional_kwargs') else "undefined"
                    all_chunks.append((content, msg_type))
                    logging.debug(f'Received message in ui.py: {message}')

                    if msg_type in ["action", "step"]:
                        # Close previous CoT div if open
                        if last_msg_type == "output":
                            thinking_buffer += "</div>\n"
                            thinking_text += thinking_buffer
                            thinking_container.markdown(thinking_text, unsafe_allow_html=True)
                            thinking_buffer = ""
                        # Add complete div for action or step
                        thinking_buffer += f'<div class="intermediate-step">{content}</div>\n'
                        thinking_text += thinking_buffer
                        thinking_container.markdown(thinking_text, unsafe_allow_html=True)
                        thinking_buffer = ""
                        last_non_output_index = i
                    elif msg_type == "output":
                        # Start new CoT div only if not continuing an output
                        if last_msg_type != "output":
                            thinking_buffer += f'<div class="intermediate-step"><strong>Chain-of-Thought: </strong>{content}'
                        else:
                            # Append to existing CoT div
                            thinking_buffer += content
                    else:
                        # Handle undefined or other types
                        if last_msg_type == "output":
                            thinking_buffer += "</div>\n"
                            thinking_text += thinking_buffer
                            thinking_container.markdown(thinking_text, unsafe_allow_html=True)
                            thinking_buffer = ""
                        thinking_buffer += f'<div class="intermediate-step">{content}</div>\n'
                        thinking_text += thinking_buffer
                        thinking_container.markdown(thinking_text, unsafe_allow_html=True)
                        thinking_buffer = ""
                    current_query_steps.append(content)
                    last_msg_type = msg_type
                    logging.debug(f'Thinking buffer: {thinking_buffer}')
                    st.session_state.messages.append(
                        AIMessage(content=content, additional_kwargs={"type": msg_type})
                    )

                # Flush any remaining buffered content
                if thinking_buffer:
                    if last_msg_type == "output":
                        thinking_buffer += "</div>\n"
                    thinking_text += thinking_buffer
                    thinking_container.markdown(thinking_text, unsafe_allow_html=True)

                for i, (content, msg_type) in enumerate(all_chunks):
                    if i > last_non_output_index and msg_type == "output":
                        final_response_chunks.append(content)

                #     if msg_type in ["action", "step"]:
                #         # Close previous CoT div if open
                #         if last_msg_type == "output":
                #             thinking_text += "</div>\n"
                #         # Start new div for action or step
                #         thinking_text += f'<div class="intermediate-step">{content}</div>\n'
                #         last_non_output_index = i
                #     elif msg_type == "output":
                #         # Start new CoT div only if not continuing an output
                #         if last_msg_type != "output":
                #             thinking_text += f'<div class="intermediate-step"><strong>Chain-of-Thought: </strong>{content}'
                #         else:
                #             # Append to existing CoT div
                #             thinking_text += content
                #     else:
                #         # Handle undefined or other types
                #         if last_msg_type == "output":
                #             thinking_text += "</div>\n"
                #         thinking_text += f'<div class="intermediate-step">{content}</div>\n'
                #     current_query_steps.append(content)
                #     last_msg_type = msg_type
                #     thinking_container.markdown(thinking_text + ("" if msg_type != "output" else "</div>"), unsafe_allow_html=True)
                #     logging.debug(f'Thinking container updated with {thinking_text}')
                #     st.session_state.messages.append(
                #         AIMessage(content=content, additional_kwargs={"type": msg_type})
                #     )

                # if last_msg_type == "output":
                #     thinking_text += "</div>\n"
                #     thinking_container.markdown(thinking_text, unsafe_allow_html=True)

                # for i, (content, msg_type) in enumerate(all_chunks):
                #     if i > last_non_output_index and msg_type == "output":
                #         final_response_chunks.append(content)

                response_text = "".join(final_response_chunks)
                if response_text:
                    response_container.markdown(response_text, unsafe_allow_html=True)
                    st.session_state.messages.append(
                        AIMessage(content=response_text, additional_kwargs={"type": "output"})
                    )

                if current_query_steps:
                    st.session_state.intermediate_steps_history.append(current_query_steps)
                if len(st.session_state.intermediate_steps_history) > MAX_HISTORY // 2:
                    st.session_state.intermediate_steps_history = st.session_state.intermediate_steps_history[-MAX_HISTORY // 2:]

                st.session_state.messages.append(
                    AIMessage(content=response_text, additional_kwargs={"type": "output"})
                )