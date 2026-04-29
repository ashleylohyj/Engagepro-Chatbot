"""
ui.py
Main Streamlit user interface for the EngagePro chatbot
Handles chat display, user input, sidebar settings, history tab, and analytics tab
"""

import streamlit as st
import time
import os
import re
from datetime import datetime
from router_agent import route_and_respond
from memory_manager import (
    save_current_session,
    load_session,
    get_all_sessions,
    remove_session,
    get_preferred_response_style,
    update_preferred_response_style
)
from follow_up_questions import generate_follow_up_questions
from conversation_summary import summarize_conversation
from analytics import get_analytics

# File paths for avatars and configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHATBOT_AVATAR = os.path.join(SCRIPT_DIR, "Chatbot Profile Pic.png")
WIKI_AVATAR = os.path.join(SCRIPT_DIR, "Wiki Profile Pic.png")
CONFIG_PATH = os.path.join(SCRIPT_DIR, ".streamlit", "config.toml")


def get_current_theme():
    """Reads the current theme setting (light or dark) from the config file"""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'base = "dark"' in content or 'base="dark"' in content:
                    return "dark"
                elif 'base = "light"' in content or 'base="light"' in content:
                    return "light"
        return "light"
    except Exception:
        return "light"


def update_theme(theme):
    """Updates the theme setting in the config file and writes it to disk"""
    try:
        if theme == "dark":
            config_content = """[theme]
base = "dark"
primaryColor = "#667eea"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
font = "sans serif"
"""
        else:
            config_content = """[theme]
base = "light"
primaryColor = "#667eea"
backgroundColor = "#f9fafb"
secondaryBackgroundColor = "#ffffff"
textColor = "#111827"
font = "sans serif"
"""
        
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            f.write(config_content)
            f.flush()
            os.fsync(f.fileno())
        
        return True
    except Exception as e:
        st.error(f"Error updating theme: {str(e)}")
        return False


def get_avatar_for_source(source_type):
    """
    Returns the avatar image path based on whether the response came from RAG or Wikipedia
    Falls back to emoji if image file doesn't exist
    
    Parameters:
    source_type: str - 'rag' or 'wikipedia'
    
    Returns:
    str - Path to avatar image or emoji fallback
    """
    if source_type == 'rag':
        return CHATBOT_AVATAR if os.path.exists(CHATBOT_AVATAR) else "🤖"
    elif source_type == 'wikipedia':
        return WIKI_AVATAR if os.path.exists(WIKI_AVATAR) else "🌐"
    else:
        return CHATBOT_AVATAR if os.path.exists(CHATBOT_AVATAR) else "🤖"


def format_response(response_text):
    """
    Removes any markdown formatting from the response to ensure plain text display
    Strips bold, italic, headers, lists, code blocks, and links but keeps the text content
    
    Parameters:
    response_text: str - The raw response text from the LLM
    
    Returns:
    str - Plain text response with all markdown removed
    """
    if not response_text:
        return response_text
    
    # Ensure proper line breaks
    response_text = response_text.strip()
    
    # Remove markdown formatting symbols
    # Remove bold markers (**text** or __text__)
    response_text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', response_text)
    response_text = re.sub(r'__([^_]+)__', r'\1', response_text)
    
    # Remove italic markers (*text* or _text_)
    response_text = re.sub(r'(?<!\*)\*(?!\*)([^\*]+)\*(?!\*)', r'\1', response_text)
    response_text = re.sub(r'(?<!_)_(?!_)([^_]+)_(?!_)', r'\1', response_text)
    
    # Remove headers (# ## ###)
    response_text = re.sub(r'^#+\s+', '', response_text, flags=re.MULTILINE)
    
    # Remove horizontal rules (--- or ===)
    response_text = re.sub(r'^[-=]{3,}\s*$', '', response_text, flags=re.MULTILINE)
    
    # Remove markdown list markers but keep the text
    response_text = re.sub(r'^[-•*]\s+', '', response_text, flags=re.MULTILINE)
    response_text = re.sub(r'^\d+[.)]\s+', '', response_text, flags=re.MULTILINE)
    
    # Remove code blocks (```code```)
    response_text = re.sub(r'```[^`]*```', '', response_text, flags=re.DOTALL)
    response_text = re.sub(r'`([^`]+)`', r'\1', response_text)
    
    # Remove links [text](url) but keep the text
    response_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', response_text)
    
    # Clean up extra whitespace but preserve paragraph breaks
    lines = response_text.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
        else:
            # Preserve blank lines for paragraph breaks
            if cleaned_lines and cleaned_lines[-1] != '':
                cleaned_lines.append('')
    
    return '\n'.join(cleaned_lines)

def main():
    """Main function that runs the Streamlit chat interface"""
    
    # Set up the page configuration
    st.set_page_config(
        page_title="EngagePro Chatbot",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state variables that persist across page reruns
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_prompt" not in st.session_state:
        st.session_state.processing_prompt = None
    if "followup_placeholders" not in st.session_state:
        st.session_state.followup_placeholders = []
    
    # Sidebar section with chatbot info, settings, and preferences
    with st.sidebar:
        # Display chatbot avatar and title side by side
        header_img = CHATBOT_AVATAR if os.path.exists(CHATBOT_AVATAR) else None
        
        col_img, col_title = st.columns([1, 2])
        with col_img:
            if header_img:
                st.image(header_img, width=80, use_container_width=False)
        
        with col_title:
            st.title("EngagePro Chatbot")
        
        st.caption("Your intelligent assistant for EngagePro information and general knowledge")
        
        # New Chat button - Starts a fresh conversation
        st.write("")
        if st.button("➕ New Chat"):
            # Clear session state and rerun
            st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.messages = []
            st.session_state.processing_prompt = None
            st.session_state.followup_placeholders = []
            st.rerun()

        # Dark mode toggle
        current_theme = get_current_theme()
        is_dark = current_theme == "dark"
        
        new_dark_mode = st.toggle("🌙 Dark Mode", value=is_dark, key="dark_mode_toggle")
        
        if new_dark_mode != is_dark:
            new_theme = "dark" if new_dark_mode else "light"
            update_theme(new_theme)
            time.sleep(0.1)
            st.rerun()
        
        st.divider()

        st.subheader("Features")
        st.write("- Ask about EngagePro services and products \n - Search general or technical knowledge via Wikipedia \n - Get instant, accurate responses")
        
        st.divider()
        
        # Response Length Preference - User can choose concise or detailed responses
        st.subheader("Response Length")
        current_style = get_preferred_response_style()
        
        # Display buttons to toggle between concise and detailed response styles
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 Concise", key="style_concise", use_container_width=True, 
                        type="primary" if current_style == "concise" else "secondary"):
                update_preferred_response_style("concise")
                st.rerun()
        
        with col2:
            if st.button("📄 Detailed", key="style_detailed", use_container_width=True,
                        type="primary" if current_style == "detailed" else "secondary"):
                update_preferred_response_style("detailed")
                st.rerun()
        
        # Display current response style preference
        style_display = "📝 Concise" if current_style == "concise" else "📄 Detailed"
        st.caption(f"Current: {style_display}")
        
        st.divider()
    
    # Create main tabs for different views
    chat_tab, history_tab, analytics_tab = st.tabs(["💬 Chat", "📜 History", "📊 Analytics"])
    
    # Chat Tab - Main conversation interface
    with chat_tab:
        # Reset follow-up placeholders list at start of each render cycle
        # Placeholders will be repopulated as messages with follow-up questions are rendered
        st.session_state.followup_placeholders = []
        
        # Display all previous messages in the chat history
        for msg_idx, message in enumerate(st.session_state.messages):
            role = message["role"]
            content = message["content"]
            source = message.get("source", "rag")
            wiki_link = message.get("wiki_link", None)
            follow_up_questions = message.get("follow_up_questions", None)
            
            # Get avatar image path based on source type (RAG or Wikipedia)
            avatar = get_avatar_for_source(source) if role == "assistant" else None
            
            # Use file path if avatar file exists, otherwise use emoji fallback
            avatar_path = avatar if isinstance(avatar, str) and os.path.exists(avatar) else (avatar if role == "assistant" else None)
            
            # Display message with appropriate avatar
            with st.chat_message(role, avatar=avatar_path):
                # Remove any markdown formatting and display as plain text
                cleaned_content = format_response(content)
                st.write(cleaned_content)
                # Display Wikipedia source link below assistant responses
                if wiki_link and role == "assistant":
                    st.markdown("---")
                    st.markdown(f"📚 **Source:** {wiki_link}")
                
                # Display follow-up question suggestions below assistant messages
                # Uses st.empty() placeholders for instant clearing when new question is asked
                if follow_up_questions and role == "assistant" and not st.session_state.processing_prompt:
                    followup_placeholder = st.empty()
                    # Store placeholder reference for instant clearing functionality
                    st.session_state.followup_placeholders.append(followup_placeholder)
                    
                    with followup_placeholder.container():
                        st.markdown("---")
                        st.markdown("**💡 Suggested follow-up questions:**")
                        num_questions = len(follow_up_questions)
                        
                        # Display 1-2 questions in individual columns
                        if num_questions <= 2:
                            cols = st.columns(num_questions)
                            for i, question in enumerate(follow_up_questions):
                                with cols[i]:
                                    button_key = f"followup_history_{msg_idx}_{i}_{hash(question) % 10000}"
                                    if st.button(question, key=button_key, use_container_width=True):
                                        # Clear all follow-up question placeholders instantly from UI
                                        for placeholder in st.session_state.followup_placeholders:
                                            placeholder.empty()
                                        st.session_state.followup_placeholders = []
                                        
                                        # Remove follow-up questions from all assistant messages
                                        for msg in st.session_state.messages:
                                            if msg.get("role") == "assistant" and "follow_up_questions" in msg:
                                                msg["follow_up_questions"] = None
                                        
                                        # Add clicked follow-up question as new user message
                                        st.session_state.messages.append({
                                            "role": "user",
                                            "content": question,
                                            "source": None
                                        })
                                        st.session_state.processing_prompt = question
                                        st.rerun()
                        else:
                            # Display 3-4 questions in 2 columns (2 questions per column)
                            cols = st.columns(2)
                            for i, question in enumerate(follow_up_questions):
                                col_idx = i % 2
                                with cols[col_idx]:
                                    button_key = f"followup_history_{msg_idx}_{i}_{hash(question) % 10000}"
                                    if st.button(question, key=button_key, use_container_width=True):
                                        # Clear all follow-up question placeholders instantly from UI
                                        for placeholder in st.session_state.followup_placeholders:
                                            placeholder.empty()
                                        st.session_state.followup_placeholders = []
                                        
                                        # Remove follow-up questions from all assistant messages
                                        for msg in st.session_state.messages:
                                            if msg.get("role") == "assistant" and "follow_up_questions" in msg:
                                                msg["follow_up_questions"] = None
                                        
                                        # Add clicked follow-up question as new user message
                                        st.session_state.messages.append({
                                            "role": "user",
                                            "content": question,
                                            "source": None
                                        })
                                        st.session_state.processing_prompt = question
                                        st.rerun()
                elif follow_up_questions and role == "assistant" and st.session_state.processing_prompt:
                    # Hide follow-up questions when a new question is being processed
                    st.empty()
        
        # Loading indicator placeholder (displays above chat input)
        loading_placeholder = st.empty()
        
        # Process user prompt if one is pending
        if st.session_state.processing_prompt:
            prompt = st.session_state.processing_prompt
            
            # Display loading spinner while generating response
            with loading_placeholder.container():
                with st.spinner("Thinking..."):
                    # Route query to RAG or Wikipedia and get response
                    try:
                        response, source_type, wiki_link = route_and_respond(prompt)
                    except Exception as e:
                        # Handle errors gracefully with user-friendly message
                        response = f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."
                        source_type = "rag"
                        wiki_link = None
            
            # Clear loading indicator
            loading_placeholder.empty()
            
            # Get avatar image based on response source (RAG or Wikipedia)
            avatar = get_avatar_for_source(source_type)
            avatar_path = avatar if isinstance(avatar, str) and os.path.exists(avatar) else None
            
            # Display assistant response with appropriate avatar
            with st.chat_message("assistant", avatar=avatar_path):
                # Stream response word-by-word with markdown rendering
                response_container = st.empty()
                accumulated_text = ""
                
                # Accumulate words and render plain text after each word for smooth streaming effect
                words = response.split()
                for i, word in enumerate(words):
                    accumulated_text += word
                    if i < len(words) - 1:
                        accumulated_text += " "
                    # Remove markdown and display as plain text
                    cleaned_text = format_response(accumulated_text)
                    response_container.write(cleaned_text)
                    time.sleep(0.03)  # Small delay for streaming effect
                
                response_text = accumulated_text.strip()
                # Clean final response to remove any markdown
                cleaned_response_text = format_response(response_text)
                
                # Display Wikipedia source link below response if available
                if wiki_link:
                    st.markdown("---")
                    st.markdown(f"📚 **Source:** {wiki_link}")
                
                # Generate follow-up question suggestions using LLM
                follow_up_questions = generate_follow_up_questions(prompt, response_text)
                followup_placeholder = st.empty()
                # Store placeholder for instant clearing when new question is asked
                st.session_state.followup_placeholders.append(followup_placeholder)

                # Display follow-up questions as clickable buttons
                if follow_up_questions and not st.session_state.processing_prompt:
                    with followup_placeholder.container():
                        st.markdown("---")
                        st.markdown("**💡 Suggested follow-up questions:**")
                        num_questions = len(follow_up_questions)
                        message_index = len(st.session_state.messages)
                        
                        # Display 1-2 questions in individual columns
                        if num_questions <= 2:
                            cols = st.columns(num_questions)
                            for i, question in enumerate(follow_up_questions):
                                with cols[i]:
                                    button_key = f"followup_{message_index}_{i}_{hash(question) % 10000}"
                                    if st.button(question, key=button_key, use_container_width=True):
                                        # Clear all follow-up question placeholders instantly from UI
                                        for placeholder in st.session_state.followup_placeholders:
                                            placeholder.empty()
                                        st.session_state.followup_placeholders = []
                                        
                                        # Remove follow-up questions from all assistant messages
                                        for msg in st.session_state.messages:
                                            if msg.get("role") == "assistant" and "follow_up_questions" in msg:
                                                msg["follow_up_questions"] = None
                                        
                                        # Add clicked follow-up question as new user message
                                        st.session_state.messages.append({
                                            "role": "user",
                                            "content": question,
                                            "source": None
                                        })
                                        st.session_state.processing_prompt = question
                                        st.rerun()
                        else:
                            # Display 3-4 questions in 2 columns (2 questions per column)
                            cols = st.columns(2)
                            for i, question in enumerate(follow_up_questions):
                                col_idx = i % 2
                                with cols[col_idx]:
                                    button_key = f"followup_{message_index}_{i}_{hash(question) % 10000}"
                                    if st.button(question, key=button_key, use_container_width=True):
                                        # Clear all follow-up question placeholders instantly from UI
                                        for placeholder in st.session_state.followup_placeholders:
                                            placeholder.empty()
                                        st.session_state.followup_placeholders = []
                                        
                                        # Remove follow-up questions from all assistant messages
                                        for msg in st.session_state.messages:
                                            if msg.get("role") == "assistant" and "follow_up_questions" in msg:
                                                msg["follow_up_questions"] = None
                                        
                                        # Add clicked follow-up question as new user message
                                        st.session_state.messages.append({
                                            "role": "user",
                                            "content": question,
                                            "source": None
                                        })
                                        st.session_state.processing_prompt = question
                                        st.rerun()
                elif follow_up_questions and st.session_state.processing_prompt:
                    # Hide follow-up questions when a new question is being processed
                    followup_placeholder.empty()
            
            # Save assistant response to chat history with metadata
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "source": source_type,
                "wiki_link": wiki_link,
                "follow_up_questions": follow_up_questions if follow_up_questions else None
            })
            
            # Clear processing flag to indicate response is complete
            st.session_state.processing_prompt = None
            
            # Save current session to long-term memory for history tab
            if st.session_state.messages:
                # Count only user messages (exclude assistant responses)
                user_message_count = len([msg for msg in st.session_state.messages if msg.get("role") == "user"])
                metadata = {
                    "created_at": st.session_state.current_session_id,
                    "message_count": user_message_count,
                    "last_updated": datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                save_current_session(st.session_state.current_session_id, st.session_state.messages, metadata)
            
            # Rerun to refresh UI and display new response
            st.rerun()
        
        # Chat input box (always positioned at bottom of chat area)
        if prompt := st.chat_input("💬 Ask me anything about EngagePro or search for general information..."):
            # Clear all follow-up question placeholders instantly when new question is submitted
            for placeholder in st.session_state.followup_placeholders:
                placeholder.empty()
            st.session_state.followup_placeholders = []
            
            # Remove follow-up questions from all assistant messages
            for msg in st.session_state.messages:
                if msg.get("role") == "assistant" and "follow_up_questions" in msg:
                    msg["follow_up_questions"] = None
            
            # Add user message to chat history
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt,
                "source": None
            })
            
            # Set processing flag to trigger response generation on next rerun
            st.session_state.processing_prompt = prompt
            st.rerun()
    
    # History Tab - Display and manage past chat sessions
    with history_tab:
        st.subheader("Chat History")
        
        sessions = get_all_sessions()
        
        if not sessions:
            st.info("No previous chat sessions found. Start chatting to create your first session!")
        else:
            # Display sessions in reverse chronological order (newest first)
            session_ids = sorted(sessions.keys(), reverse=True)
            
            for session_id in session_ids:
                session_data = sessions[session_id]
                messages = session_data.get("messages", [])
                metadata = session_data.get("metadata", {})
                
                # Skip empty sessions
                if not messages:
                    continue
                
                # Generate short summary of conversation using LLM
                summary = summarize_conversation(messages)
                
                # Display session information and controls
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**Session:** {session_id}")
                    st.caption(f"**Summary:** {summary}")
                    # Count only user messages (exclude assistant responses)
                    user_message_count = len([msg for msg in messages if msg.get("role") == "user"])
                    st.caption(f"Messages: {user_message_count}")
                
                with col2:
                    # Load button - Restores selected session to chat tab
                    if st.button("Load", key=f"load_{session_id}"):
                        # Save current session before loading a different one
                        if st.session_state.messages:
                            user_message_count = len([msg for msg in st.session_state.messages if msg.get("role") == "user"])
                            current_metadata = {
                                "created_at": st.session_state.current_session_id,
                                "message_count": user_message_count
                            }
                            save_current_session(st.session_state.current_session_id, st.session_state.messages, current_metadata)
                        
                        # Load selected session into current chat
                        st.session_state.current_session_id = session_id
                        st.session_state.messages = messages.copy()
                        st.rerun()
                    
                    # Delete button - Removes session from history
                    if st.button("Delete", key=f"delete_{session_id}"):
                        remove_session(session_id)
                        st.rerun()
                
                st.divider()
    
    # Analytics Tab - Display usage statistics and insights
    with analytics_tab:
        st.subheader("📊 Analytics & Insights")
        
        # Retrieve analytics data from all chat sessions
        analytics = get_analytics()
        
        if analytics["total_sessions"] == 0:
            st.info("No chat sessions found. Start chatting to see analytics!")
        else:
            # Overview Metrics - Key statistics at a glance
            st.markdown("### 📈 Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Sessions", analytics["total_sessions"])
            
            with col2:
                st.metric("Total Messages", analytics["total_messages"])
            
            with col3:
                st.metric("User Messages", analytics["total_user_messages"])
            
            with col4:
                st.metric("Avg Messages/Session", analytics["average_messages_per_session"])
            
            st.divider()
            
            # Source Distribution - Breakdown of RAG vs Wikipedia responses
            st.markdown("### 🔍 Source Distribution")
            source_dist = analytics["source_distribution"]
            
            if source_dist:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display source counts and percentages as metrics
                    rag_count = source_dist.get("rag", 0)
                    wiki_count = source_dist.get("wikipedia", 0)
                    total_responses = rag_count + wiki_count
                    
                    if total_responses > 0:
                        st.metric("EngagePro (RAG)", rag_count, f"{round(rag_count/total_responses*100, 1)}%")
                        st.metric("Wikipedia", wiki_count, f"{round(wiki_count/total_responses*100, 1)}%")
                    else:
                        st.write("No responses yet")
                
                with col2:
                    # Visual progress bars showing source distribution
                    if total_responses > 0:
                        st.write("**Distribution:**")
                        rag_pct = rag_count / total_responses * 100
                        wiki_pct = wiki_count / total_responses * 100
                        st.progress(rag_pct / 100, text=f"RAG: {rag_pct:.1f}%")
                        st.progress(wiki_pct / 100, text=f"Wikipedia: {wiki_pct:.1f}%")
            else:
                st.info("No source data available yet")
            
            st.divider()
            
            # Top Topics - Most frequently discussed topics
            st.markdown("### 🎯 Top Topics")
            top_topics = analytics["top_topics"]
            
            if top_topics:
                # Display top 5 most discussed topics with mention counts
                for i, topic_data in enumerate(top_topics[:5], 1):
                    topic = topic_data["topic"]
                    count = topic_data["count"]
                    st.write(f"{i}. **{topic.capitalize()}** - {count} mentions")
            else:
                st.info("No topic data available yet")
            
            st.divider()
            
            # Additional Insights - Detailed activity and source breakdown
            st.markdown("### 💡 Insights")
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.write("**Activity Summary:**")
                st.write(f"- Total chat sessions: {analytics['total_sessions']}")
                st.write(f"- Total interactions: {analytics['total_messages']}")
                st.write(f"- Average messages per session: {analytics['average_messages_per_session']}")
            
            with insights_col2:
                st.write("**Response Sources:**")
                if source_dist:
                    rag_count = source_dist.get("rag", 0)
                    wiki_count = source_dist.get("wikipedia", 0)
                    total = rag_count + wiki_count
                    if total > 0:
                        st.write(f"- EngagePro knowledge: {rag_count} ({round(rag_count/total*100, 1)}%)")
                        st.write(f"- Wikipedia searches: {wiki_count} ({round(wiki_count/total*100, 1)}%)")
                else:
                    st.write("No data available")

if __name__ == "__main__":
    main()
