"""
memory_manager.py
Main interface for accessing and updating long-term memory
Other files should import from here instead of directly from long_term_memory.py
"""

from long_term_memory import (
    load_long_term_memory,
    update_topic_frequency,
    get_preferred_response_style,
    update_preferred_response_style,
    save_chat_session,
    load_chat_session,
    get_all_chat_sessions,
    delete_chat_session
)


def get_memory_context():
    """
    Gets memory context to add to prompts
    Currently returns empty string since we don't use conversational context
    Long-term memory is used through topic frequency and response style preferences
    
    Returns:
    str: Empty string
    """
    return ""


def update_memory_from_interaction(topic=None, intent=None, tool=None, style=None, query=None, response=None):
    """
    Updates long-term memory based on an interaction
    Tracks topic frequency and response style preferences
    
    Parameters:
    topic: str or None - The topic discussed
    intent: str or None - Not used, kept for compatibility
    tool: str or None - Not used, kept for compatibility
    style: str or None - Preferred response style ("concise" or "detailed")
    query: str or None - Not used, kept for compatibility
    response: str or None - Not used, kept for compatibility
    """
    # Update topic frequency if topic provided
    if topic:
        update_topic_frequency(topic)
    
    # Update response style preference if provided
    if style:
        update_preferred_response_style(style)


# Chat session management functions
def save_current_session(session_id, messages, metadata=None):
    """
    Saves the current chat session to long-term memory
    
    Parameters:
    session_id: str - Unique identifier for the session
    messages: list - List of message dictionaries
    metadata: dict or None - Optional metadata
    """
    save_chat_session(session_id, messages, metadata)


def load_session(session_id):
    """
    Loads a chat session from long-term memory
    
    Parameters:
    session_id: str - Unique identifier for the session
    
    Returns:
    dict or None: Session data or None if not found
    """
    return load_chat_session(session_id)


def get_all_sessions():
    """
    Gets all saved chat sessions
    
    Returns:
    dict: Dictionary mapping session_id to session data
    """
    return get_all_chat_sessions()


def remove_session(session_id):
    """
    Deletes a chat session from long-term memory
    
    Parameters:
    session_id: str - Unique identifier for the session
    """
    delete_chat_session(session_id)
