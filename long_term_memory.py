"""
long_term_memory.py
Stores long-term memory that persists across chat sessions
Tracks topic frequency, response style preferences, and saved chat sessions
"""

import json
import os

# Path to the JSON file that stores all memory
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "user_memory.json")

# Default structure when creating a new memory file
_DEFAULT_MEMORY = {
    "topic_frequency": {},
    "preferred_response_style": "concise",
    "chat_sessions": {}
}


def _ensure_memory_file():
    """
    Makes sure the memory file exists, creates it with default values if it doesn't
    """
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(_DEFAULT_MEMORY, f, indent=2)


def load_long_term_memory():
    """
    Loads long-term memory from the JSON file
    
    Returns:
    dict: Dictionary containing topic_frequency, preferred_response_style, and chat_sessions
    """
    _ensure_memory_file()
    
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            memory = json.load(f)
            # Ensure all required keys exist
            if "topic_frequency" not in memory:
                memory["topic_frequency"] = {}
            if "preferred_response_style" not in memory:
                memory["preferred_response_style"] = "concise"
            if "chat_sessions" not in memory:
                memory["chat_sessions"] = {}
            return memory
    except (json.JSONDecodeError, IOError):
        # If file is corrupted or unreadable, return default
        return _DEFAULT_MEMORY.copy()


def save_long_term_memory(memory_dict):
    """
    Saves long-term memory to the JSON file
    
    Parameters:
    memory_dict: dict - Dictionary containing topic_frequency, preferred_response_style, and chat_sessions
    """
    _ensure_memory_file()
    
    # Ensure all required keys exist
    if "topic_frequency" not in memory_dict:
        memory_dict["topic_frequency"] = {}
    if "preferred_response_style" not in memory_dict:
        memory_dict["preferred_response_style"] = "concise"
    if "chat_sessions" not in memory_dict:
        memory_dict["chat_sessions"] = {}
    
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(memory_dict, f, indent=2)
    except IOError:
        # Silently fail if file cannot be written
        pass


def update_topic_frequency(topic):
    """
    Increases the count for how many times a topic has been discussed
    
    Parameters:
    topic: str - The topic to update
    """
    if not topic or not isinstance(topic, str):
        return
    
    memory = load_long_term_memory()
    topic_lower = topic.lower().strip()
    
    if topic_lower:
        if topic_lower in memory["topic_frequency"]:
            memory["topic_frequency"][topic_lower] += 1
        else:
            memory["topic_frequency"][topic_lower] = 1
        
        save_long_term_memory(memory)


def get_preferred_response_style():
    """
    Gets the user's preferred response style setting
    
    Returns:
    str: "concise" or "detailed"
    """
    memory = load_long_term_memory()
    style = memory.get("preferred_response_style", "concise")
    # Ensure it's a valid value
    if style not in ["concise", "detailed"]:
        return "concise"
    return style


def update_preferred_response_style(style):
    """
    Updates the user's preferred response style setting
    
    Parameters:
    style: str - "concise" or "detailed"
    """
    if style not in ["concise", "detailed"]:
        return
    
    memory = load_long_term_memory()
    memory["preferred_response_style"] = style
    save_long_term_memory(memory)


def save_chat_session(session_id, messages, metadata=None):
    """
    Saves a chat session to long-term memory
    
    Parameters:
    session_id: str - Unique identifier for the session
    messages: list - List of message dictionaries with role and content
    metadata: dict or None - Optional metadata
    """
    memory = load_long_term_memory()
    
    session_data = {
        "messages": messages,
        "metadata": metadata or {}
    }
    
    memory["chat_sessions"][session_id] = session_data
    save_long_term_memory(memory)


def load_chat_session(session_id):
    """
    Loads a chat session from long-term memory
    
    Parameters:
    session_id: str - Unique identifier for the session
    
    Returns:
    dict or None: Session data with messages and metadata, or None if not found
    """
    memory = load_long_term_memory()
    return memory["chat_sessions"].get(session_id)


def get_all_chat_sessions():
    """
    Gets all saved chat sessions
    
    Returns:
    dict: Dictionary mapping session_id to session data
    """
    memory = load_long_term_memory()
    return memory.get("chat_sessions", {})


def delete_chat_session(session_id):
    """
    Deletes a chat session from long-term memory
    
    Parameters:
    session_id: str - Unique identifier for the session
    """
    memory = load_long_term_memory()
    if session_id in memory["chat_sessions"]:
        del memory["chat_sessions"][session_id]
        save_long_term_memory(memory)
