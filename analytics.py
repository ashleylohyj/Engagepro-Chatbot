"""
analytics.py
Generates analytics and statistics from all saved chat sessions
Calculates metrics like total messages, source distribution, top topics, etc.
"""

from memory_manager import get_all_sessions
from collections import Counter
from datetime import datetime


def get_analytics():
    """
    Calculates analytics from all saved chat sessions
    Counts messages, tracks sources, finds top topics, calculates averages
    
    Returns:
    dict: Dictionary containing analytics metrics like total_sessions, total_messages, etc.
    """
    sessions = get_all_sessions()
    
    if not sessions:
        return {
            "total_sessions": 0,
            "total_messages": 0,
            "total_user_messages": 0,
            "total_assistant_messages": 0,
            "source_distribution": {},
            "top_topics": [],
            "average_messages_per_session": 0,
            "most_active_day": None
        }
    
    total_sessions = len(sessions)
    total_messages = 0
    total_user_messages = 0
    total_assistant_messages = 0
    source_counter = Counter()
    all_user_messages = []
    session_dates = []
    
    # Analyze all sessions
    for session_id, session_data in sessions.items():
        messages = session_data.get("messages", [])
        metadata = session_data.get("metadata", {})
        
        # Count messages
        user_msgs = [msg for msg in messages if msg.get("role") == "user"]
        assistant_msgs = [msg for msg in messages if msg.get("role") == "assistant"]
        
        total_user_messages += len(user_msgs)
        total_assistant_messages += len(assistant_msgs)
        total_messages += len(messages)
        
        # Track sources
        for msg in assistant_msgs:
            source = msg.get("source", "rag")
            source_counter[source] += 1
        
        # Extract user message content for topic analysis
        for msg in user_msgs:
            content = msg.get("content", "")
            if content:
                all_user_messages.append(content)
        
        # Track session dates
        created_at = metadata.get("created_at", session_id)
        if created_at:
            session_dates.append(created_at)
    
    # Calculate top topics (simple word frequency from user messages)
    topic_words = []
    for msg in all_user_messages:
        # Extract key words (simple approach - first few words of each message)
        words = msg.lower().split()[:5]  # First 5 words
        topic_words.extend([w for w in words if len(w) > 3])  # Filter short words
    
    topic_counter = Counter(topic_words)
    top_topics = [{"topic": topic, "count": count} for topic, count in topic_counter.most_common(10)]
    
    # Calculate average messages per session
    avg_messages = total_user_messages / total_sessions if total_sessions > 0 else 0
    
    # Find most active day (simplified - just count sessions per day prefix)
    if session_dates:
        day_counter = Counter([date[:8] for date in session_dates if len(date) >= 8])  # YYYYMMDD
        most_active_day = day_counter.most_common(1)[0][0] if day_counter else None
    else:
        most_active_day = None
    
    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "total_user_messages": total_user_messages,
        "total_assistant_messages": total_assistant_messages,
        "source_distribution": dict(source_counter),
        "top_topics": top_topics,
        "average_messages_per_session": round(avg_messages, 2),
        "most_active_day": most_active_day
    }
