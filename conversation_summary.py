"""
conversation_summary.py
Generates short summaries of chat conversations for the history tab
Uses LLM to create a brief summary that captures the main topics discussed
"""

from langchain_core.messages import SystemMessage, HumanMessage
from config import llm_local as llm
from prompts import CONVERSATION_SUMMARY_SYSTEM_PROMPT, get_conversation_summary_prompt


def summarize_conversation(messages: list) -> str:
    """
    Generates a short summary of a conversation using LLM
    Creates a brief summary for displaying in the history tab
    
    Parameters:
    messages: list - List of message dictionaries with 'role' and 'content'
    
    Returns:
    str - Short summary of the conversation
    """
    try:
        if not messages:
            return "Empty conversation"
        
        # Format messages for the prompt
        conversation_text = _format_messages_for_summary(messages)
        
        # Generate prompt for conversation summary
        prompt_content = get_conversation_summary_prompt(conversation_text)
        
        messages_llm = [
            SystemMessage(content=CONVERSATION_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=prompt_content)
        ]
        
        # Get summary from LLM
        response = llm.invoke(messages_llm)
        summary = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up the summary
        summary = summary.strip()
        
        return summary if summary else "No summary available"
        
    except Exception as e:
        print(f"Conversation summary error: {e}")
        # Fallback: return first user message preview
        first_user_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
        if first_user_msg:
            preview = first_user_msg.get("content", "")[:50]
            return preview + "..." if len(preview) < len(first_user_msg.get("content", "")) else preview
        return "No summary available"


def _format_messages_for_summary(messages: list) -> str:
    """
    Formats messages into a readable text format for the LLM to summarize
    Truncates long assistant responses to keep the summary prompt manageable
    
    Parameters:
    messages: list - List of message dictionaries
    
    Returns:
    str - Formatted conversation text
    """
    conversation_lines = []
    
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            conversation_lines.append(f"User: {content}")
        elif role == "assistant":
            # Truncate long assistant responses for summary
            content_preview = content[:200] if len(content) > 200 else content
            conversation_lines.append(f"Assistant: {content_preview}")
    
    return "\n".join(conversation_lines)
