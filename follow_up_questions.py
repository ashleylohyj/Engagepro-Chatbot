"""
follow_up_questions.py
Generates follow-up question suggestions based on the current query and response
Uses LLM to create relevant questions the user might want to ask next
"""

from langchain_core.messages import SystemMessage, HumanMessage
from config import llm_local as llm
from prompts import FOLLOW_UP_QUESTIONS_SYSTEM_PROMPT, get_follow_up_questions_prompt


def generate_follow_up_questions(query: str, response: str) -> list:
    """
    Generates follow-up question suggestions using LLM
    Creates 3-4 relevant questions the user might want to ask next
    
    Parameters:
    query: str - The user's query
    response: str - The assistant's response
    
    Returns:
    list: List of 3-4 follow-up questions (empty list if generation fails)
    """
    try:
        # Generate prompt for follow-up questions
        prompt_content = get_follow_up_questions_prompt(query, response)
        
        messages = [
            SystemMessage(content=FOLLOW_UP_QUESTIONS_SYSTEM_PROMPT),
            HumanMessage(content=prompt_content)
        ]
        
        # Get follow-up questions from LLM
        llm_response = llm.invoke(messages)
        questions_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        # Parse the response to extract questions
        questions = _parse_follow_up_questions(questions_text)
        
        # Return 3-4 questions (limit to avoid clutter)
        return questions[:4] if len(questions) > 4 else questions
        
    except Exception as e:
        print(f"Follow-up questions generation error: {e}")
        return []


def _parse_follow_up_questions(questions_text: str) -> list:
    """
    Parses the LLM response to extract individual questions
    Removes numbering, bullet points, and other formatting
    
    Parameters:
    questions_text: str - The LLM response containing questions
    
    Returns:
    list: List of cleaned question strings
    """
    if not questions_text:
        return []
    
    questions = []
    
    # Try to split by common delimiters (numbered list, bullet points, newlines)
    # Remove common prefixes like "1.", "2.", "-", "*", etc.
    lines = questions_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove common list prefixes
        line = line.lstrip('0123456789.-*•) ')
        
        # Remove quotes if present
        line = line.strip('"\'')
        
        # Only add if it looks like a question (ends with ? or is substantial)
        if line and (line.endswith('?') or len(line) > 10):
            questions.append(line)
    
    # If no questions found with parsing, try to extract from the text directly
    if not questions and questions_text:
        # Look for questions ending with ?
        import re
        found_questions = re.findall(r'[^.!?]*\?', questions_text)
        questions = [q.strip().lstrip('0123456789.-*•) ').strip('"\'') for q in found_questions if q.strip()]
    
    return questions
