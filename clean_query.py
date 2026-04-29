"""
clean_query.py
Cleans and optimizes user queries before searching Wikipedia
Uses LLM to fix spelling mistakes and extract key search terms
"""

from langchain_core.messages import SystemMessage, HumanMessage
from config import llm_local as llm
from prompts import QUERY_CLEANING_SYSTEM_PROMPT, get_query_cleaning_prompt


def clean_query(query: str) -> str:
    """
    Cleans the user query using LLM to fix spelling and extract important keywords
    Makes the query better for Wikipedia search by removing unnecessary words
    
    Parameters:
    query: str - Original user query (might have typos or extra words)
    
    Returns:
    str - Cleaned query ready for Wikipedia search
    """
    try:
        # Build the prompt asking LLM to clean the query
        prompt_content = get_query_cleaning_prompt(query)
        
        # Send to LLM for cleaning
        messages = [
            SystemMessage(content=QUERY_CLEANING_SYSTEM_PROMPT),
            HumanMessage(content=prompt_content)
        ]
        
        response = llm.invoke(messages)
        cleaned_query = response.content if hasattr(response, 'content') else str(response)
        cleaned_query = cleaned_query.strip()
        
        # If cleaning failed, return original query
        if not cleaned_query:
            return query
        
        return cleaned_query
        
    except Exception as e:
        print(f"Query cleaning error: {e}")
        return query
