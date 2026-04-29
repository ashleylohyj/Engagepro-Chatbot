"""
wiki_tools.py
Handles all Wikipedia search and response generation
Searches Wikipedia, gets content, generates response using LLM, and returns the answer with source link
"""

from urllib.parse import quote
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.utilities import WikipediaAPIWrapper
from config import llm_local as llm
from prompts import ENGAGEPRO_SYSTEM_PROMPT, get_wikipedia_response_prompt
from guardrail import apply_guardrails
from memory_manager import get_memory_context, update_memory_from_interaction, get_preferred_response_style
from clean_query import clean_query

# Error message shown when Wikipedia search fails or finds nothing
ERROR_MESSAGE = "I am sorry I am not sure about this question. Would you like to ask about something else?"


def get_wikipedia_response(query: str):
    """
    Searches Wikipedia and generates a response using the retrieved content
    Cleans the query, searches Wikipedia, uses LLM to generate answer, applies guardrails, and returns response with source link
    
    Parameters:
    query: str - User query string
    
    Returns:
    tuple: (response_text, wiki_link) where:
        - response_text: str - Generated response text or error message
        - wiki_link: str or None - Wikipedia URL (None if no results or error)
    """
    # Get memory context to add to prompt
    memory_context = get_memory_context()
    
    try:
        # Clean the query to fix spelling and extract key terms
        cleaned_query = clean_query(query)
        
        # Initialize Wikipedia search wrapper
        # Gets top 6 Wikipedia pages, max 6000 characters per page
        wikipedia = WikipediaAPIWrapper(
            top_k_results=6,
            doc_content_chars_max=6000
        )

        # Search Wikipedia using the cleaned query
        wiki_content = wikipedia.run(cleaned_query)
        
        # If nothing found, return error message
        if not wiki_content:
            return ERROR_MESSAGE, None
        
        # Get user's preferred response style
        response_style = get_preferred_response_style()
        # Build prompt with Wikipedia content and user query
        prompt_content = get_wikipedia_response_prompt(wiki_content, query, response_style)
        
        # Add memory context to prompt if available
        if memory_context:
            prompt_content = f"{memory_context}\n\n{prompt_content}"
        
        # Prepare messages for LLM
        messages = [
            SystemMessage(content=ENGAGEPRO_SYSTEM_PROMPT),
            HumanMessage(content=prompt_content)
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # If LLM returned nothing, return error
        if not response_text:
            return ERROR_MESSAGE, None
        
        # Apply guardrails to check for safety and accuracy
        final_response = apply_guardrails(response_text, source_type="wikipedia", has_context=True)
        
        # Check if response says no information found (only for short responses to avoid false positives)
        if len(final_response.strip()) < 100 and _check_if_no_information_found(final_response):
            return ERROR_MESSAGE, None
        
        # Get the Wikipedia source link for citation
        # Use load() to get document metadata including the URL
        docs = wikipedia.load(cleaned_query)

        # If no documents, return error
        if not docs:
            return ERROR_MESSAGE, None

        # Get the source link from the first (most relevant) document
        wiki_content = docs[0].page_content
        wiki_link = docs[0].metadata.get("source")

        
        # Update memory with this interaction
        # Extract topic from query (first 3 words or whole query if shorter)
        topic = query.split()[:3] if len(query.split()) > 2 else query
        update_memory_from_interaction(
            topic=" ".join(topic) if isinstance(topic, list) else topic,
            tool="Wikipedia",
            query=query,
            response=final_response
        )
        
        # Return response and source link
        return final_response, wiki_link
        
    except Exception as e:
        # Log error but don't show it to user
        print(f"Wikipedia search error: {e}")
        return ERROR_MESSAGE, None


def _check_if_no_information_found(response_text: str) -> bool:
    """
    Checks if the LLM response says it couldn't find information
    Uses regex to detect phrases like "I cannot find information" at the start
    Only checks short responses to avoid false positives on longer answers
    
    Parameters:
    response_text: str - The response text to check
    
    Returns:
    bool - True if response says no information found, False otherwise
    """
    # Empty response means no information found
    if not response_text:
        return True
    
    # Convert to lowercase for matching
    response_lower = response_text.lower().strip()
    
    # Patterns that match "no information found" statements at the start
    explicit_no_info_patterns = [
        r'^i (?:cannot|cannot|cant|don\'t|do not) (?:find|have|locate) (?:information|any information|details)',
        r'^no information (?:is |was |available|found)',
        r'^i (?:am|am not) (?:unable|not able) to (?:find|locate|access)',
        r'^information (?:is |was )?(?:not available|unavailable|not found)',
        r'^i (?:cannot|cannot|cant) find (?:sufficient|any|enough) information',
    ]
    
    # Check if response starts with any "no information" pattern
    for pattern in explicit_no_info_patterns:
        if re.match(pattern, response_lower):
            return True
    
    # For very short responses, also check if they contain "no information" anywhere
    if len(response_text) < 80:
        if "no information" in response_lower or "cannot find information" in response_lower:
            return True
    
    # Response doesn't say no information found
    return False
