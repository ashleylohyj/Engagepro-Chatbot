"""
router_agent.py
Main routing logic that decides how to answer user queries
Always tries RAG search first, then falls back to Wikipedia if no relevant results found
"""

from langchain_core.messages import SystemMessage, HumanMessage
from config import llm_local as llm
from rag_search import search_engagepro
from wiki_tools import get_wikipedia_response
from prompts import ENGAGEPRO_SYSTEM_PROMPT, get_rag_prompt
from guardrail import apply_guardrails
from memory_manager import get_memory_context, update_memory_from_interaction, get_preferred_response_style

def route_and_respond(query):
    """
    Routes the user query and generates a response
    First searches the company brochure using RAG, if nothing relevant found then uses Wikipedia
    
    Parameters:
    query: str - User query
    
    Returns:
    tuple: (response_text, source_type, wiki_link) where:
        - response_text: str - The response text
        - source_type: str - 'rag' or 'wikipedia'
        - wiki_link: str or None - Wikipedia link if source is 'wikipedia', None otherwise
    """
    # Get memory context to add to the prompt
    memory_context = get_memory_context()
    
    # Search the company brochure using vector similarity search
    context, has_relevant = search_engagepro(query)
    
    # If relevant information found in company brochure, use RAG to answer
    if has_relevant and context.strip():
        # Get user's preferred response style (concise or detailed)
        response_style = get_preferred_response_style()
        # Build the prompt with the retrieved context and user query
        prompt_content = get_rag_prompt(context, query, response_style)
        
        # Add memory context to the prompt if available
        if memory_context:
            prompt_content = f"{memory_context}\n\n{prompt_content}"
        
        # Prepare messages for the LLM
        messages = [
            SystemMessage(content=ENGAGEPRO_SYSTEM_PROMPT),
            HumanMessage(content=prompt_content)
        ]
        
        # Get response from LLM
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Apply guardrails to check for safety and accuracy
        final_response = apply_guardrails(response_text, source_type="rag", has_context=has_relevant)
        
        # Update memory with this interaction
        # Extract topic from query (first 3 words or whole query if shorter)
        topic = query.split()[0:3] if len(query.split()) > 2 else query
        update_memory_from_interaction(
            topic=" ".join(topic) if isinstance(topic, list) else topic, 
            tool="RAG",
            query=query,
            response=final_response
        )
        
        return final_response, 'rag', None
    
    # No relevant results in company brochure, use Wikipedia instead
    response_text, wiki_link = get_wikipedia_response(query)
    return response_text, 'wikipedia', wiki_link
