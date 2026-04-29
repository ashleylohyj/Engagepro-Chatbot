"""
guardrail.py
Checks and filters LLM responses for safety, accuracy, and quality
Detects harmful content, hallucinations, assumptions, and vague responses
"""

import re
from typing import Tuple, Optional

# Maximum response length to prevent overly long answers
MAX_RESPONSE_LENGTH = 2000

# Phrases that show the model is being honest about uncertainty
UNCERTAINTY_PHRASES = [
    r'\b(?:I (?:don\'t|do not) (?:know|have|have access to))\b',
    r'\b(?:I (?:am|\'m) not (?:sure|certain|able to))\b',
    r'\b(?:I (?:cannot|cannot|cant) (?:find|locate|access|provide))\b',
    r'\b(?:unable to|unavailable|not available)\b',
    r'\b(?:based on (?:the|available) (?:information|context|data))\b',
    r'\b(?:reliable information (?:is )?unavailable)\b',
    r'\b(?:no (?:relevant|available) (?:information|data|context))\b',
    r'\b(?:insufficient (?:information|data))\b',
]

# Patterns that show the model is being too confident (might be hallucinating)
GENERIC_CONFIDENT_PATTERNS = [
    r'\b(?:definitely|absolutely|certainly|without a doubt)\b.*\b(?:is|are|was|were)\b',
    r'\b(?:I (?:know|believe|think|remember|recall|am sure|am certain))\b',
]

# Patterns that show the model is making assumptions or speculating (making up info)
ASSUMPTION_PATTERNS = [
    r'\b(?:suggests?|suggesting|suggested)\b.*\b(?:that|the|it|this|which)\b',
    r'\b(?:likely|probably|possibly|perhaps|maybe|might|may|could)\b.*\b(?:is|are|was|were|has|have|been)\b',
    r'\b(?:seems?|appears?|appearing)\b.*\b(?:that|to be|to have|to)\b',
    r'\b(?:which (?:suggests?|indicates?|implies?|means?))\b',
    r'\b(?:this (?:suggests?|indicates?|implies?|means?|shows?))\b',
    r'\b(?:it (?:suggests?|seems?|appears?|looks?))\b',
]

# Patterns that match harmful, discriminatory, or offensive content
HARMFUL_PATTERNS = [
    r'\b(?:kill|murder|suicide|self-harm|violence|attack|bomb|weapon)\b',
    r'\b(?:hate|racist|sexist|discriminat|offensive|slur)\b',
    r'\b(?:illegal|unethical|fraud|scam|cheat|steal)\b',
    r'\b(?:drug|substance abuse|addiction)\b',
    r'\b(?:explicit|pornographic|sexual content)\b',
]

# Patterns that show the model is making assumptions about the user
USER_ASSUMPTION_PATTERNS = [
    r'\b(?:you (?:are|seem|look|appear|sound) (?:a|an|like))\b',
    r'\b(?:based on (?:your|what you))\b',
    r'\b(?:since you (?:are|seem|appear))\b',
    r'\b(?:people like you|users like you)\b',
    r'\b(?:your (?:background|identity|race|gender|age|location))\b',
]

# Patterns that match vague or overly generic responses
# Only catches single-word responses like "yes" or "no"
VAGUE_PATTERNS = [
    r'^(?:yes|no|maybe|perhaps|possibly|probably)$',
]

# Patterns that match illegal or unethical advice
UNETHICAL_ADVICE_PATTERNS = [
    r'\b(?:how to (?:hack|break|steal|cheat|lie|deceive|manipulate))\b',
    r'\b(?:illegal (?:way|method|approach|solution))\b',
    r'\b(?:unethical (?:practice|method|approach))\b',
]

def remove_source_mentions(text):
    """
    Removes mentions of sources, RAG, context, documents to make responses sound natural
    Users shouldn't see technical terms like "RAG" or "context" in the response
    
    Parameters:
    text: str - Response text
    
    Returns:
    str - Text with source mentions removed
    """
    # Patterns to remove or replace
    patterns_to_remove = [
        r'\b(?:based on|according to|from) (?:the |provided |available |given )?(?:RAG |rag |context|documents?|sources?|information|data|provided information|available information)\b',
        r'\b(?:the |provided |available |given )?(?:RAG |rag |context|documents?|sources?|information|data) (?:states?|indicates?|shows?|says?|mentions?|provides?)\b',
        r'\b(?:as (?:stated|mentioned|indicated|shown) (?:in|by) (?:the |provided |available )?(?:RAG |rag |context|documents?|sources?|information|data))\b',
        r'\b(?:information (?:retrieved|obtained|found|gathered) (?:from|in) (?:the |provided |available )?(?:RAG |rag |context|documents?|sources?|information|data))\b',
        r'\b(?:find more sources?:?\s*)?(?:wikipedia|Wikipedia)\b',
        r'\b(?:source:?\s*)?(?:wikipedia|Wikipedia)\b(?!\s*https?://)',
    ]
    
    processed = text
    for pattern in patterns_to_remove:
        processed = re.sub(pattern, '', processed, flags=re.IGNORECASE)
    
    # Clean up extra spaces but preserve line breaks for Source lines
    final_lines = []
    for line in processed.split('\n'):
        if re.search(r'Source:\s*https?://', line, re.IGNORECASE):
            final_lines.append(line.strip())
        else:
            cleaned = ' '.join(line.split())
            if cleaned:
                final_lines.append(cleaned)
    
    return '\n'.join(final_lines)

def detect_harmful_content(text: str) -> bool:
    """
    Checks if the response contains harmful, discriminatory, or offensive content
    
    Parameters:
    text: str - Response text to check
    
    Returns:
    bool - True if harmful content found
    """
    text_lower = text.lower()
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def detect_unethical_advice(text: str) -> bool:
    """
    Checks if the response contains illegal or unethical advice
    
    Parameters:
    text: str - Response text to check
    
    Returns:
    bool - True if unethical advice found
    """
    text_lower = text.lower()
    for pattern in UNETHICAL_ADVICE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def detect_user_assumptions(text: str) -> bool:
    """
    Checks if the response makes assumptions about the user's identity or background
    
    Parameters:
    text: str - Response text to check
    
    Returns:
    bool - True if user assumptions found
    """
    for pattern in USER_ASSUMPTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def detect_assumptions(text: str) -> bool:
    """
    Checks if the response uses speculative language that suggests the model is making up information
    Looks for words like "suggests", "likely", "probably" that indicate guessing
    
    Parameters:
    text: str - Response text to check
    
    Returns:
    bool - True if assumption-making language found
    """
    for pattern in ASSUMPTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def detect_vague_response(text: str) -> bool:
    """
    Checks if the response is too vague or generic
    Only catches very short responses or single-word answers
    
    Parameters:
    text: str - Response text to check
    
    Returns:
    bool - True if response is too vague
    """
    # Check if response is extremely short
    if len(text.strip()) < 15:
        return True
    
    # Check for single-word responses like "yes", "no", "maybe"
    for pattern in VAGUE_PATTERNS:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True
    
    return False


def add_transparency_statement(text: str, has_uncertainty: bool, source_type: Optional[str] = None) -> str:
    """
    Adds a note telling users to verify information from official sources
    Only adds if the response doesn't already express uncertainty
    
    Parameters:
    text: str - Response text
    has_uncertainty: bool - Whether uncertainty was already detected
    source_type: Optional[str] - "rag" or "wikipedia" or None
    
    Returns:
    str - Text with transparency note if needed
    """
    # If uncertainty is already expressed, no need to add more
    if has_uncertainty:
        return text
    
    # Add transparency statement for potentially incomplete information
    transparency_note = "\n\n*Note: This information is based on available sources. For critical decisions, please verify from official sources.*"
    
    # Only add if response is substantial (not already acknowledging uncertainty)
    if len(text) > 100 and not any(phrase in text.lower() for phrase in ["verify", "confirm", "check", "official"]):
        return text + transparency_note
    
    return text


def apply_guardrails(response_text: str, source_type: Optional[str] = None, has_context: bool = True) -> str:
    """
    Applies all safety checks and filters to the LLM response
    Checks for harmful content, hallucinations, assumptions, vague responses, and enforces tone
    This is the main function that runs all the guardrail checks
    
    Parameters:
    response_text: str - Raw LLM response text
    source_type: Optional[str] - "rag" or "wikipedia" or None
    has_context: bool - Whether relevant context was found (True) or not (False)
    
    Returns:
    str - Processed response with all guardrails applied
    """
    # Check if response is empty
    if not response_text or len(response_text.strip()) == 0:
        return safe_fallback()
    
    processed = response_text.strip()
    
    # Check if response is too long and truncate if needed
    if len(processed) > MAX_RESPONSE_LENGTH:
        processed = processed[:MAX_RESPONSE_LENGTH] + "..."
        processed = "⚠️ **Guardrail: Response length exceeded limit (truncated).**\n\n" + processed
        processed += "\n\n*Response truncated for length. Please ask a more specific question if you need more details.*"
    
    # Block harmful, discriminatory, or offensive content
    if detect_harmful_content(processed):
        return safe_fallback_harmful()
    
    # Block illegal or unethical advice
    if detect_unethical_advice(processed):
        return safe_fallback_unethical()
    
    # Remove assumptions about the user
    if detect_user_assumptions(processed):
        # Remove assumption patterns
        for pattern in USER_ASSUMPTION_PATTERNS:
            processed = re.sub(pattern, '', processed, flags=re.IGNORECASE)
        # Clean up extra spaces
        processed = ' '.join(processed.split())
        # Add warning note
        processed = "⚠️ **Guardrail: Detected and removed user assumptions/bias.**\n\n" + processed
    
    # Remove mentions of sources to make response sound natural
    processed = remove_source_mentions(processed)
    
    # Check if response already expresses uncertainty (this is good)
    has_uncertainty = False
    for pattern in UNCERTAINTY_PHRASES:
        if re.search(pattern, processed, re.IGNORECASE):
            has_uncertainty = True
            break
    
    # Check for assumption-making language (especially for RAG responses)
    # This catches when the model is guessing instead of using facts from the knowledge base
    has_assumptions = detect_assumptions(processed)
    if has_assumptions and source_type == "rag":
        # Add warning and remove assumption phrases
        processed = "⚠️ **Guardrail: Detected assumption-making language. Information may not be in knowledge base.**\n\n" + processed
        # Remove common assumption patterns
        assumption_replacements = [
            (r'\b(?:which|this|it) (?:suggests?|indicates?|implies?|means?|shows?)\b.*?\.', ''),
            (r'\b(?:likely|probably|possibly|perhaps|maybe|might|may|could)\b.*?\.', ''),
        ]
        for pattern, replacement in assumption_replacements:
            processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
        # Clean up extra spaces
        processed = ' '.join(processed.split())
        # Add uncertainty statement at the start
        if not processed.startswith("I don't have"):
            processed = "I don't have specific information about that in my knowledge base. " + processed
        has_uncertainty = True
    
    # If no context was found, make sure to express uncertainty
    if not has_context and not has_uncertainty:
        # Add uncertainty statement based on source type
        if source_type == "rag":
            processed = "I don't have specific information about that aspect of EngagePro in my knowledge base. " + processed
        elif source_type == "wikipedia":
            processed = "I cannot find sufficient information about this topic on Wikipedia. " + processed
        else:
            processed = "I don't have reliable information available about that. " + processed
        has_uncertainty = True
    
    # Check if response is too vague
    if detect_vague_response(processed):
        return safe_fallback_vague()
    
    # Check for overly confident responses (might be hallucinating)
    has_overconfident = False
    for pattern in GENERIC_CONFIDENT_PATTERNS:
        if re.search(pattern, processed, re.IGNORECASE):
            has_overconfident = True
            break
    
    # Add transparency note if response seems overconfident
    if has_overconfident and not has_uncertainty:
        processed = "⚠️ **Guardrail: Detected potentially overconfident response (potential hallucination).**\n\n" + processed
        processed = add_transparency_statement(processed, has_uncertainty, source_type)
    
    # Ensure polite and neutral tone
    processed = ensure_polite_tone(processed)
    processed = ensure_neutral_tone(processed)
    
    return processed

def ensure_polite_tone(text: str) -> str:
    """
    Makes sure the response uses polite and professional language
    Replaces casual words like "yeah" with "yes"
    
    Parameters:
    text: str - Response text
    
    Returns:
    str - Text with polite tone adjustments
    """
    # Replace overly casual or unprofessional phrases
    replacements = {
        r'\b(?:yeah|yep|nope)\b': 'yes',
        r'\b(?:gonna|wanna)\b': 'going to',
        r'\b(?:gotta)\b': 'have to',
        r'\b(?:nah)\b': 'no',
        r'\b(?:dunno)\b': "don't know",
    }
    
    processed = text
    for pattern, replacement in replacements.items():
        processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
    
    return processed


def ensure_neutral_tone(text: str) -> str:
    """
    Makes sure the response is neutral and respectful without bias
    Removes judgmental phrases like "obviously" or "as you should know"
    
    Parameters:
    text: str - Response text
    
    Returns:
    str - Text with neutral tone adjustments
    """
    # Remove or neutralize potentially biased or judgmental language
    neutral_replacements = {
        r'\b(?:obviously|clearly|of course)\b': '',
        r'\b(?:everyone knows|everybody knows)\b': '',
        r'\b(?:as you should know|as you know)\b': '',
    }
    
    processed = text
    for pattern, replacement in neutral_replacements.items():
        processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    processed = ' '.join(processed.split())
    
    return processed


def safe_fallback() -> str:
    """
    Returns a safe error message when the response is empty or invalid
    
    Returns:
    str - Safe fallback message
    """
    return "⚠️ **Guardrail: Detected empty or invalid response.**\n\nI apologize, but I'm unable to provide a response at this time. Please try rephrasing your question or ask about something else."


def safe_fallback_harmful() -> str:
    """
    Returns a safe message when harmful content is detected
    
    Returns:
    str - Safe fallback message
    """
    return "⚠️ **Guardrail: Detected harmful, discriminatory, or offensive content.**\n\nI cannot provide information on that topic. Please ask about EngagePro's services or general knowledge questions instead."


def safe_fallback_unethical() -> str:
    """
    Returns a safe message when unethical advice is detected
    
    Returns:
    str - Safe fallback message
    """
    return "⚠️ **Guardrail: Detected illegal or unethical advice.**\n\nI cannot provide advice on illegal or unethical activities. Please ask about EngagePro's services or general knowledge questions instead."


def safe_fallback_vague() -> str:
    """
    Returns a safe message when the response is too vague or generic
    
    Returns:
    str - Safe fallback message
    """
    return "⚠️ **Guardrail: Detected vague or overly generic response.**\n\nI apologize, but I need more specific information to provide a helpful answer. Could you please rephrase your question with more details?"
