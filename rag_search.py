"""
rag_search.py
Vector similarity search for finding relevant information in the company brochure
Uses embeddings to find the most similar document chunks to the user query
"""

import os
import pandas as pd
import numpy as np
from config import hf_embeddings

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the pickle file containing document embeddings
PICKLE_FILE = os.path.join(SCRIPT_DIR, "engagepro_index.pkl")
# Minimum similarity score needed to consider a result relevant
SIMILARITY_THRESHOLD = 0.45

def calculate_similarities(query_embedding, document_embeddings):
    """
    Calculates how similar the query is to each document chunk
    Uses cosine similarity between embedding vectors
    
    Parameters:
    query_embedding: numpy array - The query converted to an embedding vector
    document_embeddings: numpy array - All document chunks as embedding vectors
    
    Returns:
    numpy array - Similarity scores for each document
    """
    similarities = hf_embeddings.similarity(query_embedding, document_embeddings)[0]
    return similarities

def find_top_k_similar(query_embedding, document_embeddings, num_results=3):
    """
    Finds the top k most similar document chunks to the query
    Sorts by similarity and returns the best matches
    
    Parameters:
    query_embedding: numpy array - The query embedding vector
    document_embeddings: numpy array - All document embeddings
    num_results: int - How many top results to return
    
    Returns:
    list - List of (index, similarity_score) tuples, sorted by similarity
    """
    similarities = calculate_similarities(query_embedding, document_embeddings)
    top_k_indices = np.argsort(similarities)[-num_results:]
    return [(index, similarities[index]) for index in reversed(top_k_indices)]

def search_engagepro(query, min_similarity=SIMILARITY_THRESHOLD):
    """
    Searches the company brochure for information related to the query
    Converts query to embedding, finds similar chunks, and returns relevant content
    
    Parameters:
    query: str - User query string
    min_similarity: float - Minimum similarity score to consider a result relevant
    
    Returns:
    tuple: (content_string, has_relevant_results)
        - content_string: All relevant chunks joined together (empty if none found)
        - has_relevant_results: True if any chunks met the similarity threshold
    """
    if not os.path.exists(PICKLE_FILE):
        return "", False
    
    # Load the pre-computed embeddings from pickle file
    df = pd.read_pickle(PICKLE_FILE)
    
    if df is None or len(df) == 0:
        return "", False
    
    # Convert embeddings and content to numpy arrays for faster processing
    document_embeddings = np.array(df['embedding'].tolist())
    document_content = np.array(df['content'].tolist())
    
    # Convert the user query to an embedding vector
    query_embedding = hf_embeddings.encode(query)
    
    # Find the top 3 most similar document chunks
    top_results = find_top_k_similar(query_embedding, document_embeddings, num_results=3)
    
    # Filter results by similarity threshold and collect relevant chunks
    relevant_chunks = []
    has_relevant = False
    
    # Only include chunks that meet the minimum similarity threshold
    for index, similarity in top_results:
        if similarity >= min_similarity:
            relevant_chunks.append(document_content[index])
            has_relevant = True
    
    # Join all relevant chunks together with double newlines
    content = "\n\n".join(relevant_chunks) if relevant_chunks else ""
    return content, has_relevant
