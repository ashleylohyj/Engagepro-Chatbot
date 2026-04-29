# Configuration file that sets up the LLM and embedding models used throughout the chatbot
# Loads environment variables and initializes the language model and embeddings

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os

from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Initialize the local LLM client
# Connects to a local LLM server running on port 1234
# Temperature controls randomness (lower = more deterministic)
# Max tokens limits the response length
llm_local = ChatOpenAI(
    api_key="NIL",
    openai_api_base="http://localhost:1234/v1/",
    temperature=0.3,
    max_tokens=500,
)

# Initialize the embedding model for converting text to vectors
# This model is used for semantic search in the RAG system
hf_embeddings = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

