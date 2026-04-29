"""
rag_prepare.py
Prepares the knowledge base by processing the PDF and creating embeddings
This is run once to build the searchable index of the company brochure
"""

import os
import pandas as pd
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import hf_embeddings

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
PDF_FILE = os.path.join(SCRIPT_DIR, "Company_Brochure.pdf")
PICKLE_FILE = os.path.join(SCRIPT_DIR, "engagepro_index.pkl")
# Chunk size and overlap for splitting the PDF into smaller pieces
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def main():
    print("=" * 60)
    print("EngagePro RAG Knowledge Base Preparation")
    print("=" * 60)
    
    # Load the PDF file
    print(f"\n[Step 1] Loading PDF: {PDF_FILE}")
    if not os.path.exists(PDF_FILE):
        print(f"ERROR: PDF file not found: {PDF_FILE}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {SCRIPT_DIR}")
        return
    
    loader = PyPDFLoader(PDF_FILE)
    print(f"PDF file found: {PDF_FILE}")
    
    # Split the PDF into smaller chunks for better search
    print(f"\n[Step 2] Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    pages = loader.load_and_split(text_splitter)
    print(f"Created {len(pages)} document chunks")
    
    # Convert each chunk to an embedding vector
    print(f"\n[Step 3] Generating embeddings...")
    documents_data = []
    
    for idx, doc in enumerate(pages):
        # Convert text chunk to embedding vector
        doc_embedding = hf_embeddings.encode(doc.page_content)
        documents_data.append({
            "embedding": doc_embedding,
            "source": doc.metadata,
            "content": doc.page_content
        })
        
        # Show progress every 10 chunks
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(pages)} chunks...")
    
    print(f"Generated embeddings for {len(documents_data)} chunks")
    
    # Create a DataFrame to store all the data
    print(f"\n[Step 4] Creating DataFrame...")
    df = pd.DataFrame(documents_data)
    print(f"DataFrame created with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Save everything to a pickle file for fast loading later
    print(f"\n[Step 5] Saving to pickle file: {PICKLE_FILE}")
    df.to_pickle(PICKLE_FILE)
    print(f"Successfully saved embeddings to {PICKLE_FILE}")
    
    print("\n" + "=" * 60)
    print("RAG preparation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
