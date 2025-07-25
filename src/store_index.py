# store_index.py (Final Corrected Version)

import os
import sys
import logging
from src.helper import load_docs_from_directory, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
# Remove Pinecone logger

# Load environment variables from .env file
load_dotenv()

print("Step 1: Loading documents from the 'data/' directory...")
extracted_data = load_docs_from_directory('data/')
if not extracted_data:
    print("Error: No documents were loaded from the 'data' directory.")
    sys.exit(1)
print(f"-> Success: Loaded {len(extracted_data)} document(s).")

print("\nStep 2: Splitting documents into text chunks...")
text_chunks = text_split(extracted_data)
if not text_chunks:
    print("Error: Failed to split documents into chunks.")
    sys.exit(1)
print(f"-> Success: Created {len(text_chunks)} text chunks.")

print("\nStep 3: Downloading Hugging Face embeddings model...")
embeddings = download_hugging_face_embeddings()
print("-> Success: Embeddings model loaded.")

print("\nStep 4: Creating FAISS vector store and ingesting data...")
try:
    vector_store = FAISS.from_documents(
        documents=text_chunks,
        embedding=embeddings
    )
    print("\n--- INGESTION COMPLETE ---")
    # Optionally, you can save the FAISS index to disk here
    # vector_store.save_local('faiss_index')
    print("✅ Success! Your vectors have been stored in FAISS.")
except Exception as e:
    print(f"\n--- AN ERROR OCCURRED DURING FAISS OPERATIONS ---")
    print(f"Error details: {e}")