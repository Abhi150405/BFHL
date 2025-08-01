import os
import io
import requests
import tempfile
from urllib.parse import urlparse
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEmailLoader,
    UnstructuredXMLLoader,
    RSSFeedLoader
)
from langchain_unstructured import UnstructuredLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def load_doc_from_url(url: str):
    """
    Loads a document (PDF, Word, Email, XML, etc.) from a given URL by processing it
    from a temporary file for maximum compatibility and speed.
    Raises exceptions with clear error messages on failure.
    """
    # Get the filename from the URL to determine the correct suffix
    file_name = os.path.basename(urlparse(url).path)
    
    # Special handling for RSS feeds - they don't need to be downloaded first
    if file_name.lower().endswith('.rss'):
        print(f"Processing RSS feed from {url}...")
        loader = RSSFeedLoader(urls=[url])
        print("Parsing RSS content...")
        documents = loader.load()
        print("RSS parsing complete.")
        return documents
    
    try:
        # Download the document content
        print(f"Downloading document from {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        suffix = os.path.splitext(file_name)[1]

        # Create a temporary file to store the downloaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        print(f"Processing '{file_name}' from temporary file...")
        
        loader = None
        # Determine the correct loader based on the file extension
        if file_name.lower().endswith('.pdf'):
            # Use the much faster PyMuPDFLoader for PDFs
            loader = PyMuPDFLoader(file_path=temp_file_path)
        elif file_name.lower().endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(file_path=temp_file_path)
        elif file_name.lower().endswith(('.eml', '.msg')):
            loader = UnstructuredEmailLoader(file_path=temp_file_path)
        elif file_name.lower().endswith('.xml'):
            # Added handler for XML files
            loader = UnstructuredXMLLoader(file_path=temp_file_path)
        else:
            # General fallback for other file types supported by 'unstructured'
            print(f"Using generic file loader for '{file_name}'.")
            loader = UnstructuredLoader(file_path=temp_file_path)

        # Load the document.
        print("Parsing document content...")
        documents = loader.load()
        print("Document parsing complete.")
        
    finally:
        # Ensure the temporary file is always deleted
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")

    return documents

def download_hugging_face_embeddings():
    """
    Initializes and returns Google Gemini embeddings.
    Requires the GOOGLE_API_KEY environment variable to be set.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Fallback for different naming conventions
        api_key = os.getenv("GOOGLE_API_KEY1")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GOOGLE_API_KEY1 environment variable not set.")
    
    # Initialize the Gemini embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return embeddings
