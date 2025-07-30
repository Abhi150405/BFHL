import os
import tempfile
import requests
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredEmailLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from urllib.parse import urlparse

def load_doc_from_url(url: str):
    """
    Loads a document (PDF, Word, or Email) from a given URL.
    Raises exceptions with clear error messages on failure.
    """
    try:
        # Create a temporary file to store the downloaded content
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            response = requests.get(url)
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        # Determine the correct loader based on the file extension
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        if path.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif path.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(tmp_file_path)
        elif path.endswith(('.eml', '.msg')):
            loader = UnstructuredEmailLoader(tmp_file_path)
        else:
            # Clean up and raise an error for unsupported types
            os.remove(tmp_file_path)
            raise ValueError(f"Unsupported file type for URL: {url}. Only PDF, Word (.docx), and Email (.eml/.msg) are supported.")

        # Load the document and clean up the temporary file
        documents = loader.load()
        os.remove(tmp_file_path)
        return documents

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download document from URL: {url}. Network error: {e}")
    except Exception as e:
        # Catch-all for other potential errors during processing
        raise RuntimeError(f"Error processing document from URL: {url}. Error: {e}")

def load_docs_from_directory(directory_path: str):
    """
    Loads all PDF documents from a specified directory.
    """
    try:
        # Use DirectoryLoader to efficiently load all PDFs
        loader = DirectoryLoader(
            directory_path,
            glob="*.pdf", # Process only PDF files
            loader_cls=PyPDFLoader
        )
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading documents from directory: {directory_path}. Error: {e}")
        return None

def text_split(extracted_data):
    """
    Splits the document into smaller chunks for better processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    """
    Initializes and returns Google Gemini embeddings.
    Requires the GOOGLE_API_KEY environment variable to be set.
    """
    # Check if the API key is available in the environment variables
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    
    # Initialize the Gemini embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return embeddings
