import os
import tempfile
import requests
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredEmailLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from urllib.parse import urlparse

def load_doc_from_url(url: str):
    """
    Loads a document (PDF, Word, or Email) from a given URL.
    Raises exceptions with clear error messages on failure.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            response = requests.get(url)
            response.raise_for_status()
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        if path.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        elif path.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(tmp_file_path)
        elif path.endswith(('.eml', '.msg')):
            loader = UnstructuredEmailLoader(tmp_file_path)
        else:
            os.remove(tmp_file_path)
            raise ValueError(f"Unsupported file type for URL: {url}. Only PDF, Word (.docx), and Email (.eml/.msg) are supported.")

        documents = loader.load()
        os.remove(tmp_file_path)
        return documents

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download document from URL: {url}. Network error: {e}")
    except Exception as e:
        raise RuntimeError(f"Error processing document from URL: {url}. Error: {e}")

def load_docs_from_directory(directory_path: str):
    """
    Loads all PDF documents from a specified directory.
    """
    try:
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
    Splits the document into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    """
    Downloads embeddings from Hugging Face.
    """
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    return embeddings
