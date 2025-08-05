
import os
import io
import requests
import tempfile
from urllib.parse import urlparse
from typing import List
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEmailLoader,
    UnstructuredXMLLoader,
    RSSFeedLoader
)
from langchain_unstructured import UnstructuredLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

def load_doc_from_url(url: str):
    """
    Enhanced document loading with better content preservation for insurance documents.
    """
    file_name = os.path.basename(urlparse(url).path)

    # Special handling for RSS feeds
    if file_name.lower().endswith('.rss'):
        print(f"Processing RSS feed from {url}...")
        loader = RSSFeedLoader(urls=[url])
        documents = loader.load()
        return enhance_document_metadata(documents)

    try:
        print(f"Downloading document from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        suffix = os.path.splitext(file_name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        print(f"Processing '{file_name}' from temporary file...")

        loader = None
        if file_name.lower().endswith('.pdf'):
            # Use PyMuPDFLoader with enhanced settings for better text extraction
            loader = PyMuPDFLoader(file_path=temp_file_path)
        elif file_name.lower().endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(file_path=temp_file_path)
        elif file_name.lower().endswith(('.eml', '.msg')):
            loader = UnstructuredEmailLoader(file_path=temp_file_path)
        elif file_name.lower().endswith('.xml'):
            loader = UnstructuredXMLLoader(file_path=temp_file_path)
        else:
            print(f"Using generic file loader for '{file_name}'.")
            loader = UnstructuredLoader(file_path=temp_file_path)

        print("Parsing document content...")
        documents = loader.load()

        # Enhanced post-processing for insurance documents
        documents = enhance_document_metadata(documents)
        documents = clean_insurance_content(documents)

        print(f"Document parsing complete. Loaded {len(documents)} pages/sections.")

    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")

    return documents

def enhance_document_metadata(documents: List[Document]) -> List[Document]:
    """Add enhanced metadata to documents for better retrieval."""
    enhanced_docs = []

    for i, doc in enumerate(documents):
        # Add page/section numbering
        doc.metadata['section_id'] = i + 1
        doc.metadata['content_length'] = len(doc.page_content)

        # Identify content type based on keywords
        content_lower = doc.page_content.lower()
        if any(keyword in content_lower for keyword in ['claim', 'claims', 'reimbursement']):
            doc.metadata['content_type'] = 'claims_process'
        elif any(keyword in content_lower for keyword in ['dependent', 'family', 'spouse', 'child']):
            doc.metadata['content_type'] = 'dependents'
        elif any(keyword in content_lower for keyword in ['grievance', 'complaint', 'appeal']):
            doc.metadata['content_type'] = 'grievance'
        elif any(keyword in content_lower for keyword in ['network', 'hospital', 'provider']):
            doc.metadata['content_type'] = 'network_providers'
        elif any(keyword in content_lower for keyword in ['policy', 'coverage', 'benefit']):
            doc.metadata['content_type'] = 'policy_details'
        else:
            doc.metadata['content_type'] = 'general'

        enhanced_docs.append(doc)

    return enhanced_docs

def clean_insurance_content(documents: List[Document]) -> List[Document]:
    """Clean and enhance insurance document content for better retrieval."""
    cleaned_docs = []

    for doc in documents:
        content = doc.page_content

        # Remove excessive whitespace but preserve structure
        lines = content.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line:  # Keep non-empty lines
                cleaned_lines.append(line)

        # Rejoin with consistent spacing
        cleaned_content = '\n'.join(cleaned_lines)

        # Skip very short or low-quality content
        if len(cleaned_content) < 50:
            continue

        # Create new document with cleaned content
        cleaned_doc = Document(
            page_content=cleaned_content,
            metadata=doc.metadata.copy()
        )

        cleaned_docs.append(cleaned_doc)

    return cleaned_docs

def download_hugging_face_embeddings():
    """
    Initialize Google Gemini embeddings with enhanced configuration.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY1")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GOOGLE_API_KEY1 environment variable not set.")

    # Initialize with specific task type for better embeddings
    embeddings = GoogleGenerativeAIEmbeddings( 
        model="models/text-embedding-004", 
        google_api_key=api_key,
        task_type="retrieval_document"  # Optimize for document retrieval
    )
    return embeddings

def extract_key_sections(documents: List[Document], section_keywords: List[str]) -> List[Document]:
    """Extract sections that are most relevant to insurance queries."""
    relevant_docs = []

    for doc in documents:
        content_lower = doc.page_content.lower()
        relevance_score = 0

        # Score based on keyword presence
        for keyword in section_keywords:
            if keyword.lower() in content_lower:
                relevance_score += content_lower.count(keyword.lower())

        # Add document if it has relevant content
        if relevance_score > 0 or len(doc.page_content) > 200:
            doc.metadata['relevance_score'] = relevance_score
            relevant_docs.append(doc)

    # Sort by relevance score (descending)
    relevant_docs.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)

    return relevant_docs

# Insurance-specific keywords for better content filtering
INSURANCE_KEYWORDS = [
    'claim', 'claims', 'dependent', 'coverage', 'policy', 'benefit', 'benefits',
    'reimbursement', 'network', 'provider', 'hospital', 'grievance', 'complaint',
    'appeal', 'procedure', 'process', 'document', 'documentation', 'form',
    'eligibility', 'age', 'limit', 'deadline', 'contact', 'email', 'phone',
    'department', 'office', 'support', 'customer', 'service'
]
