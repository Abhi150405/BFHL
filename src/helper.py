# src/helper.py - Updated with text-embedding-004

import os
import io
import requests
import tempfile
import re
from urllib.parse import urlparse
from typing import List, Dict, Set
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
    Includes improved metadata extraction and content cleaning.
    """
    file_name = os.path.basename(urlparse(url).path)
    
    # Special handling for RSS feeds
    if file_name.lower().endswith('.rss'):
        print(f"Processing RSS feed from {url}...")
        loader = RSSFeedLoader(urls=[url])
        documents = loader.load()
        return enhance_insurance_documents(documents)
    
    try:
        print(f"Downloading document from {url}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        suffix = os.path.splitext(file_name)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        print(f"Processing '{file_name}' from temporary file...")
        
        loader = None
        if file_name.lower().endswith('.pdf'):
            # Enhanced PDF loading with better text extraction
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
        documents = enhance_insurance_documents(documents)
        documents = clean_and_structure_content(documents)
        
        print(f"Document parsing complete. Processed {len(documents)} pages/sections.")
        
    finally:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")

    return documents

def enhance_insurance_documents(documents: List[Document]) -> List[Document]:
    """Enhanced document processing with insurance-specific metadata and structure detection."""
    enhanced_docs = []
    
    for i, doc in enumerate(documents):
        content = doc.page_content
        metadata = doc.metadata.copy()
        
        # Add basic metadata
        metadata['section_id'] = i + 1
        metadata['content_length'] = len(content)
        metadata['source_page'] = metadata.get('page', i + 1)
        
        # Detect section types based on content analysis
        content_lower = content.lower()
        section_type = detect_section_type(content_lower)
        metadata['section_type'] = section_type
        
        # Extract key entities and topics
        entities = extract_insurance_entities(content)
        metadata['entities'] = entities
        
        # Calculate content quality score
        quality_score = calculate_content_quality(content)
        metadata['quality_score'] = quality_score
        
        # Skip low-quality content
        if quality_score < 0.3:
            continue
        
        enhanced_docs.append(Document(page_content=content, metadata=metadata))
    
    return enhanced_docs

def detect_section_type(content_lower: str) -> str:
    """Detect the type of insurance content section."""
    section_patterns = {
        'claims_process': [
            'claim submission', 'claim process', 'reimbursement', 'medical bills',
            'treatment bills', 'cashless', 'settlement', 'claim settlement'
        ],
        'dependent_eligibility': [
            'dependent', 'family member', 'spouse coverage', 'child coverage',
            'age limit', 'eligibility criteria', 'family floater'
        ],
        'network_providers': [
            'network hospital', 'network provider', 'empaneled', 'tie-up',
            'cashless facility', 'pre-authorization', 'network list'
        ],
        'grievance_redressal': [
            'grievance', 'complaint', 'ombudsman', 'appeal', 'dispute',
            'customer care', 'redressal', 'escalation'
        ],
        'policy_terms': [
            'policy terms', 'coverage', 'benefit', 'exclusion', 'waiting period',
            'sum insured', 'premium', 'renewal'
        ],
        'contact_information': [
            'contact us', 'customer care', 'toll free', 'email', 'address',
            'branch office', 'head office', 'phone number'
        ],
        'procedures': [
            'procedure', 'process', 'step', 'how to', 'method', 'way to',
            'required documents', 'documentation'
        ]
    }
    
    section_scores = {}
    for section_type, patterns in section_patterns.items():
        score = sum(1 for pattern in patterns if pattern in content_lower)
        if score > 0:
            section_scores[section_type] = score
    
    if section_scores:
        return max(section_scores, key=section_scores.get)
    
    return 'general'

def extract_insurance_entities(content: str) -> Dict[str, List[str]]:
    """Extract insurance-specific entities from content."""
    entities = {
        'claim_types': [],
        'relationships': [],
        'documents': [],
        'timeframes': [],
        'contact_info': [],
        'amounts': [],
        'medical_terms': []
    }
    
    content_lower = content.lower()
    
    # Define entity patterns
    entity_patterns = {
        'claim_types': [
            'dental claim', 'medical claim', 'hospitalization claim', 'surgery claim',
            'maternity claim', 'outpatient claim', 'emergency claim'
        ],
        'relationships': [
            'spouse', 'husband', 'wife', 'son', 'daughter', 'child', 'children',
            'parent', 'father', 'mother', 'sibling', 'brother', 'sister'
        ],
        'documents': [
            'discharge summary', 'medical bills', 'prescription', 'investigation reports',
            'doctor certificate', 'claim form', 'id proof', 'address proof',
            'marriage certificate', 'birth certificate'
        ],
        'medical_terms': [
            'surgery', 'operation', 'treatment', 'hospitalization', 'consultation',
            'diagnosis', 'procedure', 'therapy', 'medication', 'robotic surgery'
        ]
    }
    
    # Extract entities using pattern matching
    for entity_type, patterns in entity_patterns.items():
        found_entities = [pattern for pattern in patterns if pattern in content_lower]
        entities[entity_type] = found_entities
    
    # Extract timeframes using regex
    timeframe_patterns = [
        r'\d+\s+days?', r'\d+\s+months?', r'\d+\s+years?', r'\d+\s+hours?',
        r'within\s+\d+\s+days?', r'after\s+\d+\s+days?', r'before\s+\d+\s+days?'
    ]
    
    for pattern in timeframe_patterns:
        matches = re.findall(pattern, content_lower)
        entities['timeframes'].extend(matches)
    
    # Extract contact information using regex
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b|\b\d{10}\b|\+91[\s-]?\d{10}'
    
    emails = re.findall(email_pattern, content)
    phones = re.findall(phone_pattern, content)
    
    entities['contact_info'].extend(emails + phones)
    
    # Extract amounts
    amount_pattern = r'₹\s*\d+(?:,\d+)*|Rs\.?\s*\d+(?:,\d+)*|\d+\s*lakhs?|\d+\s*crores?'
    amounts = re.findall(amount_pattern, content, re.IGNORECASE)
    entities['amounts'] = amounts
    
    return entities

def calculate_content_quality(content: str) -> float:
    """Calculate content quality score based on various factors."""
    if not content or len(content.strip()) < 20:
        return 0.0
    
    score = 0.0
    content_lower = content.lower()
    
    # Length score (normalized)
    length_score = min(len(content) / 1000, 1.0)
    score += length_score * 0.2
    
    # Insurance keyword density
    insurance_keywords = [
        'claim', 'policy', 'coverage', 'benefit', 'premium', 'deductible',
        'reimbursement', 'dependent', 'eligible', 'procedure', 'document',
        'hospital', 'treatment', 'medical', 'grievance', 'contact'
    ]
    
    keyword_count = sum(1 for keyword in insurance_keywords if keyword in content_lower)
    keyword_score = min(keyword_count / 10, 1.0)
    score += keyword_score * 0.3
    
    # Structure score (presence of lists, numbers, clear formatting)
    structure_indicators = [
        '\n1.', '\n2.', '\n•', '\n-', 'step 1', 'step 2', 'procedure',
        'process', 'required', 'contact:', 'email:', 'phone:'
    ]
    
    structure_count = sum(1 for indicator in structure_indicators if indicator in content_lower)
    structure_score = min(structure_count / 5, 1.0)
    score += structure_score * 0.3
    
    # Contact information bonus
    if any(term in content_lower for term in ['email', '@', 'phone', 'contact']):
        score += 0.1
    
    # Process information bonus
    if any(term in content_lower for term in ['step', 'procedure', 'process', 'how to']):
        score += 0.1
    
    return min(score, 1.0)

def clean_and_structure_content(documents: List[Document]) -> List[Document]:
    """Clean and structure content for better retrieval."""
    cleaned_docs = []
    
    for doc in documents:
        content = doc.page_content
        
        # Remove excessive whitespace while preserving structure
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Multiple newlines to double
        content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces to single
        
        # Fix common OCR errors in insurance documents
        content = re.sub(r'\bcliam\b', 'claim', content, flags=re.IGNORECASE)
        content = re.sub(r'\bpolciy\b', 'policy', content, flags=re.IGNORECASE)
        content = re.sub(r'\bdocuemnt\b', 'document', content, flags=re.IGNORECASE)
        
        # Ensure proper sentence spacing
        content = re.sub(r'\.([A-Z])', r'. \1', content)
        
        # Structure enhancement: Add clear section breaks
        content = re.sub(r'\n([A-Z][A-Z\s]+):\s*\n', r'\n\n=== \1 ===\n', content)
        
        # Skip very short content
        if len(content.strip()) < 100:
            continue
        
        cleaned_doc = Document(
            page_content=content.strip(),
            metadata=doc.metadata.copy()
        )
        
        cleaned_docs.append(cleaned_doc)
    
    return cleaned_docs

def download_hugging_face_embeddings():
    """
    Initialize with the latest text-embedding-004 model for better semantic understanding.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY1")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GOOGLE_API_KEY1 environment variable not set.")
    
    # Use the latest text-embedding-004 model with optimized settings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",  # Latest and most capable model
        google_api_key=api_key,
        task_type="retrieval_document",  # Optimized for document retrieval
        # Additional parameters for better performance
        title="Insurance Document Retrieval"  # Optional: helps with context
    )
    
    return embeddings

def filter_relevant_sections(documents: List[Document], query_concepts: List[str]) -> List[Document]:
    """Filter documents to keep only sections relevant to the query concepts."""
    if not query_concepts:
        return documents
    
    relevant_docs = []
    
    for doc in documents:
        content_lower = doc.page_content.lower()
        relevance_score = 0
        
        # Score based on concept matches
        for concept in query_concepts:
            if concept.lower() in content_lower:
                relevance_score += content_lower.count(concept.lower())
        
        # Include document if it has any relevance or is high quality
        quality_threshold = 0.7
        if (relevance_score > 0 or 
            doc.metadata.get('quality_score', 0) > quality_threshold or
            doc.metadata.get('section_type') in ['claims_process', 'contact_information', 'procedures']):
            
            doc.metadata['relevance_score'] = relevance_score
            relevant_docs.append(doc)
    
    # Sort by relevance score
    relevant_docs.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
    
    return relevant_docs

# Insurance-specific constants
INSURANCE_SECTION_KEYWORDS = {
    'claims': ['claim', 'reimbursement', 'settlement', 'cashless', 'bills'],
    'dependents': ['dependent', 'family', 'spouse', 'child', 'age limit'],
    'network': ['network', 'provider', 'hospital', 'empaneled', 'tie-up'],
    'grievance': ['grievance', 'complaint', 'ombudsman', 'appeal', 'redressal'],
    'contact': ['contact', 'customer care', 'email', 'phone', 'address'],
    'eligibility': ['eligible', 'criteria', 'condition', 'requirement', 'qualify'],
    'procedure': ['procedure', 'process', 'step', 'method', 'way', 'how to']
}

CRITICAL_TERMS = [
    'email', 'phone', 'contact', 'days', 'age', 'limit', 'required',
    'documents', 'process', 'procedure', 'eligible', 'criteria'
]
