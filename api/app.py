import os
import asyncio
import hashlib
from typing import List, Tuple, Dict, Set
from datetime import datetime
import re

from fastapi import FastAPI, Depends, HTTPException, Security, Request as FastAPIRequest
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn
from pymongo import MongoClient

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from src.helper import load_doc_from_url
from src.prompt import get_advanced_scenario_prompt

# --- Configuration and Initialization ---
load_dotenv()

FAISS_INDEX_DIR = "faiss_cache"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

app = FastAPI(
    title="Advanced Scenario-Aware RAG API",
    description="Enhanced retrieval with text-embedding-004 and multi-strategy approach for complex scenarios.",
    version="12.0.0"
)

# --- Database Connection ---
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
if not MONGO_DB_URL:
    raise ValueError("MONGO_DB_URL must be set in the .env file.")
client = MongoClient(MONGO_DB_URL)
db = client.get_database("api_requests")
requests_collection = db.get_collection("requests")

# --- Authentication ---
API_KEY = os.getenv("BEARER_TOKEN")
auth_scheme = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    if not API_KEY or credentials.scheme != "Bearer" or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})
    return credentials

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the large document.")
    questions: List[str] = Field(..., description="A list of questions about the document.")

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Initialize RAG Components with Enhanced Models ---
def get_enhanced_embeddings():
    """Initialize with the newer text-embedding-004 model."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY1")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GOOGLE_API_KEY1 environment variable not set.")
    
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",  # Latest embedding model
        google_api_key=api_key,
        task_type="retrieval_document"
    )

embeddings = get_enhanced_embeddings()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY1"),
    temperature=0.0  # Most deterministic for factual queries
)

# Enhanced text splitter with insurance-specific separators
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,  # Larger chunks for better context
    chunk_overlap=300,  # Substantial overlap
    separators=[
        "\n\n\n",  # Major sections
        "\n\n",    # Paragraphs
        "\n",      # Lines
        ". ",      # Sentences
        " ",       # Words
        ""
    ]
)

# --- Advanced Question Analysis ---
class QuestionAnalyzer:
    def __init__(self):
        self.scenario_patterns = [
            r"while.*also",
            r"for.*involving.*what",
            r"when.*and.*how",
            r"process.*for.*and.*confirm",
            r"what.*needed.*and.*how",
            r"can.*continue.*if",
            r"also.*provide.*email",
            r"also.*confirm.*process"
        ]
        
        self.insurance_entities = {
            'claim_types': ['dental', 'medical', 'surgery', 'robotic surgery', 'treatment'],
            'relationships': ['spouse', 'daughter', 'sibling', 'dependent', 'child'],
            'processes': ['claim submission', 'name change', 'update', 'grievance'],
            'documents': ['supporting documents', 'bills', 'prescription', 'discharge summary'],
            'providers': ['hospital', 'network provider', 'apollo care'],
            'contacts': ['email', 'grievance', 'customer care', 'contact']
        }
    
    def is_complex_scenario(self, question: str) -> bool:
        """Enhanced scenario detection using patterns and entity count."""
        text_lower = question.lower()
        
        # Check for scenario patterns
        pattern_match = any(re.search(pattern, text_lower) for pattern in self.scenario_patterns)
        
        # Count insurance entities
        entity_count = 0
        for category, entities in self.insurance_entities.items():
            entity_count += sum(1 for entity in entities if entity in text_lower)
        
        # Count question indicators
        question_words = ['what', 'how', 'when', 'where', 'can', 'is', 'are', 'will']
        question_count = sum(1 for word in question_words if word in text_lower)
        
        return pattern_match or entity_count >= 3 or question_count >= 2
    
    def extract_key_concepts(self, question: str) -> Dict[str, List[str]]:
        """Extract key concepts for targeted retrieval."""
        text_lower = question.lower()
        concepts = {}
        
        for category, entities in self.insurance_entities.items():
            found_entities = [entity for entity in entities if entity in text_lower]
            if found_entities:
                concepts[category] = found_entities
        
        return concepts
    
    def generate_sub_queries(self, question: str) -> List[str]:
        """Generate multiple focused queries for better retrieval."""
        concepts = self.extract_key_concepts(question)
        sub_queries = [question]  # Always include original
        
        # Generate concept-based queries
        if 'claim_types' in concepts:
            for claim_type in concepts['claim_types']:
                sub_queries.append(f"{claim_type} claim process procedure")
                sub_queries.append(f"{claim_type} claim required documents")
        
        if 'relationships' in concepts:
            for relation in concepts['relationships']:
                sub_queries.append(f"{relation} dependent eligibility criteria")
                sub_queries.append(f"{relation} coverage age limit")
        
        if 'processes' in concepts:
            for process in concepts['processes']:
                sub_queries.append(f"{process} procedure steps")
        
        if 'providers' in concepts:
            sub_queries.append("network provider verification process")
            sub_queries.append("hospital empanelment check")
        
        if 'contacts' in concepts:
            sub_queries.append("grievance redressal email contact")
            sub_queries.append("customer care contact information")
        
        return list(set(sub_queries))  # Remove duplicates

analyzer = QuestionAnalyzer()

# --- Advanced Retrieval System ---
class AdvancedRetriever:
    def __init__(self, vector_store, documents):
        self.vector_store = vector_store
        self.documents = documents
        self.semantic_retriever = vector_store.as_retriever()
        
        # Create BM25 retriever for keyword matching
        doc_texts = [doc.page_content for doc in documents]
        self.bm25_retriever = BM25Retriever.from_texts(doc_texts)
        
        # Ensemble retriever combining both approaches
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.semantic_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]  # Favor semantic but include keyword matching
        )
    
    async def multi_query_retrieve(self, queries: List[str], k_per_query: int = 8) -> List[Document]:
        """Retrieve documents using multiple queries and combine results."""
        all_docs = []
        seen_content = set()
        
        for query in queries[:5]:  # Limit to 5 queries to avoid overload
            # Use ensemble retriever for each query
            try:
                # Semantic retrieval
                semantic_docs = await self.semantic_retriever.aget_relevant_documents(query)
                
                # BM25 retrieval
                bm25_docs = await self.bm25_retriever.aget_relevant_documents(query)
                
                # Combine and deduplicate
                combined_docs = semantic_docs[:k_per_query//2] + bm25_docs[:k_per_query//2]
                
                for doc in combined_docs:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        doc.metadata['query_used'] = query
                        all_docs.append(doc)
                        
            except Exception as e:
                print(f"Error retrieving for query '{query}': {e}")
                continue
        
        return all_docs[:20]  # Return top 20 unique documents
    
    async def context_aware_retrieve(self, question: str, concepts: Dict) -> List[Document]:
        """Retrieve documents with context awareness."""
        # Generate targeted queries
        queries = analyzer.generate_sub_queries(question)
        
        # Multi-query retrieval
        docs = await self.multi_query_retrieve(queries)
        
        # Context-based filtering and ranking
        scored_docs = []
        for doc in docs:
            score = self.calculate_relevance_score(doc, concepts, question)
            scored_docs.append((doc, score))
        
        # Sort by relevance score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:15]]
    
    def calculate_relevance_score(self, doc: Document, concepts: Dict, question: str) -> float:
        """Calculate relevance score based on concepts and question context."""
        content_lower = doc.page_content.lower()
        question_lower = question.lower()
        
        score = 0.0
        
        # Base score from content length (longer documents might have more info)
        score += min(len(doc.page_content) / 1000, 2.0)
        
        # Score based on concept matches
        for category, entities in concepts.items():
            for entity in entities:
                if entity in content_lower:
                    score += 2.0
                    # Bonus for exact phrase matches
                    if entity in content_lower:
                        score += 1.0
        
        # Score based on question word matches
        question_words = question_lower.split()
        for word in question_words:
            if len(word) > 3 and word in content_lower:
                score += 0.5
        
        # Bonus for insurance-specific terms
        insurance_terms = [
            'process', 'procedure', 'document', 'required', 'eligible',
            'contact', 'email', 'phone', 'department', 'days', 'limit'
        ]
        for term in insurance_terms:
            if term in content_lower:
                score += 0.3
        
        return score

# --- Enhanced Document Processing ---
def enhanced_group_pages(pages: List[Document], pages_per_group: int = 6) -> List[Document]:
    """Enhanced grouping that preserves section boundaries."""
    if len(pages) <= pages_per_group:
        return pages
    
    grouped_docs = []
    current_group = []
    current_length = 0
    max_group_length = 8000  # Maximum characters per group
    
    for page in pages:
        page_length = len(page.page_content)
        
        # If adding this page would exceed limits, finalize current group
        if (len(current_group) >= pages_per_group or 
            current_length + page_length > max_group_length) and current_group:
            
            # Create grouped document
            combined_content = "\n\n--- PAGE BREAK ---\n\n".join([p.page_content for p in current_group])
            metadata = current_group[0].metadata.copy()
            metadata['page_range'] = f"{current_group[0].metadata.get('page', 1)}-{current_group[-1].metadata.get('page', len(current_group))}"
            metadata['group_size'] = len(current_group)
            
            grouped_docs.append(Document(page_content=combined_content, metadata=metadata))
            
            # Start new group
            current_group = [page]
            current_length = page_length
        else:
            current_group.append(page)
            current_length += page_length
    
    # Add remaining pages
    if current_group:
        combined_content = "\n\n--- PAGE BREAK ---\n\n".join([p.page_content for p in current_group])
        metadata = current_group[0].metadata.copy()
        metadata['page_range'] = f"{current_group[0].metadata.get('page', 1)}-{current_group[-1].metadata.get('page', len(current_group))}"
        metadata['group_size'] = len(current_group)
        
        grouped_docs.append(Document(page_content=combined_content, metadata=metadata))
    
    return grouped_docs

async def process_group_to_embeddings(doc_group: Document) -> List[Tuple[str, List[float]]]:
    """Enhanced processing with better error handling and metadata preservation."""
    try:
        chunks = text_splitter.split_documents([doc_group])
        if not chunks:
            return []
        
        # Add source metadata to chunks
        for chunk in chunks:
            chunk.metadata.update(doc_group.metadata)
        
        chunk_texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings in batches to avoid rate limits
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i + batch_size]
            try:
                batch_embeddings = await embeddings.aembed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                await asyncio.sleep(0.1)  # Small delay to respect rate limits
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        return list(zip(chunk_texts, all_embeddings))
        
    except Exception as e:
        print(f"Error processing document group: {e}")
        return []

# --- Enhanced Answer Generation ---
async def generate_comprehensive_answer(question: str, retriever: AdvancedRetriever, question_index: int) -> str:
    """Generate comprehensive answers for complex scenario questions."""
    try:
        # Analyze question
        is_complex = analyzer.is_complex_scenario(question)
        concepts = analyzer.extract_key_concepts(question)
        
        print(f"Question {question_index} analysis - Complex: {is_complex}, Concepts: {list(concepts.keys())}")
        
        # Retrieve relevant documents
        if is_complex:
            relevant_docs = await retriever.context_aware_retrieve(question, concepts)
        else:
            relevant_docs = await retriever.semantic_retriever.aget_relevant_documents(question)
        
        print(f"Question {question_index}: Retrieved {len(relevant_docs)} documents")
        
        # Use advanced prompt
        qa_prompt = ChatPromptTemplate.from_template(get_advanced_scenario_prompt())
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        # Create a custom retrieval chain that uses our documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate answer
        result = await llm.ainvoke([
            {"role": "system", "content": get_advanced_scenario_prompt()},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ])
        
        answer = result.content.strip()
        
        # Post-process answer
        if len(answer) < 50 or any(phrase in answer.lower() for phrase in ["not available", "cannot find", "not specified"]):
            print(f"Question {question_index}: Answer too generic, trying fallback approach...")
            
            # Fallback: try with more documents and different approach
            fallback_docs = await retriever.multi_query_retrieve([question], k_per_query=15)
            fallback_context = "\n\n".join([doc.page_content for doc in fallback_docs])
            
            fallback_result = await llm.ainvoke([
                {"role": "system", "content": "You are an expert insurance analyst. Provide detailed, specific answers based on the document context. If information is partial, provide what is available and clearly state what is missing."},
                {"role": "user", "content": f"Context:\n{fallback_context}\n\nQuestion: {question}"}
            ])
            
            answer = fallback_result.content.strip()
        
        print(f"Question {question_index}: Generated answer length: {len(answer)} characters")
        return answer
        
    except Exception as e:
        print(f"Error generating answer for question {question_index}: {e}")
        return f"An error occurred while processing this question: {str(e)}"

# --- Main API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(fastapi_req: FastAPIRequest, request: HackRxRequest, user: str = Depends(get_current_user)):
    print(f"Processing request with {len(request.questions)} questions")
    
    url_hash = hashlib.sha256(request.documents.encode()).hexdigest()
    cache_path = os.path.join(FAISS_INDEX_DIR, url_hash)

    vector_store = None
    all_documents = None

    if os.path.exists(cache_path):
        print(f"Cache hit. Loading FAISS index from: {cache_path}")
        try:
            vector_store = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
            # Note: We'll need to reload documents for BM25 retriever
            pages = load_doc_from_url(request.documents)
            large_chunks = enhanced_group_pages(pages, pages_per_group=6)
            all_documents = []
            for chunk in large_chunks:
                chunk_docs = text_splitter.split_documents([chunk])
                all_documents.extend(chunk_docs)
        except Exception as e:
            print(f"Error loading from cache: {e}. Re-generating index.")
            vector_store = None
    
    if not vector_store:
        print(f"Cache miss. Generating new embeddings for URL: {request.documents}")
        
        pages = load_doc_from_url(request.documents)
        if not pages:
            raise HTTPException(status_code=400, detail="Failed to load or parse document from URL.")
        
        print(f"Loaded {len(pages)} pages from document")
        
        # Enhanced grouping
        large_chunks = enhanced_group_pages(pages, pages_per_group=6)
        print(f"Created {len(large_chunks)} large chunks")
        
        # Process embeddings
        embedding_tasks = [process_group_to_embeddings(chunk) for chunk in large_chunks]
        results = await asyncio.gather(*embedding_tasks)
        
        text_embedding_pairs = [item for sublist in results for item in sublist]
        if not text_embedding_pairs:
            raise HTTPException(status_code=500, detail="Could not process document into embeddings.")
        
        print(f"Generated embeddings for {len(text_embedding_pairs)} total chunks.")

        # Create vector store and documents list
        vector_store = FAISS.from_embeddings(text_embeddings=text_embedding_pairs, embedding=embeddings)
        
        # Create document list for BM25
        all_documents = []
        for chunk in large_chunks:
            chunk_docs = text_splitter.split_documents([chunk])
            all_documents.extend(chunk_docs)
        
        print(f"Saving new FAISS index to cache: {cache_path}")
        vector_store.save_local(cache_path)

    # Create advanced retriever
    advanced_retriever = AdvancedRetriever(vector_store, all_documents)
    
    print(f"Processing {len(request.questions)} questions with parallel execution...")
    
    # Process questions in parallel
    question_tasks = [
        generate_comprehensive_answer(question, advanced_retriever, i+1) 
        for i, question in enumerate(request.questions)
    ]
    
    answers = await asyncio.gather(*question_tasks)

    # Log the request
    try:
        log_entry = {
            "timestamp": datetime.utcnow(),
            "request_ip": fastapi_req.client.host,
            "request_body": request.dict(),
            "response_body": {"answers": answers},
            "user": user.credentials,
        }
        requests_collection.insert_one(log_entry)
    except Exception as log_e:
        print(f"Warning: Failed to log request to MongoDB. Error: {log_e}")

    return HackRxResponse(answers=answers)

# --- Health Check Endpoints ---
@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Advanced Scenario-Aware RAG API with text-embedding-004 is live."}

@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "healthy", "embedding_model": "text-embedding-004"}
