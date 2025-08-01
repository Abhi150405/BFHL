import os
import asyncio
import hashlib
from typing import List, Tuple, Dict
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, Security, Request as FastAPIRequest
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn
from pymongo import MongoClient

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# This assumes your helper and prompt files are in a 'src' directory
from src.helper import load_doc_from_url, download_hugging_face_embeddings
from src.prompt import system_prompt

# --- Configuration and Initialization ---
load_dotenv()

# Define a directory to store the cached FAISS indexes
FAISS_INDEX_DIR = "faiss_cache"
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)


app = FastAPI(
    title="Cached Parallel RAG API",
    description="Uses a file-based cache to skip embedding generation for previously processed documents.",
    version="10.0.0"
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
    """Dependency to validate the Bearer token."""
    if not API_KEY or credentials.scheme != "Bearer" or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})
    return credentials

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the large document.")
    questions: List[str] = Field(..., description="A list of questions about the document.")

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Initialize RAG Components ---
embeddings = download_hugging_face_embeddings()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY1")
)
if not os.getenv("GOOGLE_API_KEY1"):
    raise ValueError("GOOGLE_API_KEY1 must be set in the .env file.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# --- Helper Functions for Parallel Group Processing ---

def group_pages(pages: List[Document], pages_per_group: int = 10, overlap_pages: int = 2) -> List[Document]:
    """Groups document pages into larger Document objects for processing, with overlap."""
    grouped_docs = []
    # Ensure the step is at least 1 to avoid infinite loops
    step = pages_per_group - overlap_pages if pages_per_group > overlap_pages else 1
        
    for i in range(0, len(pages), step):
        group = pages[i : i + pages_per_group]
        if not group:
            break # Exit if no more pages are left
            
        combined_content = "\n\n".join([page.page_content for page in group])
        first_page_metadata = group[0].metadata if group else {}
        grouped_docs.append(Document(page_content=combined_content, metadata=first_page_metadata))
    return grouped_docs

async def process_group_to_embeddings(doc_group: Document) -> List[Tuple[str, List[float]]]:
    """
    Takes a large Document group, chunks it, and generates embeddings.
    """
    try:
        chunks = text_splitter.split_documents([doc_group])
        if not chunks: return []
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = await embeddings.aembed_documents(chunk_texts)
        return list(zip(chunk_texts, chunk_embeddings))
    except Exception as e:
        print(f"Error processing a document group (metadata: {doc_group.metadata}): {e}")
        return []

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(fastapi_req: FastAPIRequest, request: HackRxRequest, user: str = Depends(get_current_user)):
    # Create a safe filename for the cache from a hash of the URL
    url_hash = hashlib.sha256(request.documents.encode()).hexdigest()
    cache_path = os.path.join(FAISS_INDEX_DIR, url_hash)

    vector_store = None

    if os.path.exists(cache_path):
        # --- CACHE HIT ---
        print(f"Cache hit. Loading FAISS index from: {cache_path}")
        try:
            vector_store = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading from cache: {e}. Re-generating index.")
            vector_store = None # Force re-generation
    
    if not vector_store:
        # --- CACHE MISS ---
        print(f"Cache miss. Generating new embeddings for URL: {request.documents}")
        
        # 1. Load and process the document
        pages = load_doc_from_url(request.documents)
        if not pages:
            raise HTTPException(status_code=400, detail="Failed to load or parse document from URL.")
        
        large_chunks = group_pages(pages, pages_per_group=10)
        
        embedding_tasks = [process_group_to_embeddings(chunk) for chunk in large_chunks]
        results = await asyncio.gather(*embedding_tasks)
        
        text_embedding_pairs = [item for sublist in results for item in sublist]
        if not text_embedding_pairs:
            raise HTTPException(status_code=500, detail="Could not process document into embeddings.")
        print(f"Generated embeddings for {len(text_embedding_pairs)} total chunks.")

        # 2. Create Vector Store and save to cache
        vector_store = FAISS.from_embeddings(text_embeddings=text_embedding_pairs, embedding=embeddings)
        print(f"Saving new FAISS index to cache: {cache_path}")
        vector_store.save_local(cache_path)

    # --- Continue to Q&A with the (newly created or cached) vector store ---
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    async def get_answer(question: str):
        qa_prompt = ChatPromptTemplate.from_template(system_prompt) # Using the single-question prompt
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        result = await rag_chain.ainvoke({"input": question})
        return result.get("answer", "Could not find an answer.").strip()

    print(f"Answering {len(request.questions)} questions in parallel...")
    qa_tasks = [get_answer(q) for q in request.questions]
    answers = await asyncio.gather(*qa_tasks)

    # Log and return
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


# --- Root and Health Check Endpoints ---
@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "API is live. See /docs for endpoint details."}

@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "healthy"}
