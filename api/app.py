# main.py

import os
import asyncio
from typing import List, Tuple
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, Security, Request as FastAPIRequest
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn
from pymongo import MongoClient

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Imports for BM25 Hybrid Search
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Custom Module Imports
from src.helper import load_doc_from_url, download_hugging_face_embeddings
from src.prompt import get_scenario_prompt, get_simple_prompt

# --- Configuration and Initialization ---
load_dotenv()

app = FastAPI(
    title="Enhanced Scenario-Aware RAG API",
    description="Handles complex scenario-based questions with user-defined parallel processing.",
    version="14.0.0" # Version updated for user-specified parallel logic
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

# --- Initialize RAG Components ---
embeddings = download_hugging_face_embeddings()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY1"),
    temperature=0.5
)
if not os.getenv("GOOGLE_API_KEY1"):
    raise ValueError("GOOGLE_API_KEY1 must be set in the .env file.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# --- Helper Functions (including your parallel processing pattern) ---

async def embed_texts_in_parallel(texts: List[str]) -> List[List[float]]:
    """
    A helper function to embed a list of texts using the async method.
    This function will be called by each parallel worker.
    """
    return await embeddings.aembed_documents(texts)

def is_scenario_question(question: str) -> bool:
    scenario_indicators = [
        "while", "for a", "involving", "in case of", "when", "if",
        "also confirm", "also provide", "what supporting", "how to",
        "can a", "is it possible", "what happens if", "scenario",
        "situation", "process for", "steps to", "procedure for"
    ]
    question_indicators = ["what", "how", "when", "where", "why", "can", "is", "are", "does", "do"]
    question_count = sum(1 for indicator in question_indicators if indicator in question.lower())
    has_scenario_indicators = any(indicator in question.lower() for indicator in scenario_indicators)
    return has_scenario_indicators or question_count >= 2

def clean_answer(answer: str) -> str:
    cleaned = answer.replace("\n", " ").replace("**", "")
    return " ".join(cleaned.split()).strip()

# --- Q&A Processing ---

async def get_enhanced_answer(question: str, retriever, is_scenario: bool = False):
    try:
        k_value = 10 if is_scenario else 5
        for r in retriever.retrievers:
            if hasattr(r, 'search_kwargs'):
                r.search_kwargs = {"k": k_value}
            if hasattr(r, 'k'):
                r.k = k_value
        
        qa_prompt = ChatPromptTemplate.from_template(get_scenario_prompt() if is_scenario else get_simple_prompt())
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        result = await rag_chain.ainvoke({"input": question})
        answer = result.get("answer", "Could not find an answer.").strip()
        
        if len(answer) < 20 or "not available" in answer.lower():
            for r in retriever.retrievers:
                if hasattr(r, 'search_kwargs'):
                    r.search_kwargs = {"k": 15}
                if hasattr(r, 'k'):
                    r.k = 15
            result = await rag_chain.ainvoke({"input": question})
            answer = result.get("answer", "Could not find an answer.").strip()
        
        return clean_answer(answer)
        
    except Exception as e:
        print(f"Error generating answer for question '{question}': {e}")
        return "An error occurred while processing this question."

async def process_scenario_question(question: str, retriever) -> str:
    is_scenario = is_scenario_question(question)
    return await get_enhanced_answer(question, retriever, is_scenario=is_scenario)

# --- Main API Endpoint (Using Your Parallel Logic) ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(fastapi_req: FastAPIRequest, request: HackRxRequest, user: str = Depends(get_current_user)):
    
    print(f"Loading document from URL: {request.documents}")
    pages = load_doc_from_url(request.documents)
    if not pages:
        raise HTTPException(status_code=400, detail="Failed to load or parse document from URL.")

    # 1. Split documents into chunks for both retrievers
    text_chunks = text_splitter.split_documents(pages)
    if not text_chunks:
        raise HTTPException(status_code=500, detail="Could not process document into chunks.")
    print(f"Created {len(text_chunks)} document chunks.")

    # 2. Create the BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(text_chunks)
    
    # 3. Generate embeddings using your specified parallel pattern
    print("Generating embeddings for FAISS using specified parallel pattern...")
    
    # To avoid hitting API rate limits with too many parallel requests, we batch the chunks.
    batch_size = 100  # Number of chunks to embed in each parallel task
    chunk_texts = [chunk.page_content for chunk in text_chunks]
    embedding_tasks = []

    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i : i + batch_size]
        # Create a task for each batch
        embedding_tasks.append(embed_texts_in_parallel(batch))

    # Run all batch embedding tasks in parallel
    results_from_tasks = await asyncio.gather(*embedding_tasks)
    
    # Flatten the list of lists into a single list of embeddings
    chunk_embeddings = [embedding for batch_result in results_from_tasks for embedding in batch_result]
    
    if not chunk_embeddings:
        raise HTTPException(status_code=500, detail="Failed to generate document embeddings.")

    text_embedding_pairs = list(zip(chunk_texts, chunk_embeddings))
    
    # 4. Create the FAISS vector store from the pre-computed parallel embeddings
    vector_store = FAISS.from_embeddings(text_embeddings=text_embedding_pairs, embedding=embeddings)
    faiss_retriever = vector_store.as_retriever()
    
    # 5. Create the EnsembleRetriever
    print("Initializing EnsembleRetriever with BM25 and FAISS...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )
    
    # 6. Process all questions in parallel
    print(f"Processing {len(request.questions)} questions with hybrid retrieval (parallel execution)...")
    qa_tasks = [process_scenario_question(q, ensemble_retriever) for q in request.questions]
    answers = await asyncio.gather(*qa_tasks)

    # Log the request and response
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
    return {"message": "Enhanced Scenario-Aware RAG API with Hybrid Search is live. See /docs for details."}

@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "healthy"}
