import os
import asyncio
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

# Updated imports for enhanced functionality
from src.helper import load_doc_from_url, download_hugging_face_embeddings
from src.prompt import get_scenario_prompt, get_simple_prompt

# --- Configuration and Initialization ---
load_dotenv()

app = FastAPI(
    title="Enhanced Scenario-Aware RAG API",
    description="Handles complex scenario-based questions with improved context understanding.",
    version="11.0.0"
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
    temperature=0.1  # Lower temperature for more consistent answers
)
if not os.getenv("GOOGLE_API_KEY1"):
    raise ValueError("GOOGLE_API_KEY1 must be set in the .env file.")

# Improved text splitter with better chunk handling
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # Slightly smaller chunks for better granularity
    chunk_overlap=200,   # More overlap to preserve context
    separators=["\n\n", "\n", ". ", " ", ""]  # Better separation logic
)

# --- Enhanced Helper Functions ---

def is_scenario_question(question: str) -> bool:
    """Detect if a question is scenario-based and complex."""
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

def extract_sub_questions(question: str) -> List[str]:
    """Extract individual sub-questions from a complex scenario question."""
    parts = []
    connectors = [", also ", " and ", ", and ", ", what ", ", how ", ", when ", ", where ", ", can ", ", is "]
    
    current_part = question
    for connector in connectors:
        if connector in current_part.lower():
            split_parts = current_part.split(connector, 1)
            parts.append(split_parts[0].strip())
            current_part = split_parts[1].strip()
    
    if current_part:
        parts.append(current_part.strip())
    
    cleaned_parts = []
    for part in parts:
        part = part.strip()
        if part and len(part) > 10:
            if not part.endswith('?'):
                part += '?'
            cleaned_parts.append(part)
    
    return cleaned_parts if len(cleaned_parts) > 1 else [question]

def group_pages(pages: List[Document], pages_per_group: int = 8, overlap_pages: int = 3) -> List[Document]:
    """Enhanced grouping with better overlap for context preservation."""
    grouped_docs = []
    step = max(1, pages_per_group - overlap_pages)
        
    for i in range(0, len(pages), step):
        group = pages[i : i + pages_per_group]
        if not group:
            break
            
        combined_content = "\n\n".join([page.page_content for page in group])
        first_page_metadata = group[0].metadata if group else {}
        
        first_page_metadata['page_range'] = f"{i+1}-{min(i+pages_per_group, len(pages))}"
        
        grouped_docs.append(Document(page_content=combined_content, metadata=first_page_metadata))
    return grouped_docs

async def process_group_to_embeddings(doc_group: Document) -> List[Tuple[str, List[float]]]:
    """Enhanced processing with better error handling."""
    try:
        chunks = text_splitter.split_documents([doc_group])
        if not chunks: 
            return []
        
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = await embeddings.aembed_documents(chunk_texts)
        return list(zip(chunk_texts, chunk_embeddings))
    except Exception as e:
        print(f"Error processing document group: {e}")
        return []

def clean_answer(answer: str) -> str:
    """Cleans the answer to be on a single line and removes markdown."""
    cleaned = answer.replace("\n", " ").replace("**", "")
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    return " ".join(cleaned.split()).strip()

# --- Enhanced Q&A Processing ---

async def get_enhanced_answer(question: str, retriever, is_scenario: bool = False):
    """Enhanced answer generation with scenario-aware prompting."""
    try:
        if is_scenario:
            search_kwargs = {"k": 10, "fetch_k": 20}
        else:
            search_kwargs = {"k": 5, "fetch_k": 10}
        
        retriever.search_kwargs = search_kwargs
        
        if is_scenario:
            qa_prompt = ChatPromptTemplate.from_template(get_scenario_prompt())
        else:
            qa_prompt = ChatPromptTemplate.from_template(get_simple_prompt())
        
        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        result = await rag_chain.ainvoke({"input": question})
        answer = result.get("answer", "Could not find an answer.").strip()
        
        if len(answer) < 20 or "not available" in answer.lower():
            retriever.search_kwargs = {"k": 15, "fetch_k": 30}
            result = await rag_chain.ainvoke({"input": question})
            answer = result.get("answer", "Could not find an answer.").strip()
        
        return clean_answer(answer)
        
    except Exception as e:
        print(f"Error generating answer for question '{question}': {e}")
        return "An error occurred while processing this question."

async def process_scenario_question(question: str, retriever) -> str:
    """Process complex scenario questions by breaking them down."""
    if not is_scenario_question(question):
        return await get_enhanced_answer(question, retriever, is_scenario=False)
    
    sub_questions = extract_sub_questions(question)
    
    if len(sub_questions) == 1:
        return await get_enhanced_answer(question, retriever, is_scenario=True)
    
    print(f"Processing scenario question with {len(sub_questions)} parts in parallel")
    
    sub_tasks = [get_enhanced_answer(sub_q, retriever, is_scenario=True) for sub_q in sub_questions]
    sub_answers = await asyncio.gather(*sub_tasks)
    
    # Combine answers into a single-line response
    combined_answer = " ".join(sub_answers)
    
    if len(combined_answer) > 500:
        unified_answer = await get_enhanced_answer(question, retriever, is_scenario=True)
        if len(unified_answer) > 50 and "not available" not in unified_answer.lower():
            return unified_answer
    
    return combined_answer

# --- Main API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(fastapi_req: FastAPIRequest, request: HackRxRequest, user: str = Depends(get_current_user)):
    # --- Caching logic has been removed. Embeddings are generated on every request. ---
    
    print(f"Generating new embeddings for URL: {request.documents}")
    
    pages = load_doc_from_url(request.documents)
    if not pages:
        raise HTTPException(status_code=400, detail="Failed to load or parse document from URL.")
    
    large_chunks = group_pages(pages, pages_per_group=8, overlap_pages=3)
    
    embedding_tasks = [process_group_to_embeddings(chunk) for chunk in large_chunks]
    results = await asyncio.gather(*embedding_tasks)
    
    text_embedding_pairs = [item for sublist in results for item in sublist]
    if not text_embedding_pairs:
        raise HTTPException(status_code=500, detail="Could not process document into embeddings.")
    print(f"Generated embeddings for {len(text_embedding_pairs)} total chunks.")

    # Create the vector store directly from the generated embeddings
    vector_store = FAISS.from_embeddings(text_embeddings=text_embedding_pairs, embedding=embeddings)
    
    retriever = vector_store.as_retriever()
    
    print(f"Processing {len(request.questions)} questions with enhanced scenario handling...")
    
    qa_tasks = [process_scenario_question(q, retriever) for q in request.questions]
    answers = await asyncio.gather(*qa_tasks)

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
    return {"message": "Enhanced Scenario-Aware RAG API is live. See /docs for endpoint details."}

@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "healthy"}
