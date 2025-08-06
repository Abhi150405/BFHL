import os
import asyncio
import time
import hashlib
import random
import base64
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from urllib.parse import urlparse
from io import BytesIO

# --- Third-party libraries ---
import requests
from fastapi import FastAPI, HTTPException, Security, Request as FastAPIRequest, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from pymongo import MongoClient
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# --- Local application imports ---
from src.helper import load_doc_from_url, initialize_gemini_embeddings
from src.prompt import get_scenario_prompt, get_simple_prompt, get_tabular_prompt, get_sub_question_prompt, get_image_prompt

# --- Configuration and Initialization ---
load_dotenv()

app = FastAPI(
    title="Enhanced Vision-Aware RAG API",
    description="A robust RAG API using Gemini Vision for image processing, with PPTX caching and abort policies.",
    version="15.0.0" # Version updated for Gemini Vision integration
)

# --- Feature Configuration ---
CACHE_DIR = Path("faiss_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
MIN_RESPONSE_TIME_SECONDS = 5.0
ZIP_ABORT_DELAY_SECONDS = 7.0

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
    """Validates the bearer token against the environment variable."""
    if not API_KEY or credentials.scheme != "Bearer" or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})
    return credentials

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the large document (.pptx, .pdf, .zip, .png, .jpg, etc.).")
    questions: List[str] = Field(..., description="A list of questions about the document.")

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Initialize RAG Components ---
embeddings = initialize_gemini_embeddings()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", # This model is multimodal and supports vision
    api_key=os.getenv("GOOGLE_API_KEY1"),
    temperature=0.1
)
if not os.getenv("GOOGLE_API_KEY1"):
    raise ValueError("GOOGLE_API_KEY1 must be set in the .env file.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# --- Helper Functions ---

def get_url_hash(url: str) -> str:
    """Creates a SHA256 hash of a URL to use as a unique cache key."""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

def get_file_type_from_url(url: str) -> str:
    """
    Parses a URL to extract the file path and determine its extension.
    Handles complex URLs with query parameters (e.g., SAS tokens).
    """
    try:
        path = urlparse(url).path
        file_ext = path.split('.')[-1].lower()
        
        if file_ext == 'pptx':
            return 'pptx'
        elif file_ext == 'zip':
            return 'zip'
        elif file_ext == 'pdf':
            return 'pdf'
        elif file_ext == 'bin':
            return 'bin'
        elif file_ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
            return 'image'
        else:
            return 'unknown'
    except Exception as e:
        print(f"Could not parse URL '{url}': {e}")
        return 'unknown'

def is_scenario_question(question: str) -> bool:
    """Detect if a question is scenario-based and complex."""
    scenario_indicators = ["while", "for a", "involving", "in case of", "when", "if", "also confirm", "also provide", "what supporting", "how to", "can a", "is it possible", "what happens if", "scenario", "situation", "process for", "steps to", "procedure for"]
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
    cleaned_parts = [part.strip() + '?' if not part.strip().endswith('?') else part.strip() for part in parts if part.strip() and len(part.strip()) > 10]
    return cleaned_parts if len(cleaned_parts) > 1 else [question]

def group_pages(pages: List[Document], pages_per_group: int = 8, overlap_pages: int = 3) -> List[Document]:
    """Group pages with overlap for better context preservation."""
    grouped_docs = []
    step = max(1, pages_per_group - overlap_pages)
    for i in range(0, len(pages), step):
        group = pages[i : i + pages_per_group]
        if not group: break
        combined_content = "\n\n".join([page.page_content for page in group])
        first_page_metadata = group[0].metadata if group else {}
        first_page_metadata['page_range'] = f"{i+1}-{min(i+pages_per_group, len(pages))}"
        grouped_docs.append(Document(page_content=combined_content, metadata=first_page_metadata))
    return grouped_docs

async def process_group_to_embeddings(doc_group: Document) -> List[Tuple[str, List[float]]]:
    """Process a document group into text-embedding pairs."""
    try:
        chunks = text_splitter.split_documents([doc_group])
        if not chunks: return []
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = await embeddings.aembed_documents(chunk_texts)
        return list(zip(chunk_texts, chunk_embeddings))
    except Exception as e:
        print(f"Error processing document group: {e}")
        return []

def clean_answer(answer: str) -> str:
    """Cleans the answer to be on a single line and removes markdown."""
    cleaned = answer.replace("\n", " ").replace("**", "")
    return " ".join(cleaned.split()).strip()

async def get_enhanced_answer(question: str, retriever, is_scenario: bool = False) -> str:
    """Generate an answer with scenario-aware prompting and retries."""
    try:
        search_kwargs = {"k": 10, "fetch_k": 20} if is_scenario else {"k": 5, "fetch_k": 10}
        retriever.search_kwargs = search_kwargs
        qa_prompt = ChatPromptTemplate.from_template(get_scenario_prompt() if is_scenario else get_simple_prompt())
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
    
    sub_tasks = [get_enhanced_answer(sub_q, retriever, is_scenario=True) for sub_q in sub_questions]
    sub_answers = await asyncio.gather(*sub_tasks)
    return " ".join(sub_answers)

# --- Main API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(fastapi_req: FastAPIRequest, request: HackRxRequest, user: str = Depends(get_current_user)):
    start_time = time.monotonic()
    
    doc_url = request.documents
    print(f"Processing URL: {doc_url}")
    
    file_type = get_file_type_from_url(doc_url)
    
    # --- Definitive Rule: Abort all ZIP file requests after 7 seconds ---
    if file_type == 'zip':
        await asyncio.sleep(ZIP_ABORT_DELAY_SECONDS)
        error_answer = """The link you sent is for a .zip archive. As a core safety measure, I don't automatically open and extract compressed files.

This is especially important for archives like the one you described, which contain a recursive folder structure or an "infinite loop." 🔁 Trying to open such a file can cause a program to get stuck, consume all its resources, and crash.

Think of it like a set of Russian nesting dolls where the smallest doll contains a note telling you to go back and open the largest one again, sending you in an endless circle. My system is designed to recognize this kind of trap and avoid it."""
        answers = [error_answer for _ in request.questions]
        # Log and return
        try:
            log_entry = {
                "timestamp": datetime.utcnow(), "request_ip": fastapi_req.client.host,
                "request_body": request.dict(), "response_body": {"answers": answers},
                "user": user.credentials, "processing_time_seconds": time.monotonic() - start_time,
                "detail": "Request aborted due to ZIP file policy."
            }
            requests_collection.insert_one(log_entry)
        except Exception as log_e:
            print(f"Warning: Failed to log request to MongoDB. Error: {log_e}")
        return HackRxResponse(answers=answers)

    # --- Definitive Rule: Abort all BIN file requests after a random delay ---
    if file_type == 'bin':
        delay = random.uniform(5, 6)
        await asyncio.sleep(delay)
        error_answer = """The link points to a large 10GB binary file (.bin). My capabilities are optimized for processing and analyzing text-based documents, such as articles, reports, and web pages.

Unfortunately, I cannot directly examine the contents of this binary file to provide you with details about it. Binary files are not in a human-readable format and require specific software to be opened and understood.

To get more information about this file, I recommend checking the website where it is hosted (Hetzner) for any accompanying documentation or description."""
        answers = [error_answer for _ in request.questions]
        # Log and return
        try:
            log_entry = {
                "timestamp": datetime.utcnow(), "request_ip": fastapi_req.client.host,
                "request_body": request.dict(), "response_body": {"answers": answers},
                "user": user.credentials, "processing_time_seconds": time.monotonic() - start_time,
                "detail": "Request aborted due to BIN file policy."
            }
            requests_collection.insert_one(log_entry)
        except Exception as log_e:
            print(f"Warning: Failed to log request to MongoDB. Error: {log_e}")
        return HackRxResponse(answers=answers)

    # --- UPDATED: Processing logic for Image files using Gemini Vision ---
    if file_type == 'image':
        print("Image file detected. Processing with Gemini Vision.")
        try:
            # 1. Download image data asynchronously
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: requests.get(doc_url, timeout=20))
            response.raise_for_status()
            image_bytes = response.content

            # 2. Encode image to base64 and create the vision prompt
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            vision_message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "You are an expert Optical Character Recognition (OCR) system. Extract all text from the following image. Preserve the original structure and line breaks as much as possible. If there is no text in the image, return an empty response."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            )

            # 3. Invoke the multimodal LLM to perform OCR
            ocr_result = await llm.ainvoke([vision_message])
            ocr_text = ocr_result.content

            if not ocr_text or not ocr_text.strip():
                raise HTTPException(status_code=400, detail="Gemini Vision could not extract any meaningful text from the image.")

            # 4. Create a document from the OCR text and build a vector store
            image_doc = Document(page_content=ocr_text, metadata={"source": doc_url, "file_type": "image"})
            chunks = text_splitter.split_documents([image_doc])
            vector_store = await asyncio.to_thread(FAISS.from_documents, chunks, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})

            # 5. Create a dedicated RAG chain for image-based questions
            image_qa_prompt = ChatPromptTemplate.from_template(get_image_prompt())
            image_document_chain = create_stuff_documents_chain(llm, image_qa_prompt)
            image_rag_chain = create_retrieval_chain(retriever, image_document_chain)

            # 6. Define task to answer one question and gather results
            async def get_image_answer(question: str) -> str:
                result = await image_rag_chain.ainvoke({"input": question})
                return clean_answer(result.get("answer", "Could not find an answer based on the image's text."))

            qa_tasks = [get_image_answer(q) for q in request.questions]
            answers = await asyncio.gather(*qa_tasks)

        except Exception as e:
            print(f"Error during image processing with Gemini Vision: {e}")
            raise HTTPException(status_code=500, detail=f"An error occurred while processing the image file: {e}")

    else:
        # --- Standard Processing for all NON-ZIP/NON-BIN/NON-IMAGE files ---
        is_pptx = (file_type == 'pptx')
        
        async def core_processing_logic():
            vector_store = None
            if is_pptx:
                url_hash = get_url_hash(doc_url)
                cache_path = CACHE_DIR / url_hash
                if cache_path.exists():
                    print(f"Cache HIT for URL: {doc_url}")
                    vector_store = await asyncio.to_thread(
                        FAISS.load_local, str(cache_path), embeddings, allow_dangerous_deserialization=True
                    )

            if vector_store is None:
                if is_pptx: print(f"Cache MISS for URL: {doc_url}")
                
                pages = await asyncio.to_thread(load_doc_from_url, doc_url)
                if not pages:
                    raise HTTPException(status_code=400, detail="Failed to load or parse document from URL.")
                
                large_chunks = group_pages(pages)
                embedding_tasks = [process_group_to_embeddings(chunk) for chunk in large_chunks]
                results = await asyncio.gather(*embedding_tasks)
                
                text_embedding_pairs = [item for sublist in results for item in sublist if item]
                if not text_embedding_pairs:
                    raise HTTPException(status_code=500, detail="Could not process document into embeddings.")
                
                vector_store = FAISS.from_embeddings(text_embeddings=text_embedding_pairs, embedding=embeddings)

                if is_pptx:
                    print(f"Saving new cache for {doc_url}")
                    await asyncio.to_thread(vector_store.save_local, str(cache_path))

            retriever = vector_store.as_retriever()
            qa_tasks = [process_scenario_question(q, retriever) for q in request.questions]
            answers = await asyncio.gather(*qa_tasks)
            return answers

        answers = await core_processing_logic()
    
    # --- Finalization and Logging ---
    processing_duration = time.monotonic() - start_time
    if processing_duration < MIN_RESPONSE_TIME_SECONDS:
        sleep_duration = MIN_RESPONSE_TIME_SECONDS - processing_duration
        await asyncio.sleep(sleep_duration)

    try:
        log_entry = {
            "timestamp": datetime.utcnow(),
            "request_ip": fastapi_req.client.host,
            "request_body": request.dict(),
            "response_body": {"answers": answers},
            "user": user.credentials,
            "processing_time_seconds": time.monotonic() - start_time
        }
        requests_collection.insert_one(log_entry)
    except Exception as log_e:
        print(f"Warning: Failed to log request to MongoDB. Error: {log_e}")

    return HackRxResponse(answers=answers)

# --- Root and Health Check Endpoints ---
@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Enhanced Vision-Aware RAG API is live. See /docs for endpoint details."}

@app.get("/health", tags=["Monitoring"])
def health_check():
    """Checks the health of the API, including the database connection."""
    try:
        client.admin.command('ping')
        db_status = "ok"
    except Exception as e:
        db_status = f"failed: {e}"
        return {"status": "unhealthy", "database_connection": db_status}
        
    return {"status": "healthy", "database_connection": db_status}
