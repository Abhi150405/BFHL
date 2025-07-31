import os
import asyncio
from typing import List

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# This assumes your helper and prompt files are in a 'src' directory
# and you have an empty 'src/__init__.py' file.
from src.helper import load_doc_from_url, text_split, download_hugging_face_embeddings
from src.prompt import system_prompt

# --- Configuration and Initialization ---
load_dotenv()

app = FastAPI(
    title="Document Q&A API with Single Groq Model",
    description="An API that processes questions using a single Groq model.",
    version="2.4.0"
)

# --- Authentication ---
API_KEY = os.getenv("BEARER_TOKEN")
auth_scheme = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    """Dependency to validate the Bearer token."""
    if not API_KEY or credentials.scheme != "Bearer" or credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the policy document.")
    questions: List[str] = Field(..., description="A list of questions about the document.")

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Initialize RAG Components ---
embeddings = download_hugging_face_embeddings()
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Initialize single LLM client
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY must be set in the .env file.")

# Create single LLM client
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(
    request: HackRxRequest,
    user: str = Depends(get_current_user)
):
    print(f"Processing authenticated request with single Groq model")
    try:
        # Document loading, chunking, and FAISS creation
        docs = load_doc_from_url(request.documents)
        text_chunks = text_split(docs)
        chunk_texts = [chunk.page_content for chunk in text_chunks]
        chunk_embeddings = await embeddings.aembed_documents(chunk_texts)
        text_embedding_pairs = list(zip(chunk_texts, chunk_embeddings))
        vector_store = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # Helper function to process a single question
        async def get_answer(question: str):
            document_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, document_chain)
            result = await rag_chain.ainvoke({"input": question})
            return result.get("answer", "Could not find an answer.").strip()

        # Create tasks for all questions using the single LLM
        tasks = [get_answer(question) for question in request.questions]

        # Run all tasks concurrently. asyncio.gather preserves the order of the
        # tasks, so the 'answers' list will match the original question order.
        answers = await asyncio.gather(*tasks)

        return HackRxResponse(answers=answers)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- Root and Health Check Endpoints ---
@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "API is live. See /docs for endpoint details."}

@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "healthy"}
