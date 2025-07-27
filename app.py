import os
import asyncio
from typing import List

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn

from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# Local helper and prompt imports
from src.helper import load_doc_from_url, text_split, download_hugging_face_embeddings
from src.prompt import system_prompt

# --- Configuration and Initialization ---
load_dotenv()

app = FastAPI(
    title="Document Q&A API",
    description="An API that uses FAISS to answer questions about a document.",
    version="2.0.0"
)

# --- Authentication ---
API_KEY = os.getenv("BEARER_TOKEN")
auth_scheme = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    """Dependency to validate the Bearer token."""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="BEARER_TOKEN not configured on the server.")
    if credentials.scheme != "Bearer" or credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

# --- Pydantic Models for API Schema ---
class HackRxRequest(BaseModel):
    documents: str = Field(..., description="URL to the policy document.")
    questions: List[str] = Field(..., description="A list of questions about the document.")

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Initialize RAG Components ---
embeddings = download_hugging_face_embeddings()
llm = ChatGroq(model="gemma2-9b-it", temperature=0.2)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackRxResponse)
async def run_hackrx(
    request: HackRxRequest,
    user: str = Depends(get_current_user)  # Enforces authentication
):
    """
    Processes a document from a URL using a temporary FAISS index.
    Requires Bearer Token authentication.
    """
    print(f"Processing authenticated request with FAISS index")
    try:
        # 1. Load document from URL
        docs = load_doc_from_url(request.documents)
        if not docs:
            raise HTTPException(status_code=400, detail="Could not load or process document from URL.")

        # 2. Split document into chunks
        text_chunks = text_split(docs)
        if not text_chunks:
            raise HTTPException(status_code=500, detail="Failed to split the document.")

        # 3. Create FAISS vector store in memory for every request
        vector_store = FAISS.from_documents(
            documents=text_chunks,
            embedding=embeddings
        )

        # 4. Create retriever and RAG chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

        # ✅ 5. Generate answers for all questions CONCURRENTLY
        async def get_answer(question: str):
            """Helper function to invoke the RAG chain asynchronously."""
            result = await rag_chain.ainvoke({"input": question})
            return result.get("answer", "Could not find an answer.")

        # Create a list of tasks and run them all in parallel
        tasks = [get_answer(q) for q in request.questions]
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
