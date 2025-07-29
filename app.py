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
# NOTE: The content for 'system_prompt' is provided below for context.
# from src.helper import load_doc_from_url, text_split, download_hugging_face_embeddings
# from src.prompt import system_prompt

# --- Configuration and Initialization ---
load_dotenv()

app = FastAPI(
    title="Parallel Document Q&A API",
    description="An API that uses FAISS and parallel embedding to answer questions about a document.",
    version="2.1.0"
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
# This is the system prompt content you provided
system_prompt = (
"""You are a precise text analysis expert. Your task is to answer questions accurately and concisely, using ONLY the provided text context.

Follow these rules without exception:
1.  **Strict Adherence to Context:** Your entire answer must be based on the provided text. Do not use any external knowledge.
2.  **Answer with Brief Reasoning:** Provide a direct answer to the question, supported by brief reasoning or context from the document. Keep the entire response concise, with a maximum of 30 words.
3.  **Handling Unknowns:** If the text does not contain the answer, you must reply with the exact phrase: "This information is not available in the provided document."
4.  **Direct and Factual:** Provide a direct, factual answer. Avoid conversational fillers, opinions, or introductory phrases.

Context from the document:
{context}"""
)

# NOTE: The 'download_hugging_face_embeddings' function is assumed to be in your 'src/helper.py' file.
# embeddings = download_hugging_face_embeddings()
# This is a placeholder for your actual function call
embeddings = None # Replace with your actual embeddings function call

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
    Processes a document from a URL using a temporary FAISS index
    with embeddings created in parallel.
    """
    # Placeholder check for embeddings, as they are not loaded in this snippet
    if not embeddings:
         raise HTTPException(status_code=500, detail="Embeddings not loaded. Replace placeholder in the code.")

    print(f"Processing authenticated request with parallel embedding")
    try:
        # NOTE: The helper functions below are assumed to be in your 'src/helper.py' file.
        # 1. Load document from URL
        # docs = load_doc_from_url(request.documents)
        # if not docs:
        #     raise HTTPException(status_code=400, detail="Could not load or process document from URL.")

        # 2. Split document into chunks
        # text_chunks = text_split(docs)
        # if not text_chunks:
        #     raise HTTPException(status_code=500, detail="Failed to split the document.")

        # The lines above are commented out because the helper functions are not available.
        # You should uncomment them and ensure 'src/helper.py' is present.
        # For demonstration, we'll use a placeholder for text_chunks.
        text_chunks = ["This is a placeholder chunk."] # Replace with your actual text_split call

        # 3. Create Embeddings in PARALLEL
        print(f"Starting parallel embedding for {len(text_chunks)} chunks...")
        chunk_texts = [chunk.page_content if hasattr(chunk, 'page_content') else chunk for chunk in text_chunks]
        chunk_embeddings = await embeddings.aembed_documents(chunk_texts)
        print("...Parallel embedding complete.")

        # Combine the texts and their corresponding embeddings
        text_embedding_pairs = list(zip(chunk_texts, chunk_embeddings))

        # 4. Create FAISS vector store from the pre-computed embeddings
        vector_store = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=embeddings
        )

        # 5. Create retriever and RAG chain
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

        # 6. Generate answers for all questions CONCURRENTLY
        async def get_answer(question: str):
            """Helper function to invoke the RAG chain asynchronously."""
            result = await rag_chain.ainvoke({"input": question})
            answer = result.get("answer", "Could not find an answer.")
            # MODIFIED LINE: Clean the answer string before returning
            return answer.strip()

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

# NOTE: The code for running with uvicorn is typically in a separate run.py file
# or executed from the command line, e.g., `uvicorn main:app --reload`
# If you need to run this file directly, you could add:
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
