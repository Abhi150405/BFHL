# BFHL

This project is a document-based Q&A chatbot API that supports PDF, Word (.docx), and Email (.eml/.msg) files. It uses FastAPI, LangChain, and FAISS for in-memory vector search.

## Features
- **Multi-format support:** PDF, Word, and Email files via URL
- **In-memory vector search:** Uses FAISS (no external vector DB required)
- **Modern API:** Built with FastAPI
- **Embeddings:** Uses sentence-transformers/all-MiniLM-L6-v2

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/AbhishekC1005/BFHL.git
   cd Medical-Chatbot
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Running the API

```sh
python run.py
```

The API will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## API Usage

### `/hackrx/run` (POST)
- **Description:** Answer questions about a document from a URL (PDF, DOCX, EML, MSG)
- **Request Body:**
  ```json
  {
    "documents": "https://example.com/your.pdf",
    "questions": ["What is this document about?"]
  }
  ```
- **Response:**
  ```json
  {
    "answers": ["...answer..."]
  }
  ```

## Notes
- Make sure the document URL is publicly accessible.
- FAISS is in-memory; each request creates a temporary index.
- For production, consider persistent storage or a scalable vector DB.

## Troubleshooting
- If you see errors about missing packages, run:
  ```sh
  pip install -r requirements.txt
  ```
- If you see errors about document loading, check the file type and URL.

## License
See LICENSE file.
