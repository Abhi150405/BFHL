# test_scenario_questions.py

import asyncio
import os
from dotenv import load_dotenv
from src.helper import load_doc_from_url, download_hugging_face_embeddings
from src.prompt import get_scenario_prompt, get_simple_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

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

def extract_sub_questions(question: str) -> list:
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

async def test_question_processing():
    """Test the enhanced question processing system."""
    
    # Initialize components
    embeddings = download_hugging_face_embeddings()
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"), # Corrected to use a standard env variable name
        temperature=0.1
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Test questions
    test_questions = [
        "While checking the process for submitting a dental claim for a 23-year-old financially dependent daughter (who recently married and changed her surname), also confirm the process for updating her last name in the policy records and provide the company's grievance redressal email.",
        "For a claim submission involving robotic surgery for a spouse at \"Apollo Care Hospital\" (city not specified), what supporting documents are needed, how to confirm if the hospital is a network provider, and can a sibling above 26 continue as a dependent if financially dependent after job loss?"
    ]
    
    print("=== QUESTION ANALYSIS ===")
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")
        print(f"Is Scenario Question: {is_scenario_question(question)}")
        sub_questions = extract_sub_questions(question)
        print(f"Sub-questions ({len(sub_questions)}):")
        for j, sub_q in enumerate(sub_questions, 1):
            print(f"  {j}. {sub_q}")
    
    # Test with a sample document
    print("\n=== TESTING WITH DOCUMENT ===")
    document_url = input("Enter document URL for testing (or press Enter to skip): ").strip()
    
    if document_url:
        try:
            # Load document
            print("Loading document...")
            pages = load_doc_from_url(document_url)
            print(f"Loaded {len(pages)} pages")
            
            # Create embeddings
            print("Creating embeddings...")
            chunks = text_splitter.split_documents(pages)
            print(f"Created {len(chunks)} chunks")
            
            # Create vector store
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 10})
            
            # Test each question
            for i, question in enumerate(test_questions, 1):
                print(f"\n--- Testing Question {i} ---")
                print(f"Question: {question}")
                
                is_scenario = is_scenario_question(question)
                
                if is_scenario:
                    qa_prompt = ChatPromptTemplate.from_template(get_scenario_prompt())
                else:
                    qa_prompt = ChatPromptTemplate.from_template(get_simple_prompt())
                
                # Complete the chain creation and invocation
                document_chain = create_stuff_documents_chain(llm, qa_prompt)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                print("Invoking chain...")
                response = await retrieval_chain.ainvoke({"input": question})
                
                print("\n--- Answer ---")
                print(response.get("answer", "No answer found."))
                print("--------------")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please ensure the URL is correct and points to a valid PDF document.")

if __name__ == "__main__":
    asyncio.run(test_question_processing())
