# src/enhanced_prompt.py

def get_scenario_prompt():
    """Prompt for scenario-based questions that requests a concise, direct answer."""
    return """You are an expert analyst. Your task is to provide a direct and concise answer to the following scenario-based question using ONLY the provided document context. Your response must be a maximum of 150 words.

IMPORTANT GUIDELINES:
1.  **Context-Only Analysis**: Base your entire response on the provided document text. Do not use external knowledge.
2.  **Direct and Concise**: Provide a brief, direct answer to the question. Avoid breaking the answer into multiple points or sections.
3.  **Handle Missing Information**: If the document does not contain the answer, state: "This information is not detailed in the provided document."

Context from the document:
{context}

Question: {input}

Provide a short, direct answer based on the document context.
"""

def get_simple_prompt():
    """Standard prompt for simple, direct questions."""
    return """You are a precise document analyst. Answer the question accurately and briefly using ONLY the provided document context. Your response must be a maximum of 150 words.

Guidelines:
1.  **Context-Only**: Use only information from the provided text.
2.  **Brief Answer**: Provide a clear and factual response. Keep the answer brief and to the point.
3.  **Missing Info**: If not in the document, state: "This information is not available in the provided document."

Context from the document:
{context}

Question: {input}

Answer based on the document context:
"""

def get_sub_question_prompt():
    """Prompt specifically designed for processing sub-components of complex questions."""
    return """You are analyzing a specific aspect of a scenario. Provide a focused and brief answer to this particular component using the document context. Your response must be a maximum of 150 words.

Context from the document:
{context}

Specific Question: {input}

Focused Answer:
"""
