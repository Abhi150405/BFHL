# src/advanced_prompt.py
from typing import List, Tuple

def get_advanced_scenario_prompt():
    """Advanced prompt specifically designed for complex insurance scenario questions."""
    return """You are an expert insurance policy analyst with deep knowledge of claims processing, dependent management, and policy procedures. Your task is to provide comprehensive, actionable answers to complex insurance scenarios using the provided document context.

ANALYSIS APPROACH:
1. **Scenario Understanding**: Carefully analyze multi-part questions that involve multiple insurance processes, dependent situations, or combined procedures.

2. **Comprehensive Coverage**: Address ALL components of complex questions systematically. Do not leave any part unanswered.

3. **Structured Response**: For multi-part questions, organize your response clearly:
   - Use numbered sections for different aspects
   - Provide specific procedures and requirements
   - Include all relevant contact information
   - Mention timeframes and deadlines

4. **Specific Details**: Always provide:
   - Exact document requirements (with specific names)
   - Step-by-step procedures
   - Age limits, eligibility criteria, and conditions
   - Contact emails, phone numbers, department names
   - Deadlines and time limits
   - Special circumstances or exceptions

5. **Context Integration**: When multiple topics are mentioned (e.g., claims + name changes + grievances), integrate information from different sections of the document to provide complete guidance.

6. **Practical Guidance**: Provide actionable steps that the person can follow immediately.

RESPONSE GUIDELINES:
- Be comprehensive but concise
- Use specific terminology from the document
- Provide exact contact information when available
- Mention any prerequisites or conditions
- Include relevant form numbers or reference codes
- Address edge cases or special situations mentioned

For missing information, state: "This specific detail is not provided in the document" while still giving all available related information.

Context from document:
{context}

Question: {input}

Provide a detailed, actionable response addressing all aspects of this scenario:"""

def get_fallback_prompt():
    """Fallback prompt for when the main prompt doesn't work well."""
    return """You are analyzing an insurance document to answer a detailed question. The question may have multiple parts that need to be addressed comprehensively.

Your task:
1. Read through the entire context carefully
2. Identify ALL parts of the question
3. For each part, provide specific information from the document
4. Include procedures, requirements, contact details, and deadlines
5. If information is scattered across different sections, combine it coherently

Format your response to address each aspect clearly. Use the exact information from the document without adding external knowledge.

Context: {context}

Question: {input}

Comprehensive Answer:"""

def get_contact_extraction_prompt():
    """Specialized prompt for extracting contact information."""
    return """Extract all contact information from the provided context that could be relevant to insurance queries, grievances, or customer service.

Look for:
- Email addresses (especially grievance, customer care, claims)
- Phone numbers
- Department names
- Office addresses
- Website URLs
- Specific process contact points

Format as a clear list with the purpose/department for each contact.

Context: {context}

Contact Information:"""

def get_process_extraction_prompt():
    """Specialized prompt for extracting step-by-step processes."""
    return """Extract detailed procedures and processes from the context related to the specific query.

Focus on:
- Step-by-step procedures
- Required documents (with exact names)
- Timeframes and deadlines
- Eligibility criteria
- Special conditions or exceptions
- Prerequisites

Present as clear, actionable steps.

Context: {context}
Query Focus: {query_focus}

Process Details:"""

def get_eligibility_prompt():
    """Specialized prompt for dependent and eligibility questions."""
    return """Extract all eligibility criteria, age limits, and dependent-related rules from the context.

Look for:
- Age limits for different types of dependents
- Financial dependency requirements
- Marriage/status change procedures
- Continuation criteria
- Special circumstances (job loss, education, etc.)
- Required documentation for status changes

Context: {context}

Eligibility Rules:"""

# Prompt selection based on question type
def select_optimal_prompt(question: str) -> str:
    """Select the best prompt based on question characteristics."""
    question_lower = question.lower()
    
    # Check for contact-related queries
    if any(term in question_lower for term in ['email', 'contact', 'grievance', 'customer care', 'phone']):
        if len([term for term in ['email', 'contact', 'grievance'] if term in question_lower]) >= 2:
            return get_contact_extraction_prompt()
    
    # Check for process-heavy queries
    if any(term in question_lower for term in ['process', 'procedure', 'steps', 'how to', 'submit']):
        if any(term in question_lower for term in ['claim', 'submission', 'documents']):
            return get_process_extraction_prompt()
    
    # Check for eligibility queries
    if any(term in question_lower for term in ['dependent', 'eligibility', 'age', 'continue', 'sibling']):
        return get_eligibility_prompt()
    
    # Default to advanced scenario prompt for complex questions
    return get_advanced_scenario_prompt()

# Multi-prompt approach for comprehensive answers
def get_multi_prompt_strategy(question: str) -> List[Tuple[str, str]]:
    """Get multiple prompts to cover different aspects of complex questions."""
    strategies = []
    question_lower = question.lower()
    
    # Always start with the main scenario prompt
    strategies.append(("main", get_advanced_scenario_prompt()))
    
    # Add specialized prompts based on question content
    if any(term in question_lower for term in ['email', 'contact', 'grievance']):
        strategies.append(("contacts", get_contact_extraction_prompt()))
    
    if any(term in question_lower for term in ['process', 'procedure', 'documents', 'submit']):
        strategies.append(("process", get_process_extraction_prompt()))
    
    if any(term in question_lower for term in ['dependent', 'eligibility', 'age', 'continue']):
        strategies.append(("eligibility", get_eligibility_prompt()))
    
    return strategies

# Enhanced prompt for specific insurance scenarios
def get_insurance_scenario_prompts():
    """Collection of insurance-specific scenario prompts."""
    return {
        "claim_process": """
        Analyze this insurance claim scenario and provide complete guidance on:
        
        1. CLAIM SUBMISSION PROCESS:
           - Required documents (list each specifically)
           - Submission methods and deadlines
           - Processing timeframes
        
        2. DEPENDENT-RELATED PROCEDURES:
           - Eligibility verification steps
           - Required documentation for dependents
           - Age limits and continuation criteria
        
        3. ADDITIONAL REQUIREMENTS:
           - Name change procedures
           - Contact information for queries
           - Grievance process if applicable
        
        Use only information from the provided context.
        
        Context: {context}
        Scenario: {input}
        """,
        
        "dependent_management": """
        Address this dependent management scenario covering:
        
        1. ELIGIBILITY CRITERIA:
           - Age limits for different dependent types
           - Financial dependency requirements
           - Special circumstances (marriage, job loss, education)
        
        2. DOCUMENTATION PROCESS:
           - Required documents for status changes
           - Update procedures and timelines
           - Verification requirements
        
        3. POLICY UPDATES:
           - Name change procedures
           - Contact information for updates
           - Processing timelines
        
        Context: {context}
        Question: {input}
        """,
        
        "network_verification": """
        Provide comprehensive guidance on network and provider verification:
        
        1. NETWORK VERIFICATION:
           - How to check if a provider is in-network
           - Pre-authorization requirements
           - Specific steps for verification
        
        2. CLAIM PROCEDURES:
           - Different processes for network vs non-network
           - Required documentation
           - Approval processes
        
        3. CONTACT INFORMATION:
           - Relevant departments for verification
           - Customer service contacts
           - Grievance contacts if needed
        
        Context: {context}
        Query: {input}
        """
    }

def determine_scenario_type(question: str) -> str:
    """Determine the type of insurance scenario to select appropriate prompt."""
    question_lower = question.lower()
    
    if any(term in question_lower for term in ['claim', 'submission', 'documents', 'bills']):
        if any(term in question_lower for term in ['dependent', 'daughter', 'spouse', 'child']):
            return "claim_process"
    
    if any(term in question_lower for term in ['dependent', 'eligibility', 'continue', 'age']):
        return "dependent_management"
    
    if any(term in question_lower for term in ['network', 'provider', 'hospital', 'apollo']):
        return "network_verification"
    
    return "claim_process"  # Default to claim process for complex scenarios
