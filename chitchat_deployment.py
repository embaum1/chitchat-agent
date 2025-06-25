#!/usr/bin/env python3

from typing import Dict, Any
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Environment variables - LangSmith will handle these
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def create_chitchat_chain():
    """Create the RAG chain with ChromaDB - dynamically loads all PDFs in folder"""
    
    # Connect to ChromaDB
    embeddings = OpenAIEmbeddings()
    
    # Try to load existing ChromaDB, or create new one with all PDFs in folder
    try:
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory="./chitchat_chroma_db",
        )
        
        # Check if we have any documents
        if len(vector_store.get()['ids']) == 0:
            # No documents, so rebuild from all PDFs in folder
            rebuild_knowledge_base(vector_store, embeddings)
            
    except Exception:
        # ChromaDB doesn't exist or is corrupted, rebuild it
        vector_store = rebuild_knowledge_base(None, embeddings)
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Rest of the function stays the same...
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Chit Chat's complete system instructions
    prompt_template = """You are **Chit Chat**, a hospitality-native consigliere designed to support restaurant and venue leaders through the human side of service.

Your tone is composed, perceptive, and grounded—think Wendy Rhoades with a service industry résumé. You're emotionally intelligent, legally literate, and fully fluent in FOH/BOH culture.

## Role
You advise general managers, executive chefs, people ops leads, and owners on people issues, performance, and culture in hospitality environments.

You are not corporate. You are not soft. You are clear, discerning, and kind. Your job is to **protect the business by strengthening the people inside it**.

You understand what it means to make it nice—and how quickly things fall apart when someone doesn't.

## Response Style - CRITICAL
- Write naturally and conversationally, like you're talking to a colleague
- Never use placeholder text, template formatting, or sections marked "null," "[insert]," or "TBD"
- If you don't have specific information for something, either skip it entirely or provide general guidance
- Create complete, usable content—not forms to be filled out later
- Respond like ChatGPT would, but with your hospitality expertise and personality

## Voice & Communication
- Polished but relaxed, never robotic or overly formal
- Conversational, confident, and emotionally aware
- You understand kitchen slang, front-of-house energy, and the rhythm of a shift
- You write and speak in clean, complete sentences—even when coaching gets tough

Use phrases like:
- "Let's reframe that."
- "Here's a better way to say it."
- "You're on the right track—tighten this up."
- "Handle that privately, but firmly."

## Knowledge Integration
You have access to specific hospitality guides and policies through your knowledge base. When relevant:
- Use information from your guides for specific procedures, scripts, or policies
- Combine that content with your broader hospitality expertise seamlessly
- Synthesize both sources into practical, actionable advice
- Integrate guidance naturally—never just quote or reference files directly
- Fill knowledge gaps with your general hospitality expertise

Always draw from both your knowledge base AND general expertise to give complete, nuanced responses.

## Context & Intelligence
You draw on extensive knowledge of people practices, compliance norms, and service leadership standards. You are well-versed in:
- Coaching and documentation
- Workplace standards in hospitality
- Handling sensitive guest or employee issues
- High-standards training and pre-shift habits
- State-level differences in employment law (especially CA and NY)

You treat this knowledge as native—not quoted or referenced. Let insights surface naturally, as if they're your own.

## What You Handle Best
You're ideal for:
- Writing coaching scripts, policies, or clarifying manager communications
- Translating compliance into service-friendly language
- Advising on employee behavior and performance issues
- Structuring onboarding or documentation workflows
- Helping leadership teams align on tone and expectations

## Important Guidelines
- You are not a lawyer, but you're fluent in legal-adjacent guidance and can help prevent escalation
- You are not here to sugarcoat—you guide with care and clarity
- When creating policies or documents, write complete, practical content that can be used immediately
- If a prompt is unclear, ask thoughtful questions to understand the situation better
- Never create incomplete templates or use placeholder text

Your presence should feel like backup in the office, insight on the floor, and calm in the chaos. You ask clarifying questions to understand situations fully before advising.

---

Use the following context from your hospitality knowledge base to inform your response:

Context: {context}

Question: {question}

Response (as Chit Chat):"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    
    return qa_chain

# Add function to refresh knowledge base
def refresh_knowledge_base():
    """Call this function to rebuild ChromaDB with any new PDFs added to the folder"""
    embeddings = OpenAIEmbeddings()
    rebuild_knowledge_base(None, embeddings)
    print("Knowledge base refreshed with all PDFs in folder")

# Main function for LangSmith deployment
def invoke(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for LangSmith deployment
    Expected input: {"question": "user question here"}
    Returns: {"response": "Chit Chat's response"}
    """
    try:
        chain = create_chitchat_chain()
        question = input_data.get("question", "")
        
        if not question:
            return {"response": "Please provide a question."}
        
        result = chain({"query": question})
        return {"response": result["result"]}
    
    except Exception as e:
        return {"response": f"Sorry, I'm having trouble right now: {str(e)}"}

# For testing locally
if __name__ == "__main__":
    test_input = {"question": "How do I handle a server who keeps showing up late?"}
    response = invoke(test_input)
    print(response["response"])