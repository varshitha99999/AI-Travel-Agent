"""
RAG Chain
---------
Wraps TravelDocumentStore with a simple Q&A chain.
Retrieves relevant chunks via TF-IDF and passes them to the Groq LLM.
"""

from __future__ import annotations

import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.document_store import TravelDocumentStore


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are TripWeaver, an AI Travel Concierge.
The user has uploaded travel documents (guides, itineraries, visa info, etc.).
Use the retrieved context below to answer their question accurately.
If the context doesn't contain enough information, say so honestly and offer general travel advice.

Retrieved context:
{context}
""",
    ),
    ("human", "{question}"),
])


def answer_from_docs(
    question: str,
    store: TravelDocumentStore,
    groq_api_key: str | None = None,
) -> str:
    """
    Retrieve relevant chunks from the document store and generate an answer.
    Returns an empty string if no documents are loaded or nothing relevant found.
    """
    if not store.has_documents():
        return ""

    context = store.query(question, k=4)
    if not context:
        return ""

    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1024,
        timeout=30,
    )

    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
