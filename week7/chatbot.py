"""
chatbot.py
Combines retrieval (vectorstore) with generation (backend) to answer
user questions grounded in the uploaded document.
"""

from vectorstore import query_chunks
from backend import get_llm_response

RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based only on the provided document context.
If the answer is not contained in the context, say you don't have enough information — do not make up an answer.

Context:
{context}

Question: {question}

Answer:"""


def answer_question(question: str, top_k: int = 4, provider: str = None) -> dict:
    """
    Retrieve relevant chunks for the question and generate a grounded answer.
    Returns the answer plus the source chunks used, for citation display.
    """
    retrieved = query_chunks(question, top_k=top_k)

    if not retrieved:
        return {
            "answer": "No documents have been indexed yet — please upload a PDF first.",
            "sources": [],
        }

    context = "\n\n".join(f"[Chunk {r['chunk_index']}] {r['text']}" for r in retrieved)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    answer = get_llm_response(prompt, provider=provider)

    return {
        "answer": answer,
        "sources": retrieved,
    }
