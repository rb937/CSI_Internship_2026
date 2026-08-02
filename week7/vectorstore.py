"""
vectorstore.py
Handles storing document chunk embeddings in ChromaDB and retrieving
the most relevant chunks for a given query.
"""

import chromadb
from backend import get_embedding

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "documents"


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def add_chunks(chunks: list, source_name: str = "document"):
    """Embed and store a list of text chunks in ChromaDB."""
    collection = get_collection()

    embeddings = [get_embedding(chunk) for chunk in chunks]
    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


def query_chunks(query: str, top_k: int = 4) -> list:
    """Return the top_k most relevant chunks for a query, with their sources."""
    collection = get_collection()
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    return [
        {"text": doc, "source": meta.get("source", "unknown"), "chunk_index": meta.get("chunk_index")}
        for doc, meta in zip(documents, metadatas)
    ]


def clear_collection():
    """Delete all stored chunks (useful when starting fresh with a new document)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass 
