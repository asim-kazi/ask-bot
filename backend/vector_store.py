# backend/vector_store.py

import os
import json
from typing import Optional, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------
# Config
# ---------------------------
DATA_FILE = "../data/personal.json"
DB_DIR = "./db"
COLLECTION_NAME = "personal_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small & fast

# ---------------------------
# Load personal chunks (from chunker.py)
# ---------------------------
from chunker import load_and_chunk

docs = load_and_chunk(DATA_FILE)

# ---------------------------
# Setup Chroma client (persistent)
# ---------------------------
# Note: PersistentClient stores DB files in DB_DIR
client = chromadb.PersistentClient(path=DB_DIR)

# If you want to start fresh uncomment the delete line below.
# Be careful — it removes existing collection data.
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    # ignore if collection doesn't exist
    pass

collection = client.create_collection(name=COLLECTION_NAME)

# ---------------------------
# Embedding function (local)
# ---------------------------
embed_model = SentenceTransformer(EMBEDDING_MODEL)


def embed_local(text: str):
    """Return embedding vector (list[float]) for text using sentence-transformers."""
    if not text:
        return []
    return embed_model.encode(text).tolist()


# If you prefer OpenAI embeddings, implement embed_openai() and swap usage below.
# (Do NOT put keys in repo; use .env and python-dotenv in your FastAPI server.)
#
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")
# def embed_openai(text: str):
#     emb = openai.Embedding.create(model="text-embedding-3-small", input=text)["data"][0]["embedding"]
#     return emb

# ---------------------------
# Insert docs into Chroma
# ---------------------------
# Add documents one-by-one (with ids, texts, metadata and embeddings)
for idx, d in enumerate(docs):
    text = d.get("text", "")
    meta = d.get("metadata", {})
    emb = embed_local(text)
    # Skip empty embeddings/text (defensive)
    if not text or not emb:
        continue
    collection.add(
        ids=[f"doc_{idx}"],
        documents=[text],
        metadatas=[meta],
        embeddings=[emb]
    )

print(f"✅ Inserted {len(docs)} chunks into Chroma collection '{COLLECTION_NAME}'")

# ---------------------------
# Search functions
# ---------------------------
def get_top_k(query: str, k: int = 3, where: Optional[Dict[str, Any]] = None, include_scores: bool = False):
    """
    Query the Chroma collection semantically.
    :param query: user query string
    :param k: number of results
    :param where: optional metadata filter, e.g. {"section":"location"}
    :param include_scores: if True, include 'distances' in the returned result
    :return: dict with keys 'documents', 'metadatas', and optionally 'distances'
    """
    q_emb = embed_local(query)
    include = ["documents", "metadatas"]
    if include_scores:
        include.append("distances")

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        where=where,
        include=include
    )

    return res


def get_top_k_formatted(query: str, k: int = 3, where: Optional[Dict[str, Any]] = None):
    """
    Convenience wrapper returning a list of tuples: (text, metadata, distance (optional))
    """
    res = get_top_k(query, k=k, where=where, include_scores=True)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs)
    result_list = []
    for text, meta, dist in zip(docs, metas, dists):
        result_list.append((text, meta, dist))
    return result_list


# ---------------------------
# Quick test if run as script
# ---------------------------
if __name__ == "__main__":
    q = "Tell me about the Flash Card Generator"
    results = get_top_k_formatted(q, k=3, where={"section": "project"})
    print("🔍 Query:", q)
    for text, meta, dist in results:
        print("->", text[:200], "...", "| META:", meta, "| DIST:", dist)