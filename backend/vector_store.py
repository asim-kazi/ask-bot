import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

# ---------------------------
# Config
# ---------------------------

DATA_FILE = "../data/personal.json"
DB_DIR = "./db"
COLLECTION_NAME = "personal_data"

# ---------------------------
# Load personal chunks
# ---------------------------
from chunker import load_and_chunk
docs = load_and_chunk(DATA_FILE)

# ---------------------------
# Setup Chroma client
# ---------------------------
client = chromadb.PersistentClient(path=DB_DIR)

# Delete existing collection if needed
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

# Create collection
collection = client.create_collection(
    name=COLLECTION_NAME
)

# ---------------------------
# Choose Embedding Option
# ---------------------------

# Option A: Local sentence-transformers (FREE, no API cost)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_local(text: str):
    return embed_model.encode(text).tolist()

# Option B: OpenAI embeddings (if API key available)
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")
# def embed_openai(text: str):
#     emb = openai.Embedding.create(
#         model="text-embedding-3-small",
#         input=text
#     )["data"][0]["embedding"]
#     return emb

# ---------------------------
# Insert docs into Chroma
# ---------------------------
for idx, d in enumerate(docs):
    emb = embed_local(d["text"])   # 👉 switch to embed_openai if needed
    collection.add(
        ids=[f"doc_{idx}"],
        documents=[d["text"]],
        metadatas=[d["metadata"]],
        embeddings=[emb]
    )

print(f"✅ Inserted {len(docs)} chunks into Chroma collection '{COLLECTION_NAME}'")


# ---------------------------
# Search function
# ---------------------------
def get_top_k(query: str, k: int = 3):
    q_emb = embed_local(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )
    return results


if __name__ == "__main__":
    q = "Tell me about the Flash Card Generator"
    res = get_top_k(q, k=3)
    print("🔍 Query:", q)
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        print("->", doc[:100], "...", "| META:", meta)
