# backend/retrieval_qa.py
"""
Local Retrieval + (optional) local generation pipeline.

- Uses sentence-transformers (all-MiniLM-L6-v2) for query embeddings (free).
- Uses chromadb PersistentClient to query the existing local collection 'personal_data'.
- Optionally uses a small HuggingFace text2text model (google/flan-t5-small) to synthesize answers.
- If transformers is not present / model not downloaded, falls back to returning concatenated retrieved docs.
"""

import os
import json
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer

# Try to import HF pipeline (optional). If not present, we'll fallback to simple concat answers.
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ---------------------------
# Config
# ---------------------------
DB_DIR = "./db"
COLLECTION_NAME = "personal_data"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4
# Tweak threshold after observing distances. Smaller distance = more similar.
# Observed: excellent matches ~0.16, so 0.30 is a reasonable cutoff to mark "confident".
SIMILARITY_THRESHOLD = 0.30

# Local generator model (small and fairly capable)
HF_GEN_MODEL = os.getenv("HF_GEN_MODEL", "google/flan-t5-small")
HF_MAX_LENGTH = int(os.getenv("HF_MAX_LENGTH", "256"))

# ---------------------------
# Setup embedding model (local)
# ---------------------------
print(f"Loading embedding model '{EMBED_MODEL_NAME}' (this may take a moment)...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def embed_text(text: str) -> List[float]:
    if not text:
        return []
    vec = embed_model.encode(text)
    # convert numpy -> list
    try:
        return vec.tolist()
    except Exception:
        # if already list-like
        return list(vec)

# ---------------------------
# Setup Chroma client & collection
# ---------------------------
client = chromadb.PersistentClient(path=DB_DIR)

try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception as e:
    # helpful debug message
    try:
        all_cols = client.list_collections()
    except Exception:
        all_cols = []
    raise RuntimeError(
        f"Could not open Chroma collection '{COLLECTION_NAME}'. Available collections: {all_cols}. Error: {e}"
    )

# ---------------------------
# Optional: setup HF generator (if available)
# ---------------------------
GEN_PIPE = None
if HF_AVAILABLE:
    try:
        print(f"Initializing HF generation pipeline with model '{HF_GEN_MODEL}' (will download if needed)...")
        tokenizer = AutoTokenizer.from_pretrained(HF_GEN_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(HF_GEN_MODEL)
        GEN_PIPE = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)  # CPU by default
        print("HF generation pipeline ready.")
    except Exception as e:
        GEN_PIPE = None
        print("⚠️ HF pipeline init failed (will fallback to concatenated answers). Error:", e)
else:
    print("transformers not available — generation fallback will use concatenated retrieved texts.")

# ---------------------------
# Retrieval helper
# ---------------------------
def semantic_search(query: str, k: int = TOP_K, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Perform semantic search using chroma collection.
    Returns dict with keys: documents(list), metadatas(list), distances(list)
    """
    q_emb = embed_text(query)
    if not q_emb:
        return {"documents": [], "metadatas": [], "distances": []}
    # Note: chroma returns lists inside lists (for batch queries)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return {"documents": docs, "metadatas": metas, "distances": dists}

# ---------------------------
# Answer generator
# ---------------------------
def generate_answer_with_hf(context: str, question: str) -> str:
    """
    Use HF text2text pipeline to generate an answer from context+question.
    If pipeline not available, raises.
    """
    if GEN_PIPE is None:
        raise RuntimeError("HF generation pipeline not available")
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer concisely based only on the context."
    out = GEN_PIPE(prompt, max_length=HF_MAX_LENGTH, truncation=True)
    text = out[0].get("generated_text") or out[0].get("text") or str(out[0])
    return text.strip()

def simple_concat_answer(top_texts: List[str], question: str) -> str:
    """
    A safe, free fallback that synthesizes an answer by concatenating top texts and
    returning a short summary-like response (no heavy generation).
    """
    joined = "\n\n".join(top_texts)
    return f"I found these relevant details (based on my profile):\n\n{joined}"

# ---------------------------
# High-level API: answer_query
# ---------------------------
def answer_query(query: str, k: int = TOP_K, similarity_threshold: float = SIMILARITY_THRESHOLD, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns:
      {
        "answer": str,
        "source_documents": [ {"text":..., "metadata":... , "distance":...}, ... ],
        "used_retrieval": bool,
        "top_score": float or None
      }
    """
    search = semantic_search(query, k=k, where=where)
    docs = search["documents"]
    metas = search["metadatas"]
    dists = search["distances"]

    if len(docs) == 0:
        return {
            "answer": "I couldn't find relevant info in my profile. Ask something else about my projects/skills.",
            "source_documents": [],
            "used_retrieval": False,
            "top_score": None
        }

    top_score = dists[0] if len(dists) > 0 else None

    # Prepare structured source docs
    source_docs = []
    for t, m, s in zip(docs, metas, dists):
        source_docs.append({"text": t, "metadata": m, "distance": s})

    # Confident match if distance <= threshold (smaller means more similar)
    if top_score is not None and top_score <= similarity_threshold:
        top_texts = [t for t in docs[:k] if t and t.strip()]
        context = "\n\n".join(top_texts)
        # Try HF generation if available
        if GEN_PIPE:
            try:
                ans = generate_answer_with_hf(context, query)
                return {"answer": ans, "source_documents": source_docs, "used_retrieval": True, "top_score": top_score}
            except Exception as e:
                # fallback to safe concat
                ans = simple_concat_answer(top_texts, query)
                return {"answer": ans + f"\n\n(Generation failed: {e})", "source_documents": source_docs, "used_retrieval": True, "top_score": top_score}
        else:
            ans = simple_concat_answer(top_texts, query)
            return {"answer": ans, "source_documents": source_docs, "used_retrieval": True, "top_score": top_score}
    else:
        # Not confident -> return concatenated docs and say uncertain
        top_texts = [t for t in docs[:k] if t and t.strip()]
        ans = "I am not confident this is from my profile. Here are possibly related details:\n\n" + "\n\n".join(top_texts)
        return {"answer": ans, "source_documents": source_docs, "used_retrieval": False, "top_score": top_score}

# ---------------------------
# CLI quick test block
# ---------------------------
if __name__ == "__main__":
    tests = [
        "Tell me about the Flash Card Generator",
        "Where are you located?",
        "What skills do you have?",
        "Explain what a movie recommendation system is"
    ]
    for q in tests:
        print(f"\n=== QUESTION: {q}")
        out = answer_query(q)
        print("TOP SCORE:", out.get("top_score"))
        print("USED RETRIEVAL:", out.get("used_retrieval"))
        print("ANSWER:\n", out.get("answer")[:2000])
        print("SOURCES:")
        for s in out.get("source_documents", [])[:4]:
            print(" -", s.get("metadata"), "|", s.get("text")[:300], "| DIST:", s.get("distance"))
