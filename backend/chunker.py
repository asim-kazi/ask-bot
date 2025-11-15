# backend/chunker.py (improved, safe)
import json
from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    text = str(text).strip()
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def safe_join_list(x):
    if isinstance(x, list):
        return ", ".join(map(str, x))
    if isinstance(x, dict):
        # convert dict to a readable string
        return ", ".join([f"{k}: {v}" for k, v in x.items()])
    return str(x)


def load_and_chunk(file_path: str = "../data/personal.json") -> List[Dict]:
    """
    Load personal.json and return a list of documents:
    [{"text": "...", "metadata": {...}}, ...]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents: List[Dict] = []

    # --- PROFILE: store atomic fields for precise retrieval ---
    profile = data.get("profile", {})
    for key in ("name", "email", "location", "title", "bio"):
        val = profile.get(key)
        if val:
            documents.append({
                "text": str(val),
                "metadata": {"source": "personal.json", "section": key}
            })

    # also keep a friendly profile summary chunk
    links_str = safe_join_list(profile.get("links", {}))
    profile_summary = (
        f"{profile.get('name','')} - {profile.get('title','')}. "
        f"{profile.get('bio','')}. Email: {profile.get('email','')}. "
        f"Location: {profile.get('location','')}. Links: {links_str}."
    )
    for i, chunk in enumerate(chunk_text(profile_summary)):
        documents.append({
            "text": chunk,
            "metadata": {"source": "personal.json", "section": "profile", "chunk_id": i}
        })

    # --- EDUCATION ---
    for edu in data.get("education", []):
        edu_text = f"{edu.get('degree','')} at {edu.get('institute','')} ({edu.get('year','')})"
        if edu.get("cgpa"):
            edu_text += f" | CGPA: {edu.get('cgpa')}"
        if edu.get("percentage"):
            edu_text += f" | Percentage: {edu.get('percentage')}"
        for i, chunk in enumerate(chunk_text(edu_text)):
            documents.append({
                "text": chunk,
                "metadata": {"source": "personal.json", "section": "education", "chunk_id": i}
            })

    # --- SKILLS (single doc is fine) ---
    skills_text = safe_join_list(data.get("skills", []))
    if skills_text:
        documents.append({
            "text": skills_text,
            "metadata": {"source": "personal.json", "section": "skills"}
        })

    # --- PROJECTS: split into logical parts (title, desc, stack, notes, repo) ---
    for project in data.get("projects", []):
        title = project.get("title", "")
        desc = project.get("desc", "")
        stack = safe_join_list(project.get("stack", []))
        notes = project.get("notes", "")
        repo = project.get("repo")

        parts = [
            ("title", title),
            ("desc", desc),
            ("stack", f"Stack: {stack}" if stack else ""),
            ("notes", f"Notes: {notes}" if notes else "")
        ]
        if repo:
            parts.append(("repo", f"Repo: {repo}"))

        for ptype, ptext in parts:
            if ptext and str(ptext).strip():
                documents.append({
                    "text": str(ptext),
                    "metadata": {
                        "source": "personal.json",
                        "section": "project",
                        "project_id": project.get("id"),
                        "part": ptype
                    }
                })

    return documents


if __name__ == "__main__":
    docs = load_and_chunk()
    print(f"Total chunks: {len(docs)}")
    for d in docs:
        # Print a preview only to avoid huge terminal spam
        preview = d["text"][:250].replace("\n", " ")
        print({"text_preview": preview, "metadata": d["metadata"]})