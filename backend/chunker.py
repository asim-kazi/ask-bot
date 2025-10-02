import json
from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_and_chunk(file_path: str = "../data/personal.json") -> List[Dict]:
    """
    Load resume data from JSON and chunk into documents for embedding.
    Returns a list of {text, metadata}.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    # Profile
    profile = data["profile"]
    profile_text = f"{profile['name']} - {profile['title']}. {profile['bio']}. Email: {profile['email']}. Location: {profile['location']}. Links: {profile['links']}."
    for i, chunk in enumerate(chunk_text(profile_text)):
        documents.append({
            "text": chunk,
            "metadata": {"source": "personal.json", "section": "profile", "chunk_id": i}
        })

    # Education
    for edu in data["education"]:
        edu_text = f"{edu['degree']} at {edu['institute']} ({edu['year']})"
        if "cgpa" in edu:
            edu_text += f" | CGPA: {edu['cgpa']}"
        if "percentage" in edu:
            edu_text += f" | Percentage: {edu['percentage']}"
        for i, chunk in enumerate(chunk_text(edu_text)):
            documents.append({
                "text": chunk,
                "metadata": {"source": "personal.json", "section": "education", "chunk_id": i}
            })

    # Skills
    skills_text = ", ".join(data["skills"])
    for i, chunk in enumerate(chunk_text(skills_text)):
        documents.append({
            "text": chunk,
            "metadata": {"source": "personal.json", "section": "skills", "chunk_id": i}
        })

    # Projects
    for project in data["projects"]:
        project_text = f"{project['title']}: {project['desc']} | Stack: {', '.join(project['stack'])} | Notes: {project['notes']}"
        if project.get("repo"):
            project_text += f" | Repo: {project['repo']}"
        for i, chunk in enumerate(chunk_text(project_text)):
            documents.append({
                "text": chunk,
                "metadata": {
                    "source": "personal.json",
                    "section": "project",
                    "project_id": project["id"],
                    "chunk_id": i
                }
            })

    return documents


if __name__ == "__main__":
    docs = load_and_chunk()
    for d in docs:
        print(d)
