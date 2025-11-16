# backend/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
from retrieval_qa import answer_query

app = FastAPI(title="Ask-Bot Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only; in prod restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatPayload(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/chat")
async def chat(payload: ChatPayload):
    """
    Payload: {"message":"..."}
    Returns: {"reply": "...", "sources": [...], "retrieved": bool}
    """
    q = payload.message.strip()
    if not q:
        return {"reply": "Please provide a message in the 'message' field.", "sources": [], "retrieved": False}
    res = answer_query(q)
    return {"reply": res["answer"], "sources": res["source_documents"], "retrieved": res["used_retrieval"], "top_score": res.get("top_score")}

# For running with: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
