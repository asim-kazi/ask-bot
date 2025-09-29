from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load .env file (for future API keys)
load_dotenv()

app = FastAPI(title="AskMyName-Bot")

# Enable CORS (so frontend can talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/chat")
async def chat(payload: dict):
    """
    Temporary placeholder endpoint.
    payload: {"message": "user input"}
    """
    user_message = payload.get("message", "")
    return {"reply": f"Placeholder reply for: {user_message}"}
