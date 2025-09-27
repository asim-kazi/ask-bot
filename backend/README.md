# Ask-Bot 🤖

**Ask-Bot** is an AI-powered personal assistant that can:

- Answer questions about me (skills, projects, education) using my custom data.
- Answer general questions using GPT fallback.

## 📂 Project Structure

ask-bot/
├── backend/ # FastAPI + LangChain + Chroma
│ ├── README.md
│ ├── requirements.txt
│ └── .env.example
├── frontend/ # React + Tailwind chat UI
│ └── package.json
├── data/ # Personal data for embeddings
│ └── personal.json
└── docs/ # Architecture diagrams, notes

## 🚀 Setup

1. Clone repo:

```bash
git clone https://github.com/<your-username>/ask-bot.git
cd ask-bot
```

2. Install backend deps:
   cd backend
   pip install -r requirements.txt

3. Setup env:
   cp .env.example .env

4. (Frontend setup will be added later)

## 🛠 Tech Stack

Backend: FastAPI, LangChain, ChromaDB

Frontend: React, TailwindCSS

LLM: OpenAI GPT API + RAG pipeline

## 📜 License

MIT
