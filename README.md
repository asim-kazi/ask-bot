# Ask-Bot 🤖

**Ask-Bot** is an AI-powered personal assistant designed to answer questions using a specialized knowledge base. It uses a **FastAPI** backend, **LangChain** for orchestration, and a modern **React/Tailwind** frontend.

## ✨ Key Capabilities

* **Personalized Answers:** Answers questions about me (skills, projects, education) by leveraging custom data.
* **General Q&A:** Fallback to a powerful Large Language Model (LLM) for general queries.

---

## 🛠 Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Backend** | **FastAPI**, **LangChain**, **ChromaDB** |
| **Frontend** | **React**, **TailwindCSS** |
| **LLM Strategy** | OpenAI GPT API + **RAG (Retrieval-Augmented Generation)** pipeline |

---

## 📂 Project Structure

The project is logically divided to separate the API, UI, and data sources.

| Directory | Purpose | Key Files |
| :--- | :--- | :--- |
| `backend/` | The core API, LangChain logic, and database configuration. | `requirements.txt`, `.env.example` |
| `frontend/` | The React-based chat user interface. | `package.json` |
| `data/` | Stores the **personal data** (`personal.json`) used to train the bot. | `personal.json` |
| `docs/` | For architecture diagrams, design notes, and documentation. | - |

---

## 🚀 Setup Guide

Follow these steps to get the backend running locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)<your-username>/ask-bot.git
cd ask-bot
```

### 2. Install backend deps:
```bash
   cd backend
   pip install -r requirements.txt
```

### 3. Setup env:
```bash
   cp .env.example .env
```

### 4. (Frontend setup will be added later)

## 📜 License
MIT
