#  Personal AI Assistant — Agentic AI Capstone Project

A terminal-based conversational AI assistant built with LangChain, 
RAG, and AI Agents. It answers questions from a personal knowledge 
base, searches the web, and performs calculations.

##  Tech Stack
- Python 3.11
- LangChain 0.3.7
- OpenRouter API (Llama 3.3 70B)
- FAISS (vector store for RAG)
- DuckDuckGo Search (ddgs)
- HuggingFace Embeddings

##  Features
- Multi-turn conversation with memory
- RAG: answers from personal knowledge base
- Web search via DuckDuckGo
- Math via Calculator tool
- Fallback system when agent fails
- Persona-driven system prompt (Aiden)

##  How to Run

1. Clone the repo
   git clone https://github.com/dhasarath1409/personal-ai-assistant.git

2. Create virtual environment
   python -m venv venv

3. Activate it
   venv\Scripts\activate  (Windows)

4. Install dependencies
   pip install -r requirements.txt

5. Create .env file
   OPENROUTER_API_KEY=your_api_key_here

6. Run
   python assistant.py

## What I Learned
Built as a capstone project covering LLM API calls, prompt 
engineering, multi-turn chat, LangChain, memory, RAG pipelines, 
and AI agents with real tool use.