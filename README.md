# 🤖 AutoStream AI Agent: Stateful Conversational RAG Agent

AutoStream is a production-level **Stateful AI Agent** built using **LangGraph**, designed to automate customer inquiries, product education, and lead generation for a video-editing SaaS platform.

The agent leverages a hybrid architecture combining **Retrieval-Augmented Generation (RAG)** with a **Deterministic Safety Fallback** system to guarantee zero hallucination on business-critical data like pricing.

---

## 🏗️ Architecture Overview

The agent follows a modular, state-driven workflow orchestrated by LangGraph:

1.  **Intent Classification**: A hybrid node (Regex + Gemini) identifies if the user is greeting, asking about pricing, or ready to sign up.
2.  **RAG Retriever**: For pricing/product queries, the agent queries a **ChromaDB** vector store (populated from `knowledge.json`) using **HuggingFace Embeddings**.
3.  **Lead Collector**: A multi-turn state machine that gathers and validates user information (Name, Email, Platform) sequentially.
4.  **Response Generator**: A grounded LLM node that synthesizes the final reply.
5.  **Safety Layer**: A programmatic post-processor that verifies LLM output against raw RAG context to prevent hallucinations.

---

## 🚀 Key Features

*   **Stateful Memory**: Utilizes LangGraph's `MemorySaver` to maintain context across multi-turn conversations.
*   **Hybrid Intent Detection**: Combines high-speed rule-matching with LLM-based reasoning for ambiguous queries.
*   **Zero-Hallucination RAG**:
    *   Uses `sentence-transformers/all-MiniLM-L6-v2` for dense local embeddings.
    *   **Safety Fallback**: If the LLM misses core pricing facts ($29/$79), the system automatically defaults to the raw, verified knowledge base context.
*   **Deterministic Lead Flow**: Validates emails and stores lead data in a session state without LLM drift.
*   **CLI Interface**: A clean, interactive terminal UI for real-time testing.

---

## 🛠️ Tech Stack

*   **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
*   **LLM**: [Google Gemini 1.5 Flash](https://ai.google.dev/)
*   **Embeddings**: [HuggingFace (all-MiniLM-L6-v2)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
*   **Vector Database**: [ChromaDB](https://www.trychroma.com/)
*   **Framework**: [LangChain](https://www.langchain.com/)
*   **Language**: Python 3.10+

---

## 📖 How It Works

### 1. The Greeting
Simply say "Hi" or "Hello". The agent identifies the `greeting` intent and provides a warm, tone-aware welcome.

### 2. Pricing & Product Inquiry (RAG)
When you ask "How much does the Pro plan cost?", the agent:
1.  Detects the `pricing` intent.
2.  Retrieves relevant chunks from `knowledge.json`.
3.  Attempts to generate a grounded response.
4.  **Validates** the output. If the response is generic or missing specific pricing, it returns the raw plan data.

### 3. Lead Capture Flow
When you say "I want to sign up for the Pro plan", the agent:
1.  Transitions into a `lead` state.
2.  Asks for your **Name**.
3.  Asks for your **Email** (with built-in validation).
4.  Asks for your **Platform** (e.g., YouTube/Instagram).
5.  Fires a "Mock Lead Capture" tool once all data is valid.

---

## ⚙️ Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
The agent requires a Google Gemini API Key.
```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

### 4. Run the Agent
```bash
python main.py
```

---

## 💬 Example Usage

```text
You: Hello!
Bot: Hello! Welcome to AutoStream, I'm your friendly assistant here to help you.

You: Tell me pricing
Bot: Basic Plan: $29/month (10 videos/month, 720p resolution)
     Pro Plan: $79/month (Unlimited videos, 4K resolution, AI captions)

You: I want the Pro plan
Bot: Let's get you set up with the Pro plan! What is your name?
```

---

## 🔒 Safety Mechanism

A core requirement for this project was **Zero Hallucination**. 

In `agent/nodes.py`, the `response_generator` implements a **Post-Generation Validator**:
*   The LLM is prompted to answer using *only* the provided context.
*   The system then checks the string output for mandatory factual anchors (e.g., $29 and $79).
*   If the LLM output is generic (e.g., "pricing depends on needs") or missing anchors, the system **blocks** the LLM and returns the **verified Knowledge Base context** directly.

---

## 📁 Project Structure

```text
autostream_ai_agent/
├── agent/
│   ├── graph.py        # LangGraph StateGraph definition
│   ├── nodes.py        # Core logic: Intent, RAG, Lead Capture, Response
│   ├── memory.py       # AgentState schema
│   └── tools.py        # Helper functions & validation logic
├── data/
│   ├── knowledge.json  # Source of truth for RAG
│   └── chroma_db/      # Persisted vector database
├── main.py             # CLI Entry Point
├── requirements.txt    # Project dependencies
└── tests/              # Pytest suite
```

---

## 🎬 Demo Video
[Add your demo video link here]

---

## ⏩ Future Improvements
*   **Web UI**: Implementing a Streamlit or Next.js dashboard for a better user experience.
*   **Omnichannel Deployment**: Integration with WhatsApp via Twilio.
*   **API Layer**: Exposing the LangGraph agent as a FastAPI service for mobile app integration.
