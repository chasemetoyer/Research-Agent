# 🤖 Autonomous Research Agent

An intelligent AI agent that autonomously routes queries between web search and private knowledge bases, built with LangGraph's agentic workflow framework.

## 🎯 What Does This Do?

This project demonstrates an **autonomous research agent** that intelligently decides how to answer your questions:

- **Public Information** → Searches the web using Tavily API
- **Private Information** → Searches your local documents using RAG (Retrieval-Augmented Generation)

The agent uses a **cyclic graph architecture** (not a linear chain) to self-correct and re-assess retrieved data before generating final answers, making it more reliable and accurate than traditional chatbots.

## 🏗️ Technical Architecture

### Core Components

- **Orchestration Framework:** LangGraph (State Machine with Cyclic Graphs)
- **Language Model:** Google Gemini 2.5 Flash
- **Vector Database:** ChromaDB with Google Generative AI Embeddings (768-dimensional)
- **Web Search Tool:** Tavily API (optimized for AI agents)
- **Memory System:** RAG (Retrieval-Augmented Generation)

### How It Works

```text
User Query
    ↓
[Agent Node] → Analyzes intent
    ↓
[Router Logic] → Conditional edge decides:
    ├─→ Public info? → [Tavily Search Tool]
    └─→ Private info? → [RAG Retrieval Tool]
         ↓
    [Tool Node] → Executes search
         ↓
    [Agent Node] → Re-evaluates results (self-correction loop)
         ↓
    Final Answer
```

**Key Feature:** The agent can loop back to tools multiple times until it has sufficient information, enabling multi-step reasoning and verification.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Google Gemini API Key
- Tavily API Key

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd "Research Agent"
   ```

2. **Create and activate virtual environment**

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   **Option A: Using `uv` (Recommended - Fast & Modern)**

   ```bash
   # Install uv if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Install dependencies
   uv pip install -r requirements.txt
   ```

   **Option B: Using `pip` (Traditional)**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:

   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

### Initial Setup & Testing

1. **Test API connections**

   ```bash
   python test_gemini.py
   python tools.py
   ```

   This verifies your Gemini and Tavily API keys are working correctly.

2. **Build the vector database**

   ```bash
   python database.py
   ```

   This loads `notes.txt` and creates a ChromaDB vector database for RAG retrieval.

3. **Run the agent**

   ```bash
   python agent.py
   ```

## 📁 Project Structure

```plaintext
Research Agent/
├── agent.py          # Main agent logic with LangGraph workflow
├── database.py       # Vector database setup and document ingestion
├── tools.py          # Tool definitions (Tavily search, RAG retrieval)
├── test_gemini.py    # API connection test
├── requirements.txt  # Python dependencies
├── notes.txt         # Your private knowledge base (customize this!)
├── chroma_db/        # Vector database storage (auto-generated)
├── .env              # API keys (not tracked in git)
└── README.MD         # This file
```

## 💡 Usage Examples

### Example 1: Public Information Query

```text
You: What's the latest news about AI?
🤖 Agent: [Searches web via Tavily and returns current information]
```

### Example 2: Private Information Query

```text
You: What's my project deadline?
🤖 Agent: [Searches your notes.txt via RAG and returns stored information]
```

### Example 3: Mixed Query

```text
You: Compare my favorite movie to current box office hits
🤖 Agent: [Uses RAG for your favorite, Tavily for current hits, then synthesizes]
```

## 🧠 Understanding LangGraph

Unlike traditional AI agents that run linearly, this project uses **LangGraph** to create a **state machine** with loops:

### Traditional Agent (Linear)

```text
User → LLM → Tool → Answer
```

### LangGraph Agent (Cyclic)

```text
User → [Agent Node] ⟷ [Tool Node] → Answer
           ↑____________↓
        (Self-correction loop)
```

### State Management

The agent maintains conversation state using `AgentState`:

- Tracks all messages (user questions, tool results, agent responses)
- Uses `add_messages` reducer to append new messages without overwriting history
- Enables multi-turn conversations with full context

## 🔧 Customization

### Add Your Own Documents

1. Edit `notes.txt` with your private information
2. Run `python database.py` to rebuild the vector database
3. The agent will now search your custom knowledge base

### Adjust Retrieval Settings

In `agent.py`, modify the retriever parameters:

```python
retriever = db.as_retriever(search_kwargs={"k": 3})  # Return top 3 results
```

### Change the LLM

Replace Gemini with another model:

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
```

## 🛠️ Key Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | Core LLM framework |
| `langgraph` | State machine and agentic workflow orchestration |
| `langchain-google-genai` | Google Gemini integration |
| `tavily-python` | Web search optimized for AI agents |
| `chromadb` | Local vector database for semantic search |
| `python-dotenv` | Secure API key management |

## 🔍 How RAG Works

**RAG (Retrieval-Augmented Generation)** enables semantic search over your documents:

1. **Embedding:** Text is converted to 768-dimensional vectors
2. **Storage:** Vectors are stored in ChromaDB
3. **Search:** User queries are embedded and compared using cosine similarity
4. **Retrieval:** Most similar documents are returned to the LLM

**Example:** Searching for "food" will find "pizza" and "burger" even if you didn't type those exact words, because they live in the same semantic "neighborhood."

## 🎓 Learning Resources

This project demonstrates:

- ✅ Agentic AI workflows with LangGraph
- ✅ Tool-calling and function execution
- ✅ RAG implementation with vector databases
- ✅ Conditional routing and decision-making
- ✅ State management in AI applications
- ✅ Self-correcting AI systems

## 📝 License

MIT License - Feel free to use this project for learning and development.

## 🤝 Contributing

Contributions welcome! Feel free to:

- Add new tools (e.g., calculator, database queries)
- Improve the routing logic
- Enhance the RAG retrieval quality
- Add conversation memory persistence

---

**Built with ❤️ using LangGraph and Google Gemini**
