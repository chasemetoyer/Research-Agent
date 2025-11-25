# 🤖 Autonomous Researcher Agent (LangGraph + RAG)

## Project Overview

An intelligent agentic workflow designed to autonomously distinguish between general knowledge queries and proprietary internal data. Built to bridge the gap between LLMs and actionable business intelligence.

## Technical Architecture

* **Orchestration:** LangGraph (State Machine/Cyclic Graph)
* **LLM:** Google Gemini 1.5 Flash
* **Memory (RAG):** ChromaDB (Vector Store) with Google Generative Embeddings (768-dim)
* **External Tools:** Tavily API (Web Search)

## How It Works

1. **Router Logic:** A conditional edge analyzes the user query intent.
2. **Tool Routing:**
    * *Public Info* → Triggers Tavily Search Node.
    * *Private Info* → Triggers RAG Retrieval Node.
3. **Self-Correction:** The agent utilizes a cyclical graph, allowing it to re-assess retrieved data before generating a final answer.

## Setup

Notable Dependencies:

* Langchain
* Langgraph - The framework for building our agents"brain" (state machine)
* Langchain-Gemini
* Tavily - the search engine tool optimized for AI agents
* Chromadb - Our local vector database for memory (RAG)
* Python-dotenv - to keep my API keys secure

# Steps

# 1st Install all dependencies using pip

# 2nd Then you need to make a tools.py file and run it to test the connection to Gemini and Tavily

* this is basically for testing the connection to the search tools and gemini to make sure everything is working correctly.

# 3rd. Next we are going to create a langchain agent using LangGraph to build our agents brain and loops

## What is Langraph

we are using LangGraph. Standard generic agents just run in a straight line. LangGraph allows us to build a loop (a state machine)

This is basically a simple flow.

1. **Start**: User asks a question
2. **Agent Node**: The LLM thinks. "Do I know this, or do i NEED TO SEARCH?
3. *Conditional Edge:

* if it needs info -> go to **Tool Node**

* if it has the answer 0> go to **End**

To create this flow , we need to Define the state

* We need a way for the agent to keep track of the conversation (the user's question _ the search results + the final answer). In LnagGraph, this is called the **State**

This would be first start to create it by using creating a file named agent.py

## Agent.py

    1. Define the state
    2. Setup tools and LLM
    3. Define the Agent Node
    4. Define the Tool Node
    5. Build the Graph
    6. Define the Logic function
    7. Add the Conditional Edge
    8. Compile the graph    

# 4th We need to give our agent long-term memory using ChromaDB

I created a file named database.py  

* this file is responsible for loading the notes.txt file and creating a vector database from it.
*

The reason we use a vector database is because of this thing called **semantic similarity**
this allows the ai agent to search through data almost instantaneously.
We are basically turning  words and letters into numbers and then comparing them to find the most similar documents.

* an examples of this is "Blue" and "Green" live in the same neighborhood,
* pizza and burger live in a different neighborhood.
* If you search for "Food," the AI looks at that neighborhood and finds "Pizza," even if you didn't type the word "Pizza."

# 5th Step now we need to give our agent another **tool** which is the **ChromaDB** tool we call this a **Retrieval Tool**

* go back to your agent.py file and add the following and add the retreiever tool to the tools list
