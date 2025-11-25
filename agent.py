import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.graph import StateGraph, END



#1. Load environment variables
load_dotenv()

#2 Define the state
# This is the "memory" of the graph. It holds a list of messages.
# 'add_messages' is a special reducer that appends new messages to the history (automatic memory system of the graph)
# this allows the agent to have a scroll or a chat log so the agent can see what it said and the user said
# instead of overwriting them.Annoated
class AgentState(TypedDict):
    messages:Annotated[List, add_messages]


# 3. Setup tools and LLM
# A. Setup the Knowledge Base (RAG)
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# We turn the database into a "Retriever" (a search engine for our own files)
retriever = db.as_retriever(search_kwargs={"k": 1}) # k=1 means it will return the top 1 search results

# We wrap it as a Tool so the agent can use it
rag_tool = Tool(
    name="search_my_documents",
    description="Searches for private information about project codes, deadlines, and favorites",
    func=lambda query: "\n".join([doc.page_content for doc in retriever.invoke(query)])
)

#B. Add it to the tool kit
tools = [
    TavilySearchResults(max_results=3),
    rag_tool # this is our new tool that we created and added.
]


llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
llm_with_tools = llm.bind_tools(tools)



# 4. Define the Agent Node
def chatbot(state: AgentState):
    # This function takes the current history (state)
    # and asks the LLM what to do next.AgentsState
    print("agent is thinking...")

    # We pass the entire message  history to the llm
    response = llm_with_tools.invoke(state["messages"])

    # We return the new message, which 'add_messages' will append to the list
    return {"messages": [response]}


#5. Define the Tool Node
# This generic node knows how to execute any tool in our 'tools' list
tool_node = ToolNode(tools)

#6 Build the Graph
workflow = StateGraph(AgentState)

# Add our two nodes
workflow.add_node("agent", chatbot)
workflow.add_node("tools", tool_node)

# Set the entry point (where the graph starts)
workflow.set_entry_point("agent")

# Add edges between nodes, this basically draws lines between the two nodes which are called (edges)
workflow.add_edge("tools", "agent")

#7 Define the Logic function
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM asked to call a tool, go to "tools"
    if last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the graph
    return END  

#8. Add the Conditional Edge
workflow.add_conditional_edges(
    "agent",            # Start at the agent node
    should_continue,   # Run this function to decide
)

# this effectively makes a loop that the agent can keep going back to the tools as many times as it needs until it has a final answer


# 9. Compile the graph
# This turns our definition into a runnable applcations
app = workflow.compile()

# --- Test AREA --- 
if __name__ == "__main__":
    print("Research Agent is ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
            
        print("   (Thinking...)")
        
        # Run the graph
        for event in app.stream({"messages": [("user", user_input)]}):
            # The event is a dict where keys are node names (e.g., "agent", "tools")
            # We need to iterate through the values to find messages
            for node_name, node_output in event.items():
                # Check if this node output has messages
                if "messages" in node_output:
                    last_message = node_output["messages"][-1]
                    
                    # CHECK: Is this a raw Tool Message? (The huge JSON block)
                    if isinstance(last_message, ToolMessage):
                        # If yes, print a simple status update instead of the wall of text
                        print("   🔎 (Found data... reading...)")
                        continue
                    
                    # If no, it's the final answer. Print it!
                    if last_message.content:
                         print(f"🤖 Agent: {last_message.content}\n")