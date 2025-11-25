import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

# 1 load environment variables
load_dotenv()

# 2 Initalize the Gemini model
# Using gemini-2.5-flash (fast and efficient)
llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", api_key=os.getenv("GEMINI_API_KEY"))

# 3 Initialize the Search tool
# k - 3 means it wil return the top 3 search results
search_tool = TavilySearchResults(max_results=3)

# --- Test AREA -- 
if __name__ == "__main__":
    print("Testing Gemini Connection...")
    try:
        response = llm.invoke("Hello, are you ready to research?")
        print(f" ✅ Gemini Response: {response.content}\n")
    except Exception as e:
        print(f"❌ Gemini Error: {e}")

    print ("Testing Tavily Search Tool...")
    try:
        # We ask the tool to search for something specific 
        search_results = search_tool.invoke("What is the current stock price of NVIDIA?")
        print(f" ✅ Search Results Found!: {len(search_results)}")
        print(f"  Sample: {search_results[0]['content'][:100]}...") # Print first 100 chars
    except Exception as e:
        print(f"❌ Tavily Error:{e}")