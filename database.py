from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

#1. Load the document
print("📄 Loading notes.txt...")
loader = TextLoader("notes.txt")
documents = loader.load()


# 2. Split the text into chunks
# (We break long text into smaller pieces so the ai can digest it easily)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 3. Initialize the embedding model (using free local model to avoid API quotas)
# This translates text into numbers that AI understands
print("🔧 Loading embedding model...")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=api_key
)

#4 Create the Database
print("💾 Saving to ChromaDB...")
db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="./chroma_db"  # Saves to a folder on your disk
)

print("✅ Database created successfully!")