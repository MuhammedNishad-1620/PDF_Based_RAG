import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM Provider
LLM_PROVIDER = "groq"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Embedding model(hugging_face_all-MiniLM-L6-v2)
EMBEDDING_PROVIDER = "huggingface"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNKING_SIZE = 512
CHUNKING_OVERLAP = 64

# VectorDB
CHROMA_DB_PATH = "chroma_db/"

# Top_k and relevance threshold
TOP_K = 10