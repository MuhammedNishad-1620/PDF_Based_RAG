import os
from pathlib import Path

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if it exists
ENV_PATH = BASE_DIR / '.env'
if ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=ENV_PATH)
    except ImportError:
        # Fallback parser if python-dotenv is not installed yet
        try:
            with open(ENV_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            key, val = parts
                            # Clean quotes if any
                            val = val.strip().strip("'\"")
                            os.environ.setdefault(key.strip(), val)
        except Exception as e:
            print(f"[CONFIG ERROR] Failed to load env file manually: {e}")

# API Keys
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
except Exception as e:
    print(f"[CONFIG ERROR] Failed to load API keys: {e}")
    GEMINI_API_KEY = ""
    OPENAI_API_KEY = ""

# LLM Provider
try:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
except Exception as e:
    print(f"[CONFIG ERROR] Failed to load LLM provider: {e}")
    LLM_PROVIDER = "gemini"

# Embedding Model
try:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
except Exception as e:
    print(f"[CONFIG ERROR] Failed to load embedding model: {e}")
    EMBEDDING_MODEL = "text-embedding-004"

# Vector DB Path
try:
    VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", BASE_DIR / "chroma_db"))
except Exception as e:
    print(f"[CONFIG ERROR] Failed to configure vector database path: {e}")
    VECTOR_DB_PATH = BASE_DIR / "chroma_db"

# Chunking Strategy
try:
    CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive")
except Exception as e:
    print(f"[CONFIG ERROR] Failed to load chunking strategy: {e}")
    CHUNKING_STRATEGY = "recursive"

# Top K Retrieval Settings
try:
    TOP_K = int(os.getenv("TOP_K", "5"))
except Exception as e:
    print(f"[CONFIG ERROR] Failed to load top_k configuration: {e}")
    TOP_K = 5

# Ensure Vector DB Directory exists
try:
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"[CONFIG ERROR] Failed to create vector DB directory: {e}")

# Validate configuration settings
try:
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing but Gemini is chosen as LLM provider.")
    elif LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing but OpenAI is chosen as LLM provider.")
except ValueError as e:
    print(f"[CONFIG WARNING] {e}")
except Exception as e:
    print(f"[CONFIG ERROR] Validation failed: {e}")


