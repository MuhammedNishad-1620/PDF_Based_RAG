import sys
from pathlib import Path

# Ensure the root directory of the project is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.chroma_store import ChromaStore
from models.embedding import SentenceTransformerEmbeddings
from config.config import EMBEDDING_MODEL_NAME

# Global initializations to load model weights/connections once
store = ChromaStore()
embedder = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def retrieve(query: str, top_k: int = 5, block_type: str = None) -> list[tuple[str, dict]]:
    """Query Chroma DB and return top-k chunks, optionally filtering by block_type.
    
    Returns:
        List of (text, metadata) tuples.
    """
    try:
        where = {"block_type": block_type} if block_type else None
        
        docs = store.query(query, embedder, n_results=top_k, where=where)
        return [(doc.page_content, doc.metadata) for doc in docs]
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []
