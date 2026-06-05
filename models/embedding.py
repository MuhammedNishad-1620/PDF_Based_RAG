import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

# Ensure the root directory of the project is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import EMBEDDING_MODEL_NAME

class SentenceTransformerEmbeddings(Embeddings):
    """LangChain compatible wrapper for SentenceTransformer embedding models."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """Initialize SentenceTransformer model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using SentenceTransformer."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [list(e) for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query using SentenceTransformer."""
        embedding = self.model.encode(text, show_progress_bar=False)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(embedding)


def embed_documents_in_batches(
    documents: list[Document], 
    model_name: str = EMBEDDING_MODEL_NAME, 
    batch_size: int = 32
) -> list[list[float]]:
    """Extract page_content from documents and embed them in batches using SentenceTransformer.
    
    Args:
        documents: A list of LangChain Document objects.
        model_name: The SentenceTransformer model to load.
        batch_size: The batch size used for encoding.
        
    Returns:
        A list of embeddings (lists of floats).
    """
    texts = [doc.page_content for doc in documents]
    if not texts:
        return []
        
    embedder = SentenceTransformerEmbeddings(model_name)
    embeddings = embedder.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return [list(e) for e in embeddings]
