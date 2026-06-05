import sys
from pathlib import Path
import chromadb
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import CHROMA_DB_PATH

class ChromaStore:
    """Manager for Chroma DB vector database collection connections, upserts, and queries."""
    
    def __init__(self, collection_name: str = "pdf_documents"):
        """Initialize persistent Chroma client and collection."""
        db_path = Path(CHROMA_DB_PATH)
        if not db_path.is_absolute():
            db_path = BASE_DIR / db_path
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert_documents(self, documents: list[Document], embedding_model: Embeddings):
        """Generate embeddings and upsert document chunks to Chroma DB in batches."""
        if not documents:
            return
            
        ids = []
        texts = []
        metadatas = []
        
        for idx, doc in enumerate(documents):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page_num", 1)
            block_type = doc.metadata.get("block_type", "text")
            
            # Formulate unique ID
            doc_id = f"{source}_p{page}_{block_type}"
            if block_type == "text":
                chunk_idx = doc.metadata.get("chunk_index", 0)
                doc_id += f"_c{chunk_idx}_{idx}"
            elif block_type == "table":
                table_idx = doc.metadata.get("table_index", 0)
                doc_id += f"_t{table_idx}_{idx}"
            elif block_type == "image":
                image_xref = doc.metadata.get("image_xref", 0)
                doc_id += f"_img{image_xref}_{idx}"
            else:
                doc_id += f"_{idx}"
                
            ids.append(doc_id)
            texts.append(doc.page_content)
            
            # Clean metadata values to ensure compatibility with Chroma (str, int, float, bool)
            clean_metadata = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_metadata[k] = v
                else:
                    clean_metadata[k] = str(v)
            metadatas.append(clean_metadata)
            
        # Generate embeddings
        embeddings = embedding_model.embed_documents(texts)
        
        # Upsert in batches of 200
        batch_size = 200
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )

    def query(
        self, 
        query_text: str, 
        embedding_model: Embeddings, 
        n_results: int = 5, 
        where: dict = None
    ) -> list[Document]:
        """Query the vector database collection and return matching LangChain Document chunks.
        
        Args:
            query_text: The search query string.
            embedding_model: The model used to embed the query.
            n_results: The number of top documents to retrieve.
            where: A dictionary representing metadata filters (Chroma's standard where filter).
            
        Returns:
            A list of LangChain Document objects with matching chunks and metadata.
        """
        query_embedding = embedding_model.embed_query(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        documents = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metadatas = results["metadatas"][0]
            ids = results["ids"][0]
            distances = results.get("distances", [[]])[0]
            
            for idx in range(len(docs)):
                meta = metadatas[idx].copy() if metadatas[idx] else {}
                meta["id"] = ids[idx]
                if idx < len(distances):
                    meta["distance"] = distances[idx]
                    
                documents.append(Document(
                    page_content=docs[idx],
                    metadata=meta
                ))
                
        return documents
