import sys
import os
from pathlib import Path
from langchain_core.documents import Document

# Ensure the root directory of the project is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def get_embedding_model():
    from config.config import EMBEDDING_PROVIDER, EMBEDDING_MODEL_NAME
    
    if EMBEDDING_PROVIDER == "huggingface":
        from models.embedding import SentenceTransformerEmbeddings
        print(f"[INGESTION] Loading SentenceTransformer model '{EMBEDDING_MODEL_NAME}'...")
        return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    else:
        raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")


def upsert_to_chroma(documents: list[Document], embedding_model):
    from models.chroma_store import ChromaStore
    store = ChromaStore()
    print(f"[INGESTION] Preparing {len(documents)} document chunks for Chroma DB...")
    store.upsert_documents(documents, embedding_model)
    print(f"[INGESTION] Successfully upserted {len(documents)} document chunks to Chroma DB.")


def ingest_pdf(pdf_path: str) -> bool:
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        print(f"[ERROR] PDF file does not exist: {pdf_path}")
        return False
        
    print(f"\n==========================================")
    print(f"[INGESTION] Starting pipeline for: {pdf_path_obj.resolve()}")
    print(f"==========================================\n")
    
    # 1. Extraction Stage
    print("[INGESTION] Stage 1: Extraction")
    from ingestion.text_extractor import extract_text_blocks_to_documents
    from ingestion.table_extractor import extract_tables_to_documents
    from ingestion.image_extractor import extract_images_to_documents
    
    print(f"[INGESTION] Extracting text blocks...")
    text_docs = extract_text_blocks_to_documents(str(pdf_path_obj))
    print(f"[INGESTION] -> Extracted {len(text_docs)} text documents.")
    
    print(f"[INGESTION] Extracting tables...")
    table_docs = extract_tables_to_documents(str(pdf_path_obj))
    print(f"[INGESTION] -> Extracted {len(table_docs)} table documents.")
    
    print(f"[INGESTION] Extracting images (OCR + caption matching)...")
    image_docs = extract_images_to_documents(str(pdf_path_obj))
    print(f"[INGESTION] -> Extracted {len(image_docs)} image documents.")
    
    all_docs = text_docs + table_docs + image_docs
    print(f"[INGESTION] -> Total extracted documents: {len(all_docs)}")
    
    if not all_docs:
        print("[WARNING] No content extracted from PDF.")
        return False
        
    # 2. Chunking Stage
    print("\n[INGESTION] Stage 2: Chunking")
    from ingestion.chunker import chunk_documents
    from config.config import CHUNKING_SIZE, CHUNKING_OVERLAP
    chunked_docs = chunk_documents(all_docs, chunk_size=CHUNKING_SIZE, overlap=CHUNKING_OVERLAP)
    print(f"[INGESTION] -> Chunked into {len(chunked_docs)} documents using chunk_size={CHUNKING_SIZE}, overlap={CHUNKING_OVERLAP}.")
    
    # 3. Embedding and Database Upsert Stage
    print("\n[INGESTION] Stage 3: Embedding & Database Upsert")
    embedding_model = get_embedding_model()
    upsert_to_chroma(chunked_docs, embedding_model)
    
    print(f"\n==========================================")
    print(f"[INGESTION] Completed pipeline for {pdf_path_obj.name}")
    print(f"==========================================\n")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        ingest_pdf(pdf_path)
    else:
        # Fallback to the default PDF
        default_pdf = BASE_DIR / "data" / "Document.pdf"
        if default_pdf.exists():
            print(f"[INGESTION] No PDF path specified. Using default: {default_pdf}")
            ingest_pdf(str(default_pdf))
        else:
            print("[USAGE] Please specify a PDF file path to ingest:")
            print("        python ingestion/run_ingestions.py path/to/document.pdf")
