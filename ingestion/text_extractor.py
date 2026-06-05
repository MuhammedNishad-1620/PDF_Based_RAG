from pathlib import Path
from langchain_core.documents import Document
from ingestion.pdf_loader import load_pdf

def extract_text_blocks_to_documents(pdf_path: str) -> list:
    """Load PDF, extract text blocks, and convert each block into a LangChain Document.
    
    Metadata fields populated: page_num, source (file name), and block_type='text'.
    """
    documents = []
    try:
        # Resolve source name from PDF path
        source_name = Path(pdf_path).name
        # Load PDF pages
        pages_data = load_pdf(pdf_path)
        # Convert each text block into a LangChain Document
        for page in pages_data:
            page_num = page.get("page_number")
            for block in page.get("text_blocks", []):
                if block.strip():
                    doc = Document(
                        page_content=block,
                        metadata={
                            "page_num": page_num,
                            "source": source_name,
                            "block_type": "text"
                        }
                    )
                    documents.append(doc)
    except Exception as e:
        print(f"[ERROR] Failed to convert text blocks to documents for {pdf_path}: {e}")
        
    return documents
