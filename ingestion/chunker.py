from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.config import CHUNKING_SIZE, CHUNKING_OVERLAP

def chunk_documents(documents: list[Document], chunk_size: int = CHUNKING_SIZE, overlap: int = CHUNKING_OVERLAP) -> list[Document]:
    """Split text documents using RecursiveCharacterTextSplitter while keeping image and table documents atomic.
    
    Args:
        documents: A list of LangChain Document objects.
        chunk_size: The target size of each chunk.
        overlap: The overlap between chunks.
        
    Returns:
        A list of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    chunked_docs = []
    for doc in documents:
        # Check block_type metadata, defaulting to "text" if not present
        block_type = doc.metadata.get("block_type", "text")
        
        # If it is explicitly an image or a table, keep it atomic
        if block_type in ("image", "table"):
            chunked_docs.append(doc)
        else:
            # It's a text block or unspecified block type
            split_docs = text_splitter.split_documents([doc])
            # Add chunk_index to tracking metadata
            for idx, split_doc in enumerate(split_docs):
                split_doc.metadata["chunk_index"] = idx
            chunked_docs.extend(split_docs)
            
    return chunked_docs
