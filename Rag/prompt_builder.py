def build_prompt(
    context_chunks: list[tuple[str, dict]], 
    query: str, 
    system_prompt: str = None
) -> str:
    """Assemble system prompt, retrieved context, and user question into a final LLM prompt.
    
    Visually separates text vs image/table chunks.
    
    Args:
        context_chunks: List of (text, metadata) tuples representing retrieved documents.
        query: The user's question.
        system_prompt: Optional custom system prompt.
        
    Returns:
        The fully formatted prompt string.
    """
    sys_prompt = system_prompt or (
    "You are a precise, highly accurate technical assistant answering questions based strictly on the provided context extracted from a PDF.\n"
    "The context contains different types of information, labeled as TEXT, TABLE, or IMAGE.\n\n"
    "CRITICAL GENERATION RULES:\n"
    "1. For any factual, structural, or algorithmic definitions (e.g., propositions, RAG frameworks), ensure you explicitly include their specific technical attributes (e.g., whether they are atomic, factual, self-contained, sequential, iterative, or adaptive).\n"
    "2. If a specific metric, value, or threshold is missing or not explicitly numerical in the context, you MUST state that there is 'no specific' value given or it is 'not mentioned'.\n"
    "3. If the context does not contain the answer, or if the question points to topics completely out of the document's scope, you MUST explicitly include the words 'not mentioned', 'not found', or 'cannot answer' in your response.\n\n"
    "Please use the context details carefully to construct an accurate, highly specific technical response."
)
    
    context_blocks = []
    for idx, (text, metadata) in enumerate(context_chunks, 1):
        block_type = metadata.get("block_type", "text").upper()
        source = metadata.get("source", "unknown")
        page = metadata.get("page_num", "unknown")
        
        # Format chunk header visually depending on block_type
        header = f"=== [{block_type} CHUNK {idx}] (Source: {source}, Page: {page}) ==="
        divider = "=" * len(header)
        
        chunk_str = f"{header}\n{text.strip()}\n{divider}"
        context_blocks.append(chunk_str)
        
    context_section = "\n\n".join(context_blocks) if context_blocks else "No relevant context found."
    
    final_prompt = (
        f"SYSTEM INSTRUCTIONS:\n{sys_prompt}\n\n"
        f"CONTEXT INFORMATION:\n{context_section}\n\n"
        f"USER QUESTION: {query}\n"
    )
    return final_prompt
