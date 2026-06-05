import os
import sys
from pathlib import Path
from typing import Union, Generator, Tuple, List, Dict, Any

# Ensure the root directory of the project is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import TOP_K, RERANKER_MODEL_NAME, RERANK_INITIAL_K, RERANK_FINAL_K
from Rag.retriever import retrieve
from Rag.prompt_builder import build_prompt
from Rag.generator import generate

# 1. Initialize FlashRank compressor globally once to keep evaluation fast
from flashrank import Ranker, RerankRequest
try:
    # Uses a highly efficient, CPU-friendly, local cross-encoder model
    ranker = Ranker(model_name=RERANKER_MODEL_NAME)
except Exception as e:
    print(f"Warning: Failed to load FlashRank model. Falling back to default retrieval. Error: {e}")
    ranker = None

def query(
    question: str,
    stream: bool = False,
    llm: Any = None,
    top_k: int = TOP_K,
    block_type: str = None,
    system_prompt: str = None
) -> Tuple[Union[str, Generator[str, None, None]], List[Tuple[str, Dict[str, Any]]]]:
    """Runs the full RAG pipeline for a given query with FlashRank optimization.
    
    Workflow:
    1. Retrieve a broad net of chunks using retrieve().
    2. Rerank retrieved items and compress down to the most relevant top matches.
    3. Construct prompt using build_prompt().
    4. Call the generator to produce the response.
    """
    # 2. Retrieve a broader footprint of raw chunks to guarantee capturing missing pages
    initial_top_k = max(top_k, RERANK_INITIAL_K)
    source_chunks = retrieve(question, top_k=initial_top_k, block_type=block_type)
    
    # 3. Apply Cross-Encoder Reranking if FlashRank loaded successfully
    if ranker and source_chunks:
        try:
            # Reformat your (content, metadata) tuples into dictionary formats expected by FlashRank
            passages = [
                {
                    "id": idx,
                    "text": content,
                    "meta": metadata
                }
                for idx, (content, metadata) in enumerate(source_chunks)
            ]
            
            # Execute cross-encoder scoring evaluation
            rerank_request = RerankRequest(query=question, passages=passages)
            reranked_results = ranker.rerank(rerank_request)
            
            # Compress and keep only the top matches to clear prompt noise
            # Change the slice boundary if your evaluation requires more or less context
            compressed_results = reranked_results[:RERANK_FINAL_K]
            
            # Map back to your pipeline's original Tuple structure to prevent downstream breaks
            source_chunks = [
                (result["text"], result["meta"]) 
                for result in compressed_results
            ]
        except Exception as e:
            print(f"\nReranking error occurred, using raw fallback vector order: {e}")
            source_chunks = source_chunks[:top_k]
    else:
        # Fallback slicing if FlashRank package isn't initialized
        source_chunks = source_chunks[:top_k]
    
    # 4. Build the prompt using compressed, highly-sorted sources
    prompt = build_prompt(source_chunks, question, system_prompt=system_prompt)
    
    # 5. Generate response
    answer = generate(prompt, stream=stream, llm=llm)
    
    return answer, source_chunks

if __name__ == "__main__":
    while True:
        try:
            user_query = input("You: ").strip()

            if user_query.lower() in ["exit", "quit"]:
                print("\nAssistant: Goodbye!")
                break

            if not user_query:
                continue

            answer, sources = query(user_query, stream=False)

            print("\nAssistant:")
            print(answer)

            # Optional: show retrieved sources
            print("\n[Retrieved Sources]")
            for idx, (_, meta) in enumerate(sources, 1):
                print(
                    f"{idx}. "
                    f"Type={meta.get('block_type', 'text').upper()} | "
                    f"Source={meta.get('source', 'unknown')} | "
                    f"Page={meta.get('page_num', 'unknown') or meta.get('page', 'unknown')}"
                )

            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\nAssistant: Goodbye!")
            break

        except Exception as e:
            print(f"\nError: {e}\n")