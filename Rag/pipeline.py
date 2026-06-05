import os
import sys
from pathlib import Path
from typing import Union, Generator, Tuple, List, Dict, Any

# Ensure the root directory of the project is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import TOP_K
from Rag.retriever import retrieve
from Rag.prompt_builder import build_prompt
from Rag.generator import generate

def query(
    question: str,
    stream: bool = False,
    llm: Any = None,
    top_k: int = TOP_K,
    block_type: str = None,
    system_prompt: str = None
) -> Tuple[Union[str, Generator[str, None, None]], List[Tuple[str, Dict[str, Any]]]]:
    """Runs the full RAG pipeline for a given query.
    
    Workflow:
    1. Retrieve relevant chunks using retrieve().
    2. Construct prompt using build_prompt().
    3. Call the generator to produce the response.
    
    Args:
        question: The user's query/question.
        stream: Whether to stream the response generator.
        llm: Optional custom LangChain LLM instance.
        top_k: The number of relevant documents/chunks to retrieve.
        block_type: Optional filter for chunk types (e.g. TEXT, TABLE, IMAGE).
        system_prompt: Optional custom system prompt.
        
    Returns:
        A tuple of (answer, source_chunks) where:
            - answer is either a string or a generator yielding strings.
            - source_chunks is the list of (content, metadata) tuples retrieved.
    """
    # 1. Retrieve source chunks
    source_chunks = retrieve(question, top_k=top_k, block_type=block_type)
    
    # 2. Build the prompt
    prompt = build_prompt(source_chunks, question, system_prompt=system_prompt)
    
    # 3. Generate response
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
                    f"Page={meta.get('page_num', 'unknown')}"
                )

            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\nAssistant: Goodbye!")
            break

        except Exception as e:
            print(f"\nError: {e}\n")

