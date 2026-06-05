import os
import sys
from pathlib import Path
from typing import Union, Generator
from langchain_groq import ChatGroq

# Ensure the root directory of the project is in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import LLM_PROVIDER, LLM_MODEL, TEMPERATURE, MAX_TOKENS, GROQ_API_KEY

try:
    from groq import Groq
except ImportError:
    raise ImportError("Run: pip install groq")

# Initialize global Groq client
_api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
groq_client = Groq(api_key=_api_key) if _api_key else None

def _generate_groq(
    prompt: str,
    stream: bool,
) -> Union[str, Generator[str, None, None]]:
    global groq_client
    if groq_client is None:
        api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        groq_client = Groq(api_key=api_key)
 
    messages = [{"role": "user", "content": prompt}]
 
    if stream:
        def _stream():
            response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return _stream()
    else:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return response.choices[0].message.content.strip()

def generate(
    prompt: str,
    stream: bool = False,
    llm = None,
) -> Union[str, Generator[str, None, None]]:
    """Call the Groq LLM provider or a custom LLM and return a streamed generator or the full answer string.
    
    Args:
        prompt: The constructed prompt string.
        stream: Whether to stream the response chunks.
        llm: An optional LangChain LLM/ChatModel instance to use for generation.
        
    Returns:
        Generator of strings if stream=True, else full answer string.
    """
    if llm is not None:
        if stream:
            def stream_generator():
                for chunk in llm.stream(prompt):
                    if hasattr(chunk, "content"):
                        yield chunk.content
                    else:
                        yield str(chunk)
            return stream_generator()
        else:
            if hasattr(llm, "invoke"):
                res = llm.invoke(prompt)
                if hasattr(res, "content"):
                    return res.content
                return str(res)
            elif hasattr(llm, "predict"):
                return llm.predict(prompt)
            else:
                return str(llm(prompt))

    if LLM_PROVIDER == "groq":
        return _generate_groq(prompt, stream)
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")