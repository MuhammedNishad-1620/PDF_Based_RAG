import sys
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure the root directory of the project is in python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from Rag.pipeline import query

app = FastAPI(
    title="PDF RAG API",
    description="API for querying the PDF-based RAG assistant",
    version="1.0.0"
)

class AskRequest(BaseModel):
    question: str

class SourceDoc(BaseModel):
    content: str
    metadata: Dict[str, Any]

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceDoc] = []

@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        # Run the query pipeline
        answer, sources = query(payload.question, stream=False)
        sources_list = [SourceDoc(content=text, metadata=meta) for text, meta in sources]
        return AskResponse(answer=str(answer), sources=sources_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_files():
    try:
        data_dir = BASE_DIR / "data"
        if data_dir.exists():
            files = [f.name for f in data_dir.glob("*.pdf")]
            return {"files": files}
        return {"files": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
