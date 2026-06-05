# PDF-Based Retrieval-Augmented Generation (RAG) System

A complete, locally deployable PDF-based Retrieval-Augmented Generation (RAG) assistant. The system processes uploaded PDF documents by extracting text, table structures, and images (with OCR captioning), indexing them into a local vector database, and utilizing high-performance LLM generation to answer questions precisely within the scope of your documents.

It provides both a developer-friendly **FastAPI REST API** and a simple, user-friendly **Flask Web UI** to submit questions and review retrieved document sources.

---

## Tech Stack Used

* **RAG & Vector Storage**:
  * **LangChain**: Pipeline structure and document representations.
  * **Chroma DB**: High-efficiency, CPU-friendly local vector database.
  * **Sentence-Transformers (`all-MiniLM-L6-v2`)**: Local embeddings generation.
  * **FlashRank (`ms-marco-MiniLM-L-12-v2`)**: Local cross-encoder reranker for context compression.
* **Extraction Stage**:
  * **PyMuPDF (Fitz)**: High-speed text block extraction.
  * **pdfplumber**: Table structures and layout parsing.
  * **EasyOCR & Pillow**: Image extraction and OCR caption matching.
* **LLM Provider**:
  * **Groq API**: Powered by `llama-3.3-70b-versatile` for fast response generation.
* **Web Services**:
  * **FastAPI** (REST Backend) & **Uvicorn** (ASGI Server).
  * **Flask** (Frontend proxy web-server) & **HTML5/JS** (Client interface).

---

## System Architecture & Methodology

### 1. Multi-Track Document Segmentation
Before files are loaded into the vector database, documents undergo structural segmentation to separate content into three high-fidelity processing tracks:
* **Text Extraction**: PyMuPDF parses paragraphs and layout text blocks while preserving relative page references.
* **Table Structures**: `pdfplumber` identifies boundary lines and tabular grids to export tables as markdown strings, preserving columns and rows.
* **Image OCR**: Visual images are extracted, run through `EasyOCR` to capture printed text inside figures, and merged with surrounding captions.

Once extracted, these tracks are unified into standard Document blocks prior to building the database index.

### 2. Chunking & Overlap Strategy
* **Chunk Size (`512` characters/words)**: Balances semantic density with target context window constraints, preventing individual passages from diluting distinct facts.
* **Chunk Overlap (`64` characters/words)**: Provides a sliding window buffer that ensures critical information spanning chunk boundaries (such as equations, algorithms, or definitions) remains intact.

### 3. Reranking & Hallucination Control
To mitigate LLM hallucination and clean retrieved noise, we employ **Cross-Encoder Reranking** via **FlashRank** (`ms-marco-MiniLM-L-12-v2`):
1. **Raw Vector Retrieval**: The database queries a broad set of candidates (`RERANK_INITIAL_K = 15`).
2. **Relevance Scoring**: FlashRank computes cross-attention relevance scores between the prompt query and candidate passages.
3. **Prompt Compression**: The list is pruned down to the top `3` highest-scoring matches (`RERANK_FINAL_K = 3`).
4. **Impact**: Compressing context down to the most relevant matches prevents the LLM from getting distracted by noise, reducing out-of-scope hallucinations.

### 4. Model Selection & Open-Source Compatibility
* **Embeddings & Reranking**: `all-MiniLM-L6-v2` and `ms-marco-MiniLM-L-12-v2` are highly efficient open-source models designed to run locally on CPU, ensuring high environmental compatibility and low deployment footprints.
* **Llama 3 (via Groq)**: Leverages highly optimized open-source foundation models running on dedicated hardware to ensure low response latency and compatibility with privacy-oriented deployments.

---

## Evaluation Results

We ran automated verification tests to evaluate retrieval quality, accuracy, and page coverage. The results from [eval_results.json](file:///f:/Projects/PDF_Based_RAG/eval_results.json) are summarized below:

* **Total Questions Evaluated**: 12
* **Passed Verdicts**: 9 (75.0% pass rate)
* **Average Keyword Match Score**: 76.1%
* **Average Page Coverage Score**: 79.2%

*Note: Retrieval issues were observed on multi-hop questions requiring joint information across disconnected pages, while factual lookup queries achieved 100% accuracy.*

---

## System Limitations

* **Strict Semantic Dependency & Context Fragmentation**: Subdividing pages into rigid 512-token chunks can fracture tables, charts, or continuous definitions that cross boundaries. If the query requires understanding the complete document flow, fragmented context can result in incomplete answers.
* **Inherent Vulnerability to "Lost in the Middle" (Prompt Saturation)**: Large language models tend to ignore information placed in the middle of long prompts. If the reranking step places critical information in the middle of a dense context payload, the model may suffer from prompt saturation and fail to extract the correct answer.

---

## Setup Instructions

### 1. Configure the Environment
Ensure Python 3.9+ is installed. Then setup the virtual environment and install the required dependencies:

```powershell
# Create virtual environment
python -m venv PDF_RAG

# Activate virtual environment (Windows)
.\PDF_RAG\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a file named `.env` in the root of the project directory and specify your Groq API key:

```env
GROQ_API_KEY="your-groq-api-key-here"
```

### 3. Ingest Documents
Place your target PDF documents in the `data/` directory (e.g., `data/Document.pdf`). Run the ingestion pipeline to parse and index the content into Chroma DB:

```powershell
python ingestion/run_ingestions.py data/Document.pdf
```

---

## Steps to Run the Servers

To run the complete application, you need to start both the FastAPI backend and the Flask UI proxy.

### 1. Start the FastAPI Service
Runs the RAG logic, retriever, and LLM query handler.

```powershell
# In terminal 1 (with virtual environment active)
uvicorn main:app --reload --port 8000
```
The FastAPI documentation will be available at `http://127.0.0.1:8000/docs`.

### 2. Start the Flask UI Service
Serves the user web interface.

```powershell
# In terminal 2 (with virtual environment active)
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`** to access the user interface.

---

## API Endpoint Details

FastAPI serves the following REST endpoints on `http://127.0.0.1:8000`:

### 1. Query Assistant
* **URL**: `/ask`
* **Method**: `POST`
* **Content-Type**: `application/json`
* **Description**: Submits a user question, queries Chroma DB, reranks matching passages, compiles the context prompt, and returns the response answer along with source document chunks.

#### Example Request:
```json
{
  "question": "What is Retrieval-Augmented Generation?"
}
```

#### Example Response:
```json
{
  "answer": "Retrieval-Augmented Generation is a technique that combines information retrieval with text generation...",
  "sources": [
    {
      "content": "Retrieval-Augmented Generation (RAG) is a framework that combines retrieval and generation...",
      "metadata": {
        "block_type": "text",
        "page_num": 1,
        "source": "F:\\Projects\\PDF_Based_RAG\\data\\Document.pdf"
      }
    }
  ]
}
```

### 2. Ingested File Directory
* **URL**: `/files`
* **Method**: `GET`
* **Description**: Returns a list of filenames for PDF files currently stored in the RAG `data/` directory.

#### Example Response:
```json
{
  "files": [
    "Document.pdf"
  ]
}
```
