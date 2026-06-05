import pdfplumber
from pathlib import Path
from langchain_core.documents import Document

def table_to_markdown(table_data):
    """Convert a list of lists representing table rows into a markdown table string."""
    if not table_data or not table_data[0]:
        return ""
    # Clean none/newline values and ensure cells are string
    cleaned_rows = [
        [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
        for row in table_data
    ]
    # Filter out empty rows
    cleaned_rows = [r for r in cleaned_rows if any(r)]
    if not cleaned_rows:
        return ""
    # Format header, separator, and data rows
    headers = cleaned_rows[0]
    separator = ["---"] * len(headers)
    md_table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |"
    ]
    for row in cleaned_rows[1:]:
        # Pad row elements if row length is less than header length
        if len(row) < len(headers):
            row += [""] * (len(headers) - len(row))
        md_table.append("| " + " | ".join(row[:len(headers)]) + " |")
    return "\n".join(md_table)

def extract_tables_to_documents(pdf_path):
    """Detect and extract tables from a PDF using pdfplumber, returning LangChain Documents."""
    docs = []
    try:
        source = Path(pdf_path).name
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    tables = page.extract_tables()
                    for idx, tbl in enumerate(tables):
                        try:
                            md_str = table_to_markdown(tbl)
                            if not md_str.strip(): continue
                            docs.append(Document(
                                page_content=md_str,
                                metadata={
                                    "page_num": i + 1,
                                    "source": source,
                                    "block_type": "table",
                                    "table_index": idx
                                }
                            ))
                        except Exception as tbl_err:
                            print(f"[TABLE FORMAT ERROR] Page {i+1}, index {idx}: {tbl_err}")
                except Exception as pg_err:
                    print(f"[TABLE PAGE ERROR] Failed to process page {i+1}: {pg_err}")
    except Exception as doc_err:
        print(f"[TABLE DOC ERROR] Failed to open PDF {pdf_path}: {doc_err}")
    return docs
