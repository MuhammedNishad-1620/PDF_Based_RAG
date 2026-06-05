import fitz

def load_pdf(pdf_path: str) -> list:
    """Open PDF, iterate pages, and separate text blocks from image XREFs per page."""
    pages_data = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            # Extract text blocks (block_type 0 is text, block_type 1 is image)
            text_blocks = [
                block[4].strip()
                for block in page.get_text("blocks")
                if block[6] == 0 and block[4].strip()
            ]
            
            # Extract image XREFs
            image_xrefs = [img[0] for img in page.get_images()]
            
            pages_data.append({
                "page_number": i + 1,
                "text_blocks": text_blocks,
                "image_xrefs": image_xrefs
            })
        doc.close()
    except Exception as e:
        print(f"[ERROR] Failed to load PDF {pdf_path}: {e}")
        
    return pages_data
