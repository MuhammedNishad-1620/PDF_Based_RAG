import io
import os
import fitz
from PIL import Image
from pathlib import Path
from langchain_core.documents import Document

_easyocr_reader = None

def _get_ocr_reader():
    """Lazily initialize and return the EasyOCR reader."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'])
    return _easyocr_reader

def find_nearby_caption(img_bbox, text_blocks):
    """Find the most likely figure caption for an image block."""
    best_caption, candidates = "", []
    ix0, iy0, ix1, iy1 = img_bbox
    img_cx = (ix0 + ix1) / 2
    for tx0, ty0, tx1, ty1, text, _, _ in text_blocks:
        text = text.strip()
        if not text: continue
        # Identify typical caption starters
        is_cap = any(text.lower().startswith(kw) for kw in ["figure", "fig.", "fig ", "table", "eq.", "eq ", "image", "img"])
        # Compute gap between image and text block
        v_gap = max(0, ty0 - iy1) if ty0 > iy1 else (max(0, iy0 - ty1) if iy0 > ty1 else 0)
        h_diff = abs(img_cx - (tx0 + tx1) / 2)
        dist = v_gap + 0.5 * h_diff # Spatial score
        if is_cap and dist < 150:
            candidates.append((dist, text))
        elif dist < 80:
            candidates.append((dist + 50, text)) # Penalty for no caption keywords
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_caption = candidates[0][1]
    return best_caption

def extract_images_to_documents(pdf_path):
    """Crop images from PDF, run Tesseract OCR, match captions, and format as LangChain Documents."""
    docs = []
    try:
        source = Path(pdf_path).name
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                # Retrieve all text blocks on the page
                text_blocks = [b for b in page.get_text("blocks") if b[6] == 0]
                for img in page.get_image_info(xrefs=True):
                    try:
                        bbox, xref = img.get("bbox"), img.get("xref")
                        if not bbox or not xref: continue
                        ocr_txt = ""
                        # Crop image region and extract text via OCR
                        try:
                            pix = page.get_pixmap(clip=bbox, dpi=150)
                            reader = _get_ocr_reader()
                            results = reader.readtext(pix.tobytes("png"))
                            ocr_txt = " ".join([res[1] for res in results]).strip()
                        except Exception as e:
                            print(f"[OCR ERROR] Page {i+1}, xref {xref}: {e}")
                        # Find matching caption and construct LangChain Document
                        caption = find_nearby_caption(bbox, text_blocks)
                        if not ocr_txt and not caption: continue
                        docs.append(Document(
                            page_content=f"Image OCR: {ocr_txt}\nCaption: {caption}".strip(),
                            metadata={"page_num": i + 1, "source": source, "block_type": "image", "image_xref": xref}
                        ))
                    except Exception as e:
                        print(f"[IMAGE ERROR] Page {i+1}: {e}")
    except Exception as e:
        print(f"[DOC ERROR] {pdf_path}: {e}")
    return docs
