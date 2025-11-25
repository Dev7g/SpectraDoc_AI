import json
import os
from typing import List, Dict, Any
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from tqdm import tqdm

import config

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe\tesseract.exe"



def extract_text_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    chunks = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")
        if text.strip():
            chunks.append(
                {
                    "type": "text",
                    "content": text,
                    "page": page_idx + 1,
                    "source": f"Page {page_idx + 1}",
                }
            )
    doc.close()
    return chunks


def extract_image_chunks(pdf_path: str, images_dir: str) -> List[Dict[str, Any]]:
    os.makedirs(images_dir, exist_ok=True)
    pages = convert_from_path(pdf_path)
    chunks = []

    for i, pil_page in enumerate(tqdm(pages, desc="Images/OCR")):
        page_num = i + 1
        img_path = os.path.join(images_dir, f"page_{page_num}.png")
        pil_page.save(img_path)

        # OCR
        ocr_text = pytesseract.image_to_string(pil_page)
        if ocr_text.strip():
            chunks.append(
                {
                    "type": "image",
                    "content": ocr_text,
                    "page": page_num,
                    "image_path": img_path,
                    "source": f"Image page {page_num}",
                }
            )

    return chunks


def run_ingestion():
    config.ensure_dirs()
    if not os.path.exists(config.PDF_PATH):
        raise FileNotFoundError(f"Put your PDF at: {config.PDF_PATH}")

    text_chunks = extract_text_chunks(config.PDF_PATH)
    image_chunks = extract_image_chunks(config.PDF_PATH, config.IMAGES_DIR)

    all_chunks = text_chunks + image_chunks
    for idx, c in enumerate(all_chunks):
        c["id"] = idx

    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_chunks)} chunks to {config.CHUNKS_PATH}")


if __name__ == "__main__":
    run_ingestion()
