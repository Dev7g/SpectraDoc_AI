import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
VECTOR_DIR = os.path.join(DATA_DIR, "vector_store")

PDF_PATH = os.path.join(RAW_DIR, "document.pdf")
CHUNKS_PATH = os.path.join(PROCESSED_DIR, "chunks.json")
VECTOR_FAISS_PATH = os.path.join(VECTOR_DIR, "faiss_index")

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"  # LVLM
TOP_K_TEXT = 4
TOP_K_IMAGE = 4

def ensure_dirs():
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, IMAGES_DIR, VECTOR_DIR]:
        os.makedirs(d, exist_ok=True)
