import json
import os
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

import config


class MultiModalVectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = None

    def build_from_chunks(self, chunks: List[Dict[str, Any]]):
        docs: List[Document] = []
        for ch in chunks:
            meta = {
                "id": ch["id"],
                "page": ch["page"],
                "type": ch["type"],
                "source": ch["source"],
                "image_path": ch.get("image_path", None),
            }
            docs.append(Document(page_content=ch["content"], metadata=meta))
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.vectorstore.save_local(path)

    def load(self, path: str):
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )

    def search_text(self, query: str, k: int) -> List[Document]:
        docs_scores = self.vectorstore.similarity_search_with_score(query, k=k * 3)
        text_docs = [d for d, s in docs_scores if d.metadata.get("type") == "text"]
        return text_docs[:k]

    def search_image(self, query: str, k: int) -> List[Document]:
        docs_scores = self.vectorstore.similarity_search_with_score(query, k=k * 3)
        img_docs = [d for d, s in docs_scores if d.metadata.get("type") == "image"]
        return img_docs[:k]


def build_vector_store():
    if not os.path.exists(config.CHUNKS_PATH):
        raise FileNotFoundError("Run ingestion.py first.")
    with open(config.CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    store = MultiModalVectorStore()
    store.build_from_chunks(chunks)
    store.save(config.VECTOR_FAISS_PATH)
    print("FAISS vector store saved.")


if __name__ == "__main__":
    build_vector_store()
