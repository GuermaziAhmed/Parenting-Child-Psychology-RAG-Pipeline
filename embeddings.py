"""Embedding utilities wrapping SentenceTransformer.

Provides a simple interface .embed_documents(List[str]) and .embed_query(str).
"""
from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

class EmbeddingsWrapper:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()
