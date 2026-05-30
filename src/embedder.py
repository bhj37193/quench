from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

from src.types import ChatMessage

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 and openai text-embedding-3-small (reduced)


def messages_to_text(messages: list[ChatMessage]) -> str:
    """Flatten conversation to a single string for embedding.

    Includes all turns so that same last message in different contexts
    produces different embeddings.
    """
    parts = []
    for msg in messages:
        content = msg.content or ""
        parts.append(f"{msg.role}: {content}")
    return "\n".join(parts)


class EmbedderInterface(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...

    def embed_messages(self, messages: list[ChatMessage]) -> np.ndarray:
        return self.embed(messages_to_text(messages))


class LocalEmbedder(EmbedderInterface):
    _model = None

    def _get_model(self):
        if LocalEmbedder._model is None:
            from sentence_transformers import SentenceTransformer
            LocalEmbedder._model = SentenceTransformer("all-MiniLM-L6-v2")
        return LocalEmbedder._model

    def embed(self, text: str) -> np.ndarray:
        vec = self._get_model().encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)


class OpenAIEmbedder(EmbedderInterface):
    def __init__(self) -> None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_EMBED_API_KEY") or os.environ.get("UPSTREAM_API_KEY", "")
        self._client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=_EMBEDDING_DIM,
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


_embedder: EmbedderInterface | None = None


def get_embedder() -> EmbedderInterface:
    global _embedder
    if _embedder is None:
        if os.environ.get("EMBEDDER", "local") == "openai":
            _embedder = OpenAIEmbedder()
        else:
            _embedder = LocalEmbedder()
    return _embedder


# Module-level shims — proxy.py and eval import these directly
def embed(text: str) -> np.ndarray:
    return get_embedder().embed(text)


def embed_messages(messages: list[ChatMessage]) -> np.ndarray:
    return get_embedder().embed_messages(messages)
