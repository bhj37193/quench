from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from src.types import ChatCompletionRequest, ChatCompletionResponse

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
_COLLECTION = "quench_cache"


def partition_key(req: ChatCompletionRequest) -> str:
    """Stable hash over (model, generation params, system prompt).

    Semantic search runs only within a partition — prevents
    cross-context false positives.
    """
    system_text = ""
    for msg in req.messages:
        if msg.role == "system":
            system_text += (msg.content or "")

    params = {
        "model": req.model,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "top_p": req.top_p,
        "frequency_penalty": req.frequency_penalty,
        "presence_penalty": req.presence_penalty,
    }
    canonical = json.dumps(params, sort_keys=True) + "|" + system_text
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class CacheHit:
    response: ChatCompletionResponse
    similarity: float


class QdrantStore:
    def __init__(
        self,
        qdrant_url: str = ":memory:",
        collection: str = _COLLECTION,
        ttl_seconds: int = 86400,
    ) -> None:
        if qdrant_url == ":memory:":
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(url=qdrant_url)
        self._collection = collection
        self._ttl_seconds = ttl_seconds
        self._ensure_collection()
        self._partitions: set[str] = set()
        self._load_partitions()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=_EMBEDDING_DIM, distance=Distance.COSINE),
            )

    def _load_partitions(self) -> None:
        offset = None
        while True:
            results, offset = self._client.scroll(
                collection_name=self._collection,
                limit=256,
                with_payload=["partition"],
                offset=offset,
            )
            for point in results:
                if point.payload and "partition" in point.payload:
                    self._partitions.add(point.payload["partition"])
            if offset is None:
                break

    def lookup(
        self, part: str, embedding, threshold: float
    ) -> Optional[CacheHit]:
        now = int(time.time())
        result = self._client.query_points(
            collection_name=self._collection,
            query=embedding.tolist(),
            query_filter=Filter(
                must=[
                    FieldCondition(key="partition", match=MatchValue(value=part)),
                    FieldCondition(key="expires_at", range=Range(gt=now)),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not result.points or result.points[0].score < threshold:
            return None
        hit = result.points[0]
        response = ChatCompletionResponse.model_validate_json(hit.payload["response_json"])
        return CacheHit(response=response, similarity=hit.score)

    def store(
        self,
        part: str,
        embedding,
        response: ChatCompletionResponse,
    ) -> None:
        expires_at = int(time.time()) + self._ttl_seconds
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "partition": part,
                        "response_json": response.model_dump_json(),
                        "expires_at": expires_at,
                    },
                )
            ],
        )
        self._partitions.add(part)

    def evict_expired(self) -> int:
        """Delete all expired entries. Returns count of deleted points."""
        now = int(time.time())
        result = self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="expires_at", range=Range(lt=now))]
                )
            ),
        )
        return getattr(result, "deleted", 0)

    def stats(self) -> dict:
        info = self._client.get_collection(self._collection)
        return {
            "total_entries": info.points_count,
            "partitions": len(self._partitions),
        }


# Backward-compatible alias — proxy.py and eval import CacheStore
CacheStore = QdrantStore
