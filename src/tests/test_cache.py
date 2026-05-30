"""Unit tests for cache.py — partition key, hit/miss logic, cross-context isolation."""
from __future__ import annotations

import numpy as np

from src.cache import QdrantStore, partition_key
from src.types import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)

THRESHOLD = 0.92


def make_req(user_msg: str, system_msg: str = "You are helpful.", model: str = "gpt-4o-mini") -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[
            ChatMessage(role="system", content=system_msg),
            ChatMessage(role="user", content=user_msg),
        ],
        temperature=0.0,
    )


def make_response(content: str, model: str = "gpt-4o-mini") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model=model,
        choices=[ChatCompletionChoice(index=0, message=ChatCompletionMessage(role="assistant", content=content), finish_reason="stop")],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=10, total_tokens=15),
    )


def random_unit_vec() -> np.ndarray:
    v = np.random.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)


def test_partition_key_stable():
    req = make_req("Hello")
    assert partition_key(req) == partition_key(req)


def test_partition_key_differs_by_system_prompt():
    req_a = make_req("Hello", system_msg="You are a coder.")
    req_b = make_req("Hello", system_msg="You are a chef.")
    assert partition_key(req_a) != partition_key(req_b)


def test_partition_key_differs_by_model():
    req_a = make_req("Hello", model="gpt-4o-mini")
    req_b = make_req("Hello", model="gpt-4o")
    assert partition_key(req_a) != partition_key(req_b)


def test_cache_hit_on_identical_embedding():
    db = QdrantStore()
    req = make_req("What is Python?")
    part = partition_key(req)
    vec = random_unit_vec()
    resp = make_response("Python is a programming language.")

    db.store(part, vec, resp)
    hit = db.lookup(part, vec, threshold=THRESHOLD)

    assert hit is not None
    assert hit.similarity > THRESHOLD
    assert hit.response.choices[0].message.content == "Python is a programming language."


def test_cache_miss_on_low_similarity():
    db = QdrantStore()
    req = make_req("What is Python?")
    part = partition_key(req)
    vec_stored = random_unit_vec()
    # Orthogonal vector → similarity ≈ 0
    vec_query = random_unit_vec()
    # Force them to be orthogonal
    vec_query = vec_query - np.dot(vec_query, vec_stored) * vec_stored
    vec_query = (vec_query / np.linalg.norm(vec_query)).astype(np.float32)

    db.store(part, vec_stored, make_response("Python is a language."))
    hit = db.lookup(part, vec_query, threshold=THRESHOLD)
    assert hit is None


def test_cross_context_isolation():
    """Same user message, different system prompts → different partitions → no cross-hit."""
    db = QdrantStore()
    req_coding = make_req("What is a list?", system_msg="You are a Python expert.")
    req_biology = make_req("What is a list?", system_msg="You are a biology tutor.")

    part_coding = partition_key(req_coding)
    part_biology = partition_key(req_biology)
    assert part_coding != part_biology

    vec = random_unit_vec()
    db.store(part_coding, vec, make_response("A list is [1, 2, 3]."))

    # Biology partition has nothing — must miss
    hit = db.lookup(part_biology, vec, threshold=THRESHOLD)
    assert hit is None


def test_empty_cache_returns_none():
    db = QdrantStore()
    req = make_req("Anything?")
    part = partition_key(req)
    hit = db.lookup(part, random_unit_vec(), threshold=THRESHOLD)
    assert hit is None


def test_stats():
    db = QdrantStore()
    assert db.stats()["total_entries"] == 0

    req = make_req("Hello", system_msg="sys1")
    db.store(partition_key(req), random_unit_vec(), make_response("Hi."))
    req2 = make_req("Hello", system_msg="sys2")
    db.store(partition_key(req2), random_unit_vec(), make_response("Hey."))

    stats = db.stats()
    assert stats["total_entries"] == 2
    assert stats["partitions"] == 2
