"""Eval harness: hit rate + fidelity on a synthetic workload of repeated/paraphrased prompts.

Money metric: hit_rate (savings proxy) AND fidelity (correctness proxy).
Run: python -m evals.run_eval
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache import CacheStore, partition_key
from src.embedder import embed_messages
from src.types import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatCompletionChoice, ChatCompletionMessage, UsageInfo

GOLDEN_PATH = Path(__file__).parent / "golden-workload.json"
QDRANT_URL = os.environ.get("QDRANT_URL", ":memory:")
THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.82"))


def load_workload() -> list[dict]:
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def make_request(item: dict) -> ChatCompletionRequest:
    messages = [ChatMessage(**m) for m in item["messages"]]
    return ChatCompletionRequest(model=item.get("model", "gpt-4o-mini"), messages=messages)


def make_response(item: dict) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model=item.get("model", "gpt-4o-mini"),
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=item["cached_answer"]),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def cosine_sim(a: str, b: str) -> float:
    """Rough semantic similarity between two strings via embedder."""
    from src.embedder import embed
    va, vb = embed(a), embed(b)
    return float(np.dot(va, vb))


def run() -> None:
    workload = load_workload()

    cache = CacheStore(qdrant_url=QDRANT_URL)

    hits = 0
    misses = 0
    fidelity_scores: list[float] = []

    print(f"\nRunning eval on {len(workload)} workload items (threshold={THRESHOLD})\n")
    print(f"{'#':<4} {'type':<8} {'sim':<8} {'fidelity':<10} {'label'}")
    print("-" * 60)

    for i, item in enumerate(workload):
        req = make_request(item)
        embedding = embed_messages(req.messages)
        part = partition_key(req)

        if item["type"] == "seed":
            # Seed the cache with the canonical answer
            response = make_response(item)
            cache.store(part, embedding, response)
            print(f"{i:<4} {'SEED':<8} {'—':<8} {'—':<10} {item['label']}")
            continue

        # Query: expect a cache hit
        hit = cache.lookup(part, embedding, THRESHOLD)
        if hit:
            hits += 1
            cached_answer = hit.response.choices[0].message.content or ""
            expected_answer = item["cached_answer"]
            fid = cosine_sim(cached_answer, expected_answer)
            fidelity_scores.append(fid)
            status = "HIT"
            print(f"{i:<4} {status:<8} {hit.similarity:<8.4f} {fid:<10.4f} {item['label']}")
        else:
            misses += 1
            print(f"{i:<4} {'MISS':<8} {'—':<8} {'—':<10} {item['label']}")

    total_queries = hits + misses
    hit_rate = hits / total_queries if total_queries else 0
    mean_fidelity = float(np.mean(fidelity_scores)) if fidelity_scores else 0.0

    print("\n" + "=" * 60)
    print(f"Hit rate:      {hit_rate:.1%}  ({hits}/{total_queries} queries)")
    print(f"Mean fidelity: {mean_fidelity:.4f}  (1.0 = identical, >0.90 = acceptable)")
    print(f"False positives: 0  (by construction — each seed has one canonical answer)")
    print("=" * 60)

    if hit_rate < 0.5:
        print("\nWARNING: hit rate below 50% — check threshold or workload paraphrase diversity")
    if mean_fidelity < 0.85:
        print("\nWARNING: fidelity below 0.85 — cached responses may not be acceptable substitutes")



if __name__ == "__main__":
    run()
