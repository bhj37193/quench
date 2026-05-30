"""Synthetic load simulation: 2000+ requests through the real cache + embedder.

No upstream API calls — seeds the cache from the golden workload, then replays
the query set 200× to simulate sustained traffic and show hit-rate convergence.

Run:
    python -m load_test.simulate
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache import QdrantStore, partition_key
from src.embedder import embed_messages
from src.metrics import record_cost_saved
from src.types import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)

GOLDEN_PATH = Path(__file__).parent.parent / "evals" / "golden-workload.json"
THRESHOLD = 0.82
REPLAYS = 200          # 200 × 10 queries = 2000 requests
WINDOW = 100           # report hit rate every N requests
MODEL = "claude-3-5-haiku-20241022"

# Rough token counts per cached response (used for cost savings estimate)
_AVG_PROMPT_TOKENS = 25
_AVG_COMPLETION_TOKENS = 60


def _make_request(item: dict) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=item.get("model", MODEL),
        messages=[ChatMessage(**m) for m in item["messages"]],
        temperature=0.0,
    )


def _make_response(item: dict) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model=item.get("model", MODEL),
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=item["cached_answer"]),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=_AVG_PROMPT_TOKENS,
            completion_tokens=_AVG_COMPLETION_TOKENS,
            total_tokens=_AVG_PROMPT_TOKENS + _AVG_COMPLETION_TOKENS,
        ),
    )


def run() -> None:
    workload = json.loads(GOLDEN_PATH.read_text())
    seeds = [w for w in workload if w["type"] == "seed"]
    queries = [w for w in workload if w["type"] == "query" and
               "MUST MISS" not in w["label"]]  # exclude deliberate miss

    cache = QdrantStore()

    # Seed phase
    print(f"Seeding {len(seeds)} entries...")
    for item in seeds:
        req = _make_request(item)
        embedding = embed_messages(req.messages)
        cache.store(partition_key(req), embedding, _make_response(item))

    # Load simulation
    total = len(queries) * REPLAYS
    print(f"\nSimulating {total:,} requests ({len(queries)} unique queries × {REPLAYS} replays)")
    print(f"Threshold: {THRESHOLD}   Window: {WINDOW} requests\n")
    print(f"{'Requests':>10}  {'Window Hit%':>12}  {'Cumulative Hit%':>16}  {'Cost Saved':>12}")
    print("-" * 58)

    hits_total = 0
    hits_window = 0
    cost_saved_total = 0.0
    request_count = 0
    latencies_hit: list[float] = []
    latencies_miss: list[float] = []

    for replay in range(REPLAYS):
        for item in queries:
            req = _make_request(item)

            t0 = time.monotonic()
            embedding = embed_messages(req.messages)
            hit = cache.lookup(partition_key(req), embedding, THRESHOLD)
            elapsed = time.monotonic() - t0

            request_count += 1

            if hit:
                hits_total += 1
                hits_window += 1
                cost_saved = (
                    _AVG_PROMPT_TOKENS * 0.80 / 1_000_000
                    + _AVG_COMPLETION_TOKENS * 4.00 / 1_000_000
                )
                cost_saved_total += cost_saved
                latencies_hit.append(elapsed * 1000)
            else:
                latencies_miss.append(elapsed * 1000)

            if request_count % WINDOW == 0:
                window_rate = hits_window / WINDOW * 100
                cum_rate = hits_total / request_count * 100
                print(
                    f"{request_count:>10,}  {window_rate:>11.1f}%  "
                    f"{cum_rate:>15.1f}%  ${cost_saved_total:>11.4f}"
                )
                hits_window = 0

    # Final report
    hit_rate = hits_total / request_count * 100
    p95_hit = sorted(latencies_hit)[int(len(latencies_hit) * 0.95)] if latencies_hit else 0
    p95_miss = sorted(latencies_miss)[int(len(latencies_miss) * 0.95)] if latencies_miss else 0

    print("\n" + "=" * 58)
    print(f"Total requests:     {request_count:,}")
    print(f"Cache hits:         {hits_total:,}  ({hit_rate:.1f}%)")
    print(f"Cache misses:       {request_count - hits_total:,}  ({100 - hit_rate:.1f}%)")
    print(f"Cost saved (est.):  ${cost_saved_total:.4f} USD")
    print(f"P95 latency — hit:  {p95_hit:.1f} ms")
    print(f"P95 latency — miss: {p95_miss:.1f} ms")
    print("=" * 58)


if __name__ == "__main__":
    run()
