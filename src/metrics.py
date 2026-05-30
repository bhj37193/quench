from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

requests_total = Counter(
    "quench_requests_total",
    "Total requests",
    ["model", "result"],  # result: hit | miss | bypass
)

latency_seconds = Histogram(
    "quench_latency_seconds",
    "End-to-end request latency",
    ["result"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

similarity_score = Histogram(
    "quench_similarity_score",
    "Cosine similarity score on cache hits",
    buckets=[0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99, 1.0],
)

cost_saved_usd_total = Counter(
    "quench_cost_saved_usd_total",
    "Estimated USD saved by serving from cache instead of upstream",
    ["model"],
)

cache_entries_total = Gauge(
    "quench_cache_entries_total",
    "Current number of entries in the cache",
)

embed_latency_seconds = Histogram(
    "quench_embed_latency_seconds",
    "Embedding latency",
    ["embedder"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
)


# Rough cost-per-token (USD/token) for common models.
# Used to estimate savings on cache hits.
_INPUT_COST_PER_TOKEN: dict[str, float] = {
    "gpt-4o-mini": 0.150 / 1_000_000,
    "gpt-4o": 5.00 / 1_000_000,
    "claude-3-5-sonnet-20241022": 3.00 / 1_000_000,
    "claude-3-5-haiku-20241022": 0.80 / 1_000_000,
    "claude-3-haiku-20240307": 0.25 / 1_000_000,
    "claude-sonnet-4-5": 3.00 / 1_000_000,
}
_OUTPUT_COST_PER_TOKEN: dict[str, float] = {
    "gpt-4o-mini": 0.600 / 1_000_000,
    "gpt-4o": 15.00 / 1_000_000,
    "claude-3-5-sonnet-20241022": 15.00 / 1_000_000,
    "claude-3-5-haiku-20241022": 4.00 / 1_000_000,
    "claude-3-haiku-20240307": 1.25 / 1_000_000,
    "claude-sonnet-4-5": 15.00 / 1_000_000,
}
_DEFAULT_INPUT_COST = 1.00 / 1_000_000
_DEFAULT_OUTPUT_COST = 2.00 / 1_000_000


def record_cost_saved(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    inp = _INPUT_COST_PER_TOKEN.get(model, _DEFAULT_INPUT_COST)
    out = _OUTPUT_COST_PER_TOKEN.get(model, _DEFAULT_OUTPUT_COST)
    saved = inp * prompt_tokens + out * completion_tokens
    cost_saved_usd_total.labels(model=model).inc(saved)
