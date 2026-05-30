from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from src.cache import CacheStore, partition_key
from src.embedder import embed_messages
from src.metrics import (
    cache_entries_total,
    embed_latency_seconds,
    latency_seconds,
    record_cost_saved,
    requests_total,
    similarity_score,
)
from src.router import forward
from src.types import ChatCompletionRequest, ChatCompletionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Quench", version="1.0.0")

_SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.82"))
_TEMP_CACHE_MAX = float(os.environ.get("TEMP_CACHE_MAX", "0.3"))
_QDRANT_URL = os.environ.get("QDRANT_URL", ":memory:")
_DEFAULT_TTL_SECONDS = int(os.environ.get("DEFAULT_TTL_SECONDS", "86400"))
_EVICTION_INTERVAL_SECONDS = 60

_cache: CacheStore | None = None


def _get_cache() -> CacheStore:
    global _cache
    if _cache is None:
        _cache = CacheStore(qdrant_url=_QDRANT_URL, ttl_seconds=_DEFAULT_TTL_SECONDS)
    return _cache


async def _eviction_loop() -> None:
    while True:
        await asyncio.sleep(_EVICTION_INTERVAL_SECONDS)
        try:
            deleted = _get_cache().evict_expired()
            if deleted:
                log.info("EVICT deleted=%d expired entries", deleted)
            stats = _get_cache().stats()
            cache_entries_total.set(stats.get("total_entries", 0) or 0)
        except Exception as exc:
            log.warning("EVICT error: %s", exc)


async def _sse_stream(response: ChatCompletionResponse) -> AsyncGenerator[str, None]:
    """Yield a ChatCompletionResponse as SSE chunks (word-by-word)."""
    content = response.choices[0].message.content or ""

    def _chunk(delta: dict) -> str:
        return "data: " + json.dumps({
            "id": response.id,
            "object": "chat.completion.chunk",
            "created": response.created,
            "model": response.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }) + "\n\n"

    yield _chunk({"role": "assistant", "content": ""})
    words = content.split(" ")
    for i, word in enumerate(words):
        yield _chunk({"content": word if i == 0 else " " + word})
    yield "data: " + json.dumps({
        "id": response.id,
        "object": "chat.completion.chunk",
        "created": response.created,
        "model": response.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }) + "\n\n"
    yield "data: [DONE]\n\n"


@app.on_event("startup")
async def startup() -> None:
    _get_cache()
    asyncio.create_task(_eviction_loop())
    log.info("Quench ready — threshold=%.2f temp_max=%.2f ttl=%ds qdrant=%s",
             _SIMILARITY_THRESHOLD, _TEMP_CACHE_MAX, _DEFAULT_TTL_SECONDS, _QDRANT_URL)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global _SIMILARITY_THRESHOLD, _TEMP_CACHE_MAX
    temp = req.temperature if req.temperature is not None else 1.0

    if temp > _TEMP_CACHE_MAX:
        t0 = time.monotonic()
        log.info("BYPASS temp=%.2f model=%s stream=%s", temp, req.model, req.stream)
        response = await forward(req)
        latency_seconds.labels(result="bypass").observe(time.monotonic() - t0)
        requests_total.labels(model=req.model, result="bypass").inc()
        if req.stream:
            return StreamingResponse(_sse_stream(response), media_type="text/event-stream")
        return JSONResponse(response.model_dump())

    t0 = time.monotonic()
    embedder_name = os.environ.get("EMBEDDER", "local")
    t_embed = time.monotonic()
    embedding = embed_messages(req.messages)
    embed_latency_seconds.labels(embedder=embedder_name).observe(time.monotonic() - t_embed)

    part = partition_key(req)
    cache = _get_cache()

    hit = cache.lookup(part, embedding, _SIMILARITY_THRESHOLD)
    if hit:
        elapsed = time.monotonic() - t0
        log.info("HIT  sim=%.4f model=%s latency=%.1fms stream=%s",
                 hit.similarity, req.model, elapsed * 1000, req.stream)
        requests_total.labels(model=req.model, result="hit").inc()
        latency_seconds.labels(result="hit").observe(elapsed)
        similarity_score.observe(hit.similarity)
        record_cost_saved(
            req.model,
            hit.response.usage.prompt_tokens,
            hit.response.usage.completion_tokens,
        )
        hit.response.system_fingerprint = f"quench-hit-{hit.similarity:.4f}"
        if req.stream:
            return StreamingResponse(_sse_stream(hit.response), media_type="text/event-stream")
        return JSONResponse(hit.response.model_dump())

    response = await forward(req)
    cache.store(part, embedding, response)
    cache_entries_total.inc()
    elapsed = time.monotonic() - t0
    log.info("MISS model=%s latency=%.1fms stream=%s", req.model, elapsed * 1000, req.stream)
    requests_total.labels(model=req.model, result="miss").inc()
    latency_seconds.labels(result="miss").observe(elapsed)
    if req.stream:
        return StreamingResponse(_sse_stream(response), media_type="text/event-stream")
    return JSONResponse(response.model_dump())


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class TuneRequest(BaseModel):
    threshold: Optional[float] = None
    temp_max: Optional[float] = None
    ttl_seconds: Optional[int] = None


@app.post("/tune")
async def tune(body: TuneRequest) -> dict:
    global _SIMILARITY_THRESHOLD, _TEMP_CACHE_MAX, _DEFAULT_TTL_SECONDS
    updated: dict = {}
    if body.threshold is not None:
        _SIMILARITY_THRESHOLD = body.threshold
        updated["threshold"] = body.threshold
    if body.temp_max is not None:
        _TEMP_CACHE_MAX = body.temp_max
        updated["temp_max"] = body.temp_max
    if body.ttl_seconds is not None:
        _DEFAULT_TTL_SECONDS = body.ttl_seconds
        _get_cache()._ttl_seconds = body.ttl_seconds
        updated["ttl_seconds"] = body.ttl_seconds
    return {
        "status": "ok",
        "updated": updated,
        "current": {
            "threshold": _SIMILARITY_THRESHOLD,
            "temp_max": _TEMP_CACHE_MAX,
            "ttl_seconds": _DEFAULT_TTL_SECONDS,
        },
    }


@app.get("/health")
async def health() -> dict:
    stats = _get_cache().stats()
    cache_entries_total.set(stats.get("total_entries", 0) or 0)
    return {"status": "ok", **stats}


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list", "data": []}
