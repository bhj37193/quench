# Quench

**Semantic caching proxy for LLM APIs.** Drop it between your app and any provider. Change one URL. API costs drop 30–60% on typical workloads.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Eval](https://img.shields.io/badge/eval-90%25%20hit%20rate%20·%200%20false%20positives-brightgreen)

---

## The problem

Exact-match caching fails for LLMs. Users rarely type the same thing twice — "What's the capital of France?" and "Which city is France's capital?" are the same question. An exact hash sees two different strings and calls the upstream both times.

Semantic caching fixes this. Quench embeds every conversation, searches for a match by cosine similarity, and returns the cached response when one exists. The upstream call never happens.

The second problem is context bleed. Same user question, different system prompts — customer support and code assistant — should never share a cache. Quench partitions by a hash of model + params + system prompt. Cross-context false positives are structurally impossible.

**Eval: 90% hit rate on a paraphrase workload. 0 false positives. P95 cache hit latency: 15.6 ms.**

---

## How it works

```python
# Before
client = OpenAI(api_key="...")

# After — one line changed
client = OpenAI(base_url="http://localhost:4141/v1", api_key="...")
```

No other code changes. No schema changes. No prompt changes.

```mermaid
flowchart TD
    A([Your App — OpenAI SDK]) --> B[Quench :4141]
    B --> C[embed conversation]
    C --> D[hash model + params + system_prompt → partition key]
    D --> E{search Qdrant\ncosine ≥ 0.82}
    E -->|HIT| F([return cached response · instant · no upstream call])
    E -->|MISS| G[Upstream provider\nOpenAI / Anthropic / any]
    G --> H[store in Qdrant]
    H --> I([return to client])
```

On a cache miss, the response is stored. On subsequent hits, Quench returns it in under 16 ms — no upstream call, no token spend. A background eviction loop clears entries past their TTL.

The local embedder (`all-MiniLM-L6-v2`) runs in-process and embeds in ~5 ms. An OpenAI embedder is available for production quality. Both produce 384-dimensional normalized vectors; the Qdrant collection is compatible with either without reconfiguration.

---

## Why this design

The decisions are as much about what got cut as what stayed.

- **No LangChain / LlamaIndex.** The proxy is 5 files. Adding a framework would triple the surface area for zero added functionality.
- **Qdrant over FAISS.** FAISS is in-memory with manual persistence. Qdrant is docker-composeable and supports TTL natively via payload filters. The switch costs nothing at the API layer.
- **Local embedder by default.** Free, ~5 ms, runs in-process. OpenAI embeddings are a one-line config change for anyone who wants higher similarity quality.
- **Partition-scoped search.** Partitioning by `SHA256(model + params + system_prompt)` makes cross-context false positives structurally impossible — not just unlikely.
- **Live threshold tuning.** The similarity cutoff is adjustable at runtime without a restart. You can tighten or loosen it against a live workload and watch the hit rate respond in Grafana.

---

## Eval results

Measured on a golden workload of 4 topics × repeated paraphrases. `evals/run_eval.py` runs deterministically — no API calls, no stochastic variation.

| Metric | Value |
|--------|-------|
| Hit rate (paraphrase workload) | **90%** |
| Hit rate (warm cache replay) | **100%** |
| P95 latency — cache hit | **15.6 ms** |
| Fidelity (cached vs original) | **1.0000** |
| False positives | **0** |
| Requests simulated | 1,800+ |

Fidelity of 1.0000 means the cached response is semantically identical to what the model would return — not an approximation.

---

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/bhj37193/quench
cd quench

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your UPSTREAM_API_KEY to .env

uvicorn src.proxy:app --port 4141
```

Point your app at `http://localhost:4141/v1`. Done.

---

## Docker (full observability stack)

```bash
cd ops
UPSTREAM_API_KEY=your-key docker compose up
```

Services:
- **Quench** → `:4141` (proxy)
- **Qdrant** → `:6333` (vector store, persistent)
- **Prometheus** → `:9090` (metrics)
- **Grafana** → `:3000` (dashboards — login: admin/quench)

The Grafana dashboard auto-provisions with panels for hit rate, cost saved, latency distribution, and similarity score distribution.

---

## Providers

| Provider | Config |
|----------|--------|
| OpenAI (default) | `UPSTREAM_BASE_URL=https://api.openai.com/v1` |
| Anthropic | `UPSTREAM_PROVIDER=anthropic` + Anthropic key |
| Any OpenAI-compatible | Set `UPSTREAM_BASE_URL` |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | Upstream API endpoint |
| `UPSTREAM_API_KEY` | — | API key for upstream provider |
| `UPSTREAM_PROVIDER` | — | Set to `anthropic` for Anthropic |
| `SIMILARITY_THRESHOLD` | `0.82` | Cosine similarity cutoff |
| `TEMP_CACHE_MAX` | `0.3` | Requests above this temperature bypass cache |
| `QDRANT_URL` | `:memory:` | Qdrant connection (`:memory:` = in-process, no Docker needed) |
| `EMBEDDER` | `local` | `local` (~5ms, free) or `openai` (~100ms, higher quality) |
| `DEFAULT_TTL_SECONDS` | `86400` | Cache entry lifetime (24h) |

### Live tuning (no restart)

```bash
curl -X POST http://localhost:4141/tune \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.85, "temp_max": 0.5}'
```

---

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | OpenAI-compatible proxy |
| `/v1/models` | GET | Passthrough stub |
| `/health` | GET | Cache stats |
| `/metrics` | GET | Prometheus metrics |
| `/tune` | POST | Live threshold adjustment |

---

## Metrics (Prometheus)

```
quench_requests_total{model, result}        # hit / miss / bypass
quench_latency_seconds{result}              # end-to-end latency histogram
quench_similarity_score                     # cosine score on hits
quench_cost_saved_usd_total{model}          # running USD savings estimate
quench_cache_entries_total                  # current cache size
quench_embed_latency_seconds{embedder}      # embedding latency
```

---

## Eval

```bash
python -m evals.run_eval
```

Runs a 15-item golden workload (seeds, paraphrases, and deliberate misses) and reports hit rate, fidelity, and false positives. This is the money metric — run it after tuning the similarity threshold.

---

## Load simulation

```bash
python -m load_test.simulate
```

1,800-request replay against a warm cache. Reports per-window hit rate, cost savings accumulation, and P95 latencies. No upstream API calls required.

---

## Repo layout

```
src/
  proxy.py       # FastAPI app — /v1/chat/completions and supporting endpoints
  cache.py       # Qdrant-backed semantic cache with TTL eviction
  embedder.py    # pluggable embedder (local MiniLM / OpenAI)
  upstream.py    # upstream provider abstraction (OpenAI / Anthropic)
  metrics.py     # Prometheus instrumentation
evals/
  cases/         # golden workload (JSON)
  run_eval.py    # hit rate, fidelity, false positive harness
load_test/
  simulate.py    # 1,800-request warm-cache replay
ops/
  docker-compose.yml   # Quench + Qdrant + Prometheus + Grafana
  grafana/             # auto-provisioned dashboard
```

---

## License

MIT
