FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY evals/ ./evals/

ENV UPSTREAM_BASE_URL=https://api.openai.com/v1
ENV SIMILARITY_THRESHOLD=0.82
ENV TEMP_CACHE_MAX=0.3
ENV QDRANT_URL=:memory:
ENV EMBEDDER=local
ENV DEFAULT_TTL_SECONDS=86400

EXPOSE 4141

CMD ["uvicorn", "src.proxy:app", "--host", "0.0.0.0", "--port", "4141"]
