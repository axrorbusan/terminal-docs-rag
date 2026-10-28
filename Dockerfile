FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag/ rag/
COPY app/ app/
COPY scripts/ scripts/

# The documents, the embedding model and the index are mounted from ./data and ./models
# (see docker-compose.yml), so the image stays small and the index is built once.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "[ -f data/index/chunks.jsonl ] || (python -m rag.ingest && python -m rag.index); uvicorn app.main:app --host 0.0.0.0 --port 8000"]
