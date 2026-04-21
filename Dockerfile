# syntax=docker/dockerfile:1.6
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_OFFLINE=0

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Dependency layer — cached unless pyproject.toml changes.
COPY pyproject.toml /app/
COPY service/ /app/service/
COPY client/ /app/client/

RUN pip install --upgrade pip wheel setuptools && \
    pip install .

RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/huggingface && \
    chown -R appuser:appuser /app /home/appuser

USER appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
