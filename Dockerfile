FROM python:3.11-slim-bookworm

WORKDIR /app

# wget for healthcheck; git/curl intentionally omitted from the final image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

# Dependency manifest first for better layer caching.
COPY pyproject.toml /app/
COPY requirements.txt /app/

COPY service/ /app/service/
COPY client/ /app/client/

RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps .

# Non-root runtime user. Cache dirs pre-created and owned so HF downloads work.
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs /app/.cache/huggingface && \
    chown -R appuser:appuser /app

USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}" \
    HF_HOME="/app/.cache/huggingface" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget -qO- http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
