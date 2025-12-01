# Simplified single-stage build - no compilation needed
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY service/ /app/service/
COPY client/ /app/client/
COPY requirements.txt /app/

# Install Python packages with retry logic for network issues
# Install packages one by one to avoid SSL timeout on large downloads
RUN pip install --no-cache-dir fastapi==0.115.0 && \
    pip install --no-cache-dir uvicorn[standard]==0.32.0 && \
    pip install --no-cache-dir pydantic==2.9.2 && \
    pip install --no-cache-dir pydantic-settings==2.6.0 && \
    pip install --no-cache-dir python-dotenv==1.0.1 && \
    pip install --no-cache-dir google-genai==0.3.0 && \
    pip install --no-cache-dir numpy==1.26.4 && \
    pip install --no-cache-dir qdrant-client==1.11.3 && \
    pip install --no-cache-dir requests==2.32.3

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
