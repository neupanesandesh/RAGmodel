# Simplified single-stage build - no compilation needed
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml first for better Docker layer caching
# If dependencies don't change, pip install layer is cached
COPY pyproject.toml /app/

# Copy application code
# Note: We need code present for pip install . to work
COPY service/ /app/service/
COPY client/ /app/client/

# Upgrade pip and install wheel first
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install the package in production mode (not editable)
# This installs all dependencies from pyproject.toml and makes the package importable
# --no-cache-dir: Don't store pip cache (reduces image size)
RUN pip install --no-cache-dir .

# Create non-root user and logs directory with proper permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
