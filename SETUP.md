# RAG Embedding Service - Setup Guide

Production-grade embedding service using **Gemini** and **Qdrant** for Retrieval-Augmented Generation (RAG).

---

## 📋 Prerequisites

- **Python 3.11+**
- **Docker & Docker Compose**
- **Gemini API Key** ([Get free key](https://makersuite.google.com/app/apikey))
- **Git**

---

## 🚀 Quick Start

### **For Development (Local Machine)**

```bash
# 1. Clone and navigate
git clone <your-repo-url>
cd RAGmodel

# 2. Setup environment
cp .env.example .env
nano .env  # Add your GEMINI_API_KEY

# 3. Start with hot-reload
./dev.sh up

# 4. Access service
# API Docs: http://localhost:8000/docs
# Health: http://localhost:8000/health
```

### **For Production (Linode VPS)**

```bash
# 1. SSH into server
ssh root@your-server-ip

# 2. Install Docker
apt update && apt upgrade -y
apt install -y docker.io docker-compose git

# 3. Clone and setup
git clone <your-repo-url>
cd RAGmodel
cp .env.example .env
nano .env  # Add GEMINI_API_KEY

# 4. Deploy
./prod.sh up -d --build

# 5. Configure firewall
ufw allow 8000/tcp
ufw allow 22/tcp
ufw enable

# 6. Verify
curl http://localhost:8000/health
```

---

## 🛠️ Development Workflow

### **Starting the Service**

```bash
# Start with hot-reload (recommended)
./dev.sh up

# Start in background
./dev.sh up -d

# View logs
./dev.sh logs -f

# Stop services
./dev.sh down
```

### **Making Code Changes**

1. Edit any `.py` file in `service/` or `client/`
2. Save the file
3. Watch terminal - you'll see: `WARNING: WatchFiles detected changes... Reloading...`
4. Refresh browser at http://localhost:8000/docs
5. **Changes are live! No rebuild needed!**

### **When to Rebuild**

Only rebuild when you:
- Change `pyproject.toml` dependencies
- Change `Dockerfile`
- First time setup

```bash
./dev.sh down
./dev.sh up --build
```

### **What `./dev.sh` Does**

The script runs:
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml [command]
```

**Development mode features:**
- ✅ **Volume mounting** - Your code is synced into container
- ✅ **Editable install** - `pip install -e .` inside container
- ✅ **Hot-reload** - FastAPI restarts on code changes
- ✅ **Fast iteration** - Save → Reload → Test (1-2 seconds)

---

## 🚢 Production Deployment

### **Initial Deployment**

```bash
# On your Linode VPS
cd RAGmodel
./prod.sh up -d --build

# Check status
./prod.sh ps

# View logs
./prod.sh logs -f
```

### **Updating Code**

```bash
# Pull latest changes
git pull

# Rebuild and restart (IMPORTANT: --build flag required!)
./prod.sh up -d --build

# Verify update
./prod.sh logs -f
```

### **What `./prod.sh` Does**

The script runs:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml [command]
```

**Production mode features:**
- ✅ **Regular install** - `pip install .` (code baked into image)
- ✅ **Auto-restart** - Container restarts on failure
- ✅ **Optimized** - No hot-reload overhead
- ✅ **Secure** - No volume mounting, no code changes possible

### **Why `--build` is Required in Production**

In production, code is **baked into the Docker image**. When you `git pull`:
- Code updates on the **server filesystem** ✅
- Code inside **container** is still old ❌

Running `--build` rebuilds the image with new code.

---

## 📦 Installing as Python Package

This service is a proper Python package and can be installed with pip.

### **Local Installation (Development)**

```bash
# Navigate to repo
cd RAGmodel

# Install in editable mode
pip install -e .

# Now you can import from anywhere!
python
>>> from client import EmbeddingClient
>>> from service.core import GeminiEmbedder, chunk_text

# Run the service with console command
embedding-service
```

### **Installation from GitHub**

```bash
# Install package from GitHub
pip install git+https://github.com/yourusername/ragmodel.git

# Use the client library
from client import EmbeddingClient
client = EmbeddingClient("http://your-service-url:8000")
```

### **What `pyproject.toml` Defines**

- **Package name:** `embedding-service`
- **Version:** `1.0.0`
- **Dependencies:** All packages from requirements.txt
- **Packages included:** `client`, `service`, `service.core`
- **Console script:** `embedding-service` command

---

## 🧪 Testing

Simple tests for critical components. Runs in seconds, catches real bugs.

### **Running Tests**

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_chunking.py

# Run with coverage report
pytest --cov=service --cov=client
```

### **What We Test**

**✅ Chunking Logic** (`test_chunking.py`)
- Empty text handling
- Small text → single chunk
- Medium text → paragraph splitting
- Large text → sliding window
- Edge cases (whitespace, single word)

**Why:** Chunking affects search quality - bugs here break everything.

**✅ Configuration** (`test_config.py`)
- Valid configuration passes
- Missing API key caught
- Invalid embedding dimensions caught
- Missing Qdrant URL caught

**Why:** Catches deployment issues before production.

### **Test Structure**

```
tests/
├── conftest.py          # Shared test fixtures
├── test_chunking.py     # Chunking logic tests
└── test_config.py       # Configuration tests
```

**Clean and minimal - only tests critical parts.**

---

## 🔧 Service Configuration

Edit `.env` file to configure:

```bash
# Gemini API Configuration
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=models/gemini-embedding-001
EMBEDDING_DIMENSION=768

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333  # Dev: localhost, Prod: use service name

# Service Settings
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000

# Logging Configuration
ENVIRONMENT=development              # development | production
LOG_LEVEL=INFO                      # TRACE | DEBUG | INFO | WARNING | ERROR | CRITICAL
LOG_DIR=./logs                      # Directory for log files
LOG_RETENTION_DAYS=30               # Days to keep old logs
LOG_ROTATION_SIZE=100 MB            # Size before rotation (100 MB, 500 MB, 1 GB, etc.)
```

**Embedding Dimensions:**
- `768` - Fast, cost-effective (recommended)
- `1536` - Balanced
- `3072` - Maximum quality

**Log Levels:**
- `TRACE` - Ultra-verbose, every operation
- `DEBUG` - Detailed debugging info (dev recommended)
- `INFO` - General informational messages (prod recommended)
- `WARNING` - Warning messages and above
- `ERROR` - Error messages only
- `CRITICAL` - Critical errors only

---

## 📚 API Usage

### **Using Python Client**

```python
from client import EmbeddingClient

# Connect to service
client = EmbeddingClient("http://localhost:8000")

# Create collection
client.create_collection("my_docs")

# Add document (auto-chunked and embedded)
client.add_document(
    collection="my_docs",
    doc_id="doc1",
    text="Your document text here...",
    namespace="optional_namespace"
)

# Search
results = client.search(
    collection="my_docs",
    query="search query",
    k=5
)

# Display results
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.text}")
    print(f"Doc ID: {result.doc_id}")
```

### **Using HTTP Requests**

```bash
# Create collection
curl -X POST http://localhost:8000/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"my_docs"}'

# Add document
curl -X POST "http://localhost:8000/collections/my_docs/documents/text?doc_id=doc1" \
  -H "Content-Type: text/plain" \
  -d "Your document text here"

# Search
curl -X POST http://localhost:8000/collections/my_docs/search \
  -H "Content-Type: application/json" \
  -d '{"query":"search query","k":5}'
```

### **API Endpoints**

- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /collections` - Create collection
- `GET /collections` - List collections
- `GET /collections/{name}` - Get collection info
- `DELETE /collections/{name}` - Delete collection
- `POST /collections/{name}/documents/text` - Add document
- `DELETE /collections/{name}/documents/{doc_id}` - Delete document
- `POST /collections/{name}/search` - Search

---

## 🐳 Docker Architecture

### **Services**

```yaml
qdrant:           # Vector database (port 6333)
embedding-service: # FastAPI application (port 8000)
```

### **Development Setup**
- **Base:** `docker-compose.yml` (common settings)
- **Override:** `docker-compose.dev.yml` (dev-specific)
- **Command:** `./dev.sh` = `docker-compose -f docker-compose.yml -f docker-compose.dev.yml`

### **Production Setup**
- **Base:** `docker-compose.yml` (common settings)
- **Override:** `docker-compose.prod.yml` (prod-specific)
- **Command:** `./prod.sh` = `docker-compose -f docker-compose.yml -f docker-compose.prod.yml`

### **Dockerfile Stages**

```dockerfile
1. Install curl (for healthcheck)
2. Copy pyproject.toml
3. Copy application code (service/, client/)
4. Run: pip install .  (installs package + dependencies)
5. Create non-root user
6. Set healthcheck
7. Expose port 8000
8. Run: uvicorn service.main:app
```

---

## 📊 Logging & Monitoring

The service uses **Loguru** for comprehensive, production-ready logging. Logs are automatically categorized and rotated for easy monitoring on your Linode VPS.

### **Log Files Overview**

All logs are stored in the configured `LOG_DIR` (default: `./logs` or `/app/logs` in Docker):

| File | Content | Use Case |
|------|---------|----------|
| `app.log` | All application logs | General debugging and monitoring |
| `error.log` | Warnings, errors, and critical issues | Quick error checking |
| `requests.log` | HTTP requests/responses with timing | API usage monitoring |
| `performance.log` | Performance metrics (embeddings, searches) | Performance analysis |
| `app.json` | JSON-formatted logs (production only) | Log aggregation tools (ELK, Datadog) |

**Automatic Management:**
- **Rotation:** Files rotate when they reach `LOG_ROTATION_SIZE` (default: 100 MB)
- **Compression:** Old logs are automatically compressed to `.zip`
- **Retention:** Logs older than `LOG_RETENTION_DAYS` (default: 30) are deleted
- **Thread-safe:** Safe for async operations and multiple workers

---

### **Development vs Production Logging**

#### **Development Mode** (`ENVIRONMENT=development`)

**Console Output:**
```
2025-12-07 10:30:15 | INFO     | main:lifespan:54 | Starting Embedding Service
2025-12-07 10:30:16 | SUCCESS  | main:lifespan:66 | Gemini embedder initialized
2025-12-07 10:30:16 | SUCCESS  | main:lifespan:75 | Qdrant vector store connected
2025-12-07 10:30:17 | INFO     | request:log_requests:121 | POST /collections/docs/documents/text
2025-12-07 10:30:17 | INFO     | performance:add_document:262 | Document added: doc123
```

**Features:**
- ✅ **Colored output** - Easy to read in terminal (green=success, red=error, etc.)
- ✅ **DEBUG level** - Shows detailed debugging information
- ✅ **Full diagnostics** - Variable values shown in exceptions
- ✅ **File + line numbers** - Exact code location for each log
- ✅ **Human-readable** - Formatted for developers

**What you see:**
```bash
# Terminal shows colored logs
./dev.sh logs -f

# Log files created in ./logs/
ls -lh logs/
```

---

#### **Production Mode** (`ENVIRONMENT=production`)

**Console Output (JSON):**
```json
{"text": "Starting Embedding Service", "record": {"time": {"timestamp": 1733574615.123}, "level": {"name": "INFO"}}, "message": "Starting Embedding Service", "extra": {"environment": "production"}}
{"text": "Gemini embedder initialized", "record": {"time": {"timestamp": 1733574616.456}, "level": {"name": "SUCCESS"}}, "extra": {"model": "models/gemini-embedding-001", "dimensions": 768}}
{"text": "POST /collections/docs/documents/text - 201", "record": {"level": {"name": "INFO"}}, "extra": {"method": "POST", "status_code": 201, "process_time_ms": 234.56}}
```

**Features:**
- ✅ **JSON format** - Easy parsing for log aggregation tools
- ✅ **INFO level** - Less verbose, production-appropriate
- ✅ **No variable exposure** - Security-focused (no `diagnose=True`)
- ✅ **Structured data** - Machine-readable, perfect for analysis
- ✅ **Log aggregation ready** - Works with ELK, Datadog, Grafana Loki

**What you see:**
```bash
# Docker logs show JSON
./prod.sh logs -f

# Log files created in Docker volume
docker exec embedding-service ls -lh /app/logs/
```

---

### **Accessing Logs**

#### **Development (Local)**

```bash
# View all logs in terminal
./dev.sh logs -f

# View specific service
./dev.sh logs -f embedding-service

# Access log files directly
tail -f logs/app.log
tail -f logs/error.log
tail -f logs/requests.log
tail -f logs/performance.log

# Search for errors
grep "ERROR" logs/app.log
grep "Failed" logs/error.log
```

#### **Production (Linode VPS)**

```bash
# Real-time Docker logs
./prod.sh logs -f

# View logs inside container
docker exec embedding-service tail -f /app/logs/app.log
docker exec embedding-service tail -f /app/logs/error.log
docker exec embedding-service tail -f /app/logs/requests.log
docker exec embedding-service tail -f /app/logs/performance.log

# Copy logs from container to host
docker cp embedding-service:/app/logs ./local-logs

# List all log files with sizes
docker exec embedding-service ls -lh /app/logs/

# Search for specific patterns
docker exec embedding-service grep "ERROR" /app/logs/error.log
docker exec embedding-service grep "timing" /app/logs/performance.log

# View compressed old logs
docker exec embedding-service ls -lh /app/logs/*.zip
```

#### **Production Log Volume**

Logs are stored in a persistent Docker volume (`embedding_logs`):

```bash
# Inspect volume
docker volume inspect ragmodel_embedding_logs

# Find volume location on host
docker volume inspect ragmodel_embedding_logs --format '{{.Mountpoint}}'

# Access volume directly (root required)
sudo ls -lh /var/lib/docker/volumes/ragmodel_embedding_logs/_data/
```

**Benefits:**
- Logs persist even if container is removed
- Can be backed up separately
- Survives `docker-compose down` (NOT `docker-compose down -v`)

---

### **Log Categories Explained**

#### **1. Application Logs** (`app.log`)

Everything that happens in the service:
```
2025-12-07 10:30:15 | INFO     | Starting Embedding Service
2025-12-07 10:30:16 | SUCCESS  | Configuration validated successfully
2025-12-07 10:30:16 | SUCCESS  | Gemini embedder initialized
2025-12-07 10:30:17 | INFO     | Collection created: docs
2025-12-07 10:30:18 | SUCCESS  | Document added to docs: doc123 (5 chunks)
```

**Use for:** General service monitoring, understanding flow

---

#### **2. Error Logs** (`error.log`)

Only warnings and above:
```
2025-12-07 10:35:22 | WARNING  | Collection deleted: old_docs
2025-12-07 10:40:15 | ERROR    | Failed to create collection: docs
  → collection_name: docs
  → error: Collection 'docs' already exists
2025-12-07 10:45:30 | ERROR    | Search failed in nonexistent_collection
```

**Use for:** Quick error checking, alerting, troubleshooting

---

#### **3. Request Logs** (`requests.log`)

Every HTTP request/response:
```
2025-12-07 10:30:17 | INFO     | POST /collections/docs/documents/text
  → method: POST
  → path: /collections/docs/documents/text
  → query_params: doc_id=doc123
  → client_ip: 172.18.0.1
  → user_agent: python-requests/2.32.3
2025-12-07 10:30:17 | INFO     | POST /collections/docs/documents/text - 201
  → method: POST
  → path: /collections/docs/documents/text
  → status_code: 201
  → process_time_ms: 234.56
```

**Use for:** API usage monitoring, performance tracking, debugging client issues

---

#### **4. Performance Logs** (`performance.log`)

Detailed performance metrics:
```
2025-12-07 10:30:17 | INFO     | Document added: doc123
  → operation: add_document
  → doc_id: doc123
  → collection: docs
  → chunks_count: 5
  → text_length: 2548
  → timing:
    - total_ms: 234.56
    - chunking_ms: 12.34
    - embedding_ms: 180.45
    - storage_ms: 41.77

2025-12-07 10:30:20 | INFO     | Search completed: "machine learning algorithms"
  → operation: search
  → collection: docs
  → query_length: 28
  → k: 10
  → results_count: 10
  → timing:
    - total_ms: 156.78
    - embedding_ms: 120.45
    - search_ms: 36.33
```

**Use for:** Performance analysis, optimization, cost tracking (Gemini API usage)

---

#### **5. JSON Logs** (`app.json` - Production Only)

Machine-readable structured logs:
```json
{"text": "Document added: doc123", "record": {"time": {"timestamp": 1733574617.234}}, "extra": {"operation": "add_document", "chunks_count": 5, "timing": {"total_ms": 234.56}}}
```

**Use for:** Log aggregation services, automated monitoring, analytics

---

### **Production Monitoring Setup**

#### **Basic Monitoring (SSH + Commands)**

```bash
# SSH into your Linode VPS
ssh root@your-server-ip

# Watch for errors
watch -n 5 'docker exec embedding-service tail -n 20 /app/logs/error.log'

# Monitor performance
docker exec embedding-service tail -f /app/logs/performance.log

# Check request volume
docker exec embedding-service wc -l /app/logs/requests.log
```

#### **Advanced Monitoring (Log Aggregation)**

**Option 1: ELK Stack (Elasticsearch, Logstash, Kibana)**
```bash
# Ship app.json to Elasticsearch
# Visualize in Kibana dashboards
# Set up alerts on error rates
```

**Option 2: Grafana Loki + Promtail**
```bash
# Lightweight alternative to ELK
# Integrates with Grafana
# Query logs like you query metrics
```

**Option 3: Cloud Services**
- **Datadog** - Full observability platform
- **New Relic** - APM + logging
- **Papertrail** - Simple log aggregation

#### **Setting Up Alerts**

```bash
# Simple script to check error count
#!/bin/bash
ERROR_COUNT=$(docker exec embedding-service grep "ERROR" /app/logs/error.log | wc -l)
if [ $ERROR_COUNT -gt 10 ]; then
    echo "High error count: $ERROR_COUNT" | mail -s "Alert" admin@example.com
fi

# Run via cron every 5 minutes
*/5 * * * * /path/to/check_errors.sh
```

---

### **Customizing Logging**

#### **Change Log Level**

```bash
# Development - more verbose
LOG_LEVEL=DEBUG ./dev.sh up

# Production - less verbose
LOG_LEVEL=WARNING ./prod.sh up -d
```

#### **Change Rotation Size**

```bash
# Rotate at 500 MB instead of 100 MB
LOG_ROTATION_SIZE=500 MB

# Rotate at 1 GB
LOG_ROTATION_SIZE=1 GB
```

#### **Change Retention Period**

```bash
# Keep logs for 60 days instead of 30
LOG_RETENTION_DAYS=60

# Keep logs for 7 days (save disk space)
LOG_RETENTION_DAYS=7
```

#### **Change Log Directory**

```bash
# Local development
LOG_DIR=/var/log/ragmodel

# Docker (requires volume update)
LOG_DIR=/app/logs  # Already configured
```

---

### **Log File Rotation Example**

```
/app/logs/
├── app.log                    # Current log (85 MB)
├── app.log.2025-12-06.zip    # Rotated log from yesterday (12 MB compressed)
├── app.log.2025-12-05.zip    # Rotated log from 2 days ago (11 MB compressed)
├── error.log                  # Current error log (5 MB)
├── requests.log               # Current request log (50 MB)
├── performance.log            # Current performance log (30 MB)
└── app.json                   # Current JSON log (85 MB)
```

**When `app.log` reaches 100 MB:**
1. Compressed to `app.log.2025-12-07.zip`
2. New `app.log` created
3. Old logs beyond 30 days deleted automatically

---

### **Understanding Log Entries**

#### **Success Entry (Development)**
```
2025-12-07 10:30:16 | SUCCESS  | embedder:__init__:33 | GeminiEmbedder initialized
  └─ Date/Time         Level      Component:Function     Message
```

#### **Performance Entry (Production JSON)**
```json
{
  "text": "Document added: doc123",
  "record": {
    "time": {"timestamp": 1733574617.234},
    "level": {"name": "INFO"}
  },
  "extra": {
    "operation": "add_document",
    "doc_id": "doc123",
    "collection": "docs",
    "chunks_count": 5,
    "timing": {
      "total_ms": 234.56,
      "chunking_ms": 12.34,
      "embedding_ms": 180.45,
      "storage_ms": 41.77
    }
  }
}
```

---

### **Troubleshooting Logging Issues**

#### **No logs appearing**
```bash
# Check if logging is configured
docker exec embedding-service env | grep LOG

# Check if log directory exists
docker exec embedding-service ls -la /app/logs/

# Check log configuration in code
docker exec embedding-service cat /app/service/config.py | grep -A 5 "Logging"
```

#### **Logs not rotating**
```bash
# Check current log sizes
docker exec embedding-service du -h /app/logs/

# Verify rotation settings
docker exec embedding-service env | grep LOG_ROTATION
```

#### **Can't access log volume**
```bash
# Check if volume exists
docker volume ls | grep embedding_logs

# Inspect volume
docker volume inspect ragmodel_embedding_logs

# Recreate volume if needed
./prod.sh down
docker volume rm ragmodel_embedding_logs
./prod.sh up -d --build
```

---

## 🔍 Troubleshooting

### **Service won't start**
```bash
./dev.sh logs    # or ./prod.sh logs
```

### **Can't access from browser**
```bash
# Check firewall (production)
ufw status
ufw allow 8000/tcp

# Check if service is running
./dev.sh ps    # or ./prod.sh ps
```

### **Code changes not reflecting (Dev)**
```bash
# Check if you're using dev mode
./dev.sh logs | grep "Reloading"

# Make sure you started with ./dev.sh, not ./prod.sh
```

### **Code changes not reflecting (Prod)**
```bash
# Did you rebuild?
./prod.sh up -d --build

# The --build flag is REQUIRED in production
```

### **Import errors after pip install**
```bash
# Reinstall package
pip install -e .  # Development
pip install .     # Production

# Verify installation
pip list | grep embedding-service
```

### **Qdrant connection errors**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Check Qdrant logs
./dev.sh logs qdrant
```

---

## 📊 Common Commands Cheat Sheet

### **Development**
```bash
./dev.sh up              # Start with hot-reload
./dev.sh up -d           # Start in background
./dev.sh logs -f         # Follow logs
./dev.sh down            # Stop all services
./dev.sh restart         # Restart services
./dev.sh ps              # Check status
./dev.sh up --build      # Rebuild (only when deps change)
```

### **Production**
```bash
./prod.sh up -d --build  # Deploy/update (most common)
./prod.sh logs -f        # Follow logs
./prod.sh down           # Stop all services
./prod.sh restart        # Restart services
./prod.sh ps             # Check status
```

### **Package Management**
```bash
pip install -e .         # Install editable (dev)
pip install .            # Install regular (prod)
embedding-service        # Run service (console command)
```

---

## 🎯 Quick Reference

| Task | Development | Production |
|------|-------------|------------|
| **Start** | `./dev.sh up` | `./prod.sh up -d --build` |
| **Stop** | `./dev.sh down` | `./prod.sh down` |
| **Logs** | `./dev.sh logs -f` | `./prod.sh logs -f` |
| **Update code** | Save file → Auto-reload | `git pull && ./prod.sh up -d --build` |
| **Rebuild** | Only when deps change | Every code update |
| **Hot-reload** | ✅ Yes | ❌ No |
| **Speed** | Instant changes | Fast runtime |

---

## 📝 Project Structure

```
RAGmodel/
├── client/              # Python client library
│   ├── __init__.py
│   └── client.py
├── service/             # FastAPI backend
│   ├── main.py              # API endpoints + request logging
│   ├── config.py            # Configuration + logging settings
│   ├── logging_config.py    # Loguru logging setup
│   ├── models.py            # Pydantic models
│   └── core/
│       ├── embedder.py      # Gemini embeddings (with logging)
│       ├── vectorstore.py   # Qdrant operations (with logging)
│       └── chunking.py      # Text chunking (with logging)
├── tests/                    # Unit tests
│   ├── conftest.py          # Test fixtures
│   ├── test_chunking.py     # Chunking tests
│   └── test_config.py       # Config tests
├── logs/                     # Log files (auto-created, git-ignored)
│   ├── app.log              # All application logs
│   ├── error.log            # Errors and warnings only
│   ├── requests.log         # HTTP requests/responses
│   ├── performance.log      # Performance metrics
│   └── app.json             # JSON logs (production)
├── docker-compose.yml        # Base Docker config
├── docker-compose.dev.yml    # Dev overrides
├── docker-compose.prod.yml   # Prod overrides
├── Dockerfile               # Container image
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
├── dev.sh                 # Dev convenience script
├── prod.sh                # Prod convenience script
├── .env.example          # Environment template
├── .env                  # Your config (git-ignored)
└── SETUP.md              # This file
```

---

## 🚀 You're All Set!

**Development:** `./dev.sh up` → Code → Save → Refresh → Repeat 🔄

**Production:** `git pull && ./prod.sh up -d --build` → Done ✅

For detailed API usage, visit: http://localhost:8000/docs
