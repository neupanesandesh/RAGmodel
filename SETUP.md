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

**Production mode features:**
- ✅ **Regular install** - `pip install .` (code baked into image)
- ✅ **Auto-restart** - Container restarts on failure
- ✅ **Optimized** - No hot-reload overhead
- ✅ **Secure** - No volume mounting, no code changes possible

**Why `--build` is Required:**
In production, code is **baked into the Docker image**. When you `git pull`, code updates on the server filesystem but the container still has old code. Running `--build` rebuilds the image with new code.

---

## 🧪 Testing

Simple tests for critical components. Runs in seconds, catches real bugs.

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_chunking.py
```

**What We Test:**

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
LOG_ROTATION_SIZE=100 MB            # Size before rotation
```

**Embedding Dimensions:**
- `768` - Fast, cost-effective (recommended)
- `1536` - Balanced
- `3072` - Maximum quality

**Log Levels:**
- `DEBUG` - Detailed debugging info (dev recommended)
- `INFO` - General informational messages (prod recommended)
- `WARNING` - Warning messages and above
- `ERROR` - Error messages only

---

## 📚 API Usage

### **Architecture**

This service uses a **simple two-tier model**:

**Key Concepts:**
- **Collection** - One per company/organization (e.g., `"auditcity"`, `"company-b"`)
- **dataset_id** - Unique dataset identifier within a collection (e.g., `"dallas-dentist"`, `"austin-pizza"`)
  - Each dataset_id represents a distinct dataset/document
  - Raw text is automatically chunked and embedded
  - Optional when searching (provide = search specific dataset, omit = search entire collection)
- **text** - Your raw text content (gets auto-chunked)
- **metadata** - Optional structured data for filtering (location, doc_type, category, etc.)

**How it works:**
```
Collection: "auditcity"
  ├── dataset_id: "dallas-dentist"
  │     └── Raw text → Auto-chunked into ~15-20 pieces
  │         (metadata: {doc_type: "reviews", location: "Dallas"})
  └── dataset_id: "austin-pizza"
        └── Raw text → Auto-chunked into ~15-20 pieces
            (metadata: {doc_type: "reviews", location: "Austin"})
```

**You provide:** Collection + dataset_id + raw text (+ optional metadata)
**Service does:** Chunking → Embedding → Storage
**You search:** By collection + optional dataset_id + optional filters

---

### **Using Python Client**

```python
from client.client import EmbeddingClient

# Connect to service
client = EmbeddingClient("http://localhost:8000")

# Create collection (one per company)
client.create_collection("auditcity")

# Upload batch of documents (recommended approach)
documents = [
    {
        "url": "https://maps.google.com/review1",
        "text": "Great dentist! Very professional and gentle with kids.",
        "meta": {"rating": 5, "author_name": "John Smith", "doc_type": "review"}
    },
    {
        "url": "https://maps.google.com/review2",
        "text": "Amazing service! Clean facility and friendly staff.",
        "meta": {"rating": 5, "author_name": "Jane Doe", "doc_type": "review"}
    }
]

client.add_documents_batch(
    collection="auditcity",
    dataset_id="dallas-dentist",
    documents=documents
)

# Verify upload
datasets = client.list_datasets("auditcity")
print(datasets)  # ['dallas-dentist']

# Search specific dataset
results = client.search(
    collection="auditcity",
    dataset_id="dallas-dentist",
    query="professional service",
    k=5
)

# Search entire collection
all_results = client.search(
    collection="auditcity",
    query="great experience",
    k=10
)

# Search with metadata filters
filtered_results = client.search(
    collection="auditcity",
    query="great dentist",
    filters={"doc_type": "review", "rating": 5},
    k=10
)

# Display results
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.text}")
    print(f"URL: {result.metadata.get('url')}")  # Source URL
    print(f"Dataset: {result.metadata.get('dataset_id')}")
    print(f"Rating: {result.metadata.get('rating')}")
```

---

### **Using HTTP Requests**

```bash
# 1. Create collection
curl -X POST http://localhost:8000/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"auditcity"}'

# 2. Upload batch of documents
curl -X POST "http://localhost:8000/collections/auditcity/documents/batch/dallas-dentist" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "url": "https://maps.google.com/review1",
        "text": "Great dentist! Very professional.",
        "meta": {"rating": 5, "author_name": "John"}
      },
      {
        "url": "https://maps.google.com/review2",
        "text": "Amazing service!",
        "meta": {"rating": 5, "author_name": "Jane"}
      }
    ]
  }'

# 3. List datasets (verify upload)
curl http://localhost:8000/collections/auditcity/datasets

# 4. Search specific dataset
curl -X POST "http://localhost:8000/collections/auditcity/dallas-dentist/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "professional dentist"}'

# 5. Search entire collection
curl -X POST "http://localhost:8000/collections/auditcity/search?k=10" \
  -H "Content-Type: application/json" \
  -d '{"query": "great experience"}'

# 6. Search with filters
curl -X POST "http://localhost:8000/collections/auditcity/search?k=10" \
  -H "Content-Type: application/json" \
  -d '{"query": "great dentist", "filters": {"doc_type": "review", "rating": 5}}'

# 7. Delete dataset
curl -X DELETE http://localhost:8000/collections/auditcity/documents/dallas-dentist
```

---

### **API Endpoints**

| Endpoint | Method | Description | Required Params | Optional Params |
|----------|--------|-------------|-----------------|-----------------|
| `/health` | GET | Health check | - | - |
| `/docs` | GET | Interactive API documentation | - | - |
| `/collections` | POST | Create collection | `name` | - |
| `/collections` | GET | List all collections | - | - |
| `/collections/{name}` | GET | Get collection info | - | - |
| `/collections/{name}` | DELETE | Delete collection | - | - |
| `/collections/{name}/datasets` | GET | List all dataset IDs | - | - |
| `/collections/{name}/documents/batch/{dataset_id}` | POST | Upload batch of documents | `documents` (body) | - |
| `/collections/{name}/documents/{dataset_id}` | DELETE | Delete dataset | - | - |
| `/collections/{name}/search` | POST | Search entire collection | `query` (body) | `k` (default: 5), `filters` |
| `/collections/{name}/{dataset_id}/search` | POST | Search specific dataset | `query` (body) | `k` (default: 5), `filters` |

**Key Features:**
- **Batch Upload**: Upload multiple documents in one request with {url, text, meta} format
- **RESTful Design**: dataset_id in URL path for clarity
- **Clear Search Routing**: Separate endpoints for collection vs dataset search
- **Flexible Search**: Configurable top-k (default: 5)
- **Metadata Filtering**: Custom filters to narrow results
- **Source Tracking**: Each document includes its URL for citation
- **Retry Logic**: Robust retry mechanism with exponential backoff
- **No Timeout Limits**: Upload process runs until complete

---

## ⏱️ Handling Large Dataset Uploads

For large datasets (gigabytes of text, hundreds of thousands of chunks), uploads can take hours.

### **Understanding the Process**

When you upload a dataset, three main steps happen:
1. **Chunking** (fast) - Splits text into manageable pieces
2. **Embedding** (slow) - Generates vector embeddings for each chunk
   - This is the bottleneck for large datasets
   - Processes in batches of 20 chunks with 2-second delays (API rate limits)
   - Example: 10,000 chunks = ~500 batches = ~16 minutes minimum
3. **Storage** (fast) - Saves to Qdrant vector database

### **Monitoring Progress**

Watch server logs in real-time:

```bash
# Development
./dev.sh logs -f embedding-service

# Production
./prod.sh logs -f embedding-service
```

Look for progress indicators:
```
2025-12-23 10:31:42 | INFO     | [1/3] Chunking text: 5,000,000 characters
2025-12-23 10:31:45 | SUCCESS  | ✓ Chunking complete: 12,500 chunks in 2.8s
2025-12-23 10:31:45 | INFO     | [2/3] Embedding 12,500 chunks...
2025-12-23 11:47:23 | SUCCESS  | ✓ Embedding complete: 12,500 chunks in 4538.2s
2025-12-23 11:47:23 | INFO     | [3/3] Storing 12,500 chunks to Qdrant...
2025-12-23 11:47:31 | SUCCESS  | ✓ Storage complete: 12,500 chunks in 7.9s
2025-12-23 11:47:31 | SUCCESS  | ✓✓✓ UPLOAD COMPLETE: 12,500 chunks in 4628.0s
```

Verify upload succeeded:
```bash
# Check if dataset exists
curl http://localhost:8000/collections/auditcity/datasets
```

### **Estimating Upload Time**

**Rule of thumb:**
- **Chunking:** ~1-3 seconds per 1M characters
- **Embedding:** ~0.36 seconds per chunk (batch of 20 + 2s delay)
  - 1,000 chunks ≈ 18 seconds
  - 10,000 chunks ≈ 3 minutes
  - 100,000 chunks ≈ 30 minutes
  - 1,000,000 chunks ≈ 5 hours
- **Storage:** ~0.5-1 second per 1,000 chunks

**Example:**
```
5 GB text → ~1.25M chunks
Chunking: ~15 seconds
Embedding: ~6.25 hours
Storage: ~20 seconds
Total: ~6.3 hours
```

**Note:** If client disconnects, the upload continues running on the server. Check logs or use `list_datasets()` to verify completion.

---

## 🐳 Docker Architecture

### **Services**

```yaml
qdrant:           # Vector database (port 6333)
embedding-service: # FastAPI application (port 8000)
```

### **Development**
- **Command:** `./dev.sh` = `docker-compose -f docker-compose.yml -f docker-compose.dev.yml`
- **Features:** Volume mounting, hot-reload, editable install

### **Production**
- **Command:** `./prod.sh` = `docker-compose -f docker-compose.yml -f docker-compose.prod.yml`
- **Features:** Code baked into image, auto-restart, optimized

---

## 📊 Logging

The service uses **Loguru** for structured logging with automatic rotation and retention.

### **Log Files**

All logs in `LOG_DIR` (default: `./logs` or `/app/logs` in Docker):

| File | Content |
|------|---------|
| `app.log` | All application logs |
| `error.log` | Warnings and errors only |
| `requests.log` | HTTP requests/responses |
| `performance.log` | Performance metrics (timing, chunks, etc.) |
| `app.json` | JSON logs (production only) |

**Auto-managed:**
- Rotate at `LOG_ROTATION_SIZE` (default: 100 MB)
- Compress old logs to `.zip`
- Delete logs older than `LOG_RETENTION_DAYS` (default: 30)

### **Development Logging**

Colored console output, DEBUG level, human-readable:
```
2025-12-07 10:30:15 | INFO     | Starting Embedding Service
2025-12-07 10:30:16 | SUCCESS  | Gemini embedder initialized
2025-12-07 10:30:17 | INFO     | POST /collections/auditcity/documents/batch/dallas-dentist
```

### **Production Logging**

JSON format, INFO level, machine-readable:
```json
{"text": "Starting Embedding Service", "record": {"time": {"timestamp": 1733574615.123}}}
{"text": "Gemini embedder initialized", "extra": {"model": "models/gemini-embedding-001"}}
```

### **Accessing Logs**

**Development:**
```bash
./dev.sh logs -f              # Docker logs
tail -f logs/app.log          # File logs
grep "ERROR" logs/error.log   # Search errors
```

**Production:**
```bash
./prod.sh logs -f                                      # Docker logs
docker exec embedding-service tail -f /app/logs/app.log  # File logs
docker cp embedding-service:/app/logs ./local-logs       # Copy logs
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

# Check if running
./dev.sh ps    # or ./prod.sh ps
```

### **Code changes not reflecting (Dev)**
```bash
# Check if using dev mode
./dev.sh logs | grep "Reloading"

# Make sure started with ./dev.sh, not ./prod.sh
```

### **Code changes not reflecting (Prod)**
```bash
# Did you rebuild?
./prod.sh up -d --build

# --build flag is REQUIRED in production
```

### **Qdrant connection errors**
```bash
# Check if Qdrant running
docker ps | grep qdrant

# Check Qdrant logs
./dev.sh logs qdrant
```

---

## 📊 Quick Reference

| Task | Development | Production |
|------|-------------|------------|
| **Start** | `./dev.sh up` | `./prod.sh up -d --build` |
| **Stop** | `./dev.sh down` | `./prod.sh down` |
| **Logs** | `./dev.sh logs -f` | `./prod.sh logs -f` |
| **Update code** | Save file → Auto-reload | `git pull && ./prod.sh up -d --build` |
| **Rebuild** | Only when deps change | Every code update |
| **Hot-reload** | ✅ Yes | ❌ No |

**Common Commands:**
```bash
# Development
./dev.sh up              # Start with hot-reload
./dev.sh logs -f         # Follow logs
./dev.sh down            # Stop

# Production
./prod.sh up -d --build  # Deploy/update
./prod.sh logs -f        # Follow logs
./prod.sh down           # Stop

# Testing
pip install -e ".[dev]"  # Install dev dependencies
pytest                   # Run tests
pytest -v                # Verbose output
```

---

## 🚀 You're All Set!

**Development:** `./dev.sh up` → Code → Save → Refresh → Repeat 🔄

**Production:** `git pull && ./prod.sh up -d --build` → Done ✅

**API Docs:** http://localhost:8000/docs
