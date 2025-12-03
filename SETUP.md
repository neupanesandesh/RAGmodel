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
```

**Embedding Dimensions:**
- `768` - Fast, cost-effective (recommended)
- `1536` - Balanced
- `3072` - Maximum quality

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
│   ├── main.py         # API endpoints
│   ├── config.py       # Configuration
│   ├── models.py       # Pydantic models
│   └── core/
│       ├── embedder.py      # Gemini embeddings
│       ├── vectorstore.py   # Qdrant operations
│       └── chunking.py      # Text chunking
├── docker-compose.yml        # Base Docker config
├── docker-compose.dev.yml    # Dev overrides
├── docker-compose.prod.yml   # Prod overrides
├── Dockerfile               # Container image
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies (legacy)
├── dev.sh                 # Dev convenience script
├── prod.sh                # Prod convenience script
├── .env.example          # Environment template
└── .env                  # Your config (git-ignored)
```

---

## 🚀 You're All Set!

**Development:** `./dev.sh up` → Code → Save → Refresh → Repeat 🔄

**Production:** `git pull && ./prod.sh up -d --build` → Done ✅

For detailed API usage, visit: http://localhost:8000/docs
