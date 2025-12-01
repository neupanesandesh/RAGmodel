# Setup Guide - After Cloning from GitHub

Simple guide to get the embedding service running on your VPS server.

---

## What You Need

- Ubuntu 20.04 or 22.04 server
- Google Gemini API key ([Get one free here](https://makersuite.google.com/app/apikey))
- 10 minutes

---

## Step 1: Install Docker

```bash
# SSH into your server
ssh root@your-server-ip

# Update system
apt update && apt upgrade -y

# Install Docker
apt install -y docker.io docker-compose git

# Start Docker
systemctl start docker
systemctl enable docker
```

---

## Step 2: Clone and Setup

```bash
# Clone this repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO/gemini

# Create environment file
cp .env.example .env
nano .env
```

**Edit the first line with your API key:**

```
GEMINI_API_KEY=your_actual_api_key_here
```

Save: `Ctrl+X` → `Y` → `Enter`

---

## Step 3: Start Services

```bash
# Start everything
docker-compose up -d

# Check if running (wait 30 seconds)
docker-compose ps

# View logs
docker-compose logs -f
```

Wait for: `🚀 Service ready!`

Press `Ctrl+C` to exit logs.

---

## Step 4: Open Firewall

```bash
# Allow the service port
ufw allow 8000/tcp
ufw allow 22/tcp
ufw enable
```

---

## Step 5: Test It

```bash
# On the server
curl http://localhost:8000/health

# From your browser
http://your-server-ip:8000/docs
```

**Done!** Your service is running at `http://your-server-ip:8000`

---

## How to Use It

### Option 1: Using Python

```python
# Copy client/client.py to your project
from client.client import EmbeddingClient

# Connect to your service
client = EmbeddingClient("http://your-server-ip:8000")

# Create collection
client.create_collection("my_docs")

# Add document
client.add_document(
    collection="my_docs",
    doc_id="doc1",
    text="Your document text here..."
)

# Search
results = client.search(
    collection="my_docs",
    query="search query",
    k=5
)

for result in results:
    print(f"{result.score}: {result.text}")
```

### Option 2: Using HTTP Requests

```bash
# Create collection
curl -X POST http://your-server-ip:8000/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"my_docs","vector_size":768}'

# Add document
curl -X POST "http://your-server-ip:8000/collections/my_docs/documents/text?doc_id=doc1" \
  -H "Content-Type: text/plain" \
  -d "Your document text here"

# Search
curl -X POST http://your-server-ip:8000/collections/my_docs/search \
  -H "Content-Type: application/json" \
  -d '{"query":"search query","k":5}'
```

---

## Common Commands

```bash
# Start service
docker-compose up -d

# Stop service
docker-compose down

# View logs
docker-compose logs -f

# Restart service
docker-compose restart

# Update code
git pull origin main
docker-compose up -d --build
```

---

## Troubleshooting

**Service won't start?**

```bash
docker-compose logs
```

**Can't access from browser?**

```bash
# Check firewall
ufw status

# Make sure port 8000 is allowed
ufw allow 8000/tcp
```

**Need to change API key?**

```bash
nano .env
# Edit the file
docker-compose restart
```

---

## API Documentation

Full interactive docs: `http://your-server-ip:8000/docs`

**Main Endpoints:**

- `GET /health` - Check service status
- `POST /collections` - Create collection
- `POST /collections/{name}/documents/text` - Add document
- `POST /collections/{name}/search` - Search

That's it! Your embedding service is ready to use. 🚀
