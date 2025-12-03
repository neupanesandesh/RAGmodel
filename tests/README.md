# Tests

Simple tests for critical parts only.

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=service --cov=client

# Run specific test file
pytest tests/test_chunking.py -v
```

## What We Test

### ✅ test_chunking.py
**Why:** Chunking affects search quality - bugs here break everything.

Tests:
- Empty text handling
- Small text (single chunk)
- Medium text (paragraph splitting)
- Large text (sliding window)
- Edge cases (whitespace, single word)

### ✅ test_config.py
**Why:** Catches missing environment variables before deployment.

Tests:
- Valid configuration passes
- Missing GEMINI_API_KEY fails
- Invalid embedding dimensions fail
- Missing QDRANT_URL fails

## Test Structure

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures (sample texts)
├── test_chunking.py     # Chunking logic tests
└── test_config.py       # Configuration tests
```

**Clean and simple. No complex mocking, no heavy setup.**

## Adding More Tests

Only add tests for **critical parts that break things**.

Don't test:
- ❌ Third-party libraries
- ❌ Simple getters/setters
- ❌ Constants

Do test:
- ✅ Core business logic
- ✅ Error handling
- ✅ Configuration validation
