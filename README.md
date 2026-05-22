# RAG Document QA

Production-style Retrieval-Augmented Generation service. Upload markdown or
text documents, ask questions, get answers grounded in your documents with
source attribution.

## Architecture

```
   +-------------+      +----------------+      +----------------+
   |   Client    | ---> |   FastAPI      | ---> |  Claude API    |
   |  (curl/UI)  | <--- |   (this app)   | <--- |  (Anthropic)   |
   +-------------+      +-------+--------+      +----------------+
                                |
                                v
                        +---------------+
                        |  PostgreSQL   |
                        |  + pgvector   |
                        +---------------+

   Embedding model: sentence-transformers/all-MiniLM-L6-v2 (runs in-process)
```

### Why these choices

- **pgvector over Pinecone/Weaviate**: avoids vendor lock-in, runs in any
  Postgres, free for production at small scale, and the query interface is
  familiar SQL.
- **Local embeddings (sentence-transformers)**: removes a third-party
  dependency from the hot path, no per-token cost, predictable latency. The
  trade-off is slightly lower retrieval quality vs OpenAI ada-002. Acceptable
  for this scope.
- **Claude as the generator**: large context window (200K tokens) makes RAG
  prompt assembly more forgiving, and the streaming API is straightforward.
- **FastAPI + Uvicorn**: async-friendly, matches existing microservice
  patterns, integrates with existing OpenAPI tooling.

## Setup

Prereqs: Docker, Docker Compose, an Anthropic API key.

```bash
cp .env.example .env
# Edit .env and paste your ANTHROPIC_API_KEY

docker compose up --build
```

The service comes up on http://localhost:8000. Swagger UI at /docs.

Health check:
```bash
curl http://localhost:8000/health
```

## Usage

Ingest a document:
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "my-notes", "content": "Your text here..."}'
```

Ask a question:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say about X?"}'
```

## Project layout

```
app/
  api/        FastAPI routes
  core/       config, logging, db connection
  services/   business logic (chunking, embedding, retrieval, generation)
  models/     Pydantic request/response models
tests/        pytest tests with mocked LLM calls
scripts/      DB init, schema migration
```

## Limitations and future work

- No authentication or rate limiting. Add before exposing publicly.
- Chunking is naive fixed-size; semantic chunking would improve retrieval.
- No reranking step; top-k cosine is a baseline. Cross-encoder reranking
  would lift answer quality on ambiguous queries.
- Evaluation is manual. A proper eval harness (golden Q/A set, retrieval
  recall@k, generation faithfulness scoring) is needed before any claim
  of production-readiness.
- No streaming on the /query endpoint yet (TODO in app/api/query.py).

## Running tests

```bash
docker compose run --rm api pytest -v
```

## Learning checkpoints

This repo intentionally leaves some core logic as `TODO` so you write it
yourself. Each one teaches something specific.

| File | Concept |
|------|---------|
| `app/services/chunking.py` | Token-aware splitting, overlap, why chunk size matters |
| `app/services/embedding.py` | Embedding model loading, batching for throughput |
| `app/services/retrieval.py` | Cosine similarity in SQL, top-k tradeoffs |
| `app/services/generation.py` | Prompt assembly, context window budgeting, source attribution |
| `app/api/query.py` | Streaming response with SSE |

Work through them in this order. After each, write a paragraph in a
NOTES.md explaining what you learned and what you would do differently.
That paragraph is what you talk about in an interview.
