# Aegis RAG Lab

Agentic RAG platform for security-focused question answering, evals, and guardrails.

## Highlights
- FastAPI service with ingestion, query, and stats endpoints.
- LangGraph orchestration with prompt injection guardrails.
- Postgres + pgvector vector store with Docker Compose setup.
- Evaluation harness for retrieval and keyword grounding.
- CLI for ingestion and evaluation.

## Architecture
- API: FastAPI
- Orchestrator: LangGraph
- Vector store: Postgres + pgvector
- Cache: Redis (reserved for future use)
- Observability: structlog (OpenTelemetry planned)

## Quickstart (local)
1. Create a venv and install deps:
   `python -m venv .venv`
   `pip install -e .[dev]`
2. Run the API:
   `uvicorn aegis_rag_lab.main:app --reload`

## Quickstart (Docker)
1. `docker compose up --build`
2. Open `http://localhost:8000/ui`

## Local-only profile (Gemma 4 via Ollama)
Run a fully local stack with no external API keys. Uses `gemma4:e2b` for chat
and `nomic-embed-text` (768-dim) for embeddings.

1. In `.env` set:
   ```
   AEGIS_LLM_PROVIDER=ollama
   AEGIS_EMBEDDINGS_PROVIDER=ollama
   AEGIS_EMBEDDING_DIM=768
   ```
2. Bring up the stack with the `ollama` profile:
   `docker compose --profile ollama up -d --build`
3. First boot pulls ~7.5 GB of model weights — watch progress with
   `docker compose logs -f ollama-init`. Queries will fail until that container
   exits successfully.
4. If you switch embedders later, the dimensionality changes and existing
   pgvector rows become unusable. Wipe the DB with
   `docker compose down -v` and re-ingest.

GPU users: add an `nvidia` deploy stanza to the `ollama` service in
`docker-compose.yml` (see Ollama docs).

## UI (React)
1. `cd ui`
2. `npm install`
3. `copy .env.example .env` and set `VITE_API_URL` if needed
4. `npm run dev`
5. Open `http://localhost:5173`

## Ingest documents
CLI:
`aegis ingest --path data/seed --recursive`

API:
`curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"source": "seed", "content": "Zero trust focuses on..."}]}'`

File upload:
`curl -X POST http://localhost:8000/ingest/files \
  -F "files=@data/seed/security_notes.md"`

## Query
`curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is least privilege?"}'`

## Evaluation harness
`aegis eval --dataset data/eval/sample_eval.jsonl --no-llm`

## Configuration
Key environment variables (see .env.example for full list):
- `AEGIS_LLM_PROVIDER` (openai | ollama | stub)
- `AEGIS_EMBEDDINGS_PROVIDER` (openai | ollama | deterministic)
- `AEGIS_OPENAI_API_KEY`, `AEGIS_OPENAI_MODEL`, `AEGIS_EMBEDDING_MODEL`
- `AEGIS_OLLAMA_BASE_URL`, `AEGIS_OLLAMA_MODEL`, `AEGIS_OLLAMA_EMBEDDING_MODEL`
- `AEGIS_VECTOR_BACKEND` (postgres or memory)
- `AEGIS_GUARDRAILS_ENABLED`

## Roadmap
- [x] ingestion pipeline + chunking
- [x] LangGraph orchestration
- [x] eval harness
- [x] guardrails + prompt injection checks
- [ ] cloud deploy (Azure/GCP)

## License
MIT. Copyright (c) 2026 @dSofikitis.
