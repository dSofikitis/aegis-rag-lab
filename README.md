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

## Ingest documents
CLI:
`aegis ingest --path data/seed --recursive`

API:
`curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"source": "seed", "content": "Zero trust focuses on..."}]}'`

## Query
`curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is least privilege?"}'`

## Evaluation harness
`aegis eval --dataset data/eval/sample_eval.jsonl --no-llm`

## Configuration
Key environment variables (see .env.example for full list):
- `AEGIS_OPENAI_API_KEY`
- `AEGIS_OPENAI_MODEL`
- `AEGIS_EMBEDDING_MODEL`
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
