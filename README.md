# Aegis RAG Lab

> **Aegis** — from the Greek *aigís* (αἰγίς), the shield of Zeus and Athena.
> Here it stands for the guardrail layer that protects retrieval-augmented
> answers: prompt-injection checks, grounded prompting, and an empty-context
> refusal contract.

Agentic, security-focused RAG platform with guardrails, evals, and a fully
local-only profile (no API keys required).

## Highlights
- FastAPI service: `/query`, **streaming `/query/stream` (SSE)**, `/ingest`,
  `/ingest/files`, `/sources`, `/stats`, `/health`.
- LangGraph orchestration with prompt-injection guardrails and an adaptive
  system prompt (greeting / topic / question modes).
- **MultiQuery decomposition** for multi-hop and list-style questions.
- **HNSW vector index** + **cross-encoder reranker**
  (`ms-marco-MiniLM-L-6-v2`) on top of bi-encoder retrieval.
- File ingestion for **`.md` / `.txt` / `.jsonl` / `.pdf` / `.docx` /
  `.html`**.
- Per-stage observability: `decompose_ms / embed_ms / search_ms /
  rerank_ms / llm_ms / total_ms` returned in the API and rendered as
  color-graded chips in the UI.
- Vercel-style React + TypeScript UI bundled with the API image; sources
  tree browser with JSON export and clear-data control.
- Local-only profile via Ollama sidecar (default `qwen2.5:3b` +
  `nomic-embed-text`); NVIDIA GPU stanza available on the `nvidia`
  branch.

## Architecture
- API: FastAPI (sync + SSE streaming).
- Orchestrator: LangGraph (guardrails → retrieve → generate).
- Retrieval: bi-encoder embed → pgvector HNSW top-N → optional
  cross-encoder rerank → similarity threshold → top-k.
- Query rewriting: optional LLM-based MultiQuery decomposition.
- Vector store: Postgres + `pgvector` (HNSW, cosine ops).
- LLM / embeddings: pluggable — `openai`, `ollama`, `stub`/`deterministic`.
- Cache: Redis (reserved).
- Observability: structlog with per-stage timings and structured event
  logs (`retrieve_complete`, `ingest_complete`, etc.).

## Quickstart (Docker, local-only profile — recommended)
Default brings up Postgres, Redis, the API, and an Ollama sidecar that
pulls `qwen2.5:3b` + `nomic-embed-text` on first boot:

```bash
cp .env.example .env
docker compose --profile ollama up -d --build
docker compose logs -f ollama-init   # watch model pull (~2.25 GB on first boot)
```

Open http://localhost:8000/ui (UI) or http://localhost:8000/docs (Swagger).

## Quickstart (Docker, OpenAI)
Set `AEGIS_OPENAI_API_KEY` in `.env`, leave `AEGIS_LLM_PROVIDER=openai` /
`AEGIS_EMBEDDINGS_PROVIDER=openai`, then:

```bash
docker compose up -d --build
```

## Quickstart (local Python)
```bash
python -m venv .venv
.venv\Scripts\activate                 # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e .[dev]
uvicorn aegis_rag_lab.main:app --reload
```

`AEGIS_VECTOR_BACKEND=memory` is the simplest option for a local run
without Docker; for pgvector point `AEGIS_DATABASE_URL` at a running
Postgres with the `vector` extension.

## NVIDIA GPU
The `nvidia` branch ships a `docker-compose.yml` with the
`deploy.resources.reservations.devices` stanza wired up for the Ollama
service. Requires the NVIDIA Container Toolkit on the host.

```bash
git checkout nvidia
docker compose --profile ollama up -d --build
docker compose exec ollama nvidia-smi   # verify the GPU is visible
```

## Picking a chat model
Override `AEGIS_OLLAMA_MODEL` for stronger answers at higher
VRAM/latency cost:

| Model | Size (q4) | Notes |
|---|---|---|
| `qwen2.5:3b` | ~2 GB | **Default.** Strong instruction-following, handles negation. |
| `gemma3:1b` | ~815 MB | Fastest, but weak on negation and multi-clue questions. |
| `qwen2.5:1.5b` | ~1 GB | Lighter Qwen with most of the grounding quality. |
| `llama3.2:3b` | ~2 GB | Comparable size, good general reasoning. |
| `phi3.5:3.8b` | ~2.2 GB | Excellent quality / size ratio. |
| `gemma4:e2b` | ~7.2 GB | Multimodal (text/image/audio), 128K ctx. |

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe. |
| GET | `/stats` | Source / chunk counts. |
| GET | `/sources` | Corpus grouped by source with chunk content. |
| DELETE | `/sources` | Wipe all chunks. Returns `{ "removed": N }`. |
| POST | `/ingest` | Ingest JSON-described documents. |
| POST | `/ingest/files` | Multipart upload (md/txt/jsonl/pdf/docx/html). |
| POST | `/query` | Buffered query, returns full answer + citations + timings. |
| POST | `/query/stream` | SSE: `meta` (citations + retrieval timings) → `token` deltas → `done` (full timings). |

Inline ingest:
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"source":"seed","content":"Zero trust focuses on..."}]}'
```

File upload (PDF / DOCX / HTML / MD / TXT / JSONL):
```bash
curl -X POST http://localhost:8000/ingest/files \
  -F "files=@data/seed/security_notes.md" \
  -F "files=@papers/zero-trust.pdf"
```

Streaming query (SSE):
```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"What is least privilege?"}'
```

## CLI
```bash
aegis ingest --path data/seed --recursive
aegis eval   --dataset data/eval/sample_eval.jsonl --no-llm
```

## Configuration
Selected env vars (see `.env.example` for the full list):

- Provider selection: `AEGIS_LLM_PROVIDER` (`openai` | `ollama` | `stub`),
  `AEGIS_EMBEDDINGS_PROVIDER` (`openai` | `ollama` | `deterministic`).
- OpenAI: `AEGIS_OPENAI_API_KEY`, `AEGIS_OPENAI_MODEL`,
  `AEGIS_EMBEDDING_MODEL`.
- Ollama: `AEGIS_OLLAMA_BASE_URL`, `AEGIS_OLLAMA_MODEL`,
  `AEGIS_OLLAMA_EMBEDDING_MODEL`, `AEGIS_OLLAMA_REQUEST_TIMEOUT_S`.
- Retrieval: `AEGIS_RETRIEVAL_K`, `AEGIS_RETRIEVAL_MIN_SIMILARITY`.
- Reranker: `AEGIS_RERANK_ENABLED`, `AEGIS_RERANK_MODEL`,
  `AEGIS_RERANK_CANDIDATES`.
- Decomposer: `AEGIS_DECOMPOSE_ENABLED`,
  `AEGIS_DECOMPOSE_MAX_SUBQUERIES`,
  `AEGIS_DECOMPOSE_MIN_QUESTION_LENGTH`.
- Storage: `AEGIS_VECTOR_BACKEND` (`postgres` | `memory`),
  `AEGIS_DATABASE_URL`.
- Safety: `AEGIS_GUARDRAILS_ENABLED`.

## Development
```bash
pip install -e .[dev]
ruff check .
pytest
```

The CI workflow (`.github/workflows/ci.yml`) runs lint and tests on
push to `main` and on pull requests.

## License
MIT — © 2026 [Dimitris Sofikitis](https://dimitrisofikitis.com). See [LICENSE](LICENSE).
