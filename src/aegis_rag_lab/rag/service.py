from __future__ import annotations

from dataclasses import asdict

from aegis_rag_lab.config import Settings
from aegis_rag_lab.logging import get_logger
from aegis_rag_lab.rag.embeddings import build_embedder
from aegis_rag_lab.rag.graph import build_graph
from aegis_rag_lab.rag.guardrails import evaluate_prompt_safety
from aegis_rag_lab.rag.ingestion import ingest_documents
from aegis_rag_lab.rag.llm import build_llm
from aegis_rag_lab.rag.models import DocumentChunk, DocumentInput
from aegis_rag_lab.rag.retrieval import retrieve_documents
from aegis_rag_lab.rag.vector_store import build_vector_store


class RagService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger()
        self._embedder = build_embedder(settings)
        self._store = build_vector_store(settings)
        self._llm = build_llm(settings)
        self._graph = build_graph(self)

    def ensure_ready(self) -> None:
        self._store.ensure_schema()

    def ingest_documents(self, documents: list[DocumentInput]) -> dict[str, int]:
        result = ingest_documents(
            documents,
            self._embedder,
            self._store,
            self._settings.chunk_size,
            self._settings.chunk_overlap,
        )
        self._logger.info("ingest_complete", **asdict(result))
        return asdict(result)

    def query(self, question: str) -> dict[str, object]:
        state = self._graph.invoke({"question": question})
        if state.get("blocked"):
            return {
                "answer": "Request blocked by guardrails.",
                "citations": [],
                "blocked": True,
                "reason": state.get("reason"),
            }
        return {
            "answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "blocked": False,
            "reason": None,
        }

    def stats(self) -> dict[str, int]:
        return self._store.stats()

    def guardrails_node(self, state: dict) -> dict:
        if not self._settings.guardrails_enabled:
            return {"blocked": False, "reason": "disabled"}
        result = evaluate_prompt_safety(state.get("question", ""))
        return {"blocked": not result.allowed, "reason": result.reason}

    def retrieve_node(self, state: dict) -> dict:
        question = state.get("question", "")
        retrieved = retrieve_documents(
            question,
            self._embedder,
            self._store,
            self._settings.retrieval_k,
        )
        return {"retrieved": retrieved}

    def generate_node(self, state: dict) -> dict:
        question = state.get("question", "")
        retrieved: list[DocumentChunk] = state.get("retrieved", [])
        context = self._build_context(retrieved)
        answer = self._llm.generate(question, context)
        citations = [doc.citation() for doc in retrieved]
        return {"answer": answer, "citations": citations}

    def _build_context(self, retrieved: list[DocumentChunk]) -> str:
        if not retrieved:
            return ""
        joined = "\n\n".join(f"Source: {doc.citation()}\n{doc.content}" for doc in retrieved)
        return joined[: self._settings.max_context_chars]
