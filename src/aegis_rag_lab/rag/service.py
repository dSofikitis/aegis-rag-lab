from __future__ import annotations

import time
from dataclasses import asdict

from aegis_rag_lab.config import Settings
from aegis_rag_lab.logging import get_logger
from aegis_rag_lab.rag.embeddings import build_embedder
from aegis_rag_lab.rag.graph import build_graph
from aegis_rag_lab.rag.guardrails import evaluate_prompt_safety
from aegis_rag_lab.rag.ingestion import ingest_documents
from aegis_rag_lab.rag.llm import NO_CONTEXT_ANSWER, build_llm
from aegis_rag_lab.rag.models import DocumentChunk, DocumentInput
from aegis_rag_lab.rag.vector_store import build_vector_store


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 1)


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
        total_start = time.perf_counter()
        state = self._graph.invoke({"question": question})
        total_ms = _ms_since(total_start)
        timings = {
            "embed_ms": state.get("embed_ms"),
            "search_ms": state.get("search_ms"),
            "llm_ms": state.get("llm_ms"),
            "total_ms": total_ms,
        }
        if state.get("blocked"):
            return {
                "answer": "Request blocked by guardrails.",
                "citations": [],
                "blocked": True,
                "reason": state.get("reason"),
                "timings": timings,
            }
        return {
            "answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "blocked": False,
            "reason": None,
            "timings": timings,
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
        embed_start = time.perf_counter()
        query_embedding = self._embedder.embed_query(question)
        embed_ms = _ms_since(embed_start)

        search_start = time.perf_counter()
        scored = self._store.similarity_search(
            query_embedding,
            self._settings.retrieval_k,
            self._settings.retrieval_min_similarity,
        )
        search_ms = _ms_since(search_start)

        self._logger.info(
            "retrieve_complete",
            k=self._settings.retrieval_k,
            min_similarity=self._settings.retrieval_min_similarity,
            hits=len(scored),
            scores=[round(score, 3) for score, _ in scored],
            citations=[doc.citation() for _, doc in scored],
            embed_ms=embed_ms,
            search_ms=search_ms,
        )
        return {"retrieved": scored, "embed_ms": embed_ms, "search_ms": search_ms}

    def generate_node(self, state: dict) -> dict:
        question = state.get("question", "")
        scored: list[tuple[float, DocumentChunk]] = state.get("retrieved", [])
        citations = [
            {
                "source": doc.citation(),
                "content": doc.content,
                "score": round(score, 3),
            }
            for score, doc in scored
        ]
        if not scored:
            return {"answer": NO_CONTEXT_ANSWER, "citations": citations, "llm_ms": 0.0}
        context = self._build_context(scored)
        llm_start = time.perf_counter()
        answer = self._llm.generate(question, context)
        llm_ms = _ms_since(llm_start)
        return {"answer": answer, "citations": citations, "llm_ms": llm_ms}

    def _build_context(self, scored: list[tuple[float, DocumentChunk]]) -> str:
        if not scored:
            return ""
        joined = "\n\n".join(
            f"Source: {doc.citation()}\n{doc.content}" for _, doc in scored
        )
        return joined[: self._settings.max_context_chars]
