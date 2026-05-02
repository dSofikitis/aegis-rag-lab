from __future__ import annotations

import time
from dataclasses import asdict
from typing import Iterator

from aegis_rag_lab.config import Settings
from aegis_rag_lab.logging import get_logger
from aegis_rag_lab.rag.decompose import build_decomposer
from aegis_rag_lab.rag.embeddings import build_embedder
from aegis_rag_lab.rag.graph import build_graph
from aegis_rag_lab.rag.guardrails import evaluate_prompt_safety
from aegis_rag_lab.rag.ingestion import ingest_documents
from aegis_rag_lab.rag.llm import build_llm
from aegis_rag_lab.rag.models import DocumentChunk, DocumentInput
from aegis_rag_lab.rag.reranker import build_reranker
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
        self._reranker = build_reranker(settings)
        self._decomposer = build_decomposer(settings, self._llm)
        self._graph = build_graph(self)

    def _retrieve(
        self, question: str
    ) -> tuple[list[tuple[float, DocumentChunk]], dict[str, float]]:
        decompose_ms = 0.0
        sub_queries: list[str] = [question]
        if self._decomposer:
            decompose_start = time.perf_counter()
            sub_queries = self._decomposer.decompose(question)
            decompose_ms = _ms_since(decompose_start)

        embed_start = time.perf_counter()
        embeddings = self._embedder.embed_documents(sub_queries)
        embed_ms = _ms_since(embed_start)

        candidate_k = (
            max(self._settings.retrieval_k, self._settings.rerank_candidates)
            if self._reranker
            else self._settings.retrieval_k
        )
        search_start = time.perf_counter()
        seen_ids: set[str] = set()
        candidates: list[tuple[float, DocumentChunk]] = []
        for emb in embeddings:
            for score, doc in self._store.similarity_search(emb, candidate_k, 0.0):
                if doc.id in seen_ids:
                    continue
                seen_ids.add(doc.id)
                candidates.append((score, doc))
        search_ms = _ms_since(search_start)

        rerank_ms = 0.0
        if self._reranker and candidates:
            rerank_start = time.perf_counter()
            new_scores = self._reranker.rerank(
                question, [doc.content for _, doc in candidates]
            )
            candidates = [(score, doc) for score, (_, doc) in zip(new_scores, candidates)]
            candidates.sort(key=lambda item: item[0], reverse=True)
            rerank_ms = _ms_since(rerank_start)

        threshold = self._settings.retrieval_min_similarity
        scored = [item for item in candidates if item[0] >= threshold][
            : self._settings.retrieval_k
        ]

        timings = {"embed_ms": embed_ms, "search_ms": search_ms}
        if self._decomposer:
            timings["decompose_ms"] = decompose_ms
        if self._reranker:
            timings["rerank_ms"] = rerank_ms

        self._logger.info(
            "retrieve_complete",
            k=self._settings.retrieval_k,
            min_similarity=self._settings.retrieval_min_similarity,
            reranked=self._reranker is not None,
            sub_queries=sub_queries,
            hits=len(scored),
            scores=[round(score, 3) for score, _ in scored],
            citations=[doc.citation() for _, doc in scored],
            **{k: v for k, v in timings.items()},
        )
        return scored, timings

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
            "decompose_ms": state.get("decompose_ms"),
            "embed_ms": state.get("embed_ms"),
            "search_ms": state.get("search_ms"),
            "rerank_ms": state.get("rerank_ms"),
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

    def list_sources(self) -> list[dict]:
        documents = self._store.list_documents()
        grouped: dict[str, list[dict]] = {}
        for doc in documents:
            grouped.setdefault(doc.source, []).append(
                {
                    "id": doc.id,
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "format": doc.metadata.get("format"),
                    "content": doc.content,
                }
            )
        return [
            {"source": source, "chunks": chunks}
            for source, chunks in sorted(grouped.items())
        ]

    def query_stream(self, question: str) -> Iterator[dict]:
        total_start = time.perf_counter()

        if self._settings.guardrails_enabled:
            result = evaluate_prompt_safety(question)
            if not result.allowed:
                yield {
                    "type": "blocked",
                    "reason": result.reason,
                    "timings": {"total_ms": _ms_since(total_start)},
                }
                return

        scored, timings = self._retrieve(question)
        citations = [
            {"source": doc.citation(), "content": doc.content, "score": round(score, 3)}
            for score, doc in scored
        ]

        yield {"type": "meta", "citations": citations, **timings}

        context = self._build_context(scored)
        llm_start = time.perf_counter()
        for token in self._llm.generate_stream(question, context):
            yield {"type": "token", "value": token}
        llm_ms = _ms_since(llm_start)

        yield {
            "type": "done",
            "timings": {**timings, "llm_ms": llm_ms, "total_ms": _ms_since(total_start)},
        }

    def guardrails_node(self, state: dict) -> dict:
        if not self._settings.guardrails_enabled:
            return {"blocked": False, "reason": "disabled"}
        result = evaluate_prompt_safety(state.get("question", ""))
        return {"blocked": not result.allowed, "reason": result.reason}

    def retrieve_node(self, state: dict) -> dict:
        question = state.get("question", "")
        scored, timings = self._retrieve(question)
        return {"retrieved": scored, **timings}

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
