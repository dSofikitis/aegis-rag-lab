from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Protocol

from aegis_rag_lab.config import Settings
from aegis_rag_lab.rag.models import DocumentChunk


class VectorStore(Protocol):
    def ensure_schema(self) -> None:
        ...

    def add_documents(self, documents: list[DocumentChunk]) -> int:
        ...

    def similarity_search(self, embedding: list[float], k: int) -> list[DocumentChunk]:
        ...

    def stats(self) -> dict[str, int]:
        ...


@dataclass
class InMemoryVectorStore:
    _documents: list[DocumentChunk]

    def __init__(self) -> None:
        self._documents = []

    def ensure_schema(self) -> None:
        return None

    def add_documents(self, documents: list[DocumentChunk]) -> int:
        self._documents.extend(documents)
        return len(documents)

    def similarity_search(self, embedding: list[float], k: int) -> list[DocumentChunk]:
        scored = [
            (self._cosine_similarity(embedding, doc.embedding or []), doc)
            for doc in self._documents
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def stats(self) -> dict[str, int]:
        sources = {doc.source for doc in self._documents}
        return {"sources": len(sources), "chunks": len(self._documents)}

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return dot / (left_norm * right_norm)


class PostgresVectorStore:
    def __init__(self, settings: Settings) -> None:
        self._database_url = self._normalize_url(settings.database_url)
        self._embedding_dim = settings.embedding_dim

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id uuid PRIMARY KEY,
                    source text NOT NULL,
                    content text NOT NULL,
                    metadata jsonb NOT NULL,
                    embedding vector({self._embedding_dim}) NOT NULL,
                    created_at timestamptz DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_embedding
                ON documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                """
            )

    def add_documents(self, documents: list[DocumentChunk]) -> int:
        if not documents:
            return 0
        with self._connect() as conn:
            rows = [
                (
                    doc.id,
                    doc.source,
                    doc.content,
                    json.dumps(doc.metadata),
                    doc.embedding,
                )
                for doc in documents
            ]
            conn.executemany(
                """
                INSERT INTO documents (id, source, content, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                rows,
            )
        return len(documents)

    def similarity_search(self, embedding: list[float], k: int) -> list[DocumentChunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, source, content, metadata
                FROM documents
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (embedding, k),
            ).fetchall()
        return [
            DocumentChunk(
                id=str(row[0]),
                source=row[1],
                content=row[2],
                metadata=_normalize_metadata(row[3]),
            )
            for row in rows
        ]

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*)::int, COUNT(DISTINCT source)::int FROM documents"
            ).fetchone()
        if row is None:
            return {"sources": 0, "chunks": 0}
        return {"chunks": row[0], "sources": row[1]}

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self._database_url)
        register_vector(conn)
        return conn

    @staticmethod
    def _normalize_url(database_url: str) -> str:
        prefix = "postgresql+psycopg://"
        if database_url.startswith(prefix):
            return database_url.replace(prefix, "postgresql://", 1)
        return database_url


def build_vector_store(settings: Settings) -> VectorStore:
    if settings.vector_backend.lower() == "memory":
        return InMemoryVectorStore()
    return PostgresVectorStore(settings)


def _normalize_metadata(raw_value) -> dict:
    if raw_value is None:
        return {}
    if isinstance(raw_value, str):
        return json.loads(raw_value)
    return raw_value
