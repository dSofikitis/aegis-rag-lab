from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from aegis_rag_lab.rag.chunking import chunk_text
from aegis_rag_lab.rag.embeddings import Embedder
from aegis_rag_lab.rag.models import DocumentChunk, DocumentInput
from aegis_rag_lab.rag.vector_store import VectorStore


@dataclass
class IngestResult:
    documents: int
    chunks: int


def load_documents_from_path(
    path: Path,
    recursive: bool,
    extensions: Iterable[str] | None = None,
) -> list[DocumentInput]:
    exts = {ext.lower() for ext in (extensions or [".md", ".txt", ".jsonl"])}
    if path.is_file():
        files = [path]
    else:
        pattern = "**/*" if recursive else "*"
        files = [file for file in path.glob(pattern) if file.is_file()]

    documents: list[DocumentInput] = []
    for file in files:
        if file.suffix.lower() not in exts:
            continue
        if file.suffix.lower() == ".jsonl":
            content = file.read_text(encoding="utf-8")
            documents.extend(_load_jsonl_text(content, source_prefix=file.as_posix()))
            continue
        content = file.read_text(encoding="utf-8")
        documents.append(
            DocumentInput(
                source=file.as_posix(),
                content=content,
                metadata={"source_path": file.as_posix()},
            )
        )
    return documents


def load_documents_from_bytes(filename: str, content: bytes) -> list[DocumentInput]:
    suffix = Path(filename).suffix.lower()
    text = content.decode("utf-8", errors="ignore")
    if suffix == ".jsonl":
        return _load_jsonl_text(text, source_prefix=filename)
    if not text.strip():
        return []
    return [
        DocumentInput(
            source=filename,
            content=text,
            metadata={"source_path": filename},
        )
    ]


def ingest_documents(
    documents: list[DocumentInput],
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestResult:
    chunks: list[DocumentChunk] = []
    for document in documents:
        parts = chunk_text(document.content, chunk_size, chunk_overlap)
        for index, part in enumerate(parts):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = index
            chunks.append(
                DocumentChunk(
                    id=str(uuid4()),
                    source=document.source,
                    content=part,
                    metadata=metadata,
                )
            )

    if not chunks:
        return IngestResult(documents=len(documents), chunks=0)

    embeddings = embedder.embed_documents([chunk.content for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    stored = store.add_documents(chunks)
    return IngestResult(documents=len(documents), chunks=stored)


def _load_jsonl_text(raw_text: str, source_prefix: str) -> list[DocumentInput]:
    documents: list[DocumentInput] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        content = record.get("content", "")
        source = record.get("source") or source_prefix
        metadata = record.get("metadata", {})
        if not content:
            continue
        documents.append(
            DocumentInput(
                source=source,
                content=content,
                metadata=metadata,
            )
        )
    return documents
