from __future__ import annotations

import hashlib
from typing import Protocol

from aegis_rag_lab.config import Settings


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._timeout = settings.request_timeout_s

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
            timeout=self._timeout,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class DeterministicEmbedder:
    def __init__(self, settings: Settings) -> None:
        self._dim = settings.embedding_dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_to_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def _hash_to_vector(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        counter = 0
        while len(values) < self._dim:
            digest = hashlib.sha256(seed + counter.to_bytes(2, "little")).digest()
            values.extend([(byte / 255.0) * 2.0 - 1.0 for byte in digest])
            counter += 1
        return values[: self._dim]


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embeddings_provider.lower()
    if provider == "deterministic":
        return DeterministicEmbedder(settings)
    return OpenAIEmbedder(settings)
