from __future__ import annotations

import math
from typing import Protocol

from aegis_rag_lab.config import Settings


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, settings: Settings) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(settings.rerank_model)

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [[query, content] for content in candidates]
        logits = self._model.predict(pairs)
        return [_sigmoid(float(score)) for score in logits]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def build_reranker(settings: Settings) -> Reranker | None:
    if not settings.rerank_enabled:
        return None
    return CrossEncoderReranker(settings)
