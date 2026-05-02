from __future__ import annotations

from aegis_rag_lab.config import Settings
from aegis_rag_lab.rag.llm import LLMClient


_SYSTEM = (
    "You rewrite a user's question into atomic sub-questions that can be "
    "answered independently. Output ONE self-contained sub-question per "
    "line, no numbering, no commentary, no preamble. Output AT MOST {max} "
    "lines. If the original is already a single atomic question (or a "
    "simple topic / keyword), output ONLY the original on a single line."
)


class QueryDecomposer:
    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        self._settings = settings
        self._llm = llm

    def decompose(self, question: str) -> list[str]:
        if len(question) < self._settings.decompose_min_question_length:
            return [question]

        try:
            response = self._llm.complete(
                system=_SYSTEM.format(max=self._settings.decompose_max_subqueries),
                user=question,
                max_tokens=200,
            )
        except Exception:
            return [question]

        sub_queries: list[str] = []
        seen: set[str] = set()
        for raw in (response or "").splitlines():
            line = raw.strip().lstrip("-*0123456789.) ").strip()
            if not line:
                continue
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            sub_queries.append(line)

        if not sub_queries:
            return [question]

        if question.lower() not in seen:
            sub_queries.insert(0, question)

        return sub_queries[: self._settings.decompose_max_subqueries + 1]


def build_decomposer(settings: Settings, llm: LLMClient) -> QueryDecomposer | None:
    if not settings.decompose_enabled:
        return None
    return QueryDecomposer(settings, llm)
