from __future__ import annotations

from typing import Protocol

from aegis_rag_lab.config import Settings
from aegis_rag_lab.logging import get_logger


NO_CONTEXT_ANSWER = "I cannot answer that based on the provided context."

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. You must answer the user's "
    "question using ONLY the facts stated in the Context block. Treat the "
    "Context as ground truth even when it contradicts your prior knowledge. "
    "Never use prior knowledge or invent facts. If the Context is empty or "
    f'does not contain the answer, reply exactly: "{NO_CONTEXT_ANSWER}" '
    "and nothing else."
)


def _build_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Use only the Context below to answer the Question. After your "
                "answer, on a new line, write 'Sources:' followed by a "
                "comma-separated list of the source identifiers from the "
                "Context that you actually used.\n\n"
                f"Context:\n{context if context else '(empty)'}\n\n"
                f"Question: {question}"
            ),
        },
    ]


class LLMClient(Protocol):
    def generate(self, question: str, context: str) -> str: ...


class OpenAIChatLLM:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._timeout = settings.request_timeout_s

    def generate(self, question: str, context: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=_build_messages(question, context),
            temperature=0.1,
            top_p=0.9,
            max_tokens=600,
            timeout=self._timeout,
        )
        return response.choices[0].message.content or ""


class StubLLM:
    def generate(self, question: str, context: str) -> str:
        if not context:
            return "No context available to answer the question."
        snippet = context[:400].strip()
        return f"Based on the available context: {snippet}"


class OllamaChatLLM:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            api_key="ollama",
            base_url=f"{settings.ollama_base_url.rstrip('/')}/v1",
        )
        self._model = settings.ollama_model
        self._timeout = settings.ollama_request_timeout_s

    def generate(self, question: str, context: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=_build_messages(question, context),
            temperature=0.1,
            top_p=0.9,
            max_tokens=600,
            timeout=self._timeout,
        )
        return response.choices[0].message.content or ""


def build_llm(settings: Settings) -> LLMClient:
    provider = settings.llm_provider.lower()
    if provider == "stub":
        return StubLLM()
    if provider in ("ollama", "gemma"):
        return OllamaChatLLM(settings)
    if not settings.openai_api_key:
        logger = get_logger()
        logger.warning("openai_api_key_missing", component="llm", fallback="stub")
        return StubLLM()
    return OpenAIChatLLM(settings)
