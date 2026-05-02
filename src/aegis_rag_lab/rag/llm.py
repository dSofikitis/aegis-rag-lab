from __future__ import annotations

from typing import Protocol

from aegis_rag_lab.config import Settings
from aegis_rag_lab.logging import get_logger


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
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a security-focused assistant. "
                        "Answer with concise, cited responses."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Context:\n"
                        f"{context}\n\n"
                        "Question:\n"
                        f"{question}\n\n"
                        "Provide a short answer with citations."
                    ),
                },
            ],
            temperature=0.2,
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
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a security-focused assistant. "
                        "Answer with concise, cited responses."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Context:\n"
                        f"{context}\n\n"
                        "Question:\n"
                        f"{question}\n\n"
                        "Provide a short answer with citations."
                    ),
                },
            ],
            temperature=0.2,
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
