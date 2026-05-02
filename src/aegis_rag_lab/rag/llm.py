from __future__ import annotations

from typing import Iterator, Protocol

from aegis_rag_lab.config import Settings
from aegis_rag_lab.logging import get_logger


NO_CONTEXT_ANSWER = "I cannot answer that based on the provided context."

SYSTEM_PROMPT = (
    "You are a retrieval-augmented assistant. Adapt your response to the "
    "user's input style:\n\n"
    "1. Greeting / thanks / small-talk (e.g. 'hi', 'thanks'): respond "
    "briefly and politely, ignoring the Context.\n\n"
    "2. Topic / keyword (a noun phrase, not a full question): present the "
    "facts from the Context that concern that topic, in plain prose. If "
    "the Context contains nothing about the topic, reply exactly: "
    f'"{NO_CONTEXT_ANSWER}"\n\n'
    "3. Substantive question: answer using ONLY the facts in the Context. "
    "Treat the Context as ground truth even when it contradicts your "
    "prior knowledge. Do not invent facts. If the Context is empty or "
    "does not contain the answer, reply exactly: "
    f'"{NO_CONTEXT_ANSWER}"\n\n'
    "Never write a 'Sources:' line; the system surfaces citations "
    "separately."
)


def _build_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context if context else '(empty)'}\n\n"
                f"User input: {question}"
            ),
        },
    ]


def _complete_chat(client, model: str, system: str, user: str, max_tokens: int, timeout: float) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return response.choices[0].message.content or ""


class LLMClient(Protocol):
    def generate(self, question: str, context: str) -> str: ...

    def generate_stream(self, question: str, context: str) -> Iterator[str]: ...

    def complete(self, system: str, user: str, max_tokens: int = 200) -> str: ...


def _stream_openai_chat(client, model: str, question: str, context: str, timeout: float) -> Iterator[str]:
    stream = client.chat.completions.create(
        model=model,
        messages=_build_messages(question, context),
        temperature=0.1,
        top_p=0.9,
        max_tokens=600,
        timeout=timeout,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


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

    def generate_stream(self, question: str, context: str) -> Iterator[str]:
        return _stream_openai_chat(self._client, self._model, question, context, self._timeout)

    def complete(self, system: str, user: str, max_tokens: int = 200) -> str:
        return _complete_chat(self._client, self._model, system, user, max_tokens, self._timeout)


class StubLLM:
    def generate(self, question: str, context: str) -> str:
        if not context:
            return "No context available to answer the question."
        snippet = context[:400].strip()
        return f"Based on the available context: {snippet}"

    def generate_stream(self, question: str, context: str) -> Iterator[str]:
        yield self.generate(question, context)

    def complete(self, system: str, user: str, max_tokens: int = 200) -> str:
        return ""


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

    def generate_stream(self, question: str, context: str) -> Iterator[str]:
        return _stream_openai_chat(self._client, self._model, question, context, self._timeout)

    def complete(self, system: str, user: str, max_tokens: int = 200) -> str:
        return _complete_chat(self._client, self._model, system, user, max_tokens, self._timeout)


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
