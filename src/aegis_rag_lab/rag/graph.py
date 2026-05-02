from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from aegis_rag_lab.rag.models import DocumentChunk


class RagState(TypedDict, total=False):
    question: str
    retrieved: list[DocumentChunk]
    answer: str
    citations: list[dict]
    blocked: bool
    reason: str
    decompose_ms: float
    embed_ms: float
    search_ms: float
    rerank_ms: float
    llm_ms: float


def build_graph(service) -> object:
    graph = StateGraph(RagState)
    graph.add_node("guardrails", service.guardrails_node)
    graph.add_node("retrieve", service.retrieve_node)
    graph.add_node("generate", service.generate_node)

    graph.set_entry_point("guardrails")
    graph.add_conditional_edges(
        "guardrails",
        lambda state: "blocked" if state.get("blocked") else "retrieve",
        {"blocked": END, "retrieve": "retrieve"},
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
