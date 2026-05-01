from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aegis_rag_lab.rag.models import DocumentChunk
from aegis_rag_lab.rag.service import RagService


@dataclass
class EvalCase:
    question: str
    expected_keywords: list[str]
    expected_source: str | None = None


def load_eval_cases(dataset_path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cases.append(
                EvalCase(
                    question=record["question"],
                    expected_keywords=record.get("expected_keywords", []),
                    expected_source=record.get("expected_source"),
                )
            )
    return cases


def run_eval(service: RagService, dataset_path: Path, use_llm: bool) -> str:
    cases = load_eval_cases(dataset_path)
    results: list[dict[str, object]] = []
    retrieval_hits = 0
    answer_hits = 0

    for case in cases:
        retrieval_state = service.retrieve_node({"question": case.question})
        retrieved = retrieval_state.get("retrieved", [])
        if use_llm:
            response = service.query(case.question)
            answer = response.get("answer", "")
            blocked = response.get("blocked", False)
        else:
            answer = _synthesize_answer(retrieved)
            blocked = False

        retrieval_hit = _retrieval_hit(retrieved, case)
        answer_hit = _answer_hit(answer, case)
        retrieval_hits += 1 if retrieval_hit else 0
        answer_hits += 1 if answer_hit else 0

        results.append(
            {
                "question": case.question,
                "blocked": blocked,
                "retrieval_hit": retrieval_hit,
                "answer_keyword_hit": answer_hit,
                "citations": [doc.citation() for doc in retrieved],
            }
        )

    total = max(1, len(cases))
    report = {
        "cases": len(cases),
        "retrieval_hit_rate": retrieval_hits / total,
        "answer_keyword_rate": answer_hits / total,
        "results": results,
    }
    return json.dumps(report, indent=2)


def _retrieval_hit(retrieved: list[DocumentChunk], case: EvalCase) -> bool:
    if not retrieved:
        return False
    if case.expected_source:
        if any(case.expected_source in doc.source for doc in retrieved):
            return True
    keywords = [keyword.lower() for keyword in case.expected_keywords]
    for doc in retrieved:
        content = doc.content.lower()
        if any(keyword in content for keyword in keywords):
            return True
    return False


def _answer_hit(answer: str, case: EvalCase) -> bool:
    if not answer:
        return False
    answer_text = answer.lower()
    return any(keyword.lower() in answer_text for keyword in case.expected_keywords)


def _synthesize_answer(retrieved: list[DocumentChunk]) -> str:
    if not retrieved:
        return ""
    return "\n\n".join(doc.content for doc in retrieved)[:600]
