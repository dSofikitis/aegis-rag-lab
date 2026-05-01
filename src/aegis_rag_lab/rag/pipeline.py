from aegis_rag_lab.deps import get_rag_service


def run_query(question: str) -> dict[str, object]:
    service = get_rag_service()
    return service.query(question)
