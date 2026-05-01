from fastapi import APIRouter
from pydantic import BaseModel, Field

from aegis_rag_lab.deps import get_rag_service
from aegis_rag_lab.rag.models import DocumentInput

router = APIRouter()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    blocked: bool = False
    reason: str | None = None


class DocumentIn(BaseModel):
    source: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    metadata: dict[str, str] | None = None


class IngestRequest(BaseModel):
    documents: list[DocumentIn]


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class StatsResponse(BaseModel):
    sources: int
    chunks: int


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    service = get_rag_service()
    result = service.query(payload.question)
    return QueryResponse(**result)


@router.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest) -> IngestResponse:
    service = get_rag_service()
    documents = [
        DocumentInput(
            source=item.source,
            content=item.content,
            metadata=item.metadata or {},
        )
        for item in payload.documents
    ]
    result = service.ingest_documents(documents)
    return IngestResponse(**result)


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    service = get_rag_service()
    result = service.stats()
    return StatsResponse(**result)
