from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from aegis_rag_lab.deps import get_rag_service
from aegis_rag_lab.logging import get_logger
from aegis_rag_lab.rag.ingestion import load_documents_from_bytes
from aegis_rag_lab.rag.models import DocumentInput

router = APIRouter()
logger = get_logger()


class QueryRequest(BaseModel):
    question: str


class Citation(BaseModel):
    source: str
    content: str
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
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
    try:
        result = service.query(payload.question)
    except Exception as exc:
        logger.exception("query_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Query failed: {exc}") from exc
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
    try:
        result = service.ingest_documents(documents)
    except Exception as exc:
        logger.exception("ingest_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Ingest failed: {exc}") from exc
    return IngestResponse(**result)


@router.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(files: list[UploadFile] = File(...)) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    service = get_rag_service()
    documents: list[DocumentInput] = []
    for uploaded in files:
        content = await uploaded.read()
        filename = uploaded.filename or "upload"
        documents.extend(load_documents_from_bytes(filename, content))
    if not documents:
        raise HTTPException(status_code=400, detail="No supported content found.")
    try:
        result = service.ingest_documents(documents)
    except Exception as exc:
        logger.exception("ingest_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Ingest failed: {exc}") from exc
    return IngestResponse(**result)


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    service = get_rag_service()
    try:
        result = service.stats()
    except Exception as exc:
        logger.exception("stats_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Stats failed: {exc}") from exc
    return StatsResponse(**result)
