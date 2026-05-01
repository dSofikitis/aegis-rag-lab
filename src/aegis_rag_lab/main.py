from fastapi import FastAPI

from aegis_rag_lab.api import router as api_router
from aegis_rag_lab.deps import get_rag_service
from aegis_rag_lab.logging import configure_logging


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Aegis RAG Lab", version="0.1.0")
    app.include_router(api_router)

    @app.on_event("startup")
    def on_startup() -> None:
        service = get_rag_service()
        service.ensure_ready()

    return app


app = create_app()
