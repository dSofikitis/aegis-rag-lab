from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from aegis_rag_lab.api import router as api_router
from aegis_rag_lab.config import get_settings
from aegis_rag_lab.deps import get_rag_service
from aegis_rag_lab.logging import configure_logging


def _resolve_ui_dist() -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    ui_dist = project_root / "ui" / "dist"
    if ui_dist.exists():
        return ui_dist
    return None


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    app = FastAPI(title="Aegis RAG Lab", version="0.1.0")
    origins = [origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(api_router)

    ui_dist = _resolve_ui_dist()
    if ui_dist:
        app.mount("/ui", StaticFiles(directory=str(ui_dist), html=True), name="ui")

    @app.get("/", response_model=None)
    def root() -> Response | dict[str, str]:
        if ui_dist:
            return RedirectResponse(url="/ui")
        return {"name": "Aegis RAG Lab", "status": "ok", "docs": "/docs"}

    @app.on_event("startup")
    def on_startup() -> None:
        service = get_rag_service()
        service.ensure_ready()

    return app


app = create_app()
