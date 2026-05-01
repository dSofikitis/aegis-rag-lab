from functools import lru_cache

from aegis_rag_lab.config import get_settings
from aegis_rag_lab.rag.service import RagService


@lru_cache
def get_rag_service() -> RagService:
    settings = get_settings()
    return RagService(settings)


def reset_service_cache() -> None:
    get_rag_service.cache_clear()
