import pytest

from aegis_rag_lab.config import reset_settings_cache
from aegis_rag_lab.deps import reset_service_cache


@pytest.fixture(autouse=True)
def _configure_test_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AEGIS_VECTOR_BACKEND", "memory")
    monkeypatch.setenv("AEGIS_EMBEDDINGS_PROVIDER", "deterministic")
    monkeypatch.setenv("AEGIS_LLM_PROVIDER", "stub")
    monkeypatch.setenv("AEGIS_GUARDRAILS_ENABLED", "false")
    reset_settings_cache()
    reset_service_cache()
    yield
    reset_settings_cache()
    reset_service_cache()
