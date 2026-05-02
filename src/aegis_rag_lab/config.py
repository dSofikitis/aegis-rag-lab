from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embeddings_provider: str = "openai"
    llm_provider: str = "openai"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_request_timeout_s: float = 180.0
    vector_backend: str = "postgres"
    database_url: str = "postgresql+psycopg://aegis:aegis@localhost:5432/aegis"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"
    allowed_origins: str = ""
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_k: int = 5
    retrieval_min_similarity: float = 0.0
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20
    max_context_chars: int = 4000
    guardrails_enabled: bool = True
    request_timeout_s: float = 20.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AEGIS_",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
