"""Central settings loaded from .env (pydantic-settings)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM (Gemini REST)
    gemini_api_key: str = ""
    llm_model: str = "gemini-flash-latest"
    embed_model: str = "gemini-embedding-001"
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 2048

    # Chat provider: "lmstudio" (local, quota-free) or "gemini"
    chat_provider: str = "lmstudio"
    lmstudio_chat_model: str = "qwen-chat"
    lmstudio_chat_max_tokens: int = 2048
    # Qwen3.x thinking mode is slow on a 9B local model; default off, opt-in per call.
    lmstudio_disable_thinking: bool = True

    # Embedding provider: "lmstudio" (local, quota-free) or "gemini" (free tier)
    embed_provider: str = "lmstudio"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"
    lmstudio_embed_model: str = "nomic-embed"
    lmstudio_embed_batch: int = 64

    # Remote OpenAI-compatible provider (Vast.ai: vLLM Qwen3-30B-A3B + nomic-embed)
    # Used when chat_provider / embed_provider == "openai_compat".
    openai_chat_base_url: str = ""
    openai_embed_base_url: str = ""
    openai_api_key: str = ""
    openai_chat_model: str = "qwen3-30b-a3b"
    openai_embed_model: str = "nomic-embed-text-v1.5"
    openai_chat_max_tokens: int = 2048
    openai_embed_batch: int = 32
    openai_disable_thinking: bool = True

    # LLM disk cache (protects free-tier quota; makes evals reproducible)
    llm_cache_enabled: bool = True
    llm_cache_path: str = str(ROOT / ".cache" / "llm_cache.db")
    llm_cache_max_entries: int = 5000

    # Postgres
    postgres_user: str = "aiocopilot"
    postgres_password: str = "aiocopilot"
    postgres_db: str = "aiocopilot"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "aiocopilot"

    # Auth
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480

    # Services
    api_base_url: str = "http://localhost:8000"
    model_service_url: str = "http://localhost:8001"

    # Ingestion limits (0 = ingest all pages). Embeddings are disk-cached, so
    # re-ingests are cheap; the quota cost is paid once per distinct chunk text.
    ingest_max_pages_per_doc: int = 0
    ingest_chunk_size: int = 700
    ingest_chunk_overlap: int = 120
    ingest_extract_entities: bool = False

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
