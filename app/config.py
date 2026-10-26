"""
Configuración centralizada del sistema.

Toda la configuración (modelos, índices, URLs, pesos del retriever) se lee de
variables de entorno / `.env` y se valida con Pydantic. Ningún módulo del
proyecto lee `os.environ` directamente ni tiene IDs de modelos o índices
hard-codeados: todo pasa por `get_settings()`.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["anthropic", "gemini", "openai", "ollama"] = "ollama"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    # Vacío = 600 s para Ollama (inferencia en CPU) y 120 s para APIs cloud
    llm_timeout_seconds: Optional[float] = Field(default=None, gt=0)

    anthropic_api_key: Optional[SecretStr] = None
    anthropic_model: str = "claude-sonnet-5-5"

    # Acepta el nombre de LangChain (GOOGLE_API_KEY) o el de la doc de Google (GEMINI_API_KEY)
    google_api_key: Optional[SecretStr] = Field(
        default=None, validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY")
    )
    gemini_model: str = "gemini-3.8-flash"
    # Google recomienda temperatura 1.0 para los modelos Gemini 3.x
    gemini_temperature: float = Field(default=1.0, ge=0.0, le=2.0)

    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-4o-mini"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_num_ctx: int = Field(default=8192, ge=2048)

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_provider: Literal["ollama", "gemini", "openai"] = "ollama"
    ollama_embedding_model: str = "nomic-embed-text"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "gemini-embedding-2"
    # Dimensión de salida (Matryoshka); vacío = dimensión nativa del modelo
    gemini_embedding_dimensions: Optional[int] = Field(default=768, ge=128, le=3072)

    # ── Vector store ─────────────────────────────────────────────────────────
    vector_store: Literal["redis", "pinecone", "memory"] = "redis"
    # Réplica de failover: si el principal falla o tarda, se consulta esta (luego BM25)
    vector_store_fallback: Optional[Literal["redis", "pinecone", "memory"]] = None
    vector_search_timeout_seconds: float = Field(default=5.0, gt=0)
    redis_index_name: str = "intelligence-kb"

    pinecone_api_key: Optional[SecretStr] = None
    pinecone_index_name: str = "intelligence-kb"
    pinecone_namespace: str = "documents"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ── RAG ──────────────────────────────────────────────────────────────────
    documents_dir: Path = ROOT_DIR / "data" / "documents"
    chunk_size: int = Field(default=600, gt=50)
    chunk_overlap: int = Field(default=100, ge=0)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    auto_ingest: bool = True

    # ── Research tools ───────────────────────────────────────────────────────
    tavily_api_key: Optional[SecretStr] = None
    wikipedia_lang: str = "es"
    # Wikimedia exige un User-Agent con datos de contacto (https://w.wiki/4wJS)
    wikipedia_user_agent: str = "IntelligenceSystem/1.0 (https://github.com/ctorres014; educational project)"
    http_timeout_seconds: float = Field(default=15.0, gt=0)

    # ── Redis (checkpointer + jobs + vector store) ──────────────────────────
    redis_url: str = "redis://localhost:6379"
    job_ttl_seconds: int = Field(default=86400, gt=0)
    # TTL de los checkpoints en minutos (None = sin expiración)
    checkpoint_ttl_minutes: Optional[int] = None

    # ── Orquestación ─────────────────────────────────────────────────────────
    max_iterations: int = Field(default=8, ge=2, le=30)
    history_window: int = Field(default=6, ge=0, le=50)
    # El primer paso de cada turno consulta siempre la base de conocimiento
    rag_first: bool = True

    # ── Observabilidad ───────────────────────────────────────────────────────
    phoenix_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006"
    phoenix_project_name: str = "intelligence-system"
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP debe ser menor que CHUNK_SIZE")
        if self.bm25_weight + self.vector_weight == 0:
            raise ValueError("BM25_WEIGHT y VECTOR_WEIGHT no pueden ser ambos 0")
        if self.vector_store_fallback == self.vector_store:
            raise ValueError("VECTOR_STORE_FALLBACK debe ser distinto de VECTOR_STORE")
        return self

    @property
    def llm_timeout(self) -> float:
        if self.llm_timeout_seconds:
            return self.llm_timeout_seconds
        return 600.0 if self.llm_provider == "ollama" else 120.0

    @property
    def llm_model_name(self) -> str:
        return {
            "anthropic": self.anthropic_model,
            "gemini": self.gemini_model,
            "openai": self.openai_model,
            "ollama": self.ollama_model,
        }[self.llm_provider]

    @property
    def embedding_model_name(self) -> str:
        return {
            "ollama": self.ollama_embedding_model,
            "gemini": self.gemini_embedding_model,
            "openai": self.openai_embedding_model,
        }[self.embedding_provider]

    @property
    def embedding_fingerprint(self) -> str:
        """Identifica el espacio vectorial: si cambia, el índice debe recrearse."""
        dims = self.gemini_embedding_dimensions if self.embedding_provider == "gemini" else None
        return f"{self.embedding_provider}:{self.embedding_model_name}:{dims or 'native'}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
