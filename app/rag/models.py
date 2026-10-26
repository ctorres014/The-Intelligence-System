"""Modelos Pydantic del subsistema RAG."""

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Fragmento de documento con sus metadatos. El `id` es determinístico."""

    id: str
    text: str = Field(min_length=1)
    doc_id: str
    source: str
    title: str = ""
    section: str = ""
    category: str = "general"
    chunk_index: int = Field(ge=0)


class RetrievedChunk(Chunk):
    score: float
    retrievers: list[Literal["bm25", "vector"]] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    strategy: Literal["hybrid", "bm25_only", "vector_only"]
    degraded: bool = False
    warning: str | None = None
    # Backend vectorial que respondió (p. ej. "pinecone", o "redis" si actuó el failover)
    vector_backend: str | None = None


class IngestReport(BaseModel):
    documents: int
    chunks: int
    vector_store: str
    embedding_model: str
    dimensions: int
    seconds: float
