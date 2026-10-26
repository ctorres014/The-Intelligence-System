"""Contratos Pydantic de la API (entradas y salidas de todos los endpoints)."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.rag.models import IngestReport, RetrievalResult


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, description="Mensaje / tarea del usuario")
    thread_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_\-]{1,64}$",
        description="Omitir para iniciar una conversación nueva; reutilizar para continuarla",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "¿Qué es Reciprocal Rank Fusion y por qué se usa en búsqueda híbrida?"},
                {"message": "¿Y cuánto ocupan 1M de vectores de 1536 dimensiones en float32?", "thread_id": "demo-1"},
            ]
        }
    }


class ApprovalBody(BaseModel):
    approved: bool = Field(description="true para aprobar, false para rechazar")
    comment: str | None = Field(default=None, max_length=1000)


class Source(BaseModel):
    ref: str
    source: str
    doc_id: str
    section: str = ""
    score: float
    retrievers: list[str]
    preview: str = ""


class Step(BaseModel):
    node: str
    summary: str
    decision: str | None = None
    timestamp: str


class PendingApproval(BaseModel):
    reason: str
    query: str
    planned_action: str
    context_preview: str
    sources: list[str] = Field(default_factory=list)


TurnStatus = Literal["completed", "awaiting_approval", "rejected"]


class ChatResponse(BaseModel):
    thread_id: str
    status: TurnStatus
    answer: str | None = None
    sources: list[Source] = Field(default_factory=list)
    agents_used: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    iterations: int = 0
    pending_approval: PendingApproval | None = None
    latency_ms: int


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ThreadState(BaseModel):
    thread_id: str
    status: TurnStatus | Literal["empty"]
    messages: list[ChatTurn]
    last_steps: list[Step] = Field(default_factory=list)
    pending_approval: PendingApproval | None = None
    checkpoint_id: str | None = None


JobStatus = Literal["pending", "running", "completed", "awaiting_approval", "rejected", "failed"]


class TaskAccepted(BaseModel):
    job_id: str
    thread_id: str
    status: JobStatus
    message: str


class JobRecord(BaseModel):
    job_id: str
    thread_id: str
    status: JobStatus
    message: str
    result: ChatResponse | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class KnowledgeQuery(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = Field(default=5, ge=1, le=20)


class KnowledgeSearchResponse(RetrievalResult):
    pass


class IngestResponse(IngestReport):
    pass


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    redis: Literal["connected", "unavailable"]
    checkpointer: str
    vector_store: str
    indexed_vectors: int
    knowledge_chunks: int
    llm_provider: str
    llm_model: str
    embedding_model: str
    tracing: str


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None
    request_id: str | None = None
