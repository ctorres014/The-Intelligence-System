"""
API REST del Intelligence System.

    POST /v1/chat                      Turno de conversación (espera la respuesta)
    POST /v1/chat/stream               Igual, pero con Server-Sent Events por nodo
    GET  /v1/threads/{thread_id}       Historial y estado del hilo (desde el checkpointer)
    POST /v1/threads/{thread_id}/approve  Resolver un HITL pendiente
    POST /v1/tasks                     Encolar una tarea en background (202)
    GET  /v1/tasks/{job_id}            Estado de la tarea
    POST /v1/tasks/{job_id}/approve    Resolver el HITL de una tarea
    POST /v1/knowledge/search          Recuperación híbrida directa (debug del RAG)
    POST /v1/knowledge/ingest          Re-ingestar la base de conocimiento
    GET  /health                       Estado de dependencias
"""

import json
import logging
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from langgraph.checkpoint.base import BaseCheckpointSaver
from redis.asyncio import Redis
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.agents.graph import build_graph
from app.api.jobs import JobRunner, JobStore
from app.api.schemas import (
    ApprovalBody,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    JobRecord,
    KnowledgeQuery,
    KnowledgeSearchResponse,
    TaskAccepted,
    ThreadState,
)
from app.config import Settings, get_settings
from app.llm import build_chat_model, build_embeddings
from app.observability import init_tracing, setup_logging
from app.rag.knowledge_base import KnowledgeBase
from app.rag.vectorstore import build_vector_store
from app.service import ApprovalStateError, IntelligenceService, ThreadBusyError

logger = logging.getLogger("app.api")

_ID_PATTERN = r"^[A-Za-z0-9_\-]{1,64}$"
ThreadId = Annotated[str, Path(pattern=_ID_PATTERN, description="ID de la conversación")]
JobId = Annotated[str, Path(pattern=_ID_PATTERN, description="ID de la tarea")]


@dataclass
class AppContext:
    settings: Settings
    redis: Redis
    redis_ok: bool
    checkpointer_name: str
    kb: KnowledgeBase
    service: IntelligenceService
    jobs: JobRunner
    tracing: str


async def _build_checkpointer(
    settings: Settings, redis_ok: bool, stack: AsyncExitStack
) -> tuple[BaseCheckpointSaver, str]:
    if redis_ok:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        ttl = (
            {"default_ttl": settings.checkpoint_ttl_minutes, "refresh_on_read": True}
            if settings.checkpoint_ttl_minutes
            else None
        )
        saver = await stack.enter_async_context(AsyncRedisSaver.from_conn_string(settings.redis_url, ttl=ttl))
        await saver.asetup()
        return saver, "AsyncRedisSaver"

    from langgraph.checkpoint.memory import InMemorySaver

    logger.error("Redis no disponible: usando InMemorySaver (las conversaciones NO sobreviven a un reinicio)")
    return InMemorySaver(), "InMemorySaver (sin persistencia)"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    tracing = init_tracing(settings)

    async with AsyncExitStack() as stack:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        stack.push_async_callback(redis.aclose)
        try:
            redis_ok = bool(await redis.ping())
        except Exception as exc:
            logger.error("No se pudo conectar a Redis (%s): %s", settings.redis_url, exc)
            redis_ok = False

        checkpointer, checkpointer_name = await _build_checkpointer(settings, redis_ok, stack)

        kb = KnowledgeBase(settings, build_vector_store(settings), build_embeddings(settings))
        stack.push_async_callback(kb.close)
        await kb.load()
        if settings.auto_ingest:
            try:
                await kb.ensure_ingested()
            except Exception as exc:
                logger.error("Auto-ingesta falló (%s); el RAG funcionará en modo BM25", exc)

        graph = build_graph(build_chat_model(settings), kb, settings, checkpointer)
        service = IntelligenceService(graph)
        jobs = JobRunner(JobStore(redis, settings.job_ttl_seconds), service)
        stack.push_async_callback(jobs.shutdown)

        app.state.ctx = AppContext(
            settings=settings, redis=redis, redis_ok=redis_ok, checkpointer_name=checkpointer_name,
            kb=kb, service=service, jobs=jobs, tracing=tracing,
        )
        logger.info(
            "API lista · LLM=%s/%s · vector_store=%s · checkpointer=%s · tracing=%s",
            settings.llm_provider, settings.llm_model_name, kb.store_name, checkpointer_name, tracing,
        )
        yield
    logger.info("API detenida")


app = FastAPI(
    title="Intelligence System API",
    version="1.0.0",
    description=(
        "Sistema de IA de grado producción: **RAG híbrido** (BM25 + vectorial + RRF) + "
        "**multi-agente con Supervisor** (LangGraph) + **HITL** + **persistencia** con "
        "AsyncRedisSaver + **trazas** en Arize Phoenix.\n\n"
        "Entrega final — CoderHouse AI Engineering."
    ),
    lifespan=lifespan,
    responses={500: {"model": ErrorResponse}},
)


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


# ── Middleware y manejo de errores ──────────────────────────────────────────


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["x-request-id"] = request_id
    logger.info("%s %s → %s (%.0f ms) [%s]", request.method, request.url.path, response.status_code, elapsed, request_id)
    return response


def _error(request: Request, status: int, error: str, detail=None) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, request_id=getattr(request.state, "request_id", None))
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(StarletteHTTPException)
async def _http_handler(request: Request, exc: StarletteHTTPException):
    errors = {404: "not_found", 409: "conflict", 503: "service_unavailable"}
    return _error(request, exc.status_code, errors.get(exc.status_code, "http_error"), exc.detail)


@app.exception_handler(ThreadBusyError)
async def _busy_handler(request: Request, exc: ThreadBusyError):
    return _error(request, 409, "thread_busy", str(exc))


@app.exception_handler(ApprovalStateError)
async def _approval_handler(request: Request, exc: ApprovalStateError):
    return _error(request, 409, "approval_state", str(exc))


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    logger.exception("Error no controlado")
    return _error(request, 500, "internal_error", str(exc))


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


@app.post("/v1/chat", response_model=ChatResponse, tags=["chat"], responses={409: {"model": ErrorResponse}})
async def chat(body: ChatRequest, ctx: AppContext = Depends(get_ctx)) -> ChatResponse:
    """Envía un mensaje. Sin `thread_id` crea una conversación nueva; con `thread_id` la continúa."""
    return await ctx.service.chat(body.message, body.thread_id)


@app.post("/v1/chat/stream", tags=["chat"], responses={200: {"content": {"text/event-stream": {}}}})
async def chat_stream(body: ChatRequest, ctx: AppContext = Depends(get_ctx)):
    """Igual que `/v1/chat` pero emite eventos SSE: `start`, un `step` por nodo y `final`."""
    thread_id = body.thread_id or ctx.service.new_thread_id()
    await ctx.service.ensure_can_start(thread_id)

    async def events():
        try:
            async for event in ctx.service.stream(body.message, thread_id):
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("Error durante el stream")
            yield f"event: error\ndata: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/v1/threads/{thread_id}", response_model=ThreadState, tags=["threads"], responses={404: {"model": ErrorResponse}})
async def get_thread(thread_id: ThreadId, ctx: AppContext = Depends(get_ctx)) -> ThreadState:
    """Historial persistido del hilo (se lee del checkpointer, sobrevive a reinicios)."""
    state = await ctx.service.thread_state(thread_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No existe el hilo '{thread_id}'")
    return state


@app.post(
    "/v1/threads/{thread_id}/approve",
    response_model=ChatResponse,
    tags=["threads"],
    responses={409: {"model": ErrorResponse}},
)
async def approve_thread(body: ApprovalBody, thread_id: ThreadId, ctx: AppContext = Depends(get_ctx)) -> ChatResponse:
    """Aprueba o rechaza la operación crítica pendiente y reanuda el grafo."""
    return await ctx.service.resume(thread_id, body.approved, body.comment)


@app.post("/v1/tasks", response_model=TaskAccepted, status_code=202, tags=["tasks"])
async def submit_task(body: ChatRequest, ctx: AppContext = Depends(get_ctx)) -> TaskAccepted:
    """Encola la tarea y responde al instante; consultar con `GET /v1/tasks/{job_id}`."""
    if not ctx.redis_ok:
        raise HTTPException(status_code=503, detail="Redis no disponible: las tareas en background requieren Redis")
    thread_id = body.thread_id or ctx.service.new_thread_id()
    job = await ctx.jobs.submit(body.message, thread_id)
    return TaskAccepted(
        job_id=job.job_id, thread_id=thread_id, status=job.status,
        message=f"Tarea encolada. Consultá GET /v1/tasks/{job.job_id}",
    )


@app.get("/v1/tasks/{job_id}", response_model=JobRecord, tags=["tasks"], responses={404: {"model": ErrorResponse}})
async def get_task(job_id: JobId, ctx: AppContext = Depends(get_ctx)) -> JobRecord:
    job = await ctx.jobs.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No existe la tarea '{job_id}'")
    return job


@app.post("/v1/tasks/{job_id}/approve", response_model=TaskAccepted, status_code=202, tags=["tasks"])
async def approve_task(body: ApprovalBody, job_id: JobId, ctx: AppContext = Depends(get_ctx)) -> TaskAccepted:
    job = await ctx.jobs.store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No existe la tarea '{job_id}'")
    if job.status != "awaiting_approval":
        raise HTTPException(status_code=409, detail=f"La tarea está en estado '{job.status}', no 'awaiting_approval'")
    ctx.jobs.resume(job, body.approved, body.comment)
    verdict = "aprobada" if body.approved else "rechazada"
    return TaskAccepted(job_id=job.job_id, thread_id=job.thread_id, status="running", message=f"Tarea {verdict}; reanudando")


@app.post("/v1/knowledge/search", response_model=KnowledgeSearchResponse, tags=["knowledge"])
async def knowledge_search(body: KnowledgeQuery, ctx: AppContext = Depends(get_ctx)) -> KnowledgeSearchResponse:
    """Recuperación híbrida directa, sin LLM (útil para depurar el RAG)."""
    result = await ctx.kb.search(body.query, body.k)
    return KnowledgeSearchResponse.model_validate(result.model_dump())


@app.post("/v1/knowledge/ingest", response_model=IngestResponse, tags=["knowledge"])
async def knowledge_ingest(ctx: AppContext = Depends(get_ctx)) -> IngestResponse:
    """Re-procesa `data/documents/` y recrea el índice vectorial."""
    report = await ctx.kb.ingest()
    return IngestResponse.model_validate(report.model_dump())


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(ctx: AppContext = Depends(get_ctx)) -> HealthResponse:
    try:
        redis_ok = bool(await ctx.redis.ping())
    except Exception:
        redis_ok = False
    try:
        indexed = await ctx.kb.indexed_count()
    except Exception:
        indexed = 0
    healthy = redis_ok and indexed > 0
    return HealthResponse(
        status="ok" if healthy else "degraded",
        redis="connected" if redis_ok else "unavailable",
        checkpointer=ctx.checkpointer_name,
        vector_store=ctx.kb.store_name,
        indexed_vectors=indexed,
        knowledge_chunks=ctx.kb.chunk_count,
        llm_provider=ctx.settings.llm_provider,
        llm_model=ctx.settings.llm_model_name,
        embedding_model=ctx.settings.embedding_model_name,
        tracing=ctx.tracing,
    )
