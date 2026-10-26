"""
Observabilidad: logging + trazas OpenTelemetry hacia Arize Phoenix.

`LangChainInstrumentor` (OpenInference) captura automáticamente cada
ejecución del grafo, cada nodo, cada llamada al LLM (prompts, tokens,
latencia) y cada tool call (argumentos y salida). Las trazas se agrupan por
`session.id` = `thread_id`, así en la pestaña *Sessions* de Phoenix se ve una
conversación completa.

Alternativa: LangSmith, activando LANGSMITH_TRACING=true + LANGSMITH_API_KEY
(LangChain lo detecta solo, sin código adicional).
"""

import logging
import os
import sys
from contextlib import contextmanager
from typing import Iterator

from app.config import Settings

logger = logging.getLogger(__name__)

_instrumented = False


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        stream=sys.stdout,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def init_tracing(settings: Settings) -> str:
    """Inicializa el backend de trazas y devuelve su nombre."""
    global _instrumented
    if settings.phoenix_enabled:
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
            from phoenix.otel import register

            endpoint = settings.phoenix_collector_endpoint.rstrip("/") + "/v1/traces"
            tracer_provider = register(
                endpoint=endpoint,
                project_name=settings.phoenix_project_name,
                batch=True,
                set_global_tracer_provider=True,
                verbose=False,
            )
            if not _instrumented:
                LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
                _instrumented = True
            logger.info("Trazas → Arize Phoenix (%s, proyecto '%s')", endpoint, settings.phoenix_project_name)
            return "phoenix"
        except Exception as exc:  # Phoenix es opcional: la API arranca igual
            logger.warning("No se pudo inicializar Phoenix (%s); continuando sin trazas OTel", exc)

    if os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "")).lower() == "true":
        logger.info("Trazas → LangSmith")
        return "langsmith"

    logger.warning("Sin backend de trazas configurado")
    return "none"


@contextmanager
def trace_session(thread_id: str, **metadata: str) -> Iterator[None]:
    """Etiqueta los spans con session.id / metadata (no-op si OpenInference no está)."""
    try:
        from openinference.instrumentation import using_attributes
    except ImportError:
        yield
        return
    with using_attributes(session_id=thread_id, metadata=metadata):
        yield
