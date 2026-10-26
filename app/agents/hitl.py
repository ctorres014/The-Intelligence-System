"""
Human-in-the-Loop.

`human_approval_node` llama a `interrupt()`: LangGraph persiste el checkpoint
completo (en Redis) y detiene la ejecución. La API reanuda el hilo con
`Command(resume=ApprovalDecision)` — incluso después de reiniciar el servidor,
porque el estado vive en el checkpointer y no en memoria.
"""

import logging

from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from app.agents.common import truncate
from app.agents.state import AgentState, add_step

logger = logging.getLogger(__name__)


class ApprovalRequest(BaseModel):
    """Payload que ve el operador mientras el hilo está pausado."""

    reason: str
    query: str
    planned_action: str
    context_preview: str
    sources: list[str] = Field(default_factory=list)


class ApprovalDecision(BaseModel):
    approved: bool
    comment: str | None = Field(default=None, max_length=1000)


async def human_approval_node(state: AgentState) -> dict:
    request = ApprovalRequest(
        reason="La consulta implica una operación con efectos colaterales (borrar, desplegar, ejecutar, pagar…).",
        query=state.get("user_query", ""),
        planned_action=state.get("next_agent", ""),
        context_preview=truncate(state.get("knowledge_context") or state.get("research_data"), 600),
        sources=[s.get("ref", "") for s in state.get("sources", [])],
    )
    logger.info("HITL: pausa para aprobación humana")

    raw = interrupt(request.model_dump())

    try:
        decision = ApprovalDecision.model_validate(raw)
    except ValidationError:
        decision = ApprovalDecision(approved=bool(raw))

    logger.info("HITL: reanudado con approved=%s", decision.approved)
    verdict = "aprobado" if decision.approved else "rechazado"
    return {
        "is_approved": decision.approved,
        "approval_comment": decision.comment,
        "steps": add_step(state, "human_approval", f"{verdict}. {decision.comment or ''}".strip(), decision=verdict),
    }


def route_after_approval(state: AgentState) -> str:
    if state.get("is_approved") and state.get("next_agent") == "analyst":
        return "analyst"
    return "synthesizer"
