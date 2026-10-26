"""
Estado compartido del grafo y modelos Pydantic de las decisiones.

`messages` guarda SOLO la conversación visible (turnos del usuario y respuestas
finales del Synthesizer); es lo que el checkpointer persiste entre turnos y
reinicios. El resto de los campos es el "espacio de trabajo" del turno actual
y se reinicia en cada mensaje nuevo (ver `new_turn_input`).
"""

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

WorkerName = Literal["knowledge", "research", "analyst"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    next_agent: str
    instructions: str | None
    knowledge_context: str | None
    sources: list[dict[str, Any]]
    research_data: str | None
    analysis_result: str | None
    final_answer: str | None
    iteration_count: int
    steps: list[dict[str, Any]]
    is_critical: bool
    is_approved: bool | None
    approval_comment: str | None


class SupervisorDecision(BaseModel):
    """Salida estructurada del Supervisor (validada por Pydantic)."""

    next: Literal["knowledge", "research", "analyst", "FINISH"] = Field(
        description="Siguiente agente a ejecutar, o FINISH para pasar al Synthesizer"
    )
    reasoning: str = Field(description="Por qué se eligió esta acción (breve)")
    instructions: str | None = Field(
        default=None, description="Instrucción concreta para el agente elegido (qué buscar / calcular)"
    )


class StepRecord(BaseModel):
    node: str
    summary: str
    decision: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def add_step(state: AgentState, node: str, summary: str, decision: str | None = None) -> list[dict[str, Any]]:
    record = StepRecord(node=node, summary=summary[:500], decision=decision)
    return [*state.get("steps", []), record.model_dump()]


def new_turn_input(query: str) -> dict[str, Any]:
    """Entrada de un turno nuevo: agrega el mensaje y resetea el espacio de trabajo."""
    return {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "next_agent": "",
        "instructions": None,
        "knowledge_context": None,
        "sources": [],
        "research_data": None,
        "analysis_result": None,
        "final_answer": None,
        "iteration_count": 0,
        "steps": [],
        "is_critical": False,
        "is_approved": None,
        "approval_comment": None,
    }
