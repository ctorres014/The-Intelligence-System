"""
Supervisor: decide qué agente actúa a continuación.

Usa salida estructurada validada con Pydantic (`SupervisorDecision`). Si el
LLM falla o devuelve algo inválido, aplica reglas determinísticas. Además
aplica guardas duras: tope de iteraciones y máximo de 2 ejecuciones por agente
en un mismo turno, para garantizar que el grafo siempre termine.
"""

import logging
import re
from collections import Counter
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.common import format_history, truncate
from app.agents.prompts import SUPERVISOR_PROMPT
from app.agents.state import AgentState, SupervisorDecision, add_step
from app.config import Settings

logger = logging.getLogger(__name__)

MAX_RUNS_PER_WORKER = 2

# Operaciones con efectos colaterales: requieren aprobación humana (HITL)
_CRITICAL_PATTERNS = re.compile(
    r"\b(deploy|delete|drop|truncate|wipe|purge|execute|publish|send email|purchase|payment|"
    r"transfer|overwrite|terminate|"
    r"borr(ar|á|e|en)|elimin(ar|á|e|en)|desplegar|despleg(á|ue)|ejecut(ar|á|e)|public(ar|á)|"
    r"comprar|pagar|transferir|sobrescribir|vaciar|purgar|resetear)\b",
    re.IGNORECASE,
)


# Pedidos explícitos que un modelo chico suele "olvidar" al decidir FINISH
_CALC_REQUEST = re.compile(r"\b(calcul\w*|estim\w*|porcentaje|cuánto\w* (gb|mb|cuesta|ocupa\w*))\b", re.IGNORECASE)
_EXTERNAL_REQUEST = re.compile(
    r"\b(quién(es)? (desarroll|cre|invent|propus)\w*|investig(á|ar)|wikipedia|en internet|en la web|noticias)\b",
    re.IGNORECASE,
)


def is_critical_query(query: str) -> bool:
    return bool(_CRITICAL_PATTERNS.search(query or ""))


def _worker_runs(state: AgentState) -> Counter:
    return Counter(s["node"] for s in state.get("steps", []) if s["node"] in {"knowledge", "research", "analyst"})


def rule_based_decision(state: AgentState) -> SupervisorDecision:
    """Fallback determinístico: knowledge → analyst (si se piden números) → FINISH."""
    runs = _worker_runs(state)
    query = state.get("user_query", "").lower()
    if not runs["knowledge"]:
        return SupervisorDecision(next="knowledge", reasoning="fallback: falta contexto interno")
    needs_numbers = bool(re.search(r"\d|calcul|cost|cuánt|cuant|estim|compar|porcentaje", query))
    if needs_numbers and not runs["analyst"]:
        return SupervisorDecision(next="analyst", reasoning="fallback: la consulta requiere cálculo")
    return SupervisorDecision(next="FINISH", reasoning="fallback: información suficiente")


def enforce_coverage(state: AgentState, decision: SupervisorDecision) -> SupervisorDecision:
    """Evita cerrar el turno sin el agente que la consulta pide de forma explícita."""
    if decision.next != "FINISH":
        return decision
    runs = _worker_runs(state)
    query = state.get("user_query", "")
    if _EXTERNAL_REQUEST.search(query) and not runs["research"]:
        return SupervisorDecision(next="research", reasoning="cobertura: la consulta pide información externa",
                                  instructions=query)
    if _CALC_REQUEST.search(query) and not runs["analyst"]:
        return SupervisorDecision(next="analyst", reasoning="cobertura: la consulta pide un cálculo explícito",
                                  instructions=query)
    return decision


def _context_for_llm(state: AgentState, settings: Settings) -> str:
    runs = _worker_runs(state)
    parts = [
        f"Consulta actual del usuario: {state.get('user_query', '')}",
        f"Iteración: {state.get('iteration_count', 0)} / {settings.max_iterations}",
        f"Agentes ya ejecutados en este turno: {dict(runs) or 'ninguno'}",
    ]
    history = format_history(state, settings.history_window)
    if history:
        parts.append(f"\nHistorial de la conversación:\n{history}")
    for label, key in (
        ("Contexto interno (knowledge)", "knowledge_context"),
        ("Investigación externa (research)", "research_data"),
        ("Análisis (analyst)", "analysis_result"),
    ):
        value = state.get(key)
        parts.append(f"\n{label}: " + (truncate(value, 1200) if value else "— (vacío)"))
    return "\n".join(parts)


def make_supervisor_node(llm: BaseChatModel, settings: Settings, structured_method: str) -> Callable:
    structured_llm = llm.with_structured_output(SupervisorDecision, method=structured_method)

    async def supervisor_node(state: AgentState) -> dict:
        iteration = state.get("iteration_count", 0)
        critical = state.get("is_critical") or is_critical_query(state.get("user_query", ""))

        if iteration >= settings.max_iterations:
            decision = SupervisorDecision(next="FINISH", reasoning=f"tope de {settings.max_iterations} iteraciones alcanzado")
        elif iteration == 0 and settings.rag_first:
            # Política RAG-first: todo turno se ancla primero en la base de conocimiento interna
            decision = SupervisorDecision(
                next="knowledge",
                reasoning="política RAG-first: anclar la respuesta en la base de conocimiento interna",
                instructions=state.get("user_query"),
            )
        else:
            try:
                decision = await structured_llm.ainvoke(
                    [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=_context_for_llm(state, settings))]
                )
                if not isinstance(decision, SupervisorDecision):
                    decision = SupervisorDecision.model_validate(decision)
            except Exception as exc:
                logger.warning("Supervisor: salida estructurada inválida (%s); usando reglas", exc)
                decision = rule_based_decision(state)
            decision = enforce_coverage(state, decision)

            if decision.next != "FINISH" and _worker_runs(state)[decision.next] >= MAX_RUNS_PER_WORKER:
                decision = SupervisorDecision(
                    next="FINISH", reasoning=f"'{decision.next}' ya se ejecutó {MAX_RUNS_PER_WORKER} veces"
                )

        logger.info("Supervisor → %s (%s)", decision.next, decision.reasoning)
        return {
            "next_agent": decision.next,
            "instructions": decision.instructions,
            "is_critical": critical,
            "iteration_count": iteration + 1,
            "steps": add_step(state, "supervisor", decision.reasoning, decision=decision.next),
        }

    return supervisor_node


def route_from_supervisor(state: AgentState) -> str:
    """
    Arista condicional del Supervisor.

    Las consultas críticas pasan por `human_approval` antes de producir un
    resultado accionable (análisis o respuesta final).
    """
    next_agent = state.get("next_agent", "FINISH")
    if next_agent in {"analyst", "FINISH"} and state.get("is_critical") and state.get("is_approved") is None:
        return "human_approval"
    if next_agent in {"knowledge", "research", "analyst"}:
        return next_agent
    return "synthesizer"
