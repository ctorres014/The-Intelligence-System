"""Synthesizer: redacta la respuesta final a partir del trabajo de los agentes."""

import logging
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.common import format_history, message_text, truncate
from app.agents.prompts import SYNTHESIZER_PROMPT
from app.agents.state import AgentState, add_step
from app.config import Settings

logger = logging.getLogger(__name__)

REJECTED_ANSWER = (
    "La solicitud fue marcada como crítica y un operador humano **rechazó** su ejecución, "
    "por lo que no se generó el plan ni el análisis solicitado."
)


def _build_context(state: AgentState, settings: Settings) -> str:
    parts = [f"Consulta del usuario: {state.get('user_query', '')}"]
    history = format_history(state, settings.history_window)
    if history:
        parts.append(f"Historial de la conversación:\n{history}")
    if state.get("knowledge_context"):
        parts.append("Contexto de la base de conocimiento interna:\n" + truncate(state["knowledge_context"], 8000))
    if state.get("research_data"):
        parts.append("Investigación externa:\n" + truncate(state["research_data"], 5000))
    if state.get("analysis_result"):
        parts.append("Análisis cuantitativo:\n" + truncate(state["analysis_result"], 5000))
    if state.get("is_approved") is True:
        parts.append(f"Aprobación humana: APROBADO. {state.get('approval_comment') or ''}")
    return "\n\n".join(parts)


def make_synthesizer_node(llm: BaseChatModel, settings: Settings) -> Callable:
    async def synthesizer_node(state: AgentState) -> dict:
        if state.get("is_approved") is False:
            answer = REJECTED_ANSWER
            if state.get("approval_comment"):
                answer += f"\n\nComentario del operador: {state['approval_comment']}"
        else:
            try:
                response = await llm.ainvoke(
                    [SystemMessage(content=SYNTHESIZER_PROMPT), HumanMessage(content=_build_context(state, settings))],
                    config={"run_name": "synthesizer_llm"},
                )
                answer = message_text(response).strip() or "No pude generar una respuesta."
            except Exception as exc:
                logger.exception("Synthesizer falló")
                answer = f"No pude generar la respuesta final por un error del modelo: {type(exc).__name__}: {exc}"

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer, name="assistant")],
            "steps": add_step(state, "synthesizer", truncate(answer, 200)),
        }

    return synthesizer_node
