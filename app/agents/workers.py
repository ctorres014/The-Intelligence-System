"""
Agentes especialistas (Knowledge, Research, Analyst).

Cada uno es un agente ReAct (`langchain.agents.create_agent`) compilado UNA
sola vez al construir el grafo y ejecutado con `ainvoke` dentro de su nodo.
El nodo traduce entre el estado global y la conversación privada del agente,
de modo que los mensajes internos (tool calls) no contaminan el historial
visible del usuario.
"""

import json
import logging
from typing import Any, Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.agents.common import format_history, last_ai_text, message_text, truncate
from app.agents.prompts import ANALYST_PROMPT, KNOWLEDGE_PROMPT, RESEARCH_PROMPT
from app.agents.state import AgentState, add_step
from app.config import Settings
from app.rag.knowledge_base import KnowledgeBase
from app.tools import KNOWLEDGE_TOOL_NAME
from app.tools.schemas import KnowledgeSearchOutput, SourceSnippet

logger = logging.getLogger(__name__)

AGENT_RECURSION_LIMIT = 12


def _task_message(state: AgentState, settings: Settings, extra_context: str = "") -> HumanMessage:
    parts = [f"Consulta del usuario: {state.get('user_query', '')}"]
    if state.get("instructions"):
        parts.append(f"Instrucción del Supervisor: {state['instructions']}")
    history = format_history(state, settings.history_window)
    if history:
        parts.append(f"Historial de la conversación (para resolver referencias):\n{history}")
    if extra_context:
        parts.append(extra_context)
    return HumanMessage(content="\n\n".join(parts))


async def _run_agent(agent: Any, message: HumanMessage, name: str) -> tuple[str, list]:
    result = await agent.ainvoke(
        {"messages": [message]},
        config={"recursion_limit": AGENT_RECURSION_LIMIT, "run_name": f"{name}_agent"},
    )
    messages = result.get("messages", [])
    return last_ai_text(messages), messages


def _extract_sources(messages: list) -> list[SourceSnippet]:
    """Recolecta (sin duplicados) los fragmentos devueltos por knowledge_base_search."""
    sources: dict[str, SourceSnippet] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == KNOWLEDGE_TOOL_NAME:
            try:
                output = KnowledgeSearchOutput.model_validate_json(message_text(msg))
            except (ValidationError, json.JSONDecodeError):
                continue
            for snippet in output.results:
                if snippet.ref not in sources or snippet.score > sources[snippet.ref].score:
                    sources[snippet.ref] = snippet
    return list(sources.values())


def _format_snippets(snippets: list[SourceSnippet]) -> str:
    return "\n\n".join(f"{s.ref} ({s.source} · {s.section})\n{s.text}" for s in snippets)


def make_knowledge_node(llm: BaseChatModel, tool: BaseTool, kb: KnowledgeBase, settings: Settings) -> Callable:
    agent = create_agent(llm, [tool], system_prompt=KNOWLEDGE_PROMPT, name="knowledge_agent")

    async def direct_retrieval(query: str) -> list[SourceSnippet]:
        result = await kb.search(query)
        return [
            SourceSnippet(
                ref=f"[{c.doc_id}#{c.chunk_index}]", source=c.source, doc_id=c.doc_id,
                section=c.section, score=c.score, retrievers=list(c.retrievers), text=c.text,
            )
            for c in result.chunks
        ]

    async def knowledge_node(state: AgentState) -> dict:
        query = state.get("user_query", "")
        try:
            summary, messages = await _run_agent(agent, _task_message(state, settings), "knowledge")
            snippets = _extract_sources(messages)
            if not snippets:
                # El modelo no llamó a la herramienta: recuperación directa como red de seguridad
                logger.info("Knowledge Agent no usó la herramienta; ejecutando recuperación directa")
                snippets, summary = await direct_retrieval(query), ""
        except Exception as exc:
            # La recuperación no depende del LLM: si el modelo falla, el contexto igual llega
            logger.exception("Knowledge Agent falló; ejecutando recuperación directa")
            try:
                snippets, summary = await direct_retrieval(query), ""
            except Exception:
                logger.exception("Recuperación directa falló")
                return {"steps": add_step(state, "knowledge", f"error: {type(exc).__name__}: {exc}")}

        context = (summary + "\n\nFragmentos recuperados:\n" + _format_snippets(snippets)).strip()
        sources = [s.model_dump(exclude={"text"}) | {"preview": truncate(s.text, 240)} for s in snippets]
        return {
            "knowledge_context": context,
            "sources": sources,
            "steps": add_step(state, "knowledge", f"{len(snippets)} fragmentos recuperados. {summary}"),
        }

    return knowledge_node


def make_research_node(llm: BaseChatModel, tools: list[BaseTool], settings: Settings) -> Callable:
    agent = create_agent(llm, tools, system_prompt=RESEARCH_PROMPT, name="research_agent")

    async def research_node(state: AgentState) -> dict:
        try:
            output, _ = await _run_agent(agent, _task_message(state, settings), "research")
        except Exception as exc:
            logger.exception("Research Agent falló")
            output = f"No se pudo completar la investigación externa: {type(exc).__name__}: {exc}"
        return {
            "research_data": output or "Sin hallazgos externos.",
            "steps": add_step(state, "research", output or "sin hallazgos"),
        }

    return research_node


def make_analyst_node(llm: BaseChatModel, tools: list[BaseTool], settings: Settings) -> Callable:
    agent = create_agent(llm, tools, system_prompt=ANALYST_PROMPT, name="analyst_agent")

    async def analyst_node(state: AgentState) -> dict:
        context_parts = []
        if state.get("knowledge_context"):
            context_parts.append("Contexto interno:\n" + truncate(state["knowledge_context"], 6000))
        if state.get("research_data"):
            context_parts.append("Investigación externa:\n" + truncate(state["research_data"], 4000))
        if state.get("approval_comment"):
            context_parts.append(f"Comentario del aprobador humano: {state['approval_comment']}")
        try:
            output, _ = await _run_agent(
                agent, _task_message(state, settings, "\n\n".join(context_parts)), "analyst"
            )
        except Exception as exc:
            logger.exception("Analyst Agent falló")
            output = f"No se pudo completar el análisis: {type(exc).__name__}: {exc}"
        return {
            "analysis_result": output or "Sin resultados de análisis.",
            "steps": add_step(state, "analyst", output or "sin resultados"),
        }

    return analyst_node
