"""Utilidades compartidas por los nodos del grafo."""

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from app.agents.state import AgentState


def message_text(message: AnyMessage) -> str:
    """Texto plano de un mensaje (soporta contenido en bloques, ej. Anthropic)."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def last_ai_text(messages: list[AnyMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            text = message_text(msg).strip()
            if text:
                return text
    return ""


def format_history(state: AgentState, window: int) -> str:
    """Historial previo al turno actual (excluye el último mensaje del usuario)."""
    previous = state.get("messages", [])[:-1]
    if window <= 0 or not previous:
        return ""
    lines = []
    for msg in previous[-window:]:
        role = "Usuario" if isinstance(msg, HumanMessage) else "Asistente"
        lines.append(f"{role}: {truncate(message_text(msg), 800)}")
    return "\n".join(lines)


def truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    return text if len(text) <= limit else text[:limit] + "…"
