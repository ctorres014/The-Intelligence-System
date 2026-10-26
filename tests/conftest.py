"""
Fixtures compartidas.

`ScriptedChatModel` es un chat model determinístico que soporta `bind_tools`
y `with_structured_output`, de modo que el grafo completo (Supervisor,
agentes ReAct, HITL, Synthesizer) se ejecuta sin un LLM real.
"""

from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.agents.state import SupervisorDecision
from app.config import Settings
from app.rag import KnowledgeBase, build_vector_store

TOOL_ARGS = {
    "knowledge_base_search": {"query": "reciprocal rank fusion", "k": 3},
    "math_calculator": {"expression": "2_000_000 * 1536 * 4 / 1024**3"},
}


class ScriptedChatModel(BaseChatModel):
    decisions: list[str] = []
    bound_tools: list[str] = []
    calls: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"bound_tools": [t.name for t in tools]})

    def with_structured_output(self, schema, **kwargs):
        def decide(_messages: Any) -> SupervisorDecision:
            if not self.decisions:
                raise ValueError("sin decisiones guionadas")
            nxt = self.decisions.pop(0)
            self.calls.append(f"supervisor:{nxt}")
            return SupervisorDecision(next=nxt, reasoning=f"guion → {nxt}")

        return RunnableLambda(decide)

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        last = messages[-1]
        if isinstance(last, ToolMessage):
            msg = AIMessage(content=f"Hallazgos de {last.name}: {str(last.content)[:120]}")
        else:
            tool = next((t for t in self.bound_tools if t in TOOL_ARGS), None)
            if tool:
                self.calls.append(f"tool:{tool}")
                msg = AIMessage(content="", tool_calls=[{"name": tool, "args": TOOL_ARGS[tool], "id": f"call_{tool}"}])
            else:
                self.calls.append("synthesizer")
                msg = AIMessage(content="Respuesta final basada en [hybrid_search#1].")
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        vector_store="memory",
        phoenix_enabled=False,
        max_iterations=8,
        redis_url="redis://localhost:6380",
    )


@pytest.fixture
async def kb(settings) -> KnowledgeBase:
    knowledge_base = KnowledgeBase(settings, build_vector_store(settings), DeterministicFakeEmbedding(size=64))
    await knowledge_base.ingest()
    return knowledge_base


@pytest.fixture
def llm() -> ScriptedChatModel:
    return ScriptedChatModel(decisions=[], calls=[])
