"""
Construcción del grafo multi-agente.

    START ─► supervisor ─┬─► knowledge ──┐
                ▲        ├─► research ───┤
                │        ├─► analyst ────┤
                └────────┴───────────────┘
                         ├─► human_approval ─┬─► analyst
                         │                   └─► synthesizer
                         └─► synthesizer ─► END
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.hitl import human_approval_node, route_after_approval
from app.agents.state import AgentState
from app.agents.supervisor import make_supervisor_node, route_from_supervisor
from app.agents.synthesizer import make_synthesizer_node
from app.agents.workers import make_analyst_node, make_knowledge_node, make_research_node
from app.config import Settings
from app.llm import structured_output_method
from app.rag.knowledge_base import KnowledgeBase
from app.tools import build_analysis_tools, build_knowledge_tool, build_research_tools


def build_graph(
    llm: BaseChatModel,
    kb: KnowledgeBase,
    settings: Settings,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", make_supervisor_node(llm, settings, structured_output_method(settings)))
    workflow.add_node("knowledge", make_knowledge_node(llm, build_knowledge_tool(kb), kb, settings))
    workflow.add_node("research", make_research_node(llm, build_research_tools(settings), settings))
    workflow.add_node("analyst", make_analyst_node(llm, build_analysis_tools(), settings))
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("synthesizer", make_synthesizer_node(llm, settings))

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        ["knowledge", "research", "analyst", "human_approval", "synthesizer"],
    )
    for worker in ("knowledge", "research", "analyst"):
        workflow.add_edge(worker, "supervisor")
    workflow.add_conditional_edges("human_approval", route_after_approval, ["analyst", "synthesizer"])
    workflow.add_edge("synthesizer", END)

    return workflow.compile(checkpointer=checkpointer, name="intelligence_system")
