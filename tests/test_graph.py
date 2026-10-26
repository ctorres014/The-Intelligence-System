from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agents import build_graph, new_turn_input
from app.agents.supervisor import is_critical_query, route_from_supervisor, rule_based_decision


def _config(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def test_critical_query_detection():
    assert is_critical_query("Necesito borrar el índice de producción")
    assert is_critical_query("deploy the new version")
    assert not is_critical_query("¿Qué es un checkpointer?")


def test_routing_intercepts_critical_tasks():
    base = {"is_critical": True, "is_approved": None}
    assert route_from_supervisor({**base, "next_agent": "analyst"}) == "human_approval"
    assert route_from_supervisor({**base, "next_agent": "FINISH"}) == "human_approval"
    assert route_from_supervisor({**base, "next_agent": "knowledge"}) == "knowledge"
    assert route_from_supervisor({**base, "is_approved": True, "next_agent": "FINISH"}) == "synthesizer"
    assert route_from_supervisor({"next_agent": "FINISH"}) == "synthesizer"


def test_rule_based_fallback():
    assert rule_based_decision({"user_query": "q", "steps": []}).next == "knowledge"
    knowledge_done = {"user_query": "calculá el costo", "steps": [{"node": "knowledge"}]}
    assert rule_based_decision(knowledge_done).next == "analyst"


async def test_full_turn_rag_then_analyst(llm, kb, settings):
    llm.decisions = ["analyst", "FINISH"]
    graph = build_graph(llm, kb, settings, InMemorySaver())
    final = await graph.ainvoke(new_turn_input("¿Cuánto ocupan 2M de vectores de 1536 dims?"), _config("t1"))

    nodes = [s["node"] for s in final["steps"]]
    assert nodes == ["supervisor", "knowledge", "supervisor", "analyst", "supervisor", "synthesizer"]
    assert final["sources"], "el Knowledge Agent debe exponer las fuentes del RAG"
    assert "11.44" in final["analysis_result"]  # salida de math_calculator
    assert isinstance(final["messages"][-1], AIMessage)
    assert final["final_answer"]


async def test_supervisor_fallback_when_llm_fails(llm, kb, settings):
    llm.decisions = []  # with_structured_output lanza error → reglas
    graph = build_graph(llm, kb, settings, InMemorySaver())
    final = await graph.ainvoke(new_turn_input("¿Qué es RRF?"), _config("t2"))
    assert final["steps"][-1]["node"] == "synthesizer"
    assert any("fallback" in s["summary"] for s in final["steps"])


async def test_iteration_guard_always_terminates(llm, kb, settings):
    llm.decisions = ["knowledge"] * 20
    graph = build_graph(llm, kb, settings, InMemorySaver())
    final = await graph.ainvoke(new_turn_input("loop"), _config("t3"))
    assert final["steps"][-1]["node"] == "synthesizer"
    assert sum(s["node"] == "knowledge" for s in final["steps"]) <= 2


async def test_multi_turn_memory(llm, kb, settings):
    graph = build_graph(llm, kb, settings, InMemorySaver())
    llm.decisions = ["FINISH"]
    await graph.ainvoke(new_turn_input("¿Qué es BM25?"), _config("t4"))
    llm.decisions = ["FINISH"]
    final = await graph.ainvoke(new_turn_input("¿Y RRF?"), _config("t4"))

    kinds = [type(m) for m in final["messages"]]
    assert kinds == [HumanMessage, AIMessage, HumanMessage, AIMessage]
    assert final["user_query"] == "¿Y RRF?"
    assert final["steps"][0]["node"] == "supervisor"  # el espacio de trabajo se reinicia por turno


async def test_hitl_interrupt_and_approve(llm, kb, settings):
    llm.decisions = ["analyst", "FINISH"]
    graph = build_graph(llm, kb, settings, InMemorySaver())
    cfg = _config("t5")
    await graph.ainvoke(new_turn_input("Borrar el índice de producción y reingestar"), cfg)

    snapshot = await graph.aget_state(cfg)
    assert snapshot.next == ("human_approval",)
    assert snapshot.interrupts[0].value["planned_action"] == "analyst"

    final = await graph.ainvoke(Command(resume={"approved": True, "comment": "ok"}), cfg)
    assert final["is_approved"] is True
    assert final["analysis_result"] and final["final_answer"]


async def test_hitl_reject(llm, kb, settings):
    llm.decisions = ["FINISH"]
    graph = build_graph(llm, kb, settings, InMemorySaver())
    cfg = _config("t6")
    await graph.ainvoke(new_turn_input("Deploy a producción eliminando el namespace"), cfg)
    final = await graph.ainvoke(Command(resume={"approved": False}), cfg)
    assert final["is_approved"] is False
    assert "rechaz" in final["final_answer"]
    assert final.get("analysis_result") is None


def test_enforce_coverage_adds_missing_explicit_agents():
    from app.agents.state import SupervisorDecision
    from app.agents.supervisor import enforce_coverage

    finish = SupervisorDecision(next="FINISH", reasoning="listo")
    knowledge_only = [{"node": "knowledge", "summary": ""}]
    calc = {"user_query": "Calculá cuántos GB ocupan 2M de vectores", "steps": knowledge_only}
    assert enforce_coverage(calc, finish).next == "analyst"
    ext = {"user_query": "¿Quiénes desarrollaron BM25?", "steps": knowledge_only}
    assert enforce_coverage(ext, finish).next == "research"
    plain = {"user_query": "¿Qué es RRF?", "steps": knowledge_only}
    assert enforce_coverage(plain, finish).next == "FINISH"
    done = {"user_query": "Calculá el costo", "steps": knowledge_only + [{"node": "analyst", "summary": ""}]}
    assert enforce_coverage(done, finish).next == "FINISH"
