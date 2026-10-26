import asyncio

import httpx
import pytest
from fakeredis.aioredis import FakeRedis
from langgraph.checkpoint.memory import InMemorySaver

from app.agents import build_graph
from app.api.jobs import JobRunner, JobStore
from app.api.main import AppContext, app
from app.service import IntelligenceService


@pytest.fixture
async def client(llm, kb, settings):
    graph = build_graph(llm, kb, settings, InMemorySaver())
    service = IntelligenceService(graph)
    redis = FakeRedis(decode_responses=True)
    app.state.ctx = AppContext(
        settings=settings, redis=redis, redis_ok=True, checkpointer_name="InMemorySaver",
        kb=kb, service=service, jobs=JobRunner(JobStore(redis, 60), service), tracing="none",
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_health(client):
    body = (await client.get("/health")).json()
    assert body["status"] == "ok"
    assert body["indexed_vectors"] == body["knowledge_chunks"] > 0


@pytest.mark.parametrize(
    "payload",
    [{}, {"message": ""}, {"message": "hola", "thread_id": "con espacios"}, {"message": "x" * 4001}],
)
async def test_chat_validates_input(client, payload):
    r = await client.post("/v1/chat", json=payload)
    assert r.status_code == 422


async def test_chat_and_thread_history(client, llm):
    llm.decisions = ["FINISH"]
    r = await client.post("/v1/chat", json={"message": "¿Qué es RRF?", "thread_id": "api-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed" and body["answer"]
    assert body["agents_used"] == ["knowledge"]
    assert body["sources"] and body["sources"][0]["ref"].startswith("[")
    assert r.headers["x-request-id"]

    llm.decisions = ["FINISH"]
    await client.post("/v1/chat", json={"message": "¿Y BM25?", "thread_id": "api-1"})
    thread = (await client.get("/v1/threads/api-1")).json()
    assert [m["role"] for m in thread["messages"]] == ["user", "assistant", "user", "assistant"]


async def test_unknown_thread_returns_404(client):
    r = await client.get("/v1/threads/no-existe", headers={"x-request-id": "req-123"})
    assert r.status_code == 404
    assert r.json()["error"] == "not_found" and r.json()["request_id"] == "req-123"


async def test_hitl_flow_over_api(client, llm):
    llm.decisions = ["FINISH"]
    r = (await client.post("/v1/chat", json={"message": "Borrar el índice de producción", "thread_id": "api-2"})).json()
    assert r["status"] == "awaiting_approval" and r["answer"] is None
    assert r["pending_approval"]["planned_action"] == "FINISH"

    blocked = await client.post("/v1/chat", json={"message": "otra cosa", "thread_id": "api-2"})
    assert blocked.status_code == 409 and blocked.json()["error"] == "approval_state"

    done = (await client.post("/v1/threads/api-2/approve", json={"approved": False, "comment": "no"})).json()
    assert done["status"] == "rejected"
    again = await client.post("/v1/threads/api-2/approve", json={"approved": True})
    assert again.status_code == 409


async def test_background_task(client, llm):
    llm.decisions = ["FINISH"]
    r = await client.post("/v1/tasks", json={"message": "¿Qué es un checkpointer?"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    for _ in range(50):
        job = (await client.get(f"/v1/tasks/{job_id}")).json()
        if job["status"] == "completed":
            break
        await asyncio.sleep(0.05)
    assert job["status"] == "completed"
    assert job["result"]["answer"]


async def test_stream_emits_node_events(client, llm):
    llm.decisions = ["FINISH"]
    async with client.stream("POST", "/v1/chat/stream", json={"message": "¿Qué es RRF?"}) as r:
        text = "".join([chunk async for chunk in r.aiter_text()])
    events = [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("event: ")]
    assert events[0] == "start" and events[-1] == "final"
    assert events.count("step") == 4  # supervisor, knowledge, supervisor, synthesizer


async def test_knowledge_search_endpoint(client):
    body = (await client.post("/v1/knowledge/search", json={"query": "namespaces Pinecone", "k": 3})).json()
    assert len(body["chunks"]) == 3
    assert (await client.post("/v1/knowledge/search", json={"query": "x", "k": 3})).status_code == 422
