"""
Integración con Redis real: la conversación sobrevive a un "reinicio".

Se simula el reinicio cerrando el checkpointer y el grafo, y creando otros
nuevos (como haría un proceso de la API recién arrancado). Se omite si no hay
Redis 8 accesible en REDIS_URL.
"""

import uuid

import pytest
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.types import Command
from redis.asyncio import Redis

from app.agents import build_graph, new_turn_input


@pytest.fixture
async def redis_url(settings):
    client = Redis.from_url(settings.redis_url)
    try:
        await client.ping()
    except Exception:
        pytest.skip(f"Redis no disponible en {settings.redis_url}")
    finally:
        await client.aclose()
    return settings.redis_url


async def test_conversation_survives_restart(llm, kb, settings, redis_url):
    cfg = {"configurable": {"thread_id": f"persist-{uuid.uuid4().hex[:8]}"}}

    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        llm.decisions = ["FINISH"]
        await build_graph(llm, kb, settings, saver).ainvoke(new_turn_input("¿Qué es BM25?"), cfg)

    # ── "reinicio": nuevo checkpointer + nuevo grafo ──
    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        graph = build_graph(llm, kb, settings, saver)
        restored = await graph.aget_state(cfg)
        assert len(restored.values["messages"]) == 2

        llm.decisions = ["FINISH"]
        final = await graph.ainvoke(new_turn_input("¿Y RRF?"), cfg)
        assert len(final["messages"]) == 4


async def test_pending_approval_survives_restart(llm, kb, settings, redis_url):
    cfg = {"configurable": {"thread_id": f"hitl-{uuid.uuid4().hex[:8]}"}}

    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        await saver.asetup()
        llm.decisions = ["FINISH"]
        await build_graph(llm, kb, settings, saver).ainvoke(new_turn_input("Borrar el índice de producción"), cfg)

    async with AsyncRedisSaver.from_conn_string(redis_url) as saver:
        graph = build_graph(llm, kb, settings, saver)
        assert (await graph.aget_state(cfg)).interrupts, "la aprobación pendiente debe persistir"
        final = await graph.ainvoke(Command(resume={"approved": True}), cfg)
        assert final["is_approved"] is True and final["final_answer"]
