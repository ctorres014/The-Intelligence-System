"""Integración con Redis 8: el vector store devuelve los IDs de chunk (no las claves de Redis)."""

import uuid

import pytest
from redis.asyncio import Redis

from app.rag.models import Chunk
from app.rag.vectorstore import RedisVectorStore


@pytest.fixture
async def store(settings):
    client = Redis.from_url(settings.redis_url)
    try:
        await client.ping()
    except Exception:
        pytest.skip(f"Redis no disponible en {settings.redis_url}")
    finally:
        await client.aclose()
    vector_store = RedisVectorStore(settings.redis_url, f"test-kb-{uuid.uuid4().hex[:8]}")
    yield vector_store
    if vector_store._index is not None:
        await vector_store._index.delete(drop=True)
    await vector_store.close()


async def test_redis_roundtrip_preserves_chunk_ids(store):
    chunks = [
        Chunk(id=f"doc-{i:03d}-abc", text=f"texto {i}", doc_id="doc", source="doc.md", chunk_index=i)
        for i in range(3)
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]]
    await store.recreate(dimensions=3)
    assert await store.upsert(chunks, vectors) == 3
    assert await store.count() == 3

    results = await store.search([1.0, 0.0, 0.0], k=2)
    assert [r.id for r in results] == ["doc-000-abc", "doc-002-abc"]
    assert results[0].score == pytest.approx(1.0, abs=1e-3)
    assert results[0].chunk_index == 0 and results[0].retrievers == ["vector"]
