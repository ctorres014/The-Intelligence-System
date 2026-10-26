"""Failover entre vector stores (p. ej. Pinecone → Redis → BM25), sin red."""

import asyncio

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from pydantic import ValidationError

from app.config import Settings
from app.rag import KnowledgeBase
from app.rag.vectorstore import FallbackVectorStore, InMemoryVectorStore, build_vector_store


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, phoenix_enabled=False, **overrides)


class BrokenStore(InMemoryVectorStore):
    """Principal caído: falla (o se cuelga) en la búsqueda."""

    name = "broken"

    def __init__(self, hang: bool = False):
        super().__init__()
        self._hang = hang

    async def search(self, vector, k):
        if self._hang:
            await asyncio.sleep(10)
        raise ConnectionError("principal caído")


async def _kb(store) -> KnowledgeBase:
    kb = KnowledgeBase(_settings(), store, DeterministicFakeEmbedding(size=16))
    await kb.ingest()
    return kb


async def test_primary_serves_when_healthy():
    primary, secondary = InMemoryVectorStore(), InMemoryVectorStore()
    primary.name = "pinecone"
    kb = await _kb(FallbackVectorStore(primary, secondary, timeout=5))
    result = await kb.search("búsqueda híbrida")
    assert result.vector_backend == "pinecone" and result.strategy == "hybrid"


@pytest.mark.parametrize("hang", [False, True])
async def test_failover_to_replica(hang):
    secondary = InMemoryVectorStore()
    kb = await _kb(FallbackVectorStore(BrokenStore(hang=hang), secondary, timeout=0.2))
    result = await kb.search("búsqueda híbrida")
    assert result.vector_backend == "memory"
    assert result.strategy == "hybrid" and not result.degraded


async def test_ingest_writes_both_stores():
    primary, secondary = InMemoryVectorStore(), InMemoryVectorStore()
    kb = await _kb(FallbackVectorStore(primary, secondary, timeout=5))
    assert await primary.count() == await secondary.count() == len(kb._chunks) > 0


async def test_empty_replica_triggers_reingest():
    primary, secondary = InMemoryVectorStore(), InMemoryVectorStore()
    kb = await _kb(FallbackVectorStore(primary, secondary, timeout=5))
    await secondary.recreate(16)  # la réplica se vació
    assert await kb.ensure_ingested() is not None
    assert await secondary.count() == len(kb._chunks)


async def test_diverging_fingerprints_trigger_reingest():
    primary, secondary = InMemoryVectorStore(), InMemoryVectorStore()
    kb = await _kb(FallbackVectorStore(primary, secondary, timeout=5))
    await secondary.set_fingerprint("otro:modelo:768")
    assert await kb.ensure_ingested() is not None
    assert await secondary.get_fingerprint() == await primary.get_fingerprint()


def test_builder_wraps_with_fallback():
    store = build_vector_store(
        _settings(vector_store="pinecone", vector_store_fallback="redis", pinecone_api_key="pc-test")
    )
    assert isinstance(store, FallbackVectorStore)
    assert store.name == "pinecone+redis"
    assert type(build_vector_store(_settings(vector_store="redis"))).__name__ == "RedisVectorStore"


def test_fallback_must_differ_from_primary():
    with pytest.raises(ValidationError, match="VECTOR_STORE_FALLBACK"):
        _settings(vector_store="redis", vector_store_fallback="redis")
