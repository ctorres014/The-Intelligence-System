"""
Backends de base de datos vectorial, todos con interfaz 100% async.

- RedisVectorStore    → Redis 8 (RediSearch) vía RedisVL `AsyncSearchIndex`.
                        Opción por defecto: corre local junto al checkpointer.
- PineconeVectorStore → Pinecone Serverless vía `PineconeAsyncio`
                        (continuación de la pre-entrega del Módulo 4).
- InMemoryVectorStore → coseno en memoria (tests / demos sin infraestructura).
- FallbackVectorStore → principal + réplica de failover (p. ej. Pinecone → Redis).

El backend se elige con VECTOR_STORE (y la réplica con VECTOR_STORE_FALLBACK); nombres de índices y namespaces salen
de variables de entorno.
"""

import asyncio
import logging
import math
import re
from abc import ABC, abstractmethod

from app.config import Settings
from app.rag.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

_METADATA_FIELDS = ["id", "text", "doc_id", "source", "title", "section", "category", "chunk_index"]
# RedisVL devuelve la clave de Redis en "id": el ID del chunk se guarda en "chunk_id"
_REDIS_RETURN_FIELDS = ["chunk_id", *_METADATA_FIELDS[1:]]


class VectorStore(ABC):
    name: str

    @abstractmethod
    async def recreate(self, dimensions: int) -> None:
        """Crea (o vacía y recrea) el índice para la dimensión dada."""

    @abstractmethod
    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int: ...

    @abstractmethod
    async def search(self, vector: list[float], k: int) -> list[RetrievedChunk]: ...

    async def search_with_source(self, vector: list[float], k: int) -> tuple[list[RetrievedChunk], str]:
        """Como `search`, pero informa qué backend respondió."""
        return await self.search(vector, k), self.name

    @abstractmethod
    async def count(self) -> int:
        """Cantidad de vectores indexados (0 si el índice no existe)."""

    async def get_fingerprint(self) -> str | None:
        """Huella del modelo de embeddings con el que se construyó el índice."""
        return getattr(self, "_fingerprint", None)

    async def set_fingerprint(self, fingerprint: str) -> None:
        self._fingerprint = fingerprint

    def normalize_fingerprint(self, fingerprint: str) -> str:
        """Forma en que el backend almacena la huella (para compararla)."""
        return fingerprint

    async def close(self) -> None:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Redis (RedisVL)
# ─────────────────────────────────────────────────────────────────────────────


class RedisVectorStore(VectorStore):
    name = "redis"

    def __init__(self, redis_url: str, index_name: str):
        self._redis_url = redis_url
        self._index_name = index_name
        self._index = None
        self._meta_key = f"{index_name}:meta"

    def _schema(self, dimensions: int):
        from redisvl.schema import IndexSchema

        return IndexSchema.from_dict(
            {
                "index": {
                    "name": self._index_name,
                    "prefix": f"{self._index_name}:chunk",
                    "storage_type": "hash",
                },
                "fields": [
                    {"name": "chunk_id", "type": "tag"},
                    {"name": "doc_id", "type": "tag"},
                    {"name": "source", "type": "tag"},
                    {"name": "category", "type": "tag"},
                    {"name": "title", "type": "text"},
                    {"name": "section", "type": "text"},
                    {"name": "text", "type": "text"},
                    {"name": "chunk_index", "type": "numeric"},
                    {
                        "name": "embedding",
                        "type": "vector",
                        "attrs": {
                            "dims": dimensions,
                            "distance_metric": "cosine",
                            "algorithm": "hnsw",
                            "datatype": "float32",
                        },
                    },
                ],
            }
        )

    async def _existing_index(self):
        if self._index is not None:
            return self._index
        from redisvl.index import AsyncSearchIndex

        try:
            self._index = await AsyncSearchIndex.from_existing(
                self._index_name, redis_url=self._redis_url
            )
        except Exception:
            return None
        return self._index

    async def recreate(self, dimensions: int) -> None:
        from redisvl.index import AsyncSearchIndex

        if self._index is not None:
            await self._index.disconnect()
        self._index = AsyncSearchIndex(self._schema(dimensions), redis_url=self._redis_url)
        await self._index.create(overwrite=True, drop=True)

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        from redisvl.redis.utils import array_to_buffer

        index = await self._existing_index()
        if index is None:
            raise RuntimeError("El índice de Redis no existe; ejecutá la ingesta primero")
        records = [
            {
                **chunk.model_dump(exclude={"id"}),
                "chunk_id": chunk.id,
                "embedding": array_to_buffer(vector, dtype="float32"),
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        keys = await index.load(records, id_field="chunk_id")
        return len(keys)

    async def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        from redisvl.query import VectorQuery

        index = await self._existing_index()
        if index is None:
            raise RuntimeError("El índice de Redis no existe; ejecutá la ingesta primero")
        query = VectorQuery(
            vector=vector,
            vector_field_name="embedding",
            return_fields=_REDIS_RETURN_FIELDS,
            num_results=k,
        )
        rows = await index.query(query)
        return [
            RetrievedChunk(
                id=row["chunk_id"],
                **{f: row.get(f, "") for f in _METADATA_FIELDS if f not in ("id", "chunk_index")},
                chunk_index=int(row.get("chunk_index", 0)),
                # RediSearch devuelve distancia coseno (0 = idéntico)
                score=round(1.0 - float(row.get("vector_distance", 1.0)), 4),
                retrievers=["vector"],
            )
            for row in rows
        ]

    async def count(self) -> int:
        index = await self._existing_index()
        if index is None:
            return 0
        info = await index.info()
        return int(info.get("num_docs", 0))

    async def get_fingerprint(self) -> str | None:
        from redis.asyncio import Redis

        async with Redis.from_url(self._redis_url, decode_responses=True) as client:
            return await client.hget(self._meta_key, "embedding_fingerprint")

    async def set_fingerprint(self, fingerprint: str) -> None:
        from redis.asyncio import Redis

        async with Redis.from_url(self._redis_url, decode_responses=True) as client:
            await client.hset(self._meta_key, "embedding_fingerprint", fingerprint)

    async def close(self) -> None:
        if self._index is not None:
            await self._index.disconnect()
            self._index = None


# ─────────────────────────────────────────────────────────────────────────────
# Pinecone (async)
# ─────────────────────────────────────────────────────────────────────────────


class PineconeVectorStore(VectorStore):
    name = "pinecone"

    def __init__(self, api_key: str, index_name: str, namespace: str, cloud: str, region: str):
        self._api_key = api_key
        self._index_name = index_name
        self._namespace = namespace
        self._cloud = cloud
        self._region = region
        self._host: str | None = None

    async def _resolve_host(self) -> str | None:
        if self._host:
            return self._host
        from pinecone import PineconeAsyncio

        async with PineconeAsyncio(api_key=self._api_key) as pc:
            if not await pc.has_index(self._index_name):
                return None
            self._host = (await pc.describe_index(self._index_name)).host
        return self._host

    async def recreate(self, dimensions: int) -> None:
        from pinecone import PineconeAsyncio, ServerlessSpec

        async with PineconeAsyncio(api_key=self._api_key) as pc:
            if await pc.has_index(self._index_name):
                desc = await pc.describe_index(self._index_name)
                if desc.dimension == dimensions:
                    self._host = desc.host
                    async with pc.IndexAsyncio(host=self._host) as idx:
                        try:
                            await idx.delete(delete_all=True, namespace=self._namespace)
                        except Exception:  # namespace inexistente
                            pass
                    return
                logger.warning(
                    "Índice Pinecone '%s' con dimensión %s ≠ %s: se elimina y se recrea",
                    self._index_name, desc.dimension, dimensions,
                )
                await pc.delete_index(self._index_name)
            await pc.create_index(
                name=self._index_name,
                dimension=dimensions,
                metric="cosine",
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
            self._host = (await pc.describe_index(self._index_name)).host

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        from pinecone import PineconeAsyncio

        host = await self._resolve_host()
        if host is None:
            raise RuntimeError("El índice de Pinecone no existe; ejecutá la ingesta primero")
        payload = [
            {"id": c.id, "values": v, "metadata": c.model_dump(exclude={"id"})}
            for c, v in zip(chunks, vectors, strict=True)
        ]
        async with PineconeAsyncio(api_key=self._api_key) as pc:
            async with pc.IndexAsyncio(host=host) as idx:
                for start in range(0, len(payload), 100):
                    await idx.upsert(vectors=payload[start : start + 100], namespace=self._namespace)
        return len(payload)

    async def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        from pinecone import PineconeAsyncio

        host = await self._resolve_host()
        if host is None:
            raise RuntimeError("El índice de Pinecone no existe; ejecutá la ingesta primero")
        async with PineconeAsyncio(api_key=self._api_key) as pc:
            async with pc.IndexAsyncio(host=host) as idx:
                res = await idx.query(
                    vector=vector, top_k=k, namespace=self._namespace, include_metadata=True
                )
        return [
            RetrievedChunk(
                id=m.id,
                **dict(m.metadata or {}),
                score=round(float(m.score), 4),
                retrievers=["vector"],
            )
            for m in res.matches
        ]

    async def count(self) -> int:
        from pinecone import PineconeAsyncio

        host = await self._resolve_host()
        if host is None:
            return 0
        async with PineconeAsyncio(api_key=self._api_key) as pc:
            async with pc.IndexAsyncio(host=host) as idx:
                stats = await idx.describe_index_stats()
        ns = (stats.namespaces or {}).get(self._namespace)
        return int(ns.vector_count) if ns else 0

    @staticmethod
    def _tag(fingerprint: str) -> str:
        # Los tags de Pinecone solo admiten [A-Za-z0-9_-]
        return re.sub(r"[^A-Za-z0-9_-]", "_", fingerprint)[:120]

    async def get_fingerprint(self) -> str | None:
        from pinecone import PineconeAsyncio

        async with PineconeAsyncio(api_key=self._api_key) as pc:
            if not await pc.has_index(self._index_name):
                return None
            tags = (await pc.describe_index(self._index_name)).tags or {}
        return tags.get("embedding")

    async def set_fingerprint(self, fingerprint: str) -> None:
        from pinecone import PineconeAsyncio

        async with PineconeAsyncio(api_key=self._api_key) as pc:
            await pc.configure_index(self._index_name, tags={"embedding": self._tag(fingerprint)})

    def normalize_fingerprint(self, fingerprint: str) -> str:
        return self._tag(fingerprint)


# ─────────────────────────────────────────────────────────────────────────────
# Memoria
# ─────────────────────────────────────────────────────────────────────────────


class InMemoryVectorStore(VectorStore):
    name = "memory"

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    async def recreate(self, dimensions: int) -> None:
        self._items.clear()

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._items[chunk.id] = (chunk, vector)
        return len(chunks)

    async def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
            return dot / norm if norm else 0.0

        scored = sorted(
            ((cosine(vector, v), c) for c, v in self._items.values()),
            key=lambda t: t[0],
            reverse=True,
        )[:k]
        return [
            RetrievedChunk(**c.model_dump(), score=round(s, 4), retrievers=["vector"])
            for s, c in scored
        ]

    async def count(self) -> int:
        return len(self._items)


# ─────────────────────────────────────────────────────────────────────────────
# Principal + réplica de failover
# ─────────────────────────────────────────────────────────────────────────────

# Huella que nunca coincide: fuerza la reingesta cuando las réplicas divergen
_FINGERPRINT_MISMATCH = "mismatch"


class FallbackVectorStore(VectorStore):
    """
    Consulta el backend principal y, si falla o supera el timeout, la réplica.

    La ingesta escribe en ambos para que la réplica esté lista en un failover.
    Un error en la réplica no interrumpe la ingesta (best-effort); si los dos
    fallan en una búsqueda, `KnowledgeBase` degrada a BM25.
    """

    def __init__(self, primary: VectorStore, secondary: VectorStore, timeout: float):
        self.primary = primary
        self.secondary = secondary
        self._timeout = timeout
        self.name = f"{primary.name}+{secondary.name}"

    async def _on_both(self, action: str, *args) -> None:
        primary_res, secondary_res = await asyncio.gather(
            getattr(self.primary, action)(*args), getattr(self.secondary, action)(*args), return_exceptions=True
        )
        if isinstance(primary_res, BaseException):
            raise primary_res
        if isinstance(secondary_res, BaseException):
            logger.warning("Réplica '%s' falló en %s: %s", self.secondary.name, action, secondary_res)

    async def recreate(self, dimensions: int) -> None:
        await self._on_both("recreate", dimensions)

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        await self._on_both("upsert", chunks, vectors)
        return len(chunks)

    async def set_fingerprint(self, fingerprint: str) -> None:
        await self._on_both("set_fingerprint", fingerprint)

    async def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        return (await self.search_with_source(vector, k))[0]

    async def search_with_source(self, vector: list[float], k: int) -> tuple[list[RetrievedChunk], str]:
        try:
            return await asyncio.wait_for(self.primary.search(vector, k), self._timeout), self.primary.name
        except Exception as exc:
            logger.warning(
                "Vector store '%s' no disponible (%s); usando réplica '%s'",
                self.primary.name, type(exc).__name__, self.secondary.name,
            )
        return await self.secondary.search(vector, k), self.secondary.name

    async def count(self) -> int:
        # El mínimo: si la réplica quedó vacía, la auto-ingesta la completa.
        # Si el principal no responde se propaga el error (no se puede reingestar).
        primary_count, secondary_count = await asyncio.gather(
            self.primary.count(), self.secondary.count(), return_exceptions=True
        )
        if isinstance(primary_count, BaseException):
            raise primary_count
        if isinstance(secondary_count, BaseException):
            logger.warning("Réplica '%s' no disponible: %s", self.secondary.name, secondary_count)
            return primary_count
        return min(primary_count, secondary_count)

    async def get_fingerprint(self) -> str | None:
        primary_fp, secondary_fp = await asyncio.gather(
            self.primary.get_fingerprint(), self.secondary.get_fingerprint(), return_exceptions=True
        )
        if isinstance(primary_fp, BaseException):
            # Sin principal no se puede reingestar: se informa la huella de la réplica
            return None if isinstance(secondary_fp, BaseException) else secondary_fp
        if isinstance(secondary_fp, BaseException):
            return primary_fp
        # Cada backend guarda la huella a su manera: se comparan normalizadas
        if self.primary.normalize_fingerprint(primary_fp or "") != self.primary.normalize_fingerprint(secondary_fp or ""):
            return _FINGERPRINT_MISMATCH
        return primary_fp

    def normalize_fingerprint(self, fingerprint: str) -> str:
        return self.primary.normalize_fingerprint(fingerprint)

    async def close(self) -> None:
        await asyncio.gather(self.primary.close(), self.secondary.close(), return_exceptions=True)


def build_vector_store(settings: Settings) -> VectorStore:
    primary = _build_single(settings.vector_store, settings)
    if settings.vector_store_fallback is None:
        return primary
    return FallbackVectorStore(
        primary, _build_single(settings.vector_store_fallback, settings), settings.vector_search_timeout_seconds
    )


def _build_single(kind: str, settings: Settings) -> VectorStore:
    if kind == "pinecone":
        if settings.pinecone_api_key is None:
            raise ValueError("VECTOR_STORE=pinecone requiere PINECONE_API_KEY")
        return PineconeVectorStore(
            api_key=settings.pinecone_api_key.get_secret_value(),
            index_name=settings.pinecone_index_name,
            namespace=settings.pinecone_namespace,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
    if kind == "memory":
        return InMemoryVectorStore()
    return RedisVectorStore(settings.redis_url, settings.redis_index_name)
