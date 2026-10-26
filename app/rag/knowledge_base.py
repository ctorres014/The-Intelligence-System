"""
Base de conocimiento con recuperación híbrida (BM25 + vectorial) y RRF.

Evolución del `EnsembleRetriever` de la pre-entrega del Módulo 4, reescrito
como servicio async:

    consulta ──┬──► BM25 (léxico, en memoria) ─────────┐
               └──► embeddings ─► VectorStore (async) ─┴─► Reciprocal Rank Fusion ─► top-k

Si el vector store no responde, el sistema degrada a BM25 en lugar de fallar
y lo informa en `RetrievalResult.degraded`.
"""

import asyncio
import logging
import re
import time
import unicodedata

from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from app.config import Settings
from app.rag.loader import load_chunks
from app.rag.models import Chunk, IngestReport, RetrievalResult, RetrievedChunk
from app.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

_STOPWORDS = {
    # español
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a", "en",
    "y", "o", "que", "es", "son", "por", "para", "con", "sin", "se", "su", "sus", "lo",
    "como", "mas", "pero", "sobre", "entre", "cual", "cuales", "qué", "que", "este",
    "esta", "estos", "estas", "ese", "esa", "hay", "muy", "ya", "le", "les", "me", "mi",
    # inglés
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "for", "with", "on",
    "what", "how", "which", "this", "that", "it", "be", "by", "as",
}


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return [t for t in re.findall(r"[a-z0-9]+", normalized) if t not in _STOPWORDS and len(t) > 1]


def reciprocal_rank_fusion(
    ranked_lists: list[tuple[list[RetrievedChunk], float]], k: int, rrf_k: int
) -> list[RetrievedChunk]:
    """RRF ponderado: score(d) = Σ w_i / (rrf_k + rank_i(d))."""
    fused: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}
    for results, weight in ranked_lists:
        for rank, chunk in enumerate(results, start=1):
            fused[chunk.id] = fused.get(chunk.id, 0.0) + weight / (rrf_k + rank)
            if chunk.id in by_id:
                merged = sorted(set(by_id[chunk.id].retrievers) | set(chunk.retrievers))
                by_id[chunk.id] = by_id[chunk.id].model_copy(update={"retrievers": merged})
            else:
                by_id[chunk.id] = chunk
    ordered = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
    return [by_id[cid].model_copy(update={"score": round(fused[cid], 6)}) for cid in ordered]


class KnowledgeBase:
    def __init__(self, settings: Settings, store: VectorStore, embeddings: Embeddings):
        self._settings = settings
        self._store = store
        self._embeddings = embeddings
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._documents = 0

    @property
    def store_name(self) -> str:
        return self._store.name

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    async def indexed_count(self) -> int:
        return await self._store.count()

    async def load(self) -> None:
        """Carga y fragmenta los documentos locales y construye el índice BM25."""
        s = self._settings
        self._documents, self._chunks = await load_chunks(s.documents_dir, s.chunk_size, s.chunk_overlap)
        corpus = [tokenize(f"{c.title} {c.section} {c.text}") for c in self._chunks]
        self._bm25 = await asyncio.to_thread(BM25Okapi, corpus) if corpus else None
        logger.info("BM25 listo: %d documentos, %d chunks", self._documents, len(self._chunks))

    async def ensure_ingested(self) -> IngestReport | None:
        """Ingesta automática si el índice vectorial está vacío o incompleto."""
        current = self._store.normalize_fingerprint(self._settings.embedding_fingerprint)
        try:
            indexed = await self._store.count()
            stored = await self._store.get_fingerprint()
        except Exception as exc:
            logger.warning("No se pudo consultar el vector store (%s); se omite la auto-ingesta", exc)
            return None
        if indexed and stored is not None and stored != current:
            logger.info("El modelo de embeddings cambió (%s → %s): recreando el índice", stored, current)
            return await self.ingest()
        if indexed >= len(self._chunks) > 0:
            logger.info("Vector store '%s' ya contiene %d vectores", self._store.name, indexed)
            return None
        logger.info("Vector store con %d/%d vectores: ejecutando ingesta", indexed, len(self._chunks))
        return await self.ingest()

    async def ingest(self) -> IngestReport:
        t0 = time.perf_counter()
        await self.load()
        if not self._chunks:
            raise ValueError(f"No hay documentos en {self._settings.documents_dir}")

        texts = [f"{c.title}\n{c.section}\n{c.text}" for c in self._chunks]
        vectors = await self._embeddings.aembed_documents(texts)
        dimensions = len(vectors[0])

        await self._store.recreate(dimensions)
        await self._store.upsert(self._chunks, vectors)
        await self._store.set_fingerprint(self._settings.embedding_fingerprint)

        report = IngestReport(
            documents=self._documents,
            chunks=len(self._chunks),
            vector_store=self._store.name,
            embedding_model=self._settings.embedding_model_name,
            dimensions=dimensions,
            seconds=round(time.perf_counter() - t0, 2),
        )
        logger.info("Ingesta completada: %s", report.model_dump())
        return report

    def _bm25_search(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            RetrievedChunk(**self._chunks[i].model_dump(), score=round(float(scores[i]), 4), retrievers=["bm25"])
            for i in ranked
            if scores[i] > 0
        ]

    async def _vector_search(self, query: str, k: int) -> tuple[list[RetrievedChunk], str]:
        vector = await self._embeddings.aembed_query(query)
        return await self._store.search_with_source(vector, k)

    async def search(self, query: str, k: int | None = None) -> RetrievalResult:
        s = self._settings
        k = k or s.retrieval_top_k
        candidates = k * 3  # se recuperan más candidatos de cada lado antes de fusionar

        bm25_task = asyncio.to_thread(self._bm25_search, query, candidates)
        vector_task = self._vector_search(query, candidates)
        bm25_res, vector_res = await asyncio.gather(bm25_task, vector_task, return_exceptions=True)

        if isinstance(bm25_res, BaseException):
            logger.warning("BM25 falló: %s", bm25_res)
            bm25_res = []
        if isinstance(vector_res, BaseException):
            logger.warning("Búsqueda vectorial falló, degradando a BM25: %s", vector_res)
            return RetrievalResult(
                query=query,
                chunks=bm25_res[:k],
                strategy="bm25_only",
                degraded=True,
                warning=f"vector store no disponible: {vector_res}",
            )

        vector_chunks, backend = vector_res
        fused = reciprocal_rank_fusion(
            [(bm25_res, s.bm25_weight), (vector_chunks, s.vector_weight)], k=k, rrf_k=s.rrf_k
        )
        return RetrievalResult(query=query, chunks=fused, strategy="hybrid", vector_backend=backend)

    async def close(self) -> None:
        await self._store.close()
