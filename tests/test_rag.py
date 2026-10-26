from langchain_core.embeddings import DeterministicFakeEmbedding

from app.rag import KnowledgeBase, RetrievedChunk
from app.rag.knowledge_base import reciprocal_rank_fusion, tokenize
from app.rag.loader import chunk_text, load_chunks_sync
from app.rag.vectorstore import InMemoryVectorStore


def _rc(cid: str, retriever: str) -> RetrievedChunk:
    return RetrievedChunk(id=cid, text="t", doc_id="d", source="d.md", chunk_index=0, score=0, retrievers=[retriever])


def test_chunking_is_deterministic(settings):
    _, first = load_chunks_sync(settings.documents_dir, settings.chunk_size, settings.chunk_overlap)
    _, second = load_chunks_sync(settings.documents_dir, settings.chunk_size, settings.chunk_overlap)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)
    assert all(len(c.text) <= settings.chunk_size for c in first)


def test_chunk_metadata():
    chunks = chunk_text("hybrid_search", "hybrid_search.md", "# Título\n\n## Sección A\n\nTexto de prueba.", 600, 100)
    assert chunks[0].title == "Título"
    assert chunks[0].category == "busqueda-hibrida"
    assert chunks[-1].section == "Sección A"


def test_tokenize_normalizes_accents_and_stopwords():
    assert tokenize("¿Qué es la Búsqueda Híbrida?") == ["busqueda", "hibrida"]


def test_rrf_rewards_documents_in_both_lists():
    bm25 = [_rc("a", "bm25"), _rc("b", "bm25")]
    vector = [_rc("b", "vector"), _rc("c", "vector")]
    fused = reciprocal_rank_fusion([(bm25, 0.4), (vector, 0.6)], k=3, rrf_k=60)
    assert fused[0].id == "b"
    assert fused[0].retrievers == ["bm25", "vector"]
    assert {c.id for c in fused} == {"a", "b", "c"}


async def test_hybrid_search_finds_expected_document(kb):
    result = await kb.search("BM25 Reciprocal Rank Fusion búsqueda híbrida", 5)
    assert result.strategy == "hybrid" and not result.degraded
    assert result.chunks[0].doc_id == "hybrid_search"


async def test_degrades_to_bm25_when_vector_store_fails(settings):
    class BrokenStore(InMemoryVectorStore):
        async def search(self, vector, k):
            raise ConnectionError("vector store caído")

    kb = KnowledgeBase(settings, BrokenStore(), DeterministicFakeEmbedding(size=16))
    await kb.load()
    result = await kb.search("checkpointer LangGraph", 3)
    assert result.degraded and result.strategy == "bm25_only"
    assert result.chunks and result.chunks[0].doc_id == "langgraph_agents"


async def test_ensure_ingested_is_idempotent(kb):
    assert await kb.ensure_ingested() is None
    assert await kb.indexed_count() == kb.chunk_count
