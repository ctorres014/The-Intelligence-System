"""
Carga y fragmentación de documentos.

El chunking es determinístico (mismo documento → mismos `chunk_id`), lo que
permite que el índice BM25 en memoria y el índice vectorial compartan IDs y
puedan fusionarse por Reciprocal Rank Fusion.
"""

import asyncio
import hashlib
import re
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.rag.models import Chunk

SUPPORTED_SUFFIXES = {".md", ".txt"}

CATEGORY_MAP = {
    "rag_fundamentals": "fundamentos-rag",
    "vector_databases": "bases-de-datos-vectoriales",
    "embeddings_and_chunking": "embeddings-chunking",
    "hybrid_search": "busqueda-hibrida",
    "langchain_overview": "langchain",
    "langgraph_agents": "agentes",
    "observability": "observabilidad",
    "production_apis": "apis-produccion",
}


def _read_documents(directory: Path) -> list[tuple[Path, str]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"No existe el directorio de documentos: {directory}")
    return [
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(directory.iterdir())
        if path.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def _title_of(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


def chunk_text(
    doc_id: str, source: str, text: str, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Divide primero por encabezados Markdown y luego por tamaño."""
    title = _title_of(text, doc_id)
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for section_doc in header_splitter.split_text(text):
        section = section_doc.metadata.get("h2") or section_doc.metadata.get("h1") or ""
        for piece in size_splitter.split_text(section_doc.page_content):
            piece = piece.strip()
            if not piece:
                continue
            index = len(chunks)
            digest = hashlib.sha1(piece.encode("utf-8")).hexdigest()[:8]
            chunks.append(
                Chunk(
                    id=f"{doc_id}-{index:03d}-{digest}",
                    text=piece,
                    doc_id=doc_id,
                    source=source,
                    title=title,
                    section=section,
                    category=CATEGORY_MAP.get(doc_id, "general"),
                    chunk_index=index,
                )
            )
    return chunks


def load_chunks_sync(directory: Path, chunk_size: int, chunk_overlap: int) -> tuple[int, list[Chunk]]:
    docs = _read_documents(directory)
    chunks: list[Chunk] = []
    for path, text in docs:
        chunks.extend(chunk_text(path.stem, path.name, text, chunk_size, chunk_overlap))
    return len(docs), chunks


async def load_chunks(directory: Path, chunk_size: int, chunk_overlap: int) -> tuple[int, list[Chunk]]:
    """Versión async: la lectura de disco se delega a un thread."""
    return await asyncio.to_thread(load_chunks_sync, directory, chunk_size, chunk_overlap)
