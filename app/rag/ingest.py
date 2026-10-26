"""
CLI de ingesta: `python -m app.rag.ingest`

Carga `data/documents/`, fragmenta, genera embeddings y (re)crea el índice en
el vector store configurado con VECTOR_STORE.
"""

import asyncio
import json
import logging

from app.config import get_settings
from app.llm import build_embeddings
from app.rag.knowledge_base import KnowledgeBase
from app.rag.vectorstore import build_vector_store


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s  %(message)s")
    settings = get_settings()
    kb = KnowledgeBase(settings, build_vector_store(settings), build_embeddings(settings))
    try:
        report = await kb.ingest()
        print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))
    finally:
        await kb.close()


if __name__ == "__main__":
    asyncio.run(main())
