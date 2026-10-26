"""Herramienta RAG: búsqueda híbrida en la base de conocimiento interna."""

from langchain_core.tools import BaseTool, StructuredTool

from app.rag.knowledge_base import KnowledgeBase
from app.tools.schemas import KnowledgeSearchInput, KnowledgeSearchOutput, SourceSnippet

KNOWLEDGE_TOOL_NAME = "knowledge_base_search"


def build_knowledge_tool(kb: KnowledgeBase) -> BaseTool:
    async def knowledge_base_search(query: str, k: int = 5) -> str:
        result = await kb.search(query, k)
        output = KnowledgeSearchOutput(
            query=result.query,
            strategy=result.strategy,
            degraded=result.degraded,
            warning=result.warning,
            results=[
                SourceSnippet(
                    ref=f"[{c.doc_id}#{c.chunk_index}]",
                    source=c.source,
                    doc_id=c.doc_id,
                    section=c.section,
                    score=c.score,
                    retrievers=list(c.retrievers),
                    text=c.text,
                )
                for c in result.chunks
            ],
        )
        return output.model_dump_json()

    return StructuredTool.from_function(
        coroutine=knowledge_base_search,
        name=KNOWLEDGE_TOOL_NAME,
        description=(
            "Busca en la base de conocimiento INTERNA (documentación de arquitectura de IA: "
            "RAG, chunking, embeddings, bases vectoriales, Pinecone, búsqueda híbrida, LangChain, "
            "LangGraph, agentes, observabilidad, APIs de producción). Devuelve fragmentos con "
            "su referencia para citar, ej. [vector_databases#2]."
        ),
        args_schema=KnowledgeSearchInput,
    )
