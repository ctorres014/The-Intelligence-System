"""
Herramientas de investigación externa (async).

- web_search       → Tavily (cliente async del SDK vía langchain-tavily).
- wikipedia_search → API REST de MediaWiki con `httpx.AsyncClient`
                     (reemplaza al paquete `wikipedia`, que es síncrono).
"""

import logging

import httpx
from langchain_core.tools import BaseTool, StructuredTool

from app.config import Settings
from app.tools.schemas import (
    WebResult,
    WebSearchInput,
    WebSearchOutput,
    WikipediaOutput,
    WikipediaSearchInput,
)

logger = logging.getLogger(__name__)


def build_research_tools(settings: Settings) -> list[BaseTool]:
    async def web_search(query: str, max_results: int = 5) -> str:
        if settings.tavily_api_key is None:
            return WebSearchOutput(
                query=query,
                provider="unavailable",
                note="Búsqueda web deshabilitada (falta TAVILY_API_KEY). Usá wikipedia_search.",
            ).model_dump_json()
        try:
            from langchain_tavily import TavilySearch

            tavily = TavilySearch(
                max_results=max_results,
                tavily_api_key=settings.tavily_api_key.get_secret_value(),
            )
            raw = await tavily.ainvoke({"query": query})
            items = raw.get("results", []) if isinstance(raw, dict) else []
            results = [
                WebResult(
                    title=str(r.get("title", "")),
                    url=str(r.get("url", "")),
                    content=str(r.get("content", ""))[:1500],
                )
                for r in items
            ]
            return WebSearchOutput(query=query, provider="tavily", results=results).model_dump_json()
        except Exception as exc:
            logger.warning("web_search falló: %s", exc)
            return WebSearchOutput(
                query=query, provider="unavailable", note=f"Error en búsqueda web: {exc}"
            ).model_dump_json()

    async def wikipedia_search(topic: str, lang: str | None = None) -> str:
        lang = lang or settings.wikipedia_lang
        api = f"https://{lang}.wikipedia.org/w/api.php"
        try:
            async with httpx.AsyncClient(
                timeout=settings.http_timeout_seconds, headers={"User-Agent": settings.wikipedia_user_agent}
            ) as client:
                search = await client.get(
                    api,
                    params={"action": "query", "list": "search", "srsearch": topic, "srlimit": 1, "format": "json"},
                )
                search.raise_for_status()
                hits = search.json().get("query", {}).get("search", [])
                if not hits:
                    return WikipediaOutput(topic=topic, lang=lang, found=False).model_dump_json()

                title = hits[0]["title"]
                page = await client.get(
                    api,
                    params={
                        "action": "query",
                        "prop": "extracts",
                        "exintro": 1,
                        "explaintext": 1,
                        "redirects": 1,
                        "titles": title,
                        "format": "json",
                    },
                )
                page.raise_for_status()
                pages = page.json().get("query", {}).get("pages", {})
                extract = next(iter(pages.values()), {}).get("extract", "")
            return WikipediaOutput(
                topic=topic,
                lang=lang,
                found=True,
                title=title,
                url=f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
                extract=extract[:3000],
            ).model_dump_json()
        except httpx.HTTPError as exc:
            logger.warning("wikipedia_search falló: %s", exc)
            return WikipediaOutput(topic=topic, lang=lang, found=False, error=str(exc)).model_dump_json()

    return [
        StructuredTool.from_function(
            coroutine=web_search,
            name="web_search",
            description="Busca información actual en la web (noticias, precios, versiones recientes).",
            args_schema=WebSearchInput,
        ),
        StructuredTool.from_function(
            coroutine=wikipedia_search,
            name="wikipedia_search",
            description="Obtiene el resumen enciclopédico de Wikipedia sobre un tema o concepto.",
            args_schema=WikipediaSearchInput,
        ),
    ]
