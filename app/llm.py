"""
Factories de modelos de chat y de embeddings.

Todos los clientes devueltos son asíncronos de punta a punta: se usan con
`ainvoke` / `astream` / `aembed_*`, que internamente llaman a los SDKs async
oficiales (AsyncAnthropic, AsyncOpenAI, ollama.AsyncClient sobre httpx).
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    s = settings or get_settings()

    if s.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if s.anthropic_api_key is None:
            raise ValueError("LLM_PROVIDER=anthropic requiere ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=s.anthropic_model,
            api_key=s.anthropic_api_key,
            temperature=s.llm_temperature,
            timeout=s.llm_timeout,
            max_tokens=4096,
        )

    if s.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if s.google_api_key is None:
            raise ValueError("LLM_PROVIDER=gemini requiere GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=s.gemini_model,
            google_api_key=s.google_api_key,
            temperature=s.gemini_temperature,
            timeout=s.llm_timeout,
            max_retries=2,
        )

    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        if s.openai_api_key is None:
            raise ValueError("LLM_PROVIDER=openai requiere OPENAI_API_KEY")
        return ChatOpenAI(
            model=s.openai_model,
            api_key=s.openai_api_key,
            temperature=s.llm_temperature,
            timeout=s.llm_timeout,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=s.ollama_model,
        base_url=s.ollama_base_url,
        temperature=s.llm_temperature,
        num_ctx=s.ollama_num_ctx,
        client_kwargs={"timeout": s.llm_timeout},
    )


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()

    if s.embedding_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if s.google_api_key is None:
            raise ValueError("EMBEDDING_PROVIDER=gemini requiere GOOGLE_API_KEY")
        return GoogleGenerativeAIEmbeddings(
            model=s.gemini_embedding_model,
            google_api_key=s.google_api_key,
            output_dimensionality=s.gemini_embedding_dimensions,
        )

    if s.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if s.openai_api_key is None:
            raise ValueError("EMBEDDING_PROVIDER=openai requiere OPENAI_API_KEY")
        return OpenAIEmbeddings(model=s.openai_embedding_model, api_key=s.openai_api_key)

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=s.ollama_embedding_model, base_url=s.ollama_base_url)


@lru_cache
def get_chat_model() -> BaseChatModel:
    return build_chat_model()


def structured_output_method(settings: Settings | None = None) -> str:
    """Método de salida estructurada más fiable para cada proveedor."""
    s = settings or get_settings()
    return "json_schema" if s.llm_provider in ("ollama", "gemini") else "function_calling"
