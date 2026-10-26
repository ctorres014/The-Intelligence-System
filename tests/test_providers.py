"""Factories de proveedores (sin llamadas de red) y cambio de modelo de embeddings."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.config import Settings
from app.llm import build_chat_model, build_embeddings, structured_output_method
from app.rag import KnowledgeBase
from app.rag.vectorstore import InMemoryVectorStore


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, phoenix_enabled=False, **overrides)


@pytest.mark.parametrize(
    ("provider", "extra", "class_name", "model"),
    [
        ("anthropic", {"anthropic_api_key": "sk-ant-test"}, "ChatAnthropic", "claude-sonnet-5-5"),
        ("gemini", {"google_api_key": "test-key"}, "ChatGoogleGenerativeAI", "gemini-3.8-flash"),
        ("openai", {"openai_api_key": "sk-test"}, "ChatOpenAI", "gpt-4o-mini"),
        ("ollama", {}, "ChatOllama", "qwen2.5:7b"),
    ],
)
def test_chat_model_factory(provider, extra, class_name, model):
    s = _settings(llm_provider=provider, **extra)
    llm = build_chat_model(s)
    assert type(llm).__name__ == class_name
    assert s.llm_model_name == model
    assert model in str(getattr(llm, "model", "") or getattr(llm, "model_name", ""))


@pytest.mark.parametrize(
    ("provider", "extra", "class_name"),
    [
        ("gemini", {"google_api_key": "test-key"}, "GoogleGenerativeAIEmbeddings"),
        ("openai", {"openai_api_key": "sk-test"}, "OpenAIEmbeddings"),
        ("ollama", {}, "OllamaEmbeddings"),
    ],
)
def test_embeddings_factory(provider, extra, class_name):
    assert type(build_embeddings(_settings(embedding_provider=provider, **extra))).__name__ == class_name


@pytest.mark.parametrize("provider", ["anthropic", "gemini", "openai"])
def test_missing_api_key_fails_fast(provider):
    with pytest.raises(ValueError, match="requiere"):
        build_chat_model(_settings(llm_provider=provider))


def test_gemini_embeddings_require_key():
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        build_embeddings(_settings(embedding_provider="gemini"))


def test_structured_output_method_per_provider():
    assert structured_output_method(_settings(llm_provider="gemini")) == "json_schema"
    assert structured_output_method(_settings(llm_provider="ollama")) == "json_schema"
    assert structured_output_method(_settings(llm_provider="anthropic")) == "function_calling"


def test_fingerprint_changes_with_embedding_model():
    ollama = _settings(embedding_provider="ollama").embedding_fingerprint
    gemini = _settings(embedding_provider="gemini").embedding_fingerprint
    gemini_3072 = _settings(embedding_provider="gemini", gemini_embedding_dimensions=3072).embedding_fingerprint
    assert len({ollama, gemini, gemini_3072}) == 3


async def test_reingests_when_embedding_model_changes():
    store = InMemoryVectorStore()
    first = KnowledgeBase(_settings(embedding_provider="ollama"), store, DeterministicFakeEmbedding(size=16))
    await first.ingest()
    assert await first.ensure_ingested() is None  # mismo modelo → no reingesta

    # Simula pasar de Ollama a Gemini: otro modelo y otra dimensión
    second = KnowledgeBase(_settings(embedding_provider="gemini"), store, DeterministicFakeEmbedding(size=32))
    await second.load()
    report = await second.ensure_ingested()
    assert report is not None and report.dimensions == 32
    assert await store.get_fingerprint() == _settings(embedding_provider="gemini").embedding_fingerprint


def test_llm_timeout_depends_on_provider():
    assert _settings(llm_provider="ollama").llm_timeout == 600
    assert _settings(llm_provider="gemini").llm_timeout == 120
    assert _settings(llm_provider="anthropic", llm_timeout_seconds=30).llm_timeout == 30


def test_gemini_key_accepts_both_env_names(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-gemini-name")
    assert _settings().google_api_key.get_secret_value() == "from-gemini-name"
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google-name")
    assert _settings().google_api_key.get_secret_value() == "from-google-name"
