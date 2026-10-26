"""
Contratos Pydantic de las herramientas de los agentes.

Cada herramienta declara un modelo de entrada (`args_schema`, que LangChain
usa para validar los argumentos que genera el LLM) y un modelo de salida que
se serializa a JSON antes de devolverse al agente.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ── Entradas ────────────────────────────────────────────────────────────────


class KnowledgeSearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=500, description="Consulta en lenguaje natural sobre la base de conocimiento interna")
    k: int = Field(default=5, ge=1, le=10, description="Cantidad de fragmentos a recuperar")


class WebSearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="Consulta para el buscador web")
    max_results: int = Field(default=5, ge=1, le=10)


class WikipediaSearchInput(BaseModel):
    topic: str = Field(min_length=2, max_length=200, description="Tema o término a buscar en Wikipedia")
    lang: str | None = Field(default=None, pattern=r"^[a-z]{2,3}$", description="Idioma de Wikipedia (ej. 'es', 'en')")


class CalculatorInput(BaseModel):
    expression: str = Field(
        min_length=1,
        max_length=300,
        description="Expresión matemática Python-like, ej. '1_000_000 * 1536 * 4 / 1024**3' o 'sqrt(16) + log10(1000)'",
    )

    @field_validator("expression")
    @classmethod
    def _no_dunder(cls, v: str) -> str:
        if "__" in v:
            raise ValueError("expresión no permitida")
        return v


class SentimentInput(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class SummarizerInput(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    max_words: int = Field(default=100, ge=10, le=500)


class DataValidatorInput(BaseModel):
    data: str = Field(min_length=2, description="Objeto JSON a validar, como string")
    required_fields: list[str] = Field(min_length=1, description="Campos obligatorios")
    field_types: dict[str, Literal["string", "number", "boolean", "array", "object"]] = Field(
        default_factory=dict, description="Tipos esperados por campo (opcional)"
    )


# ── Salidas ─────────────────────────────────────────────────────────────────


class SourceSnippet(BaseModel):
    ref: str = Field(description="Identificador para citar, ej. [vector_databases#2]")
    source: str
    doc_id: str
    section: str = ""
    score: float
    retrievers: list[str]
    text: str


class KnowledgeSearchOutput(BaseModel):
    query: str
    strategy: str
    degraded: bool = False
    warning: str | None = None
    results: list[SourceSnippet]


class WebResult(BaseModel):
    title: str
    url: str
    content: str


class WebSearchOutput(BaseModel):
    query: str
    provider: Literal["tavily", "unavailable"]
    results: list[WebResult] = Field(default_factory=list)
    note: str | None = None


class WikipediaOutput(BaseModel):
    topic: str
    lang: str
    found: bool
    title: str | None = None
    url: str | None = None
    extract: str | None = None
    error: str | None = None


class CalculatorOutput(BaseModel):
    expression: str
    result: float | int | None = None
    error: str | None = None


class SentimentOutput(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    polarity: float = Field(ge=-1.0, le=1.0)
    positive_hits: int = Field(ge=0)
    negative_hits: int = Field(ge=0)


class SummarizerOutput(BaseModel):
    summary: str
    original_words: int
    summary_words: int


class DataValidatorOutput(BaseModel):
    valid: bool
    errors: list[str]
    warnings: list[str]
