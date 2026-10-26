"""
Herramientas del Analyst Agent (portadas del Módulo 6 con contratos Pydantic).

Son funciones de CPU puras y rápidas; se exponen como corutinas para que el
ToolNode las ejecute sin bloquear el event loop.
"""

import ast
import json
import math
import re

from langchain_core.tools import BaseTool, StructuredTool

from app.tools.schemas import (
    CalculatorInput,
    CalculatorOutput,
    DataValidatorInput,
    DataValidatorOutput,
    SentimentInput,
    SentimentOutput,
    SummarizerInput,
    SummarizerOutput,
)

# ── math_calculator ─────────────────────────────────────────────────────────

_SAFE_NAMES: dict[str, object] = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "pow": pow, "min": min, "max": max,
    "log": math.log, "log2": math.log2, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "ceil": math.ceil, "floor": math.floor, "factorial": math.factorial,
    "pi": math.pi, "e": math.e,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
)


def evaluate_expression(expression: str) -> CalculatorOutput:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise ValueError(f"operación no permitida: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES:
                raise ValueError(f"nombre desconocido: {node.id}")
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                raise ValueError("solo se permiten constantes numéricas")
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307
        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError("resultado no finito")
        return CalculatorOutput(expression=expression, result=result)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return CalculatorOutput(expression=expression, error=str(exc))


# ── sentiment_analyzer ──────────────────────────────────────────────────────

_POSITIVE = {
    "good", "great", "excellent", "amazing", "love", "best", "positive", "improve", "fast",
    "reliable", "bueno", "buena", "excelente", "mejor", "positivo", "positiva", "rápido",
    "confiable", "eficiente", "robusto", "ventaja", "ventajas", "éxito", "útil",
}
_NEGATIVE = {
    "bad", "terrible", "awful", "worst", "negative", "slow", "fail", "failure", "broken",
    "malo", "mala", "peor", "negativo", "negativa", "lento", "falla", "fallo", "error",
    "costoso", "riesgo", "problema", "desventaja", "desventajas", "alucinación",
}


def analyze_sentiment(text: str) -> SentimentOutput:
    words = re.findall(r"\w+", text.lower())
    pos = sum(w in _POSITIVE for w in words)
    neg = sum(w in _NEGATIVE for w in words)
    polarity = 0.0 if pos + neg == 0 else round((pos - neg) / (pos + neg), 4)
    label = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
    return SentimentOutput(label=label, polarity=polarity, positive_hits=pos, negative_hits=neg)


# ── text_summarizer ─────────────────────────────────────────────────────────


def summarize(text: str, max_words: int) -> SummarizerOutput:
    original_words = len(text.split())
    if original_words <= max_words:
        return SummarizerOutput(summary=text, original_words=original_words, summary_words=original_words)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    freq: dict[str, int] = {}
    for w in re.findall(r"\w{4,}", text.lower()):
        freq[w] = freq.get(w, 0) + 1

    def score(sentence: str) -> float:
        tokens = re.findall(r"\w{4,}", sentence.lower())
        return sum(freq.get(t, 0) for t in tokens) / (len(tokens) or 1)

    chosen: set[int] = set()
    count = 0
    for idx in sorted(range(len(sentences)), key=lambda i: score(sentences[i]), reverse=True):
        n = len(sentences[idx].split())
        if count + n <= max_words:
            chosen.add(idx)
            count += n
    summary = " ".join(sentences[i] for i in sorted(chosen)) or " ".join(text.split()[:max_words])
    return SummarizerOutput(summary=summary, original_words=original_words, summary_words=len(summary.split()))


# ── data_validator ──────────────────────────────────────────────────────────

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def validate_data(data: str, required_fields: list[str], field_types: dict[str, str]) -> DataValidatorOutput:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        return DataValidatorOutput(valid=False, errors=[f"JSON inválido: {exc}"], warnings=[])
    if not isinstance(parsed, dict):
        return DataValidatorOutput(valid=False, errors=["se esperaba un objeto JSON"], warnings=[])

    errors = [f"falta el campo obligatorio '{f}'" for f in required_fields if f not in parsed]
    for field, expected in field_types.items():
        if field in parsed and not _TYPE_CHECKS[expected](parsed[field]):
            errors.append(f"'{field}' debería ser {expected}, es {type(parsed[field]).__name__}")
    warnings = [f"'{k}' está vacío" for k, v in parsed.items() if v in (None, "", [], {})]
    return DataValidatorOutput(valid=not errors, errors=errors, warnings=warnings)


# ── Registro ────────────────────────────────────────────────────────────────


def build_analysis_tools() -> list[BaseTool]:
    async def math_calculator(expression: str) -> str:
        return evaluate_expression(expression).model_dump_json()

    async def sentiment_analyzer(text: str) -> str:
        return analyze_sentiment(text).model_dump_json()

    async def text_summarizer(text: str, max_words: int = 100) -> str:
        return summarize(text, max_words).model_dump_json()

    async def data_validator(data: str, required_fields: list[str], field_types: dict[str, str] | None = None) -> str:
        return validate_data(data, required_fields, field_types or {}).model_dump_json()

    return [
        StructuredTool.from_function(
            coroutine=math_calculator,
            name="math_calculator",
            description="Evalúa expresiones matemáticas de forma segura (aritmética, potencias, sqrt, log, etc.). Usala para TODO cálculo numérico.",
            args_schema=CalculatorInput,
        ),
        StructuredTool.from_function(
            coroutine=sentiment_analyzer,
            name="sentiment_analyzer",
            description="Analiza el sentimiento (positivo/negativo/neutral) de un texto en español o inglés.",
            args_schema=SentimentInput,
        ),
        StructuredTool.from_function(
            coroutine=text_summarizer,
            name="text_summarizer",
            description="Resume extractivamente un texto largo a un máximo de palabras.",
            args_schema=SummarizerInput,
        ),
        StructuredTool.from_function(
            coroutine=data_validator,
            name="data_validator",
            description="Valida un objeto JSON contra campos obligatorios y tipos esperados.",
            args_schema=DataValidatorInput,
        ),
    ]
