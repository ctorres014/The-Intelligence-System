import json

import pytest
from pydantic import ValidationError

from app.tools import build_analysis_tools, build_knowledge_tool
from app.tools.analysis import analyze_sentiment, evaluate_expression, summarize, validate_data
from app.tools.schemas import CalculatorInput, KnowledgeSearchOutput


def test_calculator_evaluates_expressions():
    assert evaluate_expression("2_000_000 * 1536 * 4 / 1024**3").result == pytest.approx(11.444, rel=1e-3)
    assert evaluate_expression("sqrt(16) + log10(1000)").result == 7.0


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "'a' * 3", "1/0", "x + 1"])
def test_calculator_rejects_unsafe_or_invalid(expr):
    if "__" in expr:
        with pytest.raises(ValidationError):
            CalculatorInput(expression=expr)
    assert evaluate_expression(expr).error is not None


def test_sentiment_bilingual():
    assert analyze_sentiment("Es un sistema excelente, rápido y confiable").label == "positive"
    assert analyze_sentiment("This is slow, broken and the worst").label == "negative"
    assert analyze_sentiment("El índice tiene 768 dimensiones").label == "neutral"


def test_summarizer_respects_max_words():
    text = " ".join(f"Oración número {i} sobre búsqueda híbrida y embeddings." for i in range(60))
    out = summarize(text, 40)
    assert out.summary_words <= 40 < out.original_words


def test_data_validator():
    ok = validate_data('{"name": "kb", "dims": 768}', ["name", "dims"], {"dims": "number"})
    assert ok.valid
    bad = validate_data('{"name": 3}', ["name", "dims"], {"name": "string"})
    assert not bad.valid and len(bad.errors) == 2
    assert not validate_data("no-json", ["a"], {}).valid


async def test_tools_validate_arguments_with_pydantic(kb):
    tool = build_knowledge_tool(kb)
    with pytest.raises(ValidationError):
        await tool.ainvoke({"query": "rrf", "k": 50})  # k > 10

    calc = next(t for t in build_analysis_tools() if t.name == "math_calculator")
    assert json.loads(await calc.ainvoke({"expression": "2 + 2"}))["result"] == 4


async def test_knowledge_tool_returns_validated_output(kb):
    raw = await build_knowledge_tool(kb).ainvoke({"query": "Reciprocal Rank Fusion", "k": 3})
    output = KnowledgeSearchOutput.model_validate_json(raw)
    assert output.strategy == "hybrid"
    assert len(output.results) == 3
    assert all(r.ref.startswith("[") for r in output.results)
