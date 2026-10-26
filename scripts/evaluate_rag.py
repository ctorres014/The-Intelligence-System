"""
Evaluación del recuperador sobre el golden set.

    python scripts/evaluate_rag.py            # k=5
    python scripts/evaluate_rag.py --k 3

Métricas: Hit Rate@k, MRR (a nivel documento) y Precision@k (fracción de los
k chunks que pertenecen al documento esperado). Además compara la estrategia
híbrida contra BM25 solo y vectorial solo (ablación).
Guarda el detalle en docs/evidence/rag_evaluation.json.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.llm import build_embeddings  # noqa: E402
from app.rag import KnowledgeBase, build_vector_store  # noqa: E402

ABLATIONS = {
    "bm25_only": {"bm25_weight": 1.0, "vector_weight": 0.0},
    "vector_only": {"bm25_weight": 0.0, "vector_weight": 1.0},
}


async def evaluate(kb: KnowledgeBase, golden: list[dict], k: int, verbose: bool) -> tuple[dict, list[dict]]:
    rows = []
    for item in golden:
        result = await kb.search(item["pregunta"], k)
        docs = list(dict.fromkeys(c.doc_id for c in result.chunks))
        expected = item["documento_id_esperado"]
        rank = docs.index(expected) + 1 if expected in docs else None
        rows.append({
            "id": item["id"],
            "question": item["pregunta"],
            "expected": expected,
            "retrieved_docs": docs,
            "retrieved_chunks": [c.id for c in result.chunks],
            "rank": rank,
            "precision_at_k": sum(c.doc_id == expected for c in result.chunks) / k,
        })
        if verbose:
            print(f"[{'HIT ' if rank else 'MISS'}] {item['id']:>4} rank={rank or '-'}  {item['pregunta'][:70]}")

    n = len(rows)
    metrics = {
        "hit_rate": round(sum(r["rank"] is not None for r in rows) / n, 4),
        "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 4),
        "mean_precision_at_k": round(sum(r["precision_at_k"] for r in rows) / n, 4),
    }
    return metrics, rows


async def main(k: int) -> None:
    settings = get_settings()
    golden = json.loads((ROOT / "data" / "golden_set.json").read_text(encoding="utf-8"))
    store = build_vector_store(settings)
    embeddings = build_embeddings(settings)

    kb = KnowledgeBase(settings, store, embeddings)
    await kb.load()
    await kb.ensure_ingested()
    hybrid, rows = await evaluate(kb, golden, k, verbose=True)

    comparison = {"hybrid": hybrid}
    for name, weights in ABLATIONS.items():
        variant = KnowledgeBase(settings.model_copy(update=weights), store, embeddings)
        await variant.load()
        comparison[name], _ = await evaluate(variant, golden, k, verbose=False)

    metrics = {
        "k": k,
        "questions": len(golden),
        "vector_store": kb.store_name,
        "embedding_model": settings.embedding_model_name,
        "weights": {"bm25": settings.bm25_weight, "vector": settings.vector_weight, "rrf_k": settings.rrf_k},
        "comparison": comparison,
    }
    print("\n" + json.dumps(metrics, indent=2))

    out = ROOT / "docs" / "evidence" / "rag_evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metrics": metrics, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detalle guardado en {out.relative_to(ROOT)}")
    await kb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    asyncio.run(main(parser.parse_args().k))
