"""
Pruebas end-to-end del sistema (generan las trazas para Arize Phoenix).

    python scripts/run_scenarios.py [--url http://localhost:8000] [--only 1,4]

Escenarios:
  1. RAG puro                       → Knowledge Agent con citas
  2. RAG + cálculo                  → Knowledge + Analyst (math_calculator)
  3. Investigación externa + RAG    → Research (Wikipedia) + Knowledge
  4. Memoria multi-turno            → mismo thread_id que el escenario 2
  5. HITL aprobado                  → awaiting_approval → approve → completed
  6. HITL rechazado vía /v1/tasks   → tarea en background → reject → rejected

Resultados: docs/evidence/scenarios_results.json y scenarios_report.md
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "evidence"

Q_RAG = "¿Qué es Reciprocal Rank Fusion y cómo combina los resultados de BM25 con la búsqueda vectorial?"
Q_CALC = (
    "Según nuestra documentación, ¿cuántas dimensiones tiene el modelo text-embedding-3-small? "
    "Calculá cuántos GB ocupan 2 millones de esos vectores almacenados en float32 (4 bytes por dimensión)."
)
Q_FOLLOWUP = "¿Y si usara un modelo de 768 dimensiones? ¿Qué porcentaje de almacenamiento ahorraría?"
Q_RESEARCH = (
    "¿Quiénes desarrollaron el algoritmo BM25 (Okapi BM25) y en qué sistema se usó originalmente? "
    "Relacionalo con el rol que cumple BM25 en nuestra búsqueda híbrida."
)
Q_HITL_APPROVE = (
    "Necesito borrar el índice vectorial de producción y reingestar todos los documentos con chunks de 400 tokens. "
    "Dame el plan paso a paso y estimá cuántos chunks se generarían para 1.500 páginas de 500 tokens cada una."
)
Q_HITL_REJECT = "Desplegar a producción la nueva versión del índice eliminando el namespace 'documents' de Pinecone."


def summarize(name: str, data: dict, elapsed: float) -> dict:
    answer = data.get("answer") or ""
    print(f"   status={data.get('status')} agents={data.get('agents_used')} iter={data.get('iterations')} "
          f"sources={len(data.get('sources', []))} t={elapsed:.1f}s")
    print("   " + (answer[:300].replace("\n", " ") + ("…" if len(answer) > 300 else "")))
    return {"scenario": name, "elapsed_s": round(elapsed, 1), "response": data}


async def chat(client: httpx.AsyncClient, message: str, thread_id: str | None = None) -> tuple[dict, float]:
    t0 = time.perf_counter()
    r = await client.post("/v1/chat", json={"message": message, "thread_id": thread_id})
    r.raise_for_status()
    return r.json(), time.perf_counter() - t0


async def scenario_1(client, ctx):
    data, t = await chat(client, Q_RAG)
    return summarize("1. RAG puro", data, t)


async def scenario_2(client, ctx):
    data, t = await chat(client, Q_CALC, ctx.setdefault("calc_thread", f"calc-{int(time.time())}"))
    return summarize("2. RAG + cálculo", data, t)


async def scenario_3(client, ctx):
    data, t = await chat(client, Q_RESEARCH)
    return summarize("3. Investigación externa + RAG", data, t)


async def scenario_4(client, ctx):
    thread = ctx.get("calc_thread")
    if thread is None:
        await scenario_2(client, ctx)
        thread = ctx["calc_thread"]
    data, t = await chat(client, Q_FOLLOWUP, thread)
    history = (await client.get(f"/v1/threads/{thread}")).json()
    print(f"   historial persistido: {len(history['messages'])} mensajes en el hilo {thread}")
    result = summarize("4. Memoria multi-turno", data, t)
    result["thread_history"] = history
    return result


async def scenario_5(client, ctx):
    data, t1 = await chat(client, Q_HITL_APPROVE)
    print(f"   → {data['status']}: {(data.get('pending_approval') or {}).get('reason')}")
    if data["status"] != "awaiting_approval":
        return summarize("5. HITL aprobado (no se activó la aprobación)", data, t1)
    t0 = time.perf_counter()
    r = await client.post(
        f"/v1/threads/{data['thread_id']}/approve",
        json={"approved": True, "comment": "Aprobado: ejecutar fuera del horario pico."},
    )
    r.raise_for_status()
    result = summarize("5. HITL aprobado", r.json(), t1 + time.perf_counter() - t0)
    result["paused_response"] = data
    return result


async def poll_task(client, job_id: str, until: set[str]) -> dict:
    for _ in range(1200):
        job = (await client.get(f"/v1/tasks/{job_id}")).json()
        if job["status"] in until:
            return job
        await asyncio.sleep(2)
    raise TimeoutError(job_id)


async def scenario_6(client, ctx):
    t0 = time.perf_counter()
    r = await client.post("/v1/tasks", json={"message": Q_HITL_REJECT})
    r.raise_for_status()
    accepted = r.json()
    print(f"   202 Accepted job_id={accepted['job_id']}")
    job = await poll_task(client, accepted["job_id"], {"awaiting_approval", "completed", "failed", "rejected"})
    print(f"   → {job['status']}")
    if job["status"] == "awaiting_approval":
        r = await client.post(
            f"/v1/tasks/{accepted['job_id']}/approve",
            json={"approved": False, "comment": "Rechazado: no se borra producción sin ventana de mantenimiento."},
        )
        r.raise_for_status()
        job = await poll_task(client, accepted["job_id"], {"completed", "failed", "rejected"})
    result = summarize("6. HITL rechazado (tarea en background)", job.get("result") or {"status": job["status"]}, time.perf_counter() - t0)
    result["job"] = {k: v for k, v in job.items() if k != "result"}
    return result


SCENARIOS = [scenario_1, scenario_2, scenario_3, scenario_4, scenario_5, scenario_6]


def write_report(results: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scenarios_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Resultados de los escenarios de prueba",
        "",
        "| Escenario | Estado | Agentes | Iteraciones | Fuentes | Latencia |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        d = r["response"]
        lines.append(
            f"| {r['scenario']} | {d.get('status')} | {', '.join(d.get('agents_used', [])) or '—'} | "
            f"{d.get('iterations', '—')} | {len(d.get('sources', []))} | {r['elapsed_s']} s |"
        )
    for r in results:
        d = r["response"]
        lines += ["", f"## {r['scenario']}", "", f"**thread_id:** `{d.get('thread_id')}`", "", "**Recorrido del grafo:**", ""]
        lines += [f"- `{s['node']}`" + (f" → **{s['decision']}**" if s.get("decision") else "") + f": {s['summary'][:160]}"
                  for s in d.get("steps", [])]
        lines += ["", "**Respuesta:**", "", (d.get("answer") or "_(sin respuesta)_")]
    (OUT_DIR / "scenarios_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main(url: str, only: set[int] | None, timeout: float) -> int:
    ctx: dict = {}
    results = []
    async with httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(timeout, connect=10.0)) as client:
        health = (await client.get("/health")).json()
        print("Health:", json.dumps(health, ensure_ascii=False))
        for i, scenario in enumerate(SCENARIOS, start=1):
            if only and i not in only:
                continue
            print(f"\n▶ Escenario {i}")
            try:
                results.append(await scenario(client, ctx))
            except Exception as exc:
                print(f"   ✗ falló: {exc!r}")
                results.append({"scenario": f"{i}. (error)", "elapsed_s": 0, "response": {"status": f"error: {exc}"}})
            write_report(results)  # incremental: una corrida interrumpida conserva lo ya ejecutado
    print(f"\nResultados en {OUT_DIR.relative_to(ROOT)}/ — revisá las trazas en Phoenix (http://localhost:6006)")
    return 0 if all(not str(r["response"].get("status", "")).startswith("error") for r in results) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--only", default="", help="Lista de escenarios, ej. 1,4")
    parser.add_argument("--timeout", type=float, default=900, help="Timeout por request en segundos (Ollama en CPU: usar 2400)")
    args = parser.parse_args()
    selected = {int(x) for x in args.only.split(",") if x.strip()} or None
    sys.exit(asyncio.run(main(args.url, selected, args.timeout)))
