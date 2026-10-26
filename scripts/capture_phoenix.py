"""
Capturas de Arize Phoenix con Playwright (evidencia de trazabilidad).

    pip install -r requirements-dev.txt && python -m playwright install chromium
    python scripts/run_scenarios.py          # genera las trazas
    python scripts/capture_phoenix.py [--phoenix http://localhost:6006]

Usa el hilo multi-turno de la última corrida (escenarios 2 y 4, leído de
docs/evidence/scenarios_results.json) y guarda en docs/screenshots/:
  phoenix_traces.png        trazas de los endpoints de la API (sin el polling de tareas)
  phoenix_trace_detail.png  supervisor → knowledge_agent → knowledge_base_search → ... → synthesizer
  phoenix_sessions.png      sesión multi-turno agrupada por thread_id
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
from playwright.async_api import Page, async_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

OUT_DIR = ROOT / "docs" / "screenshots"
RESULTS = ROOT / "docs" / "evidence" / "scenarios_results.json"
# Las trazas del grafo tienen decenas de spans; el polling GET /v1/tasks, uno o dos
TRACES_FILTER = "num_spans >= 5"


def multi_turn_thread() -> str:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    for r in results:
        if r["scenario"].startswith("4."):
            return r["response"]["thread_id"]
    raise SystemExit("No hay escenario 4 en scenarios_results.json: ejecutá run_scenarios.py primero")


async def find_trace(client: httpx.AsyncClient, project: str, thread_id: str) -> str:
    """Primera traza (escenario 2) del hilo multi-turno, vía la REST API de Phoenix."""
    spans, cursor = [], None
    for _ in range(20):
        params = {"limit": 1000, **({"cursor": cursor} if cursor else {})}
        page = (await client.get(f"/v1/projects/{project}/spans", params=params)).json()
        spans += page["data"]
        cursor = page.get("next_cursor")
        if not cursor:
            break
    matches = [s for s in spans if (s.get("attributes") or {}).get("session.id") == thread_id]
    if not matches:
        raise SystemExit(f"No se encontraron spans del hilo {thread_id} en Phoenix")
    return min(matches, key=lambda s: s["start_time"])["context"]["trace_id"]


async def settle(page: Page, ms: int = 2500) -> None:
    await page.wait_for_load_state("networkidle")
    await page.wait_for_timeout(ms)
    # Oculta el aviso de actualización de Phoenix si aparece
    await page.add_style_tag(content="[class*='toast'], [role='alert'] {display:none !important}")
    close = page.get_by_text("Update available").locator("xpath=ancestor::*[.//button][1]//button")
    if await close.count():
        await close.first.click()


async def main(phoenix: str) -> None:
    settings = get_settings()
    thread_id = multi_turn_thread()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(base_url=phoenix, timeout=30) as client:
        projects = (await client.get("/v1/projects")).json()["data"]
        project_id = next(p["id"] for p in projects if p["name"] == settings.phoenix_project_name)
        trace_id = await find_trace(client, settings.phoenix_project_name, thread_id)
    base = f"{phoenix}/projects/{project_id}"

    async with async_playwright() as p:
        browser = await p.chromium.launch()

        page = await browser.new_page(viewport={"width": 1600, "height": 1000})
        await page.goto(f"{base}/traces?timeRangeKey=1d")
        await settle(page)
        # El filtro es un editor CodeMirror, no un <input>
        await page.locator(".cm-content").first.click()
        await page.keyboard.type(TRACES_FILTER)
        await page.keyboard.press("Enter")
        await page.keyboard.press("Escape")  # cierra el autocompletado
        await page.get_by_text("Overall volume of traces").click()
        await settle(page)
        await page.screenshot(path=OUT_DIR / "phoenix_traces.png")
        print("✔ phoenix_traces.png")

        page = await browser.new_page(viewport={"width": 1600, "height": 1900})
        await page.goto(f"{base}/traces/{trace_id}")
        await settle(page)
        await page.get_by_text("knowledge_base", exact=False).first.click()
        await settle(page, 1500)
        await page.screenshot(path=OUT_DIR / "phoenix_trace_detail.png")
        print(f"✔ phoenix_trace_detail.png (trace {trace_id})")

        page = await browser.new_page(viewport={"width": 1600, "height": 1100})
        await page.goto(f"{base}/sessions?timeRangeKey=1d")
        await settle(page)
        await page.get_by_text(thread_id, exact=True).first.click()
        await settle(page)
        await page.screenshot(path=OUT_DIR / "phoenix_sessions.png")
        print(f"✔ phoenix_sessions.png (sesión {thread_id})")

        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phoenix", default="http://localhost:6006")
    asyncio.run(main(parser.parse_args().phoenix))
