# Capturas de Arize Phoenix

Generadas con Playwright (`python scripts/capture_phoenix.py`) después de ejecutar `python scripts/run_scenarios.py` con el sistema levantado.
Requiere `pip install -r requirements-dev.txt` y `python -m playwright install chromium`.
Dashboard: http://localhost:6006 → proyecto **intelligence-system**.

| Archivo | Qué muestra |
|---|---|
| `phoenix_traces.png` | Lista de trazas de los escenarios con latencia, tokens y estado |
| `phoenix_trace_detail.png` | Una traza expandida: grafo → supervisor → knowledge_agent → `knowledge_base_search` → synthesizer |
| `phoenix_sessions.png` | Sesión multi-turno (escenario 2 + 4) agrupada por `thread_id` |
