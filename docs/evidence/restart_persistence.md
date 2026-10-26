# Persistencia ante reinicio del servidor (Docker)

Prueba manual del 2026-10-26 con el sistema levantado por `./run.sh`. Configuración: LLM `gemini-3.5-flash-lite`, embeddings `nomic-embed-text` (Ollama del host), Pinecone con failover a Redis y checkpointer `AsyncRedisSaver`.

| Paso | Acción | Resultado |
|---|---|---|
| 1 | `POST /v1/chat` con el hilo `docker-restart-1793039163`: "¿Para qué sirve el checkpointer de LangGraph en este sistema?" | `completed`, agente `knowledge`, 5 fuentes, 18 s |
| 2 | `docker compose restart api` | Se reinicia el contenedor (Redis sigue arriba) |
| 3 | `GET /v1/threads/docker-restart-1793039163` | 2 mensajes (`user`, `assistant`) recuperados del checkpointer |
| 4 | `POST /v1/chat` en el mismo hilo: "¿Y qué pasa con eso si se reinicia el servidor?" | Resuelve "eso" = el checkpointer gracias al historial y cita `[langgraph_agents#5, langgraph_agents#6]` |
| 5 | `GET /v1/threads/docker-restart-1793039163` | 4 mensajes en el hilo |

La prueba automatizada equivalente es `tests/test_persistence.py`: simula el reinicio con un Redis real y verifica que sobreviven el historial y una aprobación humana pendiente.
