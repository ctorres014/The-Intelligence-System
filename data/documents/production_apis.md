# APIs de IA en producción: FastAPI asíncrono, Pydantic y Redis

## Asincronía de punta a punta

Las aplicaciones con LLMs pasan la mayor parte del tiempo esperando I/O: llamadas HTTP a proveedores de modelos, consultas a bases vectoriales y a Redis. Con `async`/`await` un único proceso atiende cientos de requests concurrentes mientras espera respuestas. La regla de oro es **no bloquear el event loop**: dentro de una función `async def` nunca se deben usar clientes síncronos como `requests`, el SDK síncrono de OpenAI o `redis.Redis`. Se usan en cambio `httpx.AsyncClient`, `AsyncOpenAI`, `AsyncAnthropic`, `redis.asyncio` y los métodos `ainvoke`, `astream` y `aembed_query` de LangChain. El trabajo de CPU intensivo o las librerías sin versión async se delegan a un thread con `asyncio.to_thread`.

Para ejecutar varias operaciones independientes en paralelo se usa `asyncio.gather`, por ejemplo la búsqueda BM25 y la búsqueda vectorial de un recuperador híbrido.

## FastAPI

FastAPI es un framework ASGI que integra Pydantic para validar requests y responses y genera automáticamente documentación OpenAPI (Swagger en `/docs`, ReDoc en `/redoc`). Buenas prácticas:

- Inicializar recursos compartidos (clientes, grafos compilados, conexiones) en el `lifespan` y no por request.
- Declarar `response_model` en cada endpoint para validar y documentar la salida.
- Usar códigos HTTP semánticos: 202 para tareas encoladas, 404 para recursos inexistentes, 409 para conflictos de estado, 422 para validación, 503 para dependencias caídas.
- Manejadores de excepciones globales que devuelvan un esquema de error consistente con un `request_id` para correlacionar con logs y trazas.
- Para respuestas largas, Server-Sent Events (SSE) con `StreamingResponse` permiten mostrar progreso en tiempo real.

## Pydantic como contrato

Pydantic v2 valida datos en tiempo de ejecución a partir de anotaciones de tipo. En un sistema de agentes se usa en tres lugares: (1) entradas y salidas de la API, (2) argumentos de las herramientas (`args_schema`), de modo que un argumento inválido generado por el LLM se rechaza con un mensaje claro que el agente puede corregir, y (3) salidas estructuradas del LLM, como la decisión del Supervisor. `pydantic-settings` permite además validar la configuración leída de variables de entorno, evitando hard-codear IDs de modelos o nombres de índices.

## Redis en arquitecturas de IA

Redis cumple varios roles: almacén de estado de tareas en background (con TTL), checkpointer de LangGraph para persistir conversaciones, caché semántica de respuestas, rate limiting y base de datos vectorial gracias al Query Engine (índices HNSW o FLAT con distancia coseno, L2 o producto interno). Redis 8 incluye estos módulos en la imagen oficial.

## Tareas en background

Cuando una ejecución puede tardar decenas de segundos, el patrón recomendado es responder `202 Accepted` con un `job_id`, procesar en background y exponer un endpoint de polling. Para cargas moderadas alcanza con tareas de asyncio dentro del mismo proceso; para alta escala se usa una cola dedicada (Celery, ARQ, RQ) con workers separados.

## Despliegue local reproducible

Docker Compose permite levantar la API, Redis y Phoenix con un solo comando (`docker compose up --build`). Conviene usar healthchecks, volúmenes para persistir datos entre reinicios, versiones de imágenes fijadas y un archivo `.env` (nunca versionado) con las credenciales, más un `.env.example` documentado.
