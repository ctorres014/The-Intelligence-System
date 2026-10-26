# Observabilidad de sistemas LLM: trazas, métricas y Arize Phoenix

## ¿Por qué observabilidad?

Un sistema con LLMs y agentes es no determinístico: la misma consulta puede tomar caminos distintos por el grafo, llamar a herramientas diferentes y consumir cantidades muy distintas de tokens. Sin observabilidad, el sistema es una caja negra: no se puede explicar por qué respondió algo, dónde se fue la latencia ni cuánto costó. La observabilidad permite depurar alucinaciones, detectar loops de agentes, medir costos y evaluar la calidad de la recuperación en RAG.

## Conceptos: trazas y spans

Una **traza** representa una ejecución completa de punta a punta (por ejemplo, un turno de conversación). Está compuesta por **spans** anidados: cada span es una unidad de trabajo con inicio, fin, atributos y estado. En un sistema multi-agente típico, la traza raíz contiene spans para cada nodo del grafo; dentro de ellos aparecen spans de tipo LLM (prompt, respuesta, tokens de entrada y salida), de tipo TOOL (argumentos y resultado) y de tipo RETRIEVER (documentos recuperados con su score).

Las trazas pueden agruparse en **sesiones** usando el atributo `session.id`, por ejemplo con el `thread_id` de la conversación, para ver todos los turnos de un usuario juntos.

## OpenTelemetry y OpenInference

OpenTelemetry (OTel) es el estándar abierto para emitir trazas, métricas y logs. Las trazas se exportan por OTLP (HTTP o gRPC) a un colector. OpenInference es un conjunto de convenciones semánticas y de instrumentaciones sobre OTel específicas para aplicaciones de IA: define atributos como `llm.token_count.prompt`, `llm.model_name`, `tool.name` o `retrieval.documents`. El paquete `openinference-instrumentation-langchain` instrumenta automáticamente LangChain y LangGraph mediante callbacks, sin modificar el código de los agentes.

## Arize Phoenix

Phoenix es una plataforma open source de observabilidad y evaluación de LLMs. Se puede ejecutar localmente (`phoenix serve` o la imagen Docker `arizephoenix/phoenix`) y expone su UI y el colector OTLP en el puerto 6006. Funcionalidades principales:

- Vista de trazas con el árbol de spans, latencia por paso y tokens consumidos.
- Sesiones para seguir conversaciones multi-turno.
- Proyectos para separar entornos o aplicaciones.
- Evaluaciones (LLM-as-a-judge) de relevancia de recuperación, alucinaciones y calidad de respuesta.
- Datasets y experimentos para comparar versiones de prompts.

La configuración en Python es `phoenix.otel.register(endpoint=..., project_name=...)` seguida de `LangChainInstrumentor().instrument(tracer_provider=...)`. Se recomienda `batch=True` en producción para no agregar latencia a cada request.

## LangSmith

LangSmith es la alternativa SaaS de LangChain. Se activa con las variables `LANGSMITH_TRACING=true` y `LANGSMITH_API_KEY`, sin cambios de código. Ofrece funcionalidades similares (trazas, datasets, evaluaciones, monitoreo).

## Métricas clave a monitorear

- **Latencia** p50/p95/p99 por endpoint y por nodo del grafo.
- **Tokens y costo** por request, por agente y por modelo.
- **Tasa de error** de LLMs y herramientas (timeouts, rate limits, validaciones fallidas).
- **Iteraciones del Supervisor** por request: un valor alto indica loops o instrucciones ambiguas.
- **Calidad del RAG**: Hit Rate@k, MRR, Precision@k y Recall@k sobre un golden set.
