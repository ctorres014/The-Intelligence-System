# Agentes con LangGraph: Supervisor, Checkpointers y Human-in-the-Loop

## ¿Qué es LangGraph?

LangGraph es una librería para construir aplicaciones con LLMs como grafos de estado. Cada nodo es una función (sincrónica o asíncrona) que recibe el estado compartido y devuelve una actualización parcial. Las aristas pueden ser fijas o condicionales: una función de routing lee el estado y decide el próximo nodo. A diferencia de una cadena lineal, un grafo admite ciclos, lo que permite que un agente razone en varios pasos, revise resultados y vuelva a intentar.

El estado se define con un `TypedDict` o un modelo Pydantic. Los campos pueden tener reducers: por ejemplo `add_messages` agrega mensajes a la lista en lugar de reemplazarla, y respeta los IDs de mensaje para permitir ediciones.

## Patrón Supervisor

En el patrón Supervisor (también llamado jerárquico), un nodo central decide qué agente especialista actúa en cada paso. Cada especialista tiene su propio prompt y sus propias herramientas, y al terminar devuelve el control al Supervisor. Ventajas:

- **Routing dinámico**: el Supervisor puede saltear pasos innecesarios o repetir un agente con instrucciones más precisas.
- **Separación de responsabilidades**: cada agente tiene pocas herramientas, lo que mejora la precisión del tool calling.
- **Terminación controlada**: un único punto decide cuándo terminar. Se recomienda una guarda dura (máximo de iteraciones) para evitar loops infinitos.
- **Extensibilidad**: agregar un agente nuevo implica registrar un nodo y una arista.

Se recomienda que la decisión del Supervisor sea una salida estructurada validada con Pydantic (por ejemplo un campo `next` con valores `Literal`), con un fallback determinístico si el modelo devuelve algo inválido.

## Agentes ReAct

Un agente ReAct alterna razonamiento y acción: el LLM decide llamar a una herramienta, el ToolNode la ejecuta, el resultado vuelve al LLM y el ciclo continúa hasta que el modelo responde sin tool calls. En LangChain 1.x se construye con `create_agent(model, tools, system_prompt=...)`. Para producción conviene fijar un `recursion_limit` que acote la cantidad de pasos.

## Checkpointers y persistencia

Un checkpointer guarda una instantánea (checkpoint) del estado después de cada super-step del grafo, indexada por `thread_id`. Esto habilita:

1. **Memoria conversacional**: invocar el grafo con el mismo `thread_id` continúa la conversación con todo el historial.
2. **Tolerancia a fallos**: si el proceso se reinicia, el estado sigue en la base de datos.
3. **Human-in-the-Loop**: el grafo se pausa y se reanuda más tarde, incluso desde otro proceso.
4. **Time travel**: se puede inspeccionar o bifurcar desde checkpoints anteriores.

Implementaciones: `InMemorySaver` (solo desarrollo, se pierde al reiniciar), `SqliteSaver`, `PostgresSaver` y `RedisSaver` / `AsyncRedisSaver` (paquete `langgraph-checkpoint-redis`, requiere Redis 8 o Redis Stack con RediSearch y RedisJSON). En aplicaciones async se debe usar la variante asíncrona del checkpointer y llamar a `asetup()` una vez para crear los índices. El estado de un hilo se consulta con `graph.aget_state(config)`.

## Human-in-the-Loop con interrupt()

La función `interrupt(payload)` dentro de un nodo detiene la ejecución, persiste el checkpoint y expone el payload al cliente (en `StateSnapshot.interrupts`). Para continuar se invoca el grafo con `Command(resume=valor)` y el mismo `thread_id`; ese valor es lo que devuelve `interrupt()` dentro del nodo. Es importante no combinar `interrupt()` con `interrupt_before` sobre el mismo nodo, porque genera una doble interrupción. Casos de uso típicos: aprobar operaciones destructivas (borrar índices, desplegar, ejecutar pagos), validar datos sensibles o corregir la salida de un agente.
