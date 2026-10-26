# Retrieval-Augmented Generation (RAG): Fundamentos

## ¿Qué es RAG?

Retrieval-Augmented Generation (RAG) es una arquitectura de inteligencia artificial que combina la recuperación de información con la generación de texto mediante modelos de lenguaje de gran escala (LLMs). En lugar de depender únicamente del conocimiento almacenado en los pesos del modelo durante el entrenamiento, RAG permite que el modelo consulte una base de conocimiento externa en tiempo de inferencia.

## Componentes principales

### 1. Base de conocimiento
La base de conocimiento es el repositorio de documentos que el sistema puede consultar. Puede contener PDFs, páginas web, artículos técnicos, manuales o cualquier tipo de texto estructurado o no estructurado. Los documentos se procesan y almacenan en una base de datos vectorial.

### 2. Pipeline de ingesta
El pipeline de ingesta es el proceso mediante el cual los documentos son:
- Cargados desde sus fuentes originales
- Divididos en fragmentos (chunks) de tamaño apropiado
- Convertidos en vectores de embeddings
- Almacenados junto con metadatos relevantes en la base de datos vectorial

### 3. Recuperador (Retriever)
El recuperador es el componente que, dada una consulta del usuario, busca los fragmentos de texto más relevantes en la base de conocimiento. Utiliza métricas de similitud semántica (coseno, producto punto) para encontrar los vectores más cercanos al embedding de la consulta.

### 4. Generador (Generator)
El generador es el LLM que recibe el contexto recuperado junto con la pregunta del usuario y produce una respuesta coherente y fundamentada en los documentos recuperados.

## Ventajas de RAG sobre Fine-tuning

1. **Actualización dinámica**: La base de conocimiento se puede actualizar sin reentrenar el modelo.
2. **Verificabilidad**: Las respuestas pueden atribuirse a fuentes específicas.
3. **Costo**: Es significativamente más económico que el fine-tuning continuo.
4. **Reducción de alucinaciones**: Al anclar las respuestas a documentos reales, se reduce la generación de información falsa.

## Métricas de evaluación

Las métricas más comunes para evaluar sistemas RAG incluyen:
- **Faithfulness**: ¿La respuesta está sustentada por los documentos recuperados?
- **Answer Relevancy**: ¿La respuesta responde a la pregunta?
- **Context Precision**: ¿Los documentos recuperados son relevantes para la pregunta?
- **Context Recall**: ¿Se recuperaron todos los documentos necesarios para responder?
