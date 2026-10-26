# Embeddings y Estrategias de Chunking

## ¿Qué son los embeddings?

Los embeddings son representaciones numéricas de texto en un espacio vectorial de alta dimensionalidad. Textos con significados similares quedan ubicados cerca en este espacio, lo que permite realizar búsquedas semánticas (por significado, no por palabras exactas).

## Modelos de embedding populares

### OpenAI
- **text-embedding-3-small**: 1536 dimensiones, excelente relación calidad/costo
- **text-embedding-3-large**: 3072 dimensiones, mayor precisión
- **text-embedding-ada-002**: 1536 dimensiones (modelo anterior, legacy)

### Open Source
- **sentence-transformers/all-MiniLM-L6-v2**: 384 dims, muy rápido, ideal para CPU
- **BAAI/bge-large-en-v1.5**: 1024 dims, alta calidad
- **intfloat/multilingual-e5-large**: Soporte multilingüe

## Estrategias de chunking

El chunking (fragmentación) es uno de los pasos más críticos en un pipeline RAG. Una estrategia inadecuada puede degradar significativamente la calidad de la recuperación.

### Fixed-size chunking
Divide el texto en fragmentos de tamaño fijo (por número de tokens o caracteres), con un solapamiento (overlap) entre fragmentos consecutivos.

**Ventajas**: Simple, predecible, fácil de implementar.
**Desventajas**: Puede cortar oraciones a la mitad, perdiendo contexto.

**Recomendación de tamaño**: 500-800 tokens con 10-15% de overlap.

### Recursive Character Text Splitter
Intenta dividir por separadores jerárquicos: primero párrafos (`\n\n`), luego saltos de línea (`\n`), luego oraciones, finalmente por caracteres. Es la estrategia más común y recomendada para texto general.

### Semantic Chunking
Agrupa oraciones semanticamente relacionadas usando embeddings. Más costoso computacionalmente pero produce chunks más coherentes.

### Document-based Chunking
Respeta la estructura natural del documento:
- En Markdown: divide por encabezados (`#`, `##`, `###`)
- En PDFs: divide por páginas o secciones
- En código: divide por funciones o clases

## Metadatos importantes a preservar

Al chunkar documentos, es crítico preservar metadatos que permitan:
1. Identificar la fuente del fragmento (nombre de archivo, URL)
2. Ubicar el fragmento dentro del documento (número de página, posición)
3. Categorizar el contenido (tipo de documento, tema, autor)
4. Rastrear la versión del documento (fecha de ingesta, hash)

## El problema del "lost in the middle"

Investigaciones han demostrado que los LLMs tienden a prestar menos atención a la información ubicada en el medio de un contexto largo. Estrategias para mitigarlo:
- **Reranking**: Reordenar los chunks recuperados por relevancia antes de pasarlos al LLM
- **Lost in the Middle Reranker**: Colocar los chunks más relevantes al inicio y al final
- **Chunk size reduction**: Usar chunks más pequeños para aumentar la densidad de información relevante
