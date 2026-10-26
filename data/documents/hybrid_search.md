# Búsqueda Híbrida: BM25 + Vectorial

## ¿Por qué búsqueda híbrida?

Los sistemas de recuperación puramente vectoriales (dense retrieval) son excelentes para capturar similitud semántica, pero tienen debilidades conocidas:
- **Términos técnicos precisos**: Siglas, nombres de productos, identificadores de errores (ej. "HTTPError 429", "SKU-78923")
- **Nombres propios**: Personas, empresas, lugares
- **Coincidencias exactas**: Cuando el usuario necesita encontrar una frase específica

Los sistemas léxicos como BM25 (sparse retrieval) complementan estas debilidades con sus fortalezas:
- Alta precisión en términos exactos
- No requieren embeddings (más rápido en indexado)
- Funcionan bien con vocabulario técnico especializado

La **búsqueda híbrida** combina ambos enfoques para obtener lo mejor de cada mundo.

## BM25 (Best Match 25)

BM25 es un algoritmo de ranking léxico basado en TF-IDF (Term Frequency-Inverse Document Frequency) con mejoras:

### Fórmula básica
```
Score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl))
```

Donde:
- `f(qi, D)`: frecuencia del término qi en el documento D
- `|D|`: longitud del documento
- `avgdl`: longitud promedio de documentos en la colección
- `k1` y `b`: parámetros de saturación y normalización de longitud

### Parámetros
- `k1 = 1.5`: controla la saturación de la frecuencia del término
- `b = 0.75`: penalización por longitud del documento

## EnsembleRetriever en LangChain

LangChain provee `EnsembleRetriever` para combinar múltiples retrievers mediante fusión de rankings con Reciprocal Rank Fusion (RRF):

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore

bm25_retriever = BM25Retriever.from_documents(docs, k=5)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)
```

## Reciprocal Rank Fusion (RRF)

RRF es el algoritmo de fusión de rankings más común. Dado un conjunto de listas de rankings:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Donde `k = 60` es una constante de suavizado y `rank_i(d)` es la posición del documento d en la lista i.

### Ventajas de RRF
- No requiere normalización de scores
- Robusto ante distribuciones de scores diferentes
- Simple de implementar

## Casos de uso donde el híbrido es superior

| Tipo de consulta | Dense | Sparse | Híbrido |
|-----------------|-------|--------|---------|
| "¿Qué es machine learning?" | Excelente | Bueno | Excelente |
| "Error código ORA-12541" | Pobre | Excelente | Excelente |
| "Diferencia entre BERT y GPT" | Bueno | Pobre | Excelente |
| "Número de serie SN-78432-A" | Pobre | Excelente | Excelente |
| "Conceptos similares a transformers" | Excelente | Pobre | Excelente |
