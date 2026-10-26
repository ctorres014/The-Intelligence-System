# Bases de Datos Vectoriales

## Introducción

Las bases de datos vectoriales son sistemas de almacenamiento diseñados específicamente para guardar y consultar embeddings de alta dimensionalidad de manera eficiente. A diferencia de las bases de datos relacionales tradicionales, que buscan por valores exactos, las bases de datos vectoriales realizan búsquedas por similitud aproximada (Approximate Nearest Neighbor, ANN).

## Pinecone

Pinecone es una base de datos vectorial administrada en la nube (DBaaS). Ofrece dos modalidades de despliegue:

### Pinecone Serverless
- Sin gestión de infraestructura
- Escalado automático según la demanda
- Pago por uso (basado en lecturas/escrituras y almacenamiento)
- Ideal para prototipos y producción a cualquier escala
- Soporta múltiples índices con dimensiones configurables

### Pinecone Pod-based
- Mayor control sobre la infraestructura
- Rendimiento predecible
- Más adecuado para cargas de trabajo de alto rendimiento y baja latencia

## Conceptos clave en Pinecone

### Índice (Index)
Un índice es la unidad principal de almacenamiento en Pinecone. Al crear un índice se define:
- **Dimensión**: El número de dimensiones del vector (ej. 1536 para text-embedding-3-small de OpenAI)
- **Métrica**: La función de distancia (cosine, dotproduct, euclidean)
- **Tipo**: Serverless o Pod-based

### Namespace
Los namespaces permiten particionar los datos dentro de un mismo índice. Esto es útil para:
- Separar datos de diferentes clientes (multi-tenancy)
- Organizar documentos por tipo o categoría
- Ejecutar búsquedas aisladas sin interferencia entre particiones

### Metadata
Cada vector en Pinecone puede tener metadatos asociados en formato JSON. Los metadatos pueden usarse para:
- Filtrar resultados durante la búsqueda
- Almacenar el texto original del chunk (evitando consultas adicionales)
- Guardar información contextual como fuente, fecha, autor, categoría

## Otras bases de datos vectoriales populares

| Base de datos | Tipo | Características |
|--------------|------|----------------|
| Weaviate | Open source / Cloud | Grafos semánticos, módulos de ML |
| Qdrant | Open source / Cloud | Alto rendimiento, filtros avanzados |
| Chroma | Open source | Ligero, ideal para desarrollo local |
| Milvus | Open source / Cloud | Escalabilidad masiva |
| pgvector | Extensión PostgreSQL | Integración con BD relacional existente |

## Algoritmos ANN

Las bases de datos vectoriales utilizan algoritmos de vecinos más cercanos aproximados para escalar a millones de vectores:
- **HNSW** (Hierarchical Navigable Small World): Alta velocidad de consulta
- **IVF** (Inverted File Index): Balance entre velocidad y precisión
- **PQ** (Product Quantization): Compresión de vectores para reducir memoria
