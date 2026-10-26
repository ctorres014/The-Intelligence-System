# Resultados de los escenarios de prueba

| Escenario | Estado | Agentes | Iteraciones | Fuentes | Latencia |
|---|---|---|---|---|---|
| 1. RAG puro | completed | knowledge | 2 | 5 | 89.1 s |
| 2. RAG + cálculo | completed | knowledge, analyst | 3 | 5 | 184.2 s |
| 3. Investigación externa + RAG | completed | knowledge, research | 3 | 6 | 306.6 s |
| 4. Memoria multi-turno | completed | knowledge, analyst | 3 | 5 | 182.8 s |
| 5. HITL aprobado | completed | knowledge, analyst | 3 | 10 | 246.8 s |
| 6. HITL rechazado (tarea en background) | rejected | knowledge, research | 3 | 8 | 207.2 s |

## 1. RAG puro

**thread_id:** `d514e6a074f84842`

**Recorrido del grafo:**

- `supervisor` → **knowledge**: política RAG-first: anclar la respuesta en la base de conocimiento interna
- `knowledge`: 5 fragmentos recuperados. Resumen estructurado de los hallazgos en la base de conocimiento interna:

- **Definición de Reciprocal Rank Fusion (RRF):** Es el alg
- `supervisor` → **FINISH**: La base de conocimiento interna ya proporcionó una definición clara de Reciprocal Rank Fusion, su fórmula con la constante de suavizado y cómo combina los resul
- `synthesizer`: El **Reciprocal Rank Fusion (RRF)** es el algoritmo de fusión de rankings más comúnmente utilizado en la búsqueda híbrida [hybrid_search#6]. Su propósito princi

**Respuesta:**

El **Reciprocal Rank Fusion (RRF)** es el algoritmo de fusión de rankings más comúnmente utilizado en la búsqueda híbrida [hybrid_search#6]. Su propósito principal es combinar múltiples listas de resultados ordenados sin requerir normalización de puntuaciones (*scores*), lo que lo hace muy robusto ante distribuciones de puntajes diferentes y simple de implementar [hybrid_search#7].

### ¿Cómo combina BM25 con la búsqueda vectorial?

Para fusionar los resultados de un retriever léxico como **BM25** (basado en frecuencias de términos y TF-IDF) con los de una **búsqueda vectorial** (búsqueda semántica), RRF opera de la siguiente manera:

1. **Independencia de puntajes numéricos:** No compara ni depende de las métricas numéricas directas de cada buscador (las cuales pueden tener escalas completamente distintas), sino que evalúa **únicamente la posición relativa (el ranking)** en la que aparece cada documento dentro de cada lista [hybrid_search#6, hybrid_search#7].
2. **Aplicación de la fórmula de RRF:** Para cada documento $d$ encontrado en las distintas listas, se calcula un puntaje final acumulado utilizando la siguiente ecuación [hybrid_search#6]:

$$\text{RRF\_score}(d) = \sum \frac{1}{k + \text{rank}_i(d)}$$

Donde:
* $\text{rank}_i(d)$ es la posición (ranking) del documento $d$ en la lista generada por el retriever $i$ (por ejemplo, la lista de BM25 o la lista del retriever vectorial) [hybrid_search#6].
* $k$ es una constante de suavizado cuyo valor por defecto es $60$ [hybrid_search#6]. Evita que los documentos con posiciones muy altas dominen desproporcionadamente el puntaje y mitiga el impacto de los primeros puestos.

### Implementación práctica
En frameworks de desarrollo como **LangChain**, esta combinación se implementa de manera directa utilizando herramientas como `EnsembleRetriever`, permitiendo integrar un `BM25Retriever` y un retriever basado en vectores (por ejemplo, conectado a una base de datos vectorial) bajo este esquema de fusión [hybrid_search#5].

## 2. RAG + cálculo

**thread_id:** `calc-1793028343`

**Recorrido del grafo:**

- `supervisor` → **knowledge**: política RAG-first: anclar la respuesta en la base de conocimiento interna
- `knowledge`: 5 fragmentos recuperados. Resumen de hallazgos en la base de conocimiento interna:

- **Dimensiones de `text-embedding-3-small`**: El modelo posee **1536 dimens
- `supervisor` → **analyst**: cobertura: la consulta pide un cálculo explícito
- `analyst`: **Análisis y Resultados:**

1. **Dimensiones del modelo:**
   - Según la documentación interna (`embeddings_and_chunking#1` y `vector_databases#4`), el modelo `
- `supervisor` → **FINISH**: La base de conocimiento interna proporcionó las dimensiones y el analista completó todos los cálculos solicitados de almacenamiento. La información es completa.
- `synthesizer`: Según nuestra documentación, el modelo **`text-embedding-3-small`** posee **1536 dimensiones** [embeddings_and_chunking#1, vector_databases#4].

A continuación,

**Respuesta:**

Según nuestra documentación, el modelo **`text-embedding-3-small`** posee **1536 dimensiones** [embeddings_and_chunking#1, vector_databases#4].

A continuación, te presento el cálculo detallado del espacio de almacenamiento necesario para 2 millones de esos vectores en formato `float32` (4 bytes por dimensión):

### Supuestos del cálculo
* **Cantidad de vectores:** 2.000.000
* **Dimensiones por vector:** 1536
* **Tamaño por dimensión (`float32`):** 4 bytes

### Fórmula
$$\text{Espacio total} = \text{Cantidad de vectores} \times \text{Dimensiones} \times \text{Bytes}$$
$$\text{Espacio total} = 2.000.000 \times 1536 \times 4 = 12.288.000.000 \text{ bytes}$$

### Números finales según la unidad de medida
* **En bytes:** **12.288.000.000 bytes**
* **En Gigabytes (Decimal / Sistema Internacional: $1 \text{ GB} = 10^9 \text{ bytes}$):** **12,29 GB** 
* **En Gibibytes (Binario: $1 \text{ GiB} = 1024^3 \text{ bytes} \approx 1.073.741.824 \text{ bytes}$):** **11,44 GiB**

## 3. Investigación externa + RAG

**thread_id:** `aca16e4fc5804fbb`

**Recorrido del grafo:**

- `supervisor` → **knowledge**: política RAG-first: anclar la respuesta en la base de conocimiento interna
- `knowledge`: 6 fragmentos recuperados. Basándome en la base de conocimiento interna, aquí están los hallazgos relevantes:

* **Desarrolladores y sistema original de BM25:** 
- `supervisor` → **research**: La base de conocimiento interna no tiene información sobre quiénes desarrollaron el algoritmo BM25 ni el sistema original donde se usó. Es necesario buscar esta
- `research`: Resumen estructurado de la investigación sobre el algoritmo BM25 (Okapi BM25):

1. **Desarrolladores y base teórica:**
   - El algoritmo BM25 se basa en los mod
- `supervisor` → **FINISH**: Ya se cuenta con la información sobre los desarrolladores y el sistema original de BM25 (aportada por research) y sobre su rol en la búsqueda híbrida (aportada 
- `synthesizer`: Aquí tienes la respuesta a tu consulta basada en la base de conocimiento interna y la investigación externa:

### ¿Quiénes desarrollaron el algoritmo BM25 (Okap

**Respuesta:**

Aquí tienes la respuesta a tu consulta basada en la base de conocimiento interna y la investigación externa:

### ¿Quiénes desarrollaron el algoritmo BM25 (Okapi BM25) y en qué sistema se usó originalmente?

* **Desarrolladores:** El algoritmo BM25 se basa en modelos probabilísticos de recuperación de información iniciados por **Stephen E. Robertson** y **Karen Spärck Jones** en los años 70. La función final de BM25 y sus variantes (dentro de la familia *Best Matching*) fue desarrollada y refinada a lo largo de los años 80 y 90 por Stephen E. Robertson junto con otros investigadores como Steve Walker, Susan Jones, Micheline Hancock-Beaulieu y Mike Gatford.
* **Sistema original:** El nombre **"Okapi"** proviene del **sistema de recuperación de información Okapi**, un sistema experimental de búsqueda de texto completo desarrollado en la *City University de Londres* a principios de la década de 1980 (liderado por Stephen Walker y su equipo), que fue el lugar donde esta función de ranking se implementó y probó por primera vez de forma masiva (especialmente en las evaluaciones de TREC) [Wikipedia - Okapi BM25].

---

### Rol de BM25 en nuestra búsqueda híbrida

En nuestra arquitectura de búsqueda híbrida, BM25 cumple un rol fundamental como componente de **recuperación léxica (*sparse retrieval*)**:

* **Fundamento técnico:** Funciona como un algoritmo de ranking basado en TF-IDF con mejoras que incorporan los parámetros $k_1 = 1.5$ (para controlar la saturación de la frecuencia del término) y $b = 0.75$ (para aplicar una penalización por la longitud del documento) [hybrid_search#2, hybrid_search#4].
* **Fortalezas principales:** 
  * Proporciona **alta precisión en términos exactos** (como frases específicas, nombres propios, siglas, identificadores de errores como *"HTTPError 429"* o códigos como *"SKU-78923"*) [hybrid_search#0].
  * Funciona excelente con vocabulario técnico especializado [hybrid_search#1].
  * Al no requerir embeddings, permite un proceso de indexado más rápido [hybrid_search#1].
* **Complementariedad:** Se combina con la búsqueda vectorial (*dense retrieval* —ideal para capturar similitud semántica y conceptual, pero débil en coincidencias exactas y términos precisos) para unificar lo mejor de ambos mundos en la búsqueda híbrida [hybrid_search#0, hybrid_search#1].

## 4. Memoria multi-turno

**thread_id:** `calc-1793028343`

**Recorrido del grafo:**

- `supervisor` → **knowledge**: política RAG-first: anclar la respuesta en la base de conocimiento interna
- `knowledge`: 5 fragmentos recuperados. - El modelo `text-embedding-3-small` original tiene **1536 dimensiones** [embeddings_and_chunking#1, vector_databases#4].
- Si se usar
- `supervisor` → **analyst**: cobertura: la consulta pide un cálculo explícito
- `analyst`: $$\text{Espacio total} = 6.144.000.000 \text{ bytes}$$

En Gigabytes (sistema internacional / decimal):
$$\frac{6.144.000.000}{10^9} = 6,144 \text{ GB}$$

#### 
- `supervisor` → **FINISH**: Ya se consultó la base de conocimiento y el analista calculó tanto el espacio absoluto como el porcentaje de ahorro. Toda la información necesaria para responde
- `synthesizer`: Si usaras un modelo de **768 dimensiones** manteniendo el mismo tipo de dato (`float32`) y la misma cantidad de vectores (2 millones), el porcentaje de almacena

**Respuesta:**

Si usaras un modelo de **768 dimensiones** manteniendo el mismo tipo de dato (`float32`) y la misma cantidad de vectores (2 millones), el porcentaje de almacenamiento que ahorrarías sería exactamente del **50%** [embeddings_and_chunking#1, vector_databases#4].

### Supuestos del cálculo
* **Cantidad de vectores:** 2.000.000
* **Dimensiones originales (`text-embedding-3-small`):** 1536 dimensiones [embeddings_and_chunking#1, vector_databases#4]
* **Dimensiones nuevas:** 768 dimensiones
* **Tamaño por dimensión (`float32`):** 4 bytes

---

### Análisis cuantitativo y cálculos

#### 1. Porcentaje de ahorro
Como la cantidad de vectores y el tamaño en bytes por dimensión (`float32`) se mantienen constantes, el ahorro se calcula directamente a partir de la reducción en el número de dimensiones:

$$\text{Porcentaje de ahorro} = \left(1 - \frac{\text{Dimensiones nuevas}}{\text{Dimensiones originales}}\right) \times 100$$

$$\text{Porcentaje de ahorro} = \left(1 - \frac{768}{1536}\right) \times 100 = \mathbf{50\%}$$

#### 2. Comparativa de espacio de almacenamiento (para 2 millones de vectores)
* **Espacio original (1536 dimensiones):** 12,288 GB (12.288.000.000 bytes)
* **Espacio con el modelo nuevo (768 dimensiones):** 6,144 GB (6.144.000.000 bytes)

### Conclusión
Reducir las dimensiones a la mitad disminuye proporcionalmente el consumo de memoria a la mitad, liberando **6,144 GB** de espacio para tus 2 millones de vectores.

## 5. HITL aprobado

**thread_id:** `be2c89b887d64e5c`

**Recorrido del grafo:**

- `supervisor` → **knowledge**: política RAG-first: anclar la respuesta en la base de conocimiento interna
- `knowledge`: 10 fragmentos recuperados. En relación con la solicitud del usuario, la base de conocimiento interna no contiene una fórmula exacta o específica de cálculo mate
- `supervisor` → **analyst**: La base de conocimiento interna ya aportó las directrices del pipeline de ingesta y chunking, pero se necesita un cálculo cuantitativo exacto para estimar cuánt
- `human_approval` → **aprobado**: aprobado. Aprobado: ejecutar fuera del horario pico.
- `analyst`: Para cumplir con la instrucción del Supervisor y los requerimientos del análisis, presento el desglose cuantitativo y los supuestos técnicos utilizados:

### 1.
- `supervisor` → **FINISH**: El agente 'knowledge' ya proporcionó el contexto técnico del pipeline de ingesta y 'analyst' completó el cálculo cuantitativo requerido para los chunks. Toda la
- `synthesizer`: Aquí tienes el plan paso a paso y la estimación cuantitativa solicitada para borrar el índice de producción y reingestar tus documentos con chunks de 400 tokens

**Respuesta:**

Aquí tienes el plan paso a paso y la estimación cuantitativa solicitada para borrar el índice de producción y reingestar tus documentos con chunks de 400 tokens.

---

### 📊 Estimación Cuantitativa de Chunks

Para calcular la cantidad de fragmentos (*chunks*), se han considerado los siguientes **supuestos**:
* **Volumen total:** 1.500 páginas con un promedio de 500 tokens cada una ($1.500 \times 500 = 750.000$ tokens en total).
* **Tamaño del chunk:** 400 tokens fijos.

#### Resultados del cálculo:
1. **Sin solapamiento (*overlap* = 0%):**
   * **Cálculo:** $\frac{750.000 \text{ tokens}}{400 \text{ tokens/chunk}} =$ **1.875 chunks**.
2. **Con un solapamiento estándar del 10% al 15%** (recomendado para mantener contexto entre fragmentos consecutivos) [embeddings_and_chunking#4]:
   * **Estimación ajustada:** Aproximadamente entre **2.063 y 2.156 chunks**.

---

### 📋 Plan Paso a Paso para el Borrado y Reingesta

Dado que la operación cuenta con **aprobación humana** (con la condición de **ejecutarse fuera del horario pico**), seguiremos este flujo basado en el pipeline de ingesta estándar [rag_fundamentals#2] y la gestión de bases vectoriales [vector_databases#4]:

#### Paso 1: Planificación y Ventana de Mantenimiento
* Programar la ejecución en el horario de menor tráfico (fuera de pico) para mitigar impactos en los usuarios o aplicaciones que consuman el índice de producción.

#### Paso 2: Eliminación del Índice Vectorial Actual en Pinecone
* Conectarse a la consola de Pinecone o utilizar su API/SDK para eliminar el índice de producción obsoleto. 
* *Nota:* La base de conocimiento no detalla un script específico de borrado de índice en Pinecone, por lo que se debe utilizar el método estándar de la plataforma (`pc.delete_index("nombre-de-indice")` o desde el dashboard).

#### Paso 3: Reconfiguración y Recreación del Índice (si aplica)
* Crear un nuevo índice en Pinecone configurando los parámetros clave [vector_databases#4]:
  * **Dimensión:** Acorde al modelo de *embeddings* que utilices (ej. 1536 para OpenAI `text-embedding-3-small`) [vector_databases#4].
  * **Métrica:** Función de distancia requerida (ej. *cosine* o *euclidean*) [vector_databases#4].
  * **Tipo:** Serverless o Pod-based [vector_databases#4].

#### Paso 4: Actualización del Pipeline de Ingesta
* Configurar el script de división de documentos (*chunking*) para establecer el tamaño fijo en **400 tokens** [embeddings_and_chunking#4].

#### Paso 5: Ejecución de la Ingesta de Documentos
* **Carga:** Iniciar el proceso cargando las 1.500 páginas desde sus fuentes originales [rag_fundamentals#2].
* **Chunking:** Dividir el contenido aplicando el nuevo límite de 400 tokens [rag_fundamentals#2].
* **Embeddings:** Convertir los fragmentos resultantes (estimados entre 1.875 y ~2.156) en vectores de *embeddings* [rag_fundamentals#2].
* **Almacenamiento:** Guardar los vectores junto con sus metadatos correspondientes en el nuevo índice de Pinecone [rag_fundamentals#2].

#### Paso 6: Pruebas de Verificación
* Realizar consultas de prueba (*retrieval test*) para validar que la búsqueda semántica responde correctamente y que el índice operativo procesa las peticiones sin latencias anómalas.

## 6. HITL rechazado (tarea en background)

**thread_id:** `1ba3713044ee42a4`

**Recorrido del grafo:**

- `supervisor` → **knowledge**: política RAG-first: anclar la respuesta en la base de conocimiento interna
- `knowledge`: 8 fragmentos recuperados. Resumen estructurado de los hallazgos en la base de conocimiento interna:

- **Concepto de Namespace en Pinecone**: Los namespaces per
- `supervisor` → **research**: La base de conocimiento interna no tiene procedimientos específicos ni guías de migración para eliminar un namespace en producción en Pinecone, por lo que se re
- `research`: Como el entorno actual tiene la herramienta de búsqueda web deshabilitada y Wikipedia no contiene guías operativas detalladas de Pinecone sobre comandos específ
- `supervisor` → **FINISH**: Ya se cuenta con la información conceptual de Pinecone sobre namespaces y la estrategia general de eliminación a través de la API, y dado que no se requieren cá
- `human_approval` → **rechazado**: rechazado. Rechazado: no se borra producción sin ventana de mantenimiento.
- `synthesizer`: La solicitud fue marcada como crítica y un operador humano **rechazó** su ejecución, por lo que no se generó el plan ni el análisis solicitado.

Comentario del 

**Respuesta:**

La solicitud fue marcada como crítica y un operador humano **rechazó** su ejecución, por lo que no se generó el plan ni el análisis solicitado.

Comentario del operador: Rechazado: no se borra producción sin ventana de mantenimiento.
