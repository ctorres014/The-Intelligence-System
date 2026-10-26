# LangChain: Framework para aplicaciones LLM

## ¿Qué es LangChain?

LangChain es un framework open source en Python y JavaScript para construir aplicaciones potenciadas por modelos de lenguaje de gran escala. Provee abstracciones para encadenar llamadas a LLMs, gestionar memoria, integrar herramientas externas y construir agentes autónomos.

## Componentes principales

### Chains (Cadenas)
Las cadenas son la abstracción central de LangChain. Una cadena combina múltiples componentes en un flujo de trabajo:
- `LLMChain`: LLM + prompt template
- `RetrievalQA`: Recuperador + LLM para responder preguntas
- `ConversationalRetrievalChain`: Historial de conversación + recuperador + LLM

Con la llegada de LangChain Expression Language (LCEL), las cadenas se construyen de forma más declarativa usando el operador `|`.

### Retrievers
Los retrievers son componentes que reciben una consulta y devuelven documentos relevantes. Tipos principales:
- **VectorStoreRetriever**: Busca por similitud vectorial
- **BM25Retriever**: Búsqueda léxica BM25
- **EnsembleRetriever**: Combina múltiples retrievers
- **MultiQueryRetriever**: Genera variaciones de la consulta para mejorar el recall
- **ContextualCompressionRetriever**: Reranking y compresión de contexto

### Document Loaders
LangChain incluye loaders para decenas de formatos y fuentes:
- `PyPDFLoader`: Archivos PDF
- `UnstructuredMarkdownLoader`: Archivos Markdown
- `JSONLoader`: Archivos JSON con jq paths
- `WebBaseLoader`: Páginas web
- `GitLoader`: Repositorios de código

### Text Splitters
Splitters para fragmentar documentos:
- `RecursiveCharacterTextSplitter`: El más recomendado para texto general
- `MarkdownHeaderTextSplitter`: Respeta estructura de Markdown
- `TokenTextSplitter`: Divide exactamente por tokens

### Vector Stores
Integraciones con bases de datos vectoriales:
- `PineconeVectorStore`: Integración con Pinecone
- `Chroma`: Base de datos local en memoria o persistente
- `FAISS`: Búsqueda vectorial en memoria de Facebook
- `Weaviate`, `Qdrant`, `Milvus`

## LangChain Expression Language (LCEL)

LCEL es la forma moderna de construir pipelines en LangChain usando el operador pipe `|`:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Responde en español: {question}")
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | llm | parser
response = chain.invoke({"question": "¿Qué es RAG?"})
```

## Integración con Pinecone

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="rag-coderhouse",
    embedding=embeddings,
    namespace="documents"
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
```

## LangSmith: Observabilidad

LangSmith es la plataforma de trazabilidad y evaluación para aplicaciones LangChain. Permite:
- Visualizar cada paso de una cadena
- Medir latencia y costos por llamada
- Crear datasets de evaluación
- Comparar distintas versiones de prompts
