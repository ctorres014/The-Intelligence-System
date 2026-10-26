#!/usr/bin/env bash
# Levanta todo el sistema (Redis + Phoenix + API) con una sola instrucción.
#   ./run.sh          → build + up + espera a que la API esté sana
#   ./run.sh down     → detiene los contenedores (los datos persisten en volúmenes)
#   ./run.sh logs     → sigue los logs de la API
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-up}" in
  down) docker compose down; exit 0 ;;
  logs) docker compose logs -f api; exit 0 ;;
esac

if [ ! -f .env ]; then
  cp .env.example .env
  echo "→ Se creó .env a partir de .env.example (editalo para usar Anthropic/OpenAI/Pinecone/Tavily)."
fi

# Lee una clave de .env sin ejecutar el archivo (los valores pueden tener espacios)
env_get() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- | sed -e 's/^"//' -e 's/"$//'; }
LLM_PROVIDER=$(env_get LLM_PROVIDER); EMBEDDING_PROVIDER=$(env_get EMBEDDING_PROVIDER)
API_PORT=$(env_get API_PORT); PHOENIX_PORT=$(env_get PHOENIX_PORT)

if [ "${LLM_PROVIDER:-ollama}" = "ollama" ] || [ "${EMBEDDING_PROVIDER:-ollama}" = "ollama" ]; then
  echo "→ Proveedor Ollama: verificá que Ollama esté corriendo en el host y tenga los modelos:"
  echo "    ollama pull $(env_get OLLAMA_MODEL) && ollama pull $(env_get OLLAMA_EMBEDDING_MODEL)"
fi

docker compose up --build -d

API_URL="http://localhost:${API_PORT:-8000}"
echo -n "→ Esperando a la API en $API_URL "
for _ in $(seq 1 90); do
  if curl -fsS "$API_URL/health" >/dev/null 2>&1; then
    echo " ✓"
    curl -s "$API_URL/health"; echo
    echo
    echo "  API (Swagger) : $API_URL/docs"
    echo "  Phoenix       : http://localhost:${PHOENIX_PORT:-6006}"
    echo "  Pruebas       : python scripts/run_scenarios.py"
    exit 0
  fi
  echo -n "."; sleep 2
done
echo
echo "La API no respondió a tiempo. Revisá: docker compose logs api"
exit 1
