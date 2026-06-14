#!/usr/bin/env bash
# Arranca el servidor RAG en el puerto configurado (default 8001).
# Ejecutar desde la raíz del proyecto rag-service.

set -e

cd "$(dirname "$0")/.."

if [[ ! -d "venv" ]]; then
    echo "✗ No se encontró venv/. Creá uno con: python3.11 -m venv venv"
    exit 1
fi

source venv/bin/activate

# Cargar variables de .env si existe
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
fi

PORT=${RAG_PORT:-8001}
HOST=${RAG_HOST:-127.0.0.1}

echo "Arrancando rag-service en $HOST:$PORT"
echo "Documentación interactiva: http://$HOST:$PORT/docs"
echo ""

uvicorn src.app:app --host "$HOST" --port "$PORT" --reload
