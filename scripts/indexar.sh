#!/usr/bin/env bash
# Indexa el PDF de cosmovisión en ChromaDB.
# Ejecutar desde la raíz del proyecto rag-service.

set -e

cd "$(dirname "$0")/.."

if [[ ! -d "venv" ]]; then
    echo "✗ No se encontró venv/. Creá uno con: python3.11 -m venv venv"
    exit 1
fi

source venv/bin/activate

if [[ ! -f "data/cosmovision.pdf" ]]; then
    echo "✗ No se encontró data/cosmovision.pdf"
    echo "   Copialo allí antes de indexar."
    exit 1
fi

python -m src.indexer
