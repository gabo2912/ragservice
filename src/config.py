"""
config.py — Configuración centralizada del servidor RAG.

Lee variables de .env (si existe) o de variables de entorno del sistema.
Provee valores por defecto razonables para que el servidor arranque
incluso sin .env configurado.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env si existe (no falla si no está)
ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(ROOT / ".env")

# ── Servidor HTTP ────────────────────────────────────────────────────────────
HOST = os.getenv("RAG_HOST", "127.0.0.1")
PORT = int(os.getenv("RAG_PORT", "8001"))

# ── Comportamiento del responder ─────────────────────────────────────────────
MODO_DEFAULT = os.getenv("RAG_MODO_DEFAULT", "simple")  # "simple" | "llm"
SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "1.2"))

# ── Modelo de embeddings ─────────────────────────────────────────────────────
# Acepta tanto identificador de HuggingFace ("paraphrase-multilingual-MiniLM-L6-v2")
# como una ruta local absoluta a una carpeta con el modelo descargado
# (útil si tu red bloquea HuggingFace o querés trabajar offline).
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ── Parámetros de chunking (usados por indexer.py) ──────────────────────────
# CHUNK_SIZE reducido de 900 a 600 para chunks más enfocados (mejor calidad
# de respuesta en chatbot). Si cambiás este valor, hay que re-indexar:
#     bash scripts/indexar.sh
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

# ── Rutas ────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"
PDF_PATH = DATA_DIR / "cosmovision.pdf"
INDEX_DIR = DATA_DIR / "chroma_index"

# Nombre de la colección dentro de ChromaDB
COLLECTION = "cosmovision_shipibo"

# ── Configuración del LLM (Camino A) ─────────────────────────────────────────
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3.5:latest")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))