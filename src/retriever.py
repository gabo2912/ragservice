"""
retriever.py — Carga del índice ChromaDB y búsqueda de chunks.

Singleton perezoso: el índice y el modelo de embeddings se cargan UNA SOLA VEZ
al primer uso, y quedan en memoria para queries siguientes (latencia <200ms
después de la carga inicial).

Implementación con LangChain (langchain-chroma + langchain-huggingface).
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from . import config

logger = logging.getLogger(__name__)

# ── Estado interno (caché perezosa) ───────────────────────────────────────────
_db: Optional[Chroma] = None
_carga_intentada: bool = False
_carga_ok: bool = False
_carga_error: Optional[str] = None


def _cargar_indice() -> bool:
    """
    Carga el índice ChromaDB en memoria (una sola vez).
    Devuelve True si está disponible, False si hubo error.
    """
    global _db, _carga_intentada, _carga_ok, _carga_error

    if _carga_intentada:
        return _carga_ok

    _carga_intentada = True

    if not config.INDEX_DIR.exists():
        msg = f"Índice no encontrado en {config.INDEX_DIR}. Ejecutá: python -m src.indexer"
        _carga_error = msg
        logger.warning("retriever: %s", msg)
        return False

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _db = Chroma(
            collection_name=config.COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(config.INDEX_DIR),
        )
        _carga_ok = True
        n = _db._collection.count() if _db._collection else 0
        logger.info("retriever: índice cargado (%d chunks) desde %s", n, config.INDEX_DIR)
        return True
    except Exception as e:
        msg = f"error al cargar índice: {e}"
        _carga_error = msg
        logger.exception("retriever: %s", msg)
        return False


def disponible() -> bool:
    """True si el índice está cargado y listo para consultar."""
    return _cargar_indice()


def estado() -> Dict[str, Any]:
    """Devuelve estado del retriever para el endpoint /health y /stats."""
    ok = _cargar_indice()
    if not ok:
        return {"ok": False, "error": _carga_error, "chunks": 0}
    try:
        n = _db._collection.count() if _db and _db._collection else 0
    except Exception:
        n = 0
    return {
        "ok": True,
        "chunks": n,
        "collection": config.COLLECTION,
        "embed_model": config.EMBED_MODEL,
        "index_dir": str(config.INDEX_DIR),
    }


def buscar_chunks(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Busca los k chunks más relevantes del PDF para la consulta dada.

    Args:
        query: pregunta del usuario en lenguaje natural
        k: número de chunks a devolver (default 3)

    Returns:
        Lista de dicts con keys: 'texto', 'pagina', 'score' (menor=mejor).
        Lista vacía si el índice no está disponible o no hay resultados.
    """
    if not _cargar_indice():
        return []
    if not query or not query.strip():
        return []

    try:
        resultados = _db.similarity_search_with_score(query, k=k)
    except Exception as e:
        logger.warning("retriever: error en búsqueda: %s", e)
        return []

    chunks = []
    for doc, score in resultados:
        chunks.append({
            "texto":  (doc.page_content or "").strip(),
            "pagina": doc.metadata.get("page", "?"),
            "score":  float(score),
        })
    return chunks
