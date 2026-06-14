#!/usr/bin/env python3
"""
indexer.py — Construye el índice ChromaDB del PDF de cosmovisión shipibo.

Implementación con LangChain moderno (PyPDFLoader + RecursiveCharacterTextSplitter
+ langchain-chroma + langchain-huggingface). Esto es posible porque este
servidor corre en SU PROPIO venv, sin restricciones de Rasa.

Ejecución (UNA SOLA VEZ, o cuando cambie el PDF):
    python -m src.indexer
    # o bien:
    bash scripts/indexar.sh

Tiempo estimado: 1-3 minutos (primera vez descarga el modelo de embeddings,
                ~90 MB; siguientes corridas usan caché local).
"""

import sys
import shutil
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("indexer")


def cargar_pdf() -> list:
    """Carga el PDF como documentos LangChain (uno por página)."""
    if not config.PDF_PATH.exists():
        log.error("PDF no encontrado en %s", config.PDF_PATH)
        log.error("Copialo a esa ruta y volvé a ejecutar.")
        sys.exit(1)
    log.info("Leyendo PDF: %s (%.1f MB)",
             config.PDF_PATH.name, config.PDF_PATH.stat().st_size / (1024 * 1024))
    loader = PyPDFLoader(str(config.PDF_PATH))
    docs = loader.load()
    log.info("Páginas cargadas: %d", len(docs))
    return docs


def chunkear(docs: list) -> list:
    """
    Divide los documentos en chunks de tamaño manejable.
    RecursiveCharacterTextSplitter intenta cortar en separadores naturales
    (párrafos, oraciones) antes que partir a la fuerza.
    """
    log.info("Chunkeando con chunk_size=%d, overlap=%d...",
             config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    log.info("Chunks generados: %d", len(chunks))

    # Filtrar chunks muy cortos (headers, pies de página, números sueltos)
    utiles = [c for c in chunks if len(c.page_content.strip()) >= 80]
    descartados = len(chunks) - len(utiles)
    if descartados:
        log.info("Descartados %d chunks < 80 caracteres", descartados)
    return utiles


def construir_indice(chunks: list) -> Chroma:
    """Construye el índice ChromaDB persistente, sobrescribiendo el anterior."""
    if config.INDEX_DIR.exists():
        log.info("Borrando índice previo en %s", config.INDEX_DIR)
        shutil.rmtree(config.INDEX_DIR)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Cargando modelo de embeddings: %s", config.EMBED_MODEL)
    log.info("  (primera ejecución descarga ~90 MB del modelo, paciencia)")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    log.info("Construyendo índice ChromaDB (genera embeddings para cada chunk)...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION,
        persist_directory=str(config.INDEX_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    log.info("Índice persistido")
    return db


def probar(db: Chroma) -> None:
    """Prueba el índice con consultas representativas."""
    log.info("\n══ Pruebas de búsqueda ══")
    consultas = [
        "¿qué es Ronin?",
        "espíritus del agua",
        "el kené y su significado",
        "ayahuasca y curanderos",
        "cosmovisión shipibo",
    ]
    for q in consultas:
        resultados = db.similarity_search_with_score(q, k=1)
        if resultados:
            doc, score = resultados[0]
            preview = doc.page_content.replace("\n", " ")[:120]
            pagina = doc.metadata.get("page", "?")
            log.info("  '%s' (dist=%.3f, pág. %s)", q, score, pagina)
            log.info("    → %s...", preview)
        else:
            log.info("  '%s' → sin resultados", q)


def main():
    log.info("═══════════════════════════════════════════════════════════")
    log.info("  Indexación del PDF de cosmovisión shipibo")
    log.info("═══════════════════════════════════════════════════════════")

    docs = cargar_pdf()
    chunks = chunkear(docs)
    db = construir_indice(chunks)
    probar(db)

    tam = sum(f.stat().st_size for f in config.INDEX_DIR.rglob("*") if f.is_file())
    log.info("\n══ Resumen ══")
    log.info("  Páginas procesadas:  %d", len(docs))
    log.info("  Chunks indexados:    %d", len(chunks))
    log.info("  Tamaño del índice:   %.1f MB", tam / (1024 * 1024))
    log.info("  Ubicación:           %s", config.INDEX_DIR)
    log.info("\n✓ Índice listo. Arrancá el servidor con: bash scripts/arrancar.sh")


if __name__ == "__main__":
    main()