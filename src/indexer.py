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

import re
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


# ── Limpieza de texto del PDF (pre-chunking) ─────────────────────────────────
# El PDF de cosmovisión es prosa limpia (fuentes embebidas, no es OCR), pero
# arrastra tres artefactos que ensucian los chunks y degradan tanto el
# retrieval como la respuesta mostrada:
#   1. Encabezado de página repetido ("shipibo: territorio, historia y
#      cosmovisión") pegado al cuerpo -> produce "cosmovisiónvisión".
#   2. Números de página sueltos en su propia línea.
#   3. Guiones de corte de línea ("caste-\nllano" -> "castellano").
_HEADER_RE = re.compile(
    r"^\s*shipibo:\s*territorio,?\s*historia\s*y\s*cosmovisi[oó]n\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_PAGENUM_RE = re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)


def limpiar_texto_pdf(texto: str) -> str:
    """Quita encabezados/numeración repetida y reúne guiones de corte."""
    if not texto:
        return texto
    texto = _HEADER_RE.sub("", texto)
    texto = _PAGENUM_RE.sub("", texto)
    texto = re.sub(r"(\w+)-\n(\w+)", r"\1\2", texto)          # de-hyphenation
    texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)            # unir saltos simples
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# Palabras-función del español para medir si un chunk es prosa o tabla/glosario.
# La prosa narrativa tiene ~40% de palabras-función; las tablas bilingües del
# PDF (que pypdf linealiza como "ensalada" de columnas) tienen muy pocas.
_PALABRAS_FUNCION = set(
    "de la que el en y a los del se las por un para con no una su al lo como "
    "mas pero sus le ya o este si porque esta entre cuando muy sin sobre tambien "
    "me hasta hay donde quien desde todo nos durante todos uno les ni contra "
    "otros ese eso ante ellos esto mi antes algunos unos otro otras otra tanto "
    "esa estos mucho quienes nada muchos cual sea".split()
)

_UMBRAL_PROSA = 0.18


def _ratio_prosa(texto: str) -> float:
    """Fracción de palabras-función. Prosa ~0.4; tablas/glosarios < 0.15."""
    toks = re.findall(r"[a-záéíóúñ]+", texto.lower())
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in _PALABRAS_FUNCION) / len(toks)


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
    # Limpieza pre-chunking: encabezados repetidos, numeración y guiones de
    # corte. Se hace acá para que los embeddings se calculen sobre texto limpio
    # (mejora el retrieval, no solo lo que se muestra).
    for d in docs:
        d.page_content = limpiar_texto_pdf(d.page_content)
    log.info("Texto limpiado (encabezados, numeración, guiones de corte)")
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
    descartados_cortos = len(chunks) - len(utiles)

    # Filtrar chunks tipo tabla/glosario: el PDF tiene listas bilingües a
    # varias columnas que pypdf linealiza como "ensalada" de palabras. Estos
    # chunks ensucian el retrieval (matchean preguntas que no deberían) y dan
    # respuestas incoherentes. Se detectan por baja densidad de palabras-función.
    prosa = [c for c in utiles if _ratio_prosa(c.page_content) >= _UMBRAL_PROSA]
    descartados_tabla = len(utiles) - len(prosa)

    if descartados_cortos or descartados_tabla:
        log.info("Descartados %d chunks cortos y %d chunks tipo tabla/glosario",
                 descartados_cortos, descartados_tabla)
    return prosa


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