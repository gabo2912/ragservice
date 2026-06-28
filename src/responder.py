"""
responder.py — Genera respuestas a partir de los chunks recuperados.

Implementa dos modos:
  - "simple" (Camino B, ACTIVO): retrieval + framing textual. Cero LLM,
    cero alucinaciones, respuestas verificables literalmente del PDF.
  - "llm"    (Camino A, PREPARADO): retrieval + síntesis con LLM local
    vía Ollama. Activable cambiando RAG_MODO_DEFAULT=llm en .env, o
    pasando modo="llm" en la request HTTP.

Ver docs/rag_camino_a.md para el procedimiento completo de migración B→A.

── MEJORA DE CALIDAD (multi-chunk) ───────────────────────────────────────────
Antes responder_simple() recuperaba UN solo chunk (k=1) y respondía con ese o
nada. Con CHUNK_SIZE=600 una respuesta suele quedar partida entre 2 chunks
contiguos, y la versión anterior tiraba el segundo. Ahora:
  1. Recupera config.RETRIEVE_K chunks (default 4).
  2. Conserva los que pasan el umbral de relevancia.
  3. Combina hasta config.COMBINAR_TOP_N pasajes (en orden de lectura del PDF)
     en una sola respuesta coherente, de-duplicando el solape del chunking.
Esto sube la tasa de acierto sin re-indexar.
"""

import re
import random
import logging
from typing import Optional, List, Dict, Any

from . import config
from .retriever import buscar_chunks

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Pulido de chunks (sin IA) — limpieza pre-respuesta
# ═══════════════════════════════════════════════════════════════════════════

_MAX_ORACIONES = 3
_MAX_CHARS = 450

_REGEX_INICIO_ORACION = re.compile(r"[A-ZÁÉÍÓÚÑ¿¡]|^\d+\.\s")

_PATRONES_RUIDO = [
    re.compile(r"\(?\b[Pp]ág(?:ina)?\.?\s*\d+\b\)?"),    # "pág. 23", "(página 87)"
    re.compile(r"\[\d+\]"),                              # "[12]" (referencias)
    re.compile(r"\(\s*\d+\s*\)"),                        # "(23)" números sueltos
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),            # líneas con solo número
    re.compile(r"^\s*[A-Z\s]{8,}\s*$", re.MULTILINE),    # líneas TODO MAYÚSCULAS
]


def _pulir_chunk(texto: str, max_oraciones: int = _MAX_ORACIONES,
                 max_chars: int = _MAX_CHARS) -> str:
    """Limpia un chunk del PDF para que se vea como respuesta natural."""
    if not texto:
        return texto

    t = re.sub(r"(\w+)-\n(\w+)", r"\1\2", texto)   # reunir guiones de corte
    t = re.sub(r"\n(?!\n)", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    for patron in _PATRONES_RUIDO:
        t = patron.sub("", t)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r"\s+", " ", t).strip()

    if t and not _REGEX_INICIO_ORACION.match(t):
        m = re.search(r"[.!?¿¡]\s+(?=[A-ZÁÉÍÓÚÑ¿¡])", t)
        if m:
            t = t[m.end():]

    if t and t[-1] not in ".!?":
        ult = max(t.rfind(s) for s in (".", "!", "?"))
        if ult > 0:
            t = t[:ult + 1]

    oraciones = re.split(r"(?<=[.!?])\s+", t)
    if len(oraciones) > max_oraciones:
        t = " ".join(oraciones[:max_oraciones])

    if len(t) > max_chars:
        recortado = t[:max_chars]
        ult = max(recortado.rfind(s) for s in (".", "!", "?"))
        if ult > 100:
            t = recortado[:ult + 1]
        else:
            t = recortado.rstrip() + "…"

    return t.strip()


def _dedup_oraciones(a: str, b: str) -> str:
    """
    Devuelve `b` sin las oraciones iniciales que ya aparecen al final de `a`.
    El chunk_overlap del indexer hace que oraciones enteras se repitan entre
    chunks contiguos; esto las elimina para que la respuesta combinada no
    repita frases. Si `b` queda vacío, devuelve "".
    """
    if not a or not b:
        return b
    def _norm(s):
        return re.sub(r"\s+", " ", s).strip().lower()
    cola = set(_norm(o) for o in re.split(r"(?<=[.!?])\s+", a) if len(o.strip()) > 15)
    ors_b = re.split(r"(?<=[.!?])\s+", b)
    # saltar oraciones iniciales de b que ya están en la cola de a
    i = 0
    while i < len(ors_b) and _norm(ors_b[i]) in cola:
        i += 1
    return " ".join(ors_b[i:]).strip()


def _solapan(a: str, b: str, min_solape: int = 40) -> bool:
    """True si el final de `a` se solapa textualmente con el inicio de `b`."""
    if not a or not b:
        return False
    cola = re.sub(r"\s+", " ", a[-min_solape * 3:]).strip().lower()
    cabeza = re.sub(r"\s+", " ", b[:min_solape * 3]).strip().lower()
    for n in range(min(len(cola), len(cabeza)), min_solape - 1, -1):
        if cola[-n:] == cabeza[:n]:
            return True
    return False


def _combinar_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Combina hasta config.COMBINAR_TOP_N chunks relevantes en texto coherente."""
    if not chunks:
        return ""

    def _pag_key(c):
        p = c.get("pagina", "?")
        return (1, 0) if not isinstance(p, int) else (0, p)

    elegidos = sorted(chunks, key=_pag_key)

    partes: List[str] = []
    total = 0
    limite_total = config.RESPUESTA_MAX_CHARS
    for c in elegidos:
        pulido = _pulir_chunk(
            c["texto"],
            max_oraciones=_MAX_ORACIONES if not partes else 2,
            max_chars=_MAX_CHARS if not partes else 280,
        )
        if not pulido:
            continue
        # Quitar oraciones de este chunk que ya aparecen al final del anterior
        # (consecuencia del chunk_overlap del indexer).
        if partes:
            pulido = _dedup_oraciones(partes[-1], pulido)
            if not pulido or _solapan(partes[-1], pulido):
                continue
        if total + len(pulido) > limite_total and partes:
            break
        partes.append(pulido)
        total += len(pulido)
        if len(partes) >= config.COMBINAR_TOP_N:
            break

    return " ".join(partes).strip()


# ═══════════════════════════════════════════════════════════════════════════
# CAMINO B (ACTIVO) — retrieval puro con framing textual
# ═══════════════════════════════════════════════════════════════════════════

_MARCOS_INTRODUCCION = [
    "📚 Sobre eso, el documento de cosmovisión shipiba dice:\n\n{texto}",
    "📚 En el documento cultural encontré este pasaje relevante:\n\n{texto}",
    "📚 Te comparto lo que el documento de cosmovisión cuenta:\n\n{texto}",
    "📚 Mira lo que dice el documento cultural sobre tu pregunta:\n\n{texto}",
    "📚 Esto es lo que el documento de cosmovisión describe:\n\n{texto}",
]

_MARCOS_CIERRE = [
    "\n\n¿Quieres que busque otro pasaje sobre este tema?",
    "\n\n¿Te muestro otro fragmento del documento?",
    "\n\n¿Algo más sobre cosmovisión que quieras explorar?",
    "",
]


def responder_simple(query: str) -> Dict[str, Any]:
    """
    Camino B: retrieval multi-chunk + framing.

    Recupera varios chunks, conserva los que pasan el umbral de relevancia y
    combina los mejores en una respuesta coherente. Reduce respuestas truncadas
    por el chunking.

    Returns:
        Dict con keys 'respuesta' (str|None) y 'chunks' (list).
        respuesta es None si NINGÚN chunk pasa el umbral.
    """
    chunks = buscar_chunks(query, k=config.RETRIEVE_K)
    if not chunks:
        return {"respuesta": None, "chunks": []}

    relevantes = [c for c in chunks if c["score"] <= config.SCORE_THRESHOLD]

    if not relevantes:
        mejor = chunks[0]
        logger.debug(
            "responder: query=%r sin match relevante (mejor score=%.3f > threshold=%.3f)",
            query, mejor["score"], config.SCORE_THRESHOLD,
        )
        return {"respuesta": None, "chunks": chunks}

    texto_combinado = _combinar_chunks(relevantes)
    if not texto_combinado:
        return {"respuesta": None, "chunks": chunks}

    marco_intro = random.choice(_MARCOS_INTRODUCCION)
    marco_cierre = random.choice(_MARCOS_CIERRE)
    respuesta = marco_intro.format(texto=texto_combinado) + marco_cierre

    return {"respuesta": respuesta, "chunks": relevantes}


# ═══════════════════════════════════════════════════════════════════════════
# CAMINO A (PREPARADO, DESACTIVADO) — retrieval + síntesis con LLM local
# ═══════════════════════════════════════════════════════════════════════════
#
# Para activar el Camino A:
#   1. Instalar Ollama y descargar el modelo:   ollama pull phi3.5
#   2. Verificar:                                curl http://localhost:11434/
#   3. En requirements.txt, descomentar:         langchain-ollama>=0.2.0
#      Reinstalar:                               pip install -r requirements.txt
#   4. Descomentar el bloque de responder_llm() más abajo y borrar el stub.
#   5. En .env cambiar:                          RAG_MODO_DEFAULT=llm
#
# Ver docs/rag_camino_a.md para el procedimiento completo (15 minutos).
# ─────────────────────────────────────────────────────────────────────────
#
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
#
# _PROMPT_CULTURAL = ChatPromptTemplate.from_messages([
#     ("system",
#      "Eres Pishico, un asistente educativo cultural shipibo-konibo. "
#      "Responde la pregunta del usuario usando ÚNICAMENTE la información "
#      "del CONTEXTO proporcionado. Si el contexto no contiene la respuesta, "
#      "responde 'No encontré información sobre eso en el documento de cosmovisión'. "
#      "Mantén la respuesta breve (3 oraciones máximo) y en español neutro.\n\n"
#      "CONTEXTO:\n{context}"),
#     ("human", "{question}"),
# ])
#
# _llm = None
#
# def _get_llm():
#     global _llm
#     if _llm is None:
#         base_url = config.OLLAMA_URL.rstrip("/")
#         if base_url.endswith("/api"):
#             base_url = base_url[:-4]
#         _llm = ChatOllama(
#             model=config.OLLAMA_MODEL,
#             base_url=base_url,
#             temperature=config.OLLAMA_TEMPERATURE,
#             timeout=config.OLLAMA_TIMEOUT,
#         )
#     return _llm
#
# def responder_llm(query: str) -> Dict[str, Any]:
#     """Camino A: retrieval + síntesis con phi3.5 vía Ollama. Cae a B si falla."""
#     chunks = buscar_chunks(query, k=config.RETRIEVE_K)
#     relevantes = [c for c in chunks if c["score"] <= config.SCORE_THRESHOLD]
#     if not relevantes:
#         return {"respuesta": None, "chunks": chunks}
#     contexto = "\n\n---\n\n".join(c["texto"] for c in relevantes)
#     try:
#         llm = _get_llm()
#         chain = _PROMPT_CULTURAL | llm
#         resp = chain.invoke({"context": contexto, "question": query})
#         return {"respuesta": resp.content, "chunks": relevantes}
#     except Exception as e:
#         logger.warning("Camino A LLM falló (%s), cayendo a Camino B", e)
#         return responder_simple(query)


def responder_llm(query: str) -> Dict[str, Any]:
    """Stub: Camino A no está activo. Ver docs/rag_camino_a.md para activarlo."""
    raise NotImplementedError(
        "Camino A (RAG con LLM) no está activado. "
        "Ver docs/rag_camino_a.md para el procedimiento de activación."
    )


# ═══════════════════════════════════════════════════════════════════════════
# DISPATCHER — selecciona el modo según parámetro
# ═══════════════════════════════════════════════════════════════════════════

def responder(query: str, modo: str = None) -> Dict[str, Any]:
    """Dispatcher principal. Selecciona simple o llm; cae a MODO_DEFAULT si None."""
    modo_efectivo = modo or config.MODO_DEFAULT
    if modo_efectivo not in {"simple", "llm"}:
        logger.warning("modo desconocido %r, usando 'simple'", modo_efectivo)
        modo_efectivo = "simple"

    if modo_efectivo == "llm":
        try:
            result = responder_llm(query)
            result["modo_usado"] = "llm"
            return result
        except NotImplementedError:
            logger.warning("Camino A no activado, cayendo a 'simple'")
            result = responder_simple(query)
            result["modo_usado"] = "simple"
            return result

    result = responder_simple(query)
    result["modo_usado"] = "simple"
    return result