"""
responder.py — Genera respuestas a partir de los chunks recuperados.

Implementa dos modos:
  - "simple" (Camino B, ACTIVO): retrieval + framing textual. Cero LLM,
    cero alucinaciones, respuestas verificables literalmente del PDF.
  - "llm"    (Camino A, PREPARADO): retrieval + síntesis con LLM local
    vía Ollama. Activable cambiando RAG_MODO_DEFAULT=llm en .env, o
    pasando modo="llm" en la request HTTP. Requiere descomentar el bloque
    de Camino A más abajo y reinstalar con langchain-ollama.

Ver docs/rag_camino_a.md para el procedimiento completo de migración B→A.
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
#
# Los chunks del PDF llegan en bruto: pueden empezar/terminar a mitad de
# oración, tener saltos de línea espurios, números de página, referencias
# académicas, etc. Esta función los pule SIN IA para que la respuesta del
# bot se vea natural y bien formada.

# Cuánto del chunk dejar como máximo. Más corto = más enfocado.
_MAX_ORACIONES = 3
_MAX_CHARS = 450

# Regex para detectar inicio de oración: empieza con mayúscula (con o sin
# apertura de signo, comilla, etc.). Toleramos números también, ej. "2. Algo..."
_REGEX_INICIO_ORACION = re.compile(r"[A-ZÁÉÍÓÚÑ¿¡]|^\d+\.\s")

# Patrones de ruido típicos del PDF que conviene eliminar
_PATRONES_RUIDO = [
    re.compile(r"\(?\b[Pp]ág(?:ina)?\.?\s*\d+\b\)?"),    # "pág. 23", "(página 87)"
    re.compile(r"\[\d+\]"),                              # "[12]" (referencias)
    re.compile(r"\(\s*\d+\s*\)"),                        # "(23)" números sueltos
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),            # líneas con solo número
    re.compile(r"^\s*[A-Z\s]{8,}\s*$", re.MULTILINE),    # líneas TODO MAYÚSCULAS (headers)
]


def _pulir_chunk(texto: str) -> str:
    """
    Limpia un chunk del PDF para que se vea como respuesta natural:
      1. Une saltos de línea internos en espacios
      2. Quita ruido (números de página, referencias)
      3. Recorta texto colgante al inicio (si empieza a mitad de oración)
      4. Recorta texto colgante al final (si termina a mitad de oración)
      5. Limita a N oraciones / M caracteres
    """
    if not texto:
        return texto

    # 1. Saltos de línea internos → espacios; preservamos párrafos (\n\n)
    t = re.sub(r"\n(?!\n)", " ", texto)
    t = re.sub(r"\s+", " ", t).strip()

    # 2. Quitar patrones de ruido
    for patron in _PATRONES_RUIDO:
        t = patron.sub("", t)
    # Limpiar espacios sueltos antes de signos de puntuación (quedan tras
    # remover paréntesis/referencias)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    # Limpiar comas o signos duplicados (ej. ",,", ", ." que pueden surgir)
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r"\s+", " ", t).strip()

    # 3. Recortar texto colgante al inicio: si no empieza con mayúscula o ¿/¡,
    # buscar la primera oración completa
    if t and not _REGEX_INICIO_ORACION.match(t):
        # Buscar el primer punto/fin de oración seguido de espacio + mayúscula
        m = re.search(r"[.!?¿¡]\s+(?=[A-ZÁÉÍÓÚÑ¿¡])", t)
        if m:
            t = t[m.end():]

    # 4. Recortar texto colgante al final: cortar hasta el último punto/fin
    # de oración. Si no hay puntos al final, dejar tal cual (mejor algo que nada).
    if t and t[-1] not in ".!?":
        # buscar último signo de cierre seguido de espacio o fin de string
        ult = max(t.rfind(s) for s in (".", "!", "?"))
        if ult > 0:
            t = t[:ult + 1]

    # 5. Limitar a N oraciones y/o M caracteres (lo que ocurra primero)
    # Partir por delimitadores conservando los signos finales
    oraciones = re.split(r"(?<=[.!?])\s+", t)
    if len(oraciones) > _MAX_ORACIONES:
        t = " ".join(oraciones[:_MAX_ORACIONES])

    if len(t) > _MAX_CHARS:
        # Cortar en la última oración completa antes del límite
        recortado = t[:_MAX_CHARS]
        ult = max(recortado.rfind(s) for s in (".", "!", "?"))
        if ult > 100:  # asegurar que sobre algo útil
            t = recortado[:ult + 1]
        else:
            t = recortado.rstrip() + "…"

    return t.strip()


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
    "",  # variabilidad: a veces sin pregunta de cierre
]


def responder_simple(query: str) -> Dict[str, Any]:
    """
    Camino B: retrieval puro + framing.

    Args:
        query: pregunta del usuario

    Returns:
        Dict con keys: 'respuesta' (str|None), 'chunks' (list)
        - respuesta es None si no hay match relevante (score > threshold)
    """
    chunks = buscar_chunks(query, k=1)
    if not chunks:
        return {"respuesta": None, "chunks": []}

    mejor = chunks[0]
    if mejor["score"] > config.SCORE_THRESHOLD:
        logger.debug("responder: query=%r descartada (score=%.3f > threshold=%.3f)",
                     query, mejor["score"], config.SCORE_THRESHOLD)
        return {"respuesta": None, "chunks": chunks}

    marco_intro = random.choice(_MARCOS_INTRODUCCION)
    marco_cierre = random.choice(_MARCOS_CIERRE)
    # Pulir el chunk antes de inyectarlo en el marco textual: cortar
    # oraciones incompletas al inicio/fin, limpiar saltos de línea, quitar
    # ruido de paginación y limitar a 3 oraciones máximo.
    texto_pulido = _pulir_chunk(mejor["texto"])
    respuesta = marco_intro.format(texto=texto_pulido) + marco_cierre

    return {"respuesta": respuesta, "chunks": chunks}


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
#      (O en cada request HTTP pasar modo="llm")
#
# Ver docs/rag_camino_a.md para el procedimiento completo (15 minutos).
#
# El siguiente código está LISTO para descomentar y funcionar:
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
#         # Parsear URL de Ollama (puede venir con o sin /api)
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
#     """
#     Camino A: retrieval + síntesis con phi3.5 vía Ollama.
#     Si Ollama falla, cae al Camino B.
#     """
#     chunks = buscar_chunks(query, k=3)
#     if not chunks or chunks[0]["score"] > config.SCORE_THRESHOLD:
#         return {"respuesta": None, "chunks": chunks}
#     contexto = "\n\n---\n\n".join(c["texto"] for c in chunks)
#     try:
#         llm = _get_llm()
#         chain = _PROMPT_CULTURAL | llm
#         resp = chain.invoke({"context": contexto, "question": query})
#         return {"respuesta": resp.content, "chunks": chunks}
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
    """
    Dispatcher principal. Selecciona simple o llm según el parámetro modo,
    cayendo al MODO_DEFAULT de .env si no se especifica.

    Args:
        query: pregunta del usuario
        modo: "simple" | "llm" | None (usa MODO_DEFAULT)

    Returns:
        Dict con keys: 'respuesta', 'chunks', 'modo_usado'
    """
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