"""
responder.py — Genera respuestas a partir de los chunks recuperados.

Implementa dos modos:
  - "simple" (Camino B, ACTIVO): retrieval + framing textual. Cero LLM,
    cero alucinaciones, respuestas verificables literalmente del PDF.
  - "llm"    (Camino A, ACTIVO con Gemini): retrieval + síntesis con la API
    de Google Gemini (nube). Activable con RAG_MODO_DEFAULT=llm en .env, o
    pasando modo="llm" en la request HTTP. Requiere GEMINI_API_KEY en .env.
    Restringido al contexto recuperado para no alucinar; cae al Camino B si
    el LLM falla, se agota el rate limit, o el contexto no responde.

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
# CAMINO A (ACTIVO con Gemini) — retrieval + síntesis con LLM en la nube
# ═══════════════════════════════════════════════════════════════════════════
#
# Usa la API de Google Gemini (Google AI Studio) para sintetizar una respuesta
# a partir de los chunks recuperados del PDF. A diferencia del Camino B (que
# muestra el pasaje tal cual), el Camino A redacta una respuesta en lenguaje
# natural, PERO restringida al contexto recuperado para no alucinar.
#
# Requisitos:
#   1. pip install google-genai  (ver requirements.txt)
#   2. En .env:  GEMINI_API_KEY=<tu_key>  y  RAG_MODO_DEFAULT=llm
#
# Seguridad anti-alucinación: el prompt obliga al modelo a responder SOLO con
# el contexto; si el contexto no alcanza, devuelve una señal de "sin datos" y
# caemos al Camino B. Además, si no hay chunks relevantes (bajo el umbral),
# ni siquiera se llama al LLM.
#
# Rate limit (free tier): las llamadas se reintentan con backoff exponencial
# ante error 429 (RESOURCE_EXHAUSTED). Si se agotan los reintentos, cae a B.
# ─────────────────────────────────────────────────────────────────────────

import time

_SIN_DATOS = "SIN_DATOS_EN_CONTEXTO"

_PROMPT_SISTEMA = (
    "Eres Pishico, un asistente educativo cultural shipibo-konibo. "
    "Responde la pregunta del usuario usando ÚNICAMENTE la información del "
    "CONTEXTO proporcionado, que proviene de un documento de cosmovisión "
    "shipibo. No agregues datos que no estén en el contexto ni inventes nada. "
    f"Si el contexto no contiene la respuesta, responde exactamente: {_SIN_DATOS}. "
    "Cuando sí haya respuesta, escríbela breve (3 oraciones máximo), en español "
    "neutro y con respeto por la cultura shipibo-konibo."
)

# Cliente Gemini perezoso (se crea una sola vez).
_gemini_client = None
_gemini_import_ok = None  # None = no probado, True/False = resultado


def _get_gemini_client():
    """Crea (una vez) el cliente de Gemini. Devuelve None si no está disponible."""
    global _gemini_client, _gemini_import_ok
    if _gemini_import_ok is False:
        return None
    if _gemini_client is not None:
        return _gemini_client
    try:
        from google import genai  # import perezoso: no rompe el Camino B si falta
        _gemini_import_ok = True
    except ImportError:
        logger.warning(
            "responder: google-genai no está instalado; Camino A no disponible. "
            "Instalá con: pip install google-genai"
        )
        _gemini_import_ok = False
        return None

    if not config.GEMINI_API_KEY:
        logger.warning(
            "responder: GEMINI_API_KEY vacía; Camino A no disponible. "
            "Configurala en el .env."
        )
        _gemini_import_ok = False
        return None

    try:
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
        logger.info("responder: cliente Gemini inicializado (modelo=%s)",
                    config.GEMINI_MODEL)
        return _gemini_client
    except Exception as e:
        logger.warning("responder: no se pudo crear el cliente Gemini: %s", e)
        _gemini_import_ok = False
        return None


def _es_error_rate_limit(exc: Exception) -> bool:
    """True si la excepción parece un 429 / RESOURCE_EXHAUSTED de Gemini."""
    txt = str(exc).lower()
    return "429" in txt or "resource_exhausted" in txt or "rate limit" in txt or "quota" in txt


def _llamar_gemini(contexto: str, query: str) -> Optional[str]:
    """
    Llama a Gemini con el contexto y la pregunta. Reintenta con backoff
    exponencial ante 429. Devuelve el texto de la respuesta, o None si falla
    definitivamente (para que el caller caiga al Camino B).
    """
    client = _get_gemini_client()
    if client is None:
        return None

    from google.genai import types

    prompt_usuario = f"CONTEXTO:\n{contexto}\n\nPREGUNTA: {query}"

    for intento in range(config.GEMINI_MAX_REINTENTOS + 1):
        try:
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt_usuario,
                config=types.GenerateContentConfig(
                    system_instruction=_PROMPT_SISTEMA,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=config.GEMINI_THINKING_LEVEL
                    ),
                    http_options=types.HttpOptions(timeout=int(config.GEMINI_TIMEOUT * 1000)),
                ),
            )
            texto = (resp.text or "").strip()
            return texto or None
        except Exception as e:
            if _es_error_rate_limit(e) and intento < config.GEMINI_MAX_REINTENTOS:
                espera = config.GEMINI_BACKOFF_BASE * (2 ** intento)
                logger.warning(
                    "responder: rate limit de Gemini (intento %d/%d); "
                    "reintento en %.1fs",
                    intento + 1, config.GEMINI_MAX_REINTENTOS, espera,
                )
                time.sleep(espera)
                continue
            # Otro error, o se agotaron los reintentos → caer a Camino B
            logger.warning("responder: Gemini falló (%s); cae a Camino B", e)
            return None
    return None


def responder_llm(query: str) -> Dict[str, Any]:
    """
    Camino A: retrieval + síntesis con Gemini. Restringido al contexto para no
    alucinar. Cae al Camino B (simple) si: no hay chunks relevantes, el LLM no
    está disponible, se agota el rate limit, o el modelo responde SIN_DATOS.

    El campo 'modo_usado' del dict devuelto refleja el origen REAL de la
    respuesta ('llm' si la sintetizó Gemini, 'simple' si cayó al fallback).
    """
    chunks = buscar_chunks(query, k=config.RETRIEVE_K)
    relevantes = [c for c in chunks if c["score"] <= config.SCORE_THRESHOLD]
    if not relevantes:
        # Sin contexto suficiente: no llamamos al LLM (evita alucinar e inventar)
        return {"respuesta": None, "chunks": chunks, "modo_usado": "simple"}

    contexto = "\n\n---\n\n".join(c["texto"] for c in relevantes)
    texto = _llamar_gemini(contexto, query)

    if texto is None:
        # Falla del LLM o rate limit → fallback a Camino B
        result = responder_simple(query)
        result["modo_usado"] = "simple"
        return result

    if _SIN_DATOS in texto:
        # Gemini juzgó que el contexto recuperado NO responde la pregunta.
        # Mostrar igual ese pasaje crudo (Camino B) sería contraproducente: es
        # justo el pasaje que el modelo descartó por irrelevante (así se veían
        # mal "Ronin" y "onanya"). En su lugar devolvemos respuesta=None para
        # que el bot dé su mensaje honesto ("no encontré información
        # específica"), en vez de un chunk que no viene al caso.
        logger.debug("responder: Gemini reportó SIN_DATOS para query=%r", query)
        return {"respuesta": None, "chunks": relevantes, "modo_usado": "llm"}

    # Enmarcar como respuesta cultural, coherente con el estilo del Camino B
    respuesta = f"📚 {texto}"
    return {"respuesta": respuesta, "chunks": relevantes, "modo_usado": "llm"}


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
            # responder_llm ya setea modo_usado real ('llm' o 'simple' si cayó).
            result.setdefault("modo_usado", "llm")
            return result
        except Exception as e:
            # Red de seguridad ante cualquier error no contemplado.
            logger.warning("responder: Camino A falló (%s); cae a 'simple'", e)
            result = responder_simple(query)
            result["modo_usado"] = "simple"
            return result

    result = responder_simple(query)
    result["modo_usado"] = "simple"
    return result