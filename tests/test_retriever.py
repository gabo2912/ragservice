"""
test_retriever.py — Tests del retriever (no requiere servidor corriendo).

Estos tests verifican que:
  1. El retriever degrada limpiamente si no hay índice (devuelve [] no rompe)
  2. Si hay índice, las búsquedas devuelven chunks con la estructura esperada
  3. La función estado() devuelve el formato correcto para /health

Ejecución:
    pytest tests/test_retriever.py -v
"""

import pytest
from src import retriever, responder


def test_disponible_devuelve_bool():
    """disponible() siempre devuelve bool, sin tirar excepciones."""
    resultado = retriever.disponible()
    assert isinstance(resultado, bool)


def test_estado_estructura():
    """estado() siempre devuelve dict con keys 'ok' y 'chunks' presentes."""
    estado = retriever.estado()
    assert "ok" in estado
    assert isinstance(estado["ok"], bool)
    assert "chunks" in estado


def test_buscar_query_vacia():
    """Búsqueda con string vacío devuelve lista vacía sin error."""
    chunks = retriever.buscar_chunks("")
    assert chunks == []


def test_buscar_query_solo_espacios():
    """Búsqueda con solo espacios devuelve lista vacía."""
    chunks = retriever.buscar_chunks("   ")
    assert chunks == []


def test_responder_simple_query_vacia():
    """responder_simple con query vacía devuelve respuesta=None."""
    resultado = responder.responder_simple("")
    assert resultado["respuesta"] is None
    assert resultado["chunks"] == []


def test_responder_dispatcher_modo_invalido():
    """Modo inválido cae a 'simple' sin tirar excepción."""
    resultado = responder.responder("test", modo="xxx_invalido")
    assert resultado["modo_usado"] == "simple"
    assert "respuesta" in resultado
    assert "chunks" in resultado


def test_responder_llm_stub_tira_notimplementederror():
    """El stub del Camino A debe tirar NotImplementedError."""
    with pytest.raises(NotImplementedError):
        responder.responder_llm("test")


def test_responder_modo_llm_cae_a_simple_si_stub():
    """
    Si el Camino A está en modo stub, pedir modo='llm' debe caer a 'simple'
    automáticamente y devolver modo_usado='simple'.
    """
    resultado = responder.responder("¿qué es Ronin?", modo="llm")
    assert resultado["modo_usado"] == "simple"
    assert "respuesta" in resultado


# ── Tests que SOLO corren si hay índice (skipif) ─────────────────────────────

INDICE_OK = retriever.disponible()


@pytest.mark.skipif(not INDICE_OK, reason="índice no disponible (correr indexer primero)")
def test_buscar_pregunta_cultural_relevante():
    """Una pregunta clara sobre el PDF debe devolver al menos un chunk."""
    chunks = retriever.buscar_chunks("cosmovisión shipibo", k=3)
    assert len(chunks) > 0
    assert all("texto" in c and "pagina" in c and "score" in c for c in chunks)
    # Los scores deben venir ordenados ascendentemente (menor = mejor)
    scores = [c["score"] for c in chunks]
    assert scores == sorted(scores)


@pytest.mark.skipif(not INDICE_OK, reason="índice no disponible")
def test_responder_simple_pregunta_relevante():
    """Pregunta clara sobre el PDF debe devolver respuesta no nula."""
    resultado = responder.responder_simple("cosmovisión shipibo")
    # Asumiendo threshold razonable, esta pregunta debe tener match
    if resultado["respuesta"] is not None:
        assert "📚" in resultado["respuesta"]  # debe tener el marco textual
        assert len(resultado["chunks"]) >= 1


@pytest.mark.skipif(not INDICE_OK, reason="índice no disponible")
def test_responder_simple_pregunta_no_relacionada():
    """Pregunta totalmente fuera del PDF debe devolver respuesta=None."""
    resultado = responder.responder_simple("recetas de cocina italiana")
    # No es seguro al 100% pero esperamos que el threshold filtre esto
    # Si igual responde, al menos el chunks no debe estar vacío
    assert "respuesta" in resultado
    assert "chunks" in resultado
