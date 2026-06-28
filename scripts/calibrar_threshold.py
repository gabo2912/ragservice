#!/usr/bin/env python3
"""
calibrar_threshold.py — Calibra RAG_SCORE_THRESHOLD con datos reales.

Problema que resuelve: el umbral decide cuándo el RAG responde y cuándo calla.
Si está muy alto, el bot responde con pasajes irrelevantes (parece que "inventa").
Si está muy bajo, calla respuestas que sí estaban en el PDF. El valor 1.2 venía
puesto a ojo; este script lo calibra midiendo los scores reales que produce tu
índice sobre dos grupos de preguntas:

  • RELEVANTES  → preguntas que SÍ tienen respuesta en el PDF de cosmovisión.
                  Sus scores deberían quedar POR DEBAJO del umbral.
  • IRRELEVANTES → preguntas ajenas al PDF (cocina, deportes, etc.).
                  Sus scores deberían quedar POR ENCIMA del umbral.

El umbral ideal separa ambos grupos. El script sugiere el punto medio entre el
peor relevante y el mejor irrelevante, y reporta si hay solape (zona gris).

Uso (desde la raíz de rag-service, con el venv activo y el índice ya construido):
    python -m scripts.calibrar_threshold
    # o:  python scripts/calibrar_threshold.py
"""

import sys
from pathlib import Path

# Permitir ejecución como script suelto o como módulo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import retriever, config  # noqa: E402

# ── Preguntas RELEVANTES (deben tener respuesta en el PDF) ───────────────────
# Construidas con términos reales del documento de cosmovisión shipibo.
RELEVANTES = [
    "¿qué es el kené?",
    "¿qué significa Jene?",
    "¿qué es Nete en la cosmovisión shipiba?",
    "¿qué representa Niwe, el viento?",
    "¿quién es Ronin, la anaconda?",
    "¿qué papel tiene la ayahuasca para los shipibo?",
    "¿qué es Jain o el espacio de los espíritus?",
    "háblame del río Ucayali",
    "¿cómo es la cosmovisión shipibo-konibo?",
    "¿qué importancia tiene el curandero o meraya?",
]


# ── Preguntas IRRELEVANTES (NO deben tener respuesta en el PDF) ──────────────
IRRELEVANTES = [
    "¿cómo hago una pizza margarita?",
    "¿cuál es la capital de Francia?",
    "¿quién ganó el mundial de fútbol 2022?",
    "¿cómo configuro una impresora HP?",
    "receta de lomo saltado paso a paso",
    "¿cuánto cuesta un iPhone nuevo?",
    "explícame la teoría de la relatividad",
    "¿qué tiempo hará mañana en Lima?",
]


def _mejor_score(query: str):
    """Devuelve el menor score (mejor match) para la query, o None si sin índice."""
    chunks = retriever.buscar_chunks(query, k=config.RETRIEVE_K)
    if not chunks:
        return None
    return min(c["score"] for c in chunks)


def main():
    if not retriever.disponible():
        print("✗ Índice no disponible. Construilo primero con: bash scripts/indexar.sh")
        sys.exit(1)

    print("=" * 64)
    print("  Calibración de RAG_SCORE_THRESHOLD")
    print(f"  Umbral actual: {config.SCORE_THRESHOLD}")
    print(f"  (menor score = más relevante; distancia coseno 0–2)")
    print("=" * 64)

    print("\n── RELEVANTES (scores deberían ser BAJOS) ──")
    rel_scores = []
    for q in RELEVANTES:
        s = _mejor_score(q)
        if s is not None:
            rel_scores.append(s)
            marca = "✓" if s <= config.SCORE_THRESHOLD else "✗ (el umbral la silencia)"
            print(f"  {s:.3f}  {marca}  {q}")

    print("\n── IRRELEVANTES (scores deberían ser ALTOS) ──")
    irr_scores = []
    for q in IRRELEVANTES:
        s = _mejor_score(q)
        if s is not None:
            irr_scores.append(s)
            marca = "✓" if s > config.SCORE_THRESHOLD else "✗ (el umbral la deja pasar)"
            print(f"  {s:.3f}  {marca}  {q}")

    if not rel_scores or not irr_scores:
        print("\nNo hay suficientes datos para sugerir umbral.")
        return

    peor_rel = max(rel_scores)   # la relevante más difícil
    mejor_irr = min(irr_scores)  # la irrelevante más "tentadora"

    print("\n" + "=" * 64)
    print("  Análisis")
    print("=" * 64)
    print(f"  Relevante más difícil (score máx):   {peor_rel:.3f}")
    print(f"  Irrelevante más tentadora (score mín): {mejor_irr:.3f}")

    if peor_rel < mejor_irr:
        sugerido = round((peor_rel + mejor_irr) / 2, 2)
        print(f"\n  ✓ Separación limpia. Umbral sugerido: {sugerido}")
        print(f"    (punto medio entre ambos grupos)")
        print(f"\n  Poné en tu .env:  RAG_SCORE_THRESHOLD={sugerido}")
    else:
        print(f"\n  ⚠ ZONA GRIS: hay solape entre relevantes e irrelevantes.")
        print(f"    Algunas preguntas válidas puntúan peor que algunas ajenas.")
        print(f"    Opciones: (a) priorizar no-alucinar → usar umbral ~{peor_rel - 0.05:.2f}")
        print(f"    (silencia algunas válidas pero no inventa);")
        print(f"    (b) priorizar cobertura → usar ~{mejor_irr + 0.05:.2f}")
        print(f"    (responde más pero puede colar alguna irrelevante);")
        print(f"    (c) mejorar el chunking/modelo de embeddings para separar mejor.")

    # Distribución resumida
    import statistics as st
    print(f"\n  Relevantes:   min={min(rel_scores):.3f}  "
          f"media={st.mean(rel_scores):.3f}  max={max(rel_scores):.3f}")
    print(f"  Irrelevantes: min={min(irr_scores):.3f}  "
          f"media={st.mean(irr_scores):.3f}  max={max(irr_scores):.3f}")


if __name__ == "__main__":
    main()
