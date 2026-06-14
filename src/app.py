"""
app.py — Servidor FastAPI del RAG.

Endpoints:
  POST /buscar  — consulta RAG (request: query+k+modo; response: respuesta+chunks+modo_usado)
  GET  /health  — health check (devuelve 200 si el índice está cargado)
  GET  /stats   — info del índice (chunks, modelo, ubicación)

Arranque:
    uvicorn src.app:app --host 127.0.0.1 --port 8001
    # o bien:
    bash scripts/arrancar.sh

Documentación interactiva: http://localhost:8001/docs (Swagger UI generado por FastAPI)
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config, retriever, responder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Schemas Pydantic (contrato HTTP) ─────────────────────────────────────────

class BuscarRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500,
                       description="Pregunta del usuario en lenguaje natural")
    k: int = Field(1, ge=1, le=10,
                   description="Cuántos chunks devolver (default 1)")
    modo: Optional[Literal["simple", "llm"]] = Field(
        None,
        description="Modo de respuesta. None = usa RAG_MODO_DEFAULT del .env"
    )


class Chunk(BaseModel):
    texto: str
    pagina: object  # int | str ("?")
    score: float


class BuscarResponse(BaseModel):
    respuesta: Optional[str] = Field(
        None,
        description="Respuesta lista para mostrar al usuario. None si no hay match relevante."
    )
    chunks: List[Chunk] = Field(
        default_factory=list,
        description="Chunks recuperados con metadata (para debug y UI avanzada)"
    )
    modo_usado: str = Field(..., description="Modo efectivo usado ('simple' o 'llm')")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    rag_disponible: bool
    detalle: Optional[str] = None


class StatsResponse(BaseModel):
    ok: bool
    chunks: int
    collection: Optional[str] = None
    embed_model: Optional[str] = None
    index_dir: Optional[str] = None
    error: Optional[str] = None


# ── Lifespan: pre-carga del índice al arrancar ───────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-carga el índice y el modelo al arrancar el servidor (warm-up)."""
    logger.info("Iniciando servidor RAG en %s:%d", config.HOST, config.PORT)
    logger.info("Modo por defecto: %s", config.MODO_DEFAULT)
    logger.info("Threshold de relevancia: %.2f", config.SCORE_THRESHOLD)
    ok = retriever.disponible()
    if ok:
        logger.info("✓ Índice pre-cargado, servidor listo para queries")
    else:
        logger.warning("⚠ Servidor arrancando SIN índice. /buscar devolverá respuesta=null")
        logger.warning("  Para indexar: python -m src.indexer")
    yield
    logger.info("Servidor RAG detenido")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Pishico RAG Service",
    description=(
        "Servidor RAG para Pishico Bot (tesis PUCP). "
        "Provee búsqueda semántica sobre el PDF de cosmovisión shipibo. "
        "Camino B (retrieval + framing) activo; Camino A (LLM) preparado para migración."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/buscar", response_model=BuscarResponse,
          summary="Buscar respuesta cultural en el PDF")
async def buscar(req: BuscarRequest) -> BuscarResponse:
    """
    Procesa una pregunta del usuario y devuelve una respuesta basada en el PDF
    de cosmovisión shipibo, o None si no hay match suficientemente relevante.
    """
    try:
        result = responder.responder(req.query, modo=req.modo)
        # Limitar chunks al k pedido si vino con menos
        chunks = result.get("chunks", [])[:req.k]
        return BuscarResponse(
            respuesta=result.get("respuesta"),
            chunks=chunks,
            modo_usado=result.get("modo_usado", "simple"),
        )
    except Exception as e:
        logger.exception("Error procesando /buscar: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse,
         summary="Health check del servicio")
async def health() -> HealthResponse:
    """Devuelve 200 con status=ok si el RAG está completamente disponible,
    status=degraded si el índice no está cargado (el servidor responde pero
    las búsquedas devolverán respuesta=null)."""
    estado = retriever.estado()
    if estado["ok"]:
        return HealthResponse(status="ok", rag_disponible=True)
    return HealthResponse(
        status="degraded",
        rag_disponible=False,
        detalle=estado.get("error", "índice no cargado")
    )


@app.get("/stats", response_model=StatsResponse,
         summary="Estadísticas del índice")
async def stats() -> StatsResponse:
    """Información del índice: cantidad de chunks, modelo de embeddings, etc."""
    return StatsResponse(**retriever.estado())


# ── Ejecutar directamente: python -m src.app ─────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app:app", host=config.HOST, port=config.PORT, reload=False)
