# rag-service

Servidor RAG independiente para Pishico Bot (tesis PUCP).
Expone búsqueda semántica sobre el PDF de cosmovisión shipibo
mediante una API HTTP simple.

## ¿Por qué un servicio separado?

El proyecto Rasa de Pishico depende de pydantic 1.x (requisito de Rasa 3.6).
LangChain moderno requiere pydantic 2.x. Estos no conviven en el mismo venv.

Solución: **dos servicios independientes**, cada uno con su venv y sus
versiones, comunicándose por HTTP local. Cero conflictos de versiones,
y el RAG queda reutilizable por otros proyectos a futuro.

```
chatbot-shi/  (Rasa, pydantic 1)            rag-service/  (este proyecto)
─────────────────────────────                ─────────────────────────────
actions.py                                   FastAPI (puerto 8001)
   ↓                                            ↓
rag_client.py  ──── HTTP POST /buscar ───►   src/app.py
                                                ↓
                                             retriever + responder
                                                ↓
                                             ChromaDB + sentence-transformers
                                             (LangChain moderno, pydantic 2)
```

## Stack técnico

- **FastAPI**: servidor HTTP async con documentación automática
- **LangChain**: pipeline RAG estándar (PyPDFLoader, RecursiveCharacterTextSplitter, Chroma)
- **ChromaDB**: base vectorial local (persistencia en disco)
- **sentence-transformers**: modelo de embeddings multilingüe (384 dim)
- **Pydantic 2**: validación automática del contrato HTTP

## Setup (primera vez)

### 1. Crear venv y activar

```bash
cd rag-service
python3.11 -m venv venv     # 3.10/3.11/3.12 todos OK
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configurar (opcional)

```bash
cp .env.example .env
# Editar .env si quieres cambiar puerto, threshold, etc.
```

### 4. Copiar el PDF de cosmovisión

```bash
cp /ruta/a/tu_pdf.pdf data/cosmovision.pdf
```

### 5. Indexar (UNA SOLA VEZ)

```bash
bash scripts/indexar.sh
# o bien:
python -m src.indexer
```

Output esperado:

```
[INFO] Leyendo PDF: cosmovision.pdf (3.7 MB)
[INFO] Páginas cargadas: 174
[INFO] Chunkeando con chunk_size=900, overlap=100...
[INFO] Chunks generados: 198
[INFO] Descartados 12 chunks < 80 caracteres
[INFO] Cargando modelo de embeddings: paraphrase-multilingual-MiniLM-L6-v2
[INFO]   (primera ejecución descarga ~90 MB del modelo, paciencia)
[INFO] Construyendo índice ChromaDB...
[INFO] Índice persistido

══ Pruebas de búsqueda ══
  '¿qué es Ronin?' (dist=0.512, pág. 87)
    → La boa Ronin es el espíritu protector...
  ...

✓ Índice listo. Arrancá el servidor con: bash scripts/arrancar.sh
```

### 6. Arrancar el servidor

```bash
bash scripts/arrancar.sh
# o bien:
uvicorn src.app:app --host 127.0.0.1 --port 8001
```

El servidor pre-carga el índice al arrancar. Vas a ver:

```
[INFO] Iniciando servidor RAG en 127.0.0.1:8001
[INFO] Modo por defecto: simple
[INFO] Threshold de relevancia: 1.20
[INFO] retriever: índice cargado (198 chunks) desde data/chroma_index
[INFO] ✓ Índice pre-cargado, servidor listo para queries
INFO:     Uvicorn running on http://127.0.0.1:8001
```

### 7. Probar con curl o el Swagger UI

```bash
# Health check
curl http://localhost:8001/health

# Estadísticas
curl http://localhost:8001/stats

# Búsqueda
curl -X POST http://localhost:8001/buscar \
     -H "Content-Type: application/json" \
     -d '{"query": "¿qué es Ronin?", "k": 1}'
```

O abrí en el navegador: **http://localhost:8001/docs**

FastAPI genera automáticamente la documentación interactiva con Swagger UI.

## Endpoints

### `POST /buscar`

Busca una respuesta cultural en el PDF.

**Request**:
```json
{
  "query": "¿qué es el Ronin?",
  "k": 1,
  "modo": "simple"
}
```

- `query` (string, requerido): pregunta del usuario
- `k` (int, default 1): cuántos chunks devolver, máximo 10
- `modo` (string, opcional): `"simple"` o `"llm"`. Si se omite, usa `RAG_MODO_DEFAULT` del `.env`

**Response (éxito)**:
```json
{
  "respuesta": "📚 Sobre eso, el documento de cosmovisión shipiba dice:\n\nRonin es la boa cósmica...",
  "chunks": [
    {"texto": "Ronin es la boa cósmica...", "pagina": 87, "score": 0.512}
  ],
  "modo_usado": "simple"
}
```

**Response (sin match relevante)**:
```json
{
  "respuesta": null,
  "chunks": [],
  "modo_usado": "simple"
}
```

### `GET /health`

Health check para monitoreo y para que el cliente Rasa verifique al arrancar.

```json
{ "status": "ok", "rag_disponible": true }
```

Si el índice no está cargado:
```json
{ "status": "degraded", "rag_disponible": false, "detalle": "..." }
```

### `GET /stats`

Información del índice:

```json
{
  "ok": true,
  "chunks": 198,
  "collection": "cosmovision_shipibo",
  "embed_model": "paraphrase-multilingual-MiniLM-L6-v2",
  "index_dir": "/path/to/data/chroma_index"
}
```

## Tests

```bash
pytest -v
```

Hay dos suites:
- `tests/test_retriever.py`: tests del retriever sin servidor
- `tests/test_api.py`: tests de los endpoints con TestClient

Tests que requieren índice indexado están marcados con `skipif`, así
los corres antes y después de indexar sin problema.

## Ajuste del threshold

`RAG_SCORE_THRESHOLD` en `.env` (default 1.2) controla cuán estricto es
el RAG. Distancia más baja = chunk más relevante. Si ves muchos
`respuesta: null` que deberían responder, subí a 1.4. Si ves respuestas
no relacionadas, bajá a 1.0.

## Re-indexación

Si actualizás el PDF, simplemente vuelves a correr:

```bash
bash scripts/indexar.sh
```

El script borra el índice previo y lo reconstruye. El modelo de embeddings
ya descargado se reusa desde caché.

## Camino A (LLM): activación

Ver `docs/rag_camino_a.md` (documento separado en el proyecto Rasa).
La activación del LLM en este servicio requiere:

1. Instalar Ollama y descargar phi3.5
2. Descomentar `langchain-ollama>=0.2.0` en `requirements.txt` y reinstalar
3. Descomentar el bloque de `responder_llm()` en `src/responder.py`
4. En `.env`: `RAG_MODO_DEFAULT=llm`
5. Reiniciar el servidor

El cliente Rasa no necesita cambios: el contrato HTTP es el mismo.

## Estructura

```
rag-service/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── cosmovision.pdf          # PDF fuente (tú lo copiás)
│   └── chroma_index/            # generado por scripts/indexar.sh
├── src/
│   ├── __init__.py
│   ├── config.py                # lee .env
│   ├── indexer.py               # PDF → chunks → embeddings → chromadb
│   ├── retriever.py             # carga índice, buscar_chunks()
│   ├── responder.py             # responder_simple (B) + responder_llm (A stub)
│   └── app.py                   # FastAPI con /buscar, /health, /stats
├── tests/
│   ├── test_retriever.py
│   └── test_api.py
└── scripts/
    ├── indexar.sh
    └── arrancar.sh
```
