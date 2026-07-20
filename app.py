from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from src.pipeline import VideoRAGPipeline

logger = logging.getLogger("uvicorn.error")

pipeline: VideoRAGPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Initializing Video RAG Pipeline...")
    import os
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEYS") and not os.getenv("GOOGLE_API_KEY"):
        logger.warning("WARNING: GOOGLE_API_KEYS is not set in environment or .env. Gemini LLM stages (Decouple and Answer Generation) will be disabled, but search queries will fall back to exact matching.")
    try:
        pipeline = VideoRAGPipeline("configs/config.yaml")
        logger.info("Video RAG Pipeline initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize Video RAG Pipeline: {e}", exc_info=True)
        raise e
    yield
    if pipeline:
        logger.info("Closing Video RAG Pipeline...")
        pipeline.close()
        logger.info("Video RAG Pipeline closed.")

app = FastAPI(title="Video RAG Search Interface", lifespan=lifespan)

# Enable CORS for local testing if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

METADATA_FILE = Path("data/uploaded_metadata.json")

def load_uploaded_metadata() -> list:
    if METADATA_FILE.exists():
        try:
            return json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_uploaded_metadata(metadata: list):
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

class QueryRequest(BaseModel):
    query: str
    collection_prefix: str | None = None

@app.post("/api/ask")
async def ask_query(request: QueryRequest):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        logger.info(f"Processing query: {request.query} on collection: {request.collection_prefix}")
        try:
            decomposition, candidates, answer, model_name, key_index = pipeline.answer(
                request.query, collection_prefix=request.collection_prefix
            )
        except Exception as e:
            logger.warning(f"Generation or decoupling failed ({e}). Falling back to vector-search only.")
            decomposition, candidates = pipeline.search(
                request.query, collection_prefix=request.collection_prefix
            )
            answer = "Внимание: Генерация текстового ответа недоступна (требуется настроить GOOGLE_API_KEY в файле .env). Ниже представлены результаты поиска по локальной векторной базе Qdrant."
            model_name = "fallback-search-only"
            key_index = None
        
        # Serialize candidates for JSON response
        serialized_candidates = []
        for c in candidates:
            serialized_hits = []
            for h in c.hits:
                serialized_hits.append({
                    "modality": h.modality,
                    "start": h.start,
                    "end": h.end,
                    "score": h.score,
                    "text": h.text
                })
            
            # Format filename to be shorter/pretty
            video_filename = Path(c.video_file).name
            
            serialized_candidates.append({
                "video_file": video_filename,
                "video_full_path": str(c.video_file),
                "start": c.start,
                "end": c.end,
                "score": c.score,
                "hits": serialized_hits
            })
            
        return {
            "query": request.query,
            "answer": answer,
            "model_name": model_name or "Unknown",
            "key_index": key_index,
            "decomposition": {
                "asr_query": decomposition.asr_query or "",
                "det_queries": decomposition.det_queries or [],
                "det_mode": decomposition.det_mode or "relation"
            },
            "candidates": serialized_candidates
        }
    except Exception as e:
        logger.error(f"Error answering query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    metadata = load_uploaded_metadata()
    return {
        "main_index_available": True,
        "uploaded_videos": metadata
    }

active_sessions = {}

@app.post("/api/upload")
async def upload_videos(files: list[UploadFile] = File(...)):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized")
        
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
        
    try:
        session_id = str(uuid.uuid4())
        videos_dir = Path("data/videos")
        videos_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        new_metadata = []
        
        for file in files:
            upload_id = str(uuid.uuid4())[:8]
            safe_name = "".join(c if c.isalnum() or c in (".", "_", "-") else "_" for c in file.filename)
            filename = f"upload_{upload_id}_{safe_name}"
            dest_path = videos_dir / filename
            
            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                
            saved_paths.append(dest_path)
            size_bytes = dest_path.stat().st_size
            
            new_metadata.append({
                "filename": filename,
                "original_name": file.filename,
                "size_bytes": size_bytes,
                "timestamp": time.time()
            })
            
        active_sessions[session_id] = {
            "paths": saved_paths,
            "metadata": new_metadata
        }
        
        logger.info(f"Saved {len(saved_paths)} uploaded video files for session {session_id}. Ready for streaming.")
        
        return {
            "status": "uploaded",
            "session_id": session_id,
            "videos": new_metadata
        }
        
    except Exception as e:
        logger.error(f"Error saving video files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/upload/events/{session_id}")
async def upload_events(session_id: str):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized")
        
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    async def event_generator():
        try:
            saved_paths = session["paths"]
            metadata = session["metadata"]
            
            # Start the extraction and indexing stream generator
            generator = pipeline.index_uploaded_videos_generator(
                saved_paths, prefix="uploaded_videos", recreate=True
            )
            
            for event in generator:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.05)
                
            # Persist custom metadata once indexing succeeds
            save_uploaded_metadata(metadata)
            
            yield f"data: {json.dumps({'step': 'done', 'status': 'success'}, ensure_ascii=False)}\n\n"
            
        except Exception as e:
            logger.error(f"Error inside index streaming generator: {e}", exc_info=True)
            yield f"data: {json.dumps({'step': 'error', 'status': 'failed', 'message': str(e)}, ensure_ascii=False)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/clear-uploads")
async def clear_uploads():
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized")
        
    try:
        metadata = load_uploaded_metadata()
        videos_dir = Path("data/videos")
        artifacts_dir = Path("data/artifacts")
        
        for item in metadata:
            filename = item["filename"]
            video_file = videos_dir / filename
            if video_file.exists():
                video_file.unlink()
                
            artifact_file = artifacts_dir / f"{video_file.stem}.json"
            if artifact_file.exists():
                artifact_file.unlink()
                
        store = pipeline._get_store()
        for modality in pipeline.enabled_modalities():
            store.recreate_collection(modality, prefix="uploaded_videos")
            
        if METADATA_FILE.exists():
            METADATA_FILE.unlink()
            
        logger.info("Custom uploaded videos and index cleared successfully!")
        return {"status": "success", "message": "Custom index and uploads cleared."}
        
    except Exception as e:
        logger.error(f"Error clearing uploads: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files for frontend
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def read_index():
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found in static folder")
    
    # Force browser to always fetch the fresh version, bypassing local cache
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"), headers=headers)

app.mount("/videos", StaticFiles(directory="data/videos"), name="videos")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

