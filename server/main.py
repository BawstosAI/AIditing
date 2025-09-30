from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import re
from fastapi.responses import FileResponse, JSONResponse

from .config import FRONTEND_ORIGINS, FRONTEND_ORIGIN_REGEX
from .jobs import Job, job_store
from .pipeline import run_real_pipeline
from .storage import ensure_job_dirs, get_result_path, get_upload_paths, save_stream_to_path

app = FastAPI(title="AIditing API", version="0.1.0")

# CORS for frontend dev
compiled_origin_regex = None
if FRONTEND_ORIGIN_REGEX:
    try:
        compiled_origin_regex = re.compile(FRONTEND_ORIGIN_REGEX)
    except re.error:
        compiled_origin_regex = None


def _allow_origin_fn(origin: str) -> bool:  # FastAPI calls this when using allow_origin_regex
    if not origin:
        return False
    if origin in FRONTEND_ORIGINS:
        return True
    if compiled_origin_regex and compiled_origin_regex.match(origin):
        return True
    return False


app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_origin_regex=FRONTEND_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> JSONResponse:
    # Fast and simple healthcheck endpoint
    return JSONResponse({"status": "ok"})


@app.post("/process")
async def process_endpoint(
    background_tasks: BackgroundTasks,
    video: Annotated[UploadFile, File(description="Video file (mp4)")],
    audio: Annotated[UploadFile, File(description="Clean audio (wav)")],
) -> JSONResponse:
    job_id = str(uuid.uuid4())
    job_store.create(Job(job_id=job_id, status="queued", progress=0))

    # Prepare directories and file paths
    dirs = ensure_job_dirs(job_id)
    upload_paths = get_upload_paths(job_id)

    # Save uploaded files to disk
    with video.file as vf:
        save_stream_to_path(vf, upload_paths["video"])
    with audio.file as af:
        save_stream_to_path(af, upload_paths["audio"])

    # Launch background mock pipeline
    def _task() -> None:
        try:
            run_real_pipeline(job_id, upload_paths["video"], upload_paths["audio"])
        except Exception as exc:  # noqa: BLE001
            job_store.update(job_id, status="failed", progress=100, message=str(exc))

    background_tasks.add_task(_task)

    return JSONResponse({"job_id": job_id})


@app.get("/status/{job_id}")
async def status_endpoint(job_id: str) -> JSONResponse:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JSONResponse({
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
    })


@app.get("/result/{job_id}")
async def result_endpoint(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != "completed" or not job.result_path:
        raise HTTPException(status_code=409, detail="result not ready")

    path = Path(job.result_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="result missing")

    return FileResponse(path, media_type="video/mp4", filename="out.mp4")

