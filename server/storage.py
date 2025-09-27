from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from .config import UPLOADS_DIR, RESULTS_DIR, JOBS_DIR


def ensure_job_dirs(job_id: str) -> dict[str, Path]:
    job_root = JOBS_DIR / job_id
    uploads_dir = job_root / "uploads"
    results_dir = job_root / "results"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "job_root": job_root,
        "uploads_dir": uploads_dir,
        "results_dir": results_dir,
    }


def save_stream_to_path(stream: BinaryIO, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def get_result_path(job_id: str) -> Path:
    return (JOBS_DIR / job_id / "results" / "out.mp4").resolve()


def get_upload_paths(job_id: str) -> dict[str, Path]:
    base = JOBS_DIR / job_id / "uploads"
    return {
        "video": (base / "video.mp4").resolve(),
        "audio": (base / "clean.wav").resolve(),
    }

