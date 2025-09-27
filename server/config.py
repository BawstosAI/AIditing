from __future__ import annotations

import os
from pathlib import Path

# Base directories
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
RESULTS_DIR: Path = DATA_DIR / "results"
JOBS_DIR: Path = DATA_DIR / "jobs"

# Frontend origins for CORS (comma-separated)
FRONTEND_ORIGINS: list[str] = [
    o.strip()
    for o in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,https://bawstosai.github.io",
    ).split(",")
    if o.strip()
]

# Ensure directories exist
for _dir in (DATA_DIR, UPLOADS_DIR, RESULTS_DIR, JOBS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

