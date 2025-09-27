from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Dict, Optional


@dataclass
class Job:
    job_id: str
    status: str = "queued"  # queued | running | completed | failed
    progress: int = 0        # 0-100
    message: str = ""
    result_path: Optional[str] = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = RLock()

    def create(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)


job_store = JobStore()

