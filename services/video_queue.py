from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from loguru import logger

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CELERY_AVAILABLE = False
celery_app = None

try:
    from celery import Celery
    celery_app = Celery(
        "video_tasks",
        broker=REDIS_URL,
        backend=REDIS_URL,
    )
    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        result_expires=timedelta(hours=24),
    )
    CELERY_AVAILABLE = True
    logger.info("Celery app initialized with Redis: {}", REDIS_URL)
except ImportError:
    logger.warning("Celery/Redis not installed. Video queue will use synchronous processing.")
except Exception as exc:
    logger.error("Celery init failed: {}", exc)


_JOB_STORE: Dict[str, Dict[str, Any]] = {}
_job_counter = 0


def _next_job_id() -> str:
    global _job_counter
    _job_counter += 1
    return f"video_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_job_counter}"


if CELERY_AVAILABLE and celery_app is not None:

    @celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
    def process_video_task(self, video_path: str, appliance_override: Optional[str] = None,
                            output_dir: str = "output") -> Dict[str, Any]:
        from scripts.inference import InspectionPipeline
        pipeline = InspectionPipeline()
        try:
            result = pipeline.inspect_video(
                video_path=video_path,
                appliance_override=appliance_override,
                output_dir=output_dir,
            )
            return result
        except Exception as exc:
            logger.error("Video processing failed: {}", exc)
            raise self.retry(exc=exc)
        finally:
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass


def submit_video_job(
    video_path: str,
    appliance_override: Optional[str] = None,
    output_dir: str = "output",
) -> Dict[str, Any]:
    job_id = _next_job_id()
    if CELERY_AVAILABLE and celery_app is not None:
        task = process_video_task.delay(video_path, appliance_override, output_dir)
        _JOB_STORE[job_id] = {
            "job_id": job_id,
            "task_id": task.id,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None,
        }
        logger.info("Video job {} submitted (task {})", job_id, task.id)
    else:
        from scripts.inference import InspectionPipeline
        pipeline = InspectionPipeline()
        try:
            result = pipeline.inspect_video(
                video_path=video_path,
                appliance_override=appliance_override,
                output_dir=output_dir,
            )
            _JOB_STORE[job_id] = {
                "job_id": job_id,
                "task_id": None,
                "status": "completed",
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "result": result,
                "error": None,
            }
        except Exception as exc:
            _JOB_STORE[job_id] = {
                "job_id": job_id,
                "task_id": None,
                "status": "failed",
                "created_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "result": None,
                "error": str(exc),
            }
        finally:
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass
    return {"job_id": job_id, "status": _JOB_STORE[job_id]["status"]}


def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    job = _JOB_STORE.get(job_id)
    if job is None:
        return None
    if CELERY_AVAILABLE and celery_app is not None and job.get("task_id"):
        try:
            from celery.result import AsyncResult
            async_result = AsyncResult(job["task_id"], app=celery_app)
            if async_result.ready():
                if async_result.successful():
                    job["status"] = "completed"
                    job["result"] = async_result.result
                    job["completed_at"] = datetime.now().isoformat()
                else:
                    job["status"] = "failed"
                    job["error"] = str(async_result.result)
                    job["completed_at"] = datetime.now().isoformat()
            else:
                state = async_result.state
                job["status"] = {
                    "PENDING": "queued",
                    "RECEIVED": "queued",
                    "STARTED": "processing",
                    "RETRY": "retrying",
                }.get(state, state.lower())
        except Exception as exc:
            logger.error("Failed to check task status: {}", exc)
    return dict(job)
