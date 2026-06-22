"""
FastAPI service for the AI Appliance Inspection Platform.
git status --short | head -50
Endpoints:
- GET  /health                     Health check
- GET  /api/v1/info               API information
- POST /api/v1/inspect/image      Single image inspection
- POST /api/v1/inspect/multi      Multi-image inspection (up to 6)
- POST /api/v1/inspect/video      Video inspection
- POST /api/v1/quality            Image quality check
- POST /api/v1/fraud/advanced     Advanced fraud analysis
- POST /api/v1/severity           Severity computation
- GET  /api/v1/claims             List claims
- GET  /api/v1/claims/{id}        Get claim details
- GET  /api/v1/claims/{id}/pdf    Download claim PDF
- GET  /api/v1/monitor/stats      Monitoring statistics
- GET  /api/v1/sample-report      Sample report data
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import API_CONFIG, MVP_APPLIANCE_CLASSES, PROJECT_NAME, VERSION
from scripts.inference import InspectionPipeline
from services.claim_service import get_claim_by_id, get_claim_stats, get_claims, save_claim
from services.fraud_service import AdvancedFraudEngine
from services.image_quality import check_image_quality
from services.monitoring import InferenceEvent, monitor
from services.multi_image_service import MultiImageInspector
from services.video_queue import get_job_status, submit_video_job

API_KEYS: List[str] = []


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    if API_KEYS and (not x_api_key or x_api_key not in API_KEYS):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


app_state: Dict[str, Any] = {
    "pipeline": None, "advanced_fraud": None, "multi_inspector": None,
    "executor": None,
}

_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)


async def _run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, lambda: fn(*args, **kwargs))


async def _track_async(module: str, operation: str, fn, *args, **kwargs):
    start = time.perf_counter()
    error: Optional[str] = None
    success = True
    try:
        result = await _run_in_thread(fn, *args, **kwargs)
        return result
    except Exception as e:
        success = False
        error = str(e)
        raise
    finally:
        duration = (time.perf_counter() - start) * 1000
        monitor.log_inference(InferenceEvent(
            module=module, operation=operation,
            duration_ms=duration, success=success, error=error,
        ))


def _save_upload(uploaded_file: UploadFile) -> str:
    suffix = os.path.splitext(uploaded_file.filename or "")[1] or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


@asynccontextmanager
async def lifespan(_: FastAPI):
    app_state["pipeline"] = InspectionPipeline()
    app_state["advanced_fraud"] = AdvancedFraudEngine()
    app_state["multi_inspector"] = MultiImageInspector()
    app_state["executor"] = _thread_pool
    yield
    _thread_pool.shutdown(wait=True)


app = FastAPI(title=PROJECT_NAME, version=VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": PROJECT_NAME,
        "version": VERSION,
        "supported_appliances": MVP_APPLIANCE_CLASSES,
        "irdai_compliance": {
            "regulated": True,
            "regulator": "Insurance Regulatory and Development Authority of India (IRDAI)",
            "guidelines": "IRDAI (Insurance Advertisements and Disclosure) Regulations, 2021",
            "disclaimer": "This AI-generated report is an advisory tool and does not replace a licensed insurance surveyor's assessment. All claim decisions must be verified by a qualified adjuster in accordance with IRDAI guidelines.",
            "compliant": True,
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
    }


@app.get("/api/v1/info")
async def info():
    return {
        "appliances": MVP_APPLIANCE_CLASSES,
        "damage_classes": ["crack", "dent", "display_lines"],
        "fraud_modules": ["ela", "metadata", "advanced_7factor", "screenshot", "ai_gen", "copy_move"],
        "video_formats": API_CONFIG["allowed_video_formats"],
        "model_version": VERSION,
    }


@app.post("/api/v1/quality")
async def check_quality(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in API_CONFIG["allowed_image_formats"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")
    path = _save_upload(file)
    try:
        from utils import read_image
        image = read_image(path)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image")
        result = check_image_quality(image)
        return JSONResponse(result.to_dict())
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/api/v1/inspect/image")
async def inspect_image(
    file: UploadFile = File(...),
    appliance_override: Optional[str] = Form(None),
    _: None = Depends(verify_api_key),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in API_CONFIG["allowed_image_formats"]:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    path = _save_upload(file)
    try:
        from utils import read_image
        image = read_image(path)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image")

        quality = check_image_quality(image)
        if not quality.passed:
            return JSONResponse({
                "error": "Image quality check failed",
                "quality": quality.to_dict(),
                "guidance": quality.guidance,
            }, status_code=400)

        pipeline = app_state["pipeline"]
        result = await _track_async(
            "api", "inspect_image",
            pipeline.inspect_image,
            path, appliance_override=appliance_override,
            save_visualizations=True, output_dir="output",
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        result["quality_check"] = quality.to_dict()
        return JSONResponse(result)
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/api/v1/inspect/multi")
async def inspect_multi(
    files: List[UploadFile] = File(...),
    appliance_override: Optional[str] = Form(None),
    _: None = Depends(verify_api_key),
):
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="At least one image required")
    if len(files) > 6:
        raise HTTPException(status_code=400, detail="Maximum 6 images per request")

    paths = []
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in API_CONFIG["allowed_image_formats"]:
            for p in paths:
                if os.path.exists(p):
                    os.remove(p)
            raise HTTPException(status_code=400, detail=f"Unsupported format: {f.filename}")
        paths.append(_save_upload(f))

    try:
        report = await _track_async(
            "api", "inspect_multi",
            app_state["multi_inspector"].inspect, paths,
            appliance_override=appliance_override,
        )
        return JSONResponse(report.to_dict())
    finally:
        for p in paths:
            if os.path.exists(p):
                os.remove(p)


@app.post("/api/v1/inspect/video")
async def inspect_video(
    file: UploadFile = File(...),
    appliance_override: Optional[str] = Form(None),
    _: None = Depends(verify_api_key),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in API_CONFIG["allowed_video_formats"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")
    path = _save_upload(file)
    try:
        result = await _track_async(
            "api", "inspect_video",
            app_state["pipeline"].inspect_video,
            path, appliance_override=appliance_override, output_dir="output",
        )
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return JSONResponse(result)
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/api/v1/inspect/video/async")
async def inspect_video_async(
    file: UploadFile = File(...),
    appliance_override: Optional[str] = Form(None),
    _: None = Depends(verify_api_key),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in API_CONFIG["allowed_video_formats"]:
        raise HTTPException(status_code=400, detail="Unsupported video format")
    path = _save_upload(file)
    try:
        result = await _run_in_thread(
            submit_video_job, path,
            appliance_override=appliance_override,
            output_dir="output",
        )
        return JSONResponse(result)
    except Exception as exc:
        if os.path.exists(path):
            os.remove(path)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/video/job/{job_id}")
async def get_video_job(job_id: str):
    job = get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job)


@app.post("/api/v1/fraud/advanced")
async def advanced_fraud_analysis(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in API_CONFIG["allowed_image_formats"]:
        raise HTTPException(status_code=400, detail="Unsupported format")
    path = _save_upload(file)
    try:
        from utils import read_image
        image = read_image(path)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not read image")
        result = await _track_async(
            "api", "fraud_analysis",
            app_state["advanced_fraud"].analyze, image, path,
        )
        return JSONResponse(result.to_dict())
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/api/v1/severity")
async def compute_severity(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in API_CONFIG["allowed_image_formats"]:
        raise HTTPException(status_code=400, detail="Unsupported format")
    path = _save_upload(file)
    try:
        result = app_state["pipeline"].inspect_image(path, save_visualizations=False)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        report = result["report"]
        return JSONResponse({
            "damage_percentage": report.get("damage_percentage", 0),
            "severity": report.get("severity", "None"),
            "condition_score": report.get("condition_score", 100),
            "grade": report.get("grade", "A"),
            "damage_detected": report.get("damage_detected", False),
            "damage_count": len(report.get("damage_detections", [])),
        })
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.get("/api/v1/claims")
async def list_claims(limit: int = 50, offset: int = 0):
    claims = get_claims(limit=limit, offset=offset)
    stats = get_claim_stats()
    return JSONResponse({"claims": claims, "stats": stats})


@app.get("/api/v1/claims/{claim_id}")
async def get_claim(claim_id: str):
    claim = get_claim_by_id(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return JSONResponse(claim)


@app.get("/api/v1/claims/{claim_id}/pdf")
async def download_claim_pdf(claim_id: str):
    claim = get_claim_by_id(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    try:
        from services.pdf_service import generate_pdf_report
        report_data = json.loads(claim["full_report"]) if claim.get("full_report") else claim
        pdf_path = generate_pdf_report(report_data, output_dir="reports")
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"claim_{claim_id}.pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")


@app.get("/api/v1/monitor/stats")
async def monitor_stats():
    return JSONResponse({
        "session": monitor.get_session_stats(),
        "performance": monitor.get_performance_summary(),
        "recent_errors": monitor.get_recent_errors(limit=10),
    })


@app.get("/api/v1/sample-report")
async def sample_report():
    return JSONResponse({
        "appliance": "television",
        "appliance_confidence": 0.93,
        "damage_detected": True,
        "damage_type": "crack",
        "damage_confidence": 0.88,
        "damage_percentage": 12,
        "severity": "Moderate",
        "condition_score": 68,
        "grade": "C",
        "repair_cost_min": 3500,
        "repair_cost_max": 6300,
        "repair_cost_display": "\u20b93,500 - \u20b96,300",
        "fraud_score": 18,
        "fraud_risk_level": "Low",
        "claim_score": 42,
        "claim_risk": "Medium",
        "decision": "MANUAL_REVIEW",
    })


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    uvicorn.run("api:app", host=host, port=port, reload=reload)
