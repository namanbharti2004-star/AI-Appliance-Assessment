# AI-Powered Appliance Inspection & Insurance Claim Platform

A production-grade computer vision system that detects damage to consumer electronics and appliances, estimates repair costs, detects fraud, and generates professional insurance claim reports.

## Features

| Feature | Details |
|---------|---------|
| **Appliance Classification** | Detects phone, laptop, television, refrigerator with confidence scoring |
| **Damage Detection** | Identifies cracks, dents, display lines, rust, scratches via YOLO11 + heuristics |
| **Damage Localization** | NMS-filtered bounding boxes with appliance-aware location inference |
| **Severity Scoring** | Location-weighted (screen 2x vs body 0.8x), type-weighted with defect multiplier |
| **Repair Cost Estimation** | Brand-aware (Apple 2.5x, Samsung 1.8x), severity-based multipliers |
| **Fraud Detection** | 12-factor engine: ELA, PRNU noise, metadata, screenshot, AI-gen, copy-move, duplicate hash |
| **Explainable AI** | 5-section natural language explanation for every decision |
| **Claim Recommendation** | Weighted engine (severity 40%, fraud 30%, condition 20%, count 10%) |
| **Image Quality Gate** | Blur, exposure, resolution, compression, motion blur checks before inference |
| **Multi-Image Inspection** | Upload 2-6 angles, IoU-based damage merge across views |
| **Video Analysis** | Smart frame extraction, damage persistence tracking, inconsistency detection |
| **Professional PDF Reports** | Insurer-ready with decision badge, XAI reasoning, images, metadata |
| **IRDAI Compliant** | Regulatory disclaimer in API, PDF, and dashboard |
| **REST API** | 13 endpoints with API key auth, async inference, monitoring |
| **Streamlit Dashboard** | 6-tab UI with analytics, history, and system monitoring |

## Architecture

```
User Upload → Image Quality Gate → Appliance Detector (YOLO11s)
                                ↓
                    Damage Detector (YOLO + CLIP + heuristics)
                                ↓
         ┌──────────────────────┼──────────────────────┐
         ↓                      ↓                      ↓
   Severity Engine       Fraud Engine (12-factor)   Brand Detector
         ↓                      ↓                      ↓
   Repair Estimator      Claim Recommender       Rep. Cost Multiplier
         ↓                      ↓                      ↓
   Explainable AI ────→ Structured Report ←─── PDF Generator
                                ↓
                   Streamlit Dashboard / FastAPI / Gradio
```

## Tech Stack

- **Models**: YOLO11s, YOLO11s-seg, CLIP zero-shot, EasyOCR
- **Backend**: FastAPI (async), Uvicorn, ThreadPoolExecutor
- **Frontend**: Streamlit (6 tabs), Gradio, HTML/CSS
- **Fraud**: ELA, PRNU noise, perceptual hashing (SQLite), metadata analysis
- **PDF**: fpdf2 with IRDAI compliance
- **Infrastructure**: Docker, Docker Compose, Redis (Celery optional)
- **CI/CD**: GitHub Actions (test → lint → build)

## Setup

```bash
# Clone and enter
git clone https://github.com/namanbharti2004-star/AI-Appliance-Assessment.git
cd AI-Appliance-Assessment

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run API server
python scripts/run_api.py

# Run dashboard (separate terminal)
streamlit run dashboard/app.py

# Inspect a single image
python scripts/inference.py --image path/to/photo.jpg --save-vis
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/info` | API information |
| `POST` | `/api/v1/inspect/image` | Single image inspection |
| `POST` | `/api/v1/inspect/multi` | Multi-image (2-6) inspection |
| `POST` | `/api/v1/inspect/video` | Video inspection |
| `POST` | `/api/v1/inspect/video/async` | Async video (Redis/Celery) |
| `POST` | `/api/v1/quality` | Image quality check |
| `POST` | `/api/v1/fraud/advanced` | Advanced fraud analysis |
| `GET` | `/api/v1/claims` | List claims |
| `GET` | `/api/v1/claims/{id}` | Claim details |
| `GET` | `/api/v1/claims/{id}/pdf` | Download claim PDF |
| `GET` | `/api/v1/monitor/stats` | System monitoring |

## Fraud Detection Factors

| # | Factor | Weight |
|---|--------|--------|
| 1 | Error Level Analysis (ELA) | 15% |
| 2 | Metadata anomalies | 40% |
| 3 | Screenshot detection | 30% |
| 4 | AI-generated image detection | 30% |
| 5 | Copy-move detection | 20% |
| 6 | Color diversity analysis | 15% |
| 7 | Tampering edge detection | 10% |
| 8 | Resolution mismatch | 20% |
| 9 | Compression anomalies | 15% |
| 10 | Duplicate hash detection | 40% |
| 11 | PRNU noise consistency | 25% |
| 12 | GPS-location consistency | 15% |

## Claim Decision Formula

```
CLAIM_SCORE = severity_risk × 0.4 + fraud_score × 0.3
              + condition_risk × 0.1 + damage_count_risk × 0.1

< 25  → APPROVE (fast-track)
25-75 → MANUAL_REVIEW (adjuster needed)
≥ 75  → REJECT (high risk)
```

## Project Structure

```
AI-Appliance-Assessment/
├── api/                  # FastAPI server (13 endpoints)
├── configs/              # Thresholds, weights, rules
├── dashboard/            # Streamlit UI (6 tabs)
├── fraud_detection/      # ELA + metadata analyzers
├── missing_part_detector/
├── models/
│   ├── appliance_detector/   # YOLO11s for appliance classification
│   └── damage_detector/      # YOLO + CLIP + heuristic detectors
├── report_engine/        # Report generation + enrichment
├── risk_engine/          # Risk assessment pipeline
├── scripts/              # Inference, training, evaluation
├── services/
│   ├── fraud_service.py       # 12-factor fraud engine
│   ├── severity_service.py    # Location-weighted severity
│   ├── repair_service.py      # Brand-aware cost estimation
│   ├── pdf_service.py         # IRDAI-compliant PDFs
│   ├── explain_service.py     # XAI explanations
│   ├── claim_recommendation.py # Claim decision engine
│   ├── image_quality.py       # Pre-inference quality gate
│   ├── multi_image_service.py # Multi-angle merge
│   └── video_queue.py         # Redis/Celery async video
├── tests/                # 52 tests (19 core + 33 verification)
├── utils/                # Image I/O, normalization, helpers
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Limitations

- Phone crack model detects phone body instead of crack (dataset label issue)
- Fridge damage model mAP50 = 0.249 (trained on ~50 images)
- TV damage uses CLIP zero-shot fallback (no YOLO TV damage dataset)
- HEIC/HEIF support requires `pillow-heif` package
- Video processing is synchronous without Redis/Celery

## Future Improvements

- Active learning pipeline for uncertain samples
- CLIP-based zero-shot damage for all appliances
- Claim velocity monitoring (rate-limit per phone/email)
- Real-time PRNU fingerprint database for camera authentication
- Mobile SDK (iOS/Android) for field inspector use

## License

MIT — see LICENSE file for details.
