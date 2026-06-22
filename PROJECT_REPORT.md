# AI-Powered Appliance Inspection & Insurance Claim Platform

## Project Overview

**Version:** 3.0.0  
**Language:** Python 3.10+  
**Total Codebase:** ~7,800 lines across 38 Python files  
**Model Weights:** ~250 MB total (9 trained models + 1 fallback)  
**Database:** SQLite (claim history, fraud hashes, monitoring)  
**License:** Proprietary (Insurance Industry Use)

A production-grade **Computer Vision + AI platform** purpose-built for insurance companies (Zopper, ACKO, Digit, ICICI Lombard, Bajaj Allianz, PolicyBazaar) to automate appliance damage assessment and claim processing. The system takes photos/videos of damaged appliances, classifies the appliance type, detects and localizes damage, assesses severity, estimates repair costs, detects fraud, and generates insurer-ready claim recommendations — all with transparent explainable AI reasoning.

---

## What The System Does (End-to-End)

```
User Uploads Photo/Video
        │
        ▼
┌─────────────────────────────┐
│ 1. Image Quality Validation │  ← Blur, exposure, resolution, compression check
└─────────────┬───────────────┘
              │ (fail → retake guidance)
              ▼
┌─────────────────────────────┐
│ 2. Appliance Classification │  ← YOLO11s → "Refrigerator (97%)"
└─────────────┬───────────────┘
              │ (< 35% confidence → "Unknown" → manual review)
              ▼
┌─────────────────────────────┐
│ 3. Damage Detection         │  ← YOLO + CV heuristics + NMS
│    (per damage region)      │      shadow filter + area validation
└─────────────┬───────────────┘
              │ (no valid damage → "No Damage Detected")
              ▼
┌─────────────────────────────┐
│ 4. Damage Localization      │  ← Location labels: "Upper Door",
│    + Segmentation           │      "Screen", "Keyboard", etc.
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ 5. Fraud Detection          │  ← 10-factor engine (ELA, screenshot,
│    (0-100 score)            │      AI-gen, copy-move, duplicate, etc.)
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ 6. Severity Assessment      │  ← Type × Area × Defect count
│    + Repair Cost Estimate   │      → 4 severity bands
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ 7. Claim Recommendation     │  ← APPROVE / MANUAL_REVIEW / REJECT
│    + Explainable AI         │      + justification text
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ 8. Professional Dashboard   │  ← Side-by-side images, confidence bars,
│    + PDF Report             │      XAI panel, insurer-ready PDF
└─────────────────────────────┘
```

---

## Technology Stack

### Core AI/ML
| Technology | Version/Purpose |
|---|---|
| **Ultralytics YOLO11s** | Primary model architecture (appliance detection, damage segmentation) — 640×640 input |
| **Ultralytics YOLOv8n** | Fallback appliance detector (~6.2 MB, runs on low-resource devices) |
| **PyTorch 2.0+** | Deep learning framework — MPS support on Apple Silicon, CUDA on NVIDIA GPUs |
| **OpenCV 4.8+** | Image processing, heuristics, annotation, frame extraction |
| **scikit-learn 1.3+** | ML utilities, clustering for damage analysis |
| **scikit-image 0.21+** | Advanced image processing (DCT, ELA, segmentation helpers) |
| **NumPy 1.24+ / Pandas 2.0+** | Numerical computation, data manipulation |
| **Pillow 10.0+** | EXIF metadata reading, image format support |

### Web & API
| Technology | Purpose |
|---|---|
| **FastAPI** | REST API backend (13 endpoints) |
| **Uvicorn** | ASGI server |
| **Streamlit** | Professional dashboard UI (6-tab navigation) |
| **Python-Multipart** | File upload handling |
| **Pydantic** | Request/response validation |
| **CORS Middleware** | Cross-origin support |

### Data & Storage
| Technology | Purpose |
|---|---|
| **SQLite** | Claim history (`claim_history.db`), fraud hash database (`fraud_hashes.db`), monitoring database (`monitor.db`) |
| **fpdf2** | PDF report generation with images and structured layout |
| **PyYAML** | Configuration files and repair cost rules |
| **JSON** | Report serialization and API responses |
| **Loguru** | Structured logging with file rotation |

### Infrastructure
| Technology | Purpose |
|---|---|
| **Docker** | Containerized deployment (Python 3.10-slim) |
| **Docker Compose** | API + Dashboard orchestration |
| **GitHub Actions** | CI/CD pipeline (test → lint → build) |
| **Pytest** | Unit testing (19 tests, all passing) |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Streamlit    │  │    FastAPI   │  │   CLI/       │              │
│  │  Dashboard    │  │    Backend   │  │   Scripts    │              │
│  │  (port 8501)  │  │  (port 8000) │  │              │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
└─────────┼─────────────────┼─────────────────┼───────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    IMAGE QUALITY GATE                                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Blur Detection  │  Exposure Check  │  Resolution Check     │   │
│  │  (Laplacian var) │  (30-230 mean)   │  (<300px → reject)    │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │  Compression     │  Motion Blur     │  Guidance Text         │   │
│  │  Artifacts       │  Detection       │  ("Please retake...")  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  MODEL INFERENCE PIPELINE                            │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  1. APPLIANCE DETECTOR (YOLO11s)                              │ │
│  │     • Model: yolo11s.pt (18 MB)                               │ │
│  │     • Classes: phone, television, laptop (+6 Phase 2)         │ │
│  │     • Returns: top-3 predictions with confidence scores        │ │
│  │     • Threshold: 35% (below → "Unknown Appliance")            │ │
│  │     • Fallback: YOLOv8n if primary model unavailable           │ │
│  └──────────────┬─────────────────────────────────────────────────┘ │
│                 │                                                    │
│                 ▼                                                    │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  2. ROI CROP → Run detection only on appliance region          │ │
│  └──────────────┬─────────────────────────────────────────────────┘ │
│                 │                                                    │
│        ┌────────┴────────┐                                           │
│        ▼                  ▼                                          │
│  ┌──────────┐    ┌──────────────┐                                   │
│  │ DAMAGE   │    │ DAMAGE       │  ← Optional segmentation          │
│  │ DETECTOR │    │ SEGMENTATION │     (YOLO11s-seg)                 │
│  │ (YOLO)   │    │ (YOLO11s-seg)│                                   │
│  └────┬─────┘    └──────┬───────┘                                   │
│       │                  │                                           │
│       └──────┬───────────┘                                           │
│              ▼                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  3. POST-PROCESSING                                            │ │
│  │     • Non-Max Suppression (IoU threshold 0.5)                  │ │
│  │     • Area validation (0.1%–60% of image)                      │ │
│  │     • Reflection/shadow filter (brightness gradient)           │ │
│  │     • Location inference ("Upper Door", "Screen", etc.)        │ │
│  │     • Confidence filter: 0.4 global minimum                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  4. MISSING PART DETECTOR (Rule-based)                        │ │
│  │     • Checks for missing camera, buttons, keys, hinges, etc.  │ │
│  │     • Appliance-specific component templates                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ANALYSIS SERVICES                               │
│                                                                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐ │
│  │ SEVERITY   │  │  FRAUD     │  │  REPAIR    │  │  CLAIM       │ │
│  │ SERVICE    │  │  SERVICE   │  │  COST      │  │  RECOMMEND   │ │
│  │            │  │            │  │  SERVICE   │  │              │ │
│  │ Type×Area  │  │ 10-factor  │  │ Base×Sev   │  │ Weighted     │ │
│  │ ×Defect    │  │ engine     │  │ ×Conf      │  │ decision     │ │
│  │ 0-100      │  │ 0-100      │  │ Breakdown  │  │ engine       │ │
│  │ 4 bands    │  │ 4 levels   │  │ per damage │  │ Justification│ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └──────┬───────┘ │
│        │               │               │                │          │
│        └───────────────┴───────────────┴────────────────┘          │
│                              │                                      │
│                              ▼                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  5. EXPLAINABLE AI (XAI)                                      │ │
│  │     Generates 5-section natural language explanation:          │ │
│  │     • Appliance Classification: "Detected Refrigerator (97%)" │ │
│  │     • Damage Assessment: "Found 2 damages: crack at upper     │ │
│  │       door (85% confidence, ~2% area)..."                     │ │
│  │     • Fraud Analysis: "Fraud Score: 25/100 (Low)..."         │ │
│  │     • Repair Estimate: "Estimated cost: ₹3,500-₹6,300..."    │ │
│  │     • Claim Decision: "Approved — condition is good..."      │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OUTPUT & REPORTING                             │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ DASHBOARD    │  │  PDF REPORT  │  │  CLAIM       │              │
│  │              │  │              │  │  HISTORY     │              │
│  │ Side-by-side │  │ Original +   │  │ SQLite       │              │
│  │ images       │  │ Annotated    │  │ storage      │              │
│  │ Confidence   │  │ images       │  │ Searchable   │              │
│  │ bars         │  │ Severity     │  │ Exportable   │              │
│  │ XAI panel    │  │ breakdown    │  │ Audit trail  │              │
│  │ Analytics    │  │ XAI section  │  │              │              │
│  │ Monitor tab  │  │ Decision     │  │              │              │
│  │ Multi-image  │  │ justification│  │              │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  MONITORING & OBSERVABILITY (SQLite + In-Memory)              │ │
│  │  • Per-module inference timing    • Error rate tracking        │ │
│  │  • Model version logging          • Session performance stats  │ │
│  │  • Confidence trends              • Audit trail for decisions  │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module-by-Module Deep Dive

### 1. Configuration System (`configs/config.py` — 294 lines)

Central configuration hub with **every threshold exposed** as a configurable constant:

```python
MODEL_CONFIG = {
    "appliance_detector": {
        "model_type": "yolo11", "input_size": 640,
        "confidence_threshold": 0.35, "iou_threshold": 0.45,
    },
    "damage_detector": {
        "confidence_threshold": 0.35,
        "min_damage_confidence": 0.35,
        "min_damage_area_ratio": 0.001,
        "max_damage_area_ratio": 0.6,
        "nms_iou_threshold": 0.5,
        "global_confidence_filter": 0.4,
    },
}
```

**Key configurable thresholds:**
- Appliance confidence: 35% (below → "Unknown")
- Damage confidence: 35% per detection, 40% global filter
- Damage area: 0.1%–60% of image (outside → noise)
- Fraud scores: Low < 30, Medium < 50, High < 75, Critical ≥ 75
- Claim decision: Approve < 25, Manual Review < 50, High < 75, Reject ≥ 75
- Severity bands: Minor < 10%, Moderate < 30%, Major < 60%, Severe ≥ 60%
- Condition grades: A (85–100), B (70–84), C (50–69), D (0–49)
- Damage type weights: crack=1.5, dent=1.0, display_lines=2.0, etc.
- Repair severity multipliers: Minor=0.5×, Moderate=1.0×, Major=1.8×, Severe=3.0×

### 2. Image Quality Validation (`services/image_quality.py` — 149 lines)

**Purpose:** Gate inference behind quality checks — prevents unreliable results.

**Checks performed:**
| Check | Method | Threshold | Penalty |
|---|---|---|---|
| **Blur** | Laplacian variance | < 80.0 | −25 points |
| **Underexposure** | Mean brightness | < 30/255 | −15 points |
| **Overexposure** | Mean brightness | > 230/255 | −15 points |
| **Low Resolution** | Dimensions | < 300×300 | −20 points |
| **JPEG Artifacts** | 16×16 block grid edge density | > 30% blocks | −10 points |
| **Motion Blur** | Laplacian ratio test | < 0.3 ratio | −20 points |

**Output:** `QualityResult(passed, score 0-100, issues[], guidance_text)`
- Pass threshold: score ≥ 40 AND fewer than 4 issues
- Failure → returns structured guidance ("Please retake with the device held steady")

### 3. Appliance Detector (`models/appliance_detector/__init__.py` — 202 lines)

**Model:** YOLO11s (`yolo11s.pt`, 18 MB) — fallback YOLOv8n (`yolov8n.pt`, 6.2 MB)  
**Input:** 640×640 RGB image  
**Output:** Top-3 appliance predictions with confidence scores

**Key Behavior:**
- `detect_all(image)` → returns list of up to 5 predictions sorted by confidence
- `detect_single(image)` → returns highest-confidence prediction or `None`
- Confidence threshold 0.35: below this → returns `None` → "Unknown Appliance"
- Never forces a wrong class — if unsure, escalates to manual review
- Tracks inference time and model version

**Supported Appliances:**
| Tier | Appliances | Damage Types |
|---|---|---|
| **MVP** | phone, television, laptop | crack, dent, display_lines |
| **Phase 2** | tablet, monitor, refrigerator, washing_machine, air_conditioner, microwave | Appliance-specific (see config) |

**Sample output:**
```json
[
  {"class_name": "refrigerator", "confidence": 0.97, "bbox": [45, 30, 590, 610]},
  {"class_name": "washing_machine", "confidence": 0.02, "bbox": ...},
  {"class_name": "microwave", "confidence": 0.01, "bbox": ...}
]
```

### 4. Damage Detector (`models/damage_detector/__init__.py` — 307 lines)

**Primary:** YOLO bbox-based damage detection (phone_damage_best.pt 23 MB, laptop_damage_best.pt 18 MB, refrigerator_damage_best.pt 6 MB)  
**Fallback:** CV heuristic pipeline (Canny edge detection, contour analysis, Hough line detection)  
**Segmentation Option:** YOLO11s-seg (`yolo11s-seg.pt`, 20 MB) for polygon masks

**Post-Processing Pipeline:**
1. **Non-Max Suppression (NMS)** — IoU threshold 0.5, removes duplicate detections on same damage
2. **Area Validation** — damage bbox must be 0.1%–60% of image area (too small = noise, too large = full-image false positive)
3. **Reflection/Shadow Filter** — checks region brightness (> 220 mean) and standard deviation (< 8 with > 180 mean)
4. **Location Inference** — maps bbox position to appliance-aware labels:
   - **Refrigerator:** Upper Door (cy < 50%), Lower Door (cy ≥ 50%)
   - **Phone/Laptop/TV:** Screen, Keyboard, Hinge, Trackpad, Body, Edges, etc.
   - **General:** Surface, Top, Bottom, Center
5. **Confidence Filter** — global minimum 0.4 (configurable)

**Heuristic Fallback (when YOLO model unavailable or returns nothing):**
- **Crack detection:** Canny edges → dilate → largest contour → if area > 0.2% of image and min dimension > 40px
- **Dent detection:** Gaussian blur difference → threshold → contours → 0.3%–20% area range
- **Display lines detection:** HoughLinesP — if ≥ 6 lines found, groups into bbox

**Output:**
```json
[
  {
    "class_name": "crack",
    "confidence": 0.85,
    "bbox": [120, 340, 200, 390],
    "location": "upper_door",
    "source": "yolo"
  },
  {
    "class_name": "dent",
    "confidence": 0.62,
    "bbox": [400, 200, 480, 260],
    "location": "lower_door",
    "source": "yolo"
  }
]
```

### 5. Damage Segmentation (`models/damage_segmentation/__init__.py` — 165 lines)

**Model:** YOLO11s-seg (`yolo11s-seg.pt`, 20 MB)  
**Purpose:** Provides exact polygon masks of damage regions instead of bounding boxes

**Features:**
- Returns contours as polygon point lists
- Converts masks to bounding boxes for backward compatibility
- Calculates mask pixel area for precise damage percentage
- Checks if damage is within beam/scan bounds
- Enables future heatmap visualization

### 6. Fraud Detection Engine (`services/fraud_service.py` — 457 lines)

**10-factor fraud scoring engine** (0–100) with **4 risk levels** (Low/Medium/High/Critical):

| Factor | Detection Method | Max Score |
|---|---|---|
| **1. Error Level Analysis (ELA)** | JPEG re-compression analysis | Weighted 20% of total |
| **2. Metadata Anomalies** | EXIF analysis — missing camera, date, orientation; detects Photoshop, GIMP, Stable Diffusion, Midjourney, etc. | 40 |
| **3. Screenshot Detection** | Edge density (< 2%), solid borders (std < 5), uniform color blocks | 30 |
| **4. AI-Generated Image** | DCT frequency ratio (low/high > 50×), noise analysis (std < 2.0), Laplacian variance < 5 | 30 |
| **5. Copy-Move Detection** | 16×16 block grid matching — identical blocks at different positions | 20 |
| **6. Color Diversity** | Unique pixel count / total pixels ratio (< 0.001 → synthetic) | 15 |
| **7. Tampering Edges** | High-pass filter → contour analysis → suspicious edge clusters | 10 |
| **8. Duplicate Image** | Persistent perceptual hash in SQLite — cross-session and same-session detection | 40 |
| **9. Resolution Mismatch** | Too small (< 150px), too large (> 8000px), unusual aspect ratio | 20 |
| **10. Compression Anomalies** | Laplacian var < 10 (over-compressed), 8×8 DCT block uniformity | 15 |

**Duplicate detection** uses a **persistent SQLite database** (`fraud_hashes.db`) storing 16×16 perceptual hashes with first-seen timestamps and occurrence counts. If the same hash (or near-identical hash within 2-bit Hamming distance) is seen across sessions, it flags it as a potentially fraudulent claim reuse.

**Output:**
```json
{
  "fraud_score": 65,
  "risk_level": "High",
  "reasons": [
    "High ELA score — possible digital manipulation",
    "No metadata / file path missing",
    "Uniform 8x8 blocks — JPEG compression artifacts",
    "Suspicious repeated blocks — possible copy-move manipulation"
  ],
  "explanation": "Fraud Score: **65/100** (Risk: **High**). Indicators: ... Multiple fraud indicators present. Manual review strongly recommended."
}
```

### 7. Severity Service (`services/severity_service.py` — 153 lines)

**Core Formula:**
```
severity_pct = damage_area_pct × damage_type_weight × defect_multiplier
condition_score = 100 − Σ(severity_impact × weight × area_factor)
```

**Damage Type Weights:**
| Type | Weight | Rationale |
|---|---|---|
| panel_damage | 2.5 | Structural → highest impact |
| display_lines | 2.0 | Display → expensive repair |
| screen_crack | 1.8 | Screen → common high-cost |
| crack | 1.5 | Structural compromise |
| rust | 1.2 | Progressive deterioration |
| body_damage | 1.1 | Cosmetic + structural |
| dent | 1.0 | Baseline cosmetic |
| scratch | 0.6 | Minor cosmetic |
| dead_pixels | 0.8 | Display but low impact |

**Defect Multiplier:** 1.0 + (count − 1) × 0.15 — multiple same-type damages compound severity

**Severity Bands:**
| Label | Weighted % Range | Interpretation |
|---|---|---|
| None | 0% | No damage |
| Minor | 0–10% | Cosmetic only |
| Moderate | 10–30% | Repairable, still usable |
| Major | 30–60% | Significant functional impact |
| Severe | 60–100% | May be beyond economical repair |

**Condition Grade:**
| Grade | Score Range | Meaning |
|---|---|---|
| A | 85–100 | Excellent — like new |
| B | 70–84 | Good — minor wear |
| C | 50–69 | Fair — damage present |
| D | 0–49 | Poor — significant damage |

### 8. Repair Cost Service (`services/repair_service.py` — 119 lines)

**Core Formula:**
```
cost = base_cost × severity_multiplier × confidence_factor
```

**Base Costs (per damage type):**
| Damage Type | Min (₹) | Max (₹) |
|---|---|---|
| crack | 150 | 400 |
| dent | 100 | 300 |
| display_lines | 500 | 1,500 |
| screen_crack | 300 | 800 |
| panel_damage | 400 | 1,500 |
| rust | 200 | 600 |
| scratch | 50 | 150 |

**Severity Multipliers:**
| Severity | Multiple |
|---|---|
| Minor | 0.5× |
| Moderate | 1.0× |
| Major | 1.8× |
| Severe | 3.0× |

**Confidence Factor:** 0.5 + (confidence × 0.5) — low-confidence detections have lower cost impact

**Output includes per-damage breakdown:**
```json
{
  "total_display": "₹3,500 - ₹6,300",
  "breakdown": [
    {"damage_type": "crack", "base_range": "₹150-₹400", "severity_multiplier": 1.8, "cost_min": 315, "cost_max": 840},
    {"damage_type": "display_lines", "base_range": "₹500-₹1,500", "severity_multiplier": 1.0, "cost_min": 650, "cost_max": 1950}
  ]
}
```

### 9. Claim Recommendation Engine (`services/claim_recommendation.py` — 105 lines)

**Weighted decision formula:**
```
claim_score = severity_risk × 0.4 + fraud_score × 0.3 + condition_risk × 0.2 + damage_count_risk × 0.1
```

**Risk Weights:**
| Factor | Weight | Source |
|---|---|---|
| Severity | 40% | None=0, Minor=10, Moderate=30, Major=60, Severe=80 |
| Fraud Score | 30% | Direct from fraud engine (0–100) |
| Condition | 20% | A=0, B=15, C=40, D=70 (inverted) |
| Damage Count | 10% | 0=0, 1=10, 2-3=30, 4+=50 |

**Decision Thresholds:**
| Score Range | Decision | Action |
|---|---|---|
| 0–24 | **APPROVE** | Fast-track, no human needed |
| 25–49 | **MANUAL_REVIEW** | Requires adjuster review |
| 50–74 | **MANUAL_REVIEW** | High-risk — senior adjuster |
| 75–100 | **REJECT** | Deny claim, escalate to fraud team |

**Justification Generation:**
- **APPROVE:** "Claim qualifies for automatic approval. Appliance condition is good. No significant fraud indicators detected."
- **MANUAL_REVIEW:** "Manual review needed because damage severity is high and fraud score is elevated."
- **REJECT:** "Claim recommended for rejection. Fraud indicators present. Damage severity is at maximum level."

### 10. Explainable AI (XAI) Service (`services/explain_service.py` — 167 lines)

Generates 5-section natural language explanation for every inspection:

**1. Appliance Explanation:**
> "Detected appliance: **Refrigerator** (confidence: 97%). Alternative predictions: Washing Machine (2%), Microwave (1%). Low confidence — manual verification recommended."

**2. Damage Explanation:**
> "Found **2** damage(s): 1. **crack** at **upper_door** (confidence: 85%, area: ~2% of image). 2. **dent** at **lower_door** (confidence: 62%, area: ~1% of image).  
> Overall severity: **Moderate**. Condition score: **82/100** (grade **B**).  
> Moderate damage — repair recommended but appliance still usable."

**3. Fraud Explanation:**
> "Fraud Score: **25/100** (Risk: **Low**). No significant fraud indicators detected."

**4. Repair Explanation:**
> "Estimated repair cost: **₹3,500 – ₹6,300**. Breakdown: crack: ₹315-₹840. dent: ₹180-₹540. Moderate damage — cost estimate is moderate."

**5. Claim Explanation:**
> "Claim assessment: risk **low** (score: **22/100**). Recommended decision: **APPROVE**. Low risk — no issues detected. Fast-track processing."

### 11. Multi-Image Inspection Service (`services/multi_image_service.py` — 329 lines)

**Purpose:** Upload 2–6 photos from different angles → single unified report.

**View Classification:**
| Aspect Ratio | Edge Density | Classification |
|---|---|---|
| > 1.8 | Any | Close-up |
| < 0.5 | Any | Detail |
| Any | < 0.01 | Unknown |
| Bottom quarter bright (> 180 mean) | Any | Top view |
| Default | Any | Front view |

**Detection Merge Algorithm:**
1. Run inference independently on each image
2. Group detections by class name across images
3. Within each group, apply IoU-based clustering (threshold 0.3)
4. For each cluster: keep detection with highest confidence, average bbox coordinates
5. Track `source_images` to identify which images contributed to each merged detection
6. Select best evidence image per damage type for report inclusion

### 12. Monitoring & Observability (`services/monitoring.py` — 208 lines)

**Three-tier monitoring:**
1. **SQLite Persistence** (`monitor.db`): Every inference logged with module, operation, duration, success/fail, error, model version, confidence
2. **In-Memory Session Stats:** Running averages per module:operation for dashboard display
3. **Error Tracking:** Recent errors with full context for debugging

**Tracked Metrics:**
- Total calls per module
- Average/max/min duration (ms)
- Success rate (%)
- Average confidence
- Error count and types

**Usage in code:**
```python
with monitor.track("api", "inspect_image"):
    result = pipeline.inspect_image(path)
```

### 13. Professional PDF Reports (`services/pdf_service.py` — 267 lines)

**Insurer-ready PDF structure:**

| Page | Section | Content |
|---|---|---|
| 1 | **Header** | Dark blue banner (RGB 25,55,109), claim ID, timestamp |
| 1 | **Decision Badge** | Green check (APPROVE), yellow warning (MANUAL_REVIEW), red X (REJECT) |
| 1 | **1. Appliance Details** | Type, confidence, condition score + grade |
| 1 | **2. Damage Assessment** | Per-damage: type, location, confidence, area % |
| 1 | **3. Severity Breakdown** | Base range, severity multiplier, total estimate |
| 1 | **4. Fraud Analysis** | Score, risk level, all reasons |
| 2 | **5. XAI Reasoning** | 5 sections: appliance, damage, fraud, repair, claim |
| 2 | **6. Inspection Images** | Original + annotated |
| 3 | **7. Claim Summary** | Decision, risk score, justification, metadata |

### 14. Video Inspection (`scripts/inference.py` — 285 lines)

**Smart Frame Extraction:**
- Strategically samples frames from beginning, 25%, 50%, 75%, and end of video
- Each frame runs through the full pipeline independently
- Damage **persistence tracking** — damages that appear in ≥ 30% of frames are flagged as persistent
- Frame **inconsistency detection** — if a frame lacks persistent damages, it's flagged as potentially fraudulent
- **Best frame selection** — scores frames by `damage_confidence × 0.6 + (1 − fraud) × 0.4`, picks the best
- Outputs annotated video and per-frame JSON summary

### 15. Streamlit Dashboard (`dashboard/app.py` — 787 lines)

**6-tab navigation:**

| Tab | Features |
|---|---|
| **Image** | Upload image → quality check → run inspection → side-by-side original/annotated → confidence bars → 5 detail tabs (Confidence, Damage, Repair Cost, Fraud, Claim) → XAI panel → Save PDF |
| **Video** | Upload video → frame extraction → per-frame analysis → persistence tracking → annotated video output → summary download |
| **Multi-Image** | Upload 2-6 images → view classification → per-image inference → IoU merge → unified report → per-image quality scores → merged damage table |
| **History** | Claim list with stats (total, high-risk, avg fraud, avg condition) → searchable table → view details → download PDF |
| **Analytics** | Key metrics (total inspections, avg fraud, avg claim, avg condition, avg repair) → severity distribution → claim risk distribution → appliance distribution → fraud score trends → condition trends → decision breakdown → damage detection rate |
| **Monitor** | Total calls, error rate, module timing table, session stats, recent errors |

**Professional UI Elements:**
- Decision banner (green/yellow/red) at top with icon
- Colored severity indicator dots
- Confidence bars (green ≥ 70%, yellow 40–69%, red < 40%)
- Explanation boxes with blue left border
- Metric cards with background shading
- Responsive multi-column layouts

### 16. FastAPI Backend (`api/__init__.py` — 330 lines)

**13 API Endpoints:**

| Method | Endpoint | Purpose | Auth |
|---|---|---|---|
| GET | `/` | Project info | No |
| GET | `/health` | Health check | No |
| GET | `/api/v1/info` | API metadata | No |
| POST | `/api/v1/quality` | Image quality check | No |
| POST | `/api/v1/inspect/image` | Single image inspection | API key |
| POST | `/api/v1/inspect/multi` | Multi-image (2–6) inspection | API key |
| POST | `/api/v1/inspect/video` | Video inspection | API key |
| POST | `/api/v1/fraud/advanced` | Fraud analysis only | No |
| POST | `/api/v1/severity` | Severity computation | No |
| GET | `/api/v1/claims` | List claims | No |
| GET | `/api/v1/claims/{id}` | Get claim details | No |
| GET | `/api/v1/claims/{id}/pdf` | Download claim PDF | No |
| GET | `/api/v1/monitor/stats` | Monitoring stats | No |

### 17. Claim History (`services/claim_service.py` — 150 lines)

**SQLite schema** (`data/claim_history.db`):
```sql
CREATE TABLE claims (
    claim_id TEXT PRIMARY KEY,
    timestamp TEXT,
    appliance TEXT,
    severity TEXT,
    fraud_score REAL,
    repair_cost REAL,
    condition_score REAL,
    claim_risk TEXT,
    decision TEXT,
    full_report TEXT   -- JSON blob
);
```

**Supported operations:** `save_claim`, `get_claims` (pagination), `get_claim_by_id`, `get_claim_stats`

### 18. Fraud Detection Base (`fraud_detection/__init__.py` — 129 lines)

Base classes used by the advanced fraud engine:
- **ELADetector:** JPEG re-compress at 90% quality → pixel-wise difference → mean squared error → normalized ELA score
- **MetadataAnalyzer:** EXIF extraction via Pillow → TAGS mapping → camera model, date, software detection
- **FraudDetectionEngine:** Combines ELA + metadata into unified analysis

### 19. Risk Engine (`risk_engine/__init__.py` — 164 lines)

Legacy risk engine providing backward compatibility:
- **RiskAssessment:** Full report assessment with damage %, severity, condition, grade, repair cost, decision
- **DamageSeverityEstimator:** Original area-based severity calculation
- **RepairCostEstimator:** Original cost rules from REPAIR_COST_RULES config
- **RiskEngine:** Orchestrates assessment using weighted risk formula

### 20. Utils (`utils/__init__.py` — 467 lines)

Comprehensive utility library:
- **Image I/O:** `read_image` (handle Unicode paths), `save_image`, `resize_image`, `normalize_image`, `denormalize_image`
- **CV Operations:** `calculate_iou`, `non_max_suppression`
- **Class Mapping:** `get_appliance_from_class_id`, `get_class_id_from_appliance`
- **File Validation:** `validate_image_file` (PIL verify), `validate_video_file` (OpenCV check)
- **Config:** `load_config`, `save_config` (YAML)
- **Data:** `load_json`, `save_json`
- **Helpers:** `setup_logging`, `AverageMeter`, `format_size`
- **Device Detection:** `get_device` → auto-selects CUDA / MPS / CPU

---

## Model Files

| File | Size | Architecture | Purpose |
|---|---|---|---|
| `yolo11s.pt` | 18 MB | YOLO11s | Primary appliance detector |
| `yolo11s-seg.pt` | 20 MB | YOLO11s-seg | Damage segmentation (polygons) |
| `yolov8n.pt` | 6.2 MB | YOLOv8n | Fallback appliance detector |
| `phone_damage_best.pt` | 23 MB | YOLO11 | Phone damage detection |
| `laptop_damage_best.pt` | 18 MB | YOLO11 | Laptop damage detection |
| `refrigerator_damage_best.pt` | 6 MB | YOLO11 | Refrigerator damage detection |
| Training checkpoints | ~150 MB total | Various | Epoch checkpoints across runs |

**Device Support:** CUDA (NVIDIA), MPS (Apple Silicon M1-M4), CPU (any)

---

## Training Pipeline

Three dedicated training scripts:

| Script | Target | Classes | Dataset Structure |
|---|---|---|---|
| `train_appliance_detector.py` | Appliance detection | phone, television, laptop | `datasets/appliance_detector/{train,val,test}` |
| `train_damage_detector.py` | Damage detection | crack, dent, display_lines | `datasets/damage_detector/{appliance}/{train,val,test}` |
| `train_missing_part_detector.py` | Missing parts | Appliance-specific | Future |

**Training Configuration:**
```yaml
epochs: 50
batch_size: 8
image_size: 640
optimizer: Adam
learning_rate: 0.001
weight_decay: 0.0005
warmup_epochs: 3
patience: 15
save_period: 5
```

**Evaluation Script:** `scripts/evaluate.py` — computes precision, recall, mAP50, mAP50-95, per-class performance, confusion matrix, F1 curves

---

## Deployment

### Docker
```dockerfile
FROM python:3.10-slim
# Installs OpenCV system deps (libgl1, libglib2.0, etc.)
# Runs API server on port 8000 by default
```

### Docker Compose
```yaml
services:
  api:     # FastAPI on :8000
  dashboard:  # Streamlit on :8501
```
Both services share volumes for models, data, output, and reports.

### CI/CD (GitHub Actions)
```yaml
jobs:
  test:   # pytest on every push/PR
  lint:   # flake8 + mypy
  build:  # Docker image build (main branch only)
```

### API Authentication
Optional API key auth via `X-API-Key` header. Configured via `API_KEYS` list in `api/__init__.py`.

---

## Key Design Decisions

### 1. Never Force Wrong Classification
The appliance detector returns `None` when confidence < 35% instead of forcing the best-guess class. This prevents the classic MVP mistake of confidently misclassifying a refrigerator as a television. "Unknown Appliance" is a valid, safe output that triggers manual review.

### 2. No Fabricated Damage
When the damage detector finds no detections above threshold, it returns an empty list. The pipeline then correctly reports "No Damage Detected" with condition score 100/100 and grade A. Previously, empty detections triggered heuristic fallbacks that fabricated severity and costs.

### 3. Damage Type Weights Instead of Hard Rules
Severity uses continuous weighted scoring (crack=1.5×, dent=1.0×, display_lines=2.0×) rather than lookup tables. This generalizes to new damage types without code changes — just add a weight.

### 4. Explainability as a First-Class Feature
Every decision has a corresponding natural-language explanation. This is critical for insurance: adjusters need to understand *why* a claim was approved/reviewed/rejected, not just the outcome.

### 5. Multi-Layer Fraud Detection
A single high ELA score doesn't trigger rejection — it's combined with 9 other factors. This reduces false positives while catching sophisticated fraud (AI-generated images, copy-move, screenshots of genuine images).

### 6. Quality Gate Before Inference
Rather than letting blurry/overexposed images through the pipeline and getting unreliable results, the quality check rejects them upfront with actionable guidance. This dramatically improves trust in the system.

### 7. Persistent Duplicate Detection
Fraud hashes persist in SQLite across sessions. If someone uploads the same photo to two different claims weeks apart, the system detects it. This is essential for production insurance fraud detection.

---

## Test Coverage (19 tests, all passing)

| Test Class | Tests | Coverage |
|---|---|---|
| `TestUtils` | 8 | IoU, NMS, resize, normalize/denormalize, class mapping |
| `TestFraudDetection` | 3 | ELA, MetadataAnalyzer, FraudDetectionEngine imports |
| `TestRiskEngine` | 3 | RiskEngine, DamageSeverityEstimator, RepairCostEstimator |
| `TestReportEngine` | 2 | InspectionReport creation, serialization |
| `TestModels` | 2 | ApplianceDetector, DamageDetector imports |
| `TestMissingPartDetector` | 1 | MissingPartDetector import |

---

## Performance Characteristics

| Module | Avg Time | Notes |
|---|---|---|
| Image Quality Check | < 10 ms | Pure OpenCV operations |
| Appliance Detection | 50–150 ms | YOLO11s on MPS/CUDA |
| Damage Detection | 50–200 ms | YOLO + heuristics |
| Fraud Detection | 200–500 ms | ELA is most expensive |
| Severity + Repair | < 5 ms | Pure math operations |
| Full Pipeline | 300–900 ms | Single image, MPS/CUDA |

---

## Limitations & Known Issues

1. **Phone crack model** detects entire phone instead of crack region — this is a dataset labeling issue, not code-fixable
2. **Fridge damage model** has mAP50 = 0.249 (trained on ~50 images) — confidence filter 0.4 helps suppress false positives
3. **TV damage model** slot is `None` — no dataset available
4. **Segmentation models** not yet trained for specific appliances — placeholder architecture
5. **Missing part detector** is rule-based, not ML-driven — basic but functional
6. **HEIC/HEIF image formats** not supported (iOS default) — convert to JPEG before upload

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run API
python scripts/run_api.py

# Run Dashboard (in another terminal)
python scripts/run_dashboard.py

# CLI Inference
python scripts/inference.py --image path/to/photo.jpg --save-vis

# Run Tests
python -m pytest tests/ -v

# Docker Deployment
docker-compose up --build
```

---

*Generated for: Insurance Industry Evaluation  
Platform Version: 3.0.0  
Codebase: 7,800+ lines · 38 Python files · 250 MB model weights  
Last Updated: June 2026*
