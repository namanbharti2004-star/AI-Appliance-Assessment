"""
Configuration for the AI-Powered Appliance Inspection & Insurance Claim Platform.

Scope:
- Appliance detector: phone, television, laptop (YOLO11s)
- Damage detector: crack, dent, display_lines (bbox + segmentation ready)
- Missing-part detector for phone / television / laptop
- Fraud detection: ELA + metadata + 7-factor advanced
- Severity scoring, repair cost estimation, claim risk engine
- PDF reports, claim history (SQLite)
- Supports both images and videos

All thresholds are configurable per service.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

PROJECT_NAME = "AI Appliance Inspection & Insurance Platform"
VERSION = "3.0.0"
API_HOST = "0.0.0.0"
API_PORT = 8000

MVP_APPLIANCE_CLASSES = ["phone", "television", "laptop"]
MVP_DAMAGE_CLASSES = ["crack", "dent", "display_lines"]

MISSING_PART_CLASSES: Dict[str, List[str]] = {
    "phone": ["camera", "buttons"],
    "laptop": ["keys", "hinge_cover"],
    "television": ["stand", "bezel"],
}

PHASE_2_APPLIANCE_CLASSES = [
    "tablet", "monitor", "refrigerator", "washing_machine",
    "air_conditioner", "microwave",
]

MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "appliance_detector": {
        "name": "appliance_detector",
        "model_type": "yolo11",
        "input_size": 640,
        "confidence_threshold": 0.35,
        "iou_threshold": 0.45,
        "max_det": 5,
    },
    "damage_detector": {
        "name": "damage_detector",
        "model_type": "yolo11",
        "input_size": 640,
        "confidence_threshold": 0.35,
        "iou_threshold": 0.45,
        "max_det": 10,
        "min_damage_confidence": 0.35,
        "min_damage_area_ratio": 0.001,
        "max_damage_area_ratio": 0.6,
        "nms_iou_threshold": 0.5,
        "global_confidence_filter": 0.4,
    },
    "damage_segmentation": {
        "name": "damage_segmentation",
        "model_type": "yolo11s-seg",
        "input_size": 640,
        "confidence_threshold": 0.35,
        "iou_threshold": 0.45,
        "max_det": 10,
    },
    "missing_part_detector": {
        "name": "missing_part_detector",
        "model_type": "rules_or_yolo",
        "confidence_threshold": 0.3,
    },
}

APPLIANCE_CLASSES = MVP_APPLIANCE_CLASSES + PHASE_2_APPLIANCE_CLASSES

DAMAGE_CLASSES: Dict[str, List[str]] = {
    "phone": MVP_DAMAGE_CLASSES,
    "television": MVP_DAMAGE_CLASSES,
    "laptop": MVP_DAMAGE_CLASSES,
}

PHASE_3_DAMAGE_CLASSES: Dict[str, List[str]] = {
    "television": ["screen_crack", "dead_pixels", "display_lines", "panel_damage"],
    "laptop": ["screen_crack", "keyboard_damage", "hinge_damage", "body_dent"],
    "phone": ["screen_crack", "camera_crack", "display_lines", "body_damage"],
    "tablet": ["screen_crack", "glass_shatter", "body_damage"],
    "monitor": ["screen_crack", "display_lines", "dead_pixels"],
    "refrigerator": ["dent", "rust", "door_seal_damage", "surface_scratch"],
    "washing_machine": ["drum_damage", "door_damage", "panel_dent", "rust"],
    "air_conditioner": ["fin_damage", "body_dent", "rust"],
    "microwave": ["door_damage", "body_dent", "rust", "glass_damage"],
}

# ========== CONFIGURABLE THRESHOLDS ==========

FRAUD_THRESHOLDS: Dict[str, float] = {
    "ela_score": 0.55,
    "metadata_risk_score": 0.5,
    "fraud_score_low": 30,
    "fraud_score_medium": 50,
    "fraud_score_high": 75,
}

RISK_WEIGHTS: Dict[str, float] = {
    "appliance_confidence": 0.15,
    "damage_confidence": 0.25,
    "ela_score": 0.2,
    "metadata_risk_score": 0.2,
    "missing_part_score": 0.2,
}

DECISION_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "APPROVE": {"fraud_score_max": 30, "condition_score_min": 75, "claim_score_max": 25},
    "MANUAL_REVIEW": {"fraud_score_max": 65, "condition_score_min": 45, "claim_score_max": 75},
    "REJECT": {"fraud_score_min": 65, "claim_score_min": 75},
}

CONDITION_GRADE_RANGES: Dict[str, tuple] = {
    "A": (85, 100),
    "B": (70, 84),
    "C": (50, 69),
    "D": (0, 49),
}

REPAIR_COST_RULES: Dict[str, Dict[str, float]] = {
    "crack": {"base": 3500, "multiplier": 1.8},
    "dent": {"base": 1800, "multiplier": 1.4},
    "display_lines": {"base": 4200, "multiplier": 2.0},
    "missing_part": {"base": 1200, "multiplier": 1.3},
    "screen_crack": {"base": 4000, "multiplier": 2.0},
    "rust": {"base": 2000, "multiplier": 1.5},
    "scratch": {"base": 500, "multiplier": 0.8},
    "body_damage": {"base": 2500, "multiplier": 1.5},
    "dead_pixels": {"base": 0, "multiplier": 0.0},
    "panel_damage": {"base": 5000, "multiplier": 2.5},
}

SEVERITY_RANGES: Dict[str, tuple] = {
    "LOW": (0, 15),
    "MEDIUM": (16, 35),
    "HIGH": (36, 60),
    "CRITICAL": (61, 100),
}

SEVERITY_BANDS: List[tuple] = [
    ("Minor", 0, 10),
    ("Moderate", 10, 30),
    ("Major", 30, 60),
    ("Severe", 60, 101),
]

DAMAGE_TYPE_WEIGHTS: Dict[str, float] = {
    "crack": 1.5,
    "dent": 1.0,
    "display_lines": 2.0,
    "rust": 1.2,
    "scratch": 0.6,
    "body_damage": 1.1,
    "screen_crack": 1.8,
    "dead_pixels": 0.8,
    "panel_damage": 2.5,
    "unknown": 1.0,
}

CLAIM_RISK_CONFIG: Dict[str, Any] = {
    "severity_weights": {"None": 0, "Minor": 10, "Moderate": 30, "Major": 60, "Severe": 80},
    "condition_thresholds": {"good": 90, "fair": 75, "poor": 50},
    "weight_severity": 0.4,
    "weight_fraud": 0.3,
    "weight_condition": 0.2,
    "weight_damage_count": 0.1,
    "approve_threshold": 25,
    "review_threshold": 50,
    "reject_threshold": 75,
}

REPAIR_COST_CONFIG: Dict[str, Any] = {
    "severity_multipliers": {"Minor": 0.5, "Moderate": 1.0, "Major": 1.8, "Severe": 3.0, "None": 0.0},
    "confidence_factor_base": 0.5,
}

DAMAGE_LOCATIONS_CONFIG: Dict[str, List[str]] = {
    "phone": ["screen", "back_panel", "camera", "edges"],
    "television": ["screen", "bezel", "stand", "back_panel"],
    "laptop": ["screen", "keyboard", "hinge", "trackpad", "body"],
    "refrigerator": ["upper_door", "lower_door", "side_panel", "top", "handle"],
}

LOCATION_WEIGHTS: Dict[str, float] = {
    "screen": 2.0,
    "display": 2.0,
    "keyboard": 1.5,
    "hinge": 1.8,
    "trackpad": 1.2,
    "body": 0.8,
    "back_panel": 0.7,
    "camera": 1.6,
    "edges": 0.9,
    "upper_door": 1.0,
    "lower_door": 0.9,
    "side_panel": 0.8,
    "top": 0.7,
    "handle": 0.6,
    "bezel": 1.1,
    "stand": 0.5,
    "front": 1.0,
    "rear": 0.8,
    "left": 1.0,
    "right": 1.0,
    "close_up": 1.2,
    "surface": 1.0,
    "unknown": 1.0,
}

BRAND_COST_MULTIPLIERS: Dict[str, float] = {
    "Apple": 2.5,
    "Samsung": 1.8,
    "LG": 1.5,
    "Sony": 2.0,
    "Dell": 1.6,
    "HP": 1.4,
    "Lenovo": 1.3,
    "Google": 1.7,
    "OnePlus": 1.5,
    "Xiaomi": 1.2,
    "Huawei": 1.4,
    "Nokia": 1.1,
    "unknown": 1.0,
}

# ========== END CONFIGURABLE THRESHOLDS ==========

TRAINING_CONFIG: Dict[str, Any] = {
    "epochs": 50,
    "batch_size": 8,
    "image_size": 640,
    "device": "cuda",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "patience": 15,
    "save_period": 5,
}

PATHS: Dict[str, str] = {
    "datasets": "datasets",
    "models": "models",
    "appliance_detector": "models/appliance_detector",
    "damage_detector": "models/damage_detector",
    "damage_segmentation": "models/damage_segmentation",
    "missing_part_detector": "missing_part_detector",
    "fraud_detection": "fraud_detection",
    "risk_engine": "risk_engine",
    "report_engine": "report_engine",
    "services": "services",
    "dashboard": "dashboard",
    "logs": "logs",
    "temp": "temp",
    "output": "output",
    "reports": "reports",
    "data": "data",
}

MODEL_PATHS: Dict[str, Any] = {
    "appliance_detector": "models/appliance_detector/yolo11s.pt",
    "appliance_detector_fallback": "yolov8n.pt",
    "damage_detector": {
        "phone": "models/damage_detector/phone_damage_best.pt",
        "television": None,
        "laptop": "models/damage_detector/laptop_damage_best.pt",
        "refrigerator": "models/damage_detector/refrigerator_damage_best.pt",
    },
    "damage_segmentation": {
        "phone": None,
        "television": None,
        "laptop": None,
        "refrigerator": None,
    },
    "damage_segmentation_default": "models/damage_segmentation/yolo11s-seg.pt",
}

DATASET_TEMPLATES: Dict[str, Any] = {
    "appliance_detector": {
        "root": "datasets/appliance_detector",
        "splits": ["train", "val", "test"],
        "children": ["images", "labels"],
        "classes": MVP_APPLIANCE_CLASSES,
    },
    "damage_detector": {
        "phone": {"root": "datasets/damage_detector/phone", "classes": MVP_DAMAGE_CLASSES},
        "television": {"root": "datasets/damage_detector/television", "classes": MVP_DAMAGE_CLASSES},
        "laptop": {"root": "datasets/damage_detector/laptop", "classes": MVP_DAMAGE_CLASSES},
    },
    "missing_part_detector": {
        "phone": {"root": "datasets/missing_part_detector/phone", "classes": MISSING_PART_CLASSES["phone"]},
        "television": {"root": "datasets/missing_part_detector/television", "classes": MISSING_PART_CLASSES["television"]},
        "laptop": {"root": "datasets/missing_part_detector/laptop", "classes": MISSING_PART_CLASSES["laptop"]},
    },
}

LOG_CONFIG: Dict[str, str] = {
    "level": "INFO",
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
}

API_CONFIG: Dict[str, Any] = {
    "max_upload_size": 50 * 1024 * 1024,
    "allowed_image_formats": [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"],
    "allowed_video_formats": [".mp4", ".mov", ".avi"],
    "max_video_frames": 24,
    "cors_origins": ["*"],
}

DASHBOARD_CONFIG: Dict[str, str] = {
    "theme": "light",
    "title": "AI Appliance Inspection · Insurance Platform",
    "layout": "wide",
    "page_icon": "\u2699\ufe0f",
}


def get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except Exception:
        return "cpu"
