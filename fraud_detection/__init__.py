"""
Fraud detection for the MVP.

Priority fraud modules:
- Error Level Analysis (ELA)
- Metadata analysis
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


@dataclass
class FraudAnalysisResult:
    ela_score: float = 0.0
    metadata_risk_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ela_score": self.ela_score,
            "metadata_risk_score": self.metadata_risk_score,
            "metadata": self.metadata or {},
        }


class ELADetector:
    name = "Error Level Analysis"

    def __init__(self, quality: int = 82):
        self.quality = quality

    def analyze(self, image: np.ndarray) -> float:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        success, encoded = cv2.imencode(".jpg", image, encode_param)
        if not success:
            return 0.0
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decoded = cv2.resize(decoded, (image.shape[1], image.shape[0]))
        diff = cv2.absdiff(image.astype(np.float32), decoded.astype(np.float32))
        score = float(np.mean(diff) / 255.0)
        return min(score * 12.0, 1.0)

    def ela_map(self, image: np.ndarray) -> np.ndarray:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        success, encoded = cv2.imencode(".jpg", image, encode_param)
        if not success:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decoded = cv2.resize(decoded, (image.shape[1], image.shape[0]))
        diff = cv2.absdiff(image.astype(np.float32), decoded.astype(np.float32))
        ela = np.mean(diff, axis=2)
        ela = (ela - ela.min()) / (ela.max() - ela.min() + 1e-6)
        return (ela * 255).astype(np.uint8)


class MetadataAnalyzer:
    name = "Metadata Analyzer"
    FLAGGED_SOFTWARE = ("photoshop", "gimp", "stable diffusion")

    def analyze(self, image_path: Optional[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "camera_model": None,
            "software_used": None,
            "timestamp": None,
            "missing_metadata": True,
            "flagged_software": [],
            "risk_score": 0.0,
        }
        if not image_path or not os.path.exists(image_path):
            result["risk_score"] = 0.35
            return result

        try:
            image = Image.open(image_path)
            exif = image.getexif()
            if not exif:
                result["risk_score"] = 0.35
                return result

            result["missing_metadata"] = False
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in {"Make", "Model"}:
                    prior = result["camera_model"] or ""
                    result["camera_model"] = f"{prior} {value}".strip()
                elif tag == "Software":
                    result["software_used"] = str(value)
                elif tag in {"DateTime", "DateTimeOriginal"}:
                    result["timestamp"] = str(value)

            if result["software_used"]:
                software_lower = result["software_used"].lower()
                for flagged in self.FLAGGED_SOFTWARE:
                    if flagged in software_lower:
                        result["flagged_software"].append(flagged)
                        result["risk_score"] += 0.35

            if not result["camera_model"]:
                result["risk_score"] += 0.2
            if not result["timestamp"]:
                result["risk_score"] += 0.1
            result["risk_score"] = min(result["risk_score"], 1.0)
            return result
        except Exception:
            result["risk_score"] = 0.35
            return result


class FraudDetectionEngine:
    def __init__(self):
        self.ela_detector = ELADetector()
        self.metadata_analyzer = MetadataAnalyzer()

    def analyze(self, image: np.ndarray, image_path: Optional[str] = None) -> FraudAnalysisResult:
        metadata = self.metadata_analyzer.analyze(image_path)
        return FraudAnalysisResult(
            ela_score=self.ela_detector.analyze(image),
            metadata_risk_score=float(metadata["risk_score"]),
            metadata=metadata,
        )
