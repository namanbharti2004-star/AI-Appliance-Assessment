"""
Segmentation damage detection service.

Provides a clean API for the pipeline to use YOLO11s-seg segmentation.
Falls back to bbox-based damage detection when no segmentation model is available.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

from configs.config import MODEL_PATHS
from models.damage_segmentation import DamageSegmentationDetector


class SegmentationService:
    def __init__(self):
        self._detector: Optional[DamageSegmentationDetector] = None
        self._enabled = False

    def initialize(self, model_path: Optional[str] = None) -> bool:
        try:
            self._detector = DamageSegmentationDetector(model_path=model_path)
            self._enabled = self._detector.model is not None
            if self._enabled:
                logger.info("Segmentation service initialized with model")
            else:
                logger.info("Segmentation service initialized without model (bbox fallback)")
            return self._enabled
        except Exception as exc:
            logger.error("Segmentation service init failed: {}", exc)
            self._enabled = False
            return False

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._detector is not None and self._detector.model is not None

    def detect(
        self, image: np.ndarray, roi: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        if not self.is_enabled:
            logger.debug("Segmentation not available, returning empty")
            return []
        return self._detector.detect(image, roi=roi)

    def compute_mask_area(self, detection: Dict[str, Any]) -> float:
        mask_data = detection.get("mask")
        if mask_data:
            return float(np.sum(np.array(mask_data)))
        bbox = detection.get("bbox")
        if bbox and len(bbox) == 4:
            return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        return 0.0

    def compute_damage_percentage(
        self, damage_detections: List[Dict[str, Any]], appliance_area: float
    ) -> float:
        if not damage_detections or appliance_area <= 0:
            return 0.0
        total_damage_area = sum(self.compute_mask_area(d) for d in damage_detections)
        return min(total_damage_area / appliance_area * 100, 100.0)

    def classify_severity(self, percentage: float) -> str:
        if percentage <= 0:
            return "None"
        if percentage <= 10:
            return "Minor"
        if percentage <= 30:
            return "Moderate"
        if percentage <= 60:
            return "Major"
        return "Severe"

    def get_overlay_mask(
        self, image: np.ndarray, detection: Dict[str, Any], color: tuple = (0, 0, 255), alpha: float = 0.4
    ) -> np.ndarray:
        overlay = image.copy()
        mask_data = detection.get("mask")
        if mask_data:
            mask_np = np.array(mask_data, dtype=np.uint8)
            h, w = overlay.shape[:2]
            if mask_np.shape[:2] != (h, w):
                mask_np = cv2.resize(mask_np, (w, h))
            colored_mask = np.zeros_like(overlay)
            colored_mask[mask_np > 0] = color
            overlay = cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0)
        return overlay
