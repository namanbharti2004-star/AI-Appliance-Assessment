"""
Segmentation-based damage detector using YOLO11s-seg.

Detects exact damage regions as polygons/masks instead of bounding boxes.
Provides a migration path from bbox-based damage detection to segmentation.

Supported appliances: phone, television, laptop, refrigerator
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from configs.config import MODEL_CONFIG, MODEL_PATHS, MVP_DAMAGE_CLASSES, get_device


SEG_CLASS_NAME_MAP = {
    "cracked": "crack",
    "screen_crack": "crack",
    "body_dent": "dent",
    "lines": "display_lines",
}


class DamageSegmentationDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ):
        config = MODEL_CONFIG["damage_segmentation"]
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold or config["confidence_threshold"]
        self.iou_threshold = iou_threshold or config["iou_threshold"]
        self.device = device or get_device()
        self.model = None
        self.input_size = config["input_size"]

        resolved_path = model_path or MODEL_PATHS.get("damage_segmentation_default")
        if YOLO and resolved_path and os.path.exists(resolved_path):
            self.load_model(resolved_path)

        logger.info(
            "DamageSegmentationDetector initialized | device={} | model_path={}",
            self.device,
            resolved_path or "not loaded",
        )

    def load_model(self, model_path: str) -> bool:
        if YOLO is None:
            logger.warning("Ultralytics not installed; segmentation unavailable.")
            return False
        try:
            self.model = YOLO(model_path)
            self.model_path = model_path
            logger.info("Loaded segmentation model: {}", model_path)
            return True
        except Exception as exc:
            logger.error("Failed to load segmentation model: {}", exc)
            self.model = None
            return False

    def detect(
        self, image: np.ndarray, roi: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        if self.model is None:
            logger.warning("No segmentation model loaded.")
            return []

        detect_image = roi if roi is not None else image
        detections: List[Dict[str, Any]] = []

        try:
            results = self.model(
                detect_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            for result in results:
                names = result.names
                if result.masks is not None:
                    num_masks = len(result.masks.data)
                    num_boxes = len(result.boxes) if result.boxes is not None else 0
                    for i, mask in enumerate(result.masks.data):
                        if i < num_boxes and result.boxes is not None:
                            class_id = int(result.boxes.cls[i].item())
                            confidence = float(result.boxes.conf[i].item())
                        else:
                            class_id = 0
                            confidence = 0.0
                        raw_name = names.get(class_id, str(class_id))
                        class_name = SEG_CLASS_NAME_MAP.get(raw_name, raw_name)
                        if class_name not in MVP_DAMAGE_CLASSES:
                            continue

                        mask_np = mask.cpu().numpy()
                        mask_resized = cv2.resize(mask_np, (detect_image.shape[1], detect_image.shape[0]))
                        binary_mask = (mask_resized > 0.5).astype(np.uint8)

                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        polygon = []
                        if contours:
                            largest = max(contours, key=cv2.contourArea)
                            polygon = largest.squeeze().tolist()
                            if isinstance(polygon, list) and len(polygon) > 0:
                                if not isinstance(polygon[0], list):
                                    polygon = [polygon]

                        bbox = [0.0, 0.0, 0.0, 0.0]
                        if result.boxes is not None:
                            bbox = [float(v) for v in result.boxes.xyxy[i].tolist()]

                        detections.append({
                            "class_name": class_name,
                            "class_id": MVP_DAMAGE_CLASSES.index(class_name) if class_name in MVP_DAMAGE_CLASSES else class_id,
                            "confidence": confidence,
                            "bbox": bbox,
                            "mask": binary_mask.tolist(),
                            "polygon": polygon,
                            "mask_area": int(np.sum(binary_mask)),
                            "segmentation": True,
                        })

                elif result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        class_id = int(box.cls.item())
                        confidence = float(box.conf.item())
                        raw_name = names.get(class_id, str(class_id))
                        class_name = SEG_CLASS_NAME_MAP.get(raw_name, raw_name)
                        if class_name not in MVP_DAMAGE_CLASSES:
                            continue
                        bbox = [float(v) for v in box.xyxy[0].tolist()]
                        detections.append({
                            "class_name": class_name,
                            "class_id": MVP_DAMAGE_CLASSES.index(class_name) if class_name in MVP_DAMAGE_CLASSES else class_id,
                            "confidence": confidence,
                            "bbox": bbox,
                            "segmentation": False,
                        })

        except Exception as exc:
            logger.error("Segmentation detection failed: {}", exc)

        return detections

    def get_primary_damage(self, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not detections:
            return None
        return max(detections, key=lambda d: d["confidence"])

    def is_damaged(self, detections: List[Dict[str, Any]]) -> bool:
        return bool(detections)
