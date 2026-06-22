"""
Appliance detector with confidence-based classification.

Key improvements:
- Returns top-3 predictions instead of forcing one class
- Confidence threshold below which we say "unsure"
- Confidence scores displayed prominently
- Heuristic fallback only when YOLO unavailable (not as primary)
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from configs.config import MODEL_CONFIG, MODEL_PATHS, MVP_APPLIANCE_CLASSES, get_device

COCO_APPLIANCE_MAP = {
    "cell phone": "phone",
    "tv": "television",
    "laptop": "laptop",
    "remote": "remote",
    "keyboard": "keyboard",
    "mouse": "mouse",
    "microwave": "microwave",
    "refrigerator": "refrigerator",
}

COCO_BACKED_MODELS = {"yolov8", "yolo11"}

APPLIANCE_CONFIDENCE_THRESHOLD = 0.35


def _is_coco_model(model_path: str) -> bool:
    return any(x in os.path.basename(model_path).lower() for x in COCO_BACKED_MODELS)


class ApplianceDetector:
    _instance: Optional["ApplianceDetector"] = None

    @classmethod
    def get_instance(cls, **kwargs) -> "ApplianceDetector":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ):
        config = MODEL_CONFIG["appliance_detector"]
        self.confidence_threshold = confidence_threshold or config["confidence_threshold"]
        self.iou_threshold = iou_threshold or config["iou_threshold"]
        self.device = device or get_device()
        self.classes = MVP_APPLIANCE_CLASSES
        self.model = None
        self.model_type = "unavailable"
        self.model_version = "unknown"
        self._load_time_ms = 0.0

        resolved_path = model_path or MODEL_PATHS.get("appliance_detector")
        if YOLO and resolved_path and os.path.exists(resolved_path):
            self.load_model(resolved_path, model_type="coco" if _is_coco_model(resolved_path) else "custom")
        else:
            fallback = MODEL_PATHS.get("appliance_detector_fallback")
            if YOLO and fallback and os.path.exists(fallback):
                self.load_model(fallback, model_type="coco")

        logger.info("ApplianceDetector initialized | device={} | model={} | version={} | load={}ms",
                     self.device, self.model_type, self.model_version, round(self._load_time_ms, 1))

    def load_model(self, model_path: str, model_type: str = "custom") -> bool:
        if YOLO is None:
            logger.warning("Ultralytics not installed.")
            return False
        if not os.path.exists(model_path):
            logger.warning("Weights not found: {}", model_path)
            return False
        try:
            t0 = time.perf_counter()
            self.model = YOLO(model_path)
            self._load_time_ms = (time.perf_counter() - t0) * 1000
            self.model_path = model_path
            self.model_type = model_type
            if hasattr(self.model, "ckpt") and hasattr(self.model.ckpt, "get"):
                self.model_version = str(self.model.ckpt.get("version", "unknown"))
            return True
        except Exception as exc:
            logger.error("Failed to load model: {}", exc)
            self.model = None
            return False

    def _crop_roi(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image.shape[1], x2); y2 = min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return image
        return image[y1:y2, x1:x2]

    def _format_detection(self, image: np.ndarray, class_name: str, class_id: int,
                          confidence: float, bbox: List[float]) -> Dict[str, Any]:
        return {
            "class_name": class_name,
            "class_id": class_id,
            "confidence": float(confidence),
            "bbox": [float(v) for v in bbox],
            "roi": self._crop_roi(image, bbox),
        }

    def detect_all(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Returns (all_detections, inference_time_ms).
        Each detection has class_name, confidence, bbox.
        Sorted by confidence descending.
        """
        if self.model is None or image is None or not isinstance(image, np.ndarray):
            return [], 0.0

        detections: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        try:
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - t0) * 1000

            for result in results:
                if result.boxes is None:
                    continue
                names = result.names
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    raw = names.get(cls_id, str(cls_id))
                    mapped = COCO_APPLIANCE_MAP.get(raw, raw)
                    if mapped not in self.classes:
                        continue
                    bbox = box.xyxy[0].tolist()
                    detections.append(self._format_detection(
                        image=image, class_name=mapped,
                        class_id=self.classes.index(mapped),
                        confidence=conf, bbox=bbox))
        except Exception as exc:
            logger.error("Appliance detection failed: {}", exc)
            inference_ms = (time.perf_counter() - t0) * 1000

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections, inference_ms

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        detections, _ = self.detect_all(image)
        return detections

    def detect_single(self, image: np.ndarray,
                      preferred_classes: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        all_dets, _ = self.detect_all(image)

        if not all_dets:
            logger.warning("YOLO found no appliances. Returning None (not forcing heuristic).")
            return None

        if preferred_classes:
            filtered = [d for d in all_dets if d["class_name"] in preferred_classes]
            if filtered:
                all_dets = filtered

        best = all_dets[0]

        if best["confidence"] < APPLIANCE_CONFIDENCE_THRESHOLD:
            logger.warning("Best detection {} has confidence {:.2f} (below {:.2f}). Returning None.",
                           best["class_name"], best["confidence"], APPLIANCE_CONFIDENCE_THRESHOLD)
            return None

        top3 = [(d["class_name"], round(d["confidence"], 3)) for d in all_dets[:3]]
        logger.info("Top predictions: {}", top3)

        best["top_predictions"] = top3
        return best

    def detect_brand(self, image: np.ndarray, image_path: Optional[str] = None) -> Optional[str]:
        try:
            if image_path and os.path.exists(image_path):
                from PIL import Image as PILImage
                from PIL.ExifTags import TAGS
                pil_img = PILImage.open(image_path)
                exif = pil_img.getexif()
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "Make":
                        brand = str(value).strip().lower()
                        known = {"apple": "Apple", "samsung": "Samsung", "lg": "LG",
                                 "sony": "Sony", "dell": "Dell", "hp": "HP",
                                 "lenovo": "Lenovo", "google": "Google", "oneplus": "OnePlus",
                                 "xiaomi": "Xiaomi", "huawei": "Huawei"}
                        for key, name in known.items():
                            if key in brand:
                                return name
                        return brand.title()
        except Exception:
            pass
        try:
            from easyocr import Reader
            reader = Reader(["en"], gpu=False, verbose=False)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            results = reader.readtext(gray)
            known_brands = {"apple": "Apple", "samsung": "Samsung", "lg": "LG", "sony": "Sony",
                            "dell": "Dell", "hp": "HP", "lenovo": "Lenovo", "google": "Google",
                            "oneplus": "OnePlus", "xiaomi": "Xiaomi", "huawei": "Huawei", "nokia": "Nokia"}
            for _, text, conf in results:
                if conf > 0.5:
                    t = text.lower().strip()
                    for key, name in known_brands.items():
                        if key in t:
                            return name
        except Exception:
            pass
        return None

    @staticmethod
    def create_dataset_yaml(train_path: str, val_path: str, test_path: str, output_path: str) -> str:
        yaml_content = f"""
train: {train_path}
val: {val_path}
test: {test_path}
nc: {len(MVP_APPLIANCE_CLASSES)}
names: {MVP_APPLIANCE_CLASSES}
"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(yaml_content.strip() + "\n")
        return output_path
