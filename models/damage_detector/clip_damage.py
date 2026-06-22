from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from loguru import logger


class CLIPTVDamageDetector:
    DAMAGE_CLASSES = ["crack", "dent", "display_lines"]

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        confidence_threshold: float = 0.3,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor = None
        self._available = False
        self._load_error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> bool:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
            elif torch.backends.mps.is_available():
                self.model = self.model.to("mps")
            self._available = True
            logger.info("CLIP TV Damage Detector loaded: {}", self.model_name)
            return True
        except ImportError:
            self._load_error = "transformers / torch not installed"
            logger.warning("CLIP not available: {}", self._load_error)
        except Exception as exc:
            self._load_error = str(exc)
            logger.error("CLIP load failed: {}", exc)
        return False

    def _classify_with_clip(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if not self._available or self.model is None or self.processor is None:
            return self._fallback_heuristic(image)
        try:
            import torch
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 20:
                return self._fallback_heuristic(image)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            text_labels = [
                "a crack on a television screen",
                "a dent on a television frame",
                "display lines on a television screen",
                "a normal undamaged television screen",
                "a television showing static or noise",
            ]
            inputs = self.processor(text=text_labels, images=rgb, return_tensors="pt", padding=True)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            results = []
            for i, cls_name in enumerate(self.DAMAGE_CLASSES):
                conf = float(probs[i])
                if conf >= self.confidence_threshold:
                    h, w = image.shape[:2]
                    bbox = [w * 0.1, h * 0.1, w * 0.9, h * 0.9]
                    results.append({
                        "class_name": cls_name,
                        "class_id": i,
                        "confidence": conf,
                        "bbox": bbox,
                        "source": "clip",
                    })
            return results
        except Exception as exc:
            logger.error("CLIP classification failed: {}", exc)
            return self._fallback_heuristic(image)

    def _fallback_heuristic(self, image: np.ndarray) -> List[Dict[str, Any]]:
        results = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 120, minLineLength=int(w * 0.15), maxLineGap=5)
        if lines is not None and len(lines) >= 4:
            xs = []
            ys = []
            for line in lines[:20]:
                x1, y1, x2, y2 = line[0]
                if abs(x1 - x2) < 6 or abs(y1 - y2) < 6:
                    xs.extend([x1, x2])
                    ys.extend([y1, y2])
            if len(xs) >= 8:
                bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                results.append({
                    "class_name": "display_lines",
                    "class_id": 2,
                    "confidence": 0.4,
                    "bbox": bbox,
                    "source": "heuristic",
                })
        return results

    def detect(self, image: np.ndarray, roi: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        detect_image = roi if roi is not None else image
        return self._classify_with_clip(detect_image)
