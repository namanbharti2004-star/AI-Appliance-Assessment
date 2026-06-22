"""
Utility Functions for the AI Appliance Assessment Platform
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from loguru import logger
try:
    import torch
except ImportError:
    torch = None
import cv2
import numpy as np
from PIL import Image


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """
    Configure logging for the application

    Args:
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()
    logger.add(
        os.path.join(log_dir, "app_{time:YYYY-MM-DD}.log"),
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        rotation="500 MB",
        retention="7 days",
        compression="zip"
    )
    logger.add(lambda msg: print(msg, end=""), level=level)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        config_path: Path to save config file

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        return False


def get_device() -> str:
    """
    Get the best available device for computation

    Returns:
        'cuda' if GPU available, 'mps' if Apple Silicon, 'cpu' otherwise
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def create_directory_structure(base_path: str) -> None:
    """
    Create the standard project directory structure

    Args:
        base_path: Base directory for the project
    """
    directories = [
        "datasets/appliance_detector",
        "datasets/damage_detector/phone",
        "datasets/damage_detector/television",
        "datasets/damage_detector/laptop",
        "datasets/damage_detector/tablet",
        "datasets/damage_detector/monitor",
        "datasets/damage_detector/refrigerator",
        "datasets/damage_detector/washing_machine",
        "datasets/damage_detector/air_conditioner",
        "datasets/damage_detector/microwave",
        "models/appliance_detector",
        "models/damage_detector",
        "fraud_detection",
        "risk_engine",
        "report_engine",
        "dashboard",
        "logs",
        "temp",
        "output",
        "tests"
    ]

    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
        logger.debug(f"Created directory: {full_path}")


def load_json(json_path: str) -> Dict[str, Any]:
    """
    Load JSON file

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary containing JSON data
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {json_path}: {e}")
        return {}


def save_json(data: Dict[str, Any], json_path: str, indent: int = 2) -> bool:
    """
    Save dictionary to JSON file

    Args:
        data: Dictionary to save
        json_path: Path to save JSON file
        indent: JSON indentation level

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=indent)
        logger.debug(f"Saved JSON to {json_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON to {json_path}: {e}")
        return False


def read_image(image_path: str) -> Optional[np.ndarray]:
    import unicodedata
    import numpy as np
    import re

    def _sanitize_path(p: str) -> str:
        p = unicodedata.normalize("NFC", p).strip()
        p = re.sub(r"[\u2000-\u200A\u202F\u205F\u3000]", " ", p)
        p = re.sub(r"\s+", " ", p)
        return p

    paths_to_try = [_sanitize_path(image_path), image_path]
    for attempt_path in dict.fromkeys(paths_to_try):
        try:
            ext = os.path.splitext(attempt_path)[1].lower()
            if ext in (".heic", ".heif"):
                try:
                    from PIL import ImageOps
                    from pillow_heif import open_heif
                    heif_file = open_heif(attempt_path)
                    pil_image = heif_file.to_pillow()
                    pil_image = ImageOps.exif_transpose(pil_image)
                except Exception:
                    pil_image = Image.open(attempt_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                return image
            with open(attempt_path, "rb") as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is not None:
                return image
        except Exception:
            continue
    logger.error(f"Failed to read image from {image_path}")
    return None


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save image to file path

    Args:
        image: Image as numpy array (BGR format)
        output_path: Path to save image

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        logger.debug(f"Saved image to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {e}")
        return False


def resize_image(image: np.ndarray, target_size: int = 640, keep_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target size

    Args:
        image: Input image
        target_size: Target size for longest edge
        keep_aspect: Whether to maintain aspect ratio

    Returns:
        Resized image
    """
    h, w = image.shape[:2]

    if keep_aspect:
        if h > w:
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            new_w = target_size
            new_h = int(h * (target_size / w))
    else:
        new_h = new_w = target_size

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range

    Args:
        image: Input image (0-255)

    Returns:
        Normalized image (0-1)
    """
    return image.astype(np.float32) / 255.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [0, 1] to [0, 255]

    Args:
        image: Normalized image

    Returns:
        Denormalized image
    """
    return (image * 255).astype(np.uint8)


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]

    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def non_max_suppression(
    boxes: List[List[float]],
    scores: List[float],
    iou_threshold: float = 0.45
) -> List[int]:
    """
    Apply Non-Maximum Suppression to bounding boxes

    Args:
        boxes: List of boxes [x1, y1, x2, y2]
        scores: List of confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []

    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        current_box = boxes[current]
        rest_boxes = [boxes[i] for i in indices[1:]]

        ious = [calculate_iou(current_box, box) for box in rest_boxes]

        indices = indices[1:][np.array(ious) <= iou_threshold]

    return keep


def get_appliance_from_class_id(class_id: int, classes: List[str]) -> Optional[str]:
    """
    Get appliance name from class ID

    Args:
        class_id: Numeric class ID
        classes: List of class names

    Returns:
        Class name or None
    """
    if 0 <= class_id < len(classes):
        return classes[class_id]
    return None


def get_class_id_from_appliance(appliance: str, classes: List[str]) -> Optional[int]:
    """
    Get class ID from appliance name

    Args:
        appliance: Appliance name
        classes: List of class names

    Returns:
        Class ID or None
    """
    try:
        return classes.index(appliance)
    except ValueError:
        return None


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def validate_image_file(file_path: str, allowed_formats: List[str] = None) -> bool:
    """
    Validate if file is a valid image

    Args:
        file_path: Path to image file
        allowed_formats: List of allowed extensions

    Returns:
        True if valid image, False otherwise
    """
    if allowed_formats is None:
        allowed_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic', '.heif']

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed_formats:
        return False

    if not os.path.exists(file_path):
        return False

    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except Exception:
        return False


def validate_video_file(file_path: str, allowed_formats: List[str] = None) -> bool:
    """
    Validate if file is a valid video

    Args:
        file_path: Path to video file
        allowed_formats: List of allowed extensions

    Returns:
        True if valid video, False otherwise
    """
    if allowed_formats is None:
        allowed_formats = ['.mp4', '.avi', '.mov', '.mkv']

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed_formats:
        return False

    if not os.path.exists(file_path):
        return False

    try:
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        return ret
    except Exception:
        return False
