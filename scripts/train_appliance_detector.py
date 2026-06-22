"""
Train the MVP appliance detector for:
- phone
- television
- laptop
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger
os.environ['YOLO_VERBOSE'] = 'False'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import MVP_APPLIANCE_CLASSES, TRAINING_CONFIG, PATHS, get_device

from ultralytics import YOLO


def create_appliance_dataset_yaml(
    dataset_path: str,
    output_path: str = "configs/appliance_dataset.yaml"
) -> str:
    """
    Create dataset YAML for appliance detector training.

    Args:
        dataset_path: Path to dataset root directory
        output_path: Path to save YAML

    Returns:
        Path to created YAML
    """
    train_path = os.path.join(dataset_path, "train", "images")
    val_path = os.path.join(dataset_path, "val", "images")
    test_path = os.path.join(dataset_path, "test", "images")

    yaml_content = f"""
# Appliance Detection Dataset Configuration
train: {train_path}
val: {val_path}
test: {test_path}

nc: {len(MVP_APPLIANCE_CLASSES)}
names: {MVP_APPLIANCE_CLASSES}
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(yaml_content.strip())

    logger.info(f"Created appliance dataset YAML at {output_path}")
    return output_path


def train_appliance_detector(
    data_yaml: str,
    model_size: str = "yolo11s",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    device: str = None,
    resume: bool = False,
    weights: str = None
) -> str:
    """
    Train appliance detector (YOLO11s / YOLOv8).

    Args:
        data_yaml: Path to dataset YAML
        model_size: Model size (yolo11n/s/m/l/x or yolov8n/s/m/l/x)
        epochs: Number of training epochs
        batch_size: Batch size
        image_size: Input image size
        device: Device to use (cuda, mps, cpu)
        resume: Resume from last checkpoint
        weights: Path to pretrained weights

    Returns:
        Path to best trained model
    """
    from ultralytics import YOLO

    device = device or get_device()
    logger.info(f"Starting appliance detector training on {device}")

    # Load model
    if weights:
        model = YOLO(weights)
    else:
        model = YOLO(f'{model_size}.pt')

    # Training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        project=PATHS["appliance_detector"],
        name="appliance_detector",
        exist_ok=True,
        resume=resume,
        verbose=True,
        patience=TRAINING_CONFIG["patience"],
        save_period=TRAINING_CONFIG["save_period"],
        optimizer=TRAINING_CONFIG["optimizer"],
        lr0=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        warmup_epochs=TRAINING_CONFIG["warmup_epochs"]
    )

    best_model_path = results.save_dir / "weights" / "best.pt"
    logger.info(f"Training complete. Best model: {best_model_path}")

    return str(best_model_path)


def validate_appliance_detector(
    model_path: str,
    data_yaml: str,
    device: str = None
) -> dict:
    """
    Validate trained appliance detector.

    Args:
        model_path: Path to trained model
        data_yaml: Path to dataset YAML
        device: Device to use

    Returns:
        Validation metrics
    """
    from ultralytics import YOLO

    device = device or get_device()
    model = YOLO(model_path)

    logger.info("Validating appliance detector...")
    metrics = model.val(
        data=data_yaml,
        device=device,
        verbose=True
    )

    logger.info(f"mAP50: {metrics.box.map50:.4f}")
    logger.info(f"mAP50-95: {metrics.box.map:.4f}")

    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr)
    }


def export_appliance_detector(
    model_path: str,
    format: str = "onnx",
    device: str = None
) -> str:
    """
    Export trained model to different formats.

    Args:
        model_path: Path to trained model
        format: Export format (onnx, torchscript, tflite, etc.)
        device: Device type (cpu, gpu)

    Returns:
        Path to exported model
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    device = device or "cpu"

    logger.info(f"Exporting model to {format}...")
    export_path = model.export(format=format, device=device)

    logger.info(f"Exported to: {export_path}")
    return export_path


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train Appliance Detector")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to appliance dataset directory"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="yolo11s",
        choices=["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="Model size (YOLO11s recommended)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pretrained weights"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after training"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        choices=["onnx", "torchscript", "tflite", "coreml"],
        help="Export format after training"
    )

    args = parser.parse_args()

    # Setup logging to console only
    logger.add(sys.stderr, level="INFO")

    # Create dataset YAML
    data_yaml = create_appliance_dataset_yaml(args.dataset)

    # Train model
    best_model = train_appliance_detector(
        data_yaml=data_yaml,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
        resume=args.resume,
        weights=args.weights
    )

    # Validate if requested
    if args.validate:
        metrics = validate_appliance_detector(best_model, data_yaml, args.device)
        logger.info(f"Final metrics: {metrics}")

    # Export if requested
    if args.export:
        export_path = export_appliance_detector(best_model, args.export, args.device)
        logger.info(f"Model exported to: {export_path}")

    logger.info("Appliance detector training complete!")


if __name__ == "__main__":
    main()
