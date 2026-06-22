"""
Train the MVP damage detector for phone, television, and laptop.

Target classes:
- crack
- dent
- display_lines
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from loguru import logger
import os
os.environ['YOLO_VERBOSE'] = 'False'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DAMAGE_CLASSES, TRAINING_CONFIG, PATHS, get_device

from ultralytics import YOLO


APPLIANCE_DAMAGE_CONFIGS = {
    "phone": {
        "classes": ["crack", "dent", "display_lines"],
        "description": "Phone damage detection"
    },
    "television": {
        "classes": ["crack", "dent", "display_lines"],
        "description": "Television damage detection"
    },
    "laptop": {
        "classes": ["crack", "dent", "display_lines"],
        "description": "Laptop damage detection"
    }
}


def create_damage_dataset_yaml(
    dataset_path: str,
    appliance: str,
    output_path: str = None
) -> str:
    """
    Create dataset YAML for damage detector training.

    Args:
        dataset_path: Path to dataset root directory
        appliance: Appliance type
        output_path: Path to save YAML

    Returns:
        Path to created YAML
    """
    if appliance not in APPLIANCE_DAMAGE_CONFIGS:
        raise ValueError(f"Unknown appliance: {appliance}. Choose from: {list(APPLIANCE_DAMAGE_CONFIGS.keys())}")

    classes = APPLIANCE_DAMAGE_CONFIGS[appliance]["classes"]

    train_path = os.path.join(dataset_path, "train", "images")
    val_path = os.path.join(dataset_path, "val", "images")
    test_path = os.path.join(dataset_path, "test", "images")

    yaml_content = f"""
# {appliance.title()} Damage Detection Dataset
# {APPLIANCE_DAMAGE_CONFIGS[appliance]['description']}

train: {train_path}
val: {val_path}
test: {test_path}

nc: {len(classes)}
names: {classes}
"""

    if output_path is None:
        output_path = f"configs/damage_{appliance}_dataset.yaml"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(yaml_content.strip())

    logger.info(f"Created damage dataset YAML for {appliance} at {output_path}")
    return output_path


def train_damage_detector(
    appliance: str,
    data_yaml: str,
    model_size: str = "yolo11s",
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    device: str = None,
    resume: bool = False,
    weights: str = None,
    model_name: str = None
) -> str:
    """
    Train damage detector for specific appliance (YOLO11s / YOLOv8).

    Args:
        appliance: Appliance type
        data_yaml: Path to dataset YAML
        model_size: YOLOv8 model size
        epochs: Number of training epochs
        batch_size: Batch size
        image_size: Input image size
        device: Device to use
        resume: Resume from checkpoint
        weights: Path to pretrained weights
        model_name: Custom model name

    Returns:
        Path to best trained model
    """
    from ultralytics import YOLO

    device = device or get_device()
    model_name = model_name or f"damage_{appliance}"

    logger.info(f"Starting {appliance} damage detector training on {device}")

    if weights:
        model = YOLO(weights)
    else:
        model = YOLO(f'{model_size}.pt')

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        project=os.path.join(PATHS["models"], "damage_detector"),
        name=model_name,
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


def validate_damage_detector(
    model_path: str,
    data_yaml: str,
    device: str = None
) -> dict:
    """
    Validate trained damage detector.

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

    logger.info("Validating damage detector...")
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


def train_phone_crack_detector(
    dataset_path: str,
    epochs: int = 50,
    batch_size: int = 8,
    image_size: int = 640,
    device: str = None
) -> str:
    """
    Train phone crack detector using existing dataset.

    This uses the cracked screen dataset to train a dedicated
    phone screen crack detector.

    Args:
        dataset_path: Path to phone crack dataset
        epochs: Number of epochs
        batch_size: Batch size
        image_size: Image size
        device: Device to use

    Returns:
        Path to trained model
    """
    from ultralytics import YOLO

    device = device or get_device()
    logger.info(f"Training phone crack detector using dataset: {dataset_path}")

    # Use your existing cracked screen dataset
    # It has 1 class: 'cracked'
    phone_classes = ["screen_crack", "display_lines", "camera_crack", "body_damage", "normal"]

    train_path = os.path.join(dataset_path, "train", "images")
    val_path = os.path.join(dataset_path, "val", "images")
    test_path = os.path.join(dataset_path, "test", "images")

    # Create YAML for phone damage
    yaml_content = f"""
# Phone Damage Detection Dataset
# Based on cracked screen dataset

train: {train_path}
val: {val_path}
test: {test_path}

nc: {len(phone_classes)}
names: {phone_classes}
"""

    yaml_path = os.path.join(PATHS["models"], "damage_detector", "phone_dataset.yaml")
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())

    logger.info(f"Created phone damage dataset YAML: {yaml_path}")

    # Load and train
    model = YOLO('yolov8n.pt')

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        device=device,
        project=os.path.join(PATHS["models"], "damage_detector"),
        name="phone_crack",
        exist_ok=True,
        verbose=True,
        patience=20,
        save_period=10
    )

    best_model_path = results.save_dir / "weights" / "best.pt"
    logger.info(f"Phone crack detector training complete: {best_model_path}")

    return str(best_model_path)


def export_damage_detector(
    model_path: str,
    format: str = "onnx",
    device: str = None
) -> str:
    """Export trained damage detector"""
    from ultralytics import YOLO

    model = YOLO(model_path)
    device = device or "cpu"

    logger.info(f"Exporting model to {format}...")
    export_path = model.export(format=format, device=device)

    return export_path


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train Damage Detector")

    parser.add_argument(
        "--appliance",
        type=str,
        required=True,
        choices=list(APPLIANCE_DAMAGE_CONFIGS.keys()) + ["phone_crack"],
        help="Appliance type for damage detection"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to damage dataset directory"
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
        help="Device to use"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to pretrained weights"
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
        help="Export format"
    )

    args = parser.parse_args()

    # Setup logging to console only (sandbox restrictions prevent file logging)
    logger.add(sys.stderr, level="INFO")

    # Special handling for phone_crack (uses existing dataset)
    if args.appliance == "phone_crack":
        best_model = train_phone_crack_detector(
            dataset_path=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device
        )
    else:
        # Create dataset YAML
        data_yaml = create_damage_dataset_yaml(args.dataset, args.appliance)

        # Train model
        best_model = train_damage_detector(
            appliance=args.appliance,
            data_yaml=data_yaml,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            device=args.device,
            weights=args.weights
        )

    # Validate if requested
    if args.validate and args.appliance != "phone_crack":
        data_yaml = create_damage_dataset_yaml(args.dataset, args.appliance)
        metrics = validate_damage_detector(best_model, data_yaml, args.device)
        logger.info(f"Final metrics: {metrics}")

    # Export if requested
    if args.export:
        export_path = export_damage_detector(best_model, args.export, args.device)
        logger.info(f"Model exported to: {export_path}")

    logger.info(f"{args.appliance} damage detector training complete!")


if __name__ == "__main__":
    main()
