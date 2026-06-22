"""
Train a YOLOv8 missing-part detector for the MVP.

Recommended dataset labels:
- phone_camera_missing
- phone_buttons_missing
- television_stand_missing
- television_bezel_missing
- laptop_keys_missing
- laptop_hinge_cover_missing
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

from configs.config import PATHS, TRAINING_CONFIG, get_device

MISSING_PART_DATASET_CLASSES = [
    "phone_camera_missing",
    "phone_buttons_missing",
    "television_stand_missing",
    "television_bezel_missing",
    "laptop_keys_missing",
    "laptop_hinge_cover_missing",
]


def create_dataset_yaml(dataset_path: str, output_path: str = "configs/missing_part_dataset.yaml") -> str:
    yaml_content = f"""
train: {os.path.join(dataset_path, 'train', 'images')}
val: {os.path.join(dataset_path, 'val', 'images')}
test: {os.path.join(dataset_path, 'test', 'images')}

nc: {len(MISSING_PART_DATASET_CLASSES)}
names: {MISSING_PART_DATASET_CLASSES}
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(yaml_content.strip() + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MVP missing-part detector")
    parser.add_argument("--dataset", required=True, help="Dataset root with train/val/test folders")
    parser.add_argument("--weights", default="yolov8n.pt", help="Initial weights")
    parser.add_argument("--epochs", type=int, default=TRAINING_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=TRAINING_CONFIG["batch_size"])
    parser.add_argument("--image-size", type=int, default=TRAINING_CONFIG["image_size"])
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()

    data_yaml = create_dataset_yaml(args.dataset)
    model = YOLO(args.weights)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.image_size,
        device=args.device or get_device(),
        project=PATHS["models"],
        name="missing_part_detector",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
