"""
Ultralytics YOLOv8n Training Script for Side-Scan Sonar Artificial Anomaly Detection.

Hardware Target:
- NVIDIA GeForce RTX 3050 (4 GB VRAM) / Standard CUDA GPU / CPU fallback
- Default low-VRAM settings: batch=2 (or 1), imgsz=640, AMP=True, workers=2

Spatial Split Requirement:
- Site-separated dataset split: adjacent sonar waterfall frames from the same survey
  site must NEVER be randomly partitioned across train and validation folds.
"""

import os
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8n on Sonar Anomaly Dataset")
    parser.add_argument(
        "--data",
        type=str,
        default=os.environ.get("SONAR_DATASET_YAML", "ml/training/dataset.yaml"),
        help="Path to dataset.yaml"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("SONAR_BASE_MODEL", "yolov8n.pt"),
        help="Base model checkpoint"
    )
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("SONAR_EPOCHS", "50")))
    parser.add_argument("--batch", type=int, default=int(os.environ.get("SONAR_BATCH_SIZE", "2")),
                        help="Batch size (recommended 1-2 for 4GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=int(os.environ.get("SONAR_IMGSZ", "640")))
    parser.add_argument("--device", type=str, default=os.environ.get("SONAR_TRAIN_DEVICE", "0"),
                        help="CUDA device index e.g. 0, or 'cpu'")
    parser.add_argument("--amp", action="store_true", default=True, help="Use Automatic Mixed Precision")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience epochs")
    parser.add_argument("--project", type=str, default="runs/train_sonar", help="Output project dir")
    parser.add_argument("--name", type=str, default="yolov8n_sonar_mvp", help="Experiment name")
    return parser.parse_args()


def train():
    args = parse_args()
    print("==================================================")
    print("SONAR-INTEL: YOLOv8n Training Pipeline")
    print(f"Dataset YAML : {args.data}")
    print(f"Base Model   : {args.model}")
    print(f"Batch Size   : {args.batch} (Optimized for low VRAM 4GB RTX 3050)")
    print(f"Image Size   : {args.imgsz}")
    print(f"Epochs       : {args.epochs}")
    print(f"Device       : {args.device}")
    print(f"Mixed Prec.  : {args.amp}")
    print("==================================================")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"[WARNING] Dataset file '{args.data}' not found. Verify path.")

    # Load model
    model = YOLO(args.model)

    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            amp=args.amp,
            patience=args.patience,
            project=args.project,
            name=args.name,
            save=True,
            workers=2,
            verbose=True
        )
        print("\n[SUCCESS] Training completed successfully.")
        print(f"Best model weights saved at: {os.path.join(args.project, args.name, 'weights', 'best.pt')}")
        return results
    except RuntimeError as oom:
        if "out of memory" in str(oom).lower():
            print("\n[CUDA OUT OF MEMORY] Low VRAM detected on RTX 3050 (4GB).")
            print("Actionable recovery steps:")
            print("1. Reduce batch size to 1: --batch 1")
            print("2. Reduce image size to 512: --imgsz 512")
            print("3. Ensure workers=1 or 0: --workers 1")
        raise oom


if __name__ == "__main__":
    train()
