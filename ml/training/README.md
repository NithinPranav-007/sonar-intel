# Training & Validation Guide

## 1. Quick Training Run
```bash
python ml/training/train.py --epochs 50 --batch 2 --imgsz 640 --device 0
```

## 2. Evaluation
```bash
python ml/training/evaluate.py --weights runs/train_sonar/yolov8n_sonar_mvp/weights/best.pt --split val
```

## 3. Site-Separated Data Split
Sonar waterfalls have high along-track spatial autocorrelation. Do not use standard random K-Fold splits.
Assign entire survey tracks to `train`, `val`, and `test` directories to ensure realistic generalization metrics.
