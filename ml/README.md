# SONAR-INTEL ML Subsystem

This folder contains the complete ML training, preprocessing, inference, and acoustic-context scoring pipeline.

## Structure
- `preprocessing/`: Sonar normalization, water column blanking, CLAHE, quality scoring, and tiling.
- `inference/`: YOLO candidate detector, acoustic context (shadow & contrast), priority scoring, and NMS.
- `training/`: Training and evaluation scripts for YOLOv8n, site-separated validation policy.
- `models/`: Checkpoints and weights (`yolov8n.pt`, `best.pt`).

## Primary Developer Checklist
1. **Low VRAM Operations (RTX 3050 4GB)**:
   - Batch size: 1 or 2
   - Image size: 640x640
   - AMP enabled: `--amp`
2. **Spatial Split Discipline**:
   - Ensure consecutive tiles from the same transect are not split between `train` and `val`.
3. **Domain Honesty**:
   - Outputs are labeled `artificial_anomaly` / `AI Candidate` rather than "confirmed ghost net".
