# DRISHTI YOLOv8s Pretrained Model Baseline Specification

**Document:** `DRISHTI_BASELINE.md`  
**Model Name:** `DRISHTI-YOLOv8s`  
**Model Version:** `baseline-v1`  
**Status:** Pretrained Baseline (Frozen — Not Fine-Tuned)  
**Date Integrated:** September 2, 2026

---

## 1. Model Provenance & Identity

| Attribute | Authoritative Value |
| :--- | :--- |
| **Model Architecture** | Ultralytics YOLOv8s (DetectionModel) |
| **Model Weights File** | `ml/models/dristri/best_detector.pt` |
| **File Size** | 22,513,507 bytes (~22.5 MB) |
| **Cryptographic SHA256** | `2f55eec5d8fe6b4737706392e259c02660a8542cddbcbd603f96d606c54cb927` |
| **Primary Reference Source** | [HuggingFace: rehan9599/drishti-detector](https://huggingface.co/rehan9599/drishti-detector) |
| **Input Spatial Dimension** | $640 \times 640$ pixels (3 channels BGR) |
| **Operating Framework** | PyTorch 2.6.0 + CUDA 12.6 / CPU fallback |

---

## 2. Effective Model Classes & Product Filtering Policy

The model's classification head exposes 5 raw class indices:

```python
{
    0: "crab_pot",
    1: "submarine_pipeline",
    2: "shipwreck",
    3: "ghost_net",
    4: "mine_cylinder"
}
```

### Product Policy on `crab_pot`
The upstream DRISHTI documentation notes that the `crab_pot` class exhibits poor precision and generalization on real-world test sets due to insufficient training diversity. 
In accordance with SONAR-INTEL's data honesty and hydrographic operational integrity standards:
- **Raw Inference Preserved**: The detector preserves all raw predictions (`raw_class_id=0`, `raw_class_name="crab_pot"`, `raw_confidence`) with `is_filtered=True` and `filter_reason="Filtered per product policy"`.
- **Filtered Downstream**: `crab_pot` detections are **never** transformed into production `Contact` objects, preventing clutter from entering the operator triage queue.
- **Production Eligible Classes**:
  1. `submarine_pipeline`
  2. `shipwreck`
  3. `ghost_net`
  4. `mine_cylinder`

---

## 3. Acoustic Preprocessing Reproduction (`drishti-prep-v1`)

The pretrained detector was trained on side-scan sonar tiles preprocessed with speckle suppression and adaptive equalization. To guarantee model fidelity during inference, the `drishti-prep-v1` preprocessing pipeline reproduces this workflow:

```
Raw Sonar Image
      ↓
Input Validation & Non-Destructive Copy
      ↓
Grayscale Conversion & Dynamic Range Stretch (1%–99% Percentile)
      ↓
Vectorized Lee Speckle Filter (5×5 window, noise_var=0.04)
      ↓
CLAHE Contrast Enhancement (clipLimit=2.0, tileGridSize=8×8)
      ↓
3-Channel BGR Formatting for YOLOv8s
```

*Note: Raw sensor swaths are never overwritten or mutated; preprocessed tensors exist solely in memory for inference.*

---

## 4. Scientific Honesty & Evaluation Caveats

1. **Synthetic Training Bias**: The upstream DRISHTI dataset incorporates synthetic simulations for certain marine debris targets (notably ghost nets). Reported synthetic validation numbers must **never** be cited as true operational field accuracy.
2. **AI Candidate Status**: Every generated contact starts with `review_status = "AI_CANDIDATE"`. No detection is ever automatically marked as confirmed debris without human review.
3. **Confidence Decoupling**: Raw YOLO confidence (`model_score`) is tracked separately from calibrated confidence, acoustic shadow verification scores, and operational priority.
4. **Geolocation Provenance**: Geographic coordinates ($\text{lat}, \text{lon}$) are calculated only when a valid towfish navigation log is synchronized with the survey line. Coordinates are never synthesized.

---

## 5. Failure Analysis Taxonomy

During baseline evaluation and field triage, operator rejections and detection misses should be categorized using the following standardized taxonomy:

| Code | Failure Mode | Description |
| :--- | :--- | :--- |
| `ROCK` | Natural Geological Obstacle | Boulder or rock outcrop falsely classified as cylinder/wreck. |
| `SEABED_TEXTURE` | Sand Dunes / Ripples | High-relief acoustic backscatter ridges triggering pipeline proposals. |
| `ACOUSTIC_SHADOW` | Shadow-Only Void | Deep bathymetric trench or depression lacking reflective highlight. |
| `SPECKLE` | High Acoustic Noise | Multiplicative acoustic noise spikes evading the Lee filter. |
| `STRIPE_ARTIFACT` | Sensor Scanline Stripe | Transducer beam boundary or surface reflection banding. |
| `DROPOUT` | Missing Ping Return | Acoustic signal loss in the water column or nadir zone. |
| `LOW_SNR` | Low Dynamic Range | Swaths with weak acoustic contrast. |
| `PARTIAL_TARGET` | Truncated Boundary | Target clipped by swath or tile edge. |
| `DUPLICATE_DETECTION`| NMS Border Leak | Multiple overlapping detections across tile strides. |
| `WRONG_CLASS` | Class Misattribution | Shipwreck misclassified as pipeline or mine. |
| `LOCALIZATION_ERROR`| Navigation Offset | Layback, crab angle, or towfish navigation drift. |
