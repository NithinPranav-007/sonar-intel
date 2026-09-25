# SONAR-INTEL LearnLoop: Active Learning & Model Lifecycle Management Architecture

**Document Version:** 1.0.0  
**Author:** Person 2 — ML / Model Architecture, Verification & Lifecycle Lead  
**Status:** IMPLEMENTED & AUDITED  
**Audience:** Technical Jury, Machine Learning Operations (MLOps) Engineers, Hydrographic Systems Architects  

---

## 1. Executive Summary & Architectural Overview

**SONAR-INTEL LearnLoop** is a production-grade backend active learning and model lifecycle management subsystem. It bridges real-time human operator verification in operational side-scan sonar (SSS) surveys with automated supervised retraining, multi-dimensional candidate evaluation, safety gating, model version registry, and zero-downtime rollback capabilities.

```
+--------------------------------------------------------------------------------------------------+
|                                    SONAR-INTEL LearnLoop Architecture                           |
+--------------------------------------------------------------------------------------------------+

   SSS Survey Data & Telemetry
              |
              v
     [YOLOv8s Detector]  ----->  [SW-Net Structural Verifier]  ----->  [Multi-Modal Evidence Fusion]
                                                                                   |
                                                                                   v
                                                                        [Contact Package (v1.0)]
                                                                                   |
                                                                                   v
+--------------------------------------------------------------------------------------------------+
| 1. HUMAN OPERATOR VERIFICATION                                                                   |
|    - ACCEPTED      (True positive confirmation)                                                  |
|    - REJECTED      (False positive marked)                                                       |
|    - CORRECTED     (Class/bounding-box rectification)                                            |
|    - UNCERTAIN     (Ambiguous acoustic contact flagged for survey revisit)                       |
+--------------------------------------------------------------------------------------------------+
              |
              v
+--------------------------------------------------------------------------------------------------+
| 2. VERIFIED FEEDBACK STORAGE  (ml/learnloop/feedback_store.py)                                   |
|    - Append-only JSONL persistent store: data/feedback/verified_feedback.jsonl                   |
|    - Preserves contact_id, model_version, evidence breakdown, ROI references, operator notes    |
+--------------------------------------------------------------------------------------------------+
              |
              v
+--------------------------------------------------------------------------------------------------+
| 3. MULTI-GATE RETRAINING TRIGGER  (ml/learnloop/trigger.py)                                      |
|    - Gate A: min_verified_samples (e.g. 500 total)                                               |
|    - Gate B: min_new_samples_since_last_training (e.g. 250 new)                                  |
|    - Gate C: min_samples_per_class (e.g. >=25 mine_like_contact, >=25 shipwreck, etc.)          |
|    - Gate D: max_class_imbalance_ratio (e.g. max/min <= 10.0)                                    |
+--------------------------------------------------------------------------------------------------+
              | (When all gates pass)
              v
+--------------------------------------------------------------------------------------------------+
| 4. IMMUTABLE DATASET VERSIONING  (ml/learnloop/dataset_versioner.py)                             |
|    - Parent data/dataset_v1.0/ remains strictly READ-ONLY and IMMUTABLE                          |
|    - Builds versioned lineage manifest: data/versions/dataset_v1.1/manifest.json                |
+--------------------------------------------------------------------------------------------------+
              |
              v
+--------------------------------------------------------------------------------------------------+
| 5. CANDIDATE TRAINING ORCHESTRATION  (ml/learnloop/training_runner.py)                           |
|    - Inherits active Champion checkpoint (e.g. YOLO11N-KD-v3)                                    |
|    - Creates isolated candidate weights: outputs/models/candidates/candidate_v4.pt               |
|    - Never overwrites Champion weights                                                           |
+--------------------------------------------------------------------------------------------------+
              |
              v
+--------------------------------------------------------------------------------------------------+
| 6. CHAMPION / CHALLENGER BENCHMARKING & GATING  (ml/learnloop/evaluator.py)                      |
|    - Evaluates mAP50, mAP50-95, precision, recall, F1, per-class metrics, FPR, latency, size    |
|    - Gate 1: Recall Non-Degradation (Drop <= 0.01)                                               |
|    - Gate 2: Rare-Class Recall Preservation (Per-class drop <= 0.02)                             |
|    - Gate 3: F1 Score Maintenance / Improvement (Diff >= 0.00)                                   |
|    - Gate 4: False-Positive Rate Constraint (FPR Increase <= 0.00 / 1000 tiles)                  |
|    - Gate 5: Edge Latency Budget (<= 45.0 ms)                                                    |
|    - Gate 6: Model Checkpoint Size Budget (<= 25.0 MB)                                           |
+--------------------------------------------------------------------------------------------------+
         |                                                 |
  (All Gates PASS)                                  (Any Gate FAILS)
         v                                                 v
+-----------------------------+                   +-----------------------------+
| 7A. PROMOTION               |                   | 7B. REJECTION               |
|  - Challenger -> CHAMPION   |                   |  - Candidate -> REJECTED    |
|  - Previous -> ARCHIVED     |                   |  - Champion RETAINED        |
|  - Model Registry updated   |                   |  - Candidate archived safely|
+-----------------------------+                   +-----------------------------+
              \                                                 /
               \-----------------------+-----------------------/
                                       |
                                       v
+--------------------------------------------------------------------------------------------------+
| 8. IMMUTABLE AUDIT TRAIL & COMPLIANCE LOGGING  (ml/learnloop/audit.py)                           |
|    - Appends complete execution record to outputs/learnloop_audit_trail.jsonl                    |
|    - Stores cycle_id, gate results, metrics diffs, decision, and active model state              |
+--------------------------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------------------------+
| 9. RAPID ZERO-DOWNTIME ROLLBACK  (ml/learnloop/registry.py)                                      |
|    - If deployed Champion experiences operational degradation:                                   |
|    - Instantly restores previous validated Champion from archive ledger                          |
+--------------------------------------------------------------------------------------------------+
```

> [!IMPORTANT]
> **Supervised Active Learning Clarification**: LearnLoop is **supervised active learning with multi-gate retraining and champion/challenger lifecycle governance**. It is **NOT** reinforcement learning, and makes no unsubstantiated claims of continuous online self-training without human validation.

---

## 2. Core Subsystems & Implementation Details

### 2.1 Human Verification Storage (`ml/learnloop/feedback_store.py`)
Operator feedback is stored as structured `HumanVerificationRecord` objects conforming to the system data contract:
- `contact_id`: Unique identifier referencing the Contact Package.
- `survey_id`: Parent survey swath mission identifier.
- `original_prediction`: Canonical class predicted by detector/fused pipeline.
- `original_confidence`: Fused/calibrated confidence score.
- `original_bbox`: Bounding box `[x1, y1, x2, y2]`.
- `evidence`: Complete multi-modal evidence dictionary (acoustic shadow, highlight contrast, texture entropy, moments).
- `model_version`: Deployed detector/fusion version tag (e.g. `YOLO11N-KD-v3`).
- `human_action`: One of `ACCEPTED`, `REJECTED`, `CORRECTED`, `UNCERTAIN`.
- `corrected_class` / `corrected_bbox`: Operator adjustments when human action is `CORRECTED`.
- `verification_timestamp`: UTC ISO-8601 timestamp.
- `reviewer_id`: Operator badge/identifier for accountability.

Persistent storage is maintained in `data/feedback/verified_feedback.jsonl` with an in-memory thread-safe index for instant statistical queries.

### 2.2 Retraining Gate Evaluator (`ml/learnloop/trigger.py`)
Rather than relying on a single simplistic sample count, LearnLoop enforces a **multi-variable trigger policy**:
1. **Total Sample Gate (`min_verified_samples`)**: Ensures baseline statistical mass.
2. **New Sample Gate (`min_new_samples_since_last_training`)**: Prevents continuous retraining on tiny increments.
3. **Per-Class Distribution Gate (`min_samples_per_class`)**: Enforces sufficient representation across all 5 canonical sonar classes (`mine_like_contact`, `shipwreck`, `airplane_wreck`, `drowning_victim`, `debris`).
4. **Class Imbalance Boundary (`max_class_imbalance_ratio`)**: Blocks retraining if high-frequency classes completely drown rare classes.

*Note: Trigger parameters are operational configuration parameters, not scientifically hard-coded universals.*

### 2.3 Immutable Dataset Versioning (`ml/learnloop/dataset_versioner.py`)
To prevent data contamination:
- The base dataset `data/dataset_v1.0/` is **strictly immutable**.
- When retraining triggers, a new dataset version manifest (`data/versions/dataset_v1.1/manifest.json`) is generated.
- The manifest records:
  - `dataset_version` and `parent_version`
  - Total and incremental sample counts
  - Class distribution of newly incorporated verified samples
  - Full list of constituent `sample_ids`
  - SHA-256 provenance hashes and storage paths.

### 2.4 Candidate Training & Checkpoint Isolation (`ml/learnloop/training_runner.py`)
- Candidate jobs inherit architecture and weights from the deployed **Champion** model.
- Checkpoints are saved to an isolated directory: `outputs/models/candidates/candidate_<version>.pt`.
- The active Champion checkpoint is **never overwritten** during training.

### 2.5 Champion / Challenger Benchmarking & Promotion Gates (`ml/learnloop/evaluator.py`)
Candidates are evaluated against the current Champion on a held-out benchmark across 7 strict criteria:

| Promotion Gate | Metric Checked | Default Threshold | Rationale |
| :--- | :--- | :--- | :--- |
| **1. Recall Non-Degradation** | Total Recall | Drop $\le 0.01$ | Mission safety: cannot miss critical underwater hazards. |
| **2. Rare-Class Recall** | Per-Class Recall | Drop $\le 0.02$ | Prevents sacrificing drowning victims or airplane wrecks for mines. |
| **3. F1 Score Maintenance** | F1 Score | Diff $\ge 0.00$ | Overall harmonic balance must improve or hold. |
| **4. False-Positive Rate** | FP / 1,000 Tiles | Increase $\le 0.00$ | Operator fatigue prevention. |
| **5. Edge Compute Latency** | Inference Latency | $\le 45.0\text{ ms}$ | Real-time edge compute budget on autonomous vessels. |
| **6. Checkpoint Size** | Model Size (MB) | $\le 25.0\text{ MB}$ | Edge storage and embedded RAM constraint. |
| **7. mAP50 Preservation** | mAP50 Score | $\ge \text{Champion mAP50}$ | Standard detection fidelity preservation. |

### 2.6 Model Registry & Rollback Engine (`ml/learnloop/registry.py`)
Maintains `outputs/model_registry.json` tracking all models across four lifecycle states:
- **`CHAMPION`**: Currently active production model for survey inference.
- **`CHALLENGER`**: Retrained candidate undergoing benchmark gating.
- **`REJECTED`**: Archived candidates that failed promotion gates (never deleted, kept for post-mortem analysis).
- **`ARCHIVED`**: Previously promoted champions retired upon arrival of superior challengers.

**Zero-Downtime Rollback**: If an active champion shows operational defects in field surveys, calling `registry.rollback_to_previous_champion()` instantly restores the previous validated champion from archive and demotes the current champion.

### 2.7 Audit Trail (`ml/learnloop/audit.py`)
All LearnLoop cycles append an immutable record to `outputs/learnloop_audit_trail.jsonl` containing the `cycle_id`, trigger reason, sample volume, candidate checkpoint path, metrics delta, gate evaluation breakdown, promotion/rejection decision, and post-cycle active champion.

---

## 3. Test Verification Matrix

The LearnLoop test suite (`tests/test_learnloop_lifecycle.py`) verifies all 12 operational scenarios with 100% pass rate:

```
tests/test_learnloop_lifecycle.py::TestLearnLoopFeedbackStore::test_feedback_storage_and_query PASSED [  8%]
tests/test_learnloop_lifecycle.py::TestLearnLoopTrigger::test_trigger_threshold_not_reached PASSED [ 16%]
tests/test_learnloop_lifecycle.py::TestLearnLoopTrigger::test_trigger_threshold_reached PASSED [ 25%]
tests/test_learnloop_lifecycle.py::TestDatasetVersioning::test_dataset_versioning_immutability PASSED [ 33%]
tests/test_learnloop_lifecycle.py::TestChampionChallengerEvaluator::test_challenger_better_all_gates_pass PASSED [ 41%]
tests/test_learnloop_lifecycle.py::TestChampionChallengerEvaluator::test_challenger_worse_overall_rejected PASSED [ 50%]
tests/test_learnloop_lifecycle.py::TestChampionChallengerEvaluator::test_challenger_improves_map_but_hurts_recall PASSED [ 58%]
tests/test_learnloop_lifecycle.py::TestChampionChallengerEvaluator::test_challenger_improves_recall_but_exceeds_latency_budget PASSED [ 66%]
tests/test_learnloop_lifecycle.py::TestModelRegistryAndRollback::test_promotion_and_archive PASSED [ 75%]
tests/test_learnloop_lifecycle.py::TestModelRegistryAndRollback::test_rollback_to_previous_champion PASSED [ 83%]
tests/test_learnloop_lifecycle.py::TestEndToEndLearnLoopPipeline::test_full_cycle_success PASSED [ 91%]
tests/test_learnloop_lifecycle.py::TestEndToEndLearnLoopPipeline::test_candidate_training_failure PASSED [100%]
```

Full repository test suite: **61 passed in 8.21s**.

---

## 4. Jury Presentation Narrative & Live Demo Isolation

### 4.1 Jury-Facing Explanation
> *"In SONAR-INTEL, active learning is grounded in human-in-the-loop operational rigor. Every contact accepted, rejected, or corrected by a hydrographer is recorded in an immutable feedback store with its acoustic evidence and bounding coordinates. When configurable data volume and class-balance gates are met, an automated retraining pipeline creates a candidate Challenger model inheriting weights from the active Champion.*
> 
> *The Challenger is independently evaluated against 7 safety, accuracy, and edge-latency gates. If a model improves mAP but hurts rare-class recall or exceeds the 45 ms edge compute budget, it is rejected and archived. Only a strictly superior Challenger is promoted to Champion, with instant rollback available at all times."*

### 4.2 Demo Safety Isolation
- **No Live UI Retraining Dependency**: The live SONAR-INTEL demo frontend and backend inference services operate normally without dependency on active LearnLoop retraining cycles.
- **Zero Cosmetic Placeholders**: No fake "AI learning live" progress bars or simulated retraining popups are used.
