"""
Multi-Modal Evidence Fusion Engine for SONAR-INTEL.

Combines:
1. Stage-1 Detector Confidence
2. Stage-2 Segmentation Confidence (when available)
3. Stage-2 Morphology Physical Priors (compactness, solidity, elongation)
4. Stage-2 Acoustic Context (highlight contrast, shadow deficit, entropy)

DESIGN PRINCIPLES:
- Strictly preserves raw evidence, normalized sub-features, modality scores, and fused score as distinct fields.
- Adaptively rebalances weights when segmentation evidence is unavailable without substituting fake values.
- Accommodates shadow-less flat/buried targets by redistributing shadow weight to highlight contrast.
- Clearly tagged as 'baseline_adaptive_linear_v1' MVP baseline (pending learned fusion weights on validation corpus).
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

from ml.evidence.schemas import CandidateEvidence
from ml.evidence.normalization import (
    normalize_highlight_contrast,
    normalize_shadow_deficit,
    normalize_entropy,
    normalize_gradient_magnitude
)


@dataclass
class FusedEvidenceResult:
    """Structured container for multi-modal evidence fusion output."""
    fused_score: float                     # Composite uncalibrated fusion score in [0.0, 1.0]
    modality_scores: Dict[str, Optional[float]] # Individual normalized scores per modality
    normalized_features: Dict[str, float]  # Continuous [0.0, 1.0] feature mappings
    weights_applied: Dict[str, float]      # Active fusion weights
    fusion_status: str                     # "COMPLETE_4_MODALITIES", "DEGRADED_3_MODALITIES", "INVALID_INPUT"
    fusion_method: str = "baseline_adaptive_linear_v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceFusionEngine:
    """
    Executes adaptive multi-modal evidence fusion across candidate detections.
    """

    def __init__(
        self,
        base_weights_4mod: Optional[Dict[str, float]] = None,
        base_weights_3mod: Optional[Dict[str, float]] = None
    ):
        # Default baseline weights when 4 modalities (including segmentation) are present
        self.weights_4mod = base_weights_4mod or {
            "detector": 0.40,
            "segmentation": 0.25,
            "acoustic": 0.20,
            "morphology": 0.15
        }
        # Default baseline weights when 3 modalities (segmentation unavailable) are present
        self.weights_3mod = base_weights_3mod or {
            "detector": 0.50,
            "acoustic": 0.30,
            "morphology": 0.20
        }

    def fuse(self, evidence: CandidateEvidence) -> FusedEvidenceResult:
        """
        Fuses multi-modal candidate evidence into a composite score.

        Args:
            evidence: CandidateEvidence object from EvidenceExtractor.

        Returns:
            FusedEvidenceResult containing sub-scores, weights, and fused score.
        """
        if evidence.status.overall == "INVALID":
            return FusedEvidenceResult(
                fused_score=0.0,
                modality_scores={"detector": 0.0, "segmentation": None, "acoustic": 0.0, "morphology": 0.0},
                normalized_features={},
                weights_applied={},
                fusion_status="INVALID_INPUT",
                fusion_method="baseline_adaptive_linear_v1"
            )

        # 1. Normalized Feature Representations
        norm_contrast = normalize_highlight_contrast(evidence.acoustic.local_contrast_ratio)
        norm_shadow = normalize_shadow_deficit(evidence.acoustic.shadow_deficit)
        norm_entropy = normalize_entropy(evidence.acoustic.entropy)
        norm_grad = normalize_gradient_magnitude(evidence.acoustic.gradient_magnitude_mean)

        norm_features = {
            "highlight_contrast": round(norm_contrast, 3),
            "shadow_deficit": round(norm_shadow, 3),
            "texture_entropy": round(norm_entropy, 3),
            "edge_gradient": round(norm_grad, 3),
            "compactness": round(evidence.morphology.compactness, 3),
            "solidity": round(evidence.morphology.solidity, 3),
            "convexity": round(evidence.morphology.convexity, 3)
        }

        # 2. Compute Individual Modality Sub-Scores
        # A. Detector Sub-Score
        score_detector = float(min(1.0, max(0.0, evidence.detector.detector_confidence)))

        # B. Segmentation Sub-Score
        if evidence.segmentation.available and evidence.segmentation.confidence is not None:
            score_segmentation = float(min(1.0, max(0.0, evidence.segmentation.confidence)))
        else:
            score_segmentation = None

        # C. Morphology Sub-Score (Geometry physical priors)
        # Combines compactness (40%), solidity (40%), and bounded elongation (20%)
        elongation_factor = min(1.0, evidence.morphology.elongation / 3.0)
        score_morphology = (
            0.40 * evidence.morphology.compactness +
            0.40 * evidence.morphology.solidity +
            0.20 * elongation_factor
        )
        score_morphology = float(min(1.0, max(0.0, score_morphology)))

        # D. Acoustic Sub-Score (Backscatter, Shadow, Texture)
        if evidence.acoustic.shadow_status == "PRESENT":
            score_acoustic = (
                0.40 * norm_contrast +
                0.35 * norm_shadow +
                0.15 * norm_entropy +
                0.10 * norm_grad
            )
        elif evidence.acoustic.shadow_status in ("ABSENT", "UNCERTAIN"):
            # Flat/buried target: redistribute shadow weight to highlight contrast & texture
            score_acoustic = (
                0.60 * norm_contrast +
                0.25 * norm_entropy +
                0.15 * norm_grad
            )
        else:
            # OUT_OF_BOUNDS / Edge
            score_acoustic = (
                0.60 * norm_contrast +
                0.40 * norm_entropy
            )
        score_acoustic = float(min(1.0, max(0.0, score_acoustic)))

        # 3. Dynamic Weight Selection & Score Fusion
        modality_scores = {
            "detector": round(score_detector, 3),
            "segmentation": round(score_segmentation, 3) if score_segmentation is not None else None,
            "acoustic": round(score_acoustic, 3),
            "morphology": round(score_morphology, 3)
        }

        if score_segmentation is not None:
            # 4-Modality Fusion (Shipwreck with SW-Net)
            weights = self.weights_4mod.copy()
            fused = (
                weights["detector"] * score_detector +
                weights["segmentation"] * score_segmentation +
                weights["acoustic"] * score_acoustic +
                weights["morphology"] * score_morphology
            )
            fusion_status = "COMPLETE_4_MODALITIES"
        else:
            # 3-Modality Fusion (Other classes or pre-checkpoint)
            weights = self.weights_3mod.copy()
            fused = (
                weights["detector"] * score_detector +
                weights["acoustic"] * score_acoustic +
                weights["morphology"] * score_morphology
            )
            fusion_status = "DEGRADED_3_MODALITIES"

        fused_score = float(min(1.0, max(0.0, fused)))

        return FusedEvidenceResult(
            fused_score=round(fused_score, 3),
            modality_scores=modality_scores,
            normalized_features=norm_features,
            weights_applied=weights,
            fusion_status=fusion_status,
            fusion_method="baseline_adaptive_linear_v1"
        )
