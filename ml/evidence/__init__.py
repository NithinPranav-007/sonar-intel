"""
SONAR-INTEL: Multi-Modal Evidence Extraction Layer.

Provides structured, unweighted extraction of detector, segmentation,
morphological, and acoustic physical priors for candidate sonar detections.
"""

from ml.evidence.schemas import (
    CandidateEvidence,
    DetectorEvidence,
    SegmentationEvidence,
    MorphologyEvidence,
    AcousticEvidence,
    EvidenceStatus
)
from ml.evidence.extractor import EvidenceExtractor
from ml.evidence.morphology_extractor import extract_morphology_evidence
from ml.evidence.acoustic_extractor import (
    extract_acoustic_evidence,
    compute_shannon_entropy,
    compute_directional_consistency
)
from ml.evidence.normalization import (
    normalize_highlight_contrast,
    normalize_shadow_deficit,
    normalize_entropy,
    normalize_gradient_magnitude
)

__all__ = [
    "CandidateEvidence",
    "DetectorEvidence",
    "SegmentationEvidence",
    "MorphologyEvidence",
    "AcousticEvidence",
    "EvidenceStatus",
    "EvidenceExtractor",
    "extract_morphology_evidence",
    "extract_acoustic_evidence",
    "compute_shannon_entropy",
    "compute_directional_consistency",
    "normalize_highlight_contrast",
    "normalize_shadow_deficit",
    "normalize_entropy",
    "normalize_gradient_magnitude"
]
