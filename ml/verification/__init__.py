"""
Second-Stage Verification and Structural Segmentation Module for SONAR-INTEL.

Provides candidate ROI extraction, SW-Net directional semantic segmentation,
and morphological physical prior evaluation.
"""

from ml.verification.roi_extractor import ExtractedROI, ROIExtractor
from ml.verification.swnet_verifier import (
    BaseROIVerifier,
    SWNetVerifier,
    VerificationResult,
    compute_morphological_metrics
)

__all__ = [
    "ExtractedROI",
    "ROIExtractor",
    "BaseROIVerifier",
    "SWNetVerifier",
    "VerificationResult",
    "compute_morphological_metrics"
]
