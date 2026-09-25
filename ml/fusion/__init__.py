"""
SONAR-INTEL: Multi-Modal Evidence Fusion and Contact Package Module.

Provides:
- EvidenceFusionEngine: Adaptive combination of detector, segmentation, morphology, and acoustic evidence.
- Calibrators: BaseCalibrator, IdentityCalibrator, TemperatureCalibrator, PlattCalibrator.
- ContactPackage & ContactPackageBuilder: Machine-readable handoff contracts for Person 3.
"""

from ml.fusion.calibration import (
    BaseCalibrator,
    IdentityCalibrator,
    TemperatureCalibrator,
    PlattCalibrator
)
from ml.fusion.evidence_fusion import (
    EvidenceFusionEngine,
    FusedEvidenceResult
)
from ml.fusion.contact_package import (
    ContactPackage,
    ContactPackageBuilder,
    ContactLocation,
    ContactAssets,
    ModelProvenance,
    OperatorIndicators
)

__all__ = [
    "BaseCalibrator",
    "IdentityCalibrator",
    "TemperatureCalibrator",
    "PlattCalibrator",
    "EvidenceFusionEngine",
    "FusedEvidenceResult",
    "ContactPackage",
    "ContactPackageBuilder",
    "ContactLocation",
    "ContactAssets",
    "ModelProvenance",
    "OperatorIndicators"
]
