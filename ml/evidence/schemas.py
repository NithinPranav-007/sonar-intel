"""
SONAR-INTEL: Evidence Layer Schemas and Data Models.

Defines decoupled, structured representation of all candidate evidence:
1. Detector Evidence (YOLO candidate outputs)
2. Segmentation Evidence (SW-Net / mask metrics, explicitly distinguishing available vs unavailable)
3. Morphology Evidence (Raw geometric shapes, moments, compactness, solidity, elongation)
4. Acoustic Evidence (Highlight intensity, shadow presence/deficit, texture entropy, gradients)
5. Evidence Status (Data quality flags, boundary warnings, contrast flags)

CRITICAL DESIGN PRINCIPLE:
- Strictly separates RAW FEATURES from NORMALIZED VALUES from FUTURE FUSION SCORES.
- Explicitly represents missing/unavailable evidence without silent zeroes or manufactured ground truth.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict


@dataclass
class DetectorEvidence:
    """Raw evidence produced by Stage-1 YOLO candidate detector."""
    class_name: str
    class_id: int
    detector_confidence: float          # Raw model confidence [0.0, 1.0]
    bbox: List[int]                     # [x1, y1, x2, y2]
    bbox_width: int
    bbox_height: int
    bbox_area: int
    aspect_ratio: float                 # max(w, h) / min(w, h) (>= 1.0)
    class_probability_margin: Optional[float] = None  # Difference between top-1 and top-2 class probs if available
    class_probabilities: Optional[Dict[str, float]] = None # Full class probability distribution if available
    tile_relative_coords: Optional[Dict[str, float]] = None # Normalized tile coordinates [center_x, center_y, w, h] in [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentationEvidence:
    """Evidence produced by Stage-2 segmentation verifier (e.g. SW-Net)."""
    available: bool                     # True if supervised segmentation model is applied
    source: Optional[str] = None        # e.g., "SW-Net-Shipwreck", "None", "Heuristic-Contour"
    confidence: Optional[float] = None  # Pixel-level segmentation mean confidence
    mask_status: str = "UNAVAILABLE"    # "AVAILABLE", "UNAVAILABLE_CLASS_NOT_SUPPORTED", "UNAVAILABLE_NO_CHECKPOINT", "INVALID_ROI"
    mask_area_px: Optional[int] = None
    mask_to_box_ratio: Optional[float] = None
    mask_coverage: Optional[float] = None      # Fraction of extracted crop area covered by mask
    mask_compactness: Optional[float] = None   # Compactness 4*pi*A/P^2 of the segmentation mask
    mask_orientation_deg: Optional[float] = None # Orientation of segmentation mask [-90, +90]
    mask_elongation: Optional[float] = None    # Major / minor axis ratio of segmentation mask

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MorphologyEvidence:
    """
    Geometric and shape physical priors extracted from mask or bounding geometry.
    Raw measurements are preserved independently of any scoring function.
    """
    source: str                         # "segmentation_mask", "heuristic_contour", "bbox_geometry"
    area_px: float                      # Pixel area of target
    width_px: float                     # Spatial width
    height_px: float                    # Spatial height
    aspect_ratio: float                 # max(w, h) / min(w, h) (>= 1.0)
    perimeter_px: float                 # Perimeter length in pixels
    compactness: float                  # Isoperimetric quotient 4 * pi * Area / Perimeter^2 in [0.0, 1.0]
    eccentricity: float                 # sqrt(1 - (b/a)^2) in [0.0, 1.0]
    elongation: float                   # Major axis / minor axis length ratio (>= 1.0)
    orientation_deg: float              # Principal axis angle in [-90.0, +90.0] degrees
    solidity: float                     # Area / ConvexHullArea in [0.0, 1.0]
    convexity: float = 1.0              # ConvexHullPerimeter / ContourPerimeter in [0.0, 1.0]
    status: str = "VALID"               # "VALID", "DEGENERATE", "ESTIMATED_FROM_BBOX"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AcousticEvidence:
    """
    Image-derived acoustic backscatter, shadow, and texture features.
    """
    # 1. Highlight Features
    highlight_mean: float               # Raw mean intensity of candidate target [0.0, 255.0]
    highlight_max: float                # Maximum pixel intensity in candidate [0.0, 255.0]
    local_background_mean: float        # Raw mean intensity of ambient seabed context ring [0.0, 255.0]
    local_background_std: float         # Standard deviation of ambient seabed context ring
    local_contrast_ratio: float         # highlight_mean / max(1.0, background_mean)
    normalized_highlight_strength: float # (highlight_mean - background_mean) / 255.0 in [-1.0, 1.0]

    # 2. Acoustic Shadow Features
    shadow_status: str                  # "PRESENT", "ABSENT", "UNCERTAIN", "OUT_OF_BOUNDS"
    shadow_mean: float                  # Mean intensity of down-range shadow region [0.0, 255.0]
    shadow_contrast_ratio: float        # background_mean / max(1.0, shadow_mean)
    shadow_deficit: float               # max(0.0, background_mean - shadow_mean)
    shadow_candidate_area_px: int       # Area of dark shadow region in pixels
    shadow_to_highlight_area_ratio: float # shadow_area / max(1, target_area)
    shadow_length_px: float             # Estimated along-range shadow extension in pixels

    # 3. Acoustic Texture Features
    local_variance: float               # Intensity variance inside target ROI
    entropy: float                      # Shannon entropy of grayscale distribution in bits [0.0, 8.0]
    gradient_magnitude_mean: float      # Mean Sobel gradient magnitude across edges
    edge_density: float                 # Fraction of Canny edge pixels in candidate [0.0, 1.0]

    # 4. Directional Backscatter Features
    dominant_gradient_orientation_deg: float # Dominant gradient orientation in [-90.0, 90.0] degrees
    directional_consistency: float      # Angular coherence of gradient vectors in [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceStatus:
    """Data quality and boundary status flags for the extracted evidence."""
    overall: str                        # "COMPLETE", "PARTIAL", "DEGRADED", "INVALID"
    warnings: List[str] = field(default_factory=list)
    is_touching_boundary: bool = False
    is_low_contrast: bool = False
    is_degenerate_box: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateEvidence:
    """
    Unified Candidate Evidence Container.
    Encapsulates all measurable detector, segmentation, morphology, and acoustic evidence.
    """
    detector: DetectorEvidence
    segmentation: SegmentationEvidence
    morphology: MorphologyEvidence
    acoustic: AcousticEvidence
    status: EvidenceStatus
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector": self.detector.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "morphology": self.morphology.to_dict(),
            "acoustic": self.acoustic.to_dict(),
            "status": self.status.to_dict(),
            "raw_metadata": self.raw_metadata
        }
