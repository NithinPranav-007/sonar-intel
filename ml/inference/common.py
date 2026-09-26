"""
Common Data Structures for Hybrid Sonar Anomaly Inference.

Provides unified, backend-agnostic schemas for both Hugging Face Space API
and Local PyTorch/Ultralytics detector inference.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field


@dataclass
class CommonDetection:
    """Standardized detection representation across all backends."""
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2] (0-indexed integer pixel coordinates)
    class_id: Optional[int] = None
    tile_id: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    is_filtered: bool = False
    filter_reason: Optional[str] = None
    source_backend: str = "local"  # "huggingface" | "local"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceResult:
    """Unified result container returned by the hybrid inference pipeline."""
    success: bool
    backend: str  # "huggingface" | "local" | "none"
    detections: List[CommonDetection] = field(default_factory=list)
    annotated_image: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    error: Optional[str] = None
    model_name: str = ""
    model_version: str = ""
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "backend": self.backend,
            "detections": [d.to_dict() for d in self.detections],
            "annotated_image": self.annotated_image,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "error": self.error,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "execution_time_ms": self.execution_time_ms,
        }
