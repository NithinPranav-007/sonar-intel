"""
LearnLoop Schemas & Data Contracts.

Defines all data structures for human feedback verification, retraining triggers,
dataset manifests, champion/challenger evaluations, promotion gates, model registry,
and audit trails.
"""

from typing import Dict, Any, Optional, List, Union, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum


class HumanAction(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    CORRECTED = "corrected"
    UNCERTAIN = "uncertain"


class ModelStatus(str, Enum):
    CHAMPION = "champion"
    CHALLENGER = "challenger"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class PromotionStatus(str, Enum):
    PROMOTED = "PROMOTED"
    REJECTED = "REJECTED"


@dataclass
class HumanVerificationRecord:
    """
    Standardized feedback record captured from operator verification.
    """
    contact_id: str
    survey_id: str
    original_prediction: str
    original_confidence: float
    original_bbox: List[int]
    evidence: Dict[str, Any]
    model_version: str
    human_action: HumanAction
    roi_ref: Optional[str] = None
    mask_ref: Optional[str] = None
    corrected_class: Optional[str] = None
    corrected_bbox: Optional[List[int]] = None
    verification_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reviewer_id: Optional[str] = "operator_default"
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["human_action"] = self.human_action.value if isinstance(self.human_action, HumanAction) else str(self.human_action)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanVerificationRecord":
        action = data.get("human_action")
        if isinstance(action, str):
            action = HumanAction(action.lower())
        return cls(
            contact_id=data["contact_id"],
            survey_id=data.get("survey_id", "unknown_survey"),
            original_prediction=data["original_prediction"],
            original_confidence=float(data["original_confidence"]),
            original_bbox=data.get("original_bbox", [0, 0, 0, 0]),
            evidence=data.get("evidence", {}),
            model_version=data.get("model_version", "unknown_v1"),
            human_action=action,
            roi_ref=data.get("roi_ref"),
            mask_ref=data.get("mask_ref"),
            corrected_class=data.get("corrected_class"),
            corrected_bbox=data.get("corrected_bbox"),
            verification_timestamp=data.get("verification_timestamp", datetime.now(timezone.utc).isoformat()),
            reviewer_id=data.get("reviewer_id", "operator_default"),
            notes=data.get("notes")
        )


@dataclass
class TriggerConfig:
    """
    Configuration gates for automated retraining triggers.
    Note: These are operational configuration values, not experimentally proven optimal thresholds.
    """
    min_verified_samples: int = 500
    min_new_samples_since_last_training: int = 250
    min_samples_per_class: Dict[str, int] = field(default_factory=lambda: {
        "mine_like_contact": 25,
        "shipwreck": 25,
        "airplane_wreck": 10,
        "drowning_victim": 10,
        "debris": 25
    })
    max_class_imbalance_ratio: Optional[float] = 10.0  # max(count)/min(count) limit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TriggerResult:
    """Outcome of evaluating retraining trigger conditions."""
    is_triggered: bool
    reason: str
    total_verified_samples: int
    new_samples_count: int
    class_counts: Dict[str, int]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetManifest:
    """
    Metadata for versioned retraining datasets.
    Guarantees immutability of parent datasets (e.g. data/dataset_v1.0/).
    """
    dataset_version: str
    parent_version: str
    new_sample_count: int
    total_sample_count: int
    class_distribution: Dict[str, int]
    source_information: Dict[str, Any]
    creation_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sample_ids: List[str] = field(default_factory=list)
    manifest_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationMetrics:
    """Standardized performance and operational metrics for model benchmarking."""
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    f1: float
    per_class_recall: Dict[str, float]
    false_positives_per_1000_tiles: float
    latency_ms: float
    model_size_mb: float
    calibration_error: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionGatesConfig:
    """
    Predefined gates for comparing Challenger vs. Champion.
    Values are operational constraints, not hard-coded scientific claims.
    """
    max_recall_drop: float = 0.01                # Challenger recall >= Champion recall - 0.01
    max_rare_class_recall_drop: float = 0.02     # Per-class recall drop <= 0.02
    min_f1_improvement: float = 0.00             # F1 must not degrade (diff >= 0.0)
    max_fpr_increase: float = 0.00               # False positives per 1k tiles must not increase
    max_latency_ms: float = 45.0                 # Hard edge-compute latency budget (ms)
    max_model_size_mb: float = 25.0              # Max checkpoint size budget (MB)
    require_map50_non_degradation: bool = True   # mAP50 must not degrade

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateEvaluationResult:
    """Result of evaluating an individual promotion gate."""
    gate_name: str
    passed: bool
    champion_value: Any
    challenger_value: Any
    threshold: Any
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionDecision:
    """Complete record of a Champion vs. Challenger promotion decision."""
    promoted: bool
    decision: str                                # "PROMOTED" or "REJECTED"
    decision_timestamp: str
    champion_version: str
    challenger_version: str
    champion_metrics: EvaluationMetrics
    challenger_metrics: EvaluationMetrics
    gate_results: List[GateEvaluationResult]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "promoted": self.promoted,
            "decision": self.decision,
            "decision_timestamp": self.decision_timestamp,
            "champion_version": self.champion_version,
            "challenger_version": self.challenger_version,
            "champion_metrics": self.champion_metrics.to_dict(),
            "challenger_metrics": self.challenger_metrics.to_dict(),
            "gate_results": [g.to_dict() for g in self.gate_results],
            "reason": self.reason
        }


@dataclass
class ModelRegistryRecord:
    """Metadata record for registered models in the model lifecycle registry."""
    model_id: str
    version: str
    architecture: str
    checkpoint_path: str
    dataset_version: str
    status: str                                  # "champion", "challenger", "rejected", "archived"
    metrics: Dict[str, Any]
    latency_ms: float
    size_mb: float
    created_at: str
    teacher: Optional[str] = None
    student: Optional[str] = None
    training_run: Optional[str] = None
    promotion_status: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditLogEntry:
    """Immutable audit record for a single LearnLoop iteration."""
    cycle_id: str
    timestamp: str
    trigger_reason: str
    verified_sample_count: int
    dataset_version: str
    training_config: Dict[str, Any]
    candidate_checkpoint: str
    evaluation_metrics: Dict[str, Any]
    gate_results: List[Dict[str, Any]]
    decision: str
    active_champion_after_cycle: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
