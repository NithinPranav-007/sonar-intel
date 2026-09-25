"""
SONAR-INTEL LearnLoop Package.

Production backend framework for active learning, human verification feedback persistence,
multi-gate retraining triggers, immutable dataset versioning, Champion/Challenger evaluation,
safety promotion gates, model registry, and rollback operations.
"""

from ml.learnloop.schemas import (
    HumanAction,
    ModelStatus,
    PromotionStatus,
    HumanVerificationRecord,
    TriggerConfig,
    TriggerResult,
    DatasetManifest,
    EvaluationMetrics,
    PromotionGatesConfig,
    GateEvaluationResult,
    PromotionDecision,
    ModelRegistryRecord,
    AuditLogEntry
)
from ml.learnloop.feedback_store import FeedbackStore
from ml.learnloop.trigger import RetrainingTrigger
from ml.learnloop.dataset_versioner import DatasetVersioner
from ml.learnloop.training_runner import CandidateTrainingRunner
from ml.learnloop.evaluator import ChampionChallengerEvaluator
from ml.learnloop.registry import ModelRegistry
from ml.learnloop.audit import AuditLogger
from ml.learnloop.pipeline import LearnLoopOrchestrator

__all__ = [
    "HumanAction",
    "ModelStatus",
    "PromotionStatus",
    "HumanVerificationRecord",
    "TriggerConfig",
    "TriggerResult",
    "DatasetManifest",
    "EvaluationMetrics",
    "PromotionGatesConfig",
    "GateEvaluationResult",
    "PromotionDecision",
    "ModelRegistryRecord",
    "AuditLogEntry",
    "FeedbackStore",
    "RetrainingTrigger",
    "DatasetVersioner",
    "CandidateTrainingRunner",
    "ChampionChallengerEvaluator",
    "ModelRegistry",
    "AuditLogger",
    "LearnLoopOrchestrator"
]
