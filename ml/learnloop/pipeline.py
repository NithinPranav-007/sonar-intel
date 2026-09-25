"""
LearnLoop Pipeline Controller.

Coordinates the end-to-end active learning lifecycle:
Feedback -> Trigger Evaluation -> Dataset Versioning -> Candidate Retraining ->
Champion/Challenger Benchmarking -> Gate Evaluation -> Registry Promotion/Rejection -> Audit Logging.
"""

from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone

from ml.learnloop.schemas import (
    HumanVerificationRecord,
    HumanAction,
    TriggerConfig,
    TriggerResult,
    DatasetManifest,
    EvaluationMetrics,
    PromotionGatesConfig,
    PromotionDecision,
    ModelRegistryRecord,
    ModelStatus
)
from ml.learnloop.feedback_store import FeedbackStore
from ml.learnloop.trigger import RetrainingTrigger
from ml.learnloop.dataset_versioner import DatasetVersioner
from ml.learnloop.training_runner import CandidateTrainingRunner
from ml.learnloop.evaluator import ChampionChallengerEvaluator
from ml.learnloop.registry import ModelRegistry
from ml.learnloop.audit import AuditLogger


class LearnLoopOrchestrator:
    """
    Production LearnLoop lifecycle orchestrator.
    Executes automated retraining cycles with safe gating, candidate isolation,
    and audit tracking.
    """

    def __init__(
        self,
        feedback_store: Optional[FeedbackStore] = None,
        trigger: Optional[RetrainingTrigger] = None,
        versioner: Optional[DatasetVersioner] = None,
        training_runner: Optional[CandidateTrainingRunner] = None,
        evaluator: Optional[ChampionChallengerEvaluator] = None,
        registry: Optional[ModelRegistry] = None,
        audit_logger: Optional[AuditLogger] = None
    ):
        self.feedback_store = feedback_store or FeedbackStore()
        self.trigger = trigger or RetrainingTrigger()
        self.versioner = versioner or DatasetVersioner()
        self.training_runner = training_runner or CandidateTrainingRunner()
        self.evaluator = evaluator or ChampionChallengerEvaluator()
        self.registry = registry or ModelRegistry()
        self.audit_logger = audit_logger or AuditLogger()
        self.last_training_timestamp: Optional[str] = None

    def submit_feedback(
        self,
        contact_dict: Dict[str, Any],
        human_action: str,
        corrected_class: Optional[str] = None,
        corrected_bbox: Optional[List[int]] = None,
        reviewer_id: Optional[str] = "operator_default",
        notes: Optional[str] = None
    ) -> HumanVerificationRecord:
        """Submits human verification feedback for an operator-reviewed contact."""
        return self.feedback_store.add_feedback_from_contact(
            contact_dict=contact_dict,
            human_action=human_action,
            corrected_class=corrected_class,
            corrected_bbox=corrected_bbox,
            reviewer_id=reviewer_id,
            notes=notes
        )

    def check_and_run_cycle(
        self,
        candidate_version: str,
        dataset_version: str,
        mock_training: bool = True,
        mock_failure: bool = False,
        candidate_metrics: Optional[EvaluationMetrics] = None,
        training_hyperparams: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates trigger conditions and executes a complete retraining & evaluation cycle if met.

        Returns a detailed summary dictionary of the cycle outcome.
        """
        # Step 1: Evaluate trigger gates
        trigger_res = self.trigger.evaluate(
            feedback_store=self.feedback_store,
            last_training_timestamp=self.last_training_timestamp
        )

        if not trigger_res.is_triggered:
            return {
                "executed": False,
                "status": "TRIGGER_GATES_NOT_MET",
                "trigger_result": trigger_res.to_dict()
            }

        # Step 2: Get active Champion
        champion = self.registry.get_champion()
        if champion is None:
            raise RuntimeError("No active Champion model registered in ModelRegistry.")

        # Step 3: Create versioned dataset manifest (Immutable base dataset)
        manifest = self.versioner.create_version(
            feedback_store=self.feedback_store,
            new_version=dataset_version,
            parent_version=champion.dataset_version,
            since_timestamp=self.last_training_timestamp
        )

        # Step 4: Dispatch candidate training job
        train_success, checkpoint_path, trained_metrics, train_log = self.training_runner.trigger_candidate_job(
            champion_checkpoint=champion.checkpoint_path,
            dataset_manifest=manifest,
            candidate_version=candidate_version,
            training_hyperparams=training_hyperparams,
            mock=mock_training,
            mock_failure=mock_failure,
            mock_metrics=candidate_metrics
        )

        if not train_success or checkpoint_path is None or trained_metrics is None:
            return {
                "executed": True,
                "status": "CANDIDATE_TRAINING_FAILED",
                "trigger_result": trigger_res.to_dict(),
                "dataset_manifest": manifest.to_dict(),
                "error": train_log
            }

        # Register candidate in registry as CHALLENGER
        challenger_record = ModelRegistryRecord(
            model_id=f"candidate_{candidate_version.lower().replace('-', '_')}",
            version=candidate_version,
            architecture=champion.architecture,
            checkpoint_path=checkpoint_path,
            dataset_version=manifest.dataset_version,
            status=ModelStatus.CHALLENGER.value,
            metrics=trained_metrics.to_dict(),
            latency_ms=trained_metrics.latency_ms,
            size_mb=trained_metrics.model_size_mb,
            created_at=datetime.now(timezone.utc).isoformat(),
            teacher=champion.teacher,
            student=champion.student,
            training_run=f"learnloop_{candidate_version}",
            notes=f"Candidate trained on {manifest.dataset_version}"
        )
        self.registry.register_model(challenger_record)

        # Step 5: Convert champion metrics to EvaluationMetrics
        champ_m = champion.metrics
        champion_eval_metrics = EvaluationMetrics(
            mAP50=float(champ_m.get("mAP50", 0.87)),
            mAP50_95=float(champ_m.get("mAP50_95", 0.60)),
            precision=float(champ_m.get("precision", 0.88)),
            recall=float(champ_m.get("recall", 0.86)),
            f1=float(champ_m.get("f1", 0.87)),
            per_class_recall=champ_m.get("per_class_recall", {
                "mine_like_contact": 0.87,
                "shipwreck": 0.90,
                "airplane_wreck": 0.84,
                "drowning_victim": 0.80,
                "debris": 0.88
            }),
            false_positives_per_1000_tiles=float(champ_m.get("false_positives_per_1000_tiles", 2.0)),
            latency_ms=champion.latency_ms,
            model_size_mb=champion.size_mb,
            calibration_error=champ_m.get("calibration_error")
        )

        # Step 6: Champion vs. Challenger Evaluation & Gating
        decision = self.evaluator.evaluate_challenger(
            champion_version=champion.version,
            challenger_version=candidate_version,
            champion_metrics=champion_eval_metrics,
            challenger_metrics=trained_metrics
        )

        # Step 7: Apply Promotion or Rejection
        if decision.promoted:
            self.registry.promote_challenger(challenger_record.model_id, decision)
            active_champ = candidate_version
        else:
            self.registry.reject_challenger(challenger_record.model_id, decision)
            active_champ = champion.version

        # Step 8: Append to Immutable Audit Log
        audit_entry = self.audit_logger.log_cycle(
            trigger_reason=trigger_res.reason,
            verified_sample_count=manifest.new_sample_count,
            dataset_manifest=manifest,
            candidate_checkpoint=checkpoint_path,
            decision=decision,
            active_champion_after_cycle=active_champ,
            training_config=training_hyperparams
        )

        self.last_training_timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "executed": True,
            "status": "COMPLETED",
            "trigger_result": trigger_res.to_dict(),
            "dataset_manifest": manifest.to_dict(),
            "candidate_checkpoint": checkpoint_path,
            "promotion_decision": decision.to_dict(),
            "active_champion": active_champ,
            "audit_cycle_id": audit_entry.cycle_id
        }
