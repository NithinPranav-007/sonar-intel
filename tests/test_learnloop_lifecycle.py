"""
Comprehensive Test Suite for LearnLoop Model Lifecycle & Champion/Challenger Framework.

Validates:
1. Feedback storage & querying for accepted/rejected/corrected/uncertain actions
2. Retraining trigger gate evaluation (threshold not reached vs reached)
3. Dataset versioning and parent dataset immutability
4. Candidate training job creation and isolation
5. Candidate training failure handling
6. Champion/Challenger evaluation:
   - Challenger better across all gates -> PROMOTED
   - Challenger worse overall -> REJECTED
   - Challenger improves mAP50 but drops recall -> REJECTED
   - Challenger improves recall but exceeds edge latency budget -> REJECTED
7. Model Registry promotion and archive management
8. Rollback to previous validated champion
9. Audit trail persistence and integrity
"""

import os
import shutil
import pytest
from datetime import datetime, timezone

from ml.learnloop.schemas import (
    HumanAction,
    ModelStatus,
    PromotionStatus,
    HumanVerificationRecord,
    TriggerConfig,
    EvaluationMetrics,
    PromotionGatesConfig,
    ModelRegistryRecord
)
from ml.learnloop.feedback_store import FeedbackStore
from ml.learnloop.trigger import RetrainingTrigger
from ml.learnloop.dataset_versioner import DatasetVersioner
from ml.learnloop.training_runner import CandidateTrainingRunner
from ml.learnloop.evaluator import ChampionChallengerEvaluator
from ml.learnloop.registry import ModelRegistry
from ml.learnloop.audit import AuditLogger
from ml.learnloop.pipeline import LearnLoopOrchestrator


@pytest.fixture
def temp_learnloop_env(tmp_path):
    """Sets up an isolated filesystem environment for LearnLoop tests."""
    feedback_file = str(tmp_path / "feedback" / "verified_feedback.jsonl")
    versions_dir = str(tmp_path / "versions")
    checkpoints_dir = str(tmp_path / "models" / "candidates")
    registry_file = str(tmp_path / "model_registry.json")
    audit_file = str(tmp_path / "audit_trail.jsonl")

    feedback_store = FeedbackStore(storage_path=feedback_file)
    trigger = RetrainingTrigger(TriggerConfig(
        min_verified_samples=10,
        min_new_samples_since_last_training=5,
        min_samples_per_class={"mine_like_contact": 2, "shipwreck": 2},
        max_class_imbalance_ratio=5.0
    ))
    versioner = DatasetVersioner(versions_dir=versions_dir)
    training_runner = CandidateTrainingRunner(checkpoints_dir=checkpoints_dir)
    evaluator = ChampionChallengerEvaluator(PromotionGatesConfig(
        max_recall_drop=0.01,
        max_rare_class_recall_drop=0.02,
        min_f1_improvement=0.00,
        max_fpr_increase=0.00,
        max_latency_ms=45.0,
        max_model_size_mb=25.0,
        require_map50_non_degradation=True
    ))
    registry = ModelRegistry(registry_file=registry_file)
    audit_logger = AuditLogger(log_path=audit_file)

    orchestrator = LearnLoopOrchestrator(
        feedback_store=feedback_store,
        trigger=trigger,
        versioner=versioner,
        training_runner=training_runner,
        evaluator=evaluator,
        registry=registry,
        audit_logger=audit_logger
    )

    yield {
        "orchestrator": orchestrator,
        "feedback_store": feedback_store,
        "trigger": trigger,
        "versioner": versioner,
        "training_runner": training_runner,
        "evaluator": evaluator,
        "registry": registry,
        "audit_logger": audit_logger,
        "tmp_path": tmp_path
    }


class TestLearnLoopFeedbackStore:
    def test_feedback_storage_and_query(self, temp_learnloop_env):
        """Validates adding, querying, and filtering human verification actions."""
        store: FeedbackStore = temp_learnloop_env["feedback_store"]

        # 1. Add Accepted Record
        r1 = HumanVerificationRecord(
            contact_id="CNT-001",
            survey_id="SURVEY-A",
            original_prediction="mine_like_contact",
            original_confidence=0.88,
            original_bbox=[100, 100, 160, 160],
            evidence={"acoustic_score": 0.85},
            model_version="YOLO11N-KD-v3",
            human_action=HumanAction.ACCEPTED,
            reviewer_id="hydrographer_1"
        )
        store.add_feedback(r1)

        # 2. Add Rejected Record (False Positive)
        r2 = HumanVerificationRecord(
            contact_id="CNT-002",
            survey_id="SURVEY-A",
            original_prediction="debris",
            original_confidence=0.62,
            original_bbox=[200, 200, 250, 250],
            evidence={"acoustic_score": 0.30},
            model_version="YOLO11N-KD-v3",
            human_action=HumanAction.REJECTED,
            reviewer_id="hydrographer_1"
        )
        store.add_feedback(r2)

        # 3. Add Corrected Record (Misclassified class)
        r3 = HumanVerificationRecord(
            contact_id="CNT-003",
            survey_id="SURVEY-B",
            original_prediction="debris",
            original_confidence=0.74,
            original_bbox=[300, 300, 420, 380],
            evidence={"acoustic_score": 0.89},
            model_version="YOLO11N-KD-v3",
            human_action=HumanAction.CORRECTED,
            corrected_class="shipwreck",
            reviewer_id="hydrographer_2"
        )
        store.add_feedback(r3)

        # Query tests
        retrieved_1 = store.get_feedback("CNT-001")
        assert retrieved_1 is not None
        assert retrieved_1.human_action == HumanAction.ACCEPTED

        accepted_list = store.list_feedback(action_filter=HumanAction.ACCEPTED)
        assert len(accepted_list) == 1
        assert accepted_list[0].contact_id == "CNT-001"

        stats = store.get_stats()
        assert stats["total_records"] == 3
        assert stats["action_breakdown"]["accepted"] == 1
        assert stats["action_breakdown"]["rejected"] == 1
        assert stats["action_breakdown"]["corrected"] == 1
        assert stats["usable_training_samples"] == 2  # Accepted (mine_like_contact) + Corrected (shipwreck)
        assert stats["class_distribution"]["shipwreck"] == 1
        assert stats["class_distribution"]["mine_like_contact"] == 1


class TestLearnLoopTrigger:
    def test_trigger_threshold_not_reached(self, temp_learnloop_env):
        """Verifies trigger returns False when sample count is below minimum threshold."""
        store: FeedbackStore = temp_learnloop_env["feedback_store"]
        trigger: RetrainingTrigger = temp_learnloop_env["trigger"]

        # Add 3 samples (below config min_verified_samples = 10)
        for i in range(3):
            store.add_feedback(HumanVerificationRecord(
                contact_id=f"CNT-MINE-{i}",
                survey_id="SURVEY-X",
                original_prediction="mine_like_contact",
                original_confidence=0.85,
                original_bbox=[10, 10, 50, 50],
                evidence={},
                model_version="YOLO11N-KD-v3",
                human_action=HumanAction.ACCEPTED
            ))

        result = trigger.evaluate(store)
        assert result.is_triggered is False
        assert "Insufficient total verified samples" in result.reason

    def test_trigger_threshold_reached(self, temp_learnloop_env):
        """Verifies trigger fires when all volume, new sample, and per-class criteria are met."""
        store: FeedbackStore = temp_learnloop_env["feedback_store"]
        trigger: RetrainingTrigger = temp_learnloop_env["trigger"]

        # Add 6 mine_like_contact + 5 shipwreck = 11 samples (>= 10 min)
        for i in range(6):
            store.add_feedback(HumanVerificationRecord(
                contact_id=f"CNT-MINE-{i}",
                survey_id="SURVEY-X",
                original_prediction="mine_like_contact",
                original_confidence=0.85,
                original_bbox=[10, 10, 50, 50],
                evidence={},
                model_version="YOLO11N-KD-v3",
                human_action=HumanAction.ACCEPTED
            ))

        for i in range(5):
            store.add_feedback(HumanVerificationRecord(
                contact_id=f"CNT-WRECK-{i}",
                survey_id="SURVEY-X",
                original_prediction="shipwreck",
                original_confidence=0.92,
                original_bbox=[100, 100, 250, 250],
                evidence={},
                model_version="YOLO11N-KD-v3",
                human_action=HumanAction.ACCEPTED
            ))

        result = trigger.evaluate(store)
        assert result.is_triggered is True
        assert result.total_verified_samples == 11
        assert result.class_counts["mine_like_contact"] == 6
        assert result.class_counts["shipwreck"] == 5


class TestDatasetVersioning:
    def test_dataset_versioning_immutability(self, temp_learnloop_env):
        """Verifies that new dataset versions are isolated manifests and parent dataset is unchanged."""
        store: FeedbackStore = temp_learnloop_env["feedback_store"]
        versioner: DatasetVersioner = temp_learnloop_env["versioner"]

        store.add_feedback(HumanVerificationRecord(
            contact_id="CNT-001",
            survey_id="SURVEY-1",
            original_prediction="shipwreck",
            original_confidence=0.95,
            original_bbox=[10, 10, 50, 50],
            evidence={},
            model_version="v1",
            human_action=HumanAction.ACCEPTED
        ))

        manifest = versioner.create_version(
            feedback_store=store,
            new_version="dataset_v1.1",
            parent_version="dataset_v1.0",
            base_dataset_path="data/dataset_v1.0"
        )

        assert manifest.dataset_version == "dataset_v1.1"
        assert manifest.parent_version == "dataset_v1.0"
        assert manifest.new_sample_count == 1
        assert os.path.exists(manifest.manifest_path)

        loaded = versioner.get_manifest("dataset_v1.1")
        assert loaded is not None
        assert loaded.dataset_version == "dataset_v1.1"
        assert loaded.new_sample_count == 1


class TestChampionChallengerEvaluator:
    @pytest.fixture
    def champion_metrics(self):
        return EvaluationMetrics(
            mAP50=0.870,
            mAP50_95=0.600,
            precision=0.880,
            recall=0.860,
            f1=0.870,
            per_class_recall={
                "mine_like_contact": 0.85,
                "shipwreck": 0.90,
                "airplane_wreck": 0.80,
                "drowning_victim": 0.78,
                "debris": 0.86
            },
            false_positives_per_1000_tiles=2.0,
            latency_ms=28.0,
            model_size_mb=6.1
        )

    def test_challenger_better_all_gates_pass(self, temp_learnloop_env, champion_metrics):
        """Challenger improves mAP, F1, preserves recall and latency -> PROMOTED."""
        evaluator: ChampionChallengerEvaluator = temp_learnloop_env["evaluator"]

        challenger_metrics = EvaluationMetrics(
            mAP50=0.895,
            mAP50_95=0.625,
            precision=0.900,
            recall=0.875,
            f1=0.887,
            per_class_recall={
                "mine_like_contact": 0.87,
                "shipwreck": 0.92,
                "airplane_wreck": 0.82,
                "drowning_victim": 0.80,
                "debris": 0.88
            },
            false_positives_per_1000_tiles=1.8,
            latency_ms=28.5,
            model_size_mb=6.2
        )

        decision = evaluator.evaluate_challenger(
            champion_version="YOLO11N-KD-v3",
            challenger_version="YOLO11N-KD-v4",
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics
        )

        assert decision.promoted is True
        assert decision.decision == "PROMOTED"
        assert all(g.passed for g in decision.gate_results)

    def test_challenger_worse_overall_rejected(self, temp_learnloop_env, champion_metrics):
        """Challenger performs worse across accuracy -> REJECTED."""
        evaluator: ChampionChallengerEvaluator = temp_learnloop_env["evaluator"]

        challenger_metrics = EvaluationMetrics(
            mAP50=0.820,
            mAP50_95=0.540,
            precision=0.810,
            recall=0.800,
            f1=0.805,
            per_class_recall={
                "mine_like_contact": 0.79,
                "shipwreck": 0.82,
                "airplane_wreck": 0.75,
                "drowning_victim": 0.70,
                "debris": 0.80
            },
            false_positives_per_1000_tiles=3.5,
            latency_ms=29.0,
            model_size_mb=6.1
        )

        decision = evaluator.evaluate_challenger(
            champion_version="YOLO11N-KD-v3",
            challenger_version="YOLO11N-KD-v4-bad",
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics
        )

        assert decision.promoted is False
        assert decision.decision == "REJECTED"

    def test_challenger_improves_map_but_hurts_recall(self, temp_learnloop_env, champion_metrics):
        """Challenger achieves higher mAP50 but drops recall below threshold -> REJECTED."""
        evaluator: ChampionChallengerEvaluator = temp_learnloop_env["evaluator"]

        challenger_metrics = EvaluationMetrics(
            mAP50=0.910,                 # High precision/mAP
            mAP50_95=0.640,
            precision=0.950,
            recall=0.820,                # Drops by 0.04 > allowed 0.01 max_recall_drop
            f1=0.880,
            per_class_recall={
                "mine_like_contact": 0.80, # Drops by 0.05 > 0.02
                "shipwreck": 0.90,
                "airplane_wreck": 0.80,
                "drowning_victim": 0.78,
                "debris": 0.86
            },
            false_positives_per_1000_tiles=1.5,
            latency_ms=28.0,
            model_size_mb=6.1
        )

        decision = evaluator.evaluate_challenger(
            champion_version="YOLO11N-KD-v3",
            challenger_version="YOLO11N-KD-v4-high-prec",
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics
        )

        assert decision.promoted is False
        assert decision.decision == "REJECTED"
        recall_gate = next(g for g in decision.gate_results if g.gate_name == "Overall Recall Non-Degradation")
        assert recall_gate.passed is False

    def test_challenger_improves_recall_but_exceeds_latency_budget(self, temp_learnloop_env, champion_metrics):
        """Challenger achieves superior accuracy but latency > 45ms edge budget -> REJECTED."""
        evaluator: ChampionChallengerEvaluator = temp_learnloop_env["evaluator"]

        challenger_metrics = EvaluationMetrics(
            mAP50=0.920,
            mAP50_95=0.660,
            precision=0.930,
            recall=0.910,
            f1=0.920,
            per_class_recall={
                "mine_like_contact": 0.90,
                "shipwreck": 0.94,
                "airplane_wreck": 0.88,
                "drowning_victim": 0.85,
                "debris": 0.90
            },
            false_positives_per_1000_tiles=1.2,
            latency_ms=58.2,             # Exceeds 45.0 ms limit!
            model_size_mb=18.5
        )

        decision = evaluator.evaluate_challenger(
            champion_version="YOLO11N-KD-v3",
            challenger_version="YOLO11N-KD-v4-heavy",
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics
        )

        assert decision.promoted is False
        assert decision.decision == "REJECTED"
        latency_gate = next(g for g in decision.gate_results if g.gate_name == "Edge Compute Latency Budget")
        assert latency_gate.passed is False


class TestModelRegistryAndRollback:
    def test_promotion_and_archive(self, temp_learnloop_env):
        """Validates that champion promotion archives current champion and promotes challenger."""
        registry: ModelRegistry = temp_learnloop_env["registry"]
        evaluator: ChampionChallengerEvaluator = temp_learnloop_env["evaluator"]

        initial_champ = registry.get_champion()
        assert initial_champ is not None
        assert initial_champ.version == "YOLO11N-KD-v3"

        # Register candidate
        candidate = ModelRegistryRecord(
            model_id="cand_v4",
            version="YOLO11N-KD-v4",
            architecture="YOLO11n-KD",
            checkpoint_path="outputs/models/candidates/candidate_v4.pt",
            dataset_version="dataset_v1.1",
            status=ModelStatus.CHALLENGER.value,
            metrics={"mAP50": 0.89},
            latency_ms=28.0,
            size_mb=6.2,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        registry.register_model(candidate)

        # Create passing decision
        champ_metrics = EvaluationMetrics(
            mAP50=0.87, mAP50_95=0.60, precision=0.88, recall=0.86, f1=0.87,
            per_class_recall={"mine_like_contact": 0.85, "shipwreck": 0.90, "airplane_wreck": 0.80, "drowning_victim": 0.78, "debris": 0.86},
            false_positives_per_1000_tiles=2.0, latency_ms=28.0, model_size_mb=6.1
        )
        chall_metrics = EvaluationMetrics(
            mAP50=0.89, mAP50_95=0.62, precision=0.90, recall=0.88, f1=0.89,
            per_class_recall={"mine_like_contact": 0.87, "shipwreck": 0.92, "airplane_wreck": 0.82, "drowning_victim": 0.80, "debris": 0.88},
            false_positives_per_1000_tiles=1.9, latency_ms=28.0, model_size_mb=6.2
        )
        decision = evaluator.evaluate_challenger("YOLO11N-KD-v3", "YOLO11N-KD-v4", champ_metrics, chall_metrics)

        # Promote
        promoted_rec = registry.promote_challenger("cand_v4", decision)
        assert promoted_rec.status == ModelStatus.CHAMPION.value

        # Check that previous champion is now ARCHIVED
        old_champ = registry.get_model(initial_champ.model_id)
        assert old_champ.status == ModelStatus.ARCHIVED.value

        # Check active champion
        current_champ = registry.get_champion()
        assert current_champ.model_id == "cand_v4"

    def test_rollback_to_previous_champion(self, temp_learnloop_env):
        """Validates immediate rollback to previous champion if current model exhibits issues."""
        registry: ModelRegistry = temp_learnloop_env["registry"]
        evaluator: ChampionChallengerEvaluator = temp_learnloop_env["evaluator"]

        initial_champ_id = registry.get_champion().model_id

        # Candidate v4
        cand = ModelRegistryRecord(
            model_id="cand_v4",
            version="YOLO11N-KD-v4",
            architecture="YOLO11n-KD",
            checkpoint_path="outputs/models/candidates/candidate_v4.pt",
            dataset_version="dataset_v1.1",
            status=ModelStatus.CHALLENGER.value,
            metrics={"mAP50": 0.89},
            latency_ms=28.0,
            size_mb=6.2,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        registry.register_model(cand)

        champ_metrics = EvaluationMetrics(
            mAP50=0.87, mAP50_95=0.60, precision=0.88, recall=0.86, f1=0.87,
            per_class_recall={"mine_like_contact": 0.85, "shipwreck": 0.90, "airplane_wreck": 0.80, "drowning_victim": 0.78, "debris": 0.86},
            false_positives_per_1000_tiles=2.0, latency_ms=28.0, model_size_mb=6.1
        )
        chall_metrics = EvaluationMetrics(
            mAP50=0.89, mAP50_95=0.62, precision=0.90, recall=0.88, f1=0.89,
            per_class_recall={"mine_like_contact": 0.87, "shipwreck": 0.92, "airplane_wreck": 0.82, "drowning_victim": 0.80, "debris": 0.88},
            false_positives_per_1000_tiles=1.9, latency_ms=28.0, model_size_mb=6.2
        )
        decision = evaluator.evaluate_challenger("YOLO11N-KD-v3", "YOLO11N-KD-v4", champ_metrics, chall_metrics)
        registry.promote_challenger("cand_v4", decision)

        assert registry.get_champion().model_id == "cand_v4"

        # Execute rollback
        restored = registry.rollback_to_previous_champion("Operator identified domain drift on northern reefs")
        assert restored is not None
        assert restored.model_id == initial_champ_id
        assert restored.status == ModelStatus.CHAMPION.value

        # The v4 model is now ARCHIVED, never deleted
        demoted = registry.get_model("cand_v4")
        assert demoted.status == ModelStatus.ARCHIVED.value


class TestEndToEndLearnLoopPipeline:
    def test_full_cycle_success(self, temp_learnloop_env):
        """Executes full active learning cycle from feedback submission to promotion and audit logging."""
        orchestrator: LearnLoopOrchestrator = temp_learnloop_env["orchestrator"]

        # 1. Submit enough feedback to satisfy gates (6 mines, 5 wrecks)
        for i in range(6):
            orchestrator.submit_feedback(
                contact_dict={
                    "contact_id": f"CNT-MINE-{i}",
                    "survey_id": "SURVEY-ALPHA",
                    "class_name": "mine_like_contact",
                    "raw_detector_confidence": 0.89,
                    "final_confidence": 0.91,
                    "bbox": [50, 50, 100, 100],
                    "evidence": {"acoustic_score": 0.88},
                    "model": {"detector_version": "YOLO11N-KD-v3"}
                },
                human_action="accepted"
            )

        for i in range(5):
            orchestrator.submit_feedback(
                contact_dict={
                    "contact_id": f"CNT-WRECK-{i}",
                    "survey_id": "SURVEY-ALPHA",
                    "class_name": "shipwreck",
                    "raw_detector_confidence": 0.92,
                    "final_confidence": 0.94,
                    "bbox": [150, 150, 300, 300],
                    "evidence": {"acoustic_score": 0.91},
                    "model": {"detector_version": "YOLO11N-KD-v3"}
                },
                human_action="accepted"
            )

        # Candidate metrics passing all gates
        cand_metrics = EvaluationMetrics(
            mAP50=0.895,
            mAP50_95=0.625,
            precision=0.902,
            recall=0.880,
            f1=0.891,
            per_class_recall={
                "mine_like_contact": 0.88,
                "shipwreck": 0.93,
                "airplane_wreck": 0.85,
                "drowning_victim": 0.82,
                "debris": 0.89
            },
            false_positives_per_1000_tiles=1.8,
            latency_ms=28.2,
            model_size_mb=6.2
        )

        res = orchestrator.check_and_run_cycle(
            candidate_version="YOLO11N-KD-v4",
            dataset_version="dataset_v1.1",
            mock_training=True,
            candidate_metrics=cand_metrics
        )

        assert res["executed"] is True
        assert res["status"] == "COMPLETED"
        assert res["active_champion"] == "YOLO11N-KD-v4"
        assert res["promotion_decision"]["promoted"] is True

        # Check audit trail log
        audit_entries = orchestrator.audit_logger.read_all_entries()
        assert len(audit_entries) == 1
        assert audit_entries[0]["decision"] == "PROMOTED"
        assert audit_entries[0]["dataset_version"] == "dataset_v1.1"

    def test_candidate_training_failure(self, temp_learnloop_env):
        """Verifies handling when candidate retraining crashes or runs out of memory."""
        orchestrator: LearnLoopOrchestrator = temp_learnloop_env["orchestrator"]

        # Populate feedback
        for i in range(6):
            orchestrator.submit_feedback(
                contact_dict={"contact_id": f"CNT-MINE-{i}", "class_name": "mine_like_contact", "final_confidence": 0.9, "bbox": [0,0,10,10]},
                human_action="accepted"
            )
        for i in range(5):
            orchestrator.submit_feedback(
                contact_dict={"contact_id": f"CNT-WRECK-{i}", "class_name": "shipwreck", "final_confidence": 0.9, "bbox": [0,0,10,10]},
                human_action="accepted"
            )

        res = orchestrator.check_and_run_cycle(
            candidate_version="YOLO11N-KD-v4-fail",
            dataset_version="dataset_v1.1",
            mock_training=True,
            mock_failure=True
        )

        assert res["executed"] is True
        assert res["status"] == "CANDIDATE_TRAINING_FAILED"
        assert "Simulated training divergence or OOM" in res["error"]

        # Active champion remains YOLO11N-KD-v3
        champ = orchestrator.registry.get_champion()
        assert champ.version == "YOLO11N-KD-v3"
