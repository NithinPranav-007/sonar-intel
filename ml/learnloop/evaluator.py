"""
Champion / Challenger Model Evaluator & Promotion Gate Engine.

Executes rigorous multi-dimensional evaluation comparing a Challenger candidate
against the deployed Champion model across accuracy, safety, rare-class recall,
false-positive rate, and edge-compute latency budgets.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from ml.learnloop.schemas import (
    EvaluationMetrics,
    PromotionGatesConfig,
    GateEvaluationResult,
    PromotionDecision,
    PromotionStatus
)


class ChampionChallengerEvaluator:
    """
    Evaluates Challenger models against the active Champion.
    Enforces that candidate promotion requires passing ALL configured safety and operational gates.
    """

    def __init__(self, gates_config: Optional[PromotionGatesConfig] = None):
        self.gates_config = gates_config or PromotionGatesConfig()

    def evaluate_challenger(
        self,
        champion_version: str,
        challenger_version: str,
        champion_metrics: EvaluationMetrics,
        challenger_metrics: EvaluationMetrics
    ) -> PromotionDecision:
        """
        Runs comprehensive gate evaluations comparing Challenger to Champion.

        Args:
            champion_version: Version identifier of the active Champion.
            challenger_version: Version identifier of the Challenger candidate.
            champion_metrics: Benchmarked metrics for the Champion.
            challenger_metrics: Benchmarked metrics for the Challenger candidate.

        Returns:
            PromotionDecision recording promotion status, gate evaluations, and audit reasons.
        """
        gate_results: List[GateEvaluationResult] = []
        all_passed = True
        failure_reasons: List[str] = []

        # Gate 1: Overall Recall Degradation Gate
        recall_diff = challenger_metrics.recall - champion_metrics.recall
        recall_drop = -recall_diff
        passed_recall = recall_drop <= self.gates_config.max_recall_drop
        gate_results.append(GateEvaluationResult(
            gate_name="Overall Recall Non-Degradation",
            passed=passed_recall,
            champion_value=champion_metrics.recall,
            challenger_value=challenger_metrics.recall,
            threshold=f"Drop <= {self.gates_config.max_recall_drop:.3f} (Diff: {recall_diff:+.3f})",
            detail="Passed recall budget" if passed_recall else f"Recall dropped by {recall_drop:.3f} > allowed {self.gates_config.max_recall_drop:.3f}"
        ))
        if not passed_recall:
            all_passed = False
            failure_reasons.append(f"Recall dropped by {recall_drop:.3f}")

        # Gate 2: Rare-Class / Per-Class Recall Degradation Gate
        per_class_passed = True
        failed_classes = []
        for cls_name, champ_r in champion_metrics.per_class_recall.items():
            chall_r = challenger_metrics.per_class_recall.get(cls_name, 0.0)
            class_drop = champ_r - chall_r
            if class_drop > self.gates_config.max_rare_class_recall_drop:
                per_class_passed = False
                failed_classes.append(f"{cls_name} drop={class_drop:.3f}")

        gate_results.append(GateEvaluationResult(
            gate_name="Per-Class Recall Non-Degradation",
            passed=per_class_passed,
            champion_value=champion_metrics.per_class_recall,
            challenger_value=challenger_metrics.per_class_recall,
            threshold=f"Max drop per class <= {self.gates_config.max_rare_class_recall_drop:.3f}",
            detail="All class recalls preserved" if per_class_passed else f"Class drops: {', '.join(failed_classes)}"
        ))
        if not per_class_passed:
            all_passed = False
            failure_reasons.append(f"Per-class recall degradation: {', '.join(failed_classes)}")

        # Gate 3: F1 Score Maintenance / Improvement
        f1_diff = challenger_metrics.f1 - champion_metrics.f1
        passed_f1 = f1_diff >= self.gates_config.min_f1_improvement
        gate_results.append(GateEvaluationResult(
            gate_name="F1 Score Maintenance",
            passed=passed_f1,
            champion_value=champion_metrics.f1,
            challenger_value=challenger_metrics.f1,
            threshold=f"Improvement >= {self.gates_config.min_f1_improvement:+.3f} (Diff: {f1_diff:+.3f})",
            detail="F1 maintained or improved" if passed_f1 else f"F1 degraded by {-f1_diff:.3f}"
        ))
        if not passed_f1:
            all_passed = False
            failure_reasons.append(f"F1 score degraded by {-f1_diff:.3f}")

        # Gate 4: False-Positive Rate Constraint
        fpr_diff = challenger_metrics.false_positives_per_1000_tiles - champion_metrics.false_positives_per_1000_tiles
        passed_fpr = fpr_diff <= self.gates_config.max_fpr_increase
        gate_results.append(GateEvaluationResult(
            gate_name="False-Positive Rate Constraint",
            passed=passed_fpr,
            champion_value=champion_metrics.false_positives_per_1000_tiles,
            challenger_value=challenger_metrics.false_positives_per_1000_tiles,
            threshold=f"Increase <= {self.gates_config.max_fpr_increase:+.3f} FP/1000 tiles (Diff: {fpr_diff:+.3f})",
            detail="FPR constraint met" if passed_fpr else f"FPR increased by {fpr_diff:+.3f} FP/1000 tiles"
        ))
        if not passed_fpr:
            all_passed = False
            failure_reasons.append(f"FPR increased by {fpr_diff:+.3f}")

        # Gate 5: Edge Latency Budget
        passed_latency = challenger_metrics.latency_ms <= self.gates_config.max_latency_ms
        gate_results.append(GateEvaluationResult(
            gate_name="Edge Compute Latency Budget",
            passed=passed_latency,
            champion_value=champion_metrics.latency_ms,
            challenger_value=challenger_metrics.latency_ms,
            threshold=f"Max allowable latency <= {self.gates_config.max_latency_ms:.1f} ms",
            detail=f"Latency {challenger_metrics.latency_ms:.1f} ms is within budget" if passed_latency else f"Latency {challenger_metrics.latency_ms:.1f} ms exceeds budget {self.gates_config.max_latency_ms:.1f} ms"
        ))
        if not passed_latency:
            all_passed = False
            failure_reasons.append(f"Latency ({challenger_metrics.latency_ms:.1f} ms) exceeds edge budget ({self.gates_config.max_latency_ms:.1f} ms)")

        # Gate 6: Model Size Budget
        passed_size = challenger_metrics.model_size_mb <= self.gates_config.max_model_size_mb
        gate_results.append(GateEvaluationResult(
            gate_name="Model Size Budget",
            passed=passed_size,
            champion_value=champion_metrics.model_size_mb,
            challenger_value=challenger_metrics.model_size_mb,
            threshold=f"Max size <= {self.gates_config.max_model_size_mb:.1f} MB",
            detail=f"Size {challenger_metrics.model_size_mb:.1f} MB is within budget" if passed_size else f"Size {challenger_metrics.model_size_mb:.1f} MB exceeds budget"
        ))
        if not passed_size:
            all_passed = False
            failure_reasons.append(f"Model size ({challenger_metrics.model_size_mb:.1f} MB) exceeds budget")

        # Gate 7: mAP50 Non-Degradation (if required)
        if self.gates_config.require_map50_non_degradation:
            passed_map = challenger_metrics.mAP50 >= champion_metrics.mAP50
            gate_results.append(GateEvaluationResult(
                gate_name="mAP50 Non-Degradation",
                passed=passed_map,
                champion_value=champion_metrics.mAP50,
                challenger_value=challenger_metrics.mAP50,
                threshold="mAP50 >= Champion mAP50",
                detail="mAP50 maintained or improved" if passed_map else f"mAP50 dropped by {champion_metrics.mAP50 - challenger_metrics.mAP50:.3f}"
            ))
            if not passed_map:
                all_passed = False
                failure_reasons.append(f"mAP50 degraded from {champion_metrics.mAP50:.3f} to {challenger_metrics.mAP50:.3f}")

        decision_str = PromotionStatus.PROMOTED.value if all_passed else PromotionStatus.REJECTED.value
        summary_reason = "All promotion gates passed successfully" if all_passed else f"Challenger failed promotion gates: {'; '.join(failure_reasons)}"

        return PromotionDecision(
            promoted=all_passed,
            decision=decision_str,
            decision_timestamp=datetime.now(timezone.utc).isoformat(),
            champion_version=champion_version,
            challenger_version=challenger_version,
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            gate_results=gate_results,
            reason=summary_reason
        )
