"""
Automated Retraining Trigger and Gate Evaluator.

Evaluates multi-variable gating conditions on accumulated human-verified feedback
to decide when candidate model retraining should be initiated.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone

from ml.learnloop.schemas import TriggerConfig, TriggerResult, HumanAction
from ml.learnloop.feedback_store import FeedbackStore


class RetrainingTrigger:
    """
    Evaluates configurable gates against stored feedback samples.
    Triggers candidate training jobs only when all quality and volume criteria are met.
    """

    def __init__(self, config: Optional[TriggerConfig] = None):
        self.config = config or TriggerConfig()

    def evaluate(
        self,
        feedback_store: FeedbackStore,
        last_training_timestamp: Optional[str] = None
    ) -> TriggerResult:
        """
        Evaluates current feedback volume against configured trigger gates.

        Args:
            feedback_store: The active feedback store.
            last_training_timestamp: Timestamp of the last candidate training cycle.

        Returns:
            TriggerResult detailing whether retraining is triggered and the reason.
        """
        stats_all = feedback_store.get_stats(since_timestamp=None)
        stats_new = feedback_store.get_stats(since_timestamp=last_training_timestamp)

        total_usable = stats_all["usable_training_samples"]
        new_usable = stats_new["usable_training_samples"]
        class_dist = stats_new["class_distribution"]

        # Gate 1: Minimum total verified samples
        if total_usable < self.config.min_verified_samples:
            return TriggerResult(
                is_triggered=False,
                reason=f"Insufficient total verified samples: {total_usable}/{self.config.min_verified_samples}",
                total_verified_samples=total_usable,
                new_samples_count=new_usable,
                class_counts=class_dist
            )

        # Gate 2: Minimum new samples since last training run
        if new_usable < self.config.min_new_samples_since_last_training:
            return TriggerResult(
                is_triggered=False,
                reason=f"Insufficient new verified samples since last training: {new_usable}/{self.config.min_new_samples_since_last_training}",
                total_verified_samples=total_usable,
                new_samples_count=new_usable,
                class_counts=class_dist
            )

        # Gate 3: Per-class minimum sample requirements
        for cls_name, min_req in self.config.min_samples_per_class.items():
            count = class_dist.get(cls_name, 0)
            if count < min_req:
                return TriggerResult(
                    is_triggered=False,
                    reason=f"Class '{cls_name}' count {count} is below requirement {min_req}",
                    total_verified_samples=total_usable,
                    new_samples_count=new_usable,
                    class_counts=class_dist
                )

        # Gate 4: Class imbalance constraint (optional)
        if self.config.max_class_imbalance_ratio is not None and len(class_dist) > 1:
            counts = list(class_dist.values())
            min_c = min(counts) if min(counts) > 0 else 1
            max_c = max(counts)
            imbalance_ratio = max_c / min_c
            if imbalance_ratio > self.config.max_class_imbalance_ratio:
                return TriggerResult(
                    is_triggered=False,
                    reason=f"Class imbalance ratio {imbalance_ratio:.2f} exceeds limit {self.config.max_class_imbalance_ratio:.2f}",
                    total_verified_samples=total_usable,
                    new_samples_count=new_usable,
                    class_counts=class_dist
                )

        return TriggerResult(
            is_triggered=True,
            reason="All retraining trigger gates satisfied",
            total_verified_samples=total_usable,
            new_samples_count=new_usable,
            class_counts=class_dist
        )
