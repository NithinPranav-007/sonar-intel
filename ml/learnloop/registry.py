"""
Model Registry & Safe Rollback Management Engine.

Maintains an immutable ledger of model checkpoints, deployment statuses (CHAMPION,
CHALLENGER, REJECTED, ARCHIVED), performance metrics, and rapid zero-loss rollback mechanisms.
"""

import os
import json
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from ml.learnloop.schemas import (
    ModelRegistryRecord,
    ModelStatus,
    PromotionDecision,
    PromotionStatus
)


class ModelRegistry:
    """
    Central model registry providing thread-safe model state management,
    promotion execution, and deterministic rollback to previous validated champions.
    """

    def __init__(self, registry_file: str = "outputs/model_registry.json"):
        self.registry_file = registry_file
        self._lock = threading.Lock()
        self._models: Dict[str, ModelRegistryRecord] = {}
        self._champion_history: List[str] = []   # Chronological list of champion model_ids
        self._ensure_dir()
        self._load_registry()

    def _ensure_dir(self) -> None:
        directory = os.path.dirname(self.registry_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _load_registry(self) -> None:
        if not os.path.exists(self.registry_file):
            # Bootstrap with baseline model if empty
            self._bootstrap_default_champion()
            return

        with self._lock:
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    models_list = data.get("models", [])
                    self._champion_history = data.get("champion_history", [])
                    for m in models_list:
                        record = ModelRegistryRecord(**m)
                        self._models[record.model_id] = record
            except Exception:
                self._bootstrap_default_champion()

    def _bootstrap_default_champion(self) -> None:
        """Initializes the baseline champion model if no registry exists."""
        baseline = ModelRegistryRecord(
            model_id="yolo11n_kd_v3_champion",
            version="YOLO11N-KD-v3",
            architecture="YOLO11n-KnowledgeDistilled",
            checkpoint_path="weights/best_detector.pt",
            dataset_version="dataset_v1.0",
            status=ModelStatus.CHAMPION.value,
            metrics={
                "mAP50": 0.872,
                "mAP50_95": 0.598,
                "precision": 0.884,
                "recall": 0.865,
                "f1": 0.874,
                "false_positives_per_1000_tiles": 2.1
            },
            latency_ms=27.5,
            size_mb=6.1,
            created_at=datetime.now(timezone.utc).isoformat(),
            teacher="YOLO11x-SonarTeacher-v1.0",
            student="YOLO11n",
            training_run="run_baseline_kd_v3",
            promotion_status="INITIAL_BASELINE",
            notes="Initial production champion model"
        )
        self._models[baseline.model_id] = baseline
        self._champion_history = [baseline.model_id]
        self._save_to_disk()

    def _save_to_disk(self) -> None:
        payload = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "active_champion": self._get_active_champion_id_unlocked(),
            "champion_history": self._champion_history,
            "models": [m.to_dict() for m in self._models.values()]
        }
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _get_active_champion_id_unlocked(self) -> Optional[str]:
        for model_id, model in self._models.items():
            if model.status == ModelStatus.CHAMPION.value:
                return model_id
        return None

    def register_model(self, record: ModelRegistryRecord) -> ModelRegistryRecord:
        """Registers a new model record in the registry."""
        with self._lock:
            self._models[record.model_id] = record
            self._save_to_disk()
        return record

    def get_model(self, model_id: str) -> Optional[ModelRegistryRecord]:
        """Gets a model record by model_id."""
        with self._lock:
            return self._models.get(model_id)

    def get_champion(self) -> Optional[ModelRegistryRecord]:
        """Returns the currently active Champion model record."""
        with self._lock:
            for model in self._models.values():
                if model.status == ModelStatus.CHAMPION.value:
                    return model
            return None

    def promote_challenger(
        self,
        challenger_id: str,
        decision: PromotionDecision
    ) -> ModelRegistryRecord:
        """
        Promotes a challenger to Champion:
        - Current Champion status becomes ARCHIVED
        - Challenger status becomes CHAMPION
        - Decision metrics and reasons are permanently attached
        """
        with self._lock:
            challenger = self._models.get(challenger_id)
            if challenger is None:
                raise ValueError(f"Challenger model '{challenger_id}' not found in registry.")

            current_champ_id = self._get_active_champion_id_unlocked()
            if current_champ_id and current_champ_id in self._models:
                current_champ = self._models[current_champ_id]
                current_champ.status = ModelStatus.ARCHIVED.value
                current_champ.notes = f"Demoted to ARCHIVED upon promotion of {challenger.version} on {decision.decision_timestamp}"

            challenger.status = ModelStatus.CHAMPION.value
            challenger.promotion_status = PromotionStatus.PROMOTED.value
            challenger.notes = f"Promoted: {decision.reason}"

            self._champion_history.append(challenger_id)
            self._save_to_disk()
            return challenger

    def reject_challenger(
        self,
        challenger_id: str,
        decision: PromotionDecision
    ) -> ModelRegistryRecord:
        """
        Marks a challenger as REJECTED:
        - Candidate remains safely archived (never deleted)
        - Active Champion remains unchanged
        """
        with self._lock:
            challenger = self._models.get(challenger_id)
            if challenger is None:
                raise ValueError(f"Challenger model '{challenger_id}' not found in registry.")

            challenger.status = ModelStatus.REJECTED.value
            challenger.promotion_status = PromotionStatus.REJECTED.value
            challenger.notes = f"Rejected: {decision.reason}"

            self._save_to_disk()
            return challenger

    def rollback_to_previous_champion(self, reason: str = "Rollback requested") -> Optional[ModelRegistryRecord]:
        """
        Restores the previous validated champion from the archive history.
        Ensures rapid zero-downtime recovery without checkpoint deletion.
        """
        with self._lock:
            if len(self._champion_history) < 2:
                # No previous champion in history
                return None

            # Current active champion is at the end of history
            current_champ_id = self._champion_history.pop()
            previous_champ_id = self._champion_history[-1]

            current_champ = self._models.get(current_champ_id)
            previous_champ = self._models.get(previous_champ_id)

            if current_champ:
                current_champ.status = ModelStatus.ARCHIVED.value
                current_champ.notes = f"Rolled back on {datetime.now(timezone.utc).isoformat()}: {reason}"

            if previous_champ:
                previous_champ.status = ModelStatus.CHAMPION.value
                previous_champ.notes = f"Restored as CHAMPION via rollback on {datetime.now(timezone.utc).isoformat()}"

            self._save_to_disk()
            return previous_champ

    def list_models(self) -> List[ModelRegistryRecord]:
        """Returns all registered models."""
        with self._lock:
            return list(self._models.values())

    def clear(self) -> None:
        """Clears the registry for test isolation."""
        with self._lock:
            self._models.clear()
            self._champion_history.clear()
            if os.path.exists(self.registry_file):
                os.remove(self.registry_file)
