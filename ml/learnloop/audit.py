"""
LearnLoop Audit Trail & Compliance Logger.

Maintains an immutable, append-only ledger of every retraining cycle, gate evaluation,
promotion decision, and active model state.
"""

import os
import json
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import uuid

from ml.learnloop.schemas import AuditLogEntry, PromotionDecision, DatasetManifest


class AuditLogger:
    """
    Immutable audit logging facility for model lifecycle operations.
    """

    def __init__(self, log_path: str = "outputs/learnloop_audit_trail.jsonl"):
        self.log_path = log_path
        self._lock = threading.Lock()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        directory = os.path.dirname(self.log_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def log_cycle(
        self,
        trigger_reason: str,
        verified_sample_count: int,
        dataset_manifest: DatasetManifest,
        candidate_checkpoint: str,
        decision: PromotionDecision,
        active_champion_after_cycle: str,
        training_config: Optional[Dict[str, Any]] = None
    ) -> AuditLogEntry:
        """
        Appends a complete, auditable LearnLoop cycle record to the persistent log.
        """
        entry = AuditLogEntry(
            cycle_id=f"LL-CYCLE-{uuid.uuid4().hex[:8].upper()}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            trigger_reason=trigger_reason,
            verified_sample_count=verified_sample_count,
            dataset_version=dataset_manifest.dataset_version,
            training_config=training_config or {"framework": "YOLO11-KD", "epochs": 100},
            candidate_checkpoint=candidate_checkpoint,
            evaluation_metrics=decision.challenger_metrics.to_dict(),
            gate_results=[g.to_dict() for g in decision.gate_results],
            decision=decision.decision,
            active_champion_after_cycle=active_champion_after_cycle
        )

        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")

        return entry

    def read_all_entries(self) -> List[Dict[str, Any]]:
        """Reads all audit entries from disk."""
        if not os.path.exists(self.log_path):
            return []

        entries = []
        with self._lock:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        return entries

    def clear(self) -> None:
        """Clears audit logs (for test teardowns)."""
        with self._lock:
            if os.path.exists(self.log_path):
                os.remove(self.log_path)
