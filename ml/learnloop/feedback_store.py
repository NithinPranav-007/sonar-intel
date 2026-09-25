"""
Human Verification Feedback Storage Engine.

Persists operator-verified contact feedback (ACCEPTED, REJECTED, CORRECTED, UNCERTAIN)
with complete evidence references, ROI metadata, and model provenance.
"""

import os
import json
import threading
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone

from ml.learnloop.schemas import HumanVerificationRecord, HumanAction


class FeedbackStore:
    """
    Thread-safe, append-only feedback storage manager.
    Persists feedback records to a local JSONL file with memory index for fast lookups.
    """

    def __init__(self, storage_path: str = "data/feedback/verified_feedback.jsonl"):
        self.storage_path = storage_path
        self._lock = threading.Lock()
        self._records: Dict[str, HumanVerificationRecord] = {}
        self._initialized = False
        self._ensure_storage_dir()
        self._load_existing()

    def _ensure_storage_dir(self) -> None:
        directory = os.path.dirname(self.storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _load_existing(self) -> None:
        if not os.path.exists(self.storage_path):
            self._initialized = True
            return

        with self._lock:
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            record = HumanVerificationRecord.from_dict(data)
                            self._records[record.contact_id] = record
            except Exception as e:
                # Log or handle error gracefully
                pass
            self._initialized = True

    def add_feedback(self, record: HumanVerificationRecord) -> HumanVerificationRecord:
        """Adds a verified feedback record to the persistent store."""
        with self._lock:
            self._records[record.contact_id] = record
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        return record

    def add_feedback_from_contact(
        self,
        contact_dict: Dict[str, Any],
        human_action: Union[str, HumanAction],
        corrected_class: Optional[str] = None,
        corrected_bbox: Optional[List[int]] = None,
        reviewer_id: Optional[str] = "operator_default",
        notes: Optional[str] = None
    ) -> HumanVerificationRecord:
        """
        Constructs and stores a HumanVerificationRecord from an existing ContactPackage dictionary.
        """
        if isinstance(human_action, str):
            human_action = HumanAction(human_action.lower())

        model_info = contact_dict.get("model", {})
        model_ver = model_info.get("detector_version", "unknown") if isinstance(model_info, dict) else "unknown"
        assets = contact_dict.get("assets", {})

        record = HumanVerificationRecord(
            contact_id=contact_dict.get("contact_id", "unknown_contact"),
            survey_id=contact_dict.get("survey_id", "unknown_survey"),
            original_prediction=contact_dict.get("class_name", "unknown"),
            original_confidence=float(contact_dict.get("final_confidence", contact_dict.get("raw_detector_confidence", 0.0))),
            original_bbox=contact_dict.get("bbox", [0, 0, 0, 0]),
            evidence=contact_dict.get("evidence", {}),
            model_version=model_ver,
            human_action=human_action,
            roi_ref=assets.get("roi_ref") if isinstance(assets, dict) else None,
            mask_ref=assets.get("mask_ref") if isinstance(assets, dict) else None,
            corrected_class=corrected_class,
            corrected_bbox=corrected_bbox,
            verification_timestamp=datetime.now(timezone.utc).isoformat(),
            reviewer_id=reviewer_id,
            notes=notes
        )
        return self.add_feedback(record)

    def get_feedback(self, contact_id: str) -> Optional[HumanVerificationRecord]:
        """Retrieves a single feedback record by contact_id."""
        with self._lock:
            return self._records.get(contact_id)

    def list_feedback(
        self,
        action_filter: Optional[Union[str, HumanAction]] = None,
        class_filter: Optional[str] = None,
        since_timestamp: Optional[str] = None
    ) -> List[HumanVerificationRecord]:
        """Filters feedback records by action, effective class, or timestamp."""
        with self._lock:
            records = list(self._records.values())

        if isinstance(action_filter, str):
            action_filter = HumanAction(action_filter.lower())

        results = []
        for r in records:
            if action_filter is not None and r.human_action != action_filter:
                continue

            effective_class = r.corrected_class if r.corrected_class else r.original_prediction
            if class_filter is not None and effective_class != class_filter:
                continue

            if since_timestamp is not None and r.verification_timestamp <= since_timestamp:
                continue

            results.append(r)

        return results

    def get_stats(self, since_timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Computes summary statistics including action breakdown and class distributions.
        """
        with self._lock:
            records = list(self._records.values())

        total = len(records)
        filtered = [r for r in records if since_timestamp is None or r.verification_timestamp > since_timestamp]

        action_counts: Dict[str, int] = {}
        class_counts: Dict[str, int] = {}

        for r in filtered:
            act = r.human_action.value
            action_counts[act] = action_counts.get(act, 0) + 1

            # Only usable verified samples for training (accepted or corrected)
            if r.human_action in (HumanAction.ACCEPTED, HumanAction.CORRECTED):
                cls_name = r.corrected_class if (r.human_action == HumanAction.CORRECTED and r.corrected_class) else r.original_prediction
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        return {
            "total_records": total,
            "filtered_records": len(filtered),
            "action_breakdown": action_counts,
            "usable_training_samples": sum(class_counts.values()),
            "class_distribution": class_counts,
            "since_timestamp": since_timestamp
        }

    def clear(self) -> None:
        """Clears all records in memory and disk (primarily for unit tests)."""
        with self._lock:
            self._records.clear()
            if os.path.exists(self.storage_path):
                os.remove(self.storage_path)
