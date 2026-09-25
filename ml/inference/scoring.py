"""
Decision and Priority Scoring Module.

Calculates operational Priority Score:
    priority_score = (
        w_conf * model_confidence +
        w_ctxt * context_score +
        w_qual * data_quality +
        w_loc  * localization_quality
    )

Triage tiers:
    - HIGH   (>= 0.75)
    - MEDIUM (>= 0.50)
    - LOW    (< 0.50)

SCIENTIFIC / DOMAIN HONESTY RULE:
Terminology is strictly 'Priority Score', never 'Probability of Ghost Net'
or 'Probability of Debris'. This score is an operational triage metric.
"""

from typing import Dict, Any, Tuple


class PriorityScorer:
    def __init__(
        self,
        w_confidence: float = 0.50,
        w_context: float = 0.25,
        w_quality: float = 0.15,
        w_localization: float = 0.10,
        high_threshold: float = 0.72,
        medium_threshold: float = 0.48
    ):
        self.w_conf = w_confidence
        self.w_ctxt = w_context
        self.w_qual = w_quality
        self.w_loc = w_localization
        self.high_thresh = high_threshold
        self.med_thresh = medium_threshold

    def calculate_priority(
        self,
        model_confidence: float,
        context_score: float,
        data_quality: float,
        localization_status: str
    ) -> Tuple[str, float]:
        """
        Computes composite priority score and category (HIGH, MEDIUM, LOW).
        
        Args:
            model_confidence: YOLO detector confidence [0.0 - 1.0].
            context_score: Acoustic physics plausibility [0.0 - 1.0].
            data_quality: Sonar swath quality index [0.0 - 1.0].
            localization_status: "VERIFIED", "ESTIMATED", "UNCERTAIN", or "UNAVAILABLE".
            
        Returns:
            Tuple of (priority_label, numeric_score)
        """
        # Map localization status to numerical quality
        loc_map = {
            "VERIFIED": 1.0,
            "ESTIMATED": 0.8,
            "UNCERTAIN": 0.4,
            "UNAVAILABLE": 0.0
        }
        loc_quality = loc_map.get(localization_status.upper(), 0.0)

        # Weighted calculation
        raw_score = (
            (self.w_conf * model_confidence) +
            (self.w_ctxt * context_score) +
            (self.w_qual * data_quality) +
            (self.w_loc * loc_quality)
        )
        score = round(max(0.0, min(1.0, raw_score)), 2)

        if score >= self.high_thresh:
            priority = "HIGH"
        elif score >= self.med_thresh:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        return priority, score
