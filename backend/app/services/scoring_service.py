"""
Scoring Bridge Service for Backend.
"""

from ml.inference.scoring import PriorityScorer

scorer_instance = PriorityScorer()


def calculate_contact_priority(
    confidence: float,
    context_score: float,
    data_quality: float,
    localization_status: str
):
    """Calculates operational priority label ('HIGH', 'MEDIUM', 'LOW') and score."""
    return scorer_instance.calculate_priority(
        model_confidence=confidence,
        context_score=context_score,
        data_quality=data_quality,
        localization_status=localization_status
    )
