"""Numinous forecasting: Brier score for binary event predictions.

Participants receive binary events (e.g. "Will X happen by date Y?")
and return a probability between 0.0 and 1.0.

Scoring uses the Brier score: (prediction - outcome)²
- Lower is better: 0 = perfect, 1 = worst possible
- Predictions clipped to [0.01, 0.99] to prevent degenerate edge cases
- Missing predictions imputed as 0.5 → Brier score of 0.25

This is a strictly proper scoring rule — the optimal strategy is to
report your honest probability estimate.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ── Types ────────────────────────────────────────────────────────────


class EventInput(BaseModel):
    """What participants receive: a binary event to forecast.

    Each event is a yes/no question with a resolution deadline.
    """

    event_id: str = ""
    title: str = ""
    description: str = ""
    cutoff: str = ""  # ISO 8601 resolution deadline
    source: str = ""  # e.g. "polymarket"
    yes_price: float = 0.5  # current market price (informational)
    volume_24h: float = 0.0  # recent trading volume
    metadata: dict = Field(default_factory=dict)


class ForecastOutput(BaseModel):
    """What participants return: a probability estimate.

    prediction: float between 0.0 and 1.0
        Probability that the event resolves "Yes".
        0.0 = certain "No", 1.0 = certain "Yes", 0.5 = maximum uncertainty.
    """

    event_id: str = ""
    prediction: float = 0.5


class EventGroundTruth(BaseModel):
    """Binary resolution of an event."""

    model_config = ConfigDict(extra="allow")

    event_id: str = ""
    outcome: int = 0  # 0 = No, 1 = Yes


class BrierScoreResult(BaseModel):
    """Brier score output.

    value = brier_score = (clipped_prediction - outcome)²
    Lower is better: 0.0 = perfect forecast, 1.0 = worst possible.
    """

    model_config = ConfigDict(extra="allow")

    value: float = 0.0  # = brier_score, used for aggregation
    brier_score: float = 0.0
    clipped_prediction: float = 0.5
    outcome: int = 0
    success: bool = True
    failed_reason: str | None = None


# ── Scoring ──────────────────────────────────────────────────────────

CLIP_EPS = 0.01  # clip predictions to [0.01, 0.99]


def score_prediction(
    prediction: ForecastOutput,
    ground_truth: EventGroundTruth,
) -> BrierScoreResult:
    """Brier score: (prediction - outcome)².

    Strictly proper scoring rule — identical to Numinous Subnet 6.
    Predictions are clipped to [0.01, 0.99] to avoid degenerate scores.

    Args:
        prediction: Model's probability estimate (ForecastOutput).
        ground_truth: Binary outcome (EventGroundTruth).

    Returns:
        BrierScoreResult with the computed Brier score.
    """
    outcome = ground_truth.outcome

    # Validate outcome is binary
    if outcome not in (0, 1):
        return BrierScoreResult(
            success=False,
            failed_reason=f"outcome must be 0 or 1, got {outcome}",
        )

    # Clip prediction to [CLIP_EPS, 1 - CLIP_EPS]
    raw_pred = prediction.prediction
    clipped = max(CLIP_EPS, min(1.0 - CLIP_EPS, raw_pred))

    # Brier score
    brier = (clipped - outcome) ** 2

    return BrierScoreResult(
        value=brier,
        brier_score=brier,
        clipped_prediction=clipped,
        outcome=outcome,
    )
