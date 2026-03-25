"""Tests for Brier score scoring function.

Brier score = (prediction - outcome)²
Lower is better: 0.0 = perfect, 1.0 = worst possible.
Predictions are clipped to [0.01, 0.99].

All tests use Pydantic models — the engine always coerces to typed objects.
"""

from __future__ import annotations

from numinous.scoring import (
    CLIP_EPS,
    BrierScoreResult,
    EventGroundTruth,
    ForecastOutput,
    score_prediction,
)


class TestScoringContract:
    """Shape/type requirements — must pass for ANY valid implementation."""

    def test_returns_pydantic_model(self):
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.7),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        assert isinstance(result, BrierScoreResult)

    def test_has_value_field(self):
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.7),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        assert isinstance(result.value, (int, float))

    def test_has_success_field(self):
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.7),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        assert isinstance(result.success, bool)

    def test_has_failed_reason_field(self):
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.7),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        assert result.failed_reason is None or isinstance(result.failed_reason, str)

    def test_has_brier_score_field(self):
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.7),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        assert isinstance(result.brier_score, float)

    def test_has_clipped_prediction_field(self):
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.7),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        assert isinstance(result.clipped_prediction, float)


class TestBrierScoring:
    """Behavioral tests for Brier scoring rule."""

    def test_perfect_yes(self):
        """Predicting 0.99 when outcome=1 → near-zero Brier."""
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.99),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        assert result.brier_score == (0.99 - 1) ** 2
        assert result.brier_score < 0.01

    def test_perfect_no(self):
        """Predicting 0.01 when outcome=0 → near-zero Brier."""
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.01),
            EventGroundTruth(event_id="e1", outcome=0),
        )
        assert result.brier_score == (0.01 - 0) ** 2
        assert result.brier_score < 0.01

    def test_worst_case_yes(self):
        """Predicting 0.01 when outcome=1 → near-maximum Brier."""
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.01),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        assert result.brier_score == (CLIP_EPS - 1) ** 2
        assert result.brier_score > 0.95

    def test_worst_case_no(self):
        """Predicting 0.99 when outcome=0 → near-maximum Brier."""
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.99),
            EventGroundTruth(event_id="e1", outcome=0),
        )
        assert result.brier_score == (1 - CLIP_EPS - 0) ** 2
        assert result.brier_score > 0.95

    def test_uncertain_score(self):
        """Predicting 0.5 → Brier = 0.25 regardless of outcome."""
        for outcome in (0, 1):
            result = score_prediction(
                ForecastOutput(event_id="e1", prediction=0.5),
                EventGroundTruth(event_id="e1", outcome=outcome),
            )
            assert result.brier_score == 0.25

    def test_value_equals_brier_score(self):
        """value field equals brier_score for aggregation."""
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.7),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        assert result.value == result.brier_score

    def test_clipping_low(self):
        """Predictions below 0.01 are clipped to 0.01."""
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.0),
            EventGroundTruth(event_id="e1", outcome=0),
        )
        assert result.clipped_prediction == CLIP_EPS
        assert result.brier_score == CLIP_EPS**2

    def test_clipping_high(self):
        """Predictions above 0.99 are clipped to 0.99."""
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=1.0),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        assert result.clipped_prediction == 1 - CLIP_EPS
        assert result.brier_score == (1 - CLIP_EPS - 1) ** 2

    def test_normal_not_clipped(self):
        """Predictions in [0.01, 0.99] are not clipped."""
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.65),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        assert result.clipped_prediction == 0.65

    def test_outcome_preserved(self):
        """Outcome is preserved in the result."""
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.7),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        assert result.outcome == 1

    def test_invalid_outcome_fails(self):
        """Non-binary outcome returns failure."""
        result = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.5),
            EventGroundTruth(event_id="e1", outcome=2),
        )
        assert result.success is False
        assert result.failed_reason is not None

    def test_default_output_scores_quarter(self):
        """Default ForecastOutput (prediction=0.5) scores 0.25."""
        result = score_prediction(
            ForecastOutput(),
            EventGroundTruth(outcome=0),
        )
        assert result.brier_score == 0.25
        assert result.success is True

    def test_lower_is_better(self):
        """More confident correct prediction scores lower (better)."""
        confident = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.9),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        uncertain = score_prediction(
            ForecastOutput(event_id="e1", prediction=0.6),
            EventGroundTruth(event_id="e1", outcome=1),
        )
        assert confident.brier_score < uncertain.brier_score
