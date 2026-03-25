"""Tests for the example forecasting trackers."""

from __future__ import annotations

import pytest

from numinous.examples.baseline_tracker import BaselineTracker
from numinous.examples.calibrated_tracker import CalibratedTracker
from numinous.examples.contrarian_tracker import ContrarianTracker
from numinous.examples.keyword_tracker import KeywordTracker
from numinous.examples.market_tracker import MarketTracker


def _make_event_data(
    event_id: str = "poly-123",
    title: str = "Will X happen by 2026-03-16?",
    description: str = "Some description",
    cutoff: str = "2026-03-16T00:00:00Z",
    yes_price: float = 0.65,
    volume_24h: float = 100000.0,
) -> dict:
    return {
        "event_id": event_id,
        "title": title,
        "description": description,
        "cutoff": cutoff,
        "source": "polymarket",
        "yes_price": yes_price,
        "volume_24h": volume_24h,
        "metadata": {"slug": "will-x-happen"},
    }


BULLISH_EVENT = _make_event_data(yes_price=0.75, title="Expected: will pass")
BEARISH_EVENT = _make_event_data(yes_price=0.25, title="Unlikely to happen")
NEUTRAL_EVENT = _make_event_data(yes_price=0.50, title="Coin flip event")

NUMINOUS_EVENT = _make_event_data(
    event_id="numinous-001",
    title="Will there be a drone attack on Maykop in the next 3 days?",
    description="YES if Liveuamap records at least one strike event.",
    yes_price=0.5,
    volume_24h=0.0,
)


@pytest.fixture(
    params=[
        BaselineTracker,
        CalibratedTracker,
        ContrarianTracker,
        KeywordTracker,
        MarketTracker,
    ]
)
def tracker(request):
    """Parametrize over all example trackers."""
    return request.param()


class TestExampleContract:
    """Every example must satisfy the forecast prediction contract."""

    def test_returns_dict_with_prediction(self, tracker):
        tracker.feed_update(BULLISH_EVENT)
        result = tracker.predict(
            "polymarket", resolve_horizon_seconds=259200, step_seconds=1
        )
        assert isinstance(result, dict)
        assert "prediction" in result
        assert isinstance(result["prediction"], (int, float))

    def test_returns_dict_with_event_id(self, tracker):
        tracker.feed_update(BULLISH_EVENT)
        result = tracker.predict(
            "polymarket", resolve_horizon_seconds=259200, step_seconds=1
        )
        assert "event_id" in result
        assert isinstance(result["event_id"], str)

    def test_prediction_in_valid_range(self, tracker):
        tracker.feed_update(BULLISH_EVENT)
        result = tracker.predict(
            "polymarket", resolve_horizon_seconds=259200, step_seconds=1
        )
        assert 0.0 <= result["prediction"] <= 1.0

    def test_no_data_returns_half(self, tracker):
        """Without any feed data, default to 0.5 (maximum uncertainty)."""
        result = tracker.predict(
            "polymarket", resolve_horizon_seconds=259200, step_seconds=1
        )
        assert result["prediction"] == 0.5

    def test_different_prices_produce_different_predictions(self, tracker):
        """At least some trackers should respond to different yes_prices."""
        tracker.feed_update(_make_event_data(yes_price=0.8))
        pred_high = tracker.predict(
            "polymarket", resolve_horizon_seconds=259200, step_seconds=1
        )

        tracker2 = type(tracker)()
        tracker2.feed_update(_make_event_data(yes_price=0.2))
        pred_low = tracker2.predict(
            "polymarket", resolve_horizon_seconds=259200, step_seconds=1
        )

        # BaselineTracker always returns 0.5, so skip the assertion for it
        if not isinstance(tracker, BaselineTracker):
            assert pred_high["prediction"] != pred_low["prediction"]


class TestSpecificModels:
    """Model-specific behavioral tests."""

    def test_baseline_always_half(self):
        tracker = BaselineTracker()
        tracker.feed_update(BULLISH_EVENT)
        result = tracker.predict("polymarket", 259200, 1)
        assert result["prediction"] == 0.5

    def test_market_follows_price(self):
        tracker = MarketTracker()
        tracker.feed_update(_make_event_data(yes_price=0.73))
        result = tracker.predict("polymarket", 259200, 1)
        assert result["prediction"] == 0.73

    def test_contrarian_inverts_price(self):
        tracker = ContrarianTracker()
        tracker.feed_update(_make_event_data(yes_price=0.7))
        result = tracker.predict("polymarket", 259200, 1)
        assert abs(result["prediction"] - 0.3) < 1e-9

    def test_calibrated_shrinks_toward_half(self):
        tracker = CalibratedTracker()
        tracker.feed_update(_make_event_data(yes_price=0.8))
        result = tracker.predict("polymarket", 259200, 1)
        # With alpha=0.8: 0.8 * 0.8 + 0.2 * 0.5 = 0.74
        assert 0.5 < result["prediction"] < 0.8

    def test_keyword_adjusts_for_positive_words(self):
        tracker = KeywordTracker()
        tracker.feed_update(
            _make_event_data(
                yes_price=0.5,
                title="Expected to pass, confirmed by experts",
            )
        )
        result = tracker.predict("polymarket", 259200, 1)
        assert result["prediction"] > 0.5

    def test_keyword_adjusts_for_negative_words(self):
        tracker = KeywordTracker()
        tracker.feed_update(
            _make_event_data(
                yes_price=0.5,
                title="Unlikely to pass, rejected by committee",
            )
        )
        result = tracker.predict("polymarket", 259200, 1)
        assert result["prediction"] < 0.5


class TestNuminousEventCompat:
    """All example models must handle Numinous events (no market price)."""

    @pytest.fixture(
        params=[
            BaselineTracker,
            CalibratedTracker,
            ContrarianTracker,
            KeywordTracker,
            MarketTracker,
        ]
    )
    def tracker(self, request):
        return request.param()

    def test_returns_valid_prediction_for_numinous_event(self, tracker):
        tracker.feed_update(NUMINOUS_EVENT)
        result = tracker.predict("numinous", resolve_horizon_seconds=259200, step_seconds=1)
        assert isinstance(result, dict)
        assert "prediction" in result
        assert 0.0 <= result["prediction"] <= 1.0

    def test_returns_event_id(self, tracker):
        tracker.feed_update(NUMINOUS_EVENT)
        result = tracker.predict("numinous", resolve_horizon_seconds=259200, step_seconds=1)
        assert result.get("event_id") == "numinous-001"
