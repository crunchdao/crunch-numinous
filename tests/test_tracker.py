"""Tests for TrackerBase predict interface."""

from __future__ import annotations
from typing import Any

import pytest

from numinous.tracker import TrackerBase


class DummyTracker(TrackerBase):
    """Minimal implementation for testing."""

    def _predict(self, event: dict) -> dict:
        return {
            "event_id": event.get("event_id", "unknown"),
            "prediction": 0.7 if event else 0.5,
        }


class MockTracker(TrackerBase):
    """Minimal implementation for testing."""

    def __init__(self):
        super().__init__()

        self.next_prediction: Any = None

    def _predict(self, event: dict) -> dict:
        return self.next_prediction # pyright: ignore[reportReturnType]


class TestPredict:
    """predict() delegates to _predict() and returns the result."""

    def test_predict_returns_result(self):
        tracker = DummyTracker()
        result = tracker.predict({"event_id": "e1", "title": "Event 1"})

        assert result["event_id"] == "e1"
        assert result["prediction"] == 0.7

    def test_predict_with_empty_event(self):
        tracker = DummyTracker()
        result = tracker.predict({})

        assert result["event_id"] == "unknown"
        assert result["prediction"] == 0.5

    def test_predict_passes_event_to_implementation(self):
        """The event dict is forwarded as-is to _predict."""
        received = {}

        class CapturingTracker(TrackerBase):
            def _predict(self, event: dict) -> dict:
                received.update(event)
                return {"event_id": event.get("event_id", "x"), "prediction": 0.5}

        tracker = CapturingTracker()
        tracker.predict({"event_id": "e1", "symbol": "BTC", "price": 100})

        assert received["event_id"] == "e1"
        assert received["symbol"] == "BTC"
        assert received["price"] == 100

    def test_predict_multiple_events(self):
        tracker = DummyTracker()
        r1 = tracker.predict({"event_id": "e1", "title": "Event 1"})
        r2 = tracker.predict({"event_id": "e2", "title": "Event 2"})

        assert r1["event_id"] == "e1"
        assert r2["event_id"] == "e2"


class TestPredictBase:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            TrackerBase() # pyright: ignore[reportAbstractUsage]

    def test_subclass_must_implement_predict(self):
        class IncompleteTracker(TrackerBase):
            pass

        with pytest.raises(TypeError):
            IncompleteTracker() # pyright: ignore[reportAbstractUsage]


class TestWrongPrediction:
    def test_return_none(self):
        tracker = MockTracker()
        tracker.next_prediction = None

        with pytest.raises(ValueError, match="method must return a dict"):
            tracker.predict({})

    def test_return_non_dict(self):
        tracker = MockTracker()
        tracker.next_prediction = "not a dict"

        with pytest.raises(ValueError, match="method must return a dict"):
            tracker.predict({})

    def test_return_non_float_prediction(self):
        tracker = MockTracker()
        tracker.next_prediction = {"event_id": "e1", "prediction": "not a float"}

        with pytest.raises(ValueError, match="prediction must be a float, but got str"):
            tracker.predict({})
