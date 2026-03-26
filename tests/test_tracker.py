"""Tests for TrackerBase subject-keyed data storage."""

from __future__ import annotations

import pytest
from numinous.tracker import TrackerBase


class DummyTracker(TrackerBase):
    """Minimal implementation for testing."""

    def _predict(self, subject: str) -> dict:
        data = self._get_data(subject)
        if not isinstance(data, dict):
            return {"event_id": subject, "prediction": 0.5}
        return {
            "event_id": data.get("event_id", subject),
            "prediction": 0.7 if data else 0.5,
        }


class TestFeedUpdateSubjectKeying:
    """feed_update() must store data per-subject so multi-event works."""

    def test_stores_data_by_event_id(self):
        tracker = DummyTracker()
        tracker.feed_update({"event_id": "e1", "title": "Event 1", "yes_price": 0.7})
        tracker.feed_update({"event_id": "e2", "title": "Event 2", "yes_price": 0.3})

        assert tracker._get_data("e1")["yes_price"] == 0.7
        assert tracker._get_data("e2")["yes_price"] == 0.3

    def test_stores_data_by_symbol(self):
        tracker = DummyTracker()
        tracker.feed_update({"symbol": "BTC", "price": 100})
        tracker.feed_update({"symbol": "ETH", "price": 50})

        assert tracker._get_data("BTC")["price"] == 100
        assert tracker._get_data("ETH")["price"] == 50

    def test_updates_same_event(self):
        tracker = DummyTracker()
        tracker.feed_update({"event_id": "e1", "yes_price": 0.5})
        tracker.feed_update({"event_id": "e1", "yes_price": 0.8})

        assert tracker._get_data("e1")["yes_price"] == 0.8

    def test_default_always_updated(self):
        """_default is always updated to the latest data."""
        tracker = DummyTracker()
        tracker.feed_update({"event_id": "e1", "yes_price": 0.5})
        tracker.feed_update({"event_id": "e2", "yes_price": 0.8})

        # _default should be the latest event (e2)
        assert tracker._get_data("_default")["event_id"] == "e2"


class TestGetDataFallback:
    """_get_data() falls back to _default when no exact match."""

    def test_falls_back_to_default_for_unknown_subject(self):
        """Unknown subject falls back to _default (latest event)."""
        tracker = DummyTracker()
        tracker.feed_update({"event_id": "e1", "title": "Latest"})

        # "polymarket" subject not stored directly, but _default has data
        data = tracker._get_data("polymarket")
        assert data is not None
        assert data["event_id"] == "e1"

    def test_exact_match_takes_priority(self):
        tracker = DummyTracker()
        tracker.feed_update({"event_id": "e1", "title": "Event 1"})
        tracker.feed_update({"event_id": "e2", "title": "Event 2"})

        # Exact match for e1 should return e1, not _default (which is e2)
        assert tracker._get_data("e1")["title"] == "Event 1"
        assert tracker._get_data("e2")["title"] == "Event 2"

    def test_returns_none_when_empty(self):
        tracker = DummyTracker()
        assert tracker._get_data("anything") is None


class TestFeedUpdateEdgeCases:
    """Edge cases for feed_update() input."""

    def test_non_dict_data_stored_as_default(self):
        tracker = DummyTracker()
        tracker.feed_update("not a dict")  # type: ignore[arg-type]
        # Should not crash; stored under _default
        assert tracker._get_data("anything") == "not a dict"

    def test_no_event_id_no_symbol_stored_as_default(self):
        tracker = DummyTracker()
        tracker.feed_update({"price": 42})
        assert tracker._get_data("anything")["price"] == 42


class TestPredictBase:
    def test_not_implemented_on_base(self):
        tracker = TrackerBase()
        with pytest.raises(NotImplementedError):
            tracker.predict("test", 60, 15)
