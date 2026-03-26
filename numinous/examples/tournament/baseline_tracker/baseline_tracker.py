"""Baseline: always predict 0.5 (maximum uncertainty).

This is the simplest possible model — no information used.
With Brier scoring, this produces a guaranteed score of 0.25
regardless of outcome. Any model that does useful work should
beat this benchmark.
"""

from __future__ import annotations

from numinous.tracker import TrackerBase


class BaselineTracker(TrackerBase):
    """Always returns 0.5 — the uninformative prior."""

    def _predict(
        self, subject: str
    ) -> dict:
        data = self._get_data(subject)
        event_id = data.get("event_id", subject) if isinstance(data, dict) else subject
        return {"event_id": event_id, "prediction": 0.5}
