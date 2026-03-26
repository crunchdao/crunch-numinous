"""Calibrated: shrink the market price toward 0.5.

Applies Bayesian shrinkage to the market price:
    prediction = alpha * yes_price + (1 - alpha) * 0.5

With alpha=0.8, this pulls extreme predictions toward the center.
This is useful when markets tend to be overconfident — a common
finding in prediction market research.
"""

from __future__ import annotations

from numinous.tracker import TrackerBase

SHRINKAGE_ALPHA = 0.8


class CalibratedTracker(TrackerBase):
    """Shrinks market price toward 0.5 — a calibration correction."""

    def _predict(self, subject: str) -> dict:
        data = self._get_data(subject)
        if not isinstance(data, dict):
            return {"event_id": subject, "prediction": 0.5}

        event_id = data.get("event_id", subject)
        yes_price = data.get("yes_price", 0.5)

        clamped_price = max(0.0, min(1.0, float(yes_price)))
        prediction = SHRINKAGE_ALPHA * clamped_price + (1.0 - SHRINKAGE_ALPHA) * 0.5

        return {"event_id": event_id, "prediction": prediction}
