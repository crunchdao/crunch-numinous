"""Contrarian: bet against the crowd.

Reflects the market price around 0.5:
    prediction = 1.0 - yes_price

If the market says 70% yes, this model says 30%.
Useful as a diversity model — it will perform well exactly
when the market is wrong.
"""

from __future__ import annotations

from numinous.tracker import TrackerBase


class ContrarianTracker(TrackerBase):
    """Predicts the opposite of the market consensus."""

    def _predict(
        self, subject: str
    ) -> dict:
        data = self._get_data(subject)
        if not isinstance(data, dict):
            return {"event_id": subject, "prediction": 0.5}

        event_id = data.get("event_id", subject)
        yes_price = data.get("yes_price", 0.5)

        prediction = 1.0 - max(0.0, min(1.0, float(yes_price)))

        return {"event_id": event_id, "prediction": prediction}
