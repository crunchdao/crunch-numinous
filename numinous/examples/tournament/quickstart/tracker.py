"""Quickstart: minimal tournament model.

Follows the market price directly.
Copy this directory and modify _predict() to implement your strategy.

Input (via feed_update):
    {"event_id": "...", "title": "Will X happen?", "yes_price": 0.65, ...}

Output (from _predict):
    {"event_id": "...", "prediction": 0.65}
"""

from __future__ import annotations

from numinous.tracker import TrackerBase


class Quickstart(TrackerBase):

    def _predict(self, subject):
        data = self._get_data(subject)
        if not isinstance(data, dict):
            return {"event_id": subject, "prediction": 0.5}

        event_id = data.get("event_id", subject)
        prediction = max(0.0, min(1.0, float(data.get("yes_price", 0.5))))

        return {"event_id": event_id, "prediction": prediction}
