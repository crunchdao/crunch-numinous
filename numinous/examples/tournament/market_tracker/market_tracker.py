"""Market price tracker: use the prediction market's current price.

Uses the ``yes_price`` field from the event data as the prediction.
This is the "efficient market hypothesis" model — the market already
incorporates all public information, so the best prediction is the
current price.

In practice, this is a very strong baseline for prediction market
events. It's hard to beat consistently.
"""

from __future__ import annotations

from numinous.tracker import TrackerBase


class MarketTracker(TrackerBase):
    """Returns the current market yes_price as the prediction."""

    def _predict(self, subject: str) -> dict:
        data = self._get_data(subject)
        if not isinstance(data, dict):
            return {"event_id": subject, "prediction": 0.5}

        event_id = data.get("event_id", subject)
        yes_price = data.get("yes_price", 0.5)

        prediction = max(0.0, min(1.0, float(yes_price)))

        return {"event_id": event_id, "prediction": prediction}
