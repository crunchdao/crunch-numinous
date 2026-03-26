"""Keyword: simple heuristic based on question text.

Analyzes the event title and description for sentiment-bearing
keywords and adjusts the market price accordingly.

This is a very naive NLP model — meant to show how text-based
features could be used. Real competitors would use LLMs or
more sophisticated NLP.
"""

from __future__ import annotations

from numinous.tracker import TrackerBase

YES_KEYWORDS = [
    "will", "expected", "likely", "confirmed", "approved",
    "passed", "won", "leads", "ahead", "surges",
]

NO_KEYWORDS = [
    "unlikely", "rejected", "failed", "denied", "blocked",
    "loses", "behind", "drops", "collapse", "not",
]

KEYWORD_WEIGHT = 0.02


class KeywordTracker(TrackerBase):
    """Adjusts market price based on keyword sentiment in the question."""

    def _predict(self, subject: str) -> dict:
        data = self._get_data(subject)
        if not isinstance(data, dict):
            return {"event_id": subject, "prediction": 0.5}

        event_id = data.get("event_id", subject)
        yes_price = data.get("yes_price", 0.5)
        title = data.get("title", "").lower()
        description = data.get("description", "").lower()
        text = f"{title} {description}"

        yes_count = sum(1 for kw in YES_KEYWORDS if kw in text)
        no_count = sum(1 for kw in NO_KEYWORDS if kw in text)

        adjustment = (yes_count - no_count) * KEYWORD_WEIGHT
        prediction = max(0.0, min(1.0, float(yes_price) + adjustment))

        return {"event_id": event_id, "prediction": prediction}
