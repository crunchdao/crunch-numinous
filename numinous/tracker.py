"""Base tracker for Numinous forecasting competition.

Models receive binary event data and must return a probability estimate.

The ``predict()`` return value must match ``ForecastOutput``::

    {"event_id": "abc123", "prediction": 0.72}
    {"event_id": "abc123", "prediction": 0.72, "reasoning": "Based on..."}

The ``reasoning`` field is optional. Predictions are clipped to [0.01, 0.99]
during scoring.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class TrackerBase:
    """Base class for Numinous forecasting models.

    Subclass this and implement ``_predict()`` to compete.

    Each prediction cycle:
    1. ``feed_update(data)`` is called with event data (title, description, etc.)
    2. ``predict(subject, ...)`` is called — return ``{"event_id": ..., "prediction": float}``

    The ``prediction`` field is a probability between 0.0 and 1.0 that the
    event resolves "Yes".
    """

    def __init__(self) -> None:
        self._latest_data_by_subject: dict[str, dict[str, Any]] = {}
        self._model_name = type(self).__name__

        self._run_id = None
        self._run_id_lock = Lock()

    @property
    def run_id(self) -> str:
        value = self._run_id
        if value is None:
            raise ValueError("run_id is not set.")

        return value

    def feed_update_and_predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convenience method to feed data and predict in one step."""

        subject = self.feed_update(data)
        return self.predict(subject)

    def feed_update(self, data: dict[str, Any]) -> str:
        """Receive event data. Override to maintain state.

        Data is stored per-subject (event_id or symbol key).

        Args:
            data: Event data dict (aligned with Numinous Subnet 6 validator payload)::

                {
                    "event_id": "numinous-12345",
                    "event_type": "llm",
                    "title": "Will X happen by Y?",
                    "description": "...",
                    "cutoff": "2026-03-16T00:00:00Z",
                    "metadata": {"market_type": "LLM", "topics": [...]}
                }
        """

        if not isinstance(data, dict):
            raise ValueError(f"Expected event to be a dict, got {type(data)}")

        subject_key = (
            data.get("event_id")
            or data.get("symbol")
            or "_default"
        )

        self._latest_data_by_subject[subject_key] = data

        # Always update _default so predict() can find the latest event
        # regardless of the scope subject passed by the coordinator.
        self._latest_data_by_subject["_default"] = data

        # Log feed summary
        logger.info(
            "[%s] feed_update event=%s title=%s",
            self._model_name,
            subject_key,
            data.get("title", "")[:60],
        )

        return subject_key

    def _get_data(self, subject: str) -> dict[str, Any] | None:
        """Return the latest event data for *subject*.

        Falls back to ``"_default"`` when no exact match exists.
        """

        return (
            self._latest_data_by_subject.get(subject)
            or self._latest_data_by_subject.get("_default")
        )

    def predict(self, subject: str) -> dict[str, Any]:
        """Return a forecast for the given event.

        Args:
            subject: Event identifier (e.g. event_id or scope subject).

        Returns:
            Dict matching ``ForecastOutput`` fields::

                {"event_id": str, "prediction": float, "reasoning": str | None}
        """

        event = self._get_data(subject)
        if event is None:
            raise ValueError(f"No data found for subject '{subject}'")

        # Update the current run_id from the event data, if available.
        with self._run_id_lock:
            self._run_id = event.get("run_id", self._run_id)

            result = self._predict(subject)
            logger.info(
                "[%s] predict subject=%s → %s",
                self._model_name,
                subject,
                result,
            )

            self._run_id = None

        return result

    def _predict(self, subject: str) -> dict[str, Any]:
        """Override this in your model. See ``predict()`` for docs."""
        raise NotImplementedError("Implement _predict() or predict() in your model")
