"""Base tracker for Numinous forecasting competition.

Models receive binary event data and must return a probability estimate.

The ``predict()`` return value must match ``ForecastOutput``::

    {"event_id": "abc123", "prediction": 0.72}   # 72% chance of "Yes"
    {"event_id": "abc123", "prediction": 0.5}     # maximum uncertainty
    {"event_id": "abc123", "prediction": 0.05}    # very likely "No"

Predictions are clipped to [0.01, 0.99] during scoring.
"""

from __future__ import annotations

import logging
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

    def feed_update(self, data: dict[str, Any]) -> None:
        """Receive event data. Override to maintain state.

        Data is stored per-subject (event_id or symbol key).

        Args:
            data: Event data dict, typically containing::

                {
                    "event_id": "polymarket-12345",
                    "title": "Will X happen by Y?",
                    "description": "...",
                    "cutoff": "2026-03-16T00:00:00Z",
                    "source": "polymarket",
                    "yes_price": 0.65,
                    "volume_24h": 150000.0,
                    "metadata": {...}
                }
        """
        if isinstance(data, dict):
            # Use event_id as primary key, fall back to symbol, then _default
            subject_key = (
                data.get("event_id")
                or data.get("symbol")
                or "_default"
            )
        else:
            subject_key = "_default"

        self._latest_data_by_subject[subject_key] = data
        # Always update _default so predict() can find the latest event
        # regardless of the scope subject passed by the coordinator.
        self._latest_data_by_subject["_default"] = data

        # Log feed summary
        if isinstance(data, dict):
            title = data.get("title", "")[:60]
            logger.info(
                "[%s] feed_update event=%s title=%s",
                self._model_name,
                subject_key,
                title,
            )

    def _get_data(self, subject: str) -> dict[str, Any] | None:
        """Return the latest event data for *subject*.

        Falls back to ``"_default"`` when no exact match exists.
        """
        return self._latest_data_by_subject.get(
            subject,
            self._latest_data_by_subject.get("_default"),
        )

    def predict(
        self, subject: str, resolve_horizon_seconds: int, step_seconds: int
    ) -> dict[str, Any]:
        """Return a forecast for the given event.

        Args:
            subject: Event identifier (e.g. event_id or scope subject).
            resolve_horizon_seconds: Time until resolution (seconds).
            step_seconds: Time step between predictions (seconds).

        Returns:
            Dict matching ``ForecastOutput`` fields::

                {"event_id": str, "prediction": float}

            Where prediction is a probability between 0.0 and 1.0.
        """
        result = self._predict(subject, resolve_horizon_seconds, step_seconds)
        data = self._get_data(subject)
        if isinstance(data, dict):
            metadata = data.get("metadata") or {}
            if isinstance(metadata, dict):
                result["market_type"] = metadata.get("market_type", "")
            else:
                result["market_type"] = ""
        logger.info(
            "[%s] predict subject=%s horizon=%ds → %s",
            self._model_name,
            subject,
            resolve_horizon_seconds,
            result,
        )
        return result

    def _predict(
        self, subject: str, resolve_horizon_seconds: int, step_seconds: int
    ) -> dict[str, Any]:
        """Override this in your model. See ``predict()`` for docs."""
        raise NotImplementedError("Implement _predict() or predict() in your model")
