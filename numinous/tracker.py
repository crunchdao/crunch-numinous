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
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class TrackerBase(ABC):
    """Base class for Numinous forecasting models.

    Subclass this and implement ``_predict()`` to compete.

    The ``prediction`` field is a probability between 0.0 and 1.0 that the
    event resolves "Yes".
    """

    def __init__(self) -> None:
        pass

    def predict(self, event: dict[str, Any]) -> dict[str, Any]:
        """Return a forecast for the given event.

        Args:
            event: Event data dictionary (aligned with Numinous Subnet 6 validator payload).

        Returns:
            Dict matching ``ForecastOutput`` fields::

                {"event_id": str, "prediction": float, "reasoning": str | None}
        """

        # Update the current run_id from the event data, if available.
        result = self._predict(event)

        event_id = result.get("event_id", "unknown")
        logger.info(
            "Predicted event: %s: %s",
            event_id,
            result,
        )

        return result

    @abstractmethod
    def _predict(self, event: dict[str, Any]) -> dict[str, Any]:
        ...
