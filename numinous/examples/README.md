# Example Forecasting Models

Example notebooks for the Numinous binary event forecasting competition.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `quickstart.ipynb` | Minimal example — build an LLM-based forecaster from scratch |
| `geopolitical_tracker.ipynb` | Market price + geopolitical signal adjustment using Numinous Indicia |

## Interface

All models subclass `TrackerBase` and implement `_predict()`:

```python
from numinous.tracker import TrackerBase

class MyTracker(TrackerBase):
    def _predict(self, subject):
        data = self._get_data(subject)
        # ... your logic here ...
        return {"event_id": data["event_id"], "prediction": 0.65}
```

Models receive event data via `feed_update()`:

```python
{
    "event_id": "numinous-12345",
    "title": "Will X happen by Y?",
    "description": "...",
    "cutoff": "2026-03-16T00:00:00Z",
    "source": "numinous",
    "yes_price": 0.65,
    "volume_24h": 150000.0,
    "metadata": {"market_type": "Geopolitics", ...}
}
```

## Scoring

Predictions are scored with the **Brier score**: `(prediction - outcome)²`

- Lower is better: 0.0 = perfect, 1.0 = worst
- Predictions are clipped to [0.01, 0.99]
- Missing predictions are imputed as 0.5 → Brier score of 0.25
