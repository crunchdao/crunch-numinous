# Example Forecasting Models

These are example models for the Numinous binary event forecasting competition.
Each model implements `TrackerBase` and returns a probability estimate for
binary events (e.g., "Will X happen by date Y?").

## Models

| Model | Strategy | Description |
|-------|----------|-------------|
| `BaselineTracker` | Always 0.5 | Uninformative prior — guaranteed Brier score of 0.25 |
| `MarketTracker` | Market price | Uses the current `yes_price` as the prediction (efficient market hypothesis) |
| `CalibratedTracker` | Shrink to 0.5 | Pulls market price toward 0.5 with Bayesian shrinkage (alpha=0.8) |
| `ContrarianTracker` | Invert market | Predicts `1.0 - yes_price` — bets against the crowd |
| `KeywordTracker` | Text heuristic | Adjusts market price based on keyword sentiment in the event title/description |

## Interface

All models receive event data via `feed_update()` and return predictions
via `predict()`:

```python
# Input (via feed_update)
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

# Output (from predict)
{
    "event_id": "polymarket-12345",
    "prediction": 0.72   # probability of "Yes" (0.0 to 1.0)
}
```

## Scoring

Predictions are scored with the **Brier score**: `(prediction - outcome)²`

- Lower is better: 0.0 = perfect, 1.0 = worst
- Predictions are clipped to [0.01, 0.99]
- Missing predictions are imputed as 0.5 → Brier score of 0.25

## Building Your Own

1. Subclass `TrackerBase`
2. Implement `_predict()` returning `{"event_id": str, "prediction": float}`
3. Use `self._get_data(subject)` to access the latest event data
4. Use any Python libraries, LLMs, or search tools you want
