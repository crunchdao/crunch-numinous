# Numinous Crunch Challenge

A real-time binary event forecasting competition powered by [Numinous](https://numinouslabs.io/) (Bittensor Subnet 6) and hosted on [CrunchDAO](https://crunchdao.com).

Numinous is a forecasting protocol that aggregates agents into superhuman LLM forecasters. In this competition, models predict the probability that real-world events — sourced from [Polymarket](https://polymarket.com) — resolve **"Yes"**. Predictions are scored using the **Brier score**, a strictly proper scoring rule that rewards calibrated, honest probabilities.

## Install

```bash
pip install crunch-numinous
```

## What You Must Predict

For each event, you receive structured data and must return a **probability between 0.0 and 1.0** that the event resolves "Yes":

```python
# Input: event data pushed to your model
{
    "event_id": "polymarket-12345",
    "title": "Will X happen by Y?",
    "description": "...",
    "cutoff": "2026-03-16T00:00:00Z",
    "source": "polymarket",
    "yes_price": 0.65,          # current market price
    "volume_24h": 150000.0,
    "metadata": {}
}

# Output: your probability forecast
{"event_id": "polymarket-12345", "prediction": 0.72}
```

- `prediction = 1.0` → certain "Yes"
- `prediction = 0.0` → certain "No"
- `prediction = 0.5` → maximum uncertainty

Predictions are clipped to **[0.01, 0.99]** during scoring.

## Scoring

Predictions are evaluated using the **[Brier score](https://en.wikipedia.org/wiki/Brier_score)**:

$$
\text{Brier} = (\text{prediction} - \text{outcome})^2
$$

**Lower is better.**

| Score | Meaning |
|-------|---------|
| 0.00  | Perfect prediction |
| 0.25  | Always predicting 0.5 (no information) |
| 1.00  | Worst possible (predicted 1.0, outcome was 0) |

The Brier score is **strictly proper** — the optimal strategy is to report your honest probability estimate.

Missing predictions are imputed as 0.5 → scored at 0.25.

Leaderboard ranking is based on **`brier_72h`** — the 72-hour rolling average Brier score (ascending, lower is better).

## Create Your Tracker

A **tracker** is a model that receives event data and returns probability forecasts. It operates incrementally: events are pushed via `feed_update()`, and predictions are requested via `predict()`.

**To participate, subclass `TrackerBase` and implement `_predict()`:**

```python
from numinous.tracker import TrackerBase


class MyModel(TrackerBase):

    def _predict(self, subject, resolve_horizon_seconds, step_seconds):
        data = self._get_data(subject)
        if not isinstance(data, dict):
            return {"event_id": subject, "prediction": 0.5}

        event_id = data.get("event_id", subject)
        yes_price = data.get("yes_price", 0.5)

        # Your logic here — this example just follows the market
        prediction = yes_price

        return {"event_id": event_id, "prediction": prediction}
```

### How It Works

1. **`feed_update(data)`** is called with new event data — stored automatically by `TrackerBase`
2. **`predict(subject, ...)`** is called — use `self._get_data(subject)` to access the latest event data

### Available Event Fields

Inside `_predict()`, `self._get_data(subject)` gives you:

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | `str` | Unique event identifier |
| `title` | `str` | The question being asked |
| `description` | `str` | Additional context and resolution criteria |
| `cutoff` | `str` | ISO 8601 resolution deadline |
| `source` | `str` | Data source (e.g. `"polymarket"`) |
| `yes_price` | `float` | Current market probability (0.0–1.0) |
| `volume_24h` | `float` | Recent trading volume in USD |
| `metadata` | `dict` | Additional source-specific data (slug, condition_id, etc.) |

## Tracker Examples

The package ships with several example trackers:

| Tracker | Strategy |
|---------|----------|
| **`BaselineTracker`** | Always predicts 0.5 — the uninformative prior. Guaranteed Brier score of 0.25. |
| **`MarketTracker`** | Returns the current `yes_price` as the prediction. |
| **`CalibratedTracker`** | Shrinks market price toward 0.5 using Bayesian shrinkage (α=0.8). |
| **`ContrarianTracker`** | Predicts `1.0 - yes_price` — bets against the crowd. |
| **`KeywordTracker`** | Adjusts market price using keyword sentiment from the event title/description. |

```python
from numinous.examples import MarketTracker

# Use directly
tracker = MarketTracker()
tracker.feed_update({"event_id": "abc", "yes_price": 0.65, "title": "Will X happen?"})
result = tracker.predict("abc", resolve_horizon_seconds=3600, step_seconds=300)
print(result)  # {"event_id": "abc", "prediction": 0.65}
```

See [`numinous/examples/`](numinous/examples/) for full implementations.

## Links

- [Numinous Website](https://numinouslabs.io/)
- [Numinous GitHub](https://github.com/numinouslabs/numinous)
- [Numinous Leaderboard](https://leaderboard.numinouslabs.io/)
- [Subnet 6 on Taostats](https://taostats.io/subnets/6/chart)
- [CrunchDAO](https://crunchdao.com)
