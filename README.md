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
# Input: event data pushed to your model (aligned with Numinous Subnet 6)
{
    "event_id": "numinous-12345",
    "run_id": "run-abc",
    "track": "MAIN",
    "event_type": "llm",
    "title": "Will X happen by Y?",
    "description": "...",               # Optional
    "cutoff": "2026-03-16T00:00:00Z",   # Optional, ISO 8601
    "metadata": {"market_type": "LLM", "topics": ["Finance"]}
}

# Output: your probability forecast
{"event_id": "numinous-12345", "prediction": 0.72, "reasoning": "Based on..."}
```

- `prediction = 1.0` → certain "Yes"
- `prediction = 0.0` → certain "No"
- `prediction = 0.5` → maximum uncertainty

Predictions are clipped to **[0.01, 0.99]** during scoring.

## Scoring

Your final score is a weighted combination of **Brier score** (prediction accuracy) and **reasoning quality**. The weights depend on the track and event category.

### Weight distribution

| Pool              | Track  | Weight |
| ----------------- | ------ | ------ |
| Global Brier      | MAIN   | 5%     |
| Geopolitics Brier | MAIN   | 5%     |
| Reasoning         | MAIN   | 25%    |
| Global Brier      | SIGNAL | 30%    |
| Geopolitics Brier | SIGNAL | 15%    |
| Reasoning         | SIGNAL | 20%    |

[See emission weights.](docs/emission-weights.png)

### Brier score

Prediction accuracy is evaluated using the **[Brier score](https://en.wikipedia.org/wiki/Brier_score)**:

$$
\text{Brier} = (\text{prediction} - \text{outcome})^2
$$

**Lower is better.**

| Score | Meaning                                       |
| ----- | --------------------------------------------- |
| 0.00  | Perfect prediction                            |
| 0.25  | Always predicting 0.5 (no information)        |
| 1.00  | Worst possible (predicted 1.0, outcome was 0) |

The Brier score is **strictly proper** — the optimal strategy is to report your honest probability estimate.

Missing predictions are imputed as 0.5 → scored at 0.25.

Geopolitics events have a separate Brier pool because their resolution dates are often far away — Numinous requests intermediate predictions and uses Polymarket probabilities to score them.

### Reasoning scoring

The `reasoning` field returned by your model is **scored by an LLM** and has significant weight in the final score (25% on MAIN, 20% on SIGNAL). Models that simply relay market probabilities without genuine reasoning will be penalized.

The reasoning is evaluated on 5 criteria: sources used, evidence extracted, combination & weighting, uncertainties / counterpoints, and mapping to final probabilities. See the [full evaluation prompt](docs/reasoning-evaluation-prompt.md) for details.

### Tracks (MAIN / SIGNAL)

Each event specifies a **track** that restricts which resources your model can access:

| Track      | Resources                                                                                                                                    | Weight             |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| **MAIN**   | All gateway endpoints                                                                                                                        | Lower (35% total)  |
| **SIGNAL** | Restricted subset ([see config](https://github.com/numinouslabs/numinous/blob/main/neurons/validator/sandbox/signing_proxy/track_config.py)) | Higher (65% total) |

The `track` field is included in the event dict passed to `_predict()`. For **SIGNAL** events, do not call unauthorized gateway endpoints — the gateway enforces this and your prediction will fail.

## Create Your Tracker

A **tracker** is a model that receives event data and returns probability forecasts. The `predict()` method receives the full event dict directly.

**To participate, subclass `TrackerBase` and implement `_predict()`:**

```python
from numinous.tracker import TrackerBase


class MyModel(TrackerBase):

    def _predict(self, event):
        event_id = event.get("event_id", "unknown")
        run_id = event.get("run_id")
        track = event.get("track")

        # Your logic here
        prediction = 0.5

        return {
            "event_id": event_id,
            "prediction": prediction,
            "reasoning": "..."  # Optional, can be None
        }
```

### Available Event Fields

Inside `_predict()`, the `event` dict contains:

| Field         | Type          | Description                                                                     |
| ------------- | ------------- | ------------------------------------------------------------------------------- |
| `event_id`    | `str`         | Unique event identifier                                                         |
| `run_id`      | `str`         | Run identifier — must be forwarded to gateway calls for cost tracking           |
| `track`       | `str`         | `"MAIN"` or `"SIGNAL"` — determines which gateway resources are available       |
| `event_type`  | `str`         | Market type, lowercased (e.g. `"llm"`, `"sports"`, `"crypto"`)                  |
| `title`       | `str`         | The question being asked                                                        |
| `description` | `str \| None` | Additional context and resolution criteria                                      |
| `cutoff`      | `str \| None` | ISO 8601 resolution deadline                                                    |
| `metadata`    | `dict`        | Event metadata: `market_type`, `topics`, `trigger_name`, `polymarket_market_id` |

## Example

See the [quickstart notebook](numinous/examples/quickstart.ipynb) to get started.

## Gateway

Your model has **no direct internet access** in production. All external calls (LLMs, search, OSINT...) go through the **gateway**.
You will need to provide your own API keys (most providers offer a free tier)

- **In production**: `SANDBOX_PROXY_URL` is set automatically
- **Locally**: you use a public gateway, identical to the one used in production

[The official Numinous documentation contains a list of all available endpoints.](https://github.com/numinouslabs/numinous/blob/main/docs/gateway-guide.md)

### Use the gateway in your tracker

```python
import os
import httpx

# Specify your OpenAI's API Key
OPENAI_API_KEY = ...

# Get the URL of the Gateway
GATEWAY_URL = os.environ.get("SANDBOX_PROXY_URL", "https://public-gateway.numinous.competition.crunchdao.com")

response = httpx.post(
    f"{GATEWAY_URL}/api/gateway/openai/responses",
    json={
        # IMPORTANT: Always forward the `run_id` to the Gateway otherwise the model will fail.
        "run_id": run_id,

        "model": "gpt-5-mini",
        "input": [
            {
                "role": "user",
                "content": "Will BTC hit 100k?"
            }
        ],
    },
    headers={
        # IMPORTANT: Send the API Key header to the Gateway.
        "x-openai-api-key": OPENAI_API_KEY,
    },
    timeout=30,
)
```

### Authentication

Your model must submit the different providers' API keys that it needs to contact them. For example, if you want to use the OpenAI endpoint, you must include an OpenAI API key and pass it as the header `x-openai-api-key`.

To prevent the wrong API key being sent to the wrong provider, each of them has a unique header name:
| Provider         | Header Name                        |
| ---------------- | ---------------------------------- |
| Chutes           | `x-chutes-api-key`                 |
| Desearch         | `x-dearch-api-key`                 |
| Lightning Rod    | `x-lightning-rod-api-key`          |
| Lunar Crush      | `x-lunar-crush-api-key`            |
| Numinous Indicia | (No API key required)              |
| Numinous Signals | `x-numinous-signals-api-key`       |
| OpenAI           | `x-openai-api-key`                 |
| OpenRouter       | `x-openrouter-api-key`             |
| Perplexity       | `x-perplexity-api-key`             |
| Public Data      | (Name will depend on the service)* |
| Unusual Whales   | `x-unusual-whales-api-key`         |
| Vericore         | `x-vericore-api-key`               |

> [!NOTE]
> The header names for Public Data are always based on the data source you are trying to access. For example, if you want to use `api.stlouisfed.org`, the header will be named `x-fred-api-key`.
> 
> You can find a list of all the data sources [here (JSON format)](https://numinous.earth/api/v3/miner/public-data/sources).

## Links

- [Numinous Website](https://numinouslabs.io/)
- [Numinous GitHub](https://github.com/numinouslabs/numinous)
- [Numinous Leaderboard](https://leaderboard.numinouslabs.io/)
- [Subnet 6 on Taostats](https://taostats.io/subnets/6/chart)
- [CrunchDAO](https://crunchdao.com)
