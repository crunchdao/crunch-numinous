import asyncio
import json
import math
import os
import re
from collections import defaultdict
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import httpx
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _parse_prediction_value(text: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", text)
    assert match is not None, f"No numeric prediction in: {text}"
    return float(match.group())


def _parse_reasoning_value(text: str) -> str:
    lines = text.splitlines()
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line.startswith("REASONING:"):
            continue
        inline = line.replace("REASONING:", "", 1).strip()
        if inline:
            return inline
        trailing = [next_line.strip() for next_line in lines[idx + 1:] if next_line.strip()]
        return "\n".join(trailing)
    return ""

async def agent1():
    # ============================================================

    RUN_ID = os.getenv("RUN_ID") or str(uuid4())
    PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
    OPENAI_URL = f"{PROXY_URL}/api/gateway/openai"
    CHUTES_URL = f"{PROXY_URL}/api/gateway/chutes"
    DESEARCH_URL = f"{PROXY_URL}/api/gateway/desearch"
    TODAY_STR = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ORIGINAL_SYSTEM_PROMPT = """You are an expert forecaster for prediction markets. Estimate P(YES) with rigorous research.
    RESEARCH APPROACH - adapt your search strategy to the event type:
    For competitions and matches:
    - Search betting odds and convert to probability (decimal odds D → prob ≈ 1/D minus margin)
    - Check recent form, injuries, head-to-head records, rankings
    - Home advantage matters in team sports (+10-15%)
    For political events and elections:
    - Search Polymarket/PredictIt first - market prices are strong signals
    - For elections: check polling aggregates (538, RCP), consider historical polling errors
    - For policy/diplomatic events: prioritize official sources (Reuters, AP, government statements)
    - Check procedural requirements (votes needed, veto power, legislative calendar)
    For economic and financial events:
    - Search market-implied probabilities (CME FedWatch for rates, futures markets)
    - Check central bank communications and forward guidance
    - Economic calendar: what data releases occur before cutoff?
    For product launches and technology:
    - Check official company channels, press releases, SEC filings
    - Consider historical track record (announced vs actual delivery dates)
    - Distinguish between: announced, shipped, generally available
    For entertainment and awards:
    - Search prediction markets and expert consensus sites
    - Box office tracking, review aggregates
    - Awards predictions converge closer to ceremony date
    ALWAYS DO THESE:
    1. Search "Polymarket [topic]" - if market exists, price ≈ probability
    2. Search recent news (prioritize last 48-72 hours)
    3. Verify key claims with multiple sources
    4. Consider time until cutoff (more time = more uncertainty)
    ANALYSIS PRINCIPLES:
    - Polymarket price is your anchor - deviate only with strong contrary evidence
    - Official sources > speculation and rumors
    - Consider base rates: how often do similar events happen?
    - Resolution criteria are literal - read exact wording carefully
    - Range: never return exactly 0 or 1, use [0.01, 0.99]
    OUTPUT FORMAT:
    PREDICTION: [0.01-0.99]
    REASONING: [Key evidence, market signal if found, main uncertainties, 3-5 sentences]"""

    class CostTracker:
        """Accumulates API costs per provider within a single event."""
        def __init__(self):
            self.reset()
        def reset(self):
            self._costs: Dict[str, float] = {"openai": 0.0, "chutes": 0.0, "desearch": 0.0}
            self._calls: Dict[str, int] = {"openai": 0, "chutes": 0, "desearch": 0}
        def record(self, provider: str, cost: float = 0.0):
            if provider in self._costs:
                self._costs[provider] += cost
                self._calls[provider] += 1
        def print_summary(self):
            total = sum(self._costs.values())
            total_calls = sum(self._calls.values())
            print(f"\n{'─'*50}")
            print(f"  COST SUMMARY  ({total_calls} total API calls)")
            print(f"{'─'*50}")
            for p in ("openai", "chutes", "desearch"):
                c = self._costs[p]
                n = self._calls[p]
                print(f"  {p:>10}_cost : ${c:.6f}  ({n} calls)")
            print(f"  {'TOTAL':>10}      : ${total:.6f}")
            print(f"{'─'*50}\n")
    COST_TRACKER = CostTracker()
    # Chutes model pricing (per million tokens) — used when proxy response lacks "cost"
    _CHUTES_PRICING: Dict[str, Dict[str, float]] = {
        "tngtech/DeepSeek-R1T-Chimera":         {"input": 0.30, "output": 1.20},
        "Qwen/Qwen3-235B-A22B-Instruct-2507":   {"input": 0.08, "output": 0.55},
        "openai/gpt-oss-20b":                    {"input": 0.00, "output": 0.00},
        "openai/gpt-oss-120b":                   {"input": 0.04, "output": 0.40},
    }
    def _estimate_chutes_cost(data: dict) -> float:
        """Estimate Chutes cost from usage tokens when no explicit cost field."""
        model = data.get("model", "")
        usage = data.get("usage", {})
        if not usage:
            return 0.0
        pricing = _CHUTES_PRICING.get(model, {"input": 0.10, "output": 0.40})
        inp = usage.get("prompt_tokens", 0)
        out = usage.get("completion_tokens", 0)
        return (pricing["input"] * inp + pricing["output"] * out) / 1_000_000
    class TrackedAsyncClient(httpx.AsyncClient):
        """httpx.AsyncClient subclass that auto-records per-provider costs."""
        _PROVIDER_KEYS = {
            "/gateway/openai": "openai",
            "/gateway/chutes": "chutes",
            "/gateway/desearch": "desearch",
        }
        async def post(self, url, **kwargs):          # type: ignore[override]
            resp = await super().post(url, **kwargs)
            url_str = str(url)
            provider = None
            for fragment, name in self._PROVIDER_KEYS.items():
                if fragment in url_str:
                    provider = name
                    break
            if provider is None:
                return resp
            cost = 0.0
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    cost = float(data.get("cost", 0) or 0)
                    # Fallback: estimate Chutes cost from usage tokens
                    if cost == 0.0 and provider == "chutes":
                        cost = _estimate_chutes_cost(data)
                except Exception:
                    pass
            COST_TRACKER.record(provider, cost)
            return resp
    def extract_openai_text(response: dict) -> str:
        """Extract text content from OpenAI Responses API output. Shared across handlers."""
        for item in response.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") in ("output_text", "text") and content.get("text"):
                        return content["text"]
        return ""
    def safe_parse_json(text: str) -> Optional[dict]:
        """Try multiple strategies to extract JSON from LLM output."""
        if not text or not text.strip():
            return None
        # Strategy 1: code fence
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if m:
            try:
                return json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                pass
        # Strategy 2: raw parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass
        # Strategy 3: first { to last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
        # Strategy 4: minimal probability object
        m = re.search(r'\{[^{}]*"probability"[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except (json.JSONDecodeError, ValueError):
                pass
        return None
    # ============================================================
    # PART 2: EVENT CLASSIFIER
    # ============================================================
    VALID_CATEGORIES = ["sports", "app_store", "weather", "earnings", "box_office", "global", "geopolitics"]
    CLASSIFY_PROMPT = """Classify this prediction market event into exactly ONE category.
    Categories:
    - sports: Athletic competitions, matches, games, betting odds, team vs team
    - app_store: Apple App Store rankings, app positions, top free/paid apps
    - weather: Temperature forecasts, precipitation, weather conditions
    - earnings: Company earnings, EPS, revenue, stock moves after earnings, financial reports
    - box_office: Movie box office revenue, film performance, opening weekends
    - global: Cryptocurrency prices, stock market targets, broad financial predictions, mixed-domain markets not fitting other categories
    - geopolitics: Elections, politics, international relations, policy, legislation, wars, tariffs, economic indicators, and everything else
    Event Title: {title}
    Event Description: {description}
    Return ONLY the category name, nothing else. One word from: sports, app_store, weather, earnings, box_office, global, geopolitics"""
    def classify_by_keywords(event_data: dict) -> str:
        """Keyword-based fallback classifier."""
        title = event_data.get("title", "")
        desc = event_data.get("description", "")
        combined = f"{title} {desc}".lower()
        title_lower = title.lower()
        # Check metadata topics first
        for topic in event_data.get("metadata", {}).get("topics", []):
            t = topic.lower()
            if t in ("sports",):
                return "sports"
            if t in ("app store",):
                return "app_store"
            if t in ("weather",):
                return "weather"
            if t in ("earnings",):
                return "earnings"
        # Sports
        sport_hints = [' vs ', ' vs. ', 'upcoming game', 'stoppage time', 'cricket',
                    'both teams to score', 'championship', 'playoff', 'nfl', 'nba',
                    'mlb', 'nhl', 'ufc', 'boxing', 'premier league', 'champions league']
        title_sport = [' win ', ' win?', ' beat ', ' defeat ']
        if any(h in combined for h in sport_hints) or any(h in title_lower for h in title_sport):
            return "sports"
        # App Store
        if 'app store' in combined or ('top paid' in combined or 'top free' in combined):
            return "app_store"
        # Weather
        if 'temperature' in combined or 'weather forecast' in combined or 'precipitation' in combined:
            return "weather"
        # Earnings
        earnings_kw = ['earnings', 'eps', 'post-earnings', 'implied move', 'quarterly report']
        if any(kw in combined for kw in earnings_kw):
            return "earnings"
        if any(q in combined for q in ['q1 ', 'q2 ', 'q3 ', 'q4 ']) and 'above' in combined:
            return "earnings"
        # Box Office
        if 'box office' in combined or 'opening weekend' in combined or ('movie' in combined and 'revenue' in combined):
            return "box_office"
        # Global (crypto prices, stock targets, broad financial predictions)
        global_kw = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'price of', 'stock price',
                    'price target', 'market cap', 'token', 'solana', 'dogecoin', 'xrp']
        if any(kw in combined for kw in global_kw):
            return "global"
        return "geopolitics"
    async def classify_event_with_llm(event_data: dict) -> str:
        """Classify event using LLM with keyword fallback."""
        keyword_result = classify_by_keywords(event_data)
        # Try LLM classification for confirmation / override
        try:
            prompt = CLASSIFY_PROMPT.format(
                title=event_data.get("title", ""),
                description=event_data.get("description", "")[:500]
            )
            async with TrackedAsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{OPENAI_URL}/responses",
                    json={
                        "model": "gpt-5-nano",
                        "input": [{"role": "user", "content": prompt}],
                        "run_id": RUN_ID,
                    },
                    timeout=15.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = extract_openai_text(data).strip().lower().replace(" ", "_")
                    if text in VALID_CATEGORIES:
                        print(f"[CLASSIFIER] LLM classified as: {text}")
                        return text
        except Exception as e:
            print(f"[CLASSIFIER] LLM failed: {e}")
        print(f"[CLASSIFIER] Using keyword fallback: {keyword_result}")
        return keyword_result


    class GlobalHandler:
        """
        Handler for Global / broad market events (crypto prices, stock targets, mixed-domain).
        Architecture: Two-stage iterative refinement with OpenAI web search,
        Desearch fallback, response repair, and statistical calibration.
        """
        # --- Constants ---
        MODEL_CHAIN = ["gpt-5.2", "gpt-5", "gpt-5-mini"]
        REPAIR_MODELS = ["gpt-5-nano", "gpt-5-mini"]
        RETRY_LIMIT = 3
        BACKOFF_FACTOR = 2.0
        REQUEST_TIMEOUT = 145.0
        SYSTEM_PROMPT = """You are a professional forecasting analyst specializing in prediction markets.
    Your methodology:
    1. Search for Polymarket YES price - this is the wisdom of the crowd
    2. Find betting odds or prediction market data for sports/political events
    3. For crypto/stocks: get current prices and recent trends
    4. Review latest news and developments
    Key principles:
    - Market prices (Polymarket, betting odds) are strong signals
    - Convert betting odds to implied probability correctly
    - Check resolution criteria carefully
    - Output probabilities in [0.01, 0.99] range
    - More distant deadlines = higher uncertainty"""
        # --- Calibration parameters ---
        CALIBRATION_MAP = {
            'weather': {
                'method': 'platt',
                'a': 0.388963,
                'b': -1.648164,
            },
            'app_store': {
                'method': 'beta',
                'w': [-0.685640, 1.786524, 0.179951],
            },
            'other': {
                'method': 'isotonic',
                'X_sorted': [0.01, 0.05, 0.21, 0.32, 0.4, 0.42, 0.65, 0.99],
                'isotonic_y': [0.0, 0.025641, 0.027027, 0.142857, 0.5, 0.736842, 1.0, 1.0],
            },
        }
        # --- Prompt templates ---
        @classmethod
        def _initial_forecast_prompt(cls, event_data):
            today = datetime.now().strftime("%Y-%m-%d")
            return f"""Date: {today}
    Market Question: {event_data.get('title', '')}
    Details: {event_data.get('description', '')}
    Resolves: {event_data.get('cutoff', '')}
    Execute comprehensive research:
    - Search Polymarket for this market's current price
    - Find betting odds if applicable and calculate implied probability
    - Check current prices for crypto/stock markets
    - Review recent relevant news
    Return your analysis as JSON:
    ```json
    {{
    "market_consensus": <0-1 or null>,
    "odds_found": <string or null>,
    "price_data": <string or null>,
    "evidence": ["point1", "point2", "point3"],
    "analysis": "Your detailed reasoning in 5-7 sentences",
    "forecast": <probability 0-1>
    }}
    ```"""
        @classmethod
        def _refinement_prompt(cls, event_data, stage_one):
            today = datetime.now().strftime("%Y-%m-%d")
            parsed = stage_one.get("parsed_result", {})
            return f"""Date: {today}
    Market Question: {event_data.get('title', '')}
    Details: {event_data.get('description', '')}
    Resolves: {event_data.get('cutoff', '')}
    Initial Analysis Review:
    - Forecast: {stage_one.get('probability', 0.5)}
    - Market Consensus: {parsed.get('market_consensus', 'not found')}
    - Odds: {parsed.get('odds_found', 'not found')}
    - Evidence: {parsed.get('evidence', [])}
    - Reasoning: {stage_one.get('rationale', '')}
    Your task:
    1. Assess if critical information is missing
    2. Use web search to fill any gaps in research
    3. Verify or adjust the forecast based on complete information
    4. Provide final probability estimate
    Return JSON:
    ```json
    {{
    "additional_research": ["query1", "query2"] or null,
    "new_evidence": ["finding1", "finding2"] or null,
    "final_analysis": "3-5 sentence assessment",
    "forecast": <probability 0-1>
    }}
    ```"""
        @classmethod
        def _backup_prompt(cls, event_data, research_summary):
            today = datetime.now().strftime("%Y-%m-%d")
            return f"""Date: {today}
    Market Question: {event_data.get('title', '')}
    Details: {event_data.get('description', '')}
    Resolves: {event_data.get('cutoff', '')}
    External Research Results:
    {research_summary}
    Based on this research, provide your probability forecast.
    Return JSON:
    ```json
    {{
    "market_consensus": <0-1 or null>,
    "odds_found": <string or null>,
    "evidence": ["point1", "point2"],
    "analysis": "Your reasoning in 2-3 sentences",
    "forecast": <probability 0-1>
    }}
    ```"""
        # --- Utility methods ---
        @staticmethod
        def _validate_probability(value):
            """Normalize and validate probability value."""
            if 0.0 <= value <= 1.0:
                return value
            elif 1.0 < value <= 100.0:
                return value / 100.0
            return None
        @staticmethod
        def _clamp(value, lo=0.01, hi=0.99):
            return max(lo, min(hi, value))
        @classmethod
        def _parse_forecast_json(cls, text):
            """Parse JSON from LLM output, supporting 'forecast' and 'probability' keys."""
            parsed = safe_parse_json(text)
            if parsed:
                # Rename 'probability' to 'forecast' for consistency
                if "probability" in parsed and "forecast" not in parsed:
                    parsed["forecast"] = parsed.pop("probability")
                return parsed
            # Try finding a JSON object with forecast field
            m = re.search(r'\{[^{}]*"forecast"[^{}]*\}', text or "", re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except (json.JSONDecodeError, ValueError):
                    pass
            m = re.search(r'\{[^{}]*"probability"[^{}]*\}', text or "", re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group())
                    if "probability" in obj:
                        obj["forecast"] = obj.pop("probability")
                    return obj
                except (json.JSONDecodeError, ValueError):
                    pass
            return None
        # --- Calibration ---
        @classmethod
        def _categorize_event(cls, event_data):
            """Determine event category for calibration."""
            title = event_data.get("title", "").lower()
            description = event_data.get("description", "").lower()
            combined = f"{title} {description}"
            # Check metadata topics
            for topic in event_data.get("metadata", {}).get("topics", []):
                t = topic.lower()
                if t in ("sports", "weather", "app store", "earnings"):
                    return t
            # Pattern matching
            if any(p in combined for p in [' vs ', ' vs. ', ' win ', ' win?', 'cricket', 'game', 'match']):
                return 'sports'
            if ' app ' in combined or 'app store' in combined:
                return 'app_store'
            if 'temperature' in combined or 'weather' in combined:
                return 'weather'
            if 'earnings' in combined or (any(q in combined for q in ['q1', 'q2', 'q3', 'q4']) and
                                        ('revenue' in combined or 'profit' in combined or 'above' in combined)):
                return 'earnings'
            return 'other'
        @classmethod
        def _calibrate(cls, raw_prob, category):
            """Apply statistical calibration transformation."""
            import numpy as np
            params = cls.CALIBRATION_MAP.get(category.lower())
            if not params:
                return raw_prob
            method = params["method"]
            if method == "platt":
                z = params["a"] * raw_prob + params["b"]
                return float(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))))
            elif method == "beta":
                w = np.array(params["w"])
                p_safe = np.clip(raw_prob, 1e-10, 1 - 1e-10)
                f1 = np.log(p_safe / (1 - p_safe))
                f2 = -np.log(1 - p_safe)
                features = np.array([1.0, f1, f2])
                z = np.dot(features, w)
                return float(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))))
            elif method == "isotonic":
                X = np.array(params["X_sorted"])
                Y = np.array(params["isotonic_y"])
                if raw_prob <= X[0]:
                    return float(Y[0])
                elif raw_prob >= X[-1]:
                    return float(Y[-1])
                else:
                    idx = np.searchsorted(X, raw_prob)
                    x0, x1 = X[idx - 1], X[idx]
                    y0, y1 = Y[idx - 1], Y[idx]
                    if x1 != x0:
                        interpolated = y0 + (y1 - y0) * (raw_prob - x0) / (x1 - x0)
                    else:
                        interpolated = y0
                    return float(np.clip(interpolated, 0.0, 1.0))
            return raw_prob
        # --- LLM call with retry and model chain ---
        @classmethod
        async def _call_llm(cls, client, messages, enable_search=True):
            """Call LLM with retry logic across model chain."""
            payload = {"input": messages, "run_id": RUN_ID}
            if enable_search:
                payload["tools"] = [{"type": "web_search"}]
            last_err = None
            for model in cls.MODEL_CHAIN:
                payload["model"] = model
                for attempt in range(cls.RETRY_LIMIT):
                    try:
                        resp = await client.post(
                            f"{OPENAI_URL}/responses", json=payload, timeout=cls.REQUEST_TIMEOUT
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            return data, data.get("cost", 0.0)
                        if resp.status_code in (429, 500, 502, 503):
                            wait = cls.BACKOFF_FACTOR ** (attempt + 1)
                            print(f"[GLOBAL] {model} status {resp.status_code}, retry in {wait:.1f}s")
                            await asyncio.sleep(wait)
                            continue
                        print(f"[GLOBAL] {model} status {resp.status_code}, next model")
                        break
                    except httpx.TimeoutException:
                        print(f"[GLOBAL] {model} timeout")
                        break
                    except Exception as e:
                        last_err = e
                        print(f"[GLOBAL] {model} error: {e}")
                        break
            raise last_err or RuntimeError("All LLM attempts exhausted")
        # --- Web search fallback ---
        @classmethod
        async def _search_web(cls, client, query):
            """Perform Desearch web search."""
            try:
                resp = await client.post(
                    f"{DESEARCH_URL}/web/search",
                    json={"query": query, "num": 10, "start": 0, "run_id": RUN_ID},
                    timeout=60.0
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                results = data.get("data", [])
                if not results:
                    return None
                lines = [f"• {r.get('title', '')}: {r.get('snippet', '')}" for r in results[:5]]
                summary = "Search findings:\n" + "\n".join(lines)
                # Try crawling top result
                for r in results[:3]:
                    url = r.get("link", "")
                    if url:
                        try:
                            cr = await client.post(
                                f"{DESEARCH_URL}/web/crawl",
                                json={"url": url, "run_id": RUN_ID},
                                timeout=15.0
                            )
                            if cr.status_code == 200:
                                content = cr.json().get("content", "")
                                if content:
                                    return f"{summary}\n\nPage content:\n{content}"
                        except Exception:
                            pass
                return summary
            except Exception as e:
                print(f"[GLOBAL] Search error: {e}")
                return None
        # --- Response repair ---
        @classmethod
        async def _repair_response(cls, client, broken_text):
            """Attempt to repair malformed LLM response."""
            repair_prompt = f"""The following is a malformed response from a forecasting model.
    Extract the probability forecast and reasoning, then return valid JSON.
    Malformed response:
    {broken_text}
    Return only:
    {{"forecast": <number 0-1>, "analysis": "<reasoning>"}}"""
            for model in cls.REPAIR_MODELS:
                try:
                    payload = {
                        "model": model,
                        "input": [{"role": "user", "content": repair_prompt}],
                        "run_id": RUN_ID
                    }
                    resp = await client.post(f"{OPENAI_URL}/responses", json=payload, timeout=60.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        text = extract_openai_text(data)
                        parsed = cls._parse_forecast_json(text)
                        if parsed and "forecast" in parsed:
                            return parsed
                except Exception as e:
                    print(f"[GLOBAL] Repair with {model} failed: {e}")
                    continue
            return {"forecast": 0.35, "analysis": "Response repair failed"}
        # --- Stage 1: Initial forecast with web search ---
        @classmethod
        async def _stage_one(cls, client, event_data):
            query = cls._initial_forecast_prompt(event_data)
            messages = [
                {"role": "developer", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
            try:
                data, cost = await cls._call_llm(client, messages, enable_search=True)
                text = extract_openai_text(data)
                print(f"[GLOBAL][STAGE_1] Response: {text[:400]}...")
            except Exception as e:
                print(f"[GLOBAL][STAGE_1] Primary failed: {e}, trying Desearch fallback")
                # Fallback: manual search
                search_queries = [
                    f"polymarket {event_data.get('title', '')}",
                    f"{event_data.get('title', '')} odds probability",
                    f"{event_data.get('title', '')} news"
                ]
                research_results = []
                for sq in search_queries:
                    result = await cls._search_web(client, sq)
                    if result:
                        research_results.append(result)
                if not research_results:
                    raise RuntimeError("All research methods failed")
                research_summary = "\n\n".join(research_results)
                backup = cls._backup_prompt(event_data, research_summary)
                messages[1]["content"] = backup
                data, cost = await cls._call_llm(client, messages, enable_search=False)
                text = extract_openai_text(data)
                print(f"[GLOBAL][STAGE_1] Backup response: {text[:400]}...")
            parsed = cls._parse_forecast_json(text)
            if not parsed or "forecast" not in parsed:
                print("[GLOBAL][STAGE_1] Repairing response")
                parsed = await cls._repair_response(client, text)
            raw = float(parsed.get("forecast", 0.5))
            norm = cls._validate_probability(raw)
            prob = cls._clamp(norm if norm is not None else 0.5)
            rationale = parsed.get("analysis") or parsed.get("reasoning", "")
            return {
                "probability": prob,
                "rationale": rationale,
                "parsed_result": parsed,
                "cost": cost
            }
        # --- Stage 2: Refinement ---
        @classmethod
        async def _stage_two(cls, client, event_data, stage_one):
            query = cls._refinement_prompt(event_data, stage_one)
            messages = [
                {"role": "developer", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
            try:
                data, cost = await cls._call_llm(client, messages, enable_search=True)
                text = extract_openai_text(data)
                print(f"[GLOBAL][STAGE_2] Response: {text[:400]}...")
                parsed = cls._parse_forecast_json(text)
                if parsed and "forecast" in parsed:
                    raw = float(parsed["forecast"])
                    norm = cls._validate_probability(raw)
                    prob = cls._clamp(norm if norm is not None else stage_one["probability"])
                    rationale = parsed.get("final_analysis") or parsed.get("analysis", "")
                    return {"probability": prob, "rationale": rationale, "cost": cost}
            except Exception as e:
                print(f"[GLOBAL][STAGE_2] Failed: {e}, using stage 1 result")
            return stage_one
        # --- Main entry ---
        @classmethod
        async def run(cls, event_data: dict, category: str) -> dict:
            eid = event_data.get("event_id", "?")
            title = event_data.get("title", "")
            start_time = time.time()
            print(f"[GLOBAL] Event: {eid} | {title[:80]}")
            probabilities = []
            reasoning_parts = []
            total_cost = 0.0
            async with TrackedAsyncClient(timeout=180.0) as client:
                # Stage 1: Initial forecast with research
                try:
                    s1 = await cls._stage_one(client, event_data)
                    probabilities.append(s1["probability"])
                    reasoning_parts.append(s1["rationale"])
                    total_cost += s1.get("cost", 0.0)
                    print(f"[GLOBAL][STAGE_1] Forecast: {s1['probability']:.3f}")
                except Exception as e:
                    print(f"[GLOBAL][STAGE_1] Error: {e}")
                    return {
                        "event_id": eid,
                        "prediction": 0.35,
                        "reasoning": f"Stage 1 failed: {str(e)[:200]}"
                    }
                # Stage 2: Refinement and validation
                try:
                    s2 = await cls._stage_two(client, event_data, s1)
                    probabilities.append(s2["probability"])
                    reasoning_parts.append(s2["rationale"])
                    total_cost += s2.get("cost", 0.0)
                    print(f"[GLOBAL][STAGE_2] Refined: {s2['probability']:.3f}")
                except Exception as e:
                    print(f"[GLOBAL][STAGE_2] Error: {e}, using stage 1 only")
            # Aggregate probabilities
            if probabilities:
                avg_prob = sum(probabilities) / len(probabilities)
                final_prob = cls._clamp(avg_prob)
            else:
                final_prob = 0.35
            # Apply calibration
            # category = cls._categorize_event(event_data)
            calibrated = cls._calibrate(final_prob, category)
            calibrated = cls._clamp(calibrated)
            print(f"[GLOBAL] Calibration ({category}): {final_prob:.3f} -> {calibrated:.3f}")
            reasoning = " | ".join([r for r in reasoning_parts if r])
            duration = time.time() - start_time
            print(f"[GLOBAL] Final: {calibrated:.3f} | {duration:.1f}s | Cost: ${total_cost:.6f}")
            return {
                "event_id": eid,
                "prediction": calibrated,
                "reasoning": (reasoning or "Forecast generated")[:2000]
            }


    class WeatherHandler:
        """
        Top scorer for Weather events.
        Architecture: OpenAI web search primary + Desearch fallback + postprocessing.
        """
        MODELS = ["gpt-5.2", "gpt-5", "gpt-5-mini"]
        TIMEOUT = 145.0
        SYSTEM_PROMPT = """You are an expert forecaster with web search. Your job: estimate P(YES) for prediction markets.
    MANDATORY RESEARCH STEPS (do these searches FIRST):
    1. **Polymarket price**: Search "polymarket [topic]" to find current YES price - this is crucial market signal
    2. **Betting odds**: Search "[topic] betting odds" or "[topic] odds" for sportsbooks/prediction markets
    3. **Crypto prices**: If crypto-related, search "[coin] price " to get current exact price and the historical price for the last week
    4. **Recent news**: Search for latest developments on the specific event, use your own reasoning
    CRITICAL RULES:
    - Polymarket YES price is strong evidence - if market trades at 0.75, probability is ~0.75 unless you have strong contrary evidence
    - For sports: always check betting odds and convert to implied probability
    - For crypto price targets: current price vs target determines probability
    - Resolution criteria is ground truth - read exact wording
    - NEVER output exactly 0.0 or 1.0 - always 0.01-0.99 range
    - More time until deadline = more uncertainty = probability closer to 0.5"""
        USER_TEMPLATE = """**Today:** {today}
    **Question:** {title}
    **Description:** {description}
    **Deadline:** {cutoff}
    REQUIRED SEARCHES:
    1. Search Polymarket for this exact question to get current YES price
    2. Search for betting odds if sports/elections & convert betting odds to probability
    3. Search current and historical for the last week prices if crypto or stocks related
    4. Search recent news about this event
    If unsure - predict modest probability.
    **Output JSON only:**
    ```json
    {{
    "polymarket_price": <number 0-1 or null>,
    "betting_odds": <string or null>,
    "current_price": <string or null>,
    "key_facts": ["fact1", "fact2"],
    "reasoning": "5-7 sentences",
    "probability": <number 0-1>
    }}
    ```"""
        FALLBACK_TEMPLATE = """**Today:** {today}
    **Question:** {title}
    **Description:** {description}
    **Deadline:** {cutoff}
    **Research from web search:**
    {research}
    Based on the research above, estimate P(YES) for this prediction market.
    **Output JSON only:**
    ```json
    {{
    "polymarket_price": <number 0-1 or null>,
    "betting_odds": <string or null>,
    "current_price": <string or null>,
    "key_facts": ["fact1", "fact2"],
    "reasoning": "2-3 sentences",
    "probability": <number 0-1>
    }}
    ```"""
        # Postprocessing calibration
        ISOTONIC_BREAKPOINTS = {
            'other': (
                [0.01, 0.0196, 0.02, 0.0503, 0.0511, 0.0832, 0.087, 0.2111, 0.2123, 0.2748, 0.2752, 0.3089, 0.3098, 0.3403, 0.3421, 0.3769, 0.3775, 0.3894, 0.39, 0.42, 0.4211, 0.64, 0.6402, 0.6652, 0.6674, 0.7, 0.7028, 0.8376, 0.8381, 0.92, 0.9234, 0.9717, 0.9722, 0.978, 0.9783, 0.98, 0.9811, 0.99],
                [0.0, 0.0, 0.00147783, 0.00147783, 0.05927406, 0.05927406, 0.06500274, 0.06500274, 0.08750665, 0.08750665, 0.09934641, 0.09934641, 0.2037037, 0.2037037, 0.35227273, 0.35227273, 0.35384615, 0.35384615, 0.49159664, 0.49159664, 0.51661925, 0.51661925, 0.61111111, 0.61111111, 0.71428571, 0.71428571, 0.83089133, 0.83089133, 0.86666667, 0.86666667, 0.88782051, 0.88782051, 0.90909091, 0.90909091, 0.93421053, 0.93421053, 1.0, 1.0],
            ),
        }
        BETA_WEIGHTS = {'weather': [0.557913, 0.914001, -2.307990]}
        TEMPERATURE = {'app_store': 0.421422}
        @staticmethod
        def _logistic(val):
            val = min(max(val, -500.0), 500.0)
            return 1.0 / (1.0 + math.exp(-val))
        @staticmethod
        def _clamp(p):
            return min(max(p, 1e-10), 1.0 - 1e-10)
        
        @classmethod
        async def _call_with_retry(cls, client, payload, use_websearch=True):
            if use_websearch:
                payload["tools"] = [{"type": "web_search"}]
            elif "tools" in payload:
                del payload["tools"]
            for model in cls.MODELS:
                payload["model"] = model
                for attempt in range(3):
                    try:
                        resp = await client.post(f"{OPENAI_URL}/responses", json=payload, timeout=145.0)
                        if resp.status_code == 200:
                            data = resp.json()
                            return data, data.get("cost", 0.0)
                        if resp.status_code in (429, 500, 502, 503):
                            await asyncio.sleep(2.0 ** (attempt + 1))
                            continue
                        break
                    except httpx.TimeoutException:
                        break
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code in (429, 500, 502, 503):
                            await asyncio.sleep(2.0 ** (attempt + 1))
                        else:
                            break
            raise Exception("All models failed")
        
        @classmethod
        def _interpolate(cls, prob, xs, ys):
            prob = round(prob, 4)
            if prob <= xs[0]: return ys[0]
            if prob >= xs[-1]: return ys[-1]
            lo, hi = 0, len(xs) - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if xs[mid] <= prob: lo = mid
                else: hi = mid
            if xs[hi] == xs[lo]: return ys[lo]
            frac = (prob - xs[lo]) / (xs[hi] - xs[lo])
            return ys[lo] + frac * (ys[hi] - ys[lo])
        @classmethod
        def _detect_category(cls, event_data):
            known = {'sports', 'app store', 'weather', 'earnings'}
            for t in event_data.get('metadata', {}).get('topics', []):
                if t.lower() in known: return t.lower()
            combined = (event_data.get('title', '') + ' ' + event_data.get('description', '')).lower()
            title_lower = event_data.get('title', '').lower()
            if any(h in combined for h in [' vs ', ' vs. ', 'upcoming game', 'stoppage time', 'cricket', 'both teams to score']) or \
            any(h in title_lower for h in [' win ', ' win?']):
                return 'sports'
            if ' app ' in combined or 'app store' in combined: return 'app_store'
            if 'earnings' in combined or (('q1' in combined or 'q2' in combined or 'q3' in combined or 'q4' in combined) and 'above' in combined):
                return 'earnings'
            if 'temperature' in combined: return 'weather'
            return 'other'
        @classmethod
        def postprocess(cls, raw_prob, category):
            if category in cls.BETA_WEIGHTS:
                cp = cls._clamp(raw_prob)
                lo = math.log(cp / (1.0 - cp))
                nlc = -math.log(1.0 - cp)
                w = cls.BETA_WEIGHTS[category]
                return cls._logistic(w[0] + w[1] * lo + w[2] * nlc)
            if category in cls.TEMPERATURE:
                cp = cls._clamp(raw_prob)
                return cls._logistic(math.log(cp / (1.0 - cp)) / cls.TEMPERATURE[category])
            if category in cls.ISOTONIC_BREAKPOINTS:
                xs, ys = cls.ISOTONIC_BREAKPOINTS[category]
                return min(1.0, max(0.0, cls._interpolate(raw_prob, xs, ys)))
            return raw_prob
        @classmethod
        def _normalize(cls, val):
            if 0.0 <= val <= 1.0: return val
            if 1.0 < val <= 100.0: return val / 100.0
            return None
        @classmethod
        async def _fix_with_model(cls, client, raw_text):
            prompt = f"""Extract prediction data from this malformed response and return valid JSON.
    Raw response:
    {raw_text}
    Return ONLY valid JSON:
    {{"probability": <0-1>, "reasoning": "<extracted reasoning>"}}"""
            for model in ["gpt-5-nano", "gpt-5-mini"]:
                try:
                    payload = {"model": model, "input": [{"role": "user", "content": prompt}], "run_id": RUN_ID}
                    resp = await client.post(f"{OPENAI_URL}/responses", json=payload, timeout=60.0)
                    if resp.status_code != 200: continue
                    data = resp.json()
                    text = extract_openai_text(data)
                    cost = data.get("cost", 0.0)
                    parsed = safe_parse_json(text)
                    if parsed and "probability" in parsed:
                        p = float(parsed["probability"])
                        n = cls._normalize(p)
                        if n is not None:
                            parsed["probability"] = n
                            return {"data": parsed, "cost": cost}
                    m = re.search(r"probability[\"']?\s*[:=]\s*([\d.]+)", text, re.IGNORECASE)
                    if m:
                        p = float(m.group(1))
                        n = cls._normalize(p)
                        if n is not None:
                            return {"data": {"probability": max(0.01, min(0.99, n)), "reasoning": text}, "cost": cost}
                except Exception:
                    continue
            return {"data": {"probability": 0.35, "reasoning": "Failed to parse"}, "cost": 0.0}
        @classmethod
        async def _desearch_fallback(cls, client, query):
            try:
                resp = await client.post(f"{DESEARCH_URL}/web/search",
                    json={"query": query, "num": 10, "start": 0, "run_id": RUN_ID}, timeout=60.0)
                if resp.status_code != 200: return ""
                data = resp.json()
                results = data.get("data", [])
                if not results: return ""
                snippets = [f"- {r.get('title', '')}: {r.get('snippet', '')}" for r in results[:5]]
                text = "Search results:\n" + "\n".join(snippets)
                for r in results[:10]:
                    url = r.get("link", "")
                    if not url: continue
                    try:
                        cr = await client.post(f"{DESEARCH_URL}/web/crawl",
                            json={"url": url, "run_id": RUN_ID}, timeout=15.0)
                        if cr.status_code == 200:
                            content = cr.json().get("content", "")
                            if content:
                                return f"{text}\n\nCrawled content from {url}:\n{content}"
                    except Exception:
                        pass
                return text
            except Exception:
                return ""

        @classmethod
        async def _forecast_inner(cls, event):
            title = event.get("title", "")
            desc = event.get("description", "")
            cutoff = event.get("cutoff", "")
            today = datetime.now().strftime("%Y-%m-%d")
            total_cost = 0.0
            probs = []
            reasoning = None
            prompt = cls.USER_TEMPLATE.format(title=title, description=desc, cutoff=cutoff, today=today)
            async with TrackedAsyncClient(timeout=180.0) as client:
                payload = {"model": "gpt-5-pro",
                        "input": [{"role": "developer", "content": cls.SYSTEM_PROMPT},
                                    {"role": "user", "content": prompt}],
                        "run_id": RUN_ID}
                try:
                    data, cost = await cls._call_with_retry(client, payload, use_websearch=True)
                    total_cost += cost
                    text = extract_openai_text(data)
                except Exception:
                    # Desearch fallback
                    queries = [f"polymarket {title}", f"{title} betting odds", f"{title} latest news"]
                    parts = []
                    for q in queries:
                        r = await cls._desearch_fallback(client, q)
                        if r: parts.append(r)
                    if not parts:
                        return {"event_id": event.get("event_id", "?"), "prediction": 0.35,
                                "reasoning": "All search methods failed", "cost": 0.0}
                    fb_prompt = cls.FALLBACK_TEMPLATE.format(title=title, description=desc,
                        cutoff=cutoff, today=today, research="\n\n".join(parts))
                    payload2 = {"model": "gpt-5.2",
                                "input": [{"role": "developer", "content": cls.SYSTEM_PROMPT},
                                        {"role": "user", "content": fb_prompt}],
                                "run_id": RUN_ID}
                    data, cost = await cls._call_with_retry(client, payload2, use_websearch=False)
                    total_cost += cost
                    text = extract_openai_text(data)
                parsed = safe_parse_json(text)
                if not parsed or "probability" not in parsed:
                    fix = await cls._fix_with_model(client, text)
                    parsed = fix["data"]
                    total_cost += fix["cost"]
                raw = float(parsed.get("probability", 0.5))
                norm = cls._normalize(raw)
                prob = max(0.01, min(0.99, norm if norm is not None else 0.5))
                probs.append(prob)
                reasoning = parsed.get("reasoning", "")
            if not probs:
                return {"event_id": event.get("event_id", "?"), "prediction": 0.35,
                        "reasoning": reasoning or "Timeout fallback", "cost": total_cost}
            mean_p = sum(probs) / len(probs)
            final_p = max(0.01, min(0.99, mean_p))
            return {"event_id": event.get("event_id", "?"), "prediction": final_p,
                    "reasoning": reasoning or "", "cost": total_cost}
        @classmethod
        async def run(cls, event_data: dict, category: str) -> dict:
            # category = cls._detect_category(event_data)
            print(f"[WEATHER] category={category}")
            try:
                result = await asyncio.wait_for(cls._forecast_inner(event_data), timeout=cls.TIMEOUT)
            except asyncio.TimeoutError:
                result = {"event_id": event_data.get("event_id", "?"), "prediction": 0.35,
                        "reasoning": "Timeout fallback", "cost": 0.0}
            except Exception as e:
                result = {"event_id": event_data.get("event_id", "?"), "prediction": 0.35,
                        "reasoning": f"Error: {e}", "cost": 0.0}
            raw_p = result["prediction"]
            result["prediction"] = cls.postprocess(raw_p, category)
            print(f"[WEATHER] {category}: {raw_p:.4f} -> {result['prediction']:.4f}")
            return {"event_id": result["event_id"], "prediction": result["prediction"],
                    "reasoning": str(result.get("reasoning", ""))[:2000]}


    class SportsHandler:
        """
        Handler for Sports events.
        Architecture: Query generation (small model) → OpenAI web search forecast
        (model cascade) → repair fallback.
        """
        QUERY_MODEL = "gpt-5-nano"
        FORECAST_MODELS = ["gpt-5.2", "gpt-5", "gpt-5-mini"]
        REPAIR_MODEL = "gpt-5-nano"
        MAX_RETRIES = 5
        BACKOFF = 2.0
        QUERY_TIMEOUT = 120.0
        FORECAST_TIMEOUT = 180.0
        REPAIR_TIMEOUT = 60.0

        FORECASTER_SYSTEM_PROMPT = """You are an expert forecaster with web search. Your job: estimate P(YES) for prediction markets.
    MANDATORY RESEARCH STEPS (do these searches FIRST):
    1. **Polymarket price**: Search "polymarket [topic]" to find current YES price - this is crucial market signal
    2. **Betting odds**: Search "[topic] betting odds" or "[topic] odds" for sportsbooks/prediction markets
    3. **Crypto prices**: If crypto-related, search "[coin] price" to get current exact price
    4. **Recent news**: Search for latest developments on the specific event, use your own reasoning
    CRITICAL RULES:
    - Polymarket YES price is strong evidence - if market trades at 0.75, probability is ~0.75 unless you have strong contrary evidence
    - For sports: always check betting odds and convert to implied probability
    - For crypto price targets: current price vs target determines probability
    - Resolution criteria is ground truth - read exact wording carefully
    - NEVER output exactly 0.0 or 1.0 - always 0.01-0.99 range
    - More time until deadline = more uncertainty = probability closer to 0.5"""

        QUERY_BUILDER_PROMPT = """Today: {today}
    Event title: {title}
    Description: {description}
    Deadline: {cutoff}
    Generate 6 to 8 diverse search queries that would help a forecaster answer this question.
    Include a query to check if the event already occurred.
    Include Polymarket and betting odds queries when relevant.
    Return JSON only:
    ```json
    {{
    "search_queries": ["query 1", "query 2", "query 3"]
    }}
    ```"""

        # --- JSON / query parsing helpers ---
        @classmethod
        def _parse_json(cls, text):
            if not text or not text.strip():
                return None
            m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except (json.JSONDecodeError, ValueError):
                    pass
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                pass
            m = re.search(r'\{[^{}]*"probability"[^{}]*\}', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except (json.JSONDecodeError, ValueError):
                    pass
            return None

        @classmethod
        def _extract_queries(cls, text):
            parsed = cls._parse_json(text)
            if isinstance(parsed, dict):
                queries = parsed.get("search_queries") or parsed.get("queries")
                if isinstance(queries, list):
                    if queries and isinstance(queries[0], dict):
                        return [q.get("query") for q in queries if isinstance(q, dict) and q.get("query")]
                    return [q for q in queries if isinstance(q, str)]
            return []

        @staticmethod
        def _bound_prob(val, lo=0.01, hi=0.99):
            return max(lo, min(hi, val))

        # --- API call with model cascade and retry ---
        @classmethod
        async def _api_call(cls, client, payload, models, max_retries, timeout):
            last_err = None
            for model in models:
                payload["model"] = model
                for attempt in range(max_retries):
                    try:
                        print(f"[SPORTS][API] Model: {model}, Attempt: {attempt + 1}")
                        resp = await client.post(
                            f"{OPENAI_URL}/responses", json=payload, timeout=timeout
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            return data, data.get("cost", 0.0)
                        if resp.status_code in (429, 500, 502, 503):
                            delay = cls.BACKOFF ** (attempt + 1)
                            print(f"[SPORTS][RETRY] HTTP {resp.status_code}, waiting {delay:.1f}s")
                            await asyncio.sleep(delay)
                            continue
                        print(f"[SPORTS][ERROR] HTTP {resp.status_code}")
                        break
                    except httpx.TimeoutException:
                        delay = cls.BACKOFF ** (attempt + 1)
                        print(f"[SPORTS][TIMEOUT] Waiting {delay:.1f}s")
                        await asyncio.sleep(delay)
                    except Exception as e:
                        last_err = e
                        print(f"[SPORTS][ERROR] {type(e).__name__}: {str(e)[:100]}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(cls.BACKOFF ** attempt)
                        break
            raise last_err or RuntimeError("All API calls failed")

        # --- Query generation ---
        @classmethod
        async def _generate_queries(cls, client, event_data):
            prompt = cls.QUERY_BUILDER_PROMPT.format(
                title=event_data.get("title", ""),
                description=event_data.get("description", ""),
                cutoff=event_data.get("cutoff", ""),
                today=TODAY_STR,
            )
            payload = {
                "model": cls.QUERY_MODEL,
                "input": [{"role": "user", "content": prompt}],
                "run_id": RUN_ID,
            }
            try:
                data, _ = await cls._api_call(client, payload, [cls.QUERY_MODEL], cls.MAX_RETRIES, cls.QUERY_TIMEOUT)
                text = extract_openai_text(data)
                queries = cls._extract_queries(text)
                if queries:
                    return queries[:8]
            except Exception as e:
                print(f"[SPORTS][QUERY_GEN] Failed: {e}")
            title = event_data.get("title", "").strip()
            if not title:
                return ["latest news", "official announcement", "polymarket odds", "betting odds"]
            return [
                f"{title} latest news",
                f"{title} official announcement",
                f"polymarket {title}",
                f"{title} betting odds",
                f"{title} forecast",
            ][:8]

        # --- Repair malformed response ---
        @classmethod
        async def _repair(cls, client, broken_text):
            repair_prompt = (
                "Extract prediction data from this malformed response and return valid JSON.\n"
                f"Raw response:\n{broken_text[:2000]}\n"
                'Return ONLY valid JSON:\n{"probability": <0-1>, "reasoning": "<extracted reasoning>"}'
            )
            payload = {
                "model": cls.REPAIR_MODEL,
                "input": [{"role": "user", "content": repair_prompt}],
                "run_id": RUN_ID,
            }
            try:
                data, cost = await cls._api_call(client, payload, [cls.REPAIR_MODEL], 3, cls.REPAIR_TIMEOUT)
                text = extract_openai_text(data)
                parsed = cls._parse_json(text)
                if parsed and "probability" in parsed:
                    return {"data": parsed, "cost": cost}
                m = re.search(r"probability[\"']?\s*[:=]\s*([\d.]+)", text, re.IGNORECASE)
                if m:
                    prob = float(m.group(1))
                    return {"data": {"probability": cls._bound_prob(prob), "reasoning": text[:500]}, "cost": cost}
            except Exception as e:
                print(f"[SPORTS][REPAIR] Failed: {e}")
            return {"data": {"probability": 0.35, "reasoning": "Repair failed"}, "cost": 0.0}

        # --- Forecast with web search ---
        @classmethod
        async def _forecast(cls, client, event_data, queries):
            title = event_data.get("title", "")
            desc = event_data.get("description", "")
            cutoff = event_data.get("cutoff", "")
            query_list = "\n".join([f"- {q}" for q in queries]) or "- (no queries available)"
            user_prompt = f"""**Today:** {TODAY_STR}
    **Question:** {title}
    **Description:** {desc}
    **Deadline:** {cutoff}
    Suggested search queries:
    {query_list}
    REQUIRED SEARCHES:
    1. Search Polymarket for this exact question to get current YES price
    2. Search for betting odds if sports/elections
    3. Search current price if crypto-related
    4. Search recent news about this event
    **Output JSON only:**
    ```json
    {{
    "polymarket_price": <number 0-1 or null>,
    "betting_odds": <string or null>,
    "current_price": <string or null>,
    "key_facts": ["fact1", "fact2"],
    "reasoning": "2-3 sentences",
    "probability": <number 0-1>
    }}
    ```"""
            payload = {
                "model": cls.FORECAST_MODELS[0],
                "input": [
                    {"role": "developer", "content": cls.FORECASTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "tools": [{"type": "web_search"}],
                "run_id": RUN_ID,
            }
            total_cost = 0.0
            try:
                data, cost = await cls._api_call(client, payload, cls.FORECAST_MODELS, cls.MAX_RETRIES, cls.FORECAST_TIMEOUT)
                total_cost += cost
                text = extract_openai_text(data)
                print(f"[SPORTS][RESPONSE]\n{text[:1000]}")
                parsed = cls._parse_json(text)
                if not parsed or "probability" not in parsed:
                    print("[SPORTS] Response parsing failed, attempting repair...")
                    repair_result = await cls._repair(client, text)
                    parsed = repair_result["data"]
                    total_cost += repair_result["cost"]
                prob = float(parsed.get("probability", 0.5))
                validated = cls._bound_prob(prob)
                reasoning = parsed.get("reasoning", "")
                return {"prediction": validated, "reasoning": reasoning, "cost": total_cost}
            except Exception as e:
                print(f"[SPORTS][FORECAST] Failed: {e}")
                return {"prediction": 0.35, "reasoning": f"Forecasting error: {str(e)[:100]}", "cost": total_cost}

        # --- Main entry ---
        @classmethod
        async def run(cls, event_data: dict, category: str) -> dict:
            eid = event_data.get("event_id", "?")
            print(f"\n[SPORTS] Starting forecast for {eid}")
            start = time.time()
            async with TrackedAsyncClient(timeout=180.0) as client:
                queries = await cls._generate_queries(client, event_data)
                print(f"[SPORTS] Generated {len(queries)} queries")
                result = await cls._forecast(client, event_data, queries)
            duration = time.time() - start
            print(f"[SPORTS] Done in {duration:.1f}s | P={result['prediction']:.3f}")
            return {
                "event_id": eid,
                "prediction": result["prediction"],
                "reasoning": str(result.get("reasoning", ""))[:2000],
            }

    class AppStoreHandler:
        """
        Handler for App Store events.
        Architecture: Primary OpenAI forecast → Desearch rescue → Verification pass →
                    Tier2 Chutes fallback → Platt calibration → blended output.
        """
        # --- Engine lists ---
        TIER1_ENGINES = ["gpt-5.2", "gpt-5", "gpt-5-mini"]
        TIER2_ENGINES = ["Qwen/Qwen3-235B-A22B-Instruct-2507", "openai/gpt-oss-120b"]
        PREMIUM_ENGINE = "gpt-5.2"
        REPAIR_ENGINES = ["gpt-5-nano", "gpt-5-mini"]
        # --- Resilience ---
        MAX_ROUNDS = 4
        BACKOFF_FACTOR = 2.0
        HARD_TIMEOUT = 145.0
        SESSION_TIMEOUT = 180.0
        CRAWL_TIMEOUT = 15.0
        SEARCH_TIMEOUT = 60.0
        REPAIR_TIMEOUT = 60.0
        TRANSIENT_CODES = (429, 500, 502, 503)
        # --- Probability bounds ---
        PROB_SOFT_FLOOR = 0.05
        PROB_SOFT_CEIL = 0.95
        PROB_HARD_FLOOR = 0.01
        PROB_HARD_CEIL = 0.99
        PROB_NEUTRAL = 0.35
        # --- Platt calibration ---
        PLATT_PARAMS = {
            'all': {'method': 'platt', 'a': 5.112598, 'b': -2.982271},
            'weather': {'method': 'platt', 'a': -4.309746, 'b': -0.538805},
            'app_store': {'method': 'temperature', 'temp': 0.616896},
            'other': {'method': 'platt', 'a': 7.052620, 'b': -3.861737},
        }
        # --- Prompts ---
        ANALYST_SYSTEM = """You are an expert forecaster for prediction markets. Estimate P(YES) with rigorous research.
    RESEARCH APPROACH - adapt your search strategy to the event type:
    For competitions and matches:
    - Search betting odds and convert to probability (decimal odds D → prob ≈ 1/D minus margin)
    - Check recent form, injuries, head-to-head records, rankings
    - Home advantage matters in team sports (+10-15%)
    For political events and elections:
    - Search Polymarket/PredictIt first - market prices are strong signals
    - For elections: check polling aggregates (538, RCP), consider historical polling errors
    - For policy/diplomatic events: prioritize official sources (Reuters, AP, government statements)
    - Check procedural requirements (votes needed, veto power, legislative calendar)
    For economic and financial events:
    - Search market-implied probabilities (CME FedWatch for rates, futures markets)
    - Check central bank communications and forward guidance
    - Economic calendar: what data releases occur before cutoff?
    For product launches and technology:
    - Check official company channels, press releases, SEC filings
    - Consider historical track record (announced vs actual delivery dates)
    - Distinguish between: announced, shipped, generally available
    For entertainment and awards:
    - Search prediction markets and expert consensus sites
    - Box office tracking, review aggregates
    - Awards predictions converge closer to ceremony date
    ALWAYS DO THESE:
    1. Search "Polymarket [topic]" - if market exists, price ≈ probability
    2. Search recent news (prioritize last 48-72 hours)
    3. Verify key claims with multiple sources
    4. Consider time until cutoff (more time = more uncertainty)
    ANALYSIS PRINCIPLES:
    - Polymarket price is your anchor - deviate only with strong contrary evidence
    - Official sources > speculation and rumors
    - Consider base rates: how often do similar events happen?
    - Resolution criteria are literal - read exact wording carefully
    - Range: never return exactly 0 or 1, use [0.01, 0.99]
    OUTPUT FORMAT:
    PREDICTION: [0.01-0.99]
    REASONING: [Key evidence, market signal if found, main uncertainties, 3-5 sentences]"""

        INQUIRY_CARD = """**Today:** {today}
    **Question:** {title}
    **Description:** {description}
    **Deadline:** {cutoff}
    REQUIRED SEARCHES:
    1. Search Polymarket for this exact question to get current YES price
    2. Search for betting odds if sports/elections & convert betting odds to probability
    3. Search current and historical for the last week prices if crypto or stocks related
    4. Search recent news about this event
    If there is contradictory information, add it to the contradictory_facts list.
    If unsure - predict modest probability.
    **Output JSON only:**
    ```json
    {{
    "polymarket_price": <number 0-1 or null>,
    "betting_odds": <string or null>,
    "current_price": <string or null>,
    "key_facts": ["fact1", "fact2"],
    "contradictory_facts": ["fact1", "fact2"],
    "reasoning": "5-7 sentences",
    "probability": <number 0-1>
    }}
    ```"""

        RESCUE_CARD = """**Today:** {today}
    **Question:** {title}
    **Description:** {description}
    **Deadline:** {cutoff}
    **Research from web search:**
    {research}
    Based on the research above, estimate P(YES) for this prediction market.
    **Output JSON only:**
    ```json
    {{
    "polymarket_price": <number 0-1 or null>,
    "betting_odds": <string or null>,
    "current_price": <string or null>,
    "key_facts": ["fact1", "fact2"],
    "reasoning": "2-3 sentences",
    "probability": <number 0-1>
    }}
    ```"""

        VERIFIER_SYSTEM = """You are a research validator and refinement agent for prediction markets.
    Your job:
    1. Review the first forecast's research and reasoning
    2. Identify if critical information is missing or uncertain
    3. If gaps exist, USE WEB SEARCH to find the missing information
    4. Output final adjusted probability based on all available evidence
    You have web search capability - use it if the first forecast is missing key data like Polymarket price, betting odds, or recent news."""

        VERIFIER_CARD = """**Today:** {today}
    **Question:** {title}
    **Description:** {description}
    **Deadline:** {cutoff}
    **First Forecast Result:**
    - Probability: {probability}
    - Polymarket Price: {polymarket_price}
    - Betting Odds: {betting_odds}
    - Key Facts: {key_facts}
    - Reasoning: {reasoning}
    **Your Task:**
    1. Evaluate if this forecast has sufficient research
    2. If Polymarket price is missing or key information is uncertain or contradictory, SEARCH for it now
    3. If you find new information that changes the probability, adjust it
    4. Output your final probability (can be same as first if research was sufficient)
    **Output JSON only:**
    ```json
    {{
    "searched_for": ["query1", "query2"] or null,
    "new_findings": ["finding1", "finding2"] or null,
    "reasoning": "3-5 sentences explaining your final assessment",
    "probability": <number 0-1>
    }}
    ```"""

        # --- Signal classification ---
        _DOMAIN_SIGNALS = {
            "sports": {"MATCH", "GAME", "WIN", "VS", "CHAMPIONSHIP", "LEAGUE", "CUP", "PLAYOFF", "GOAL", "SCORE"},
            "politics": {"ELECTION", "VOTE", "POLL", "PRESIDENT", "GOVERNOR", "SENATOR", "MAYOR", "TARIFF", "SANCTION", "WAR", "TREATY"},
            "economy": {"RATE", "FED", "GDP", "INFLATION", "BITCOIN", "CRYPTO", "PRICE", "MARKET", "$", "STOCK"},
            "tech": {"LAUNCH", "RELEASE", "APP", "SOFTWARE", "UPDATE", "SHIP", "ANNOUNCE"},
            "entertainment": {"MOVIE", "FILM", "OSCAR", "GRAMMY", "EMMY", "ALBUM", "BOX_OFFICE", "AWARD"},
        }
        _SEARCH_HINTS = {
            "sports": ["betting odds", "injuries", "recent form", "head-to-head", "standings"],
            "politics": ["Polymarket", "polling", "official statement", "Reuters AP"],
            "economy": ["FedWatch", "central bank", "market expectations", "economic data"],
            "tech": ["official blog", "press release", "SEC filing", "launch date"],
            "entertainment": ["box_office", "reviews", "awards predictions", "release date"],
            "general": ["Polymarket", "recent news", "official source"],
        }
        KNOWN_TOPICS = {'sports', 'app_store', 'weather', 'earnings', 'election', 'inflation', 'price'}
        _SPORT_BODY_KWS = (' vs ', ' vs. ', 'upcoming game', 'stoppage time', 'cricket', 'both teams to score')
        _SPORT_TITLE_KWS = (' win ', ' win?')

        # --- Utility methods ---
        @staticmethod
        def _sigmoid(x):
            x = max(-500.0, min(500.0, x))
            return 1.0 / (1.0 + math.exp(-x))

        @staticmethod
        def _logit(p):
            p = max(1e-10, min(1 - 1e-10, p))
            return math.log(p / (1 - p))

        @classmethod
        def _soft_clamp(cls, v):
            return max(cls.PROB_SOFT_FLOOR, min(cls.PROB_SOFT_CEIL, v))

        @classmethod
        def _hard_clamp(cls, v):
            return max(cls.PROB_HARD_FLOOR, min(cls.PROB_HARD_CEIL, v))

        @staticmethod
        def _normalize_raw(v):
            if 0.0 <= v <= 1.0:
                return v
            if 1.0 < v <= 100.0:
                return v / 100.0
            return None

        @classmethod
        def _detect_domain(cls, ev):
            blob = (ev.get("title", "") + " " + ev.get("description", "")).upper()
            for domain, keywords in cls._DOMAIN_SIGNALS.items():
                if any(kw in blob for kw in keywords):
                    return domain
            return "general"

        @classmethod
        def _resolve_category(cls, ev):
            for t in ev.get('metadata', {}).get('topics', []):
                if t.lower() in cls.KNOWN_TOPICS:
                    return t
            title = ev.get('title', '')
            desc = ev.get('description', '')
            combined = (title + ' ' + desc).lower()
            title_low = title.lower()
            if 'election' in combined:
                return 'election'
            if any(k in combined for k in cls._SPORT_BODY_KWS) or any(k in title_low for k in cls._SPORT_TITLE_KWS):
                return 'Sports'
            if ' app ' in combined or 'app store' in combined:
                return 'App_Store'
            if ' price of ' in combined:
                return 'price'
            if 'earnings' in combined or (any(q in combined for q in ['q1', 'q2', 'q3', 'q4']) and 'above' in combined):
                return 'Earnings'
            if 'inflation' in combined:
                return 'inflation'
            if ' temperature ' in combined:
                return 'Weather'
            return 'Other'

        @classmethod
        def _platt_calibrate(cls, p, category_key):
            spec = cls.PLATT_PARAMS.get(category_key)
            if not spec:
                return p
            if spec['method'] == 'platt':
                z = spec['a'] * p + spec['b']
                return cls._sigmoid(z)
            if spec['method'] == 'temperature':
                logit_val = cls._logit(p)
                return cls._sigmoid(logit_val / spec['temp'])
            return p

        # --- JSON / text extraction ---
        _FENCED_RE = re.compile(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```')
        _PROB_FIELD_RE = re.compile(r'\{[^{}]*"probability"[^{}]*\}', re.DOTALL)
        _DIGITS_RE = re.compile(r"[\d.]+")
        _BARE_FLOAT_RE = re.compile(r"\b0\.\d{1,3}\b")
        _PROB_KEYS = ("probability", "prediction", "prob", "p", "final_probability")
        _REASON_KEYS = ("reasoning", "reason", "analysis", "explanation")
        _REGEX_CHAINS = [
            re.compile(r"probability[:\s]+(?:is\s+)?(\d*\.?\d+)"),
            re.compile(r"estimate[:\s]+(\d*\.?\d+)"),
            re.compile(r"predict[:\s]+(\d*\.?\d+)"),
            re.compile(r"(\d*\.?\d+)\s*probability"),
            re.compile(r"(\d{1,2}(?:\.\d+)?)\s*%"),
        ]

        @classmethod
        def _try_json(cls, raw):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None

        @classmethod
        def _extract_json(cls, text):
            if not text:
                return None
            m = cls._FENCED_RE.search(text)
            if m:
                r = cls._try_json(m.group(1))
                if r is not None:
                    return r
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                r = cls._try_json(text[start:end + 1])
                if r is not None:
                    return r
            return cls._try_json(text.strip())

        @classmethod
        def _extract_json_minimal(cls, raw):
            m = cls._FENCED_RE.search(raw)
            if m:
                parsed = cls._try_json(m.group(1))
                if parsed:
                    return parsed
            parsed = cls._try_json(raw)
            if parsed:
                return parsed
            m2 = cls._PROB_FIELD_RE.search(raw)
            if m2:
                return cls._try_json(m2.group())
            return None

        @classmethod
        def _full_parse(cls, text):
            """Parse probability and reasoning from text using multiple strategies."""
            if not text:
                return 0.5, "Empty response"
            prob, reason = None, ""
            # Structured lines
            for line in text.strip().splitlines():
                upper = line.upper()
                if "PREDICTION:" in upper:
                    m = cls._DIGITS_RE.search(line)
                    if m:
                        try:
                            prob = cls._hard_clamp(float(m.group()))
                        except (ValueError, TypeError):
                            pass
                elif "REASONING:" in upper:
                    _, _, after = line.partition(":")
                    reason = after.strip()
            # JSON body
            if prob is None:
                parsed = cls._extract_json(text)
                if parsed:
                    for k in cls._PROB_KEYS:
                        v = parsed.get(k)
                        if v is not None:
                            try:
                                prob = cls._hard_clamp(float(v))
                                break
                            except (ValueError, TypeError):
                                continue
                    for k in cls._REASON_KEYS:
                        v = parsed.get(k)
                        if v:
                            reason = str(v)
                            break
            # Regex chains
            if prob is None:
                lowered = text.lower()
                for pat in cls._REGEX_CHAINS:
                    m = pat.search(lowered)
                    if m:
                        try:
                            val = float(m.group(1))
                            if val > 1:
                                val /= 100.0
                            prob = cls._hard_clamp(val)
                            break
                        except (ValueError, TypeError):
                            continue
            # Bare decimal
            if prob is None:
                m = cls._BARE_FLOAT_RE.search(text)
                if m:
                    try:
                        prob = cls._hard_clamp(float(m.group()))
                    except (ValueError, TypeError):
                        pass
            if not reason and text:
                reason = text[:500].replace("\n", " ").strip()
            if prob is None:
                return 0.5, reason or "Could not parse prediction"
            return prob, reason

        @classmethod
        def _extract_text(cls, blob):
            for block in blob.get("output", []):
                if block.get("type") != "message":
                    continue
                for part in block.get("content", []):
                    if part.get("type") in ("output_text", "text") and part.get("text"):
                        return part["text"]
            return ""

        # --- API call helpers ---
        @classmethod
        async def _openai_call(cls, http, model, messages, tools=None):
            body = {"model": model, "input": messages, "run_id": RUN_ID}
            if tools:
                body["tools"] = tools
            idx = 0
            while True:
                try:
                    r = await http.post(f"{OPENAI_URL}/responses", json=body, timeout=cls.HARD_TIMEOUT)
                    if r.status_code == 200:
                        blob = r.json()
                        return cls._extract_text(blob), blob.get("cost", 0.0)
                    if r.status_code in cls.TRANSIENT_CODES and idx < cls.MAX_ROUNDS - 1:
                        await asyncio.sleep(cls.BACKOFF_FACTOR ** (idx + 1))
                        idx += 1
                        continue
                    r.raise_for_status()
                except httpx.TimeoutException:
                    if idx >= cls.MAX_ROUNDS - 1:
                        raise
                    await asyncio.sleep(cls.BACKOFF_FACTOR ** (idx + 1))
                    idx += 1
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code not in cls.TRANSIENT_CODES or idx >= cls.MAX_ROUNDS - 1:
                        raise
                    await asyncio.sleep(cls.BACKOFF_FACTOR ** (idx + 1))
                    idx += 1

        @classmethod
        async def _chutes_call(cls, http, model, prompt, max_tokens=2000):
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "run_id": RUN_ID,
            }
            try:
                idx = 0
                while True:
                    try:
                        r = await http.post(f"{CHUTES_URL}/chat/completions", json=body, timeout=60.0)
                        if r.status_code == 200:
                            blob = r.json()
                            choices = blob.get("choices", [])
                            return choices[0].get("message", {}).get("content", "") if choices else None
                        if r.status_code in cls.TRANSIENT_CODES and idx < 1:
                            await asyncio.sleep(cls.BACKOFF_FACTOR ** (idx + 1))
                            idx += 1
                            continue
                        return None
                    except httpx.TimeoutException:
                        if idx >= 1:
                            return None
                        await asyncio.sleep(cls.BACKOFF_FACTOR ** (idx + 1))
                        idx += 1
            except Exception:
                return None

        @classmethod
        async def _web_search(cls, http, query):
            body = {"query": query, "model": "NOVA", "run_id": RUN_ID}
            try:
                r = await http.post(
                    f"{DESEARCH_URL}/web/search".replace("/web/search", "/search"),
                    json=body, timeout=cls.SEARCH_TIMEOUT,
                )
                r.raise_for_status()
                return r.json().get("results", [])
            except Exception:
                return []

        @classmethod
        async def _crawl_page(cls, http, url):
            try:
                r = await http.post(
                    f"{DESEARCH_URL}/web/crawl",
                    json={"url": url, "run_id": RUN_ID},
                    timeout=cls.CRAWL_TIMEOUT,
                )
                if r.status_code == 200:
                    content = r.json().get("content", "")
                    if content:
                        return content
            except Exception:
                pass
            return None

        @classmethod
        async def _desearch_lookup(cls, http, q):
            try:
                r = await http.post(
                    f"{DESEARCH_URL}/web/search",
                    json={"query": q, "num": 10, "start": 0, "run_id": RUN_ID},
                    timeout=cls.SEARCH_TIMEOUT,
                )
                if r.status_code != 200:
                    return ""
                hits = r.json().get("data", [])
                if not hits:
                    return ""
            except Exception:
                return ""
            snippet_block = "Search results:\n" + "\n".join(
                f"- {h.get('title', '')}: {h.get('snippet', '')}" for h in hits[:5]
            )
            for h in hits[:10]:
                link = h.get("link", "")
                if not link:
                    continue
                page = await cls._crawl_page(http, link)
                if page:
                    return f"{snippet_block}\n\nCrawled content from {link}:\n{page}"
            return snippet_block

        @classmethod
        async def _repair_garbled(cls, http, garbled):
            prompt = (
                "Extract prediction data from this malformed response and return valid JSON.\n"
                f"Raw response:\n{garbled}\n"
                'Return ONLY valid JSON:\n{"probability": <0-1>, "reasoning": "<extracted reasoning>"}'
            )
            for eng in cls.REPAIR_ENGINES:
                try:
                    r = await http.post(
                        f"{OPENAI_URL}/responses",
                        json={"model": eng, "input": [{"role": "user", "content": prompt}], "run_id": RUN_ID},
                        timeout=cls.REPAIR_TIMEOUT,
                    )
                    if r.status_code != 200:
                        continue
                    blob = r.json()
                    txt = cls._extract_text(blob)
                    expense = blob.get("cost", 0.0)
                    parsed = cls._extract_json_minimal(txt)
                    if parsed and "probability" in parsed:
                        raw_p = float(parsed["probability"])
                        normed = cls._normalize_raw(raw_p)
                        if normed is not None:
                            parsed["probability"] = normed
                            return {"data": parsed, "cost": expense}
                    m = re.search(r"probability[\"']?\s*[:=]\s*([\d.]+)", txt, re.IGNORECASE)
                    if m:
                        raw_p = float(m.group(1))
                        normed = cls._normalize_raw(raw_p)
                        if normed is not None:
                            return {"data": {"probability": cls._hard_clamp(normed), "reasoning": txt}, "cost": expense}
                except Exception:
                    pass
            return {"data": {"probability": cls.PROB_NEUTRAL, "reasoning": "Failed to parse"}, "cost": 0.0}

        # --- Orchestration methods ---
        @classmethod
        async def _dispatch_with_cascade(cls, http, payload, web_tools=True):
            if web_tools:
                payload["tools"] = [{"type": "web_search"}]
            elif "tools" in payload:
                del payload["tools"]
            for eng in [cls.PREMIUM_ENGINE] + cls.TIER1_ENGINES:
                payload["model"] = eng
                for attempt in range(cls.MAX_ROUNDS):
                    try:
                        r = await http.post(f"{OPENAI_URL}/responses", json=payload, timeout=cls.HARD_TIMEOUT)
                        if r.status_code == 200:
                            blob = r.json()
                            return blob, blob.get("cost", 0.0)
                        if r.status_code in cls.TRANSIENT_CODES:
                            wait = cls.BACKOFF_FACTOR ** (attempt + 1)
                            await asyncio.sleep(wait)
                            continue
                        break
                    except httpx.TimeoutException:
                        break
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code in cls.TRANSIENT_CODES:
                            await asyncio.sleep(cls.BACKOFF_FACTOR ** (attempt + 1))
                            continue
                        break
                    except Exception:
                        break
            raise Exception("All engines exhausted")

        @classmethod
        async def _primary_forecast(cls, http, ev, today):
            user_msg = cls.INQUIRY_CARD.format(
                title=ev.get("title", ""), description=ev.get("description", ""),
                cutoff=ev.get("cutoff", ""), today=today,
            )
            payload = {
                "model": cls.PREMIUM_ENGINE,
                "input": [
                    {"role": "developer", "content": cls.ANALYST_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                "run_id": RUN_ID,
            }
            blob, cost = await cls._dispatch_with_cascade(http, payload, web_tools=True)
            txt = cls._extract_text(blob)
            print(f"[APP_STORE][PRIMARY]\n{txt[:500]}")
            return cls._extract_json_minimal(txt), cost

        @classmethod
        async def _desearch_rescue(cls, http, ev, today):
            title = ev.get("title", "")
            queries = [f"polymarket {title}", f"{title} betting odds", f"{title} latest news"]
            fragments = []
            for q in queries:
                r = await cls._desearch_lookup(http, q)
                if r:
                    fragments.append(r)
            if not fragments:
                return None, 0.0
            research = "\n\n".join(fragments)
            rescue_msg = cls.RESCUE_CARD.format(
                title=title, description=ev.get("description", ""),
                cutoff=ev.get("cutoff", ""), today=today, research=research,
            )
            payload = {
                "model": cls.PREMIUM_ENGINE,
                "input": [
                    {"role": "developer", "content": cls.ANALYST_SYSTEM},
                    {"role": "user", "content": rescue_msg},
                ],
                "run_id": RUN_ID,
            }
            blob, cost = await cls._dispatch_with_cascade(http, payload, web_tools=False)
            txt = cls._extract_text(blob)
            print(f"[APP_STORE][RESCUE]\n{txt[:500]}")
            return cls._extract_json_minimal(txt), cost

        @classmethod
        async def _verification_pass(cls, http, ev, initial, today):
            pd = initial.get("parsed_data", {})
            prompt = cls.VERIFIER_CARD.format(
                today=today,
                title=ev.get("title", ""),
                description=ev.get("description", ""),
                cutoff=ev.get("cutoff", ""),
                probability=initial.get("prediction", 0.5),
                polymarket_price=pd.get("polymarket_price", "not found"),
                betting_odds=pd.get("betting_odds", "not found"),
                key_facts=pd.get("key_facts", []),
                reasoning=initial.get("reasoning", ""),
            )
            payload = {
                "model": cls.TIER1_ENGINES[0],
                "input": [
                    {"role": "developer", "content": cls.VERIFIER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "tools": [{"type": "web_search"}],
                "run_id": RUN_ID,
            }
            try:
                blob, expense = await cls._dispatch_with_cascade(http, payload, web_tools=True)
                txt = cls._extract_text(blob)
                print(f"[APP_STORE][VERIFY]\n{txt[:500]}")
                parsed = cls._extract_json_minimal(txt)
                if parsed and "probability" in parsed:
                    return parsed, expense
                return {"probability": initial.get("prediction", 0.5), "reasoning": txt}, expense
            except Exception:
                return {"probability": initial.get("prediction", 0.5)}, 0.0

        @classmethod
        async def _tier2_forecast(cls, http, ev):
            domain = cls._detect_domain(ev)
            hints = cls._SEARCH_HINTS.get(domain, cls._SEARCH_HINTS["general"])
            blob_text = ev.get("title", "") + "\n" + ev.get("description", "")
            deadline = ev.get("cutoff", "")
            hint_csv = ", ".join(hints)
            tpl = "\n".join([
                "Generate 4 specific search queries for this forecasting event.",
                f"Consider searching for: {hint_csv}",
                "Event: {event}",
                "Cutoff: {cutoff}",
                "Today: {today}",
                '{{"queries": ["query1", "query2", "query3", "query4"]}}',
            ])
            filled = tpl.format(event=blob_text[:1000], cutoff=deadline, today=TODAY_STR)
            queries = None
            for eng in cls.TIER2_ENGINES[:2]:
                raw = await cls._chutes_call(http, eng, filled, max_tokens=500)
                if raw:
                    j = cls._extract_json(raw)
                    if j and "queries" in j:
                        queries = j["queries"][:4]
                        break
            if not queries:
                short = ev.get("title", "")[:80]
                queries = [f"Polymarket {short}", short]
            snippets = []
            for q in queries:
                results = await cls._web_search(http, q)
                for item in results[:3]:
                    url = item.get("url", "source")
                    title = item.get("title", "")
                    snip = item.get("snippet", "")[:300]
                    snippets.append(f"[{url}] {title}: {snip}")
            evidence = "\n\n".join(snippets) if snippets else "No search results found"
            analysis_tpl = "\n".join([
                "Analyze search results and predict probability of YES resolution.",
                "Event: {event}", "Cutoff: {cutoff}", "Today: {today}",
                "Search Results:", "{context}",
                "Consider:", "- Market signals (Polymarket, betting odds) as primary anchor",
                "- Quality and recency of evidence", "- Time remaining until cutoff",
                "- Resolution criteria (exact wording matters)",
                'Return JSON:', '{{"probability": 0.XX, "reasoning": "your analysis with key evidence"}}',
            ])
            final_prompt = analysis_tpl.format(
                event=blob_text[:2000], cutoff=deadline, today=TODAY_STR, context=evidence[:10000],
            )
            for eng in cls.TIER2_ENGINES:
                raw = await cls._chutes_call(http, eng, final_prompt, max_tokens=1500)
                if raw:
                    j = cls._extract_json(raw)
                    if j and "probability" in j:
                        return cls._hard_clamp(float(j["probability"])), j.get("reasoning", "")
            return None

        @classmethod
        async def _tier1_direct(cls, http, ev):
            user_msg = "\n".join([
                "EVENT TO FORECAST:",
                f"Title: {ev.get('title', '')}",
                f"Description: {ev.get('description', '')}",
                f"Cutoff: {ev.get('cutoff', '')}",
                f"Today: {TODAY_STR}",
                "Instructions:",
                "1. First, classify this event into one of the categories listed above",
                "2. Execute the category-specific searches",
                "3. Always search Polymarket for current market price",
                "4. Provide your prediction with structured reasoning",
            ])
            msgs = [
                {"role": "developer", "content": cls.ANALYST_SYSTEM},
                {"role": "user", "content": user_msg},
            ]
            tools = [{"type": "web_search"}]
            for eng in cls.TIER1_ENGINES:
                try:
                    txt, _ = await cls._openai_call(http, eng, msgs, tools)
                    if txt:
                        return cls._full_parse(txt)
                except Exception:
                    pass
            return None

        # --- Full pipeline ---
        @classmethod
        async def _full_pipeline(cls, http, ev, cat_key):
            eid = ev.get("event_id", "?")
            today = datetime.now().strftime("%Y-%m-%d")
            bill = 0.0
            samples = []
            sample_reasoning = None

            # Primary forecast
            parsed = None
            try:
                parsed, cost = await cls._primary_forecast(http, ev, today)
                bill += cost
            except Exception as exc:
                print(f"[APP_STORE][PRIMARY FAILED] {exc}, trying desearch rescue...")
                parsed, cost = await cls._desearch_rescue(http, ev, today)
                bill += cost

            # Repair if needed
            if not parsed or "probability" not in parsed:
                if parsed:
                    fix = await cls._repair_garbled(http, json.dumps(parsed))
                else:
                    fix = await cls._repair_garbled(http, "")
                parsed = fix["data"]
                bill += fix["cost"]

            # All primary paths failed -> try tier2/tier1
            if not parsed or "probability" not in parsed:
                tier2_result = await cls._tier2_forecast(http, ev)
                if tier2_result:
                    p2, r2 = tier2_result
                    cal_p = cls._platt_calibrate(p2, cat_key)
                    return {"event_id": eid, "prediction": cal_p, "reasoning": r2, "cost": bill}
                tier1_result = await cls._tier1_direct(http, ev)
                if tier1_result:
                    p1, r1 = tier1_result
                    cal_p = cls._platt_calibrate(p1, cat_key)
                    return {"event_id": eid, "prediction": cal_p, "reasoning": r1, "cost": bill}
                return {"event_id": eid, "prediction": cls.PROB_NEUTRAL, "reasoning": "All paths failed", "cost": bill}

            # Step 1: calibrate primary
            raw_p1 = float(parsed.get("probability", 0.5))
            normed_p1 = cls._normalize_raw(raw_p1)
            p1 = cls._hard_clamp(normed_p1 if normed_p1 is not None else 0.5)
            cal_p1 = cls._platt_calibrate(p1, cat_key)
            rationale = parsed.get("reasoning", "")
            samples.append(cal_p1)
            sample_reasoning = rationale
            print(f"[APP_STORE] Step 1: raw={raw_p1:.4f} clamped={p1:.4f} calibrated={cal_p1:.4f} (cat={cat_key})")

            first = {"event_id": eid, "prediction": p1, "reasoning": rationale, "cost": bill, "parsed_data": parsed}

            # Step 2: verification pass
            second_parsed, second_cost = await cls._verification_pass(http, ev, first, today)
            bill += second_cost
            if "probability" in second_parsed:
                raw_p2 = float(second_parsed["probability"])
                normed_p2 = cls._normalize_raw(raw_p2)
                p2 = cls._hard_clamp(normed_p2 if normed_p2 is not None else p1)
                cal_p2 = cls._platt_calibrate(p2, cat_key)
                final_r = second_parsed.get("reasoning", rationale)
                samples.append(cal_p2)
                print(f"[APP_STORE] Step 2: raw={raw_p2:.4f} clamped={p2:.4f} calibrated={cal_p2:.4f}")

                blended = (cal_p1 + cal_p2) / 2.0
                print(f"[APP_STORE] Blended: {blended:.4f}")
                return {"event_id": eid, "prediction": blended, "reasoning": final_r, "cost": bill}

            return {"event_id": eid, "prediction": cal_p1, "reasoning": rationale, "cost": bill}

        # --- Main entry ---
        @classmethod
        async def run(cls, event_data: dict, category: str) -> dict:
            eid = event_data.get("event_id", "?")
            # cat_key = cls._resolve_category(event_data).lower()
            print(f"\n[APP_STORE] Event: {eid} | Category: {category}")
            start = time.time()
            try:
                async with TrackedAsyncClient(timeout=cls.SESSION_TIMEOUT) as http:
                    try:
                        result = await asyncio.wait_for(
                            cls._full_pipeline(http, event_data, category),
                            timeout=cls.HARD_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        print(f"[APP_STORE] Timeout exceeded {cls.HARD_TIMEOUT}s")
                        result = {"event_id": eid, "prediction": cls.PROB_NEUTRAL,
                                "reasoning": "Timeout fallback", "cost": 0.0}
            except Exception as exc:
                print(f"[APP_STORE] Error: {exc}")
                result = {"event_id": eid, "prediction": cls.PROB_NEUTRAL,
                        "reasoning": f"Error: {exc}", "cost": 0.0}
            duration = time.time() - start
            print(f"[APP_STORE FINAL] prediction={result['prediction']:.3f} | {duration:.1f}s | ${result.get('cost', 0):.6f}")
            return {"event_id": result["event_id"], "prediction": result["prediction"],
                    "reasoning": str(result.get("reasoning", ""))[:2000]}
            

    class EarningsHandler:
        """
        Handler for Earnings events.
        Architecture: Single-phase OpenAI web search forecast with model cascade
        and PREDICTION/REASONING parsing.
        """
        MODELS = ["gpt-5.2", "gpt-5", "gpt-5-mini"]
        MAX_RETRIES = 3
        BASE_BACKOFF = 1.5

        SYSTEM_PROMPT = """<TASK_DESCRIPTION>
    You are a world-class forecasting agent.
    Given a news report, a Polymarket question, and today's date, you must estimate the probability of the question resolving YES.
    Your work will be evaluated on analysis quality and forecast accuracy.
    Be meticulous and professional.
    Important considerations:    
    Only count evidence from credible sources mentioned in the question. Speculation can cause false positives - avoid them.
    Evidence must fall within the information validity period specified in the question.
    Evidence within the period counts; evidence before the period should be ignored.
    Read the question and resolution criteria VERY CAREFULLY - pay attention to subtle wording - if it asks about events being reported, predict reports, not just the events. Some events may go unreported.
    Note on date discrepancies: Sometimes event dates like report publications or public appearances may differ slightly from actual dates.
    If it's clear the question refers to a specific event, assume minor date inaccuracies in the question are acceptable.
    Example: if the question asks about a statistics bureau report on August 11, but the actual report was August 10, assume the question meant the August 10 report.
    Similarly, if asking about a rate on October 14th but no rate exists on that exact date, use the closest available date.
    Without strong evidence for YES resolution, keep predicted probability near 0.
    </TASK_DESCRIPTION>
    <REASONING_INSTRUCTIONS>
    Provide thorough analysis and a probability forecast in JSON format. Be skeptical of search results due to potential bias.
    Your analysis should include:
    1) Search the web for recent news, articles and potential statistics about the question.
    2) Check, if able, Polymarket odds for this particular event.
    3) Question: Rephrase the question highlighting resolution criteria.
    4) Key Information: List and number the most relevant quotes from search results
    5) Time Horizon: Calculate remaining time until resolution and impact on certainty
    6) Influencing Factors: Identify factors favoring each outcome
    7) Conflicting Evidence: Analyze contradictions
    8) Probability Assessment: Final probability estimate (0-100)
    9) First Principles: What is the base probability of this being true?
    10) Base Rate: Consider typical occurrence rates for similar events
    If provided with similar Polymarket question price movements, use them to contextualize search results.
    For instance, if search results strongly confirm YES but Polymarket odds are moderate, the results may be misleading, though still indicative of likely YES.
    Be critical of search results; use Polymarket odds from similar questions as a ground truth reference.
    </REASONING_INSTRUCTIONS>"""

        USER_TEMPLATE = """<REQUIRED_OUTPUT_FORMAT>
    PREDICTION: [number between 0.0 and 1.0, probability for the YES result]
    REASONING: [2-4 sentences explaining your probability estimate, key factors considered, and main uncertainties]
    </REQUIRED_OUTPUT_FORMAT>
    Here is the question to predict:
    <event>
    {title}
    </event>
    Full Description:
    <description>
    {description}
    </description
    """

        @staticmethod
        def _clip_probability(prediction):
            return max(0.0, min(1.0, prediction))

        @classmethod
        def _build_messages(cls, event_data):
            cutoff_raw = event_data.get("cutoff", "")
            if isinstance(cutoff_raw, str):
                cutoff_str = cutoff_raw
            else:
                try:
                    cutoff_str = cutoff_raw.strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    cutoff_str = str(cutoff_raw)
            user_prompt = cls.USER_TEMPLATE.format(
                title=event_data.get("title", ""),
                description=event_data.get("description", ""),
            )
            return [
                {"role": "developer", "content": cls.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

        @classmethod
        def _parse_response(cls, response_text):
            try:
                lines = response_text.strip().split("\n")
                prediction = 0.5
                reasoning = "No reasoning provided."
                for line in lines:
                    if line.startswith("PREDICTION:"):
                        prediction = cls._clip_probability(float(line.replace("PREDICTION:", "").strip()))
                    elif line.startswith("REASONING:"):
                        reasoning = line.replace("REASONING:", "").strip()
                return prediction, reasoning
            except Exception as e:
                print(f"[EARNINGS] Failed to parse response: {e}")
                return 0.5, "Failed to parse LLM response."

        @classmethod
        async def _call_openai(cls, client, model, messages):
            openai_input = []
            for msg in messages:
                role = msg["role"]
                if role == "system":
                    role = "developer"
                openai_input.append({"role": role, "content": msg["content"]})
            payload = {
                "model": model,
                "input": openai_input,
                "tools": [{"type": "web_search"}],
                "run_id": RUN_ID,
            }
            resp = await client.post(f"{OPENAI_URL}/responses", json=payload, timeout=120.0)
            resp.raise_for_status()
            data = resp.json()
            content = extract_openai_text(data)
            cost = data.get("cost", 0.0)
            return content, cost

        @classmethod
        async def _retry_with_backoff(cls, func):
            for attempt in range(cls.MAX_RETRIES):
                try:
                    return await func()
                except httpx.TimeoutException as e:
                    if attempt < cls.MAX_RETRIES - 1:
                        delay = cls.BASE_BACKOFF ** (attempt + 1)
                        print(f"[EARNINGS][RETRY] Timeout, retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        raise Exception(f"Max retries exceeded: {e}")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt < cls.MAX_RETRIES - 1:
                        delay = cls.BASE_BACKOFF ** (attempt + 1)
                        print(f"[EARNINGS][RETRY] Rate limited (429), retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        try:
                            error_detail = e.response.json().get("detail", str(e))
                        except Exception:
                            error_detail = str(e)
                        raise Exception(f"HTTP {e.response.status_code}: {error_detail}")
                except Exception:
                    raise

        @classmethod
        async def _forecast_with_websearch(cls, client, event_data):
            messages = cls._build_messages(event_data)
            total_cost = 0.0
            for i, model in enumerate(cls.MODELS):
                print(f"[EARNINGS] Trying model {i+1}/{len(cls.MODELS)}: {model}")
                try:
                    async def llm_call(m=model):
                        return await cls._call_openai(client, m, messages)
                    response_text, cost = await cls._retry_with_backoff(llm_call)
                    total_cost += cost
                    prediction, reasoning = cls._parse_response(response_text)
                    print(f"[EARNINGS] Success with {model}: prediction={prediction}")
                    print(f"[EARNINGS] Cost: ${cost:.6f} | Total: ${total_cost:.6f}")
                    return {
                        "event_id": event_data.get("event_id", "?"),
                        "prediction": prediction,
                        "reasoning": reasoning,
                    }
                except httpx.HTTPStatusError as e:
                    try:
                        error_detail = e.response.json().get("detail", "")
                    except Exception:
                        error_detail = ""
                    print(f"[EARNINGS] HTTP error {e.response.status_code} with {model}{': ' + error_detail if error_detail else ''}. Trying next model...")
                except Exception as e:
                    print(f"[EARNINGS] Error with {model}: {e}. Trying next model...")
            print("[EARNINGS] All models failed. Returning fallback prediction.")
            return {
                "event_id": event_data.get("event_id", "?"),
                "prediction": 0.35,
                "reasoning": "Unable to generate forecast due to model availability issues. Returning neutral prediction.",
            }

        @classmethod
        async def run(cls, event_data: dict, category: str) -> dict:
            eid = event_data.get("event_id", "?")
            print(f"\n[EARNINGS] Running forecast for event: {eid}")
            print(f"[EARNINGS] Title: {event_data.get('title', '')}")
            start = time.time()
            async with TrackedAsyncClient(timeout=120.0) as client:
                result = await cls._forecast_with_websearch(client, event_data)
            elapsed = time.time() - start
            print(f"[EARNINGS] Complete in {elapsed:.2f}s")
            return {
                "event_id": result.get("event_id", eid),
                "prediction": result.get("prediction", 0.35),
                "reasoning": str(result.get("reasoning", ""))[:2000],
            }

    class BoxOfficeHandler:
        """
        Handler for Box Office events.
        Architecture: Gamma API (Polymarket) matcher → Weather/AppStore/Exact special handling →
                    LLM primary engines + Chutes fallback → segment adjustment.
        """
        # --- Constants ---
        DEFAULT_LIKELIHOOD = 0.35
        PRIMARY_ENGINES = ("gpt-5.2",)
        FALLBACK_ENGINES = ("Qwen/Qwen3-235B-A22B-Instruct-2507", "openai/gpt-oss-120b")
        MAX_TRIES = 4
        BACKOFF_FACTOR = 2.0
        REQUEST_TIMEOUT = 180.0
        TRANSIENT_CODES = (429, 500, 502, 503)
        # Gamma API
        _GAMMA_BASE = "https://gamma-api.polymarket.com"
        _GATEWAY_COST_PER_CALL = 0.0005
        _gateway_total_cost = 0.0
        # --- Adjustment coefficients ---
        ADJUSTMENT_COEFFICIENTS = {
            "weather": (0.15, -1.386),
        }
        # --- Known segments ---
        KNOWN_SEGMENT_LABELS = frozenset({
            'sports', 'app store', 'weather', 'earnings',
            'election', 'inflation', 'price',
        })
        _SPORT_BODY_KWS = (' vs ', ' vs. ', 'upcoming game', 'stoppage time', 'cricket', 'both teams to score')
        _SPORT_TITLE_KWS = (' win ', ' win?')
        # --- Sphere keywords ---
        SPHERE_KEYWORDS = (
            ("athletics", frozenset({"MATCH", "GAME", "WIN", "VS", "CHAMPIONSHIP", "LEAGUE", "CUP", "PLAYOFF", "GOAL", "SCORE"})),
            ("governance", frozenset({"ELECTION", "VOTE", "POLL", "PRESIDENT", "GOVERNOR", "SENATOR", "MAYOR", "TARIFF", "SANCTION", "WAR", "TREATY"})),
            ("financial", frozenset({"RATE", "FED", "GDP", "INFLATION", "BITCOIN", "CRYPTO", "PRICE", "MARKET", "$", "STOCK"})),
            ("technology", frozenset({"LAUNCH", "RELEASE", "APP", "SOFTWARE", "UPDATE", "SHIP", "ANNOUNCE"})),
            ("showbusiness", frozenset({"MOVIE", "FILM", "OSCAR", "GRAMMY", "EMMY", "ALBUM", "BOX_OFFICE", "AWARD"})),
        )
        SPHERE_HINTS = {
            "athletics": ("betting odds", "injuries", "recent form", "head-to-head", "standings"),
            "governance": ("Polymarket", "polling", "official statement", "Reuters AP"),
            "financial": ("FedWatch", "central bank", "market expectations", "economic data"),
            "technology": ("official blog", "press release", "SEC filing", "launch date"),
            "showbusiness": ("box_office", "reviews", "awards predictions", "release date"),
            "general": ("Polymarket", "recent news", "official source"),
        }
        # --- Incumbent table ---
        INCUMBENT_CONTEXT = """\
    --- APP STORE: Incumbent lookup table START ---
    **Strategy:** Incumbent lookup table. Match event title against known incumbents for each
    rank/store combination. If matched, predict their historical win rate. Otherwise predict 0.05.
    Use this alongside your other context, this is what historicaly work very well.
    **Incumbent table (corrected Feb 26):**
    | Store | Rank | Primary Incumbent | Win Rate | Prediction |
    |-------|------|-------------------|----------|------------|
    | Free | #1 | ChatGPT | 92% | 0.92 |
    | Free | #2 | Google Gemini | 24% | 0.24 |
    | Free | #3 | Google Gemini | 24% | 0.24 |
    | Free | #4 | Threads | 36% | 0.36 |
    | Paid | #1 | Shadowrocket | 92% | 0.92 |
    | Paid | #2 | HotSchedules | 97% | 0.97 |
    | Paid | #3 | AnkiMobile Flashcards | 78% | 0.78 |
    | Paid | #4 | Procreate Pocket | 83% | 0.83 |
    **Secondary incumbents (win rate >10%, also deserving elevated predictions):**
    | Store | Rank | App | Win Rate | Prediction |
    |-------|------|----|----------|------------|
    | Free | #2 | Freecash | 28% | 0.28 |
    | Free | #3 | Threads | 29% | 0.29 |
    | Free | #4 | Google Gemini | 25% | 0.25 |
    | Paid | #3 | Procreate Pocket | 13% | 0.13 |
    | Paid | #4 | AnkiMobile | 17% | 0.17 |
    --- APP STORE: Incumbent lookup table END ---"""

        # --- Prompts ---
        FORECASTER_SYSTEM = """You are an expert forecaster for prediction markets. Estimate P(YES) with rigorous research.
    CRITICAL: You MUST complete ALL your research in a SINGLE web_search tool call. You are NOT allowed to make multiple tool calls. Pack all your queries into one search. After receiving results, respond immediately.
    RESEARCH APPROACH - adapt your search strategy to the event type:
    For competitions and matches:
    - Search betting odds and convert to probability (decimal odds D → prob ≈ 1/D minus margin)
    - Check recent form, injuries, head-to-head records, rankings
    - Home advantage matters in team sports (+10-15%)
    For political events and elections:
    - Search Polymarket/PredictIt first - market prices are strong signals
    - For elections: check polling aggregates (538, RCP), consider historical polling errors
    - For policy/diplomatic events: prioritize official sources (Reuters, AP, government statements)
    - Check procedural requirements (votes needed, veto power, legislative calendar)
    For economic and financial events:
    - Search market-implied probabilities (CME FedWatch for rates, futures markets)
    - Check central bank communications and forward guidance
    - Economic calendar: what data releases occur before cutoff?
    For product launches and technology:
    - Check official company channels, press releases, SEC filings
    - Consider historical track record (announced vs actual delivery dates)
    - Distinguish between: announced, shipped, generally available
    For entertainment and awards:
    - Search prediction markets and expert consensus sites
    - Box office tracking, review aggregates
    - Awards predictions converge closer to ceremony date
    ALWAYS DO THESE:
    1. Search "Polymarket [topic]" - if market exists, price ≈ probability
    2. Search recent news (prioritize last 48-72 hours)
    3. Verify key claims with multiple sources
    4. Consider time until cutoff (more time = more uncertainty)
    ANALYSIS PRINCIPLES:
    - Polymarket price is your anchor - deviate only with strong contrary evidence
    - Official sources > speculation and rumors
    - Consider base rates: how often do similar events happen?
    - Resolution criteria are literal - read exact wording carefully
    - Range: never return exactly 0 or 1, use [0.01, 0.99]
    OUTPUT FORMAT:
    PREDICTION: [0.01-0.99]
    REASONING: [Key evidence, market signal if found, main uncertainties, 3-5 sentences]"""

        APP_STORE_SYSTEM = """You are an expert forecaster for prediction markets. Estimate P(YES) for app store ranking events.
    All relevant data is provided in the prompt — do not search the web.
    ANALYSIS PRINCIPLES:
    - Polymarket price is your anchor — deviate only with strong contrary evidence from the incumbent table
    - Incumbent win rates reflect historical dominance; weigh them against current market price
    - Consider time until cutoff (more time = more uncertainty)
    - Resolution criteria are literal — read exact wording carefully
    - Range: never return exactly 0 or 1, use [0.01, 0.99]
    OUTPUT FORMAT:
    PREDICTION: [0.01-0.99]
    REASONING: [Key evidence, market signal if found, main uncertainties, 3-5 sentences]"""

        # --- Utility helpers ---
        @staticmethod
        def _constrain(val, floor=0.01, ceiling=0.99):
            return max(floor, min(ceiling, val))

        @staticmethod
        def _sigmoid(x):
            x = max(-500.0, min(500.0, x))
            return 1.0 / (1.0 + math.exp(-x))

        # --- Gamma API helpers ---
        _MONTH_NUM = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
                    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
                    "november": 11, "december": 12}
        _STOP = {"will", "the", "be", "a", "an", "in", "on", "of", "by", "to", "and",
                "or", "for", "at", "is", "it", "its", "from", "that", "this", "with",
                "as", "are", "was", "end", "between", "than", "least", "more", "less"}
        _APP_RE = re.compile(
            r"Will (.+?) be (?:the )?#(\d+) (Free|Paid) [Aa]pp in the US "
            r"(?:iPhone |Apple )?App Store on (\w+ \d+)", re.I)
        _BTTS_RE = re.compile(r"^(.+?)\s+vs\.\s+(.+?):\s*Both Teams to Score$")
        _CRICKET_RE = re.compile(r"^(.+?):\s*(.+?)\s+vs\s+(.+?)\s+\(Game\s+.+?\)\s*-\s*(.+)$")
        _CRICKET_NG = re.compile(r"^(.+?):\s*(.+?)\s+vs\s+(.+?)\s*-\s*(.+)$")
        _ISO_RE = re.compile(r"scheduled for (\d{4}-\d{2}-\d{2})")

        @classmethod
        def _gamma_get_gateway(cls, url):
            for attempt in range(3):
                try:
                    resp = httpx.post(
                        f"{DESEARCH_URL}/web/crawl",
                        json={"url": url, "run_id": RUN_ID}, timeout=30.0
                    )
                    if resp.status_code in (429, 500, 502, 503) and attempt < 2:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    cost = data.get("cost", cls._GATEWAY_COST_PER_CALL)
                    if isinstance(cost, (int, float)) and cost > 0:
                        cls._gateway_total_cost += cost
                    content = data.get("content", "")
                    if not content:
                        return None
                    return json.loads(content) if isinstance(content, str) else content
                except Exception:
                    if attempt < 2:
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    return None
            return None

        @classmethod
        def _gamma_get(cls, path, params=None):
            url = f"{cls._GAMMA_BASE}{path}"
            if params:
                url += "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            return cls._gamma_get_gateway(url)

        @staticmethod
        def _market_dict(m):
            return {"market_id": m.get("id"), "question": m.get("question"),
                    "slug": m.get("slug"), "condition_id": m.get("conditionId"),
                    "active": m.get("active"), "closed": m.get("closed"),
                    "description": m.get("description", ""), "end_date": m.get("endDate", ""),
                    "outcome_prices": m.get("outcomePrices", ""),
                    "one_day_price_change": m.get("oneDayPriceChange")}

        @staticmethod
        def _yes_price_raw(op):
            if not op:
                return None
            try:
                pl = json.loads(op) if isinstance(op, str) else op
                return float(pl[0]) if isinstance(pl, list) and pl else None
            except (json.JSONDecodeError, ValueError, IndexError):
                return None

        @staticmethod
        def _normalize_text(text):
            return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower().strip()))

        @classmethod
        def _keywords(cls, title):
            clean = re.sub(r"[^\w\s-]", " ", title)
            words = [w for w in clean.lower().split() if w not in cls._STOP and len(w) > 1]
            return " ".join(words[:8])

        @staticmethod
        def _word_sim(a, b):
            ba, bb = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
            return len(ba & bb) / len(ba | bb) if ba and bb else 0.0

        @staticmethod
        def _desc_date(desc):
            if not desc:
                return None
            m = re.search(r"scheduled for (\w+ \d+ \d{4})", desc)
            return m.group(1) if m else None

        @classmethod
        def _parse_date_str(cls, ds):
            parts = ds.split()
            mo = cls._MONTH_NUM.get(parts[0].lower())
            if not mo:
                return None
            try:
                return datetime(2026, mo, int(parts[1])).date()
            except (ValueError, IndexError):
                return None

        @classmethod
        def _iso_date(cls, desc):
            m = cls._ISO_RE.search(desc) if desc else None
            return m.group(1) if m else None

        @classmethod
        def _get_date(cls, desc):
            return cls._iso_date(desc) or cls._desc_date(desc)

        @classmethod
        def _dates_ok(cls, our_date, pm_desc):
            pm = cls._get_date(pm_desc or "")
            return not our_date or not pm or pm == our_date

        # --- Matchers ---
        @classmethod
        def _find_parent(cls, queries, ta, tb, desc):
            our_date = cls._get_date(desc)
            best, best_sc = None, (-1, -1, -1.0)
            for q in queries:
                data = cls._gamma_get("/public-search", {"q": q})
                if not data or "events" not in data:
                    continue
                for ev in data["events"]:
                    for m in ev.get("markets", []):
                        mq = m.get("question", "").lower()
                        if ta.lower() not in mq or tb.lower() not in mq:
                            continue
                        dm = 1 if our_date and any(
                            cls._get_date(m2.get("description", "")) == our_date
                            for m2 in ev.get("markets", [])) else 0
                        sc = (dm, 0 if m.get("closed") else 1,
                            cls._word_sim(f"{ta} vs {tb}", m["question"]))
                        if sc > best_sc:
                            best_sc, best = sc, ev
                        break
                if best and best_sc[0] == 1:
                    break
            return best

        @classmethod
        def _match_btts(cls, title, desc, cutoff):
            m = cls._BTTS_RE.match(title)
            if not m:
                return None
            ta, tb = m.group(1).strip(), m.group(2).strip()
            parent = cls._find_parent([f"{ta} {tb}", f"{tb} {ta}"], ta, tb, desc)
            if not parent:
                return {"status": "NO_MARKETS", "exact_match": None, "related_markets": [], "context": None}
            data = cls._gamma_get(f"/events/slug/{parent['slug']}-more-markets")
            btts = next((cls._market_dict(mk) for mk in (data or {}).get("markets", [])
                        if "Both Teams to Score" in mk.get("question", "")), None)
            if not btts:
                return {"status": "NO_MARKETS", "exact_match": None, "related_markets": [], "context": None}
            our = cls._normalize_text(title)
            pm = cls._normalize_text(btts["question"])
            if our != pm and cls._normalize_text(f"{tb} vs. {ta}: Both Teams to Score") != pm:
                return {"status": "NO_MARKETS", "exact_match": None, "related_markets": [], "context": None}
            if cls._dates_ok(cls._desc_date(desc), btts.get("description")):
                return {"status": "EXACT", "exact_match": btts, "related_markets": [], "context": None}
            return {"status": "NO_MARKETS", "exact_match": None, "related_markets": [], "context": None}

        @classmethod
        def _parse_app(cls, title):
            m = cls._APP_RE.search(title)
            if not m:
                return None
            app, rank, typ, ds = m.groups()
            d = cls._parse_date_str(ds)
            return {"app": app.strip(), "rank": int(rank), "type": typ.capitalize(),
                    "date": d, "date_str": ds} if d else None

        @classmethod
        def _app_search(cls, queries, filt):
            out, seen = [], set()
            for q in queries:
                for ev in (cls._gamma_get("/public-search", {"q": q}) or {}).get("events", []):
                    for m in ev.get("markets", []):
                        mid = m.get("id")
                        if mid in seen:
                            continue
                        p = cls._parse_app(m.get("question", ""))
                        if p and filt(p):
                            seen.add(mid)
                            md = cls._market_dict(m)
                            md["parsed"] = p
                            out.append(md)
            return out

        @classmethod
        def _fmt_yes(cls, op):
            p = cls._yes_price_raw(op)
            if p is None:
                return "?"
            if p < 0.01:
                return "<1%"
            if p > 0.99:
                return ">99%"
            return f"{p:.0%}"

        @staticmethod
        def _fmt_pct(v):
            if v < 0.005:
                return "<1%"
            if v > 0.995:
                return ">99%"
            return f"{v:.0%}"

        @staticmethod
        def _delta_24h(change, closed=False):
            if closed or change is None:
                return None
            try:
                v = float(change)
            except (TypeError, ValueError):
                return None
            if abs(v) > 0.50:
                return None
            pct = round(v * 100)
            return pct if pct != 0 else None

        @classmethod
        def _fmt_24h(cls, change, closed=False):
            pct = cls._delta_24h(change, closed)
            if pct is None:
                return ""
            return f" ({'+' if pct > 0 else ''}{pct}% 24h)"

        @classmethod
        def _app_competition(cls, slot_markets, target_date):
            by_date = defaultdict(list)
            for m in slot_markets:
                by_date[str(m["parsed"]["date"])].append(m)
            if not by_date:
                return []
            ts = str(target_date)
            dd = lambda d: abs(int(d.replace("-", "")) - int(ts.replace("-", "")))
            if ts in by_date:
                chosen = ts
            else:
                active = [d for d in by_date if any(not m.get("closed") for m in by_date[d])]
                chosen = min(active, key=dd) if active else min(by_date, key=dd)
            entries = []
            for m in by_date[chosen]:
                raw = cls._yes_price_raw(m["outcome_prices"]) or 0.0
                app = m["parsed"]["app"]
                if (re.match(r"^App [A-Z]$", app) or app.lower() == "another app") and raw < 0.005:
                    continue
                entries.append({"app": app, "price": raw, "closed": m.get("closed"),
                                "date": chosen, "one_day_price_change": m.get("one_day_price_change")})
            entries.sort(key=lambda x: -x["price"])
            return entries

        @classmethod
        def _format_app_context(cls, parsed, same_date, nearby, slot_markets):
            app, rank, typ, ds, date = (parsed["app"], parsed["rank"], parsed["type"],
                                        parsed["date_str"], parsed["date"])
            lines = [f'App Store: "{app} #{rank} {typ} on {ds}?"']
            comp = cls._app_competition(slot_markets, date)
            for e in comp:
                e["delta_24h"] = cls._delta_24h(e.get("one_day_price_change"), e.get("closed"))
            if same_date:
                pm_ranks = sorted(set(m["rank"] for m in same_date))
                lines.append(f"No Polymarket market for #{rank}. PM only has {', '.join(f'#{r}' for r in pm_ranks)} for {app}.")
                for m in same_date:
                    c = " [closed]" if m.get("closed") else ""
                    lines.append(f"{app} #{m['rank']} {typ} on {ds}: {cls._fmt_yes(m['outcome_prices'])}"
                                f"{cls._fmt_24h(m.get('one_day_price_change'), m.get('closed'))}{c}.")
            elif nearby:
                lines.append(f"Data offset: {nearby[0]['date_diff']:+d}d (using {nearby[0]['date']} market for {ds} question).")
                nd = nearby[0]["date"]
                for m in (m2 for m2 in nearby if m2["date"] == nd):
                    c = " [closed]" if m.get("closed") else ""
                    lines.append(f"{app} #{m['rank']} {typ}: {cls._fmt_yes(m['outcome_prices'])}"
                                f"{cls._fmt_24h(m.get('one_day_price_change'), m.get('closed'))}{c}.")
            else:
                lines.append(f"No Polymarket markets found for {app}.")
            top = [e for e in comp if e["price"] >= 0.005][:8]
            if top:
                parts = [f"{e['app']}: {cls._fmt_pct(e['price'])}{cls._fmt_24h(e.get('one_day_price_change'), e.get('closed'))}"
                        for e in top]
                cd = top[0]["date"]
                if cd != str(date):
                    diff = (datetime.strptime(cd, "%Y-%m-%d").date() - date).days
                    lines.append(f"#{rank} {typ} slot on {cd} ({diff:+d}d): {' | '.join(parts)}")
                else:
                    lines.append(f"#{rank} {typ} slot: {' | '.join(parts)}")
            our = next((e for e in comp if e["app"].lower() == app.lower()), None)
            if our and our.get("delta_24h") is not None:
                leader = comp[0]
                rival = (next((e for e in comp[1:] if e.get("delta_24h") is not None), None)
                        if leader["app"].lower() == app.lower()
                        else (leader if leader.get("delta_24h") is not None else None))
                if rival:
                    gap = our["delta_24h"] - rival["delta_24h"]
                    if gap != 0:
                        s = lambda v: f"{'+' if v > 0 else ''}{v}%"
                        lines.append(f"Momentum: {app} {'gaining' if gap > 0 else 'losing ground'} vs "
                                    f"{rival['app']}, gap {s(gap)} ({app} {s(our['delta_24h'])}, "
                                    f"{rival['app']} {s(rival['delta_24h'])})")
            return "\n".join(lines)

        @classmethod
        def _match_app_store(cls, title, desc, cutoff):
            p = cls._parse_app(title)
            if not p:
                return None if "app store" not in title.lower() else {"status": "NO_MARKETS", "exact_match": None, "related_markets": [], "context": None}
            mo = [p["date"].strftime("%B")]
            app_markets = cls._app_search(
                [f"{p['app']} {p['type']} App Store"] + [f"{p['app']} {p['type']} App Store {m}" for m in mo],
                lambda x: x["app"].lower() == p["app"].lower() and x["type"].lower() == p["type"].lower())
            exact, same_date, nearby = None, [], []
            for m in app_markets:
                mp = m["parsed"]
                dd_val = (mp["date"] - p["date"]).days
                entry = {k: m[k] for k in ("market_id", "question", "slug", "condition_id",
                                            "active", "closed", "outcome_prices", "one_day_price_change")}
                entry.update(rank=mp["rank"], date=str(mp["date"]),
                            date_diff=dd_val, rank_diff=mp["rank"] - p["rank"])
                if dd_val == 0 and mp["rank"] == p["rank"]:
                    exact = entry
                elif dd_val == 0:
                    same_date.append(entry)
                elif abs(dd_val) <= 7:
                    nearby.append(entry)
            if exact:
                return {"status": "EXACT", "exact_match": exact, "related_markets": [], "context": None}
            same_date.sort(key=lambda x: abs(x["rank_diff"]))
            nearby.sort(key=lambda x: (abs(x["date_diff"]), abs(x["rank_diff"])))
            slot = cls._app_search(
                [f"{p['type']} App Store {p['rank']}"] + [f"{p['type']} App Store {m}" for m in mo],
                lambda x: x["rank"] == p["rank"] and x["type"].lower() == p["type"].lower())
            if same_date + nearby or slot:
                return {"status": "RELATED", "exact_match": None, "related_markets": same_date + nearby,
                        "context": cls._format_app_context(p, same_date, nearby, slot)}
            return {"status": "NO_MARKETS", "exact_match": None, "related_markets": [], "context": None}

        @classmethod
        def _match_general(cls, title, desc, cutoff):
            if not title:
                return {"status": "NO_MARKETS", "exact_match": None, "related_markets": [], "context": None}
            data = cls._gamma_get("/public-search", {"q": cls._keywords(title)})
            if not data or "events" not in data:
                return {"status": "NO_MARKETS", "exact_match": None, "related_markets": [], "context": None}
            event_date = cls._desc_date(desc)
            cands = [(cls._word_sim(title, m.get("question", "")), cls._market_dict(m))
                    for ev in data["events"] for m in ev.get("markets", [])]
            cands = [(s, m) for s, m in cands if s >= 0.999]
            if not cands:
                return {"status": "NO_MARKETS", "exact_match": None, "related_markets": [], "context": None}
            def sc(pair):
                s, c = pair
                dm = int(bool(event_date and cls._desc_date(c.get("description", "")) == event_date))
                return (dm, 0 if c.get("closed") else 1, s)
            cands.sort(key=sc, reverse=True)
            best_sim, best = cands[0]
            best["similarity"] = best_sim
            return {"status": "EXACT", "exact_match": best, "related_markets": [], "context": None}

        @classmethod
        def _agent_match(cls, title, description, cutoff):
            for fn in (cls._match_btts, cls._match_app_store):
                result = fn(title, description, cutoff)
                if result is not None:
                    return result
            return cls._match_general(title, description, cutoff)

        # --- Segment classification ---
        @classmethod
        def _classify_segment(cls, ev):
            for t in ev.get('metadata', {}).get('topics', []):
                if t.lower() in cls.KNOWN_SEGMENT_LABELS:
                    return t
            title = ev.get('title', '')
            desc = ev.get('description', '')
            combined = (title + ' ' + desc).lower()
            title_low = title.lower()
            if 'election' in combined:
                return 'election'
            if any(k in combined for k in cls._SPORT_BODY_KWS) or any(k in title_low for k in cls._SPORT_TITLE_KWS):
                return 'Sports'
            if ' app ' in combined or 'app store' in combined:
                return 'App Store'
            if ' price of ' in combined:
                return 'price'
            if 'earnings' in combined or (any(q in combined for q in ['q1', 'q2', 'q3', 'q4']) and 'above' in combined):
                return 'Earnings'
            if 'inflation' in combined:
                return 'inflation'
            if ' temperature ' in combined:
                return 'Weather'
            return 'Other'

        @classmethod
        def _apply_segment_adjustment(cls, raw, segment):
            norm = segment.lower()
            if norm == 'weather':
                slope, intercept = cls.ADJUSTMENT_COEFFICIENTS["weather"]
                return cls._sigmoid(slope * raw + intercept)
            return raw

        @staticmethod
        def _is_weather(title):
            return ' temperature ' in (title or '').lower()

        @staticmethod
        def _is_app_store(title):
            return 'app store' in (title or '').lower()

        # --- Response parsing ---
        @classmethod
        def _interpret_response(cls, raw_text):
            if not raw_text:
                return 0.5, "No response received"
            prob = None
            reasoning = ""
            # JSON parsing
            parsed = safe_parse_json(raw_text)
            if parsed:
                for k in ("likelihood", "probability", "prediction", "forecast", "prob", "p", "final_probability"):
                    v = parsed.get(k)
                    if v is not None:
                        try:
                            prob = cls._constrain(float(v))
                            break
                        except (ValueError, TypeError):
                            continue
                if not reasoning:
                    for k in ("rationale", "reasoning", "reason", "analysis", "explanation"):
                        v = parsed.get(k)
                        if v:
                            reasoning = str(v)
                            break
            # Structured lines
            if prob is None:
                for line in raw_text.strip().splitlines():
                    upper = line.upper().strip()
                    if any(upper.startswith(m) for m in ("PREDICTION:", "PROBABILITY:", "ESTIMATE:", "FORECAST:", "LIKELIHOOD:")):
                        after = line.split(":", 1)[1].strip() if ":" in line else ""
                        if after.lower().startswith("is "):
                            after = after[3:]
                        nums = re.findall(r"[\d.]+", after[:40])
                        if nums:
                            try:
                                val = float(nums[0])
                                if val > 1:
                                    val /= 100.0
                                prob = cls._constrain(val)
                                break
                            except (ValueError, TypeError):
                                pass
                    if upper.startswith("REASONING:"):
                        reasoning = line.split(":", 1)[1].strip() if ":" in line else ""
            # Percentage
            if prob is None:
                m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", raw_text)
                if m:
                    try:
                        prob = cls._constrain(float(m.group(1)) / 100.0)
                    except (ValueError, TypeError):
                        pass
            # Bare decimal
            if prob is None:
                m = re.search(r"\b0\.\d{1,3}\b", raw_text)
                if m:
                    try:
                        prob = cls._constrain(float(m.group()))
                    except (ValueError, TypeError):
                        pass
            if not reasoning and raw_text:
                reasoning = raw_text[:500].replace("\n", " ").strip()
            if prob is None:
                return 0.5, reasoning or "Unable to interpret forecast"
            return prob, reasoning

        # --- LLM call helpers ---
        @classmethod
        async def _openai_call_with_retry(cls, client, model, messages, tools=None):
            payload = {
                "model": model,
                "input": messages,
                "run_id": RUN_ID,
                "reasoning": {"effort": "medium"},
            }
            if tools is not None:
                payload["tools"] = tools
            idx = 0
            while True:
                try:
                    r = await client.post(f"{OPENAI_URL}/responses", json=payload, timeout=cls.REQUEST_TIMEOUT)
                    if r.status_code == 200:
                        data = r.json()
                        return extract_openai_text(data), data.get("cost", 0.0)
                    if r.status_code in cls.TRANSIENT_CODES and idx < cls.MAX_TRIES - 1:
                        await asyncio.sleep(cls.BACKOFF_FACTOR ** (idx + 1))
                        idx += 1
                        continue
                    r.raise_for_status()
                except httpx.TimeoutException:
                    if idx >= cls.MAX_TRIES - 1:
                        raise
                    await asyncio.sleep(cls.BACKOFF_FACTOR ** (idx + 1))
                    idx += 1
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code not in cls.TRANSIENT_CODES or idx >= cls.MAX_TRIES - 1:
                        raise
                    await asyncio.sleep(cls.BACKOFF_FACTOR ** (idx + 1))
                    idx += 1

        @classmethod
        async def _chutes_call(cls, client, model, prompt, max_tokens=2000):
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "run_id": RUN_ID,
            }
            try:
                r = await client.post(f"{CHUTES_URL}/chat/completions", json=payload, timeout=60.0)
                if r.status_code == 200:
                    choices = r.json().get("choices", [])
                    return choices[0].get("message", {}).get("content", "") if choices else None
            except Exception:
                pass
            return None

        @classmethod
        async def _desearch_call(cls, client, query):
            try:
                r = await client.post(
                    f"{DESEARCH_URL}/web/search".replace("/web/search", "/search"),
                    json={"query": query, "model": "NOVA", "run_id": RUN_ID},
                    timeout=60.0,
                )
                if r.status_code == 200:
                    return r.json().get("results", [])
            except Exception:
                pass
            return []

        # --- Enrichment ---
        @classmethod
        def _enrich_exact(cls, event_data, gamma_result, is_weather=False):
            eid = event_data.get("event_id", "unknown")
            try:
                exact = gamma_result["exact_match"]
                op = exact.get("outcome_prices", "")
                pl = json.loads(op) if isinstance(op, str) else op
                pm_price = cls._constrain(float(pl[0]))
                return {"event_id": eid, "prediction": pm_price, "reasoning": f"Polymarket exact match: {pm_price:.0%}"}
            except Exception:
                if is_weather:
                    return {"event_id": eid, "prediction": 0.20, "reasoning": ""}
                exact = gamma_result.get("exact_match") or {}
                op = exact.get("outcome_prices", "")
                price = cls.DEFAULT_LIKELIHOOD
                if op:
                    try:
                        pl = json.loads(op) if isinstance(op, str) else op
                        if isinstance(pl, list) and pl:
                            price = cls._constrain(float(pl[0]))
                    except (json.JSONDecodeError, ValueError, IndexError):
                        pass
                return {"event_id": eid, "prediction": price, "reasoning": ""}

        # --- Primary and fallback LLM flows ---
        @classmethod
        async def _primary_engines(cls, client, event_data):
            user_inquiry = "\n".join([
                "EVENT TO FORECAST:",
                f"Title: {event_data.get('title', '')}",
                f"Description: {event_data.get('description', '')}",
                f"Cutoff: {event_data.get('cutoff', '')}",
                f"Today: {TODAY_STR}",
                "Instructions:",
                "1. First, classify this event into one of the categories listed above",
                "2. Execute the category-specific searches",
                "3. Always search Polymarket for current market price",
                "4. Provide your prediction with structured reasoning",
            ])
            messages = [
                {"role": "developer", "content": cls.FORECASTER_SYSTEM},
                {"role": "user", "content": user_inquiry},
            ]
            tools = [{"type": "web_search"}]
            for eng in cls.PRIMARY_ENGINES:
                try:
                    text, _ = await cls._openai_call_with_retry(client, eng, messages, tools)
                    if text:
                        prob, reasoning = cls._interpret_response(text)
                        print(f"[BOX_OFFICE] primary {eng} -> {prob:.3f}")
                        return prob, reasoning
                except Exception as e:
                    print(f"[BOX_OFFICE] primary {eng} failed: {e}")
            return None

        @classmethod
        async def _fallback_engines(cls, client, event_data):
            # Generate search terms
            sphere = "general"
            blob = (event_data.get("title", "") + " " + event_data.get("description", "")).upper()
            for label, terms in cls.SPHERE_KEYWORDS:
                if any(t in blob for t in terms):
                    sphere = label
                    break
            hints = cls.SPHERE_HINTS.get(sphere, cls.SPHERE_HINTS["general"])
            hint_csv = ", ".join(hints)
            gen_prompt = "\n".join([
                "Generate 4 specific search queries for this forecasting event.",
                f"Consider searching for: {hint_csv}",
                f"Event: {event_data.get('title', '')[:1000]}",
                f"Cutoff: {event_data.get('cutoff', '')}",
                f"Today: {TODAY_STR}",
                '{"queries": ["query1", "query2", "query3", "query4"]}',
            ])
            queries = None
            for eng in cls.FALLBACK_ENGINES[:2]:
                raw = await cls._chutes_call(client, eng, gen_prompt, max_tokens=500)
                if raw:
                    parsed = safe_parse_json(raw)
                    if parsed:
                        queries = (parsed.get("searchTerms") or parsed.get("queries") or [])[:4]
                        if queries:
                            break
            if not queries:
                queries = [f"Polymarket {event_data.get('title', '')[:80]}", event_data.get("title", "")[:80]]
            # Search
            snippets = []
            for q in queries:
                results = await cls._desearch_call(client, q)
                for item in results[:3]:
                    url = item.get("url", "source")
                    title_r = item.get("title", "")
                    snip = item.get("snippet", "")[:300]
                    snippets.append(f"[{url}] {title_r}: {snip}")
            evidence = "\n\n".join(snippets) if snippets else "No findings from search"
            # Analysis
            analysis_prompt = "\n".join([
                "Analyze search results and predict probability of YES resolution.",
                f"Event: {event_data.get('title', '')}",
                f"Cutoff: {event_data.get('cutoff', '')}",
                f"Today: {TODAY_STR}",
                f"Search Results:\n{evidence[:10000]}",
                "Consider:", "- Market signals (Polymarket, betting odds) as primary anchor",
                "- Quality and recency of evidence", "- Time remaining until cutoff",
                "- Resolution criteria (exact wording matters)",
                '{"probability": 0.XX, "reasoning": "your analysis with key evidence"}',
            ])
            for eng in cls.FALLBACK_ENGINES:
                raw = await cls._chutes_call(client, eng, analysis_prompt, max_tokens=1500)
                if raw:
                    parsed = safe_parse_json(raw)
                    if parsed:
                        prob_val = parsed.get("likelihood") or parsed.get("probability")
                        if prob_val is not None:
                            prob = cls._constrain(float(prob_val))
                            reasoning = parsed.get("rationale") or parsed.get("reasoning", "")
                            print(f"[BOX_OFFICE] fallback {eng} -> {prob:.3f}")
                            return prob, reasoning
            return None

        @classmethod
        async def _app_store_with_context(cls, client, event_data, gamma_context):
            user_inquiry = "\n".join([
                "EVENT TO FORECAST:",
                f"Title: {event_data.get('title', '')}",
                f"Description: {event_data.get('description', '')}",
                f"Cutoff: {event_data.get('cutoff', '')}",
                f"Today: {TODAY_STR}",
                "",
                "POLYMARKET DATA (live market prices — use as primary signal):",
                gamma_context,
                "",
                cls.INCUMBENT_CONTEXT,
            ])
            messages = [
                {"role": "developer", "content": cls.APP_STORE_SYSTEM},
                {"role": "user", "content": user_inquiry},
            ]
            for eng in cls.PRIMARY_ENGINES:
                try:
                    text, _ = await cls._openai_call_with_retry(client, eng, messages)
                    if text:
                        return cls._interpret_response(text)
                except Exception:
                    pass
            return None

        @classmethod
        def _normalize_cutoff(cls, ev):
            raw = ev.get("cutoff")
            if isinstance(raw, str):
                try:
                    cleaned = raw.replace('Z', '+00:00')
                    ev["cutoff"] = datetime.fromisoformat(cleaned).strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    pass

        # --- Main entry ---
        @classmethod
        async def run(cls, event_data: dict, category: str) -> dict:
            eid = event_data.get("event_id", "?")
            title = event_data.get("title", "")
            segment = cls._classify_segment(event_data)
            print(f"\n[BOX_OFFICE] segment={segment} title={title[:80]}")
            cls._normalize_cutoff(event_data)
            cls._gateway_total_cost = 0.0
            start = time.time()

            # --- Gamma matcher ---
            cutoff_raw = event_data.get("cutoff", "")
            try:
                cutoff_dt = datetime.fromisoformat(str(cutoff_raw).replace('Z', '+00:00'))
            except Exception:
                cutoff_dt = datetime.utcnow()
            gamma_result = cls._agent_match(title, event_data.get("description", ""), cutoff_dt)
            gamma_status = gamma_result.get("status", "NO_MARKETS")
            print(f"[BOX_OFFICE] gamma={gamma_status} cost=${cls._gateway_total_cost:.4f}")

            # --- Weather: bypass LLM, use calibration ---
            if cls._is_weather(title):
                outcome = cls._enrich_exact(event_data, gamma_result, is_weather=True)
                raw_p = float(outcome["prediction"])
                adj_p = cls._apply_segment_adjustment(raw_p, "weather")
                print(f"[BOX_OFFICE] weather raw={raw_p:.3f} adjusted={adj_p:.3f}")
                return {"event_id": outcome["event_id"], "prediction": adj_p,
                        "reasoning": outcome.get("reasoning", "")[:2000]}

            # --- Exact match ---
            if gamma_status == "EXACT":
                outcome = cls._enrich_exact(event_data, gamma_result)
                print(f"[BOX_OFFICE] exact_match prediction={outcome['prediction']}")
                return {"event_id": outcome["event_id"], "prediction": float(outcome["prediction"]),
                        "reasoning": outcome.get("reasoning", "")[:2000]}

            # --- App Store RELATED: LLM with Gamma context ---
            app_store_gamma = gamma_result if (cls._is_app_store(title) and gamma_status == "RELATED") else None

            try:
                async with TrackedAsyncClient(timeout=cls.REQUEST_TIMEOUT) as client:
                    if app_store_gamma and app_store_gamma.get("context"):
                        app_result = await cls._app_store_with_context(client, event_data, app_store_gamma["context"])
                        if app_result is not None:
                            prob, reasoning = app_result
                            raw_p = float(prob)
                            adj_p = cls._apply_segment_adjustment(raw_p, segment)
                            duration = time.time() - start
                            print(f"[BOX_OFFICE FINAL] prediction={adj_p:.3f} | {duration:.1f}s")
                            return {"event_id": eid, "prediction": adj_p, "reasoning": str(reasoning)[:2000]}

                    # --- Standard LLM + web search ---
                    primary_result = await cls._primary_engines(client, event_data)
                    if primary_result is not None:
                        prob, reasoning = primary_result
                        raw_p = float(prob)
                        adj_p = cls._apply_segment_adjustment(raw_p, segment)
                        duration = time.time() - start
                        print(f"[BOX_OFFICE FINAL] prediction={adj_p:.3f} | {duration:.1f}s")
                        return {"event_id": eid, "prediction": adj_p, "reasoning": str(reasoning)[:2000]}

                    fallback_result = await cls._fallback_engines(client, event_data)
                    if fallback_result is not None:
                        prob, reasoning = fallback_result
                        raw_p = float(prob)
                        adj_p = cls._apply_segment_adjustment(raw_p, segment)
                        duration = time.time() - start
                        print(f"[BOX_OFFICE FINAL] prediction={adj_p:.3f} | {duration:.1f}s")
                        return {"event_id": eid, "prediction": adj_p, "reasoning": str(reasoning)[:2000]}
            except Exception as e:
                print(f"[BOX_OFFICE] Error: {e}")

            duration = time.time() - start
            print(f"[BOX_OFFICE FINAL] prediction={cls.DEFAULT_LIKELIHOOD:.3f} (fallback) | {duration:.1f}s")
            return {"event_id": eid, "prediction": cls.DEFAULT_LIKELIHOOD,
                    "reasoning": "Every engine failed to produce a forecast"}

    class GeopoliticsHandler:
        """
        Handler for Geopolitics events.
        Architecture: Multi-agent pipeline (6 agents via Chutes LLM) with Desearch search/crawl,
                    supervisor synthesis via OpenAI, and Platt scaling.
        """
        # --- Configuration ---
        NUM_AGENTS = 1
        MAX_SEARCH_ROUNDS = 3
        AGENT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
        AGENT_TEMPERATURE = 0.3
        SUPERVISOR_MODEL = "gpt-5.2"
        USE_SUPERVISOR = True
        USE_PLATT_SCALING = True
        PLATT_ALPHA = math.sqrt(3)
        MAX_URLS_TO_CRAWL = 5
        HTTP_TIMEOUT = 120.0
        MAX_RETRIES = 2
        BASE_BACKOFF = 1.5
        # Cost rates
        SEARCH_COST = 0.0025
        CRAWL_COST = 0.0005
        # Cost limits per provider
        COST_LIMIT = 0.10

        # --- Utility helpers ---
        @staticmethod
        def _clip(p):
            return max(0.0, min(1.0, p))

        @classmethod
        def _platt_scale(cls, p):
            eps = 1e-7
            p = max(eps, min(1 - eps, p))
            log_odds = math.log(p / (1 - p))
            scaled = cls.PLATT_ALPHA * log_odds
            return cls._clip(1 / (1 + math.exp(-scaled)))

        @staticmethod
        def _extract_json(text):
            if not text:
                return None
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            for pat in [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```', r'(\{.*\})']:
                m = re.search(pat, text, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group(1))
                    except json.JSONDecodeError:
                        continue
            return None

        @classmethod
        async def _retry(cls, func):
            for attempt in range(cls.MAX_RETRIES):
                try:
                    return await func()
                except httpx.TimeoutException as e:
                    if attempt < cls.MAX_RETRIES - 1:
                        await asyncio.sleep(cls.BASE_BACKOFF ** (attempt + 1))
                    else:
                        raise Exception(f"Max retries exceeded: {e}")
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt < cls.MAX_RETRIES - 1:
                        await asyncio.sleep(cls.BASE_BACKOFF ** (attempt + 1))
                    else:
                        raise
                except Exception:
                    raise

        # --- Desearch helpers ---
        @classmethod
        async def _desearch_search(cls, query, client, num_results=30):
            payload = {"query": query, "num": num_results, "start": 0, "run_id": RUN_ID}
            resp = await client.post(f"{DESEARCH_URL}/web/search", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])

        @classmethod
        async def _desearch_crawl(cls, url, client):
            payload = {"url": url, "run_id": RUN_ID}
            resp = await client.post(f"{DESEARCH_URL}/web/crawl", json=payload, timeout=15)
            resp.raise_for_status()
            return resp.json().get("content", "")

        @classmethod
        async def _search_and_crawl(cls, query, client):
            print(f"[GEOPOLITICS][SEARCH] Query: {query}")
            try:
                results = await cls._desearch_search(query, client)
            except Exception as e:
                print(f"[GEOPOLITICS][SEARCH] Search failed: {e}")
                return f"Search failed for query: {query}"
            if not results:
                return f"No results found for query: {query}"
            formatted = [f"Search query: {query}", f"Found {len(results)} results\n"]
            for i, r in enumerate(results[:5], 1):
                formatted.extend([
                    f"{i}. {r.get('title', 'No title')}",
                    f"   {r.get('snippet', 'No description')}",
                    f"   URL: {r.get('link', '')}\n"
                ])
            top_urls = [r.get("link") for r in results[:cls.MAX_URLS_TO_CRAWL] if r.get("link")]
            if top_urls:
                formatted.append("\n--- Full Content from Top URLs ---\n")
                for i, url in enumerate(top_urls, 1):
                    try:
                        content = await cls._desearch_crawl(url, client)
                        if content:
                            trunc = content[:5000] if len(content) > 5000 else content
                            formatted.extend([f"\nURL {i}: {url}", f"Content:\n{trunc}\n", "-" * 80])
                    except Exception as e:
                        print(f"[GEOPOLITICS][SEARCH] Crawl failed for {url}: {e}")
                        formatted.append(f"\nURL {i}: {url} (failed to crawl)")
            return "\n".join(formatted)

        # --- Chutes LLM helper ---
        @classmethod
        async def _chutes_complete(cls, model, messages, client, temperature=0.3):
            payload = {"model": model, "messages": messages, "temperature": temperature, "run_id": RUN_ID}
            resp = await client.post(f"{CHUTES_URL}/chat/completions", json=payload, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            return ""

        # --- OpenAI helper ---
        @classmethod
        async def _openai_complete(cls, model, messages, client):
            openai_input = []
            for msg in messages:
                role = msg["role"]
                if role == "system":
                    role = "developer"
                openai_input.append({"role": role, "content": msg["content"]})
            payload = {"model": model, "input": openai_input, "run_id": RUN_ID,
                    "tools": [{"type": "web_search"}]}
            resp = await client.post(f"{OPENAI_URL}/responses", json=payload, timeout=cls.HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return extract_openai_text(data)

        # --- Prompts ---
        @classmethod
        def _agent_system_prompt(cls, cutoff_str):
            return f"""You are an expert superforecaster using rigorous probabilistic reasoning. Today's date is {TODAY_STR}.
    Event cutoff: {cutoff_str}
    FORECASTING METHODOLOGY:
    1. BASE RATES: Start with reference class base rates before considering specific details
    2. DECOMPOSITION: Break complex questions into component probabilities
    3. TIME-SERIES: Consider trends, momentum, and historical patterns
    4. INSIDE vs OUTSIDE VIEW: Balance specific case details with broader statistical patterns
    5. QUANTITATIVE ANCHORS: Look for concrete numbers, dates, and measurable indicators
    6. RED TEAM: Actively seek counterarguments and reasons your forecast could be wrong
    7. UNCERTAINTY SOURCES: Separately assess different uncertainty types (epistemic vs aleatory)
    SEARCH PROTOCOL:
    To search for information, respond with:
    SEARCH: [specific, well-scoped query]
    CRITICAL: If you choose to search, STOP after writing "SEARCH: [query]" - do NOT provide reasoning or prediction yet.
    FINAL FORECAST FORMAT:
    When ready (after 0-{cls.MAX_SEARCH_ROUNDS} searches), provide:
    PREDICTION: [number 0.0-1.0]
    CONFIDENCE: [high/medium/low]
    REASONING: [structured analysis including: base rate, key evidence, main uncertainties, counterarguments]
    CALIBRATION NOTES:
    - Only use information available before the cutoff date
    - Avoid anchoring bias and excessive hedging toward 50%
    - Be willing to make predictions outside the 30-70% range when evidence warrants
    - High confidence means narrow uncertainty bounds, not high probability
    - Quantify uncertainties where possible"""

        @staticmethod
        def _agent_user_prompt(event_data):
            title = event_data.get("title", "")
            desc = event_data.get("description", "")
            cutoff = event_data.get("cutoff", "")
            return f"""Forecast this event using superforecaster methodology:
    **Event:** {title}
    **Resolution Criteria:** {desc}
    **Deadline:** {cutoff}
    **Analysis Framework:**
    1. REFERENCE CLASS: What similar events have occurred? What's the base rate?
    2. SPECIFIC FACTORS: What makes this case different from the reference class?
    - Positive indicators (increase probability)
    - Negative indicators (decrease probability)
    3. TIME DYNAMICS:
    - What's the trajectory/trend?
    - How much time remains?
    - Are there key milestones or deadlines?
    4. INFORMATION QUALITY:
    - What key information is missing?
    - How reliable are available sources?
    - What would most change your forecast?
    5. QUANTITATIVE GROUNDING:
    - Are there numbers, dates, or metrics to anchor on?
    - Can you estimate rates, frequencies, or magnitudes?
    Begin your analysis. Search as needed for specific information, then provide your probability estimate."""

        AGENT_FINAL_REQUEST = """Provide your final forecast now using this format:
    PREDICTION: [0.0-1.0]
    CONFIDENCE: [high/medium/low]
    REASONING: [Your reasoning should include:
    - Reference class base rate
    - Key evidence that updates from base rate
    - Main sources of uncertainty
    - Strongest counterargument to your forecast
    - How you weighted inside vs outside view]
    Be specific and quantitative where possible. Avoid vague hedging."""

        @classmethod
        def _supervisor_synthesis_prompt(cls, event_data, forecasts, disagreements):
            title = event_data.get("title", "")
            desc = event_data.get("description", "")
            cutoff = event_data.get("cutoff", "")
            forecasts_detail = "\n\n".join([
                f"Forecast {i+1}:\n  Probability: {f['probability']:.3f}\n  "
                f"Confidence: {f['confidence']}\n  Reasoning: {f['reasoning'][:300]}..."
                for i, f in enumerate(forecasts)
            ])
            disagreements_str = "\n".join([f"- {d}" for d in disagreements]) if disagreements else "None identified"
            return f"""As a meta-forecaster, synthesize multiple probabilistic forecasts into a final estimate.
    **Event:** {title}
    **Description:** {desc}
    **Cutoff:** {cutoff}
    **Individual Forecasts:**
    {forecasts_detail}
    **Identified Disagreements:**
    {disagreements_str}
    **Synthesis Framework:**
    1. INFORMATION AGGREGATION:
    - Which forecasts are based on the strongest evidence?
    - Are there patterns in high vs low confidence predictions?
    - Do disagreements reflect genuine uncertainty or poor reasoning?
    2. EVIDENCE QUALITY:
    - Which forecasts cite concrete, verifiable information?
    - Are there common evidence sources or unique information?
    - Which reasoning chains are most rigorous?
    3. UNCERTAINTY ASSESSMENT:
    - Is the spread in forecasts justified by genuine uncertainty?
    - Should we weight forecasts differently by confidence/quality?
    - What's the appropriate final uncertainty range?
    4. BIAS CORRECTION:
    - Are forecasts clustering around 50% due to hedging?
    - Is there groupthink or anchoring bias?
    - Do any forecasts show clear directional bias?
    **Task:** Provide a synthesized forecast that appropriately weighs the evidence and reasoning quality.
    **Output Format:**
    PREDICTION: [0.0-1.0]
    REASONING: [Explain how you weighted different forecasts, what evidence was most convincing]
    Be willing to deviate significantly from the mean if evidence quality varies substantially."""

        # --- Parse forecast response ---
        @classmethod
        def _parse_forecast(cls, response):
            probability = 0.5
            confidence = "medium"
            reasoning = "No reasoning provided"
            try:
                parsed_reasoning = _parse_reasoning_value(response)
                if parsed_reasoning:
                    reasoning = parsed_reasoning
                for line in response.strip().splitlines():
                    line = line.strip()
                    if line.startswith("PREDICTION:"):
                        prob_str = line.replace("PREDICTION:", "").strip()
                        probability = cls._clip(_parse_prediction_value(prob_str))
                    elif line.startswith("CONFIDENCE:"):
                        conf = line.replace("CONFIDENCE:", "").strip().lower()
                        confidence = conf if conf in ("high", "medium", "low") else "medium"
            except Exception as e:
                print(f"[GEOPOLITICS][PARSE] Error: {e}")
            return probability, confidence, reasoning

        # --- Single agent forecast ---
        @classmethod
        async def _run_agent(cls, agent_id, event_data, client):
            cutoff = str(event_data.get("cutoff", ""))
            try:
                cutoff_str = datetime.fromisoformat(cutoff.replace('Z', '+00:00')).strftime('%Y-%m-%d')
            except Exception:
                cutoff_str = cutoff
            messages = [
                {"role": "system", "content": cls._agent_system_prompt(cutoff_str)},
                {"role": "user", "content": cls._agent_user_prompt(event_data)},
            ]
            search_queries = []
            for round_num in range(cls.MAX_SEARCH_ROUNDS):
                try:
                    response_text = await cls._retry(
                        lambda: cls._chutes_complete(cls.AGENT_MODEL, messages, client, cls.AGENT_TEMPERATURE)
                    )
                    # Check if agent wants to search
                    if "SEARCH:" in response_text and "PREDICTION:" not in response_text:
                        query = ""
                        for line in response_text.splitlines():
                            if line.strip().startswith("SEARCH:"):
                                query = line.replace("SEARCH:", "").strip()
                                break
                        if query:
                            search_queries.append(query)
                            print(f"[GEOPOLITICS][AGENT-{agent_id}] Search {round_num+1}: {query[:60]}...")
                            search_results = await cls._search_and_crawl(query, client)
                            messages.append({"role": "assistant", "content": f"SEARCH: {query}"})
                            messages.append({"role": "user", "content": f"Search results:\n{search_results}\n\nContinue your analysis or provide final forecast."})
                        else:
                            break
                    else:
                        break
                except Exception as e:
                    print(f"[GEOPOLITICS][AGENT-{agent_id}] Error in search round {round_num}: {e}")
                    break
            # Final forecast
            messages.append({"role": "user", "content": cls.AGENT_FINAL_REQUEST})
            try:
                response_text = await cls._retry(
                    lambda: cls._chutes_complete(cls.AGENT_MODEL, messages, client, cls.AGENT_TEMPERATURE)
                )
                probability, confidence, reasoning = cls._parse_forecast(response_text)
            except Exception as e:
                print(f"[GEOPOLITICS][AGENT-{agent_id}] Error getting final forecast: {e}")
                probability, confidence, reasoning = 0.5, "low", f"Error: {str(e)}"
            print(f"[GEOPOLITICS][AGENT-{agent_id}] Forecast: {probability:.3f} ({confidence})")
            return {"probability": probability, "confidence": confidence, "reasoning": reasoning,
                    "search_queries": search_queries}

        # --- Supervisor synthesis ---
        @classmethod
        def _identify_disagreements(cls, forecasts):
            disagreements = []
            probs = [f["probability"] for f in forecasts]
            mean_p = sum(probs) / len(probs)
            std_p = math.sqrt(sum((p - mean_p) ** 2 for p in probs) / len(probs))
            if std_p > 0.15:
                disagreements.append(f"Wide probability range: {min(probs):.2f} to {max(probs):.2f} (σ={std_p:.2f})")
            high_conf = [f for f in forecasts if f["confidence"] == "high"]
            if len(high_conf) >= 2:
                hp = [f["probability"] for f in high_conf]
                if max(hp) - min(hp) > 0.3:
                    disagreements.append(f"High-confidence forecasts conflict: {min(hp):.2f} vs {max(hp):.2f}")
            return disagreements

        @classmethod
        async def _supervisor_synthesize(cls, event_data, forecasts, client):
            print(f"[GEOPOLITICS][SUPERVISOR] Reconciling {len(forecasts)} forecasts...")
            disagreements = cls._identify_disagreements(forecasts)
            if disagreements:
                print(f"[GEOPOLITICS][SUPERVISOR] Found {len(disagreements)} disagreement(s)")
            prompt = cls._supervisor_synthesis_prompt(event_data, forecasts, disagreements)
            messages = [
                {"role": "system", "content": "You are an expert meta-forecaster specializing in rigorous probability aggregation and Bayesian reasoning."},
                {"role": "user", "content": prompt},
            ]
            try:
                response_text = await cls._retry(
                    lambda: cls._openai_complete(cls.SUPERVISOR_MODEL, messages, client)
                )
                print(f"[GEOPOLITICS][SUPERVISOR] Response: {response_text[:200]}...")
                probability = sum(f["probability"] for f in forecasts) / len(forecasts)
                reasoning = _parse_reasoning_value(response_text) or "Synthesis of multiple forecasts"
                for line in response_text.strip().splitlines():
                    line = line.strip()
                    if line.startswith("PREDICTION:"):
                        probability = cls._clip(_parse_prediction_value(line.replace("PREDICTION:", "").strip()))
                print(f"[GEOPOLITICS][SUPERVISOR] Final probability: {probability:.3f}")
                return {"probability": probability, "reasoning": reasoning, "disagreements": disagreements}
            except Exception as e:
                print(f"[GEOPOLITICS][SUPERVISOR] Error synthesizing: {e}")
                mean_p = sum(f["probability"] for f in forecasts) / len(forecasts)
                return {"probability": mean_p, "reasoning": f"Synthesis failed, using mean: {e}",
                        "disagreements": disagreements}

        # --- Main entry ---
        @classmethod
        async def run(cls, event_data: dict, category: str) -> dict:
            eid = event_data.get("event_id", "?")
            title = event_data.get("title", "")
            print(f"\n[GEOPOLITICS] Starting ENHANCED SUPERFORECASTER")
            print(f"[GEOPOLITICS] Event: {title}")
            print(f"[GEOPOLITICS] Config: {cls.NUM_AGENTS} agents")
            start = time.time()

            async with TrackedAsyncClient(timeout=cls.HTTP_TIMEOUT) as client:
                # Phase 1: N parallel agents
                print(f"\n[GEOPOLITICS][PHASE 1] Generating {cls.NUM_AGENTS} forecasts...")
                tasks = [cls._run_agent(i + 1, event_data, client) for i in range(cls.NUM_AGENTS)]
                forecasts = await asyncio.gather(*tasks)
                probs = [f["probability"] for f in forecasts]
                mean_p = sum(probs) / len(probs)
                std_p = math.sqrt(sum((p - mean_p) ** 2 for p in probs) / len(probs))
                print(f"[GEOPOLITICS][PHASE 1] Forecasts: {[f'{p:.3f}' for p in probs]}")
                print(f"[GEOPOLITICS][PHASE 1] Mean: {mean_p:.3f}, Std: {std_p:.3f}")

                # Phase 2: Supervisor synthesis
                if cls.USE_SUPERVISOR:
                    print(f"\n[GEOPOLITICS][PHASE 2] Supervisor synthesis...")
                    sup = await cls._supervisor_synthesize(event_data, forecasts, client)
                    final_p = sup["probability"]
                    reasoning = sup["reasoning"]
                else:
                    print(f"\n[GEOPOLITICS][PHASE 2] Skipping supervisor (using mean)...")
                    final_p = mean_p
                    reasoning = "No supervisor - using simple mean"
                print(f"[GEOPOLITICS][PHASE 2] Reconciled: {final_p:.3f}")

                # Phase 3: Platt scaling
                if cls.USE_PLATT_SCALING:
                    print(f"\n[GEOPOLITICS][PHASE 3] Applying Platt scaling (alpha={cls.PLATT_ALPHA:.3f})...")
                    calibrated = cls._platt_scale(final_p)
                    print(f"[GEOPOLITICS][PHASE 3] Before: {final_p:.3f} -> After: {calibrated:.3f}")
                else:
                    calibrated = final_p

            elapsed = time.time() - start
            print(f"\n[GEOPOLITICS] Complete in {elapsed:.2f}s")
            print(f"[GEOPOLITICS] FINAL PROBABILITY: {calibrated:.3f}")
            return {
                "event_id": eid,
                "prediction": calibrated,
                "reasoning": str(reasoning)[:2000],
            }

    # ============================================================
    # PART 9: ROUTER & ENTRY POINT
    # ============================================================
    HANDLER_MAP = {
        "sports": SportsHandler,
        "app_store": AppStoreHandler,
        "weather": BoxOfficeHandler,
        "earnings": EarningsHandler,
        "box_office": BoxOfficeHandler,
        "global": BoxOfficeHandler,
        "geopolitics": GeopoliticsHandler,
    }
    async def _main_async(event_data: dict) -> dict:
        """Async entry: classify event, route to handler."""
        COST_TRACKER.reset()                       # ← reset per-event costs
        t0 = time.time()
        eid = event_data.get("event_id", "?")
        title = event_data.get("title", "")
        # Classify
        category = await classify_event_with_llm(event_data)
        handler_cls = HANDLER_MAP.get(category, BoxOfficeHandler)
        print(f"\n{'='*60}")
        print(f"[UNIFIED] Event: {eid}")
        print(f"[UNIFIED] Title: {title[:100]}")
        print(f"[UNIFIED] Category: {category} → {handler_cls.__name__}")
        print(f"{'='*60}")
        # Run handler
        result = await handler_cls.run(event_data, category)
        elapsed = time.time() - t0
        print(f"\n[UNIFIED] FINAL: P={result.get('prediction', 0):.3f} | {elapsed:.1f}s | handler={handler_cls.__name__}")
        COST_TRACKER.print_summary()               # ← print per-provider costs
        return result
    def agent_main(event_data: dict) -> dict:
        """
        Main entry point for the unified forecasting agent.
        Classifies the event type and routes to the best-performing handler.
        Args:
            event_data: dict with keys: event_id, title, description, cutoff, metadata
        Returns:
            dict with keys: event_id, prediction, reasoning
        """
        try:
            result = asyncio.run(_main_async(event_data))
        except Exception as e:
            print(f"[UNIFIED] Fatal error: {e}")
            result = {
                "event_id": event_data.get("event_id", "?"),
                "prediction": 0.35,
                "reasoning": f"Fatal error: {str(e)[:200]}",
            }
        # Ensure valid output
        return {
            "event_id": result.get("event_id", event_data.get("event_id", "?")),
            "prediction": result.get("prediction", 0.35),
            "reasoning": str(result.get("reasoning", ""))[:2000],
        }
    return agent_main
        
async def agent2():
    import os
    import re
    import math
    import json
    import time
    import asyncio
    import httpx
    from uuid import uuid4
    from collections import defaultdict
    from datetime import datetime


    # =============================================================================
    # Gamma API client (inlined from simplified_matcher)
    # =============================================================================

    _GAMMA_BASE = "https://gamma-api.polymarket.com"
    _GATEWAY_URL = None
    _GATEWAY_RUN_ID = None
    _GATEWAY_COST_PER_CALL = 0.0005
    _gateway_total_cost = 0.0

    def configure_gamma(gateway_url: str, run_id: str):
        global _GATEWAY_URL, _GATEWAY_RUN_ID, _gateway_total_cost
        _GATEWAY_URL = gateway_url.rstrip("/")
        _GATEWAY_RUN_ID = run_id
        _gateway_total_cost = 0.0

    def get_gateway_cost() -> float:
        return _gateway_total_cost

    def _gamma_get_gateway(url):
        global _gateway_total_cost
        for attempt in range(3):
            try:
                resp = httpx.post(f"{_GATEWAY_URL}/web/crawl",
                                json={"url": url, "run_id": _GATEWAY_RUN_ID}, timeout=30.0)
                if resp.status_code in (429, 500, 502, 503) and attempt < 2:
                    time.sleep(1.0 * (attempt + 1)); continue
                resp.raise_for_status()
                data = resp.json()
                cost = data.get("cost", _GATEWAY_COST_PER_CALL)
                if isinstance(cost, (int, float)) and cost > 0:
                    _gateway_total_cost += cost
                content = data.get("content", "")
                if not content:
                    return None
                return json.loads(content) if isinstance(content, str) else content
            except Exception as e:
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1)); continue
                return None
        return None

    def gamma_get(path, params=None):
        url = f"{_GAMMA_BASE}{path}"
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return _gamma_get_gateway(url)

    # =============================================================================
    # Gamma helpers
    # =============================================================================

    _MONTH_NUM = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
                "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
                "november": 11, "december": 12}

    _STOP = {"will", "the", "be", "a", "an", "in", "on", "of", "by", "to", "and",
            "or", "for", "at", "is", "it", "its", "from", "that", "this", "with",
            "as", "are", "was", "end", "between", "than", "least", "more", "less"}

    def _market_dict(m):
        return {"market_id": m.get("id"), "question": m.get("question"),
                "slug": m.get("slug"), "condition_id": m.get("conditionId"),
                "active": m.get("active"), "closed": m.get("closed"),
                "description": m.get("description", ""), "end_date": m.get("endDate", ""),
                "outcome_prices": m.get("outcomePrices", ""),
                "one_day_price_change": m.get("oneDayPriceChange")}

    def _yes_price_raw(op):
        if not op:
            return None
        try:
            pl = json.loads(op) if isinstance(op, str) else op
            return float(pl[0]) if isinstance(pl, list) and pl else None
        except (json.JSONDecodeError, ValueError, IndexError):
            return None

    def _fmt_yes(op):
        p = _yes_price_raw(op)
        if p is None: return "?"
        if p < 0.01: return "<1%"
        if p > 0.99: return ">99%"
        return f"{p:.0%}"

    def _fmt_pct(v):
        if v < 0.005: return "<1%"
        if v > 0.995: return ">99%"
        return f"{v:.0%}"

    def _normalize(text):
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower().strip()))

    def _keywords(title):
        clean = re.sub(r"[^\w\s-]", " ", title)
        words = [w for w in clean.lower().split() if w not in _STOP and len(w) > 1]
        return " ".join(words[:8])

    def _word_sim(a, b):
        ba, bb = set(re.findall(r"[a-z0-9]+", a.lower())), set(re.findall(r"[a-z0-9]+", b.lower()))
        return len(ba & bb) / len(ba | bb) if ba and bb else 0.0

    def _desc_date(desc):
        if not desc: return None
        m = re.search(r"scheduled for (\w+ \d+ \d{4})", desc)
        return m.group(1) if m else None

    def _parse_date_str(ds):
        parts = ds.split()
        mo = _MONTH_NUM.get(parts[0].lower())
        if not mo: return None
        try: return datetime(2026, mo, int(parts[1])).date()
        except (ValueError, IndexError): return None

    # =============================================================================
    # Matcher results
    # =============================================================================

    _NM = lambda: {"status": "NO_MARKETS", "exact_match": None, "related_markets": [], "context": None}
    _EX = lambda m: {"status": "EXACT", "exact_match": m, "related_markets": [], "context": None}
    _REL = lambda m, c: {"status": "RELATED", "exact_match": None, "related_markets": m, "context": c}

    # =============================================================================
    # Shared: parent event search (BTTS + Cricket)
    # =============================================================================

    _ISO_RE = re.compile(r"scheduled for (\d{4}-\d{2}-\d{2})")

    def _iso_date(desc):
        m = _ISO_RE.search(desc) if desc else None
        return m.group(1) if m else None

    def _get_date(desc):
        return _iso_date(desc) or _desc_date(desc)

    def _find_parent(queries, ta, tb, desc):
        our_date = _get_date(desc)
        best, best_sc = None, (-1, -1, -1.0)
        for q in queries:
            data = gamma_get("/public-search", {"q": q})
            if not data or "events" not in data:
                continue
            for ev in data["events"]:
                for m in ev.get("markets", []):
                    mq = m.get("question", "").lower()
                    if ta.lower() not in mq or tb.lower() not in mq:
                        continue
                    dm = 1 if our_date and any(
                        _get_date(m2.get("description", "")) == our_date
                        for m2 in ev.get("markets", [])) else 0
                    sc = (dm, 0 if m.get("closed") else 1,
                        _word_sim(f"{ta} vs {tb}", m["question"]))
                    if sc > best_sc:
                        best_sc, best = sc, ev
                    break
            if best and best_sc[0] == 1:
                break
        return best

    def _dates_ok(our_date, pm_desc):
        pm = _get_date(pm_desc or "")
        return not our_date or not pm or pm == our_date

    # =============================================================================
    # BTTS matcher
    # =============================================================================

    _BTTS_RE = re.compile(r"^(.+?)\s+vs\.\s+(.+?):\s*Both Teams to Score$")

    def _match_btts(title, desc, cutoff):
        m = _BTTS_RE.match(title)
        if not m:
            return None
        ta, tb = m.group(1).strip(), m.group(2).strip()

        parent = _find_parent([f"{ta} {tb}", f"{tb} {ta}"], ta, tb, desc)
        if not parent:
            return _NM()

        data = gamma_get(f"/events/slug/{parent['slug']}-more-markets")
        btts = next((_market_dict(m) for m in (data or {}).get("markets", [])
                    if "Both Teams to Score" in m.get("question", "")), None)
        if not btts:
            return _NM()

        our, pm = _normalize(title), _normalize(btts["question"])
        if our != pm and _normalize(f"{tb} vs. {ta}: Both Teams to Score") != pm:
            return _NM()
        return _EX(btts) if _dates_ok(_desc_date(desc), btts.get("description")) else _NM()

    # =============================================================================
    # Cricket matcher
    # =============================================================================

    _CRICKET_RE = re.compile(r"^(.+?):\s*(.+?)\s+vs\s+(.+?)\s+\(Game\s+.+?\)\s*-\s*(.+)$")
    _CRICKET_NG = re.compile(r"^(.+?):\s*(.+?)\s+vs\s+(.+?)\s*-\s*(.+)$")
    _CRICKET_SLUGS = {
        "Most Sixes": "-most-sixes", "Toss Match Double": "-toss-match-double",
        "Team Top Batter": "-team-top-batter", "Most Fours": "-most-fours",
    }

    def _match_cricket(title, desc, cutoff):
        m = _CRICKET_RE.match(title) or _CRICKET_NG.match(title)
        if not m:
            return None
        groups = m.groups()
        tourn, ta, tb, sub = groups[0].strip(), groups[1].strip(), groups[2].strip(), groups[-1].strip()

        parent = _find_parent(
            [f"{tourn} {ta} {tb}", f"{ta} {tb} {tourn}", f"{ta} vs {tb}"], ta, tb, desc)
        if not parent:
            return _NM()

        slug, our_date = parent["slug"], _iso_date(desc)

        sub_type = next((st for st in _CRICKET_SLUGS if sub.startswith(st)), None)
        if sub.startswith("Completed match"):
            markets = [_market_dict(mk) for mk in parent.get("markets", [])]
        elif sub_type:
            data = gamma_get(f"/events/slug/{slug}{_CRICKET_SLUGS[sub_type]}")
            markets = [_market_dict(mk) for mk in (data or {}).get("markets", [])]
        else:
            return _NM()

        our_norm = _normalize(title)
        found = next((mk for mk in markets if _normalize(mk.get("question", "")) == our_norm), None)
        if found and not _dates_ok(our_date, found.get("description", "")):
            found = None
        return _EX(found) if found else _NM()

    # =============================================================================
    # App Store matcher
    # =============================================================================

    _APP_RE = re.compile(
        r"Will (.+?) be (?:the )?#(\d+) (Free|Paid) [Aa]pp in the US "
        r"(?:iPhone |Apple )?App Store on (\w+ \d+)", re.I)

    def _parse_app(title):
        m = _APP_RE.search(title)
        if not m: return None
        app, rank, typ, ds = m.groups()
        d = _parse_date_str(ds)
        return {"app": app.strip(), "rank": int(rank), "type": typ.capitalize(),
                "date": d, "date_str": ds} if d else None

    def _app_search(queries, filt):
        out, seen = [], set()
        for q in queries:
            for ev in (gamma_get("/public-search", {"q": q}) or {}).get("events", []):
                for m in ev.get("markets", []):
                    mid = m.get("id")
                    if mid in seen: continue
                    p = _parse_app(m.get("question", ""))
                    if p and filt(p):
                        seen.add(mid)
                        md = _market_dict(m)
                        md["parsed"] = p
                        out.append(md)
        return out

    def _delta_24h(change, closed=False):
        if closed or change is None: return None
        try: v = float(change)
        except (TypeError, ValueError): return None
        if abs(v) > 0.50: return None
        pct = round(v * 100)
        return pct if pct != 0 else None

    def _fmt_24h(change, closed=False):
        pct = _delta_24h(change, closed)
        if pct is None: return ""
        return f" ({'+' if pct > 0 else ''}{pct}% 24h)"

    def _app_competition(slot_markets, target_date):
        by_date = defaultdict(list)
        for m in slot_markets:
            by_date[str(m["parsed"]["date"])].append(m)
        if not by_date: return []
        ts = str(target_date)
        dd = lambda d: abs(int(d.replace("-", "")) - int(ts.replace("-", "")))
        if ts in by_date:
            chosen = ts
        else:
            active = [d for d in by_date if any(not m.get("closed") for m in by_date[d])]
            chosen = min(active, key=dd) if active else min(by_date, key=dd)
        entries = []
        for m in by_date[chosen]:
            raw = _yes_price_raw(m["outcome_prices"]) or 0.0
            app = m["parsed"]["app"]
            if (re.match(r"^App [A-Z]$", app) or app.lower() == "another app") and raw < 0.005:
                continue
            entries.append({"app": app, "price": raw, "closed": m.get("closed"),
                            "date": chosen, "one_day_price_change": m.get("one_day_price_change")})
        entries.sort(key=lambda x: -x["price"])
        return entries

    def _format_app_context(parsed, same_date, nearby, slot_markets):
        app, rank, typ, ds, date = (parsed["app"], parsed["rank"], parsed["type"],
                                    parsed["date_str"], parsed["date"])
        lines = [f'App Store: "{app} #{rank} {typ} on {ds}?"']

        comp = _app_competition(slot_markets, date)
        for e in comp:
            e["delta_24h"] = _delta_24h(e.get("one_day_price_change"), e.get("closed"))

        if same_date:
            pm_ranks = sorted(set(m["rank"] for m in same_date))
            lines.append(f"No Polymarket market for #{rank}. PM only has {', '.join(f'#{r}' for r in pm_ranks)} for {app}.")
            for m in same_date:
                c = " [closed]" if m.get("closed") else ""
                lines.append(f"{app} #{m['rank']} {typ} on {ds}: {_fmt_yes(m['outcome_prices'])}"
                            f"{_fmt_24h(m.get('one_day_price_change'), m.get('closed'))}{c}.")
        elif nearby:
            lines.append(f"Data offset: {nearby[0]['date_diff']:+d}d (using {nearby[0]['date']} market for {ds} question).")
            nd = nearby[0]["date"]
            for m in (m for m in nearby if m["date"] == nd):
                c = " [closed]" if m.get("closed") else ""
                lines.append(f"{app} #{m['rank']} {typ}: {_fmt_yes(m['outcome_prices'])}"
                            f"{_fmt_24h(m.get('one_day_price_change'), m.get('closed'))}{c}.")
        else:
            lines.append(f"No Polymarket markets found for {app}.")

        top = [e for e in comp if e["price"] >= 0.005][:8]
        if top:
            parts = [f"{e['app']}: {_fmt_pct(e['price'])}{_fmt_24h(e.get('one_day_price_change'), e.get('closed'))}"
                    for e in top]
            cd = top[0]["date"]
            if cd != str(date):
                diff = (datetime.strptime(cd, "%Y-%m-%d").date() - date).days
                lines.append(f"#{rank} {typ} slot on {cd} ({diff:+d}d): {' | '.join(parts)}")
            else:
                lines.append(f"#{rank} {typ} slot: {' | '.join(parts)}")

        our = next((e for e in comp if e["app"].lower() == app.lower()), None)
        if our and our.get("delta_24h") is not None:
            leader = comp[0]
            rival = (next((e for e in comp[1:] if e.get("delta_24h") is not None), None)
                    if leader["app"].lower() == app.lower()
                    else (leader if leader.get("delta_24h") is not None else None))
            if rival:
                gap = our["delta_24h"] - rival["delta_24h"]
                if gap != 0:
                    s = lambda v: f"{'+' if v > 0 else ''}{v}%"
                    lines.append(f"Momentum: {app} {'gaining' if gap > 0 else 'losing ground'} vs "
                                f"{rival['app']}, gap {s(gap)} ({app} {s(our['delta_24h'])}, "
                                f"{rival['app']} {s(rival['delta_24h'])})")
        return "\n".join(lines)

    def _match_app_store(title, desc, cutoff):
        p = _parse_app(title)
        if not p:
            return None if "app store" not in title.lower() else _NM()

        mo = [p["date"].strftime("%B")]
        app_markets = _app_search(
            [f"{p['app']} {p['type']} App Store"] + [f"{p['app']} {p['type']} App Store {m}" for m in mo],
            lambda x: x["app"].lower() == p["app"].lower() and x["type"].lower() == p["type"].lower())

        exact, same_date, nearby = None, [], []
        for m in app_markets:
            mp = m["parsed"]
            dd = (mp["date"] - p["date"]).days
            entry = {k: m[k] for k in ("market_id", "question", "slug", "condition_id",
                                        "active", "closed", "outcome_prices", "one_day_price_change")}
            entry.update(rank=mp["rank"], date=str(mp["date"]),
                        date_diff=dd, rank_diff=mp["rank"] - p["rank"])
            if dd == 0 and mp["rank"] == p["rank"]:
                exact = entry
            elif dd == 0:
                same_date.append(entry)
            elif abs(dd) <= 7:
                nearby.append(entry)

        if exact:
            return _EX(exact)

        same_date.sort(key=lambda x: abs(x["rank_diff"]))
        nearby.sort(key=lambda x: (abs(x["date_diff"]), abs(x["rank_diff"])))
        slot = _app_search(
            [f"{p['type']} App Store {p['rank']}"] + [f"{p['type']} App Store {m}" for m in mo],
            lambda x: x["rank"] == p["rank"] and x["type"].lower() == p["type"].lower())

        if same_date + nearby or slot:
            return _REL(same_date + nearby, _format_app_context(p, same_date, nearby, slot))
        return _NM()

    # =============================================================================
    # General matcher (fallback — Jaccard sim=1.0)
    # =============================================================================

    def _match_general(title, desc, cutoff):
        if not title:
            return _NM()
        data = gamma_get("/public-search", {"q": _keywords(title)})
        if not data or "events" not in data:
            return _NM()

        event_date = _desc_date(desc)
        cands = [(_word_sim(title, m.get("question", "")), _market_dict(m))
                for ev in data["events"] for m in ev.get("markets", [])]
        cands = [(s, m) for s, m in cands if s >= 0.999]
        if not cands:
            return _NM()

        def sc(pair):
            s, c = pair
            dm = int(bool(event_date and _desc_date(c.get("description", "")) == event_date))
            return (dm, 0 if c.get("closed") else 1, s)

        cands.sort(key=sc, reverse=True)
        best_sim, best = cands[0]
        best["similarity"] = best_sim
        return _EX(best)

    # =============================================================================
    # Matcher router
    # =============================================================================

    def agent_match(title: str, description: str, cutoff: datetime) -> dict:
        for fn in (_match_btts, _match_cricket, _match_app_store):
            result = fn(title, description, cutoff)
            if result is not None:
                return result
        return _match_general(title, description, cutoff)


    GATEWAY_SETTINGS = {
        "baseUrl": os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy"),
        "executionId": os.getenv("RUN_ID") or str(uuid4()),
        "endpoints": {
            "openai": "/api/gateway/openai/responses",
            "chutes": "/api/gateway/chutes/chat/completions",
            "desearch": "/api/gateway/desearch/search",
        },
        "reattempt": {
            "maxTries": 4,
            "backoffFactor": 2.0,
            "requestTimeout": 180.0,
            "transientCodes": (429, 500, 502, 503),
        },
    }

    PRIMARY_ENGINES = ("gpt-5.2",)

    FALLBACK_ENGINES = (
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "openai/gpt-oss-120b",
    )

    CURRENT_DATE_STAMP = datetime.utcnow().strftime("%Y-%m-%d")

    SPHERE_KEYWORDS = (
        ("athletics", frozenset({
            "MATCH", "GAME", "WIN", "VS",
            "CHAMPIONSHIP", "LEAGUE", "CUP",
            "PLAYOFF", "GOAL", "SCORE",
        })),
        ("governance", frozenset({
            "ELECTION", "VOTE", "POLL", "PRESIDENT",
            "GOVERNOR", "SENATOR", "MAYOR", "TARIFF",
            "SANCTION", "WAR", "TREATY",
        })),
        ("financial", frozenset({
            "RATE", "FED", "GDP", "INFLATION",
            "BITCOIN", "CRYPTO", "PRICE", "MARKET",
            "$", "STOCK",
        })),
        ("technology", frozenset({
            "LAUNCH", "RELEASE", "APP", "SOFTWARE",
            "UPDATE", "SHIP", "ANNOUNCE",
        })),
        ("showbusiness", frozenset({
            "MOVIE", "FILM", "OSCAR", "GRAMMY",
            "EMMY", "ALBUM", "BOX OFFICE", "AWARD",
        })),
    )

    SPHERE_INVESTIGATION_HINTS = {
        "athletics": (
            "betting odds", "injuries", "recent form",
            "head-to-head", "standings",
        ),
        "governance": (
            "Polymarket", "polling", "official statement", "Reuters AP",
        ),
        "financial": (
            "FedWatch", "central bank", "market expectations", "economic data",
        ),
        "technology": (
            "official blog", "press release", "SEC filing", "launch date",
        ),
        "showbusiness": (
            "box office", "reviews", "awards predictions", "release date",
        ),
        "general": (
            "Polymarket", "recent news", "official source",
        ),
    }

    KNOWN_SEGMENT_LABELS = frozenset({
        'sports', 'app store', 'weather', 'earnings',
        'election', 'inflation', 'price',
    })

    ADJUSTMENT_COEFFICIENTS = {
        "weather": (0.15, -1.386),  # tight squeeze toward 0.20 (base rate for 5-option batches)
    }

    INCUMBENT_CONTEXT = """\
    --- APP STORE: Incumbent lookup table START ---

    **Strategy:** Incumbent lookup table. Match event title against known incumbents for each
    rank/store combination. If matched, predict their historical win rate. Otherwise predict 0.05.
    Use this alongside your other context, this is what historicaly work very well.

    **Incumbent table (corrected Feb 26):**

    | Store | Rank | Primary Incumbent | Win Rate | Prediction |
    |-------|------|-------------------|----------|------------|
    | Free | #1 | ChatGPT | 92% | 0.92 |
    | Free | #2 | Google Gemini | 24% | 0.24 |
    | Free | #3 | Google Gemini | 24% | 0.24 |
    | Free | #4 | Threads | 36% | 0.36 |
    | Paid | #1 | Shadowrocket | 92% | 0.92 |
    | Paid | #2 | HotSchedules | 97% | 0.97 |
    | Paid | #3 | AnkiMobile Flashcards | 78% | 0.78 |
    | Paid | #4 | Procreate Pocket | 83% | 0.83 |

    **Secondary incumbents (win rate >10%, also deserving elevated predictions):**

    | Store | Rank | App | Win Rate | Prediction |
    |-------|------|----|----------|------------|
    | Free | #2 | Freecash | 28% | 0.28 |
    | Free | #3 | Threads | 29% | 0.29 |
    | Free | #4 | Google Gemini | 25% | 0.25 |
    | Paid | #3 | Procreate Pocket | 13% | 0.13 |
    | Paid | #4 | AnkiMobile | 17% | 0.17 |

    --- APP STORE: Incumbent lookup table END ---"""

    DEFAULT_LIKELIHOOD = 0.35
    SAFE_EPSILON = 1e-10
    SIGMOID_LIMIT = 500.0

    FORECASTER_SYSTEM = """You are an expert forecaster for prediction markets. Estimate P(YES) with rigorous research.
    CRITICAL: You MUST complete ALL your research in a SINGLE web_search tool call. You are NOT allowed to make multiple tool calls. Pack all your queries into one search. After receiving results, respond immediately.
    RESEARCH APPROACH - adapt your search strategy to the event type:
    For competitions and matches:
    - Search betting odds and convert to probability (decimal odds D → prob ≈ 1/D minus margin)
    - Check recent form, injuries, head-to-head records, rankings
    - Home advantage matters in team sports (+10-15%)
    For political events and elections:
    - Search Polymarket/PredictIt first - market prices are strong signals
    - For elections: check polling aggregates (538, RCP), consider historical polling errors
    - For policy/diplomatic events: prioritize official sources (Reuters, AP, government statements)
    - Check procedural requirements (votes needed, veto power, legislative calendar)
    For economic and financial events:
    - Search market-implied probabilities (CME FedWatch for rates, futures markets)
    - Check central bank communications and forward guidance
    - Economic calendar: what data releases occur before cutoff?
    For product launches and technology:
    - Check official company channels, press releases, SEC filings
    - Consider historical track record (announced vs actual delivery dates)
    - Distinguish between: announced, shipped, generally available
    For entertainment and awards:
    - Search prediction markets and expert consensus sites
    - Box office tracking, review aggregates
    - Awards predictions converge closer to ceremony date
    ALWAYS DO THESE:
    1. Search "Polymarket [topic]" - if market exists, price ≈ probability
    2. Search recent news (prioritize last 48-72 hours)
    3. Verify key claims with multiple sources
    4. Consider time until cutoff (more time = more uncertainty)
    ANALYSIS PRINCIPLES:
    - Polymarket price is your anchor - deviate only with strong contrary evidence
    - Official sources > speculation and rumors
    - Consider base rates: how often do similar events happen?
    - Resolution criteria are literal - read exact wording carefully
    - Range: never return exactly 0 or 1, use [0.01, 0.99]
    OUTPUT FORMAT:
    PREDICTION: [0.01-0.99]
    REASONING: [Key evidence, market signal if found, main uncertainties, 3-5 sentences]"""


    def constrainToValidRange(rawValue, floor=0.01, ceiling=0.99):
        if rawValue < floor:
            return floor
        if rawValue > ceiling:
            return ceiling
        return rawValue


    def computeSigmoidActivation(inputValue):
        bounded = max(-SIGMOID_LIMIT, min(SIGMOID_LIMIT, inputValue))
        return 1.0 / (1.0 + math.exp(-bounded))


    def computeLogOdds(likelihoodValue):
        safeVal = max(SAFE_EPSILON, min(1 - SAFE_EPSILON, likelihoodValue))
        return math.log(safeVal / (1 - safeVal))


    def computeNegativeLogSurvival(likelihoodValue):
        safeVal = max(SAFE_EPSILON, min(1 - SAFE_EPSILON, likelihoodValue))
        return -math.log(1 - safeVal)


    def identifySphereByScan(questionRecord):
        titleFragment = questionRecord.get("title", "")
        descriptionFragment = questionRecord.get("description", "")
        mergedUppercase = (titleFragment + " " + descriptionFragment).upper()
        for sphereLabel, termCollection in SPHERE_KEYWORDS:
            for singleTerm in termCollection:
                if singleTerm in mergedUppercase:
                    return sphereLabel
        return "general"


    def fetchInvestigationHints(sphereLabel):
        return SPHERE_INVESTIGATION_HINTS.get(
            sphereLabel, SPHERE_INVESTIGATION_HINTS["general"]
        )


    def classifyIntoSegment(questionRecord):
        metadataBlock = questionRecord.get('metadata', {})
        topicEntries = metadataBlock.get('topics', [])
        for topicValue in topicEntries:
            normalizedTopic = topicValue.lower()
            if normalizedTopic in KNOWN_SEGMENT_LABELS:
                return topicValue

        titleText = questionRecord.get('title', '')
        descriptionText = questionRecord.get('description', '')
        mergedLowercase = (titleText + ' ' + descriptionText).lower()
        titleLowercase = titleText.lower()

        if 'election' in mergedLowercase:
            return 'election'

        athleticBodyMarkers = (
            ' vs ', ' vs. ', 'upcoming game',
            'stoppage time', 'cricket',
            'both teams to score',
        )
        athleticTitleMarkers = (' win ', ' win?')
        bodyHit = False
        for marker in athleticBodyMarkers:
            if marker in mergedLowercase:
                bodyHit = True
                break
        titleHit = False
        for marker in athleticTitleMarkers:
            if marker in titleLowercase:
                titleHit = True
                break
        if bodyHit or titleHit:
            return 'Sports'

        if ' app ' in mergedLowercase or 'app store' in mergedLowercase:
            return 'App Store'

        if ' price of ' in mergedLowercase:
            return 'price'

        hasQuarterReference = (
            'q1' in mergedLowercase
            or 'q2' in mergedLowercase
            or 'q3' in mergedLowercase
            or 'q4' in mergedLowercase
        )
        if 'earnings' in mergedLowercase or (hasQuarterReference and 'above' in mergedLowercase):
            return 'Earnings'

        if 'inflation' in mergedLowercase:
            return 'inflation'

        if ' temperature ' in mergedLowercase:
            return 'Weather'

        return 'Other'


    def applySegmentAdjustment(rawLikelihood, segmentLabel):
        normalizedLabel = segmentLabel.lower()

        if normalizedLabel == 'weather':
            slopeVal, interceptVal = ADJUSTMENT_COEFFICIENTS["weather"]
            linearCombination = slopeVal * rawLikelihood + interceptVal
            return computeSigmoidActivation(linearCombination)

        return rawLikelihood


    def enrich_exact(eventData, gammaResult, is_weather=False):
        """Enrich prediction from an exact Polymarket match."""
        eid = eventData.get("event_id", "unknown")
        try:
            exact = gammaResult["exact_match"]
            marketQuestion = exact["quetion"]
            outcomePrices = json.loads(exact["outcome_prices"]) if isinstance(exact.get("outcome_prices"), str) else exact["outcome_prices"]
            enrichmentConfig = exact["enrichment_params"]
            pmPrice = constrainToValidRange(float(outcomePrices[0]))
            conversationPayload = [
                {"role": "developer", "content": assembleSystemDirective()},
                {"role": "user", "content": f"Market: {marketQuestion}\nCurrent price: {pmPrice:.0%}\nConfig: {enrichmentConfig}"},
            ]

            async def _enrich():
                timeoutConfig = httpx.Timeout(GATEWAY_SETTINGS["reattempt"]["requestTimeout"])
                async with httpx.AsyncClient(timeout=timeoutConfig) as httpSession:
                    replyText, _ = await invokeOpenAIEndpoint(httpSession, PRIMARY_ENGINES[0], conversationPayload)
                    return interpretForecastResponse(replyText)

            estimatedLikelihood, derivedRationale = asyncio.run(_enrich())
            return {"event_id": eid, "prediction": constrainToValidRange(estimatedLikelihood), "reasoning": derivedRationale}
        except Exception:
            if is_weather:
                return {"event_id": eid, "prediction": 0.20, "reasoning": ""}
            exact = gammaResult.get("exact_match") or {}
            op = exact.get("outcome_prices", "")
            price = DEFAULT_LIKELIHOOD
            if op:
                try:
                    pl = json.loads(op) if isinstance(op, str) else op
                    if isinstance(pl, list) and pl:
                        price = constrainToValidRange(float(pl[0]))
                except (json.JSONDecodeError, ValueError, IndexError):
                    pass
            return {"event_id": eid, "prediction": price, "reasoning": ""}


    def _isWeatherEvent(title):
        return ' temperature ' in (title or '').lower()


    def _isAppStoreEvent(title):
        return 'app store' in (title or '').lower()


    APP_STORE_SYSTEM = """You are an expert forecaster for prediction markets. Estimate P(YES) for app store ranking events.
    All relevant data is provided in the prompt — do not search the web.
    ANALYSIS PRINCIPLES:
    - Polymarket price is your anchor — deviate only with strong contrary evidence from the incumbent table
    - Incumbent win rates reflect historical dominance; weigh them against current market price
    - Consider time until cutoff (more time = more uncertainty)
    - Resolution criteria are literal — read exact wording carefully
    - Range: never return exactly 0 or 1, use [0.01, 0.99]
    OUTPUT FORMAT:
    PREDICTION: [0.01-0.99]
    REASONING: [Key evidence, market signal if found, main uncertainties, 3-5 sentences]"""


    def composeAppStoreInquiry(questionRecord, gammaContext):
        """Build LLM prompt for app store events with Gamma + incumbent context."""
        parts = [
            "EVENT TO FORECAST:",
            "Title: " + questionRecord.get("title", ""),
            "Description: " + questionRecord.get("description", ""),
            "Cutoff: " + questionRecord.get("cutoff", ""),
            "Today: " + CURRENT_DATE_STAMP,
            "",
            "POLYMARKET DATA (live market prices — use as primary signal):",
            gammaContext,
            "",
            INCUMBENT_CONTEXT,
        ]
        return "\n".join(parts)


    def reformatDeadlineTimestamp(questionData):
        rawDeadline = questionData.get("cutoff")
        if not isinstance(rawDeadline, str):
            return
        try:
            sanitized = rawDeadline.replace('Z', '+00:00')
            parsedDatetime = datetime.fromisoformat(sanitized)
            questionData["cutoff"] = parsedDatetime.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            pass


    def assembleSystemDirective():
        return FORECASTER_SYSTEM


    def composeUserInquiry(questionRecord):
        parts = [
            "EVENT TO FORECAST:",
            "Title: " + questionRecord.get("title", ""),
            "Description: " + questionRecord.get("description", ""),
            "Cutoff: " + questionRecord.get("cutoff", ""),
            "Today: " + CURRENT_DATE_STAMP,
            "Instructions:",
            "1. First, classify this event into one of the categories listed above",
            "2. Execute the category-specific searches",
            "3. Always search Polymarket for current market price",
            "4. Provide your prediction with structured reasoning",
        ]
        return "\n".join(parts)


    def composeSearchTermGenerationPrompt(hintCollection):
        hintString = ", ".join(hintCollection)
        parts = [
            "Generate 4 specific search queries for this forecasting event.",
            "Consider searching for: " + hintString,
            "Event: {event}",
            "Cutoff: {cutoff}",
            "Today: {today}",
            'Return JSON: {{"queries": ["query1", "query2", "query3", "query4"]}}',
        ]
        return "\n".join(parts)


    def composeEvidenceAnalysisPrompt():
        template_lines = [
            "Analyze search results and predict probability of YES resolution.",
            "Event: {event}",
            "Cutoff: {cutoff}",
            "Today: {today}",
            "Search Results:",
            "{context}",
            "Consider:",
            "- Market signals (Polymarket, betting odds) as primary anchor",
            "- Quality and recency of evidence",
            "- Time remaining until cutoff",
            "- Resolution criteria (exact wording matters)",
            "Return JSON:",
            '{{"probability": 0.XX, "reasoning": "your analysis with key evidence"}}',
        ]
        return "\n".join(template_lines)


    def isolateJsonFromText(textBlock):
        if not textBlock:
            return None
        fenceMarker = "```"
        fenceStart = textBlock.find(fenceMarker)
        if fenceStart >= 0:
            afterOpening = textBlock.find("\n", fenceStart)
            if afterOpening >= 0:
                closingFence = textBlock.find(fenceMarker, afterOpening)
                if closingFence >= 0:
                    fencedContent = textBlock[afterOpening:closingFence].strip()
                    braceOpen = fencedContent.find("{")
                    braceClose = fencedContent.rfind("}")
                    if braceOpen >= 0 and braceClose > braceOpen:
                        try:
                            return json.loads(fencedContent[braceOpen:braceClose + 1])
                        except (json.JSONDecodeError, ValueError, TypeError):
                            pass
        braceOpen = textBlock.find("{")
        braceClose = textBlock.rfind("}")
        if braceOpen >= 0 and braceClose > braceOpen:
            try:
                return json.loads(textBlock[braceOpen:braceClose + 1])
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        try:
            return json.loads(textBlock.strip())
        except (json.JSONDecodeError, ValueError, TypeError):
            return None


    def _pluckLeadingNumber(fragment):
        numericAccumulator = ""
        for character in fragment:
            if character.isdigit() or character == '.':
                numericAccumulator += character
            elif numericAccumulator:
                break
        if not numericAccumulator:
            return None
        try:
            return float(numericAccumulator)
        except (ValueError, TypeError):
            return None


    def _scanForDelimitedContent(fullText, openDelimiter, closeDelimiter):
        openPosition = fullText.find(openDelimiter)
        if openPosition < 0:
            return None
        contentStart = openPosition + len(openDelimiter)
        closePosition = fullText.find(closeDelimiter, contentStart)
        if closePosition < 0:
            return None
        return fullText[contentStart:closePosition].strip()


    def interpretForecastResponse(rawText):
        if not rawText:
            return 0.5, "No response received"

        estimatedLikelihood = None
        derivedRationale = ""

        taggedForecast = _scanForDelimitedContent(
            rawText, "<|forecast|>", "<|/forecast|>"
        )
        taggedRationale = _scanForDelimitedContent(
            rawText, "<|rationale|>", "<|/rationale|>"
        )

        if taggedForecast is not None:
            parsedNumeric = _pluckLeadingNumber(taggedForecast)
            if parsedNumeric is not None:
                estimatedLikelihood = constrainToValidRange(parsedNumeric)
        if taggedRationale:
            derivedRationale = taggedRationale

        if estimatedLikelihood is None:
            jsonObject = isolateJsonFromText(rawText)
            if jsonObject is not None:
                for candidateKey in (
                    "likelihood", "probability", "prediction",
                    "forecast", "prob", "p", "final_probability",
                ):
                    candidateValue = jsonObject.get(candidateKey)
                    if candidateValue is not None:
                        try:
                            estimatedLikelihood = constrainToValidRange(
                                float(candidateValue)
                            )
                            break
                        except (ValueError, TypeError):
                            continue
                if not derivedRationale:
                    for candidateKey in (
                        "rationale", "reasoning", "reason",
                        "analysis", "explanation",
                    ):
                        candidateValue = jsonObject.get(candidateKey)
                        if candidateValue:
                            derivedRationale = str(candidateValue)
                            break

        if estimatedLikelihood is None:
            loweredText = rawText.lower()
            searchMarkers = (
                "prediction:", "probability:", "estimate:",
                "forecast:", "likelihood:",
            )
            for marker in searchMarkers:
                markerPosition = loweredText.find(marker)
                if markerPosition < 0:
                    continue
                afterMarker = rawText[
                    markerPosition + len(marker):markerPosition + len(marker) + 40
                ].strip()
                if afterMarker.lower().startswith("is "):
                    afterMarker = afterMarker[3:]
                parsedNumeric = _pluckLeadingNumber(afterMarker)
                if parsedNumeric is not None:
                    if parsedNumeric > 1:
                        parsedNumeric = parsedNumeric / 100.0
                    estimatedLikelihood = constrainToValidRange(parsedNumeric)
                    break

        if not derivedRationale:
            rationaleMarkers = ("reasoning:", "rationale:", "analysis:", "explanation:")
            loweredForRationale = rawText.lower()
            for marker in rationaleMarkers:
                markerPosition = loweredForRationale.find(marker)
                if markerPosition < 0:
                    continue
                afterColon = rawText[markerPosition + len(marker):].strip()
                newlinePosition = afterColon.find("\n")
                if newlinePosition >= 0:
                    derivedRationale = afterColon[:newlinePosition].strip()
                else:
                    derivedRationale = afterColon.strip()
                if derivedRationale:
                    break

        if estimatedLikelihood is None:
            percentSignPosition = rawText.find('%')
            if percentSignPosition > 0:
                precedingChunk = rawText[max(0, percentSignPosition - 15):percentSignPosition]
                reversedDigits = ""
                for character in reversed(precedingChunk):
                    if character.isdigit() or character == '.':
                        reversedDigits = character + reversedDigits
                    elif reversedDigits:
                        break
                if reversedDigits:
                    try:
                        estimatedLikelihood = constrainToValidRange(
                            float(reversedDigits) / 100.0
                        )
                    except (ValueError, TypeError):
                        pass

        if estimatedLikelihood is None:
            scanIndex = 0
            textLength = len(rawText)
            while scanIndex < textLength - 2:
                if rawText[scanIndex] == '0' and rawText[scanIndex + 1] == '.':
                    digitEnd = scanIndex + 2
                    while digitEnd < textLength and rawText[digitEnd].isdigit():
                        digitEnd += 1
                    if digitEnd > scanIndex + 2:
                        try:
                            candidateDecimal = float(rawText[scanIndex:digitEnd])
                            if 0 < candidateDecimal < 1:
                                estimatedLikelihood = constrainToValidRange(candidateDecimal)
                                break
                        except (ValueError, TypeError):
                            pass
                scanIndex += 1

        if not derivedRationale and rawText:
            derivedRationale = rawText[:500].replace("\n", " ").strip()

        if estimatedLikelihood is None:
            fallbackRationale = derivedRationale if derivedRationale else "Unable to interpret forecast"
            return 0.5, fallbackRationale

        return estimatedLikelihood, derivedRationale


    def extractTextFromOpenAIReply(apiPayload):
        for outputBlock in apiPayload.get("output", []):
            if outputBlock.get("type") != "message":
                continue
            for contentPart in outputBlock.get("content", []):
                partKind = contentPart.get("type", "")
                if partKind in ("output_text", "text") and contentPart.get("text"):
                    return contentPart["text"]
        return ""


    async def withReattempts(asyncCallable, maxTries=None):
        reattemptConfig = GATEWAY_SETTINGS["reattempt"]
        effectiveCeiling = maxTries if maxTries is not None else reattemptConfig["maxTries"]
        attemptIndex = 0
        while True:
            try:
                return await asyncCallable()
            except httpx.TimeoutException:
                if attemptIndex >= effectiveCeiling - 1:
                    raise
                waitDuration = reattemptConfig["backoffFactor"] ** (attemptIndex + 1)
                await asyncio.sleep(waitDuration)
            except httpx.HTTPStatusError as httpErr:
                statusCode = httpErr.response.status_code
                if statusCode not in reattemptConfig["transientCodes"]:
                    raise
                if attemptIndex >= effectiveCeiling - 1:
                    raise
                waitDuration = reattemptConfig["backoffFactor"] ** (attemptIndex + 1)
                await asyncio.sleep(waitDuration)
            attemptIndex += 1


    async def invokeOpenAIEndpoint(httpSession, engineName, conversationPayload, toolDefinitions=None):
        requestPayload = {
            "model": engineName,
            "input": conversationPayload,
            "run_id": GATEWAY_SETTINGS["executionId"],
            "reasoning": {"effort": "medium"},
        }
        if toolDefinitions is not None:
            requestPayload["tools"] = toolDefinitions

        fullUrl = GATEWAY_SETTINGS["baseUrl"] + GATEWAY_SETTINGS["endpoints"]["openai"]

        async def performPost():
            serverResponse = await httpSession.post(fullUrl, json=requestPayload)
            serverResponse.raise_for_status()
            return serverResponse.json()

        rawData = await withReattempts(performPost)
        extractedText = extractTextFromOpenAIReply(rawData)
        costIncurred = rawData.get("cost", 0.0)
        return extractedText, costIncurred


    async def invokeChutesEndpoint(httpSession, engineName, userPromptText, tokenCeiling=2000):
        requestPayload = {
            "model": engineName,
            "messages": [{"role": "user", "content": userPromptText}],
            "max_tokens": tokenCeiling,
            "run_id": GATEWAY_SETTINGS["executionId"],
        }
        try:
            fullUrl = GATEWAY_SETTINGS["baseUrl"] + GATEWAY_SETTINGS["endpoints"]["chutes"]

            async def performPost():
                serverResponse = await httpSession.post(fullUrl, json=requestPayload)
                serverResponse.raise_for_status()
                return serverResponse.json()

            rawData = await withReattempts(performPost, maxTries=2)
            choicesList = rawData.get("choices", [])
            if not choicesList:
                return None
            firstChoice = choicesList[0]
            messageContent = firstChoice.get("message", {})
            return messageContent.get("content", "")
        except Exception:
            return None


    async def invokeDesearchEndpoint(httpSession, searchPhrase):
        requestPayload = {
            "query": searchPhrase,
            "model": "NOVA",
            "run_id": GATEWAY_SETTINGS["executionId"],
        }
        try:
            fullUrl = GATEWAY_SETTINGS["baseUrl"] + GATEWAY_SETTINGS["endpoints"]["desearch"]
            serverResponse = await httpSession.post(fullUrl, json=requestPayload)
            serverResponse.raise_for_status()
            return serverResponse.json().get("results", [])
        except Exception:
            return []


    async def attemptPrimaryEngines(httpSession, questionRecord):
        userInquiry = composeUserInquiry(questionRecord)
        conversationPayload = [
            {"role": "developer", "content": assembleSystemDirective()},
            {"role": "user", "content": userInquiry},
        ]
        webLookupCapability = [{"type": "web_search"}]
        for engineIdentifier in PRIMARY_ENGINES:
            try:
                print(f"[BARD] primary engine={engineIdentifier} (web_search)")
                replyText, _ = await invokeOpenAIEndpoint(
                    httpSession, engineIdentifier,
                    conversationPayload, webLookupCapability,
                )
                if not replyText:
                    print(f"[BARD] {engineIdentifier} returned empty response")
                    continue
                estimatedLikelihood, derivedRationale = interpretForecastResponse(replyText)
                print(f"[BARD] {engineIdentifier} -> {estimatedLikelihood:.3f}")
                return estimatedLikelihood, derivedRationale
            except Exception as engineFailure:
                print(f"[ENGINE] {engineIdentifier} encountered failure: {engineFailure}")
        return None


    async def produceSearchTerms(httpSession, questionRecord):
        detectedSphere = identifySphereByScan(questionRecord)
        relevantHints = fetchInvestigationHints(detectedSphere)
        eventBlob = (
            questionRecord.get("title", "") + "\n"
            + questionRecord.get("description", "")
        )
        cutoffStr = questionRecord.get("cutoff", "")
        promptTemplate = composeSearchTermGenerationPrompt(relevantHints)
        filledPrompt = promptTemplate.format(
            event=eventBlob[:1000],
            cutoff=cutoffStr,
            today=CURRENT_DATE_STAMP,
        )
        for engineIdentifier in FALLBACK_ENGINES[:2]:
            engineReply = await invokeChutesEndpoint(
                httpSession, engineIdentifier, filledPrompt, tokenCeiling=500
            )
            if engineReply is None:
                continue
            parsedJson = isolateJsonFromText(engineReply)
            if parsedJson is not None:
                termsList = parsedJson.get("searchTerms") or parsedJson.get("queries")
                if termsList:
                    return termsList[:4]
        abbreviatedTitle = questionRecord.get("title", "")[:80]
        return [
            "Polymarket " + abbreviatedTitle,
            abbreviatedTitle,
        ]


    async def gatherFindingsFromSearch(httpSession, searchTermList):
        collectedExcerpts = []
        for singleTerm in searchTermList:
            searchOutcomes = await invokeDesearchEndpoint(httpSession, singleTerm)
            topOutcomes = searchOutcomes[:3]
            for entry in topOutcomes:
                sourceUrl = entry.get("url", "source")
                sourceTitle = entry.get("title", "")
                excerptRaw = entry.get("snippet", "")
                excerptTrimmed = excerptRaw[:300]
                formattedEntry = "[" + sourceUrl + "] " + sourceTitle + ": " + excerptTrimmed
                collectedExcerpts.append(formattedEntry)
        if not collectedExcerpts:
            return "No findings from search"
        return "\n\n".join(collectedExcerpts)


    async def attemptFallbackEngines(httpSession, questionRecord):
        print(f"[BARD] fallback: desearch + chutes")
        searchTermList = await produceSearchTerms(httpSession, questionRecord)
        print(f"[BARD] search terms: {searchTermList}")
        findingsText = await gatherFindingsFromSearch(httpSession, searchTermList)
        eventBlob = (
            questionRecord.get("title", "") + "\n"
            + questionRecord.get("description", "")
        )
        cutoffStr = questionRecord.get("cutoff", "")
        promptTemplate = composeEvidenceAnalysisPrompt()
        filledPrompt = promptTemplate.format(
            event=eventBlob[:2000],
            cutoff=cutoffStr,
            today=CURRENT_DATE_STAMP,
            context=findingsText[:10000],
        )
        for engineIdentifier in FALLBACK_ENGINES:
            engineReply = await invokeChutesEndpoint(
                httpSession, engineIdentifier, filledPrompt, tokenCeiling=1500
            )
            if engineReply is None:
                continue
            parsedJson = isolateJsonFromText(engineReply)
            if parsedJson is None:
                continue
            rawLikelihoodVal = parsedJson.get("likelihood")
            if rawLikelihoodVal is None:
                rawLikelihoodVal = parsedJson.get("probability")
            if rawLikelihoodVal is None:
                continue
            estimatedVal = constrainToValidRange(float(rawLikelihoodVal))
            rationaleVal = parsedJson.get("rationale") or parsedJson.get("reasoning", "")
            print(f"[BARD] fallback {engineIdentifier} -> {estimatedVal:.3f}")
            return estimatedVal, rationaleVal
        print(f"[BARD] all fallback engines failed")
        return None


    async def attemptAppStoreWithContext(httpSession, questionRecord, gammaContext):
        """Run LLM with Gamma + incumbent context injected, no web search."""
        userInquiry = composeAppStoreInquiry(questionRecord, gammaContext)
        conversationPayload = [
            {"role": "developer", "content": APP_STORE_SYSTEM},
            {"role": "user", "content": userInquiry},
        ]
        for engineIdentifier in PRIMARY_ENGINES:
            try:
                replyText, _ = await invokeOpenAIEndpoint(
                    httpSession, engineIdentifier, conversationPayload,
                )
                if not replyText:
                    continue
                estimatedLikelihood, derivedRationale = interpretForecastResponse(replyText)
                return estimatedLikelihood, derivedRationale
            except Exception as engineFailure:
                print(f"[ENGINE/APP] {engineIdentifier} encountered failure: {engineFailure}")
        return None


    async def executeFullWorkflow(questionRecord, gammaResult=None):
        questionIdentifier = questionRecord.get("event_id", "unknown")
        timeoutConfig = httpx.Timeout(GATEWAY_SETTINGS["reattempt"]["requestTimeout"])
        async with httpx.AsyncClient(timeout=timeoutConfig) as httpSession:

            # App Store with Gamma context — LLM without web search
            if gammaResult and gammaResult.get("context"):
                appOutcome = await attemptAppStoreWithContext(
                    httpSession, questionRecord, gammaResult["context"])
                if appOutcome is not None:
                    return {
                        "event_id": questionIdentifier,
                        "prediction": appOutcome[0],
                        "reasoning": appOutcome[1],
                    }

            # Standard path: primary engines with web search
            primaryOutcome = await attemptPrimaryEngines(httpSession, questionRecord)
            if primaryOutcome is not None:
                return {
                    "event_id": questionIdentifier,
                    "prediction": primaryOutcome[0],
                    "reasoning": primaryOutcome[1],
                }
            fallbackOutcome = await attemptFallbackEngines(httpSession, questionRecord)
            if fallbackOutcome is not None:
                return {
                    "event_id": questionIdentifier,
                    "prediction": fallbackOutcome[0],
                    "reasoning": fallbackOutcome[1],
                }
        return {
            "event_id": questionIdentifier,
            "prediction": DEFAULT_LIKELIHOOD,
            "reasoning": "Every engine failed to produce a forecast",
        }


    def agent_main(event_data: dict) -> dict:
        questionIdentifier = event_data.get("event_id", "?")
        title = event_data.get("title", "")
        print(f"[BARD] segment={classifyIntoSegment(event_data)} title={title[:80]}")

        segmentLabel = classifyIntoSegment(event_data)

        reformatDeadlineTimestamp(event_data)

        # --- Gamma matcher: run first to get Polymarket data ---
        configure_gamma(
            GATEWAY_SETTINGS["baseUrl"] + "/api/gateway/desearch",
            GATEWAY_SETTINGS["executionId"],
        )
        cutoff_raw = event_data.get("cutoff", "")
        try:
            cutoff_dt = datetime.fromisoformat(str(cutoff_raw).replace('Z', '+00:00'))
        except Exception:
            cutoff_dt = datetime.utcnow()
        gammaResult = agent_match(title, event_data.get("description", ""), cutoff_dt)
        gammaStatus = gammaResult.get("status", "NO_MARKETS")
        print(f"[BARD] gamma={gammaStatus} cost=${get_gateway_cost():.4f}")

        # --- Weather: always bypass LLM, use calibration to squish toward 0.20 ---
        if _isWeatherEvent(title):
            outcome = enrich_exact(event_data, gammaResult, is_weather=True)
            rawLikelihood = float(outcome["prediction"])
            adjustedLikelihood = applySegmentAdjustment(rawLikelihood, "weather")
            print(f"[BARD] weather raw={rawLikelihood:.3f} adjusted={adjustedLikelihood:.3f}")
            return {
                "event_id": outcome["event_id"],
                "prediction": adjustedLikelihood,
                "reasoning": outcome.get("reasoning", "")[:2000],
            }

        # --- Exact match (non-weather): use PM price directly ---
        if gammaStatus == "EXACT":
            outcome = enrich_exact(event_data, gammaResult)
            print(f"[BARD] exact_match prediction={outcome['prediction']}")
            return {
                "event_id": outcome["event_id"],
                "prediction": float(outcome["prediction"]),
                "reasoning": outcome.get("reasoning", "")[:2000],
            }

        # --- App Store RELATED: LLM with Gamma context + incumbent table, no web search ---
        # --- Everything else: standard LLM + web search flow ---
        appStoreGamma = gammaResult if (_isAppStoreEvent(title) and gammaStatus == "RELATED") else None
        if appStoreGamma:
            print(f"[BARD] app_store RELATED -> LLM with gamma context (no web search)")
        else:
            print(f"[BARD] llm_flow: primary engines + web search")

        try:
            outcome = asyncio.run(executeFullWorkflow(event_data, gammaResult=appStoreGamma))
        except Exception as executionError:
            outcome = {
                "event_id": questionIdentifier,
                "prediction": DEFAULT_LIKELIHOOD,
                "reasoning": "Execution error: " + str(executionError)[:200],
            }

        rawLikelihood = float(outcome["prediction"])
        adjustedLikelihood = applySegmentAdjustment(rawLikelihood, segmentLabel)
        print(f"[BARD] raw={rawLikelihood:.3f} adjusted={adjustedLikelihood:.3f} segment={segmentLabel}")

        rationaleStr = str(outcome.get("reasoning", ""))
        if len(rationaleStr) > 2000:
            rationaleStr = rationaleStr[:2000]

        return {
            "event_id": outcome["event_id"],
            "prediction": adjustedLikelihood,
            "reasoning": rationaleStr,
        }
    return agent_main


async def _get_agent_mains():
    return await asyncio.gather(agent1(), agent2())


async def _run_ensemble(event_data: dict) -> dict:
    agent1_main, agent2_main = await _get_agent_mains()
    print("[ENSEMBLE] running agent_1 and agent_2")
    result1, result2 = await asyncio.gather(
        asyncio.to_thread(agent1_main, dict(event_data)),
        asyncio.to_thread(agent2_main, dict(event_data)),
        return_exceptions=True,
    )
    failed1 = isinstance(result1, Exception)
    failed2 = isinstance(result2, Exception)
    if failed1 and failed2:
        raise RuntimeError(f"agent1={result1} | agent2={result2}")
    if failed1:
        print(f"[ENSEMBLE] agent_1 failed, using agent_2: {result1}")
        return {
            "event_id": result2.get("event_id", event_data.get("event_id", "?")),
            "prediction": float(result2["prediction"]),
            "reasoning": str(result2.get("reasoning", ""))[:2000],
        }
    if failed2:
        print(f"[ENSEMBLE] agent_2 failed, using agent_1: {result2}")
        return {
            "event_id": result1.get("event_id", event_data.get("event_id", "?")),
            "prediction": float(result1["prediction"]),
            "reasoning": str(result1.get("reasoning", ""))[:2000],
        }
    prediction = 0.75 * float(result1["prediction"]) + 0.25 * float(result2["prediction"])
    print(
        f"[ENSEMBLE] p1={float(result1['prediction']):.3f} "
        f"p2={float(result2['prediction']):.3f} "
        f"blended={prediction:.3f}"
    )
    return {
        "event_id": result1.get("event_id", event_data.get("event_id", "?")),
        "prediction": prediction,
        "reasoning": str(result1.get("reasoning", ""))[:2000],
    }


def agent_main(event_data: dict) -> dict:
    try:
        return asyncio.run(_run_ensemble(event_data))
    except Exception as e:
        return {
            "event_id": event_data.get("event_id", "?"),
            "prediction": 0.35,
            "reasoning": f"Fatal error: {str(e)[:200]}",
        }


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    test_events = [
        {
            "event_id": "test-sports",
            "title": "Will Manchester United win vs Liverpool in the Premier League?",
            "description": "Resolves YES if Manchester United wins the match.",
            "cutoff": "2026-03-01T00:00:00Z",
            "metadata": {},
        },
        {
            "event_id": "test-appstore",
            "title": "Will ChatGPT be the #1 Free app in the US iPhone App Store on March 1, 2026?",
            "description": "Resolves based on the top free apps chart.",
            "cutoff": "2026-03-02T00:00:00Z",
            "metadata": {},
        },
        {
            "event_id": "test-earnings",
            "title": "Will AAPL Q1 2026 EPS be above $2.50?",
            "description": "Resolves YES if Apple reports Q1 2026 EPS above $2.50.",
            "cutoff": "2026-02-28T00:00:00Z",
            "metadata": {},
        },
    ]
    for evt in test_events:
        print(f"\n{'#'*60}")
        result = agent_main(evt)
        print(json.dumps(result, indent=2))