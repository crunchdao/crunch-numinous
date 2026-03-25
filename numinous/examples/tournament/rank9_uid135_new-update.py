import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Union
from uuid import uuid4

import httpx
from pydantic import BaseModel

_run_tag = os.getenv("RUN_ID") or str(uuid4())
_base_url = "%s/api/gateway/openai" % os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")


@dataclass(frozen=True)
class _Cfg:
    model_priority: tuple = ("gpt-5.2", "gpt-5", "gpt-5-mini")
    fixer_model: str = "gpt-5-nano"
    max_attempts: int = 4
    backoff_base: float = 2.0
    retriable_codes: frozenset = frozenset({429, 500, 502, 503})
    spending_cap: float = 0.90
    fallback_estimate: float = 0.35
    lower_bound: float = 0.05
    upper_bound: float = 0.95


CFG = _Cfg()

_SPORT_TOKENS = frozenset({
    " vs ", " vs. ", "upcoming game", "stoppage time", "cricket",
    "both teams to score", " win ", " win?", " draw?", "btts",
    "most sixes", "most fours", "top batter", "toss match double",
    "completed match",
})

_POLITICAL_TOKENS = (
    "election", "vote", "president", "tariff", "sanction", "war", "treaty",
    "governor", "senator", "congress", "parliament", "legislation", "bill",
    "executive order", "indictment", "impeach", "nato", "ceasefire",
    "diplomatic", "ambassador", "summit", "un general", "security council",
    "refugee", "border", "invasion", "annex",
)

_TEMP_PATTERN = re.compile(
    r"highest temperature|temperature in .+? (on|be)", re.IGNORECASE)


def detect_category(heading: str, body: str, meta: dict) -> str:
    combined = "%s %s" % (heading, body)
    combined = combined.lower()

    known = {t.lower() for t in meta.get("topics", [])}
    for label in ("sports", "weather", "app store", "earnings"):
        if label in known:
            return label

    if any(tok in combined for tok in _POLITICAL_TOKENS):
        return "geopolitics"
    if _TEMP_PATTERN.search(heading):
        return "weather"
    if "app store" in combined or " app " in combined:
        return "app_store"
    if any(tok in combined for tok in _SPORT_TOKENS):
        return "sports"
    has_qtr = any(q in combined for q in ("q1", "q2", "q3", "q4"))
    if ("earnings" in combined or has_qtr) and "above" in combined:
        return "earnings"
    if any(kw in combined for kw in ("box office", "movie", "film")):
        return "box_office"
    return "general"


_RANK_HISTORY = [
    ("free", 1, "ChatGPT", 0.92),
    ("free", 2, "Google Gemini", 0.24),
    ("free", 2, "Freecash", 0.28),
    ("free", 3, "Google Gemini", 0.24),
    ("free", 3, "Threads", 0.29),
    ("free", 4, "Threads", 0.36),
    ("free", 4, "Google Gemini", 0.25),
    ("paid", 1, "Shadowrocket", 0.92),
    ("paid", 2, "HotSchedules", 0.97),
    ("paid", 3, "AnkiMobile Flashcards", 0.78),
    ("paid", 3, "Procreate Pocket", 0.13),
    ("paid", 4, "Procreate Pocket", 0.83),
    ("paid", 4, "AnkiMobile", 0.17),
]

_APP_TITLE_RX = re.compile(
    r"Will (.+?) be (?:the )?#(\d+) (Free|Paid) [Aa]pp", re.IGNORECASE)


def lookup_incumbent(heading: str) -> Union[float, None]:
    hit = _APP_TITLE_RX.search(heading)
    if hit is None:
        return None
    app = hit.group(1).strip()
    pos = int(hit.group(2))
    tier = hit.group(3).lower()
    for s_tier, s_pos, s_name, s_rate in _RANK_HISTORY:
        if s_tier != tier or s_pos != pos:
            continue
        if s_name.lower() in app.lower() or app.lower() in s_name.lower():
            return s_rate
    return 0.05 if pos <= 4 else None


_WEATHER_PRIOR = 0.20
_WEATHER_SHRINK = 0.55

_DOMAIN_BOUNDS = {
    "weather":     (0.03, 0.92),
    "geopolitics": (0.10, 0.90),
    "earnings":    (0.10, 0.90),
    "sports":      (0.05, 0.95),
    "app_store":   (0.03, 0.97),
    "box_office":  (0.03, 0.97),
}


def _clamp(val: float, lo: float, hi: float) -> float:
    return lo if val < lo else (hi if val > hi else val)


def apply_calibration(
    raw: float, category: str, market_px: Union[float, None] = None,
) -> float:
    if category == "weather":
        adjusted = _WEATHER_PRIOR + (raw - _WEATHER_PRIOR) * _WEATHER_SHRINK
        return _clamp(adjusted, 0.03, 0.92)

    if category == "geopolitics":
        anchor = market_px if market_px is not None else 0.40
        blended = raw * 0.75 + anchor * 0.25
        return _clamp(blended, 0.10, 0.90)

    bounds = _DOMAIN_BOUNDS.get(category)
    if bounds:
        return _clamp(raw, bounds[0], bounds[1])

    return _clamp(raw, CFG.lower_bound, CFG.upper_bound)


class EventPayload(BaseModel):
    event_id: str
    title: str
    description: str
    cutoff: datetime
    metadata: dict


class SpendLedger:
    def __init__(self, cap: float = CFG.spending_cap):
        self._cap = cap
        self._total = 0.0

    @property
    def total(self) -> float:
        return self._total

    def has_room(self, needed: float = 0.005) -> bool:
        return (self._total + needed) <= self._cap

    def log_cost(self, amount: float) -> None:
        self._total += amount


def _grab_text(blob: dict) -> str:
    for entry in blob.get("output", []):
        if entry.get("type") != "message":
            continue
        for part in entry.get("content", []):
            if part.get("type") in ("output_text", "text"):
                val = part.get("text", "")
                if val:
                    return val
    return ""


def _try_fenced_block(raw: str) -> Union[dict, None]:
    hit = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if hit:
        try:
            return json.loads(hit.group(1))
        except json.JSONDecodeError:
            pass
    return None


def _try_direct_parse(raw: str) -> Union[dict, None]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _try_brace_match(raw: str) -> Union[dict, None]:
    idx = raw.find("{")
    if idx == -1:
        return None
    level = 0
    for pos in range(idx, len(raw)):
        ch = raw[pos]
        if ch == "{":
            level += 1
        elif ch == "}":
            level -= 1
            if level == 0:
                try:
                    candidate = json.loads(raw[idx:pos + 1])
                    if "probability" in candidate:
                        return candidate
                except json.JSONDecodeError:
                    pass
                break
    return None


def _try_regex_prob(raw: str) -> Union[dict, None]:
    hit = re.search(
        r'\{[^{}]*"probability"\s*:\s*[\d.]+[^{}]*\}', raw, re.DOTALL)
    if hit:
        try:
            return json.loads(hit.group())
        except json.JSONDecodeError:
            pass
    return None


def _try_prediction_line(raw: str) -> Union[dict, None]:
    for ln in raw.strip().split("\n"):
        if "PREDICTION:" not in ln.upper():
            continue
        num_hit = re.search(r"[\d.]+", ln.split(":", 1)[-1])
        if num_hit:
            v = float(num_hit.group())
            return {"probability": v / 100 if v > 1 else v, "reasoning": ""}
    return None


_EXTRACTORS = (
    _try_fenced_block,
    _try_direct_parse,
    _try_brace_match,
    _try_regex_prob,
    _try_prediction_line,
)


def decode_structured(raw: str) -> Union[dict, None]:
    if not raw:
        return None
    for fn in _EXTRACTORS:
        result = fn(raw)
        if result is not None:
            return result
    return None


def read_probability(obj: Union[dict, None]) -> Union[float, None]:
    if obj is None:
        return None
    for attr in ("probability", "prediction", "forecast", "prob", "p"):
        val = obj.get(attr)
        if val is None:
            continue
        try:
            num = float(val)
            if 0 <= num <= 1:
                return num
        except (ValueError, TypeError):
            continue
    return None


def read_market_price(obj: Union[dict, None]) -> Union[float, None]:
    if obj is None:
        return None
    val = obj.get("polymarket_price")
    if val is not None:
        try:
            num = float(val)
            return num if 0 < num < 1 else None
        except (ValueError, TypeError):
            pass
    return None


async def with_retries(action, attempts: int = CFG.max_attempts):
    last_exc = None
    for n in range(attempts):
        try:
            return await action()
        except httpx.TimeoutException as exc:
            last_exc = exc
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code not in CFG.retriable_codes:
                raise
        except (httpx.ConnectError, httpx.RequestError) as exc:
            last_exc = exc
        if n < attempts - 1:
            await asyncio.sleep(CFG.backoff_base ** (n + 1))
    raise last_exc or RuntimeError("Retries exhausted")


class ModelClient:
    def __init__(self, http: httpx.AsyncClient, ledger: SpendLedger):
        self._http = http
        self._ledger = ledger

    async def invoke(
        self, model_name: str, conversation: list[dict],
        search: bool = True, wait: float = 140.0,
    ) -> tuple[str, float]:
        if not self._ledger.has_room():
            raise RuntimeError("Spending cap reached")

        transformed = [
            {"role": ("developer" if msg["role"] == "system" else msg["role"]),
             "content": msg["content"]}
            for msg in conversation
        ]
        body: dict = {
            "model": model_name, "input": transformed, "run_id": _run_tag,
        }
        if search:
            body["tools"] = [{"type": "web_search"}]

        async def _post():
            r = await self._http.post(
                "%s/responses" % _base_url, json=body, timeout=wait)
            r.raise_for_status()
            return r.json()

        payload = await with_retries(_post)
        content = _grab_text(payload)
        cost = payload.get("cost", 0.0)
        self._ledger.log_cost(cost)
        return content, cost

    async def cascade(
        self, conversation: list[dict], search: bool = True,
    ) -> tuple[str, float]:
        accumulated = 0.0
        for mdl in CFG.model_priority:
            try:
                print("[FCS] trying %s ..." % mdl)
                txt, c = await self.invoke(mdl, conversation, search)
                accumulated += c
                if txt:
                    return txt, accumulated
            except Exception as err:
                print("[FCS] %s error: %s" % (mdl, err))
        return "", accumulated

    async def fix_malformed(self, broken_text: str) -> Union[dict, None]:
        instruction = (
            "Extract the probability from this forecaster output.\n"
            'Return ONLY valid JSON: {"probability": <float 0-1>,'
            ' "reasoning": "<brief>"}\n\n'
            + broken_text[:1500]
        )
        try:
            async def _post():
                r = await self._http.post(
                    "%s/responses" % _base_url,
                    json={
                        "model": CFG.fixer_model,
                        "input": [{"role": "user", "content": instruction}],
                        "run_id": _run_tag,
                    },
                    timeout=25.0)
                r.raise_for_status()
                return r.json()

            payload = await with_retries(_post, attempts=2)
            content = _grab_text(payload)
            self._ledger.log_cost(payload.get("cost", 0.0))
            structured = decode_structured(content)
            if structured and "probability" in structured:
                return structured
            fallback = re.search(r"0\.\d+", content)
            if fallback:
                return {"probability": float(fallback.group()), "reasoning": ""}
        except Exception:
            pass
        return None


_SYSTEM_ALPHA = (
    "You are an expert forecaster with web search. Estimate P(YES)"
    " for prediction markets.\n"
    "Your predictions are scored by Brier score:"
    " (prediction - outcome)\u00b2. LOWER IS BETTER.\n"
    "\n"
    "MANDATORY RESEARCH (do ALL before answering):\n"
    "1. Search \"polymarket [topic]\" \u2014"
    " Polymarket YES price is your strongest anchor\n"
    "2. Search \"[topic] betting odds\""
    " for sportsbooks (if sports)\n"
    "3. Search \"[topic] latest news\""
    " for recent developments\n"
    "4. Check if event has ALREADY resolved"
    " \u2014 if so, use actual outcome\n"
    "\n"
    "CALIBRATION:\n"
    "- Polymarket price \u2248 your prior."
    " Only adjust with strong contrary evidence.\n"
    "- Genuinely uncertain with no signal \u2192 0.35\n"
    "- Weak evidence \u2192 0.25-0.40 or 0.55-0.65\n"
    "- Moderate evidence \u2192 0.15-0.30 or 0.65-0.80\n"
    "- Strong evidence \u2192 0.08-0.15 or 0.80-0.92\n"
    "- NEVER above 0.95 or below 0.05"
    " unless CONFIRMED resolved\n"
    "- Geopolitical events are inherently uncertain"
    " \u2014 stay moderate (0.15-0.85)\n"
    "- Cricket \"Draw\" props \u2192 very rare"
    " \u2192 predict low (<0.10)\n"
    "- Earnings: ~70% of companies beat EPS consensus\n"
    "\n"
    "ODDS CONVERSION:\n"
    "American negative (-X): P = X/(X+100)."
    " American positive (+X): P = 100/(X+100)\n"
    "\n"
    "Output ONLY valid JSON:\n"
    "```json\n"
    "{{\"polymarket_price\": <0-1 or null>,"
    " \"betting_odds\": <string or null>,"
    " \"key_facts\": [\"fact1\", \"fact2\"],"
    " \"reasoning\": \"2-3 sentences\","
    " \"probability\": <0.05-0.95>}}\n"
    "```"
)


_SYSTEM_BETA = (
    "You are a calibrated probabilistic forecaster."
    " Estimate P(YES) for a prediction market.\n"
    "Scored by Brier score:"
    " (prediction - outcome)\u00b2. LOWER IS BETTER.\n"
    "\n"
    "STEPS:\n"
    "1. Search Polymarket for current market price"
    " \u2014 this is your PRIOR\n"
    "2. Search for relevant news, odds, or official data\n"
    "3. Check if event already happened"
    " \u2014 use actual outcome if so\n"
    "4. Adjust prior only with strong evidence."
    " No evidence = trust the market.\n"
    "\n"
    "CRITICAL RULES:\n"
    "- Range [0.05, 0.95]. NEVER exceed these bounds.\n"
    "- Polymarket price is the best signal available."
    " Respect it.\n"
    "- Geopolitics: stay moderate."
    " Tariffs get delayed, bills stall, summits fail."
    " Don't be extreme.\n"
    "- Sports: use sportsbook implied probability."
    " Draws ~27%.\n"
    "- Weather exact degree:"
    " max ~0.30 for any single degree.\n"
    "- When uncertain, predict 0.35"
    " (markets skew toward NO resolution).\n"
    "\n"
    "Output ONLY valid JSON:\n"
    "```json\n"
    "{{\"polymarket_price\": <0-1 or null>,"
    " \"reasoning\": \"2-3 sentences\","
    " \"probability\": <0.05-0.95>}}\n"
    "```"
)


_QUESTION_FMT = (
    "**Today:** {now}\n"
    "**Question:** {q}\n"
    "**Description:** {desc}\n"
    "**Deadline:** {dl}\n"
    "\n"
    "Research this question thoroughly,"
    " then provide your probability estimate as JSON."
)


class ProbabilityEngine:
    def __init__(self):
        self._ledger = SpendLedger(CFG.spending_cap)

    async def estimate(self, event: EventPayload) -> dict:
        category = detect_category(
            event.title, event.description, event.metadata)
        print("[FCS] category=%s | %s" % (category, event.title[:80]))

        dl = event.cutoff
        if dl.tzinfo is None:
            dl = dl.replace(tzinfo=timezone.utc)
        dl_label = dl.strftime("%Y-%m-%d %H:%M UTC")
        now_label = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        question = _QUESTION_FMT.format(
            now=now_label, q=event.title, desc=event.description, dl=dl_label)

        async with httpx.AsyncClient() as session:
            client = ModelClient(session, self._ledger)

            task_a = self._run_single(client, _SYSTEM_ALPHA, question)
            task_b = self._run_single(client, _SYSTEM_BETA, question)
            out_a, out_b = await asyncio.gather(task_a, task_b)

            p_a, mkt_a, txt_a = out_a
            p_b, mkt_b, txt_b = out_b

            print("[FCS] alpha=%s beta=%s" % (p_a, p_b))

            market_px = mkt_a if mkt_a is not None else mkt_b

            match (p_a is not None, p_b is not None):
                case (True, True):
                    merged = (p_a + p_b) / 2.0
                    explanation = txt_a or txt_b
                case (True, False):
                    merged = p_a
                    explanation = txt_a
                case (False, True):
                    merged = p_b
                    explanation = txt_b
                case _:
                    merged = CFG.fallback_estimate
                    explanation = "No valid prediction obtained"

            if category == "app_store":
                inc = lookup_incumbent(event.title)
                if inc is not None:
                    merged = inc * 0.6 + merged * 0.4

            final = apply_calibration(merged, category, market_px)
            print("[FCS] raw=%.3f cal=%.3f mkt=%s cat=%s"
                  % (merged, final, market_px, category))

            return {
                "event_id": event.event_id,
                "prediction": final,
                "reasoning": (explanation[:1000] if explanation
                              else "Forecast from web research"),
            }

    async def _run_single(
        self, client: ModelClient, sys_prompt: str, user_msg: str,
    ) -> tuple[Union[float, None], Union[float, None], str]:
        conversation = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]

        raw_text, _ = await client.cascade(conversation, search=True)
        structured = decode_structured(raw_text) if raw_text else None

        if not structured or "probability" not in structured:
            if raw_text:
                structured = await client.fix_malformed(raw_text)

        prob = read_probability(structured)
        mkt = read_market_price(structured)
        reason = (structured or {}).get("reasoning", "")

        return prob, mkt, reason


async def _execute(event: EventPayload) -> dict:
    engine = ProbabilityEngine()
    t0 = time.time()
    print("\n[FCS] === %s ===" % event.event_id)
    print("[FCS] %s" % event.title[:100])
    output = await engine.estimate(event)
    elapsed = time.time() - t0
    print("[FCS] Done %.1fs | $%.4f | P=%.3f"
          % (elapsed, engine._ledger.total, output["prediction"]))
    return output


def agent_main(event_data: dict) -> dict:
    evt = EventPayload.model_validate(event_data)
    return asyncio.run(_execute(evt))


if __name__ == "__main__":
    sample = {
        "event_id": "test-ult7-001",
        "title": "Will Bitcoin exceed $100,000 by end of March 2026?",
        "description": (
            "Resolves YES if BTC exceeds $100,000 USD"
            " on any major exchange before March 31, 2026."
        ),
        "cutoff": datetime(2026, 3, 31, 23, 59, tzinfo=timezone.utc),
        "metadata": {},
    }
    print(json.dumps(agent_main(sample), indent=2))
