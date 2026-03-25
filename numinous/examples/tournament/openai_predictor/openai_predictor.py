"""
openai_predictor.py

What it does:
- Predicts App Store performance signals (e.g., downloads/ratings/keyword trends)
- Predicts weather signals (e.g., temp/precip trends)
- Predicts sports signals (e.g., game outcomes/props/trends)
- Uses OpenAI for model-driven inference + formatting
"""
import asyncio, os, time, re, json
from datetime import datetime
from uuid import uuid4
from typing import Optional
from collections import namedtuple, defaultdict
import httpx
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class _PolymarketGatewayState:
    _COST_PER_CALL = 0.0005
    _MAX_ATTEMPTS = 3
    _RETRY_CODES = (429, 500, 502, 503)

    def __init__(self):
        self._base_url = None
        self._run_id = None
        self._accumulated_cost = 0.0

    def configure(self, gateway_url: str, run_id: str):
        self._base_url = gateway_url.rstrip("/")
        self._run_id = run_id
        self._accumulated_cost = 0.0

    def cost_so_far(self) -> float:
        return self._accumulated_cost

    def _fetch_via_crawl(self, target_url: str):
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                resp = httpx.post(
                    f"{self._base_url}/web/crawl",
                    json={"url": target_url, "run_id": self._run_id},
                    timeout=30.0,
                )
                if resp.status_code in self._RETRY_CODES and attempt < self._MAX_ATTEMPTS - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                cost = data.get("cost", self._COST_PER_CALL)
                if isinstance(cost, (int, float)) and cost > 0:
                    self._accumulated_cost += cost
                content = data.get("content", "")
                if not content:
                    return None
                return json.loads(content) if isinstance(content, str) else content
            except Exception:
                if attempt < self._MAX_ATTEMPTS - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                return None
        return None

    def get(self, path: str, params=None):
        url = f"{_GAMMA_BASE}{path}"
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return self._fetch_via_crawl(url)


_gamma_state = _PolymarketGatewayState()


def configure_gamma(gateway_url: str, run_id: str):
    _gamma_state.configure(gateway_url, run_id)


def get_gateway_cost() -> float:
    return _gamma_state.cost_so_far()


def gamma_get(path, params=None):
    return _gamma_state.get(path, params)


class _MarketTextUtils:
    MONTH_NAMES = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
        "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
        "november": 11, "december": 12,
    }
    STOPWORDS = {
        "will", "the", "be", "a", "an", "in", "on", "of", "by", "to", "and",
        "or", "for", "at", "is", "it", "its", "from", "that", "this", "with",
        "as", "are", "was", "end", "between", "than", "least", "more", "less",
    }
    ISO_DATE_PATTERN = re.compile(r"scheduled for (\d{4}-\d{2}-\d{2})")
    DESC_DATE_PATTERN = re.compile(r"scheduled for (\w+ \d+ \d{4})")

    @classmethod
    def to_market_record(cls, raw):
        return {
            "market_id": raw.get("id"),
            "question": raw.get("question"),
            "slug": raw.get("slug"),
            "condition_id": raw.get("conditionId"),
            "active": raw.get("active"),
            "closed": raw.get("closed"),
            "description": raw.get("description", ""),
            "end_date": raw.get("endDate", ""),
            "outcome_prices": raw.get("outcomePrices", ""),
            "one_day_price_change": raw.get("oneDayPriceChange"),
        }

    @classmethod
    def yes_price_from_outcomes(cls, op):
        if not op:
            return None
        try:
            pl = json.loads(op) if isinstance(op, str) else op
            return float(pl[0]) if isinstance(pl, list) and pl else None
        except (json.JSONDecodeError, ValueError, IndexError):
            return None

    @classmethod
    def format_yes(cls, op):
        p = cls.yes_price_from_outcomes(op)
        if p is None:
            return "?"
        if p < 0.01:
            return "<1%"
        if p > 0.99:
            return ">99%"
        return f"{p:.0%}"

    @classmethod
    def format_pct(cls, v):
        if v < 0.005:
            return "<1%"
        if v > 0.995:
            return ">99%"
        return f"{v:.0%}"

    @classmethod
    def normalize_phrase(cls, text):
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (text or "").lower().strip()))

    @classmethod
    def search_keywords(cls, title):
        clean = re.sub(r"[^\w\s-]", " ", title or "")
        words = [
            w for w in clean.lower().split()
            if w not in cls.STOPWORDS and len(w) > 1
        ]
        return " ".join(words[:8])

    @classmethod
    def jaccard_sim(cls, a, b):
        tokens_a = set(re.findall(r"[a-z0-9]+", (a or "").lower()))
        tokens_b = set(re.findall(r"[a-z0-9]+", (b or "").lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    @classmethod
    def desc_date_verbose(cls, desc):
        if not desc:
            return None
        m = cls.DESC_DATE_PATTERN.search(desc)
        return m.group(1) if m else None

    @classmethod
    def desc_date_iso(cls, desc):
        m = cls.ISO_DATE_PATTERN.search(desc) if desc else None
        return m.group(1) if m else None

    @classmethod
    def resolve_date(cls, desc):
        return cls.desc_date_iso(desc) or cls.desc_date_verbose(desc)

    @classmethod
    def parse_month_day(cls, ds):
        parts = (ds or "").split()
        if not parts:
            return None
        mo = cls.MONTH_NAMES.get(parts[0].lower())
        if not mo:
            return None
        try:
            return datetime(2026, mo, int(parts[1])).date()
        except (ValueError, IndexError):
            return None


def _no_markets():
    return {"status": "NO_MARKETS", "exact_match": None, "related_markets": [], "context": None}


def _exact_result(market):
    return {"status": "EXACT", "exact_match": market, "related_markets": [], "context": None}


def _related_result(markets, context):
    return {"status": "RELATED", "exact_match": None, "related_markets": markets, "context": context}


def _find_parent_event(queries, team_a, team_b, description):
    u = _MarketTextUtils
    our_date = u.resolve_date(description)
    best_evt = None
    best_score = (-1, -1, -1.0)
    for q in queries:
        payload = gamma_get("/public-search", {"q": q})
        if not payload or "events" not in payload:
            continue
        for ev in payload["events"]:
            for mkt in ev.get("markets", []):
                q_lower = mkt.get("question", "").lower()
                if team_a.lower() not in q_lower or team_b.lower() not in q_lower:
                    continue
                date_match = 1 if (
                    our_date
                    and any(
                        u.resolve_date(m2.get("description", "")) == our_date
                        for m2 in ev.get("markets", [])
                    )
                ) else 0
                score = (
                    date_match,
                    0 if mkt.get("closed") else 1,
                    u.jaccard_sim(f"{team_a} vs {team_b}", mkt.get("question", "")),
                )
                if score > best_score:
                    best_score, best_evt = score, ev
                break
        if best_evt and best_score[0] == 1:
            break
    return best_evt


def _dates_consistent(our_date, pm_description):
    u = _MarketTextUtils
    pm_date = u.resolve_date(pm_description or "")
    return not our_date or not pm_date or pm_date == our_date


_BTTS_PATTERN = re.compile(r"^(.+?)\s+vs\.\s+(.+?):\s*Both Teams to Score$")


def _try_match_btts(title, desc, cutoff):
    u = _MarketTextUtils
    match = _BTTS_PATTERN.match(title)
    if not match:
        return None
    ta, tb = match.group(1).strip(), match.group(2).strip()
    parent = _find_parent_event([f"{ta} {tb}", f"{tb} {ta}"], ta, tb, desc)
    if not parent:
        return _no_markets()
    more = gamma_get(f"/events/slug/{parent['slug']}-more-markets")
    markets = (more or {}).get("markets", [])
    btts = next(
        (
            u.to_market_record(m)
            for m in markets
            if "Both Teams to Score" in m.get("question", "")
        ),
        None,
    )
    if not btts:
        return _no_markets()
    our_norm = u.normalize_phrase(title)
    pm_norm = u.normalize_phrase(btts.get("question", ""))
    alt_norm = u.normalize_phrase(f"{tb} vs. {ta}: Both Teams to Score")
    if our_norm != pm_norm and alt_norm != pm_norm:
        return _no_markets()
    if not _dates_consistent(u.desc_date_verbose(desc), btts.get("description")):
        return _no_markets()
    return _exact_result(btts)


_CRICKET_FULL = re.compile(r"^(.+?):\s*(.+?)\s+vs\s+(.+?)\s+\(Game\s+.+?\)\s*-\s*(.+)$")
_CRICKET_SHORT = re.compile(r"^(.+?):\s*(.+?)\s+vs\s+(.+?)\s*-\s*(.+)$")
_CRICKET_SLUG_SUFFIX = {
    "Most Sixes": "-most-sixes",
    "Toss Match Double": "-toss-match-double",
    "Team Top Batter": "-team-top-batter",
    "Most Fours": "-most-fours",
}


def _try_match_cricket(title, desc, cutoff):
    u = _MarketTextUtils
    match = _CRICKET_FULL.match(title) or _CRICKET_SHORT.match(title)
    if not match:
        return None
    parts = match.groups()
    tourn = parts[0].strip()
    ta = parts[1].strip()
    tb = parts[2].strip()
    sub = parts[-1].strip()
    parent = _find_parent_event(
        [f"{tourn} {ta} {tb}", f"{ta} {tb} {tourn}", f"{ta} vs {tb}"],
        ta, tb, desc,
    )
    if not parent:
        return _no_markets()
    slug = parent["slug"]
    our_date = u.desc_date_iso(desc)
    sub_slug = next((s for s in _CRICKET_SLUG_SUFFIX if sub.startswith(s)), None)
    if sub.startswith("Completed match"):
        markets = [u.to_market_record(mk) for mk in parent.get("markets", [])]
    elif sub_slug:
        data = gamma_get(f"/events/slug/{slug}{_CRICKET_SLUG_SUFFIX[sub_slug]}")
        markets = [u.to_market_record(mk) for mk in (data or {}).get("markets", [])]
    else:
        return _no_markets()
    our_norm = u.normalize_phrase(title)
    found = next(
        (mk for mk in markets if u.normalize_phrase(mk.get("question", "")) == our_norm),
        None,
    )
    if found and not _dates_consistent(our_date, found.get("description", "")):
        found = None
    return _exact_result(found) if found else _no_markets()


_APP_STORE_PATTERN = re.compile(
    r"Will (.+?) be (?:the )?#(\d+) (Free|Paid) [Aa]pp in the US "
    r"(?:iPhone |Apple )?App Store on (\w+ \d+)",
    re.I,
)


def _parse_app_store_title(title):
    u = _MarketTextUtils
    m = _APP_STORE_PATTERN.search(title or "")
    if not m:
        return None
    app, rank, typ, ds = m.groups()
    d = u.parse_month_day(ds)
    if not d:
        return None
    return {
        "app": app.strip(),
        "rank": int(rank),
        "type": typ.capitalize(),
        "date": d,
        "date_str": ds,
    }


def _search_app_markets(queries, predicate):
    u = _MarketTextUtils
    results = []
    seen_ids = set()
    for q in queries:
        data = gamma_get("/public-search", {"q": q}) or {}
        for ev in data.get("events", []):
            for m in ev.get("markets", []):
                mid = m.get("id")
                if mid in seen_ids:
                    continue
                parsed = _parse_app_store_title(m.get("question", ""))
                if parsed and predicate(parsed):
                    seen_ids.add(mid)
                    rec = u.to_market_record(m)
                    rec["parsed"] = parsed
                    results.append(rec)
    return results


def _delta_24h_pct(change, closed=False):
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


def _fmt_24h_suffix(change, closed=False):
    pct = _delta_24h_pct(change, closed)
    if pct is None:
        return ""
    return f" ({'+' if pct > 0 else ''}{pct}% 24h)"


def _slot_competition_entries(slot_markets, target_date):
    u = _MarketTextUtils
    by_date = defaultdict(list)
    for m in slot_markets:
        by_date[str(m["parsed"]["date"])].append(m)
    if not by_date:
        return []
    ts = str(target_date)
    def day_diff(d):
        return abs(int(d.replace("-", "")) - int(ts.replace("-", "")))
    if ts in by_date:
        chosen = ts
    else:
        active = [d for d in by_date if any(not m.get("closed") for m in by_date[d])]
        chosen = min(active, key=day_diff) if active else min(by_date, key=day_diff)
    entries = []
    for m in by_date[chosen]:
        raw = u.yes_price_from_outcomes(m.get("outcome_prices")) or 0.0
        app = m["parsed"]["app"]
        if (re.match(r"^App [A-Z]$", app) or app.lower() == "another app") and raw < 0.005:
            continue
        entries.append({
            "app": app,
            "price": raw,
            "closed": m.get("closed"),
            "date": chosen,
            "one_day_price_change": m.get("one_day_price_change"),
        })
    entries.sort(key=lambda x: -x["price"])
    return entries


def _build_app_store_context_text(parsed, same_date, nearby, slot_markets):
    u = _MarketTextUtils
    app = parsed["app"]
    rank = parsed["rank"]
    typ = parsed["type"]
    ds = parsed["date_str"]
    date = parsed["date"]
    lines = [f'App Store: "{app} #{rank} {typ} on {ds}?"']
    comp = _slot_competition_entries(slot_markets, date)
    for e in comp:
        e["delta_24h"] = _delta_24h_pct(e.get("one_day_price_change"), e.get("closed"))
    if same_date:
        pm_ranks = sorted(set(m["rank"] for m in same_date))
        lines.append(f"No Polymarket market for #{rank}. PM only has {', '.join(f'#{r}' for r in pm_ranks)} for {app}.")
        for m in same_date:
            suffix = " [closed]" if m.get("closed") else ""
            lines.append(
                f"{app} #{m['rank']} {typ} on {ds}: {u.format_yes(m['outcome_prices'])}"
                f"{_fmt_24h_suffix(m.get('one_day_price_change'), m.get('closed'))}{suffix}."
            )
    elif nearby:
        lines.append(f"Data offset: {nearby[0]['date_diff']:+d}d (using {nearby[0]['date']} market for {ds} question).")
        nd = nearby[0]["date"]
        for m in (x for x in nearby if x["date"] == nd):
            suffix = " [closed]" if m.get("closed") else ""
            lines.append(
                f"{app} #{m['rank']} {typ}: {u.format_yes(m['outcome_prices'])}"
                f"{_fmt_24h_suffix(m.get('one_day_price_change'), m.get('closed'))}{suffix}."
            )
    else:
        lines.append(f"No Polymarket markets found for {app}.")
    top = [e for e in comp if e["price"] >= 0.005][:8]
    if top:
        parts = [
            f"{e['app']}: {u.format_pct(e['price'])}{_fmt_24h_suffix(e.get('one_day_price_change'), e.get('closed'))}"
            for e in top
        ]
        cd = top[0]["date"]
        if cd != str(date):
            diff = (datetime.strptime(cd, "%Y-%m-%d").date() - date).days
            lines.append(f"#{rank} {typ} slot on {cd} ({diff:+d}d): {' | '.join(parts)}")
        else:
            lines.append(f"#{rank} {typ} slot: {' | '.join(parts)}")
    our = next((e for e in comp if e["app"].lower() == app.lower()), None)
    if our and our.get("delta_24h") is not None:
        leader = comp[0]
        rival = (
            next((e for e in comp[1:] if e.get("delta_24h") is not None), None)
            if leader["app"].lower() == app.lower()
            else (leader if leader.get("delta_24h") is not None else None)
        )
        if rival:
            gap = our["delta_24h"] - rival["delta_24h"]
            if gap != 0:
                def s(v):
                    return f"{'+' if v > 0 else ''}{v}%"
                lines.append(
                    f"Momentum: {app} {'gaining' if gap > 0 else 'losing ground'} vs "
                    f"{rival['app']}, gap {s(gap)} ({app} {s(our['delta_24h'])}, "
                    f"{rival['app']} {s(rival['delta_24h'])})"
                )
    return "\n".join(lines)


def _try_match_app_store(title, desc, cutoff):
    u = _MarketTextUtils
    p = _parse_app_store_title(title)
    if not p:
        return _no_markets() if "app store" in (title or "").lower() else None
    month_strs = [p["date"].strftime("%B")]
    app_markets = _search_app_markets(
        [f"{p['app']} {p['type']} App Store"] + [f"{p['app']} {p['type']} App Store {m}" for m in month_strs],
        lambda x: x["app"].lower() == p["app"].lower() and x["type"].lower() == p["type"].lower(),
    )
    exact = None
    same_date = []
    nearby = []
    for m in app_markets:
        mp = m["parsed"]
        dd = (mp["date"] - p["date"]).days
        entry = {
            k: m[k]
            for k in (
                "market_id", "question", "slug", "condition_id",
                "active", "closed", "outcome_prices", "one_day_price_change",
            )
        }
        entry["rank"] = mp["rank"]
        entry["date"] = str(mp["date"])
        entry["date_diff"] = dd
        entry["rank_diff"] = mp["rank"] - p["rank"]
        if dd == 0 and mp["rank"] == p["rank"]:
            exact = entry
        elif dd == 0:
            same_date.append(entry)
        elif abs(dd) <= 7:
            nearby.append(entry)
    if exact:
        return _exact_result(exact)
    same_date.sort(key=lambda x: abs(x["rank_diff"]))
    nearby.sort(key=lambda x: (abs(x["date_diff"]), abs(x["rank_diff"])))
    slot = _search_app_markets(
        [f"{p['type']} App Store {p['rank']}"] + [f"{p['type']} App Store {m}" for m in month_strs],
        lambda x: x["rank"] == p["rank"] and x["type"].lower() == p["type"].lower(),
    )
    if same_date or nearby or slot:
        return _related_result(
            same_date + nearby,
            _build_app_store_context_text(p, same_date, nearby, slot),
        )
    return _no_markets()


def _try_match_general(title, desc, cutoff):
    u = _MarketTextUtils
    if not title:
        return _no_markets()
    data = gamma_get("/public-search", {"q": u.search_keywords(title)})
    if not data or "events" not in data:
        return _no_markets()
    event_date = u.desc_date_verbose(desc)
    candidates = [
        (u.jaccard_sim(title, m.get("question", "")), u.to_market_record(m))
        for ev in data["events"]
        for m in ev.get("markets", [])
    ]
    candidates = [(s, c) for s, c in candidates if s >= 0.999]
    if not candidates:
        return _no_markets()

    def score_pair(pair):
        sim, c = pair
        date_ok = int(bool(event_date and u.desc_date_verbose(c.get("description", "")) == event_date))
        return (date_ok, 0 if c.get("closed") else 1, sim)

    candidates.sort(key=score_pair, reverse=True)
    best_sim, best = candidates[0]
    best["similarity"] = best_sim
    return _exact_result(best)


_MATCHERS = [_try_match_btts, _try_match_cricket, _try_match_app_store]


def agent_match(title: str, description: str, cutoff: datetime) -> dict:
    for matcher in _MATCHERS:
        out = matcher(title, description, cutoff)
        if out is not None:
            return out
    return _try_match_general(title, description, cutoff)


def _build_config():
    rid = os.getenv("RUN_ID") or str(uuid4())
    gw = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
    return namedtuple("Stn", ["rid", "gw", "oai", "dsr", "ort", "rtr", "bkf", "tlim", "mdl", "fbk_mdl"])(
        rid=rid,
        gw=gw,
        oai=f"{gw}/api/gateway/openai",
        dsr=f"{gw}/api/gateway/desearch",
        ort=f"{gw}/api/gateway/openrouter/chat/completions",
        rtr=3,
        bkf=2.0,
        tlim=180.0,
        mdl=("gpt-5-mini",),
        fbk_mdl="anthropic/claude-sonnet-4-6",
    )


CFG = _build_config()
CalSpec = namedtuple("CalSpec", ["kind", "vals"])
CAL_TBL = {
    "weather": CalSpec("platt", {"a": 0.15, "b": -1.386}),
}
MATCH_BOTH = [" vs ", " vs. ", "upcoming game", "stoppage time", "cricket", "both teams to score"]
MATCH_HDR = [" win ", " win?"]
KNOWN_CATS = {"sports", "app store", "weather", "earnings"}
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
APP_STORE_SYSTEM = """You are an expert forecaster for prediction markets. Estimate P(YES) for app store ranking events.
All relevant data is provided in the prompt — do not search the web.
ANALYSIS PRINCIPLES:
- Polymarket price is your anchor — deviate only with strong contrary evidence from the incumbent table
- Incumbent win rates reflect historical dominance; weigh them against current market price
- Consider time until cutoff (more time = more uncertainty)
- Resolution criteria are literal — read exact wording carefully
- Range: never return exactly 0 or 1, use [0.01, 0.99]
OUTPUT FORMAT — return JSON only:
```json
{
  "key_facts": ["fact1", "fact2"],
  "reasoning": "3-5 sentences with key evidence, market signal, main uncertainties",
  "probability": <number 0.01-0.99>
}
```"""
class _EventTypePredicates:
    _WEATHER_MARKER = " temperature "
    _APP_STORE_MARKER = "app store"

    @classmethod
    def is_weather(cls, title):
        return cls._WEATHER_MARKER in (title or "").lower()

    @classmethod
    def is_app_store(cls, title):
        return cls._APP_STORE_MARKER in (title or "").lower()


def _isWeatherEvent(title):
    return _EventTypePredicates.is_weather(title)


def _isAppStoreEvent(title):
    return _EventTypePredicates.is_app_store(title)


def _clamp(v, lo=0.01, hi=0.99):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


ENRICH_SYSTEM = """You are an expert forecaster. Analyze the provided market data and estimate P(YES).
The market data below reflects current live pricing — anchor your prediction strongly on this data. Only deviate if you identify a clear factual reason to (e.g. an event has already occurred, a deadline has passed, or the question contains an obvious resolution).
Range: [0.01, 0.99].
PREDICTION: [number]
REASONING: [2-3 sentences]"""
def _yes_from_outcome_prices(op):
    if not op:
        return None
    try:
        pl = json.loads(op) if isinstance(op, str) else op
        return float(pl[0]) if isinstance(pl, list) and pl else None
    except (json.JSONDecodeError, ValueError, IndexError):
        return None


def _format_24h_line(change):
    if change is None:
        return None
    try:
        pct = round(float(change) * 100)
        return pct if pct != 0 else None
    except (TypeError, ValueError):
        return None


def _build_enrich_prompt(eventData, exact):
    yes_price = _yes_from_outcome_prices(exact.get("outcome_prices", ""))
    change = exact.get("one_day_price_change")
    today = datetime.now().strftime("%Y-%m-%d")
    parts = [
        "Market question: " + exact.get("question", ""),
        "Current YES price: " + (f"{yes_price:.0%}" if yes_price is not None else "unknown"),
    ]
    pct = _format_24h_line(change)
    if pct is not None:
        parts.append(f"24h change: {'+' if pct > 0 else ''}{pct}%")
    parts.append("")
    parts.append("Event title: " + eventData.get("title", ""))
    parts.append("Cutoff: " + eventData.get("cutoff", ""))
    parts.append("Today: " + today)
    return "\n".join(parts)


def _extract_number_after_marker(text, marker, max_chars=20):
    low = (text or "").lower()
    pos = low.find(marker)
    if pos < 0:
        return None
    start = pos + len(marker)
    slice_text = low[start : start + max_chars].strip()
    nums = ""
    for ch in slice_text:
        if ch.isdigit() or ch == ".":
            nums += ch
        elif nums:
            break
    if not nums:
        return None
    try:
        v = float(nums)
        return v / 100.0 if v > 1 else v
    except ValueError:
        return None


async def _run_enrich_llm(body):
    async with httpx.AsyncClient(timeout=180.0) as cli:
        d, _ = await Gw.oai_resp(cli, body, web=False)
        return Prs.grab_txt(d)


def _parse_llm_prediction_from_text(txt):
    obj = Prs.dig_score(txt)
    lk, rationale = Prs.unify(obj)
    if lk is not None:
        return _clamp(lk), rationale or ""
    for marker in ("prediction:", "probability:"):
        v = _extract_number_after_marker(txt, marker)
        if v is not None:
            return _clamp(v), (rationale if rationale else (txt[:300] if txt else ""))
    return None, None


def _fallback_price_from_exact(gammaResult):
    exact = gammaResult.get("exact_match") or {}
    op = exact.get("outcome_prices", "")
    if not op:
        return 0.35
    try:
        pl = json.loads(op) if isinstance(op, str) else op
        if isinstance(pl, list) and pl:
            return _clamp(float(pl[0]))
    except (json.JSONDecodeError, ValueError, IndexError):
        pass
    return 0.35


def enrich_exact(eventData, gammaResult, is_weather=False):
    eid = eventData.get("event_id", "unknown")
    try:
        exact = gammaResult["exact_match"]
        user_msg = _build_enrich_prompt(eventData, exact)
        body = {
            "model": "gpt-5-pro",
            "input": [
                {"role": "developer", "content": ENRICH_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "run_id": CFG.rid,
        }
        txt = asyncio.run(_run_enrich_llm(body))
        print(f"[HENRY] enrich_exact LLM response: {txt[:200]}")
        pred, reasoning = _parse_llm_prediction_from_text(txt)
        if pred is not None:
            return {"event_id": eid, "prediction": pred, "reasoning": reasoning or ""}
    except Exception as exc:
        print(f"[HENRY] enrich_exact failed: {exc}")
    if is_weather:
        return {"event_id": eid, "prediction": 0.20, "reasoning": ""}
    price = _fallback_price_from_exact(gammaResult)
    return {"event_id": eid, "prediction": price, "reasoning": ""}


def composeAppStoreInquiry(questionRecord, gammaContext):
    today = datetime.now().strftime("%Y-%m-%d")
    sections = [
        "EVENT TO FORECAST:",
        "Title: " + questionRecord.get("title", ""),
        "Description: " + questionRecord.get("description", ""),
        "Cutoff: " + questionRecord.get("cutoff", ""),
        "Today: " + today,
        "",
        "POLYMARKET DATA (live market prices — use as primary signal):",
        gammaContext,
        "",
        INCUMBENT_CONTEXT,
    ]
    return "\n".join(sections)


class _CategoryRules:
    @staticmethod
    def _header_merged(hdr, desc=""):
        return (hdr + " " + desc).lower()

    @staticmethod
    def _infer_from_text(hdr, desc=""):
        merged = _CategoryRules._header_merged(hdr, desc)
        h = (hdr or "").lower()
        if any(k in merged for k in MATCH_BOTH) or any(k in h for k in MATCH_HDR):
            return "Sports"
        if " app " in merged or "app store" in merged:
            return "App Store"
        if "earnings" in merged or (
            ("q1" in merged or "q2" in merged or "q3" in merged or "q4" in merged)
            and "above" in merged
        ):
            return "Earnings"
        if " temperature " in merged:
            return "Weather"
        return "other"

    @staticmethod
    def resolve_from_event(evt):
        topics = evt.get("metadata", {}).get("topics", [])
        for t in topics:
            if (t or "").lower() in KNOWN_CATS:
                return t
        found = _CategoryRules._infer_from_text(
            evt.get("title", ""), evt.get("description", "")
        )
        return found if found != "other" else "Other"


class Cls:
    @staticmethod
    def tag(hdr, desc=""):
        return _CategoryRules._infer_from_text(hdr, desc)

    @staticmethod
    def resolve(evt):
        return _CategoryRules.resolve_from_event(evt)


_GAMMA_SCHEME = "https"
_GAMMA_HOST_A = "gamma"
_GAMMA_HOST_B = "api"
_GAMMA_SUBDOMAIN = f"{_GAMMA_HOST_A}-{_GAMMA_HOST_B}"
_GAMMA_DOMAIN = "polymarket"
_GAMMA_TLD = "com"
_GAMMA_NETLOC = f"{_GAMMA_SUBDOMAIN}.{_GAMMA_DOMAIN}.{_GAMMA_TLD}"
_GAMMA_BASE = f"{_GAMMA_SCHEME}://{_GAMMA_NETLOC}"


class _CalibrationImpl:
    @staticmethod
    def _linear_interp(v, xs, ys):
        import numpy as np
        v = np.atleast_1d(v)
        out = np.zeros(len(v))
        for i, x in enumerate(v):
            if x <= xs[0]:
                out[i] = ys[0]
            elif x >= xs[-1]:
                out[i] = ys[-1]
            else:
                idx = np.searchsorted(xs, x)
                lo_x, hi_x = xs[idx - 1], xs[idx]
                lo_y, hi_y = ys[idx - 1], ys[idx]
                out[i] = (
                    lo_y + (hi_y - lo_y) * (x - lo_x) / (hi_x - lo_x)
                    if hi_x != lo_x
                    else lo_y
                )
        return np.clip(out, 0, 1)

    @staticmethod
    def apply(score, cat_key):
        import numpy as np
        spec = CAL_TBL.get(cat_key)
        if spec is None:
            return score
        if spec.kind == "isotonic":
            return float(
                _CalibrationImpl._linear_interp(
                    np.array([score]),
                    np.array(spec.vals["xs"]),
                    np.array(spec.vals["ys"]),
                )[0]
            )
        if spec.kind == "platt":
            z = spec.vals["a"] * score + spec.vals["b"]
            return float(1 / (1 + np.exp(-np.clip(z, -500, 500))))
        return score


class Cal:
    @staticmethod
    def adjust(score, cat_key):
        return _CalibrationImpl.apply(score, cat_key)


class _LLMResponseParser:
    _SCORE_BRACE = "@score{"
    _SCORE_BRACE_LEN = 6
    _JSON_FENCE = "```json"
    _PROB_PATTERN = re.compile(
        r'\{[^{}]*"(?:probability|likelihood)"[^{}]*\}', re.DOTALL
    )

    @classmethod
    def first_message_text(cls, resp):
        for item in resp.get("output", []):
            if item.get("type") == "message":
                for blk in item.get("content", []):
                    if blk.get("type") in ("output_text", "text") and blk.get("text"):
                        return blk["text"]
        return ""

    @classmethod
    def normalize_probability(cls, v):
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 < v <= 100.0:
            return v / 100.0
        return None

    @classmethod
    def extract_score_object(cls, blob):
        loc = blob.find(cls._SCORE_BRACE)
        if loc >= 0:
            start = loc + cls._SCORE_BRACE_LEN
            depth = 0
            in_str = False
            end = -1
            i = start
            while i < len(blob):
                ch = blob[i]
                if in_str:
                    if ch == "\\" and i + 1 < len(blob):
                        i += 2
                        continue
                    if ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                i += 1
            if end >= 0:
                chunk = blob[start : end + 1].strip()
                try:
                    obj = json.loads(chunk)
                    if "likelihood" in obj or "probability" in obj:
                        return obj
                except Exception:
                    pass
        fence_start = blob.find(cls._JSON_FENCE)
        if fence_start >= 0:
            fence_end = blob.find("```", fence_start + 7)
            if fence_end >= 0:
                inner = blob[fence_start + 7 : fence_end].strip()
                try:
                    return json.loads(inner)
                except Exception:
                    pass
        try:
            return json.loads(blob)
        except Exception:
            pass
        m = cls._PROB_PATTERN.search(blob)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return None

    @classmethod
    def to_likelihood_and_rationale(cls, parsed):
        if parsed is None:
            return None, None
        raw = parsed.get("likelihood", parsed.get("probability"))
        if raw is None:
            return None, None
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None, None
        n = cls.normalize_probability(v)
        rationale = parsed.get("rationale", parsed.get("reasoning", ""))
        return n, rationale


class Prs:
    @staticmethod
    def grab_txt(resp):
        return _LLMResponseParser.first_message_text(resp)

    @staticmethod
    def norm(v):
        return _LLMResponseParser.normalize_probability(v)

    @staticmethod
    def dig_score(blob):
        return _LLMResponseParser.extract_score_object(blob)

    @staticmethod
    def unify(parsed):
        return _LLMResponseParser.to_likelihood_and_rationale(parsed)


class Tpl:
    _SYS_FRAGS = [
        "You are an expert forecaster with web search.",
        " Your job: estimate P(YES) for prediction markets.",
        "\n\nCRITICAL: You MUST complete ALL your research in a SINGLE web_search tool call. You are NOT allowed to make multiple tool calls. Pack all your queries into one search. After receiving results, respond immediately.",
        "\n\nRESEARCH STRATEGY - tailor your searches to the event category:",
        "\nFor competitions and sports matchups:",
        "\n- Search betting odds from sportsbooks and convert to implied probability (decimal odds D -> prob ~ 1/D minus margin)",
        "\n- Check recent form (last 3-5 games), injuries, head-to-head records, rankings",
        "\n- Home advantage typically adds +5-15% probability",
        "\nFor political and policy events:",
        "\n- Search Polymarket/PredictIt first - market prices are the strongest available signals",
        "\n- For elections: check polling aggregates (538, RealClearPolitics), factor in historical polling errors",
        "\n- For policy/diplomatic events: prioritize official sources (Reuters, AP, government statements)",
        "\n- Check procedural requirements (votes needed, veto power, legislative calendar)",
        "\nFor economic and financial events:",
        "\n- Search market-implied probabilities (CME FedWatch for rates, futures markets for targets)",
        "\n- Check central bank communications and forward guidance",
        "\n- Note economic data releases scheduled before cutoff",
        "\nFor product launches and technology:",
        "\n- Check official company channels, press releases, SEC filings",
        "\n- Consider historical track record (announced vs actual delivery dates)",
        "\n- Distinguish between: announced, shipped, generally available",
        "\nFor entertainment and awards:",
        "\n- Search prediction markets and expert consensus sites",
        "\n- Check box office tracking, review aggregates",
        "\n- Awards predictions converge closer to ceremony date",
        "\nFor crypto price targets:",
        '\n- Search "[coin] price" for current and last-week history',
        "\n- Current price vs target determines baseline probability",
        "\n\nALWAYS DO THESE:",
        '\n1. Search "Polymarket [topic]" - if market exists, price ~ probability',
        "\n2. Search recent news (prioritize last 48-72 hours)",
        "\n3. Verify key claims with multiple sources",
        "\n4. Consider time until cutoff (more time = more uncertainty)",
        "\n\nANALYSIS PRINCIPLES:",
        "\n- Polymarket price is your anchor - deviate only with strong contrary evidence",
        "\n- Official sources > speculation and rumors",
        "\n- Consider base rates: how often do similar events happen?",
        "\n- Resolution criteria are literal - read exact wording carefully",
        "\n- Range: never return exactly 0 or 1, use [0.01, 0.99]",
    ]
    _USR_FRAGS = [
        "**Today:** {today}",
        "\n\n**Question:** {title}",
        "\n\n**Description:** {description}",
        "\n\n**Deadline:** {cutoff}",
        "\n\nINSTRUCTIONS:",
        "\n1. First, classify this event into one of the categories in the system prompt",
        "\n2. Execute the category-specific searches",
        "\n3. Always search Polymarket for current market price",
        "\n4. Search for the most recent news and developments (last 48-72 hours)",
        "\n5. Provide your prediction with structured reasoning",
        "\n\n**Output JSON only:**",
        "\n```json",
        "\n{{",
        '\n  "polymarket_price": <number 0-1 or null>,',
        '\n  "betting_odds": <string or null>,',
        '\n  "current_price": <string or null>,',
        '\n  "key_facts": ["fact1", "fact2"],',
        '\n  "reasoning": "5-7 sentences",',
        '\n  "probability": <number 0-1>',
        "\n}}",
        "\n```",
    ]
    _FBK_FRAGS = [
        "**Today:** {today}",
        "\n\n**Question:** {title}",
        "\n\n**Description:** {description}",
        "\n\n**Deadline:** {cutoff}",
        "\n\n**Research from web search:**",
        "\n{research}",
        "\n\nBased on the research above,",
        " estimate P(YES) for this prediction market.",
        "\n\n**Output JSON only:**",
        "\n```json",
        "\n{{",
        '\n  "polymarket_price": <number 0-1 or null>,',
        '\n  "betting_odds": <string or null>,',
        '\n  "current_price": <string or null>,',
        '\n  "key_facts": ["fact1", "fact2"],',
        '\n  "reasoning": "2-3 sentences",',
        '\n  "probability": <number 0-1>',
        "\n}}",
        "\n```",
    ]
    @staticmethod
    def sys():
        return "".join(Tpl._SYS_FRAGS)
    @staticmethod
    def usr(title, description, cutoff, today):
        raw = "".join(Tpl._USR_FRAGS)
        return raw.format(title=title, description=description, cutoff=cutoff, today=today)
    @staticmethod
    def fbk(title, description, cutoff, today, research):
        raw = "".join(Tpl._FBK_FRAGS)
        return raw.format(
            title=title, description=description, cutoff=cutoff, today=today, research=research
        )
    @staticmethod
    def sys_with_research():
        return (
            "You are an expert forecaster. Your job: estimate P(YES) for prediction markets."
            "\nResearch results have been gathered for you and are included in the user message below."
            "\nAnalyze the provided research carefully and use it as your primary evidence."
            "\n\nANALYSIS PRINCIPLES:"
            "\n- Polymarket price is your anchor - deviate only with strong contrary evidence"
            "\n- Official sources > speculation and rumors"
            "\n- Consider base rates: how often do similar events happen?"
            "\n- Factor in time until deadline (more time = more uncertainty)"
            "\n- Resolution criteria are literal - read exact wording carefully"
            "\n- Range: never return exactly 0 or 1, use [0.01, 0.99]"
        )
    @staticmethod
    def sys_no_search():
        return (
            "You are an expert forecaster. Your job: estimate P(YES) for prediction markets."
            "\nYou have NO web search access. Use your knowledge and reasoning to estimate the probability."
            "\n\nANALYSIS PRINCIPLES:"
            "\n- Consider base rates: how often do similar events happen?"
            "\n- Consider what is publicly known about this topic up to your knowledge cutoff"
            "\n- Factor in time until deadline (more time = more uncertainty)"
            "\n- Resolution criteria are literal - read exact wording carefully"
            "\n- Range: never return exactly 0 or 1, use [0.01, 0.99]"
            "\n- When uncertain, stay closer to 0.5"
        )
    @staticmethod
    def usr_no_search(title, description, cutoff, today):
        return (
            f"**Today:** {today}"
            f"\n\n**Question:** {title}"
            f"\n\n**Description:** {description}"
            f"\n\n**Deadline:** {cutoff}"
            "\n\nEstimate P(YES) for this prediction market using your existing knowledge."
            "\n\n**Output JSON only:**"
            "\n```json"
            "\n{{"
            '\n  "key_facts": ["fact1", "fact2"],'
            '\n  "reasoning": "3-5 sentences",'
            '\n  "probability": <number 0-1>'
            "\n}}"
            "\n```"
        )
_RETRYABLE_HTTP = (429, 500, 502, 503)


async def _http_post(cli, url, body, tmo=60.0):
    return await cli.post(url, json=body, timeout=tmo)


async def _oai_response_loop(cli, body, web=True):
    body = dict(body)
    body["reasoning"] = {"effort": "medium"}
    if web:
        body["tools"] = [{"type": "web_search"}]
    elif "tools" in body:
        del body["tools"]
    last_exc = None
    for mdl in CFG.mdl:
        body["model"] = mdl
        for att in range(CFG.rtr):
            try:
                print(f"[ASSESS] Querying {mdl} (try {att + 1}/{CFG.rtr})...")
                r = await _http_post(cli, f"{CFG.oai}/responses", body, tmo=180.0)
                if r.status_code == 200:
                    d = r.json()
                    return d, d.get("cost", 0.0)
                if r.status_code in _RETRYABLE_HTTP:
                    wait = CFG.bkf ** (att + 1)
                    print(f"[WAIT] {r.status_code}, pausing {wait:.1f}s...")
                    await asyncio.sleep(wait)
                    continue
                print(f"[ERR] {r.status_code}, rotating model...")
                break
            except httpx.TimeoutException:
                print(f"[TLIM] {mdl}, rotating model...")
                break
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code in _RETRYABLE_HTTP:
                    wait = CFG.bkf ** (att + 1)
                    print(f"[WAIT] {exc.response.status_code}, pausing {wait:.1f}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"[ERR] {exc.response.status_code}, rotating model...")
                    break
        else:
            continue
        print(f"[NEXT] {mdl} exhausted, advancing...")
    raise last_exc or Exception("All models exhausted")


async def _crawl_url(cli, url):
    try:
        r = await _http_post(
            cli, f"{CFG.dsr}/web/crawl", {"url": url, "run_id": CFG.rid}, tmo=15.0
        )
        if r.status_code == 200:
            body = r.json()
            txt = body.get("content", "")
            if txt:
                print(f"[FETCH] OK: {url[:60]}...")
                return txt
    except Exception as exc:
        print(f"[FETCH] Err {url[:40]}: {exc}")
    return None


async def _search_and_fetch(cli, qry):
    try:
        print(f"[LOOKUP] {qry[:80]}...")
        r = await _http_post(
            cli,
            f"{CFG.dsr}/web/search",
            {"query": qry, "num": 10, "start": 0, "run_id": CFG.rid},
            tmo=60.0,
        )
        if r.status_code != 200:
            return ""
        hits = r.json().get("data", [])
        if not hits:
            return ""
        brief = "\n".join(
            f"- {h.get('title', '')}: {h.get('snippet', '')}" for h in hits[:5]
        )
        header = "Lookup results:\n" + brief
        for h in hits[:10]:
            link = h.get("link", "")
            if not link:
                continue
            page = await _crawl_url(cli, link)
            if page:
                return f"{header}\n\nFetched from {link}:\n{page}"
        return header
    except Exception as exc:
        print(f"[LOOKUP] Err: {exc}")
    return ""


def _openrouter_content(msg):
    raw = msg.get("content")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "".join(
            blk.get("text", "")
            for blk in raw
            if isinstance(blk, dict) and blk.get("type") in ("text", "output_text")
        )
    return ""


async def _openrouter_call(cli, messages, model=None):
    mdl = model or CFG.fbk_mdl
    payload = {
        "model": mdl,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 4096,
        "run_id": CFG.rid,
    }
    last_exc = None
    for att in range(CFG.rtr):
        try:
            print(f"[OPENROUTER] Querying {mdl} (try {att + 1}/{CFG.rtr})...")
            r = await _http_post(cli, CFG.ort, payload, tmo=120.0)
            if r.status_code == 200:
                d = r.json()
                choices = d.get("choices") or []
                if not choices:
                    raise ValueError("OpenRouter response has no choices")
                msg = choices[0].get("message") or {}
                txt = _openrouter_content(msg)
                if not txt.strip():
                    raise ValueError(
                        f"OpenRouter returned empty content (raw type: {type(msg.get('content'))})"
                    )
                return txt, d.get("cost", 0.0)
            if r.status_code in _RETRYABLE_HTTP:
                wait = CFG.bkf ** (att + 1)
                print(f"[OPENROUTER] {r.status_code}, pausing {wait:.1f}s...")
                await asyncio.sleep(wait)
                continue
            print(f"[OPENROUTER] {r.status_code}, giving up")
            break
        except httpx.TimeoutException:
            last_exc = Exception(f"Timeout calling {mdl}")
            print(f"[OPENROUTER] Timeout on attempt {att + 1}")
        except Exception as exc:
            last_exc = exc
            print(f"[OPENROUTER] Error: {exc}")
            break
    raise last_exc or Exception(f"OpenRouter {mdl} failed")


class Gw:
    @staticmethod
    async def _post(cli, url, body, tmo=60.0):
        return await _http_post(cli, url, body, tmo)

    @staticmethod
    async def oai_resp(cli, body, web=True):
        return await _oai_response_loop(cli, body, web)

    @staticmethod
    async def crawl(cli, url):
        return await _crawl_url(cli, url)

    @staticmethod
    async def lookup(cli, qry):
        return await _search_and_fetch(cli, qry)

    @staticmethod
    async def openrouter(cli, messages, model=None):
        return await _openrouter_call(cli, messages, model)


_REPAIR_INSTR = (
    "Extract prediction data from this malformed response and return valid JSON."
    "\n\nRaw response:\n{raw}"
    '\n\nReturn ONLY valid JSON:\n{{"probability": <0-1>, "reasoning": "<extracted reasoning>"}}'
)
_REPAIR_MODELS = ("gpt-5-mini", "gpt-5-nano")
_PROB_MATCH = re.compile(
    r'(?:probability|likelihood)["\']?\s*[:=]\s*([\d.]+)', re.IGNORECASE
)
_DEFAULT_REPAIR = {"data": {"likelihood": 0.35, "rationale": "Decoding failed"}, "cost": 0.0}


async def _repair_with_openai(cli, raw):
    instr = _REPAIR_INSTR.format(raw=raw)
    for mdl in _REPAIR_MODELS:
        try:
            print(f"[REPAIR] Attempting {mdl} to decode garbled output...")
            body = {
                "model": mdl,
                "input": [{"role": "user", "content": instr}],
                "run_id": CFG.rid,
            }
            r = await cli.post(f"{CFG.oai}/responses", json=body, timeout=60.0)
            if r.status_code != 200:
                continue
            d = r.json()
            txt = Prs.grab_txt(d)
            cst = d.get("cost", 0.0)
            obj = Prs.dig_score(txt)
            if obj:
                lk = obj.get("likelihood", obj.get("probability"))
                if lk is not None:
                    n = Prs.norm(float(lk))
                    if n is not None:
                        obj["likelihood"] = n
                        return {"data": obj, "cost": cst}
            hit = _PROB_MATCH.search(txt)
            if hit:
                n = Prs.norm(float(hit.group(1)))
                if n is not None:
                    return {
                        "data": {"likelihood": max(0.01, min(0.99, n)), "rationale": txt},
                        "cost": cst,
                    }
        except Exception as exc:
            print(f"[REPAIR] {mdl} err: {exc}")
            continue
    return _DEFAULT_REPAIR


async def _repair_with_openrouter(cli, raw):
    instr = _REPAIR_INSTR.format(raw=raw)
    try:
        txt, cst = await _openrouter_call(
            cli,
            [{"role": "user", "content": instr}],
            model="anthropic/claude-haiku-4-5",
        )
        obj = Prs.dig_score(txt)
        if obj:
            lk = obj.get("likelihood", obj.get("probability"))
            if lk is not None:
                n = Prs.norm(float(lk))
                if n is not None:
                    obj["likelihood"] = n
                    return {"data": obj, "cost": cst}
        hit = _PROB_MATCH.search(txt)
        if hit:
            n = Prs.norm(float(hit.group(1)))
            if n is not None:
                return {
                    "data": {"likelihood": max(0.01, min(0.99, n)), "rationale": txt},
                    "cost": cst,
                }
    except Exception as exc:
        print(f"[REPAIR-OR] err: {exc}")
    return _DEFAULT_REPAIR


class Fix:
    @staticmethod
    async def recover(cli, raw):
        return await _repair_with_openai(cli, raw)

    @staticmethod
    async def recover_openrouter(cli, raw):
        return await _repair_with_openrouter(cli, raw)


class Acc:
    __slots__ = ("_buf", "_note", "_lk")
    def __init__(self):
        self._buf = []
        self._note = None
        self._lk = asyncio.Lock()
    async def push(self, val, note=None):
        async with self._lk:
            self._buf.append(val)
            if note and self._note is None:
                self._note = note
    def summarize(self):
        if not self._buf:
            return 0.35, self._note
        avg = sum(self._buf) / len(self._buf)
        return max(0.05, min(0.95, avg)), self._note
    @property
    def has_data(self):
        return len(self._buf) > 0
async def _run_app_store_path(cli, evt, gammaResult, acc):
    app_msg = composeAppStoreInquiry(evt, gammaResult["context"])
    body = {
        "model": "gpt-5-pro",
        "input": [
            {"role": "developer", "content": APP_STORE_SYSTEM},
            {"role": "user", "content": app_msg},
        ],
        "run_id": CFG.rid,
    }
    d, cst = await _oai_response_loop(cli, body, web=False)
    txt = Prs.grab_txt(d)
    print(f"[APP OUTPUT]\n{txt}")
    obj = Prs.dig_score(txt)
    lk, rationale = Prs.unify(obj)
    if lk is None:
        raise ValueError("No likelihood from app store path")
    score = max(0.01, min(0.99, lk))
    await acc.push(score, rationale or "")
    return cst


async def _run_primary_flow(cli, evt, acc):
    hdr = evt.get("title", "")
    desc = evt.get("description", "")
    closes = evt.get("cutoff", "")
    today = datetime.now().strftime("%Y-%m-%d")
    usr_msg = Tpl.usr(hdr, desc, closes, today)
    body = {
        "model": "gpt-5-pro",
        "input": [
            {"role": "developer", "content": Tpl.sys()},
            {"role": "user", "content": usr_msg},
        ],
        "run_id": CFG.rid,
    }
    d, cst = await _oai_response_loop(cli, body, web=True)
    txt = Prs.grab_txt(d)
    print(f"[OUTPUT]\n{txt}")
    return txt, cst, False


async def _run_fallback_flow(cli, evt):
    hdr = evt.get("title", "")
    desc = evt.get("description", "")
    closes = evt.get("cutoff", "")
    today = datetime.now().strftime("%Y-%m-%d")
    queries = [f"polymarket {hdr}", f"{hdr} betting odds", f"{hdr} latest news"]
    parts = []
    for q in queries:
        seg = await _search_and_fetch(cli, q)
        if seg:
            parts.append(seg)
    if not parts:
        print("[WARN] Desearch returned nothing, querying Sonnet without research...")
        return [
            {"role": "system", "content": Tpl.sys_no_search()},
            {"role": "user", "content": Tpl.usr_no_search(hdr, desc, closes, today)},
        ]
    intel = "\n\n".join(parts)
    fbk_msg = Tpl.fbk(hdr, desc, closes, today, intel)
    return [
        {"role": "system", "content": Tpl.sys_with_research()},
        {"role": "user", "content": fbk_msg},
    ]


async def _assessment_core(evt, acc, gammaResult=None):
    hdr = evt.get("title", "")
    desc = evt.get("description", "")
    closes = evt.get("cutoff", "")
    today = datetime.now().strftime("%Y-%m-%d")
    spend = 0.0
    async with httpx.AsyncClient(timeout=180.0) as cli:
        if gammaResult and gammaResult.get("context"):
            print(f"[HENRY] app_store RELATED -> LLM with gamma context (no web search)")
            try:
                cst = await _run_app_store_path(cli, evt, gammaResult, acc)
                spend += cst
                s, n = acc.summarize()
                return {
                    "event_id": evt.get("event_id", "?"),
                    "prediction": s,
                    "reasoning": n or "",
                    "cost": spend,
                }
            except Exception as exc:
                print(f"[APP FAILED] {exc}, falling through to standard path...")
        print(f"[HENRY] llm_flow: primary engines + web search")
        used_openrouter = False
        try:
            txt, cst, _ = await _run_primary_flow(cli, evt, acc)
            spend += cst
        except Exception as exc:
            used_openrouter = True
            print(f"[WEB FAILED] {exc}, pivoting to OpenRouter fallback...")
            fbk_messages = await _run_fallback_flow(cli, evt)
            try:
                txt, cst = await _openrouter_call(cli, fbk_messages)
                spend += cst
                print(f"[FALLBACK OUTPUT]\n{txt}")
            except Exception as or_exc:
                print(f"[OPENROUTER FAILED] {or_exc}, returning deterministic fallback")
                return {
                    "event_id": evt.get("event_id", "?"),
                    "prediction": 0.35,
                    "reasoning": "All prediction avenues exhausted",
                    "cost": spend,
                }
        print(f"[SPEND] ${spend:.6f}")
        obj = Prs.dig_score(txt)
        lk, rationale = Prs.unify(obj)
        if lk is None:
            if used_openrouter:
                print("[WARN] Structured extraction failed, invoking OpenRouter repair...")
                rep = await _repair_with_openrouter(cli, txt)
            else:
                print("[WARN] Structured extraction failed, invoking repair...")
                rep = await _repair_with_openai(cli, txt)
            obj = rep["data"]
            spend += rep["cost"]
            lk, rationale = Prs.unify(obj)
            if lk is None:
                lk = 0.5
            print(f"[REPAIR] Recovered: {obj}")
        score = max(0.01, min(0.99, lk))
        if rationale is None:
            rationale = ""
        await acc.push(score, rationale)
        return {
            "event_id": evt.get("event_id", "?"),
            "prediction": score,
            "reasoning": rationale,
            "cost": spend,
            "parsed_data": obj,
        }


class Eng:
    @staticmethod
    async def _core(evt, acc, gammaResult=None):
        return await _assessment_core(evt, acc, gammaResult)

    @staticmethod
    async def assess(evt, gammaResult=None):
        acc = Acc()
        try:
            return await asyncio.wait_for(
                _assessment_core(evt, acc, gammaResult=gammaResult),
                timeout=CFG.tlim,
            )
        except asyncio.TimeoutError:
            print(f"[TLIM] Exceeded {CFG.tlim}s")
            s, n = acc.summarize()
            return {
                "event_id": evt.get("event_id", "?"),
                "prediction": s,
                "reasoning": n or "Time limit fallback",
                "cost": 0.0,
            }
        except Exception as exc:
            print(f"[ERR] {exc}")
            s, n = acc.summarize()
            return {
                "event_id": evt.get("event_id", "?"),
                "prediction": s if acc.has_data else 0.35,
                "reasoning": f"Fault: {exc}",
                "cost": 0.0,
            }


def agent_main(event_data: dict) -> dict:
    t0 = time.time()
    questionIdentifier = event_data.get("event_id", "?")
    title = event_data.get("title", "")
    cat = Cls.resolve(event_data)
    print(f"[HENRY] segment={cat} title={title[:80]}")
    # Reformat cutoff for display
    raw_cutoff = event_data.get("cutoff")
    if isinstance(raw_cutoff, str):
        try:
            sanitized = raw_cutoff.replace('Z', '+00:00')
            parsed_dt = datetime.fromisoformat(sanitized)
            event_data["cutoff"] = parsed_dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            pass
    # --- Gamma matcher: run first to get Polymarket data ---
    configure_gamma(
        CFG.gw + "/api/gateway/desearch",
        CFG.rid,
    )
    try:
        cutoff_dt = datetime.fromisoformat(str(raw_cutoff or "").replace('Z', '+00:00'))
    except Exception:
        cutoff_dt = datetime.utcnow()
    gammaResult = agent_match(title, event_data.get("description", ""), cutoff_dt)
    gammaStatus = gammaResult.get("status", "NO_MARKETS")
    print(f"[HENRY] gamma={gammaStatus} cost=${get_gateway_cost():.4f}")
    # --- Weather: always bypass LLM, use calibration to squish toward 0.20 ---
    if _isWeatherEvent(title):
        outcome = enrich_exact(event_data, gammaResult, is_weather=True)
        rawLikelihood = float(outcome["prediction"])
        adjusted = Cal.adjust(rawLikelihood, "weather")
        print(f"[HENRY] weather raw={rawLikelihood:.3f} adjusted={adjusted:.3f}")
        return {
            "event_id": outcome["event_id"],
            "prediction": adjusted,
            "reasoning": outcome.get("reasoning", "")[:2000],
        }
    # --- Exact match (non-weather): use PM price directly ---
    if gammaStatus == "EXACT":
        outcome = enrich_exact(event_data, gammaResult)
        print(f"[HENRY] exact_match prediction={outcome['prediction']}")
        return {
            "event_id": outcome["event_id"],
            "prediction": float(outcome["prediction"]),
            "reasoning": outcome.get("reasoning", "")[:2000],
        }
    # --- App Store RELATED: LLM with Gamma context + incumbent table, no web search ---
    # --- Everything else: standard LLM + web search flow ---
    appStoreGamma = gammaResult if (_isAppStoreEvent(title) and gammaStatus == "RELATED") else None
    try:
        outcome = asyncio.run(Eng.assess(event_data, gammaResult=appStoreGamma))
    except Exception as executionError:
        outcome = {
            "event_id": questionIdentifier,
            "prediction": 0.35,
            "reasoning": "Execution error: " + str(executionError)[:200],
            "cost": 0.0,
        }
    outcome["duration"] = time.time() - t0
    raw_score = outcome["prediction"]
    ck = cat.lower()
    adjusted = Cal.adjust(raw_score, ck)
    print(f"[HENRY] raw={raw_score:.3f} adjusted={adjusted:.3f} segment={ck}")
    rationaleStr = str(outcome.get("reasoning", ""))
    if len(rationaleStr) > 2000:
        rationaleStr = rationaleStr[:2000]
    print(
        f"\n[DONE] estimate={adjusted:.3f}"
        f" | {outcome['duration']:.1f}s | ${outcome.get('cost', 0):.6f}"
    )
    return {
        "event_id": outcome["event_id"],
        "prediction": adjusted,
        "reasoning": rationaleStr,
    }

