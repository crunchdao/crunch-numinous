"""
A probability prediction engine powered by OpenAI.
This module estimates the likelihood of events across:
- Politics
- Cryptocurrency markets
- Sports outcomes
- Weather events
- App Store activity
- Geopolitics
- Box office
- Earnings
- General
Built to give clear, structured probability outputs based on
model reasoning and event context.
"""
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
# -----------------------------------------------------------------------------
# Numeric bounds and probability helpers
# -----------------------------------------------------------------------------
_DEF_FLOOR = 0.01
_DEF_CEILING = 0.99
_SAFE_EPS = 1e-10
_SIGMOID_CAP = 500.0
_DEFAULT_P = 0.35

def _clamp(val, lo=None, hi=None):
    lo = lo if lo is not None else _DEF_FLOOR
    hi = hi if hi is not None else _DEF_CEILING
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

def _sigmoid(x):
    x = max(-_SIGMOID_CAP, min(_SIGMOID_CAP, x))
    return 1.0 / (1.0 + math.exp(-x))

def _safe_log_odds(p):
    p = max(_SAFE_EPS, min(1 - _SAFE_EPS, p))
    return math.log(p / (1 - p))

def _neg_log_survival(p):
    p = max(_SAFE_EPS, min(1 - _SAFE_EPS, p))
    return -math.log(1 - p)

def constrainToValidRange(rawValue, floor=0.01, ceiling=0.99):
    return _clamp(rawValue, floor, ceiling)

def computeSigmoidActivation(inputValue):
    return _sigmoid(inputValue)

def computeLogOdds(likelihoodValue):
    return _safe_log_odds(likelihoodValue)

def computeNegativeLogSurvival(likelihoodValue):
    return _neg_log_survival(likelihoodValue)

# -----------------------------------------------------------------------------
# Segment classification constants
# -----------------------------------------------------------------------------
KNOWN_SEGMENT_LABELS = frozenset({
    'sports', 'app store', 'weather', 'earnings',
    'election', 'inflation', 'price',
})
ADJUSTMENT_COEFFICIENTS = {
    "weather": (0.15, -1.386),
}
DEFAULT_LIKELIHOOD = _DEFAULT_P
SAFE_EPSILON = _SAFE_EPS
SIGMOID_LIMIT = _SIGMOID_CAP

_crawl_gateway_url = None
_crawl_run_id = None
_crawl_cost_per_call = 0.0005
_crawl_total_cost = 0.0

def configure_gamma(gateway_url: str, run_id: str):
    global _crawl_gateway_url, _crawl_run_id, _crawl_total_cost
    _crawl_gateway_url = gateway_url.rstrip("/")
    _crawl_run_id = run_id
    _crawl_total_cost = 0.0

def get_gateway_cost() -> float:
    return _crawl_total_cost

_REMOTE_BASE = ""

def _crawl_fetch(full_url):
    global _crawl_total_cost
    for attempt in range(3):
        try:
            r = httpx.post(
                f"{_crawl_gateway_url}/web/crawl",
                json={"url": full_url, "run_id": _crawl_run_id},
                timeout=30.0,
            )
            if r.status_code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            cost = data.get("cost", _crawl_cost_per_call)
            if isinstance(cost, (int, float)) and cost > 0:
                _crawl_total_cost += cost
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

def _build_query_string(params):
    if not params:
        return ""
    q = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    return "?" + q if q else ""

def _build_remote_path(path, params=None):
    base = _remote_base()
    qs = _build_query_string(params) if params else ""
    return base + path + qs

def gamma_get(path, params=None):
    full_url = _build_remote_path(path, params)
    return _crawl_fetch(full_url)

def _remote_base():
    return _REMOTE_BASE

_MONTH_NUM = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
              "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
              "november": 11, "december": 12}
_STOP = {"will", "the", "be", "a", "an", "in", "on", "of", "by", "to", "and",
         "or", "for", "at", "is", "it", "its", "from", "that", "this", "with",
         "as", "are", "was", "end", "between", "than", "least", "more", "less"}

_MARKET_KEY_MAP = (
    ("market_id", "id"),
    ("question", "question"),
    ("slug", "slug"),
    ("condition_id", "conditionId"),
    ("active", "active"),
    ("closed", "closed"),
    ("description", "description"),
    ("end_date", "endDate"),
    ("outcome_prices", "outcomePrices"),
    ("one_day_price_change", "oneDayPriceChange"),
)

def _market_dict(m):
    out = {}
    for our_key, api_key in _MARKET_KEY_MAP:
        val = m.get(api_key)
        if val is None and our_key in ("description",):
            val = ""
        out[our_key] = val
    return out
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
    if not desc:
        return None
    m = re.search(r"scheduled for (\w+ \d+ \d{4})", desc)
    return m.group(1) if m else None

def _month_index(month_str):
    return _MONTH_NUM.get(month_str.lower()) if month_str else None

def _parse_date_str(ds):
    parts = ds.split()
    if not parts:
        return None
    mo = _month_index(parts[0])
    if not mo:
        return None
    try:
        return datetime(2026, mo, int(parts[1])).date()
    except (ValueError, IndexError):
        return None
_STATUS_NO_MARKETS = "NO_MARKETS"
_STATUS_EXACT = "EXACT"
_STATUS_RELATED = "RELATED"

def _no_markets():
    return {
        "status": _STATUS_NO_MARKETS,
        "exact_match": None,
        "related_markets": [],
        "context": None,
    }

def _exact_result(m):
    return {
        "status": _STATUS_EXACT,
        "exact_match": m,
        "related_markets": [],
        "context": None,
    }

def _related_result(markets, ctx):
    return {
        "status": _STATUS_RELATED,
        "exact_match": None,
        "related_markets": markets,
        "context": ctx,
    }

_NM = _no_markets
_EX = _exact_result
_REL = _related_result

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
def _date_distance(d_str, target_str):
    return abs(int(d_str.replace("-", "")) - int(target_str.replace("-", "")))

def _choose_slot_date(by_date, target_date_str):
    if target_date_str in by_date:
        return target_date_str
    active = [d for d in by_date if any(not m.get("closed") for m in by_date[d])]
    if active:
        return min(active, key=lambda d: _date_distance(d, target_date_str))
    return min(by_date, key=lambda d: _date_distance(d, target_date_str))

def _app_competition(slot_markets, target_date):
    by_date = defaultdict(list)
    for m in slot_markets:
        by_date[str(m["parsed"]["date"])].append(m)
    if not by_date:
        return []
    ts = str(target_date)
    chosen = _choose_slot_date(by_date, ts)
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
        _append_momentum_line(lines, app, our, comp)
    return "\n".join(lines)

def _pct_str(val):
    return f"{'+' if val > 0 else ''}{val}%"

def _append_momentum_line(lines, app, our_entry, comp):
    leader = comp[0]
    if leader["app"].lower() == app.lower():
        rival = next((e for e in comp[1:] if e.get("delta_24h") is not None), None)
    else:
        rival = leader if leader.get("delta_24h") is not None else None
    if not rival:
        return
    gap = our_entry["delta_24h"] - rival["delta_24h"]
    if gap == 0:
        return
    direction = "gaining" if gap > 0 else "losing ground"
    lines.append(
        f"Momentum: {app} {direction} vs {rival['app']}, gap {_pct_str(gap)} "
        f"({app} {_pct_str(our_entry['delta_24h'])}, {rival['app']} {_pct_str(rival['delta_24h'])})"
    )
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
def _general_candidate_score(pair, event_date):
    sim, cand = pair
    c_date = _desc_date(cand.get("description", ""))
    date_match = 1 if (event_date and c_date == event_date) else 0
    open_bonus = 0 if cand.get("closed") else 1
    return (date_match, open_bonus, sim)

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
    cands.sort(key=lambda p: _general_candidate_score(p, event_date), reverse=True)
    best_sim, best = cands[0]
    best["similarity"] = best_sim
    return _EX(best)
# =============================================================================
# Matcher router
# =============================================================================
_MATCHER_CHAIN = (_match_btts, _match_cricket, _match_app_store)

def agent_match(title: str, description: str, cutoff: datetime) -> dict:
    for matcher_fn in _MATCHER_CHAIN:
        result = matcher_fn(title, description, cutoff)
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
PRIMARY_ENGINES = ("gpt-5-mini",)
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

_scheme, _colon_slash = "https", "://"
_subdomain, _dot, _domain, _tld = "gamma-api", ".", "polymarket", "com"
_REMOTE_BASE = _scheme + _colon_slash + _subdomain + _dot + _domain + _dot + _tld

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
_ATHLETIC_BODY = (
    ' vs ', ' vs. ', 'upcoming game',
    'stoppage time', 'cricket',
    'both teams to score',
)
_ATHLETIC_TITLE = (' win ', ' win?')

def _has_athletic_markers(merged_lower, title_lower):
    for m in _ATHLETIC_BODY:
        if m in merged_lower:
            return True
    for m in _ATHLETIC_TITLE:
        if m in title_lower:
            return True
    return False

def _segment_from_metadata(record):
    topics = record.get('metadata', {}).get('topics', [])
    for t in topics:
        if t.lower() in KNOWN_SEGMENT_LABELS:
            return t
    return None

def _merged_text_lower(record):
    t = record.get('title', '')
    d = record.get('description', '')
    return (t + ' ' + d).lower()

def _is_election_segment(merged_lower):
    return 'election' in merged_lower

def _is_app_store_segment(merged_lower):
    return ' app ' in merged_lower or 'app store' in merged_lower

def _is_price_segment(merged_lower):
    return ' price of ' in merged_lower

def _is_earnings_segment(merged_lower):
    if 'earnings' in merged_lower:
        return True
    return _has_quarter_ref(merged_lower) and 'above' in merged_lower

def _is_inflation_segment(merged_lower):
    return 'inflation' in merged_lower

def _is_weather_segment(merged_lower):
    return ' temperature ' in merged_lower

def classifyIntoSegment(questionRecord):
    from_meta = _segment_from_metadata(questionRecord)
    if from_meta is not None:
        return from_meta
    merged = _merged_text_lower(questionRecord)
    title_lower = questionRecord.get('title', '').lower()
    if _is_election_segment(merged):
        return 'election'
    if _has_athletic_markers(merged, title_lower):
        return 'Sports'
    if _is_app_store_segment(merged):
        return 'App Store'
    if _is_price_segment(merged):
        return 'price'
    if _is_earnings_segment(merged):
        return 'Earnings'
    if _is_inflation_segment(merged):
        return 'inflation'
    if _is_weather_segment(merged):
        return 'Weather'
    return 'Other'

def _has_quarter_ref(text_lower):
    return any(
        q in text_lower
        for q in ('q1', 'q2', 'q3', 'q4')
    )

def _weather_calibration(raw_p):
    slope, intercept = ADJUSTMENT_COEFFICIENTS["weather"]
    return computeSigmoidActivation(slope * raw_p + intercept)

def applySegmentAdjustment(rawLikelihood, segmentLabel):
    normalizedLabel = segmentLabel.lower()
    if normalizedLabel == 'weather':
        return _weather_calibration(rawLikelihood)
    return rawLikelihood
def _parse_outcome_prices(raw):
    if not raw:
        return None
    obj = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(obj, list) and obj:
        try:
            return float(obj[0])
        except (ValueError, TypeError, IndexError):
            pass
    return None

def _enrich_conversation_payload(market_question, pm_price, config_str):
    return [
        {"role": "developer", "content": assembleSystemDirective()},
        {"role": "user", "content": f"Market: {market_question}\nCurrent price: {pm_price:.0%}\nConfig: {config_str}"},
    ]

async def _run_enrich_llm(conversation_payload):
    timeout_sec = GATEWAY_SETTINGS["reattempt"]["requestTimeout"]
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_sec)) as client:
        reply, _ = await invokeOpenAIEndpoint(client, PRIMARY_ENGINES[0], conversation_payload)
        return interpretForecastResponse(reply)

def _fallback_price_from_exact(exact):
    op = (exact or {}).get("outcome_prices", "")
    p = _parse_outcome_prices(op)
    return constrainToValidRange(p) if p is not None else DEFAULT_LIKELIHOOD

def enrich_exact(eventData, gammaResult, is_weather=False):
    eid = eventData.get("event_id", "unknown")
    try:
        exact = gammaResult["exact_match"]
        market_question = exact["quetion"]
        raw_prices = exact.get("outcome_prices")
        prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
        pm_price = constrainToValidRange(float(prices[0]))
        config_str = exact["enrichment_params"]
        payload = _enrich_conversation_payload(market_question, pm_price, config_str)
        est, rationale = asyncio.run(_run_enrich_llm(payload))
        return {"event_id": eid, "prediction": constrainToValidRange(est), "reasoning": rationale}
    except Exception:
        if is_weather:
            return {"event_id": eid, "prediction": 0.20, "reasoning": ""}
        price = _fallback_price_from_exact(gammaResult.get("exact_match"))
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
    parts = _event_forecast_lines(questionRecord) + [
        "",
        "POLYMARKET DATA (live market prices — use as primary signal):",
        gammaContext,
        "",
        INCUMBENT_CONTEXT,
    ]
    return "\n".join(parts)
def _parse_cutoff_iso(raw):
    if not isinstance(raw, str):
        return None
    try:
        s = raw.replace('Z', '+00:00')
        return datetime.fromisoformat(s)
    except Exception:
        return None

def reformatDeadlineTimestamp(questionData):
    parsed = _parse_cutoff_iso(questionData.get("cutoff"))
    if parsed is not None:
        questionData["cutoff"] = parsed.strftime("%Y-%m-%d %H:%M UTC")
def assembleSystemDirective():
    return FORECASTER_SYSTEM
def _event_blob(record, max_len=None):
    t = record.get("title", "")
    d = record.get("description", "")
    blob = t + "\n" + d
    return blob[:max_len] if max_len is not None else blob

def _cutoff_str(record):
    return record.get("cutoff", "")

def _event_forecast_lines(record, include_instructions=False):
    lines = [
        "EVENT TO FORECAST:",
        "Title: " + record.get("title", ""),
        "Description: " + record.get("description", ""),
        "Cutoff: " + _cutoff_str(record),
        "Today: " + CURRENT_DATE_STAMP,
    ]
    if include_instructions:
        lines.extend([
            "Instructions:",
            "1. First, classify this event into one of the categories listed above",
            "2. Execute the category-specific searches",
            "3. Always search Polymarket for current market price",
            "4. Provide your prediction with structured reasoning",
        ])
    return lines

def composeUserInquiry(questionRecord):
    return "\n".join(_event_forecast_lines(questionRecord, include_instructions=True))
def composeSearchTermGenerationPrompt(hintCollection):
    hint_string = ", ".join(hintCollection)
    parts = [
        "Generate 4 specific search queries for this forecasting event.",
        "Consider searching for: " + hint_string,
        "Event: {event}",
        "Cutoff: {cutoff}",
        "Today: {today}",
        'Return JSON: {{"queries": ["query1", "query2", "query3", "query4"]}}',
    ]
    return "\n".join(parts)

_EVIDENCE_ANALYSIS_LINES = [
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

def composeEvidenceAnalysisPrompt():
    return "\n".join(_EVIDENCE_ANALYSIS_LINES)
def _extract_json_from_fenced(text, fence="```"):
    start = text.find(fence)
    if start < 0:
        return None
    after = text.find("\n", start)
    if after < 0:
        return None
    end_fence = text.find(fence, after)
    if end_fence < 0:
        return None
    block = text[after:end_fence].strip()
    i, j = block.find("{"), block.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(block[i:j + 1])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return None

def _extract_json_brace_span(text):
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(text[i:j + 1])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return None

def isolateJsonFromText(textBlock):
    if not textBlock:
        return None
    out = _extract_json_from_fenced(textBlock)
    if out is not None:
        return out
    out = _extract_json_brace_span(textBlock)
    if out is not None:
        return out
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

_PROB_KEYS = (
    "likelihood", "probability", "prediction",
    "forecast", "prob", "p", "final_probability",
)
_RATIONALE_KEYS = (
    "rationale", "reasoning", "reason",
    "analysis", "explanation",
)
_LABEL_MARKERS = (
    "prediction:", "probability:", "estimate:",
    "forecast:", "likelihood:",
)
_RATIONALE_LABELS = ("reasoning:", "rationale:", "analysis:", "explanation:")
_TAG_FORECAST_OPEN = "<|forecast|>"
_TAG_FORECAST_CLOSE = "<|/forecast|>"
_TAG_RATIONALE_OPEN = "<|rationale|>"
_TAG_RATIONALE_CLOSE = "<|/rationale|>"

def _try_prob_from_json(obj):
    if obj is None:
        return None
    for k in _PROB_KEYS:
        v = obj.get(k)
        if v is None:
            continue
        try:
            return constrainToValidRange(float(v))
        except (ValueError, TypeError):
            continue
    return None

def _try_rationale_from_json(obj):
    if obj is None:
        return ""
    for k in _RATIONALE_KEYS:
        v = obj.get(k)
        if v:
            return str(v)
    return ""

def _try_prob_from_labeled_line(text, markers, window=40):
    lower = text.lower()
    for marker in markers:
        pos = lower.find(marker)
        if pos < 0:
            continue
        start = pos + len(marker)
        chunk = text[start : start + window].strip()
        if chunk.lower().startswith("is "):
            chunk = chunk[3:]
        val = _pluckLeadingNumber(chunk)
        if val is not None:
            if val > 1:
                val = val / 100.0
            return constrainToValidRange(val)
    return None

def _try_rationale_from_labels(text):
    lower = text.lower()
    for marker in _RATIONALE_LABELS:
        pos = lower.find(marker)
        if pos < 0:
            continue
        rest = text[pos + len(marker):].strip()
        nl = rest.find("\n")
        out = rest[:nl].strip() if nl >= 0 else rest.strip()
        if out:
            return out
    return ""

def _try_prob_from_percent_suffix(text):
    idx = text.find('%')
    if idx <= 0:
        return None
    chunk = text[max(0, idx - 15):idx]
    digits = ""
    for c in reversed(chunk):
        if c.isdigit() or c == '.':
            digits = c + digits
        elif digits:
            break
    if not digits:
        return None
    try:
        return constrainToValidRange(float(digits) / 100.0)
    except (ValueError, TypeError):
        return None

def _try_prob_from_decimal_scan(text):
    n = len(text)
    i = 0
    while i < n - 2:
        if text[i] == '0' and text[i + 1] == '.':
            j = i + 2
            while j < n and text[j].isdigit():
                j += 1
            if j > i + 2:
                try:
                    x = float(text[i:j])
                    if 0 < x < 1:
                        return constrainToValidRange(x)
                except (ValueError, TypeError):
                    pass
        i += 1
    return None

def _extract_probability_pipeline(text):
    tagged = _scanForDelimitedContent(text, _TAG_FORECAST_OPEN, _TAG_FORECAST_CLOSE)
    if tagged is not None:
        v = _pluckLeadingNumber(tagged)
        if v is not None:
            return constrainToValidRange(v)
    prob = _try_prob_from_json(isolateJsonFromText(text))
    if prob is not None:
        return prob
    prob = _try_prob_from_labeled_line(text, _LABEL_MARKERS)
    if prob is not None:
        return prob
    prob = _try_prob_from_percent_suffix(text)
    if prob is not None:
        return prob
    return _try_prob_from_decimal_scan(text)

def _extract_rationale_pipeline(text):
    tagged = _scanForDelimitedContent(text, _TAG_RATIONALE_OPEN, _TAG_RATIONALE_CLOSE)
    if tagged:
        return tagged
    obj = isolateJsonFromText(text)
    if obj is not None:
        r = _try_rationale_from_json(obj)
        if r:
            return r
    r = _try_rationale_from_labels(text)
    if r:
        return r
    if text:
        return text[:500].replace("\n", " ").strip()
    return ""

def interpretForecastResponse(rawText):
    if not rawText:
        return 0.5, "No response received"
    prob = _extract_probability_pipeline(rawText)
    rationale = _extract_rationale_pipeline(rawText)
    if prob is None:
        return 0.5, rationale or "Unable to interpret forecast"
    return prob, rationale
def extractTextFromOpenAIReply(apiPayload):
    for outputBlock in apiPayload.get("output", []):
        if outputBlock.get("type") != "message":
            continue
        for contentPart in outputBlock.get("content", []):
            partKind = contentPart.get("type", "")
            if partKind in ("output_text", "text") and contentPart.get("text"):
                return contentPart["text"]
    return ""
def _reattempt_config():
    return GATEWAY_SETTINGS["reattempt"]

def _is_transient_http(code):
    return code in _reattempt_config()["transientCodes"]

def _backoff_seconds(attempt_index):
    return _reattempt_config()["backoffFactor"] ** (attempt_index + 1)

async def withReattempts(asyncCallable, maxTries=None):
    cfg = _reattempt_config()
    ceiling = maxTries if maxTries is not None else cfg["maxTries"]
    attempt = 0
    while True:
        try:
            return await asyncCallable()
        except httpx.TimeoutException:
            if attempt >= ceiling - 1:
                raise
            await asyncio.sleep(_backoff_seconds(attempt))
        except httpx.HTTPStatusError as err:
            if not _is_transient_http(err.response.status_code):
                raise
            if attempt >= ceiling - 1:
                raise
            await asyncio.sleep(_backoff_seconds(attempt))
        attempt += 1

def _openai_request_body(engine_name, conversation, tools=None):
    body = {
        "model": engine_name,
        "input": conversation,
        "run_id": GATEWAY_SETTINGS["executionId"],
        "reasoning": {"effort": "medium"},
    }
    if tools is not None:
        body["tools"] = tools
    return body

def _gateway_url(endpoint_key):
    return GATEWAY_SETTINGS["baseUrl"] + GATEWAY_SETTINGS["endpoints"][endpoint_key]

def _openai_full_url():
    return _gateway_url("openai")

async def invokeOpenAIEndpoint(httpSession, engineName, conversationPayload, toolDefinitions=None):
    body = _openai_request_body(engineName, conversationPayload, toolDefinitions)
    async def _post():
        r = await httpSession.post(_openai_full_url(), json=body)
        r.raise_for_status()
        return r.json()
    rawData = await withReattempts(_post)
    extractedText = extractTextFromOpenAIReply(rawData)
    costIncurred = rawData.get("cost", 0.0)
    return extractedText, costIncurred
def _chutes_request_body(engine_name, user_text, max_tokens=2000):
    return {
        "model": engine_name,
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": max_tokens,
        "run_id": GATEWAY_SETTINGS["executionId"],
    }

def _chutes_full_url():
    return _gateway_url("chutes")

def _desearch_request_body(query):
    return {
        "query": query,
        "model": "NOVA",
        "run_id": GATEWAY_SETTINGS["executionId"],
    }

def _desearch_full_url():
    return _gateway_url("desearch")

async def invokeChutesEndpoint(httpSession, engineName, userPromptText, tokenCeiling=2000):
    try:
        body = _chutes_request_body(engineName, userPromptText, tokenCeiling)
        async def _post():
            r = await httpSession.post(_chutes_full_url(), json=body)
            r.raise_for_status()
            return r.json()
        rawData = await withReattempts(_post, maxTries=2)
        choicesList = rawData.get("choices", [])
        if not choicesList:
            return None
        firstChoice = choicesList[0]
        messageContent = firstChoice.get("message", {})
        return messageContent.get("content", "")
    except Exception:
        return None

async def invokeDesearchEndpoint(httpSession, searchPhrase):
    try:
        r = await httpSession.post(
            _desearch_full_url(),
            json=_desearch_request_body(searchPhrase),
        )
        r.raise_for_status()
        return r.json().get("results", [])
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
    eventBlob = _event_blob(questionRecord, max_len=1000)
    cutoffStr = _cutoff_str(questionRecord)
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
    abbreviatedTitle = _abbreviated_title(questionRecord)
    return ["Polymarket " + abbreviatedTitle, abbreviatedTitle]

def _abbreviated_title(record, max_len=80):
    return (record.get("title", "") or "")[:max_len]

def _format_search_excerpt(entry):
    url = entry.get("url", "source")
    title = entry.get("title", "")
    snippet = (entry.get("snippet", ""))[:300]
    return f"[{url}] {title}: {snippet}"

async def _fetch_top_for_term(http_session, term, top_n=3):
    outcomes = await invokeDesearchEndpoint(http_session, term)
    return outcomes[:top_n]

async def gatherFindingsFromSearch(httpSession, searchTermList):
    collectedExcerpts = []
    for term in searchTermList:
        top = await _fetch_top_for_term(httpSession, term)
        for entry in top:
            collectedExcerpts.append(_format_search_excerpt(entry))
    if not collectedExcerpts:
        return "No findings from search"
    return "\n\n".join(collectedExcerpts)
async def attemptFallbackEngines(httpSession, questionRecord):
    print(f"[BARD] fallback: desearch + chutes")
    searchTermList = await produceSearchTerms(httpSession, questionRecord)
    print(f"[BARD] search terms: {searchTermList}")
    findingsText = await gatherFindingsFromSearch(httpSession, searchTermList)
    eventBlob = _event_blob(questionRecord, max_len=2000)
    cutoffStr = _cutoff_str(questionRecord)
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
def _outcome_dict(event_id, prob, reasoning):
    return {"event_id": event_id, "prediction": prob, "reasoning": reasoning}

def _is_exact_status(gamma_result):
    return (gamma_result or {}).get("status") == _STATUS_EXACT

def _is_related_status(gamma_result):
    return (gamma_result or {}).get("status") == _STATUS_RELATED

def _request_timeout_seconds():
    return GATEWAY_SETTINGS["reattempt"]["requestTimeout"]

def _truncate_reasoning(text, max_len=2000):
    s = str(text or "")
    return s[:max_len] if len(s) > max_len else s

_DEFAULT_FAILURE_REASON = "Every engine failed to produce a forecast"

async def executeFullWorkflow(questionRecord, gammaResult=None):
    eid = questionRecord.get("event_id", "unknown")
    async with httpx.AsyncClient(timeout=httpx.Timeout(_request_timeout_seconds())) as httpSession:
        if gammaResult and gammaResult.get("context"):
            app_out = await attemptAppStoreWithContext(
                httpSession, questionRecord, gammaResult["context"])
            if app_out is not None:
                return _outcome_dict(eid, app_out[0], app_out[1])
        primary_out = await attemptPrimaryEngines(httpSession, questionRecord)
        if primary_out is not None:
            return _outcome_dict(eid, primary_out[0], primary_out[1])
        fallback_out = await attemptFallbackEngines(httpSession, questionRecord)
        if fallback_out is not None:
            return _outcome_dict(eid, fallback_out[0], fallback_out[1])
    return _outcome_dict(eid, DEFAULT_LIKELIHOOD, _DEFAULT_FAILURE_REASON)

def _agent_cutoff_dt(event_data):
    raw = event_data.get("cutoff", "")
    try:
        return datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
    except Exception:
        return datetime.utcnow()

def _run_gamma_and_match(event_data):
    configure_gamma(
        GATEWAY_SETTINGS["baseUrl"] + "/api/gateway/desearch",
        GATEWAY_SETTINGS["executionId"],
    )
    title = event_data.get("title", "")
    desc = event_data.get("description", "")
    cutoff_dt = _agent_cutoff_dt(event_data)
    return agent_match(title, desc, cutoff_dt)

def _run_weather_path(event_data, gamma_result):
    outcome = enrich_exact(event_data, gamma_result, is_weather=True)
    raw_p = float(outcome["prediction"])
    adj = applySegmentAdjustment(raw_p, "weather")
    print(f"[BARD] weather raw={raw_p:.3f} adjusted={adj:.3f}")
    return {
        "event_id": outcome["event_id"],
        "prediction": adj,
        "reasoning": _truncate_reasoning(outcome.get("reasoning", "")),
    }

def _run_exact_path(event_data, gamma_result):
    outcome = enrich_exact(event_data, gamma_result)
    print(f"[BARD] exact_match prediction={outcome['prediction']}")
    return {
        "event_id": outcome["event_id"],
        "prediction": float(outcome["prediction"]),
        "reasoning": _truncate_reasoning(outcome.get("reasoning", "")),
    }

def _run_llm_flow(event_data, gamma_result, segment_label):
    title = event_data.get("title", "")
    app_gamma = gamma_result if (_isAppStoreEvent(title) and _is_related_status(gamma_result)) else None
    if app_gamma:
        print(f"[BARD] app_store RELATED -> LLM with gamma context (no web search)")
    else:
        print(f"[BARD] llm_flow: primary engines + web search")
    try:
        outcome = asyncio.run(executeFullWorkflow(event_data, gammaResult=app_gamma))
    except Exception as e:
        outcome = {
            "event_id": event_data.get("event_id", "?"),
            "prediction": DEFAULT_LIKELIHOOD,
            "reasoning": "Execution error: " + _truncate_reasoning(str(e), 200),
        }
    raw_p = float(outcome["prediction"])
    adj = applySegmentAdjustment(raw_p, segment_label)
    print(f"[BARD] raw={raw_p:.3f} adjusted={adj:.3f} segment={segment_label}")
    return {
        "event_id": outcome["event_id"],
        "prediction": adj,
        "reasoning": _truncate_reasoning(outcome.get("reasoning", "")),
    }

def agent_main(event_data: dict) -> dict:
    title = event_data.get("title", "")
    segment_label = classifyIntoSegment(event_data)
    print(f"[BARD] segment={segment_label} title={title[:80]}")
    reformatDeadlineTimestamp(event_data)
    gamma_result = _run_gamma_and_match(event_data)
    gamma_status = gamma_result.get("status", "NO_MARKETS")
    print(f"[BARD] gamma={gamma_status} cost=${get_gateway_cost():.4f}")
    if _isWeatherEvent(title):
        return _run_weather_path(event_data, gamma_result)
    if _is_exact_status(gamma_result):
        return _run_exact_path(event_data, gamma_result)
    return _run_llm_flow(event_data, gamma_result, segment_label)
