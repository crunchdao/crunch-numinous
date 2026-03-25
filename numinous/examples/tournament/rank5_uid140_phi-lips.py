"""
Phi-Lips: The one, the only, the best
"""
import asyncio
import json
import sys
import time

_GLOBAL_TIMEOUT = 230
_FALLBACK_PREDICTION = 0.35

_BASE_RATES = {
    "weather": 0.20,
    "earnings": 0.55,
    "box_office": 0.03,
    "geo_strikes": 0.55,
    "geo_other": 0.19,
    "netflix": 0.11,
    "appstore": 0.05,
}

_costs = {"openrouter": 0.0, "desearch": 0.0}

def _track_cost(service, amount):
    _costs[service] = _costs.get(service, 0.0) + amount

import re

_CRYPTO_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "solana", "sol",
    "cardano", "ada", "ripple", "xrp", "polkadot", "dot",
    "dogecoin", "doge", "chainlink", "link", "polygon", "matic",
    "litecoin", "ltc", "avalanche", "avax", "shiba inu", "shiba",
    "shib", "tron", "trx",
]
# Pre-sort by length (longest first) so "shiba inu" matches before "shiba"
_CRYPTO_KEYWORDS.sort(key=len, reverse=True)
_CRYPTO_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(kw) for kw in _CRYPTO_KEYWORDS) + r')\b',
    re.IGNORECASE,
)

_SPORTS_RE = re.compile(
    r'win on \d{4}-\d{2}-\d{2}'
    r'|end in a draw'
    r'|both teams to score'
    r'|leading at halftime'
    r'|draw at halftime'
    r'|exact score:'
    r'|win by KO|win by TKO|win by submission|win by decision'
    r'|finish in the top \d+'
    r'|^ODI |^T20 '
    r'|win the \d{4}[-–]\d{4} .*(championship|season)'
    r'|\b(?:FC|AFC|SC|CF|SK|BC)\b.*(?:win|draw|score)',
    re.IGNORECASE,
)


def simple_classifier_classify(event: dict, context: dict) -> dict:
    title = event.get("title", "")
    description = event.get("description", "").lower()
    merged = f"{title.lower()} {description}"
    topics = event.get("metadata", {}).get("topics", [])

    topics_lower = {t.lower() for t in topics}
    if topics:
        print(f"[classify] topics={topics}")

    if "app store" in merged or topics_lower & {"app store", "app_store"}:
        return {"event_type": "appstore"}

    if any(kw in merged for kw in ("temperature", "weather", "°f", "°c")):
        return {"event_type": "weather"}

    if "strikes" in topics_lower or re.search(
        r"military strike|missile strike|drone (?:attack|strike)|air\s?strike",
        merged,
    ):
        return {"event_type": "geo_strikes"}

    if _CRYPTO_RE.search(merged) or "crypto" in topics_lower:
        return {"event_type": "crypto"}

    if _SPORTS_RE.search(title) or "sports" in topics_lower:
        return {"event_type": "sports"}

    if "earnings" in topics_lower:
        return {"event_type": "earnings"}

    return {"event_type": "general"}


import hashlib, json, os, re, time
from datetime import date, datetime
from pathlib import Path

# =============================================================================
# Gamma API client
# =============================================================================

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_GATEWAY_URL = os.environ.get("SANDBOX_PROXY_URL", "").rstrip("/") or None
_GATEWAY_RUN_ID = os.environ.get("RUN_ID")
_gamma_desearch_cost = 0.0

def _gamma_get_gateway(url):
    global _gamma_desearch_cost
    import httpx
    for attempt in range(3):
        try:
            resp = httpx.post(f"{_GATEWAY_URL}/api/gateway/desearch/web/crawl",
                              json={"url": url, "run_id": _GATEWAY_RUN_ID}, timeout=30.0)
            if resp.status_code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(1.0 * (attempt + 1)); continue
            resp.raise_for_status()
            data = resp.json()
            _gamma_desearch_cost += data.get("cost", 0.0)
            content = data.get("content", "")
            if not content:
                return None
            return json.loads(content) if isinstance(content, str) else content
        except Exception as e:
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1)); continue
            return None
    return None

_CACHE_DIR = None

def _get_cache_dir():
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = Path("gamma_cache")
        _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR

def _gamma_get_direct(url):
    import httpx
    cache_dir = _get_cache_dir()
    cpath = cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
    if cpath.exists():
        return json.loads(cpath.read_text())
    try:
        resp = httpx.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None
    cpath.write_text(json.dumps(data))
    time.sleep(0.3)
    return data

def gamma_get(path, params=None):
    url = f"{_GAMMA_BASE}{path}"
    if params:
        url += "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    if _GATEWAY_URL:
        return _gamma_get_gateway(url)
    return _gamma_get_direct(url)

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

def _yes_price(m):
    try:
        op = m.get("outcome_prices", "")
        pl = json.loads(op) if isinstance(op, str) else op
        if isinstance(pl, list) and pl:
            v = float(pl[0])
            return max(0.01, min(0.99, v))
    except (json.JSONDecodeError, ValueError, IndexError, TypeError):
        pass
    return None

_NM = lambda: {"status": "NO_MARKETS", "market": None, "prediction": None}
def _EX(m):
    p = _yes_price(m)
    return {"status": "EXACT", "market": m, "prediction": p} if p is not None else _NM()

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
# Earnings matcher 
# =============================================================================

_BEAT_RE = re.compile(r"Will (.+?) \(([A-Z.]+)\) beat quarterly earnings\?")
_SLUG_RE = re.compile(
    r"([a-z.]+)-quarterly-earnings-(gaap|nongaap)-eps-"
    r"(\d{2})-(\d{2})-(\d{4})-(neg)?([\d]+(?:pt[\d]+)?)")

def _parse_slug(slug):
    m = _SLUG_RE.search(slug)
    if not m: return None
    ticker, gaap, mo, day, yr, neg, est_raw = m.groups()
    est = float(est_raw.replace("pt", "."))
    if neg: est = -est
    return {"ticker": ticker.upper(), "earnings_date": date(int(yr), int(mo), int(day)), "estimate": est}

def _match_earnings(title, desc, cutoff):
    m = _BEAT_RE.search(title)
    if not m: return None
    company, ticker = m.group(1).strip(), m.group(2)

    seen, markets = set(), []
    for q in [f"{ticker} quarterly earnings", f"{ticker} earnings"]:
        data = gamma_get("/public-search", {"q": q})
        if not data or "events" not in data: continue
        for ev in data["events"]:
            for mk in ev.get("markets", []):
                mid = mk.get("id")
                if mid in seen: continue
                ps = _parse_slug(mk.get("slug", ""))
                if not ps or ps["ticker"] != ticker: continue
                seen.add(mid)
                md = _market_dict(mk)
                md["parsed_slug"] = ps
                markets.append(md)

    if not markets: return _NM()
    cutoff_date = cutoff.date() if hasattr(cutoff, "date") else cutoff
    best = min(markets, key=lambda m: abs((m["parsed_slug"]["earnings_date"] - cutoff_date).days))
    dist = abs((best["parsed_slug"]["earnings_date"] - cutoff_date).days)
    if dist > 14: return _NM()
    if dist <= 7: return _EX(best)
    return _NM()

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

def _match_app_store(title, desc, cutoff):
    p = _parse_app(title)
    if not p:
        return None if "app store" not in title.lower() else _NM()

    app_lower, type_lower = p["app"].lower(), p["type"].lower()
    queries = [f"{p['app']} {p['type']} App Store",
               f"{p['app']} {p['type']} App Store {p['date'].strftime('%B')}"]
    seen = set()
    for q in queries:
        for ev in (gamma_get("/public-search", {"q": q}) or {}).get("events", []):
            for m in ev.get("markets", []):
                mid = m.get("id")
                if mid in seen: continue
                seen.add(mid)
                mp = _parse_app(m.get("question", ""))
                if not mp: continue
                if mp["app"].lower() != app_lower or mp["type"].lower() != type_lower: continue
                if mp["date"] == p["date"] and mp["rank"] == p["rank"]:
                    return _EX(_market_dict(m))
    return _NM()

# =============================================================================
# General matcher 
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
# Router
# =============================================================================

def gamma_matcher_matching(event: dict, context: dict) -> dict:
    """Match a prediction event to Polymarket.

    Returns {"status": "EXACT"|"NO_MARKETS", "market": dict|None, "prediction": float|None, "cost": float}.
    """
    global _gamma_desearch_cost
    _gamma_desearch_cost = 0.0
    title = event.get("title", "")
    description = event.get("description", "")
    cutoff_raw = event.get("cutoff", "")
    cutoff = datetime.fromisoformat(cutoff_raw) if cutoff_raw else None
    for fn in (_match_btts, _match_cricket, _match_earnings, _match_app_store):
        result = fn(title, description, cutoff)
        if result is not None:
            result["cost"] = _gamma_desearch_cost
            return result
    result = _match_general(title, description, cutoff)
    result["cost"] = _gamma_desearch_cost
    return result


import json
import os
import re
import time

import httpx


_appstore_desearch_cost = 0.0

class _AppStore:
    PROXY_URL = os.environ.get("SANDBOX_PROXY_URL", "").rstrip("/") or None
    RUN_ID = os.environ.get("RUN_ID")
    FEEDS = {
        "free": "https://rss.marketingtools.apple.com/api/v2/us/apps/top-free/100/apps.json",
        "paid": "https://rss.marketingtools.apple.com/api/v2/us/apps/top-paid/100/apps.json",
    }

    INCUMBENT_TABLE = """\
### Incumbent table (updated Mar 9, 2026)
Historical win rates — how often each app actually held this rank at resolution.

Free Apps:
| Rank | Incumbent         | Win% (30d) | Win% (all) | N   |
|------|-------------------|------------|------------|-----|
| #1   | ChatGPT           | 83%        | 77%        | 62  |
| #1   | Claude*           | 90% (7d)   | 91%        | 11  |
| #2   | Freecash          | 28%        | 40%        | 45  |
| #2   | ChatGPT           | 14%        | 22%        | 54  |
| #2   | Google Gemini     | 20%        | 12%        | 41  |
| #3   | Google Gemini     | 40%        | 36%        | 39  |
| #3   | Threads           | 16%        | 33%        | 46  |
| #4   | Threads           | 33%        | 38%        | 21  |
| #4   | Google Gemini     | 25%        | 22%        | 23  |
| #4   | Freecash          | 29%        | 21%        | 19  |

Paid Apps:
| Rank | Incumbent             | Win% (30d) | Win% (all) | N   |
|------|-----------------------|------------|------------|-----|
| #1   | Shadowrocket          | 88%        | 94%        | 63  |
| #2   | HotSchedules          | 91%        | 96%        | 52  |
| #3   | AnkiMobile Flashcards | 78%        | 75%        | 52  |
| #3   | Procreate Pocket      | 13%        | 12%        | 50  |
| #4   | Procreate Pocket      | 88%        | 78%        | 23  |
| #4   | AnkiMobile Flashcards | 13%        | 17%        | 23  |

*Claude hit Free #1 in early Mar 2026 due to major OpenAI PR crisis.
 7d data: Claude 90% (9/10), ChatGPT dropped to 11% (1/9).
 This may be temporary — treat live chart data as more reliable during volatile periods.

Base rates (any app asked about at this rank):
Free: #1=20% | #2=15% | #3=14% | #4=17%
Paid: #1=20% | #2=23% | #3=20% | #4=19%
"""

    FALLBACK_SNAPSHOT = """\
### Free Apps - App Store Top Charts (latest available snapshot, Mar 9, 2026)
| Rank | App |
|------|-----|
| #1 | ChatGPT |
| #2 | Claude by Anthropic |
| #3 | Google Gemini |
| #4 | Paramount+ |
| #5 | Temu |
"""

    @staticmethod
    def clock():
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    @classmethod
    def parse_query(cls, title):
        hit = re.search(
            r'(?:Will\s+)?["\']?(.+?)["\']?\s+be\s+(?:the\s+)?#(\d+)\s+(Free|Paid)\s+app',
            title, re.IGNORECASE,
        )
        if not hit:
            return None
        return {
            "app_name": hit.group(1).strip(),
            "target_rank": int(hit.group(2)),
            "category": hit.group(3).lower(),
        }

    @staticmethod
    def normalize_title(name):
        if not isinstance(name, str):
            return ""
        cleaned = "".join(ch if ch.isalnum() else " " for ch in name.lower().strip())
        return " ".join(cleaned.split())

    @classmethod
    def process_payload(cls, raw, query):
        try:
            # Support both old iTunes RSS and new Marketing Tools API
            entries = raw.get("feed", {}).get("results", [])
            if not entries:
                entries = raw.get("feed", {}).get("entry", [])
            if not entries:
                return None

            target_norm = cls.normalize_title(query["app_name"])
            if not target_norm:
                return None

            rank_list = []
            matches = []
            for pos, entry in enumerate(entries, 1):
                # New API uses "name", old uses "im:name" -> "label"
                label = entry.get("name", "") or entry.get("im:name", {}).get("label", "")
                rank_list.append(label)
                if cls.normalize_title(label) == target_norm:
                    matches.append((pos, label))

            if len(matches) != 1:
                if len(matches) > 1:
                    print(f"[{cls.clock()}] AppChart: ambiguous matches for "
                          f"\"{query['app_name']}\" ({len(matches)})")
                return None

            current_pos = matches[0][0]
            target_rank = query["target_rank"]
            occupant = rank_list[target_rank - 1] if len(rank_list) >= target_rank else None

            neighborhood = []
            for i in range(max(1, target_rank - 3), min(len(rank_list) + 1, target_rank + 4)):
                neighborhood.append((i, rank_list[i - 1]))

            return {
                "app_name": query["app_name"],
                "target_rank": target_rank,
                "category": query["category"],
                "current_rank": current_pos,
                "distance": current_pos - target_rank,
                "app_at_target": occupant,
                "nearby_apps": neighborhood,
                "in_top_100": True,
            }
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            print(f"[{cls.clock()}] AppChart: parse error: {exc}")
            return None

    @classmethod
    def format_live_block(cls, chart_data):
        parts = [
            "Live data from Apple App Store iPhone charts.",
            f"Question: Will \"{chart_data['app_name']}\" be "
            f"#{chart_data['target_rank']} {chart_data['category'].title()} App?",
            "",
        ]

        if chart_data["in_top_100"]:
            parts.append("Current Status:")
            parts.append(f"  \"{chart_data['app_name']}\" is currently #{chart_data['current_rank']}")
            if chart_data["distance"] == 0:
                parts.append("  ALREADY at target rank!")
            elif chart_data["distance"] > 0:
                parts.append(f"  Needs to move UP {chart_data['distance']} position(s)")
            else:
                parts.append(f"  Needs to move DOWN {abs(chart_data['distance'])} position(s)")
        else:
            parts.append("Current Status:")
            parts.append(f"  \"{chart_data['app_name']}\" is NOT in top 100 "
                         f"- very unlikely to reach #{chart_data['target_rank']}")

        parts.append("")
        parts.append(f"Apps near #{chart_data['target_rank']}:")
        for pos, label in chart_data["nearby_apps"]:
            tag = " <-- target" if pos == chart_data["target_rank"] else ""
            cur = " (TARGET APP)" if chart_data["current_rank"] == pos else ""
            parts.append(f"  #{pos}: {label}{tag}{cur}")

        return "\n".join(parts)

    @classmethod
    def build_output(cls, query, chart_data=None):
        parts = ["<|app_rankings|>"]

        if chart_data:
            parts.append(cls.format_live_block(chart_data))
            parts.append("")

        parts.append(cls.INCUMBENT_TABLE)

        if not chart_data:
            parts.append(cls.FALLBACK_SNAPSHOT)

        parts.append("<|/app_rankings|>")
        return "\n".join(parts)

    @classmethod
    async def fetch_feed(cls, http, feed_url):
        global _appstore_desearch_cost
        import asyncio
        hour_stamp = int(time.time() // 3600)
        cached_url = f"{feed_url}?ts={hour_stamp}"

        for attempt in range(3):
            try:
                if cls.PROXY_URL:
                    resp = await http.post(
                        f"{cls.PROXY_URL}/api/gateway/desearch/web/crawl",
                        json={"url": cached_url, "run_id": cls.RUN_ID},
                        timeout=20.0,
                    )
                else:
                    resp = await http.get(cached_url, timeout=20.0)

                if resp.status_code in (429, 500, 502, 503) and attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue

                if resp.status_code != 200:
                    print(f"[{cls.clock()}] AppChart: HTTP {resp.status_code} (attempt {attempt + 1}/3)")
                    if attempt < 2:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    return None

                body = resp.json()
                if cls.PROXY_URL:
                    _appstore_desearch_cost += body.get("cost", 0.0)
                    content = body.get("content", "")
                    if isinstance(content, str):
                        return json.loads(content)
                    return content
                return body

            except httpx.ConnectError as exc:
                print(f"[{cls.clock()}] AppChart: ConnectError (attempt {attempt + 1}/3) - {exc}")
            except httpx.TimeoutException:
                print(f"[{cls.clock()}] AppChart: Timeout (attempt {attempt + 1}/3)")
            except Exception as exc:
                print(f"[{cls.clock()}] AppChart: {type(exc).__name__}: {exc}")

            if attempt < 2:
                await asyncio.sleep(1.0 * (attempt + 1))

        return None


async def appstore_crawler_crawler(event: dict, context: dict) -> dict | None:
    event_type = context.get("classify", {}).get("event_type")
    if event_type != "appstore":
        return None

    title = event.get("title", "")
    query = _AppStore.parse_query(title)
    chart_data = None

    if not query:
        print(f"[{_AppStore.clock()}] AppChart: could not parse query from title: {title!r}")

    if query:
        async with httpx.AsyncClient() as http:
            feed_url = _AppStore.FEEDS.get(query["category"], _AppStore.FEEDS["free"])
            raw = await _AppStore.fetch_feed(http, feed_url)

        if raw:
            chart_data = _AppStore.process_payload(raw, query)
            if chart_data:
                print(f"[{_AppStore.clock()}] AppChart: {query['app_name']} is "
                      f"#{chart_data['current_rank']} (target #{query['target_rank']})")
            else:
                print(f"[{_AppStore.clock()}] AppChart: crawl succeeded but app not found in chart")
        else:
            print(f"[{_AppStore.clock()}] AppChart: crawl failed")

    return {"data": _AppStore.build_output(query, chart_data), "type": "appstore", "cost": _appstore_desearch_cost}

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone

import httpx


_crypto_desearch_cost = 0.0

class _Crypto:
    PROXY_URL = os.environ.get("SANDBOX_PROXY_URL", "").rstrip("/") or None
    RUN_ID = os.environ.get("RUN_ID")

    ASSETS = {
        "bitcoin": ("BTC", "90"), "btc": ("BTC", "90"),
        "ethereum": ("ETH", "80"), "eth": ("ETH", "80"),
        "solana": ("SOL", "48543"), "sol": ("SOL", "48543"),
        "cardano": ("ADA", "257"), "ada": ("ADA", "257"),
        "ripple": ("XRP", "58"), "xrp": ("XRP", "58"),
        "polkadot": ("DOT", "45219"), "dot": ("DOT", "45219"),
        "dogecoin": ("DOGE", "2"), "doge": ("DOGE", "2"),
        "chainlink": ("LINK", "45899"), "link": ("LINK", "45899"),
        "polygon": ("MATIC", "35793"), "matic": ("MATIC", "35793"),
        "litecoin": ("LTC", "1"), "ltc": ("LTC", "1"),
        "avalanche": ("AVAX", "44883"), "avax": ("AVAX", "44883"),
        "shiba inu": ("SHIB", "45088"), "shiba": ("SHIB", "45088"), "shib": ("SHIB", "45088"),
        "tron": ("TRX", "34"), "trx": ("TRX", "34"),
    }

    LABELS = {
        "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "ADA": "Cardano",
        "XRP": "Ripple", "DOT": "Polkadot", "DOGE": "Dogecoin", "LINK": "Chainlink",
        "MATIC": "Polygon", "LTC": "Litecoin", "AVAX": "Avalanche", "SHIB": "Shiba Inu",
        "TRX": "Tron",
    }

    @staticmethod
    def clock():
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def safe_num(val):
        try:
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def safe_int(val):
        try:
            return int(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    @classmethod
    def identify_asset(cls, title):
        lowered = title.lower()
        for keyword in sorted(cls.ASSETS.keys(), key=len, reverse=True):
            if re.search(rf'\b{re.escape(keyword)}\b', lowered):
                return cls.ASSETS[keyword]
        return None

    @classmethod
    async def crawl_url(cls, http, url):
        global _crypto_desearch_cost
        hour_stamp = int(time.time() // 3600)
        joiner = "&" if "?" in url else "?"
        url = f"{url}{joiner}_h={hour_stamp}"

        for attempt in range(3):
            try:
                if cls.PROXY_URL:
                    resp = await http.post(
                        f"{cls.PROXY_URL}/api/gateway/desearch/web/crawl",
                        json={"url": url, "run_id": cls.RUN_ID},
                        timeout=20.0,
                    )
                else:
                    resp = await http.get(url, timeout=20.0)

                if resp.status_code in (429, 500, 502, 503) and attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue

                if resp.status_code != 200:
                    print(f"[{cls.clock()}] Crypto: HTTP {resp.status_code} (attempt {attempt + 1}/3)")
                    if attempt < 2:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    return None

                body = resp.json()
                if cls.PROXY_URL:
                    _crypto_desearch_cost += body.get("cost", 0.0)
                    content = body.get("content", "")
                    if isinstance(content, str):
                        return json.loads(content)
                    return content
                return body

            except httpx.ConnectError as exc:
                print(f"[{cls.clock()}] Crypto: ConnectError (attempt {attempt + 1}/3) - {exc}")
            except httpx.TimeoutException:
                print(f"[{cls.clock()}] Crypto: Timeout (attempt {attempt + 1}/3)")
            except Exception as exc:
                print(f"[{cls.clock()}] Crypto: {type(exc).__name__}: {exc}")

            if attempt < 2:
                await asyncio.sleep(1.0 * (attempt + 1))

        return None

    @classmethod
    async def fetch_data(cls, http, symbol, coinlore_id):
        record = {
            "name": cls.LABELS.get(symbol, symbol),
            "symbol": symbol,
            "price": None, "change_1h": None, "change_24h": None, "change_7d": None,
            "market_cap": None, "volume": None, "rank": None, "sources": [],
        }

        cc_url = f"https://min-api.cryptocompare.com/data/price?fsym={symbol}&tsyms=USD"
        cl_url = f"https://api.coinlore.net/api/ticker/?id={coinlore_id}"

        cc_data, cl_data = await asyncio.gather(
            cls.crawl_url(http, cc_url),
            cls.crawl_url(http, cl_url),
        )

        if cc_data and "USD" in cc_data:
            record["price"] = float(cc_data["USD"])
            record["sources"].append("CryptoCompare")

        if cl_data and isinstance(cl_data, list) and len(cl_data) > 0:
            coin = cl_data[0]
            record["sources"].append("Coinlore")
            if record["price"] is None:
                record["price"] = cls.safe_num(coin.get("price_usd"))
            record["change_1h"] = cls.safe_num(coin.get("percent_change_1h"))
            record["change_24h"] = cls.safe_num(coin.get("percent_change_24h"))
            record["change_7d"] = cls.safe_num(coin.get("percent_change_7d"))
            record["market_cap"] = cls.safe_num(coin.get("market_cap_usd"))
            record["volume"] = cls.safe_num(coin.get("volume24"))
            record["rank"] = cls.safe_int(coin.get("rank"))

        return record

    @classmethod
    def format_block(cls, record):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        parts = [
            "<|crypto_data|>",
            f"Live data from CryptoCompare and Coinlore APIs.",
            f"VERIFIED {record['name']} ({record['symbol']}) PRICE (as of {today}):",
        ]

        if record["price"] is not None:
            if record["price"] < 0.01:
                parts.append(f"  Current Price: ${record['price']:.8f}")
            else:
                parts.append(f"  Current Price: ${record['price']:,.2f}")

        deltas = []
        if record["change_1h"] is not None:
            deltas.append(f"1h: {record['change_1h']:+.2f}%")
        if record["change_24h"] is not None:
            deltas.append(f"24h: {record['change_24h']:+.2f}%")
        if record["change_7d"] is not None:
            deltas.append(f"7d: {record['change_7d']:+.2f}%")
        if deltas:
            parts.append(f"  Price Changes: {', '.join(deltas)}")

        if record["market_cap"] is not None:
            parts.append(f"  Market Cap: ${record['market_cap']:,.0f}")
        if record["volume"] is not None:
            parts.append(f"  24h Volume: ${record['volume']:,.0f}")
        if record["rank"] is not None:
            parts.append(f"  Rank: #{record['rank']}")

        parts.append("<|/crypto_data|>")
        return "\n".join(parts)


async def crypto_crawler_crawler(event: dict, context: dict) -> dict | None:
    event_type = context.get("classify", {}).get("event_type")
    if event_type != "crypto":
        return None

    asset = _Crypto.identify_asset(event.get("title", ""))
    if not asset:
        return None

    symbol, coinlore_id = asset

    async with httpx.AsyncClient() as http:
        record = await _Crypto.fetch_data(http, symbol, coinlore_id)

    if record["sources"]:
        print(f"[{_Crypto.clock()}] Crypto: {symbol} ${record['price']:,.2f} "
              f"(sources: {', '.join(record['sources'])})")
        return {"data": _Crypto.format_block(record), "type": "crypto", "cost": _crypto_desearch_cost}

    print(f"[{_Crypto.clock()}] Crypto: both sources failed for {symbol}")
    return None

import os
import re
from datetime import datetime, timezone

import httpx


_indicia_cost = 0.0


class _Indicia:
    PROXY_URL = os.environ.get("SANDBOX_PROXY_URL", "").rstrip("/") or None
    RUN_ID = os.environ.get("RUN_ID")
    LIVEUAMAP_PATH = "/api/gateway/numinous-indicia/liveuamap"
    SIGNAL_LIMIT = 50

    @staticmethod
    def clock():
        return datetime.now().strftime("%H:%M:%S")

    @classmethod
    def extract_location(cls, title):
        """Extract city/region name from strike event title."""
        # "Will Kharkiv experience any military strike..."
        m = re.match(r"Will\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+experience", title)
        if m:
            return m.group(1)

        # "strike in Kharkiv" / "attack on Donetsk" / "strike in Unknown, Ukraine"
        m = re.search(r"(?:in|on|at|near)\s+([A-Z][a-zA-Z]+(?:[\s,]+[A-Z][a-zA-Z]+)*)", title)
        if m:
            loc = re.sub(r",?\s*Ukraine$", "", m.group(1)).strip()
            if loc.lower() in ("unknown", "the"):
                return None
            return loc

        return None

    @classmethod
    async def fetch_signals(cls, http, region="ukraine"):
        """Fetch signals from liveuamap via gateway."""
        global _indicia_cost
        if not cls.PROXY_URL:
            print(f"[{cls.clock()}] Indicia: no SANDBOX_PROXY_URL, skipping")
            return []

        url = f"{cls.PROXY_URL}{cls.LIVEUAMAP_PATH}"
        try:
            resp = await http.post(url, json={
                "run_id": cls.RUN_ID,
                "region": region,
                "limit": cls.SIGNAL_LIMIT,
            }, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            _indicia_cost += data.get("cost", 0.0)
            return data.get("signals", [])
        except Exception as e:
            print(f"[{cls.clock()}] Indicia: {type(e).__name__}: {e}")
            return []

    @classmethod
    def format_block(cls, signals, location, title):
        """Format signals into LLM-friendly context block."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Split into location-specific vs broader strike signals
        location_signals = []
        other_strikes = []

        for s in signals:
            sig_lower = s.get("signal", "").lower()
            cat = s.get("category", "")

            if location and location.lower() in sig_lower:
                location_signals.append(s)
            elif cat in ("strike", "explosion"):
                other_strikes.append(s)

        strike_total = sum(
            1 for s in signals if s.get("category") in ("strike", "explosion")
        )

        parts = [
            "<|indicia_signals|>",
            "Live conflict monitoring data from LiveUAMap (OSINT intelligence).",
            f"Data retrieved: {today}",
            f"Question: {title}",
            "",
        ]

        if location and location_signals:
            parts.append(
                f"=== SIGNALS MENTIONING {location.upper()} "
                f"({len(location_signals)} found) ==="
            )
            for s in location_signals:
                sig_text = s.get("signal", "unknown")
                parts.append(f"  [{s.get('category', '?')}] {sig_text}")
                parts.append(
                    f"    status={s.get('fact_status', '?')}, "
                    f"confidence={s.get('confidence', '?')}, "
                    f"time={s.get('timestamp', '?')}"
                )
            parts.append("")
        elif location:
            parts.append(f"=== NO RECENT SIGNALS FOR {location.upper()} ===")
            parts.append(
                f"No recent activity reported specifically for {location}."
            )
            parts.append("")

        parts.append(
            f"=== BROADER CONFLICT ACTIVITY "
            f"({strike_total} strike/explosion signals in region) ==="
        )
        for s in other_strikes[:10]:
            parts.append(f"  [{s.get('category', '?')}] {s.get('signal', 'unknown')[:150]}")
            parts.append(f"    time={s.get('timestamp', '?')}")
        parts.append("")

        parts.append(f"Total signals in region: {len(signals)}")
        parts.append(
            "NOTE: These signals reflect real-time conflict monitoring. "
            "High frequency of strikes indicates ongoing active conflict "
            "where attacks are near-daily events."
        )
        parts.append("<|/indicia_signals|>")

        return "\n".join(parts)


async def indicia_crawler_crawler(event: dict, context: dict) -> dict | None:
    global _indicia_cost
    _indicia_cost = 0.0

    event_type = context.get("classify", {}).get("event_type")
    if event_type != "geo_strikes":
        return None

    title = event.get("title", "")
    location = _Indicia.extract_location(title)

    async with httpx.AsyncClient() as http:
        signals = await _Indicia.fetch_signals(http)

    if not signals:
        print(f"[{_Indicia.clock()}] Indicia: no signals returned")
        return None

    # Count location-specific hits
    loc_hits = 0
    if location:
        loc_lower = location.lower()
        loc_hits = sum(
            1 for s in signals if loc_lower in s.get("signal", "").lower()
        )

    print(
        f"[{_Indicia.clock()}] Indicia: {len(signals)} signals"
        f"{f', {loc_hits} mention {location}' if location else ', no specific location'}"
    )

    return {
        "data": _Indicia.format_block(signals, location, title),
        "type": "indicia",
        "cost": _indicia_cost,
    }


_RN_SKIP_TYPES = {"appstore", "weather"}


def simple_research_needed_research_needed(event: dict, context: dict) -> dict:
    matching = context.get("matching", {})
    if matching.get("status") == "EXACT":
        return {"research_needed": False, "reason": "exact gamma match"}

    event_type = context.get("classify", {}).get("event_type", "general")
    if event_type in _RN_SKIP_TYPES:
        return {"research_needed": False, "reason": f"event type: {event_type}"}

    return {"research_needed": True, "reason": "no sufficient static data"}


import asyncio
import os
import re
import time

import httpx


class GrokResearcher:
    MODEL = "x-ai/grok-4.1-fast"
    MAX_TOKENS = 4096
    TIMEOUT = 180.0
    MAX_RETRIES = 3
    BACKOFF_BASE = 1.5
    NAME = "Grok Researcher"

    SYSTEM_PROMPT = """\
You are an expert forecaster for prediction markets. Research a topic and estimate P(YES) with rigorous evidence.

SEARCH BUDGET: You may use the search tool ONLY ONCE. One single search call, no more. Make it count.

RESEARCH APPROACH — adapt to the event type:

For political events and elections:
- Prediction market prices (Polymarket, PredictIt) are strong signals
- Polling aggregates (538, RCP), consider historical polling errors
- For policy/diplomatic events: official sources (Reuters, AP, government statements)
- Procedural requirements (votes needed, veto power, legislative calendar)

For economic and financial events:
- Market-implied probabilities (CME FedWatch for rates, futures markets)
- Central bank communications and forward guidance
- Economic data releases before cutoff

For product launches and technology:
- Official company channels, press releases, SEC filings
- Historical track record (announced vs actual delivery dates)

ANALYSIS PRINCIPLES:
- Prediction market price is your anchor — deviate only with strong contrary evidence
- Official sources > speculation and rumors
- Consider base rates: how often do similar events happen?
- Resolution criteria are literal — read exact wording carefully
- More time until cutoff = more uncertainty
- Range: never return exactly 0 or 1, use [0.01, 0.99]

OUTPUT FORMAT:

SOURCES:
1. [Source title] (source_name, date)
   URL: [url]
   FINDINGS: [2-3 bullet points with the most important facts from this source]

2. [Source title] (source_name, date)
   URL: [url]
   FINDINGS: [2-3 bullet points]

[...repeat for each source found...]

SYNTHESIS: [A concise paragraph synthesizing all sources — current situation, key developments, trajectory. Cite specific sources. Stick to facts.]

PREDICTION: [single number between 0.01 and 0.99]"""

    USER_PROMPT_TEMPLATE = """\
Research topic: {title}
Context: {description}
Resolution deadline: {cutoff}

Search for the latest credible information on this topic, then respond with sources, synthesis, and your PREDICTION."""

    def _build_messages(self, event: dict) -> list[dict]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(
                title=event.get("title", ""),
                description=event.get("description", ""),
                cutoff=event.get("cutoff", ""),
            )},
        ]

    def _parse_response(self, text: str) -> tuple[float, str]:
        prediction = 0.5
        research = text

        match = re.search(r"^PREDICTION:\s*([\d.]+)\s*$", text, re.MULTILINE | re.IGNORECASE)
        if match:
            try:
                raw = float(match.group(1))
                prediction = max(0.01, min(0.99, raw))
            except ValueError:
                pass
            research = text[:match.start()].rstrip()

        return prediction, research

    async def research(self, event: dict) -> dict | None:
        gateway_base = os.environ.get("SANDBOX_PROXY_URL", "http://localhost:8000")
        run_id = os.environ.get("RUN_ID", "unknown")
        url = f"{gateway_base.rstrip('/')}/api/gateway/openrouter/chat/completions"
        messages = self._build_messages(event)
        payload = {
            "model": self.MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.MAX_TOKENS,
            "run_id": run_id,
            "plugins": [{"id": "web", "engine": "native"}],
        }

        start = time.time()
        last_error = None

        print(f"[{self.NAME}] calling {self.MODEL} for: {event.get('title', '?')[:80]}")

        for attempt in range(self.MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                text = data["choices"][0]["message"].get("content") or ""
                cost = data.get("cost", 0.0)
                usage = data.get("usage", {})
                prediction, research = self._parse_response(text)

                elapsed = time.time() - start
                print(f"[{self.NAME}] done in {elapsed:.1f}s, prediction={prediction:.3f}, "
                      f"cost=${cost:.6f}, tokens={usage.get('prompt_tokens', '?')}/{usage.get('completion_tokens', '?')}")

                return {
                    "prediction": prediction,
                    "research": research,
                    "cost": cost,
                    "name": self.NAME,
                }

            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BACKOFF_BASE ** (attempt + 1)
                    print(f"[{self.NAME}] attempt {attempt + 1} failed, retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)

        elapsed = time.time() - start
        print(f"[{self.NAME}] all retries failed after {elapsed:.1f}s: {last_error}")
        return None


_GROK_SKIP_TYPES = {"sports", "crypto", "earnings"}


async def grok_researcher_research(event: dict, context: dict) -> dict | None:
    event_type = context.get("classify", {}).get("event_type")
    if event_type in _GROK_SKIP_TYPES:
        print(f"[{GrokResearcher.NAME}] skipping for event_type={event_type}")
        return None
    if any(cr.get("type") == "indicia" for cr in context.get("crawler", [])):
        print(f"[{GrokResearcher.NAME}] skipping — indicia data available")
        return None
    researcher = GrokResearcher()
    return await researcher.research(event)


import asyncio
import os
import re
import time
from datetime import datetime

import httpx


class _OpenAIResearcher:
    MODEL = "openai/gpt-5-mini"
    MAX_TOKENS = 4096
    TIMEOUT = 120.0
    MAX_RETRIES = 3
    BACKOFF_BASE = 1.5
    NAME = "OpenAI Researcher"

    SYSTEM_PROMPT = """\
You are an expert forecaster for prediction markets. Estimate P(YES) with rigorous research.
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

    USER_PROMPT_TEMPLATE = """\
Question: {title}
Context: {description}
Resolution deadline: {cutoff}
Today: {today}

Search for the latest credible information on this topic, then respond with sources, synthesis, and your PREDICTION."""

    def _build_messages(self, event: dict) -> list[dict]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(
                title=event.get("title", ""),
                description=event.get("description", ""),
                cutoff=event.get("cutoff", ""),
                today=datetime.utcnow().strftime("%Y-%m-%d"),
            )},
        ]

    @staticmethod
    def _parse_response(text: str) -> tuple[float, str]:
        prediction = 0.5
        research = text

        match = re.search(r"^PREDICTION:\s*([\d.]+)\s*$", text, re.MULTILINE | re.IGNORECASE)
        if match:
            try:
                raw = float(match.group(1))
                prediction = max(0.01, min(0.99, raw))
            except ValueError:
                pass

        match_r = re.search(r"^REASONING:\s*(.+)", text, re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if match_r:
            research = match_r.group(1).strip()

        return prediction, research

    async def run(self, event: dict) -> dict | None:
        gateway_base = os.environ.get("SANDBOX_PROXY_URL", "http://localhost:8000")
        run_id = os.environ.get("RUN_ID", "unknown")
        url = f"{gateway_base.rstrip('/')}/api/gateway/openrouter/chat/completions"
        messages = self._build_messages(event)
        payload = {
            "model": self.MODEL,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.MAX_TOKENS,
            "run_id": run_id,
            "reasoning": {"effort": "medium"},
            "plugins": [{"id": "web", "engine": "native"}],
        }

        start = time.time()
        last_error = None

        print(f"[{self.NAME}] calling {self.MODEL} for: {event.get('title', '?')[:80]}")

        for attempt in range(self.MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                text = data["choices"][0]["message"].get("content") or ""
                cost = data.get("cost", 0.0)
                usage = data.get("usage", {})
                prediction, research = self._parse_response(text)

                elapsed = time.time() - start
                print(f"[{self.NAME}] done in {elapsed:.1f}s, prediction={prediction:.3f}, "
                      f"cost=${cost:.6f}, tokens={usage.get('prompt_tokens', '?')}/{usage.get('completion_tokens', '?')}")

                return {
                    "prediction": prediction,
                    "research": research,
                    "cost": cost,
                    "name": self.NAME,
                }

            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.BACKOFF_BASE ** (attempt + 1)
                    print(f"[{self.NAME}] attempt {attempt + 1} failed, retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)

        elapsed = time.time() - start
        print(f"[{self.NAME}] all retries failed after {elapsed:.1f}s: {last_error}")
        return None


async def openai_researcher_research(event: dict, context: dict) -> dict | None:
    if any(cr.get("type") == "indicia" for cr in context.get("crawler", [])):
        print(f"[{_OpenAIResearcher.NAME}] skipping — indicia data available")
        return None
    researcher = _OpenAIResearcher()
    return await researcher.run(event)

import asyncio
import os
import re
import time

import httpx


class _PerplexityResearcher:
    MODEL = "perplexity/sonar"
    TIMEOUT = 120.0
    NAME = "Perplexity Researcher"

    SYSTEM_PROMPT = """\
You are a prediction market researcher. Find current, credible information and estimate the probability a given event resolves YES.

## Guidelines
- Present conflicting sources — don't reconcile them
- Cite sources with dates
- Flag uncertainty: hedging language, speculation vs confirmed
- Note timing: recent vs historical information

## Required output format

SOURCES: [Key findings with citations and dates]

SYNTHESIS: [One paragraph combining all evidence. Cite sources. Facts only.]

PREDICTION: [single number between 0.01 and 0.99]

The last line of your response must always be PREDICTION: followed by a number."""

    USER_PROMPT_TEMPLATE = """\
Question: {title}
Context: {description}
Resolution deadline: {cutoff}

Search for the latest credible information on this topic, then respond with sources, synthesis, and your PREDICTION."""

    def _build_messages(self, event: dict) -> list[dict]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT_TEMPLATE.format(
                title=event.get("title", ""),
                description=event.get("description", ""),
                cutoff=event.get("cutoff", ""),
            )},
        ]

    @staticmethod
    def _parse_response(text: str) -> tuple[float, str]:
        prediction = 0.5
        research = text

        match = re.search(r"^PREDICTION:\s*([\d.]+)\s*$", text, re.MULTILINE | re.IGNORECASE)
        if match:
            try:
                raw = float(match.group(1))
                prediction = max(0.01, min(0.99, raw))
            except ValueError:
                pass
            research = text[:match.start()].rstrip()

        return prediction, research

    async def run(self, event: dict) -> dict | None:
        gateway_base = os.environ.get("SANDBOX_PROXY_URL", "http://localhost:8000")
        run_id = os.environ.get("RUN_ID", "unknown")
        url = f"{gateway_base.rstrip('/')}/api/gateway/openrouter/chat/completions"
        messages = self._build_messages(event)
        payload = {
            "model": self.MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 4096,
            "run_id": run_id,
        }

        start = time.time()
        print(f"[{self.NAME}] calling {self.MODEL} for: {event.get('title', '?')[:80]}")

        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                text = data["choices"][0]["message"].get("content") or ""
                if not text:
                    print(f"[{self.NAME}] empty content")
                    continue

                cost = data.get("cost", 0.0)
                prediction, research = self._parse_response(text)

                elapsed = time.time() - start
                print(f"[{self.NAME}] done in {elapsed:.1f}s, prediction={prediction:.3f}, cost=${cost:.6f}")

                return {
                    "prediction": prediction,
                    "research": research,
                    "cost": cost,
                    "name": self.NAME,
                }

            except httpx.TimeoutException:
                print(f"[{self.NAME}] timeout after {self.TIMEOUT:.0f}s")
                break  # Don't retry on timeout

            except (httpx.HTTPStatusError, Exception) as e:
                print(f"[{self.NAME}] attempt {attempt + 1} failed: {e}")
                if attempt < 1:
                    await asyncio.sleep(1)

        elapsed = time.time() - start
        print(f"[{self.NAME}] failed after {elapsed:.1f}s")
        return None


_PPLX_SKIP_TYPES = {"crypto", "sports"}


async def perplexity_researcher_research(event: dict, context: dict) -> dict | None:
    event_type = context.get("classify", {}).get("event_type")
    if event_type in _PPLX_SKIP_TYPES:
        print(f"[{_PerplexityResearcher.NAME}] skipping for event_type={event_type}")
        return None
    researcher = _PerplexityResearcher()
    return await researcher.run(event)


import os
import re
import time
from datetime import datetime

import httpx


class _LLMSupervisor:
    MODEL = "anthropic/claude-sonnet-4.6"
    FALLBACK_MODEL = "openai/gpt-5.4"
    MAX_TOKENS = 4096
    TIMEOUT = 90.0
    MAX_RETRIES = 2
    NAME = "LLM Supervisor"

    SYSTEM_PROMPT = """\
You are a senior prediction market analyst. You will be given a question, \
and evidence from multiple sources: a research team, live data feeds, and \
possibly a Polymarket price. Synthesize everything and produce a final \
probability estimate.

You have a team of researchers who independently searched for information \
and formed their own predictions. Their research text is valuable context; \
their individual predictions are rough estimates that you may override.

POLYMARKET SIGNAL:
- Polymarket price is THE strongest signal available — it reflects real \
money from informed traders
- If a Polymarket price is provided (via exact match OR mentioned in \
researcher findings), anchor heavily on it
- Deviate from Polymarket only with strong, concrete contrary evidence
- If no Polymarket data exists, rely on researcher evidence and base rates

REVIEW PROCESS:
1. Check for Polymarket price first — this is your anchor
2. Read all researcher findings and live data
3. Identify where sources agree and disagree
4. Weigh evidence quality: Polymarket > live data > official sources > news > speculation
5. Consider base rates and historical patterns
6. Produce a single final probability

CRITICAL RULES:
- If live data directly answers the question (e.g., app is already at target rank), weight it very heavily
- When sources conflict, explain why you favor one over another
- Never blindly average — reason about the evidence
- Resolution criteria are literal — read exact wording carefully
- Range: [0.01, 0.99], never exactly 0 or 1

OUTPUT FORMAT:
REASONING: [2-4 sentences explaining your synthesis and key factors]
PREDICTION: [single number between 0.01 and 0.99]"""

    @classmethod
    def _build_prompt(cls, event: dict, context: dict) -> str:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        parts = [f"Question: {event.get('title', '')}"]

        description = event.get("description", "")
        if description:
            parts.append(f"Description: {description}")

        cutoff = event.get("cutoff", "")
        if cutoff:
            parts.append(f"Resolution deadline: {cutoff}")

        parts.append(f"Today: {today}")

        # Add matching data
        matching = context.get("matching", {})
        if matching.get("status") == "EXACT":
            market = matching.get("market", {})
            parts.append(f"\n--- POLYMARKET EXACT MATCH ---")
            parts.append(f"Market price (YES): {matching['prediction']:.2f}")
            if market.get("question"):
                parts.append(f"Market question: {market['question']}")

        # Add crawler data
        crawler_data = context.get("crawler", [])
        if crawler_data:
            parts.append("\n--- LIVE DATA ---")
            for cr in crawler_data:
                parts.append(cr.get("data", ""))

        # Add research results
        research = context.get("research", [])
        if research:
            parts.append("\n--- RESEARCH TEAM ---")
            for i, r in enumerate(research, 1):
                name = r.get("name", f"Researcher {i}")
                parts.append(f"\n[{name}] (prediction: {r['prediction']:.3f})")
                parts.append(r.get("research", ""))

        if not research and not crawler_data:
            parts.append("\nNo research or live data available. Use your best judgment based on the question alone.")

        parts.append("\n--- END OF DATA ---")
        parts.append("\nReview all evidence above and provide your final assessment.")
        return "\n".join(parts)

    @staticmethod
    def _parse_response(text: str) -> tuple[float, str]:
        prediction = 0.5
        reasoning = text

        match = re.search(r"^PREDICTION:\s*([\d.]+)\s*$", text, re.MULTILINE | re.IGNORECASE)
        if match:
            try:
                raw = float(match.group(1))
                prediction = max(0.01, min(0.99, raw))
            except ValueError:
                pass

        match_r = re.search(r"^REASONING:\s*(.+?)(?=\nPREDICTION:|\Z)", text,
                            re.MULTILINE | re.IGNORECASE | re.DOTALL)
        if match_r:
            reasoning = match_r.group(1).strip()

        return prediction, reasoning

    async def _call_model(self, url: str, payload: dict, model: str) -> dict | None:
        """Single attempt to call a model. Returns parsed response or None."""
        payload = {**payload, "model": model}
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            text = data["choices"][0]["message"].get("content") or ""
            cost = data.get("cost", 0.0)
            prediction, reasoning = self._parse_response(text)
            return {"prediction": prediction, "reasoning": reasoning, "cost": cost}

        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            print(f"[{self.NAME}] {model} failed: {e}")
            return None

    async def run(self, event: dict, context: dict) -> dict | None:
        gateway_base = os.environ.get("SANDBOX_PROXY_URL", "http://localhost:8000")
        run_id = os.environ.get("RUN_ID", "unknown")
        url = f"{gateway_base.rstrip('/')}/api/gateway/openrouter/chat/completions"

        user_prompt = self._build_prompt(event, context)
        payload = {
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": self.MAX_TOKENS,
            "run_id": run_id,
        }

        start = time.time()

        # Try primary model
        print(f"[{self.NAME}] calling {self.MODEL}")
        result = await self._call_model(url, payload, self.MODEL)
        if result:
            elapsed = time.time() - start
            print(f"[{self.NAME}] done in {elapsed:.1f}s, prediction={result['prediction']:.3f}, cost=${result['cost']:.6f}")
            return {"prediction": result["prediction"], "reasoning": result["reasoning"], "cost": result["cost"]}

        # Try fallback model
        print(f"[{self.NAME}] trying fallback {self.FALLBACK_MODEL}")
        result = await self._call_model(url, payload, self.FALLBACK_MODEL)
        if result:
            elapsed = time.time() - start
            print(f"[{self.NAME}] fallback done in {elapsed:.1f}s, prediction={result['prediction']:.3f}, cost=${result['cost']:.6f}")
            return {"prediction": result["prediction"], "reasoning": result["reasoning"], "cost": result["cost"]}

        # All models failed — let scaffolding handle fallback
        elapsed = time.time() - start
        print(f"[{self.NAME}] all models failed after {elapsed:.1f}s")
        return None


async def llm_supervisor_supervision(event: dict, context: dict) -> dict | None:
    supervisor = _LLMSupervisor()
    return await supervisor.run(event, context)


import math


class _PlattCalibrator:
    COEFFICIENTS = {
        "weather": (0.125, -1.45),
    }

    @staticmethod
    def _sigmoid(x):
        x = max(-500.0, min(500.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    @classmethod
    def calibrate(cls, prediction, event_type):
        coeffs = cls.COEFFICIENTS.get(event_type)
        if coeffs:
            slope, intercept = coeffs
            prediction = cls._sigmoid(slope * prediction + intercept)

        return max(0.01, min(0.99, prediction))


def platt_calibrator_calibration(event: dict, context: dict) -> dict:
    supervision = context.get("supervision", {})
    prediction = supervision.get("prediction", 0.35)
    event_type = context.get("classify", {}).get("event_type", "general")

    calibrated = _PlattCalibrator.calibrate(prediction, event_type)

    if event_type in _PlattCalibrator.COEFFICIENTS:
        print(f"[calibration] {event_type}: {prediction:.3f} -> {calibrated:.3f}")

    return {"prediction": calibrated}



# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

# Partial context — accessible from agent_main on timeout/crash
_context = {}


def _best_fallback(context: dict) -> float:
    matching = context.get("matching", {})
    if matching.get("status") == "EXACT" and "prediction" in matching:
        return matching["prediction"]
    research = context.get("research", [])
    if research:
        return sum(r["prediction"] for r in research) / len(research)
    event_type = context.get("classify", {}).get("event_type", "general")
    if event_type in _BASE_RATES:
        return _BASE_RATES[event_type]
    return _FALLBACK_PREDICTION


async def _run(event_data: dict) -> dict:
    global _context
    _context = context = {}

    # Classify
    print("[classify] starting")
    try:
        context["classify"] = simple_classifier_classify(event_data, context)
    except Exception as e:
        print(f"[classify] failed: {e}, defaulting to general")
        context["classify"] = {"event_type": "general"}
    event_type = context["classify"]["event_type"]
    print(f"[classify] done, event_type={event_type}")
    # Matching + Crawler — run in parallel, isolated
    print("[matching+crawler] starting")
    matching_result, crawler_results = await asyncio.gather(
        _run_matching(event_data, context),
        _run_crawlers(event_data, context),
        return_exceptions=True,
    )

    if isinstance(matching_result, BaseException):
        print(f"[matching] failed: {matching_result}")
        context["matching"] = {"status": "NO_MARKETS", "prediction": None}
    else:
        context["matching"] = matching_result
        _track_cost("desearch", matching_result.get("cost", 0))

    if isinstance(crawler_results, BaseException):
        print(f"[crawler] failed: {crawler_results}")
        context["crawler"] = []
    else:
        context["crawler"] = [r for r in crawler_results if r is not None]
        for cr in context["crawler"]:
            _track_cost("desearch", cr.get("cost", 0))

    print(f"[matching] done, status={context['matching']['status']}")
    print(f"[crawler] done, {len(context['crawler'])} results")
    for cr in context["crawler"]:
        print(f"[crawler] [{cr.get('type', '?')}]\n{cr.get('data', '')}")
    print("[research_needed] starting")
    try:
        context["research_needed"] = simple_research_needed_research_needed(event_data, context)
    except Exception as e:
        print(f"[research_needed] failed: {e}, defaulting to yes")
        context["research_needed"] = {"research_needed": True, "reason": "error fallback"}
    do_research = context["research_needed"]["research_needed"]
    print(f"[research_needed] done, needed={do_research}, reason={context['research_needed']['reason']}")
    if do_research:
        # Research (parallel fan-out, isolated)
        print("[research] starting")
        results = await asyncio.gather(
            grok_researcher_research(event_data, context),
            openai_researcher_research(event_data, context),
            perplexity_researcher_research(event_data, context),
            return_exceptions=True,
        )
        context["research"] = [r for r in results
                               if r is not None and not isinstance(r, BaseException)]
        for r in results:
            if isinstance(r, BaseException):
                print(f"[research] one researcher failed: {r}")
        # Track research costs
        for r in context["research"]:
            _track_cost("openrouter", r.get("cost", 0.0))
        print(f"[research] done, {len(context['research'])} results")
    else:
        context["research"] = []
        print("[research] skipped")

    # Supervision
    print("[supervision] starting")
    try:
        result = await llm_supervisor_supervision(event_data, context)
    except Exception as e:
        print(f"[supervision] failed: {e}")
        result = None
    if result is None:
        fallback = _best_fallback(context)
        context["supervision"] = {"prediction": fallback, "reasoning": "Supervision failed."}
    else:
        context["supervision"] = result
        _track_cost("openrouter", result.get("cost", 0.0))
    print(f"[supervision] done, prediction={context['supervision'].get('prediction', '?')}")
    # Calibration
    print("[calibration] starting")
    try:
        context["calibration"] = platt_calibrator_calibration(event_data, context)
    except Exception as e:
        print(f"[calibration] failed: {e}, using supervision prediction")
        context["calibration"] = {"prediction": context["supervision"]["prediction"]}
    print(f"[calibration] done, prediction={context['calibration'].get('prediction', '?')}")
    return {
        "event_id": event_data.get("event_id", ""),
        "prediction": context["calibration"]["prediction"],
        "reasoning": context["supervision"].get("reasoning", ""),
    }


async def _run_matching(event_data, context):
    """Wrapper to run sync matching in the event loop."""
    return gamma_matcher_matching(event_data, context)


async def _run_crawlers(event_data, context):
    """Run all crawlers in parallel, isolated."""
    results = await asyncio.gather(
        appstore_crawler_crawler(event_data, context),
        crypto_crawler_crawler(event_data, context),
        indicia_crawler_crawler(event_data, context),
        return_exceptions=True,
    )
    # Filter out exceptions — log and discard
    clean = []
    for r in results:
        if isinstance(r, BaseException):
            print(f"[crawler] one crawler failed: {r}")
        else:
            clean.append(r)
    return clean


def agent_main(event_data: dict) -> dict:
    start = time.time()
    try:
        result = asyncio.run(
            asyncio.wait_for(_run(event_data), timeout=_GLOBAL_TIMEOUT)
        )
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"[agent_main] GLOBAL TIMEOUT after {elapsed:.1f}s")
        fallback = _best_fallback(_context)
        result = {
            "event_id": event_data.get("event_id", ""),
            "prediction": fallback,
            "reasoning": "Agent timed out.",
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"[agent_main] FATAL ERROR after {elapsed:.1f}s: {e}")
        fallback = _best_fallback(_context)
        result = {
            "event_id": event_data.get("event_id", ""),
            "prediction": fallback,
            "reasoning": f"Agent failed: {type(e).__name__}",
        }

    elapsed = time.time() - start
    cost_parts = [f"{k}=${v:.6f}" for k, v in _costs.items() if v > 0]
    print(f"[agent_main] done in {elapsed:.1f}s, prediction={result['prediction']}")
    print(f"[costs] {' | '.join(cost_parts) if cost_parts else 'no costs tracked'}")
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        event_data = json.loads(sys.argv[1])
    else:
        event_data = json.loads(sys.stdin.read())

    print(json.dumps(agent_main(event_data), indent=2))
