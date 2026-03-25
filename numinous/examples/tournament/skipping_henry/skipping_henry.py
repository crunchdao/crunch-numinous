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


Stn = namedtuple("Stn", ["rid", "gw", "oai", "dsr", "ort", "rtr", "bkf", "tlim", "mdl", "fbk_mdl"])

_rid = os.getenv("RUN_ID") or str(uuid4())
_gw = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")

CFG = Stn(
    rid=_rid,
    gw=_gw,
    oai=f"{_gw}/api/gateway/openai",
    dsr=f"{_gw}/api/gateway/desearch",
    ort=f"{_gw}/api/gateway/openrouter/chat/completions",
    rtr=3,
    bkf=2.0,
    tlim=180.0,
    mdl=("gpt-5-mini",),
    fbk_mdl="anthropic/claude-sonnet-4-6",
)

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


def _isWeatherEvent(title):
    return ' temperature ' in (title or '').lower()


def _isAppStoreEvent(title):
    return 'app store' in (title or '').lower()


def _clamp(v, lo=0.01, hi=0.99):
    if v < lo: return lo
    if v > hi: return hi
    return v


ENRICH_SYSTEM = """You are an expert forecaster. Analyze the provided market data and estimate P(YES).
The market data below reflects current live pricing — anchor your prediction strongly on this data. Only deviate if you identify a clear factual reason to (e.g. an event has already occurred, a deadline has passed, or the question contains an obvious resolution).
Range: [0.01, 0.99].
PREDICTION: [number]
REASONING: [2-3 sentences]"""


def _build_enrich_prompt(eventData, exact):
    op = exact.get("outcome_prices", "")
    try:
        pl = json.loads(op) if isinstance(op, str) else op
        yes_price = float(pl[0]) if isinstance(pl, list) and pl else None
    except (json.JSONDecodeError, ValueError, IndexError):
        yes_price = None
    change = exact.get("one_day_price_change")
    today = datetime.now().strftime("%Y-%m-%d")
    parts = [
        "Market question: " + exact.get("question", ""),
        "Current YES price: " + (f"{yes_price:.0%}" if yes_price is not None else "unknown"),
    ]
    if change is not None:
        try:
            pct = round(float(change) * 100)
            if pct != 0:
                parts.append(f"24h change: {'+' if pct > 0 else ''}{pct}%")
        except (TypeError, ValueError):
            pass
    parts.append("")
    parts.append("Event title: " + eventData.get("title", ""))
    parts.append("Cutoff: " + eventData.get("cutoff", ""))
    parts.append("Today: " + today)
    return "\n".join(parts)


def enrich_exact(eventData, gammaResult, is_weather=False):
    """Enrich prediction from an exact Polymarket match via LLM."""
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

        async def _enrich():
            async with httpx.AsyncClient(timeout=180.0) as cli:
                d, cst = await Gw.oai_resp(cli, body, web=False)
                txt = Prs.grab_txt(d)
                print(f"[HENRY] enrich_exact LLM response: {txt[:200]}")
                obj = Prs.dig_score(txt)
                lk, rationale = Prs.unify(obj)
                if lk is not None:
                    return _clamp(lk), rationale or ""
                # Try text-based extraction as fallback
                for marker in ("prediction:", "probability:"):
                    pos = txt.lower().find(marker)
                    if pos >= 0:
                        after = txt[pos + len(marker):pos + len(marker) + 20].strip()
                        nums = ""
                        for ch in after:
                            if ch.isdigit() or ch == '.': nums += ch
                            elif nums: break
                        if nums:
                            v = float(nums)
                            if v > 1: v /= 100.0
                            return _clamp(v), rationale or txt[:300]
                return None, None

        result = asyncio.run(_enrich())
        if result[0] is not None:
            return {"event_id": eid, "prediction": result[0], "reasoning": result[1]}
    except Exception as exc:
        print(f"[HENRY] enrich_exact failed: {exc}")

    # Fallback: extract raw PM price
    if is_weather:
        return {"event_id": eid, "prediction": 0.20, "reasoning": ""}
    exact = gammaResult.get("exact_match") or {}
    op = exact.get("outcome_prices", "")
    price = 0.35
    if op:
        try:
            pl = json.loads(op) if isinstance(op, str) else op
            if isinstance(pl, list) and pl:
                price = _clamp(float(pl[0]))
        except (json.JSONDecodeError, ValueError, IndexError):
            pass
    return {"event_id": eid, "prediction": price, "reasoning": ""}


def composeAppStoreInquiry(questionRecord, gammaContext):
    """Build LLM prompt for app store events with Gamma + incumbent context."""
    today = datetime.now().strftime("%Y-%m-%d")
    parts = [
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
    return "\n".join(parts)


class Cls:
    @staticmethod
    def tag(hdr, desc=""):
        merged = (hdr + " " + desc).lower()
        h = hdr.lower()
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
    def resolve(evt):
        topics = evt.get("metadata", {}).get("topics", [])
        for t in topics:
            if t.lower() in KNOWN_CATS:
                return t
        found = Cls.tag(evt.get("title", ""), evt.get("description", ""))
        return found if found != "other" else "Other"


class Cal:
    @staticmethod
    def _interp(v, xs, ys):
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
                out[i] = lo_y + (hi_y - lo_y) * (x - lo_x) / (hi_x - lo_x) if hi_x != lo_x else lo_y
        return np.clip(out, 0, 1)

    @staticmethod
    def adjust(score, cat_key):
        import numpy as np
        spec = CAL_TBL.get(cat_key)
        if spec is None:
            return score
        if spec.kind == "isotonic":
            return float(
                Cal._interp(
                    np.array([score]),
                    np.array(spec.vals["xs"]),
                    np.array(spec.vals["ys"]),
                )[0]
            )
        if spec.kind == "platt":
            z = spec.vals["a"] * score + spec.vals["b"]
            return float(1 / (1 + np.exp(-np.clip(z, -500, 500))))
        return score


class Prs:
    @staticmethod
    def grab_txt(resp):
        for item in resp.get("output", []):
            if item.get("type") == "message":
                for blk in item.get("content", []):
                    if blk.get("type") in ("output_text", "text") and blk.get("text"):
                        return blk["text"]
        return ""

    @staticmethod
    def norm(v):
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 < v <= 100.0:
            return v / 100.0
        return None

    @staticmethod
    def dig_score(blob):
        loc = blob.find("@score{")
        if loc >= 0:
            brace_start = loc + 6
            depth = 0
            in_str = False
            end = -1
            i = brace_start
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
                chunk = blob[brace_start : end + 1].strip()
                try:
                    obj = json.loads(chunk)
                    if "likelihood" in obj or "probability" in obj:
                        return obj
                except Exception:
                    pass

        fence_start = blob.find("```json")
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

        m = re.search(r'\{[^{}]*"(?:probability|likelihood)"[^{}]*\}', blob, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return None

    @staticmethod
    def unify(parsed):
        if parsed is None:
            return None, None
        raw = parsed.get("likelihood", parsed.get("probability"))
        if raw is None:
            return None, None
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None, None
        n = Prs.norm(v)
        rationale = parsed.get("rationale", parsed.get("reasoning", ""))
        return n, rationale


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


class Gw:
    @staticmethod
    async def _post(cli, url, body, tmo=60.0):
        return await cli.post(url, json=body, timeout=tmo)

    @staticmethod
    async def oai_resp(cli, body, web=True):
        last_exc = None
        body["reasoning"] = {"effort": "medium"}
        if web:
            body["tools"] = [{"type": "web_search"}]
        elif "tools" in body:
            del body["tools"]

        for mdl in CFG.mdl:
            body["model"] = mdl
            for att in range(CFG.rtr):
                try:
                    print(f"[ASSESS] Querying {mdl} (try {att + 1}/{CFG.rtr})...")
                    r = await Gw._post(cli, f"{CFG.oai}/responses", body, tmo=180.0)
                    if r.status_code == 200:
                        d = r.json()
                        return d, d.get("cost", 0.0)
                    if r.status_code in (429, 500, 502, 503):
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
                    if exc.response.status_code in (429, 500, 502, 503):
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

    @staticmethod
    async def crawl(cli, url):
        try:
            r = await Gw._post(
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

    @staticmethod
    async def lookup(cli, qry):
        try:
            print(f"[LOOKUP] {qry[:80]}...")
            r = await Gw._post(
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
                page = await Gw.crawl(cli, link)
                if page:
                    return f"{header}\n\nFetched from {link}:\n{page}"
            return header
        except Exception as exc:
            print(f"[LOOKUP] Err: {exc}")
        return ""

    @staticmethod
    async def openrouter(cli, messages, model=None):
        """Call an OpenRouter model through the gateway. Returns (text, cost)."""
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
                r = await Gw._post(cli, CFG.ort, payload, tmo=120.0)
                if r.status_code == 200:
                    d = r.json()
                    choices = d.get("choices") or []
                    if not choices:
                        raise ValueError("OpenRouter response has no choices")
                    msg = choices[0].get("message") or {}
                    raw_content = msg.get("content")
                    if isinstance(raw_content, str):
                        txt = raw_content
                    elif isinstance(raw_content, list):
                        # Handle structured content blocks: [{"type":"text","text":"..."},...]
                        txt = "".join(
                            blk.get("text", "") for blk in raw_content
                            if isinstance(blk, dict) and blk.get("type") in ("text", "output_text")
                        )
                    else:
                        txt = ""
                    if not txt.strip():
                        raise ValueError(f"OpenRouter returned empty content (raw type: {type(raw_content)})")
                    cost = d.get("cost", 0.0)
                    return txt, cost
                if r.status_code in (429, 500, 502, 503):
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


class Fix:
    @staticmethod
    async def recover(cli, raw):
        instr = (
            "Extract prediction data from this malformed response and return valid JSON."
            f"\n\nRaw response:\n{raw}"
            '\n\nReturn ONLY valid JSON:\n{"probability": <0-1>, "reasoning": "<extracted reasoning>"}'
        )
        for mdl in ("gpt-5-mini", "gpt-5-nano"):
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
                hit = re.search(
                    r"(?:probability|likelihood)[\"']?\s*[:=]\s*([\d.]+)", txt, re.IGNORECASE
                )
                if hit:
                    n = Prs.norm(float(hit.group(1)))
                    if n is not None:
                        return {
                            "data": {
                                "likelihood": max(0.01, min(0.99, n)),
                                "rationale": txt,
                            },
                            "cost": cst,
                        }
            except Exception as exc:
                print(f"[REPAIR] {mdl} err: {exc}")
                continue
        return {"data": {"likelihood": 0.35, "rationale": "Decoding failed"}, "cost": 0.0}

    @staticmethod
    async def recover_openrouter(cli, raw):
        """Repair fallback using OpenRouter when OpenAI is unavailable."""
        instr = (
            "Extract prediction data from this malformed response and return valid JSON."
            f"\n\nRaw response:\n{raw}"
            '\n\nReturn ONLY valid JSON:\n{"probability": <0-1>, "reasoning": "<extracted reasoning>"}'
        )
        try:
            txt, cst = await Gw.openrouter(
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
            hit = re.search(
                r"(?:probability|likelihood)[\"']?\s*[:=]\s*([\d.]+)", txt, re.IGNORECASE
            )
            if hit:
                n = Prs.norm(float(hit.group(1)))
                if n is not None:
                    return {
                        "data": {
                            "likelihood": max(0.01, min(0.99, n)),
                            "rationale": txt,
                        },
                        "cost": cst,
                    }
        except Exception as exc:
            print(f"[REPAIR-OR] err: {exc}")
        return {"data": {"likelihood": 0.35, "rationale": "Decoding failed"}, "cost": 0.0}


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


class Eng:
    @staticmethod
    async def _core(evt, acc, gammaResult=None):
        hdr = evt.get("title", "")
        desc = evt.get("description", "")
        closes = evt.get("cutoff", "")
        today = datetime.now().strftime("%Y-%m-%d")
        spend = 0.0

        async with httpx.AsyncClient(timeout=180.0) as cli:

            # App Store with Gamma context — LLM without web search
            if gammaResult and gammaResult.get("context"):
                print(f"[HENRY] app_store RELATED -> LLM with gamma context (no web search)")
                app_msg = composeAppStoreInquiry(evt, gammaResult["context"])
                body = {
                    "model": "gpt-5-pro",
                    "input": [
                        {"role": "developer", "content": APP_STORE_SYSTEM},
                        {"role": "user", "content": app_msg},
                    ],
                    "run_id": CFG.rid,
                }
                try:
                    d, cst = await Gw.oai_resp(cli, body, web=False)
                    spend += cst
                    txt = Prs.grab_txt(d)
                    print(f"[APP OUTPUT]\n{txt}")
                    obj = Prs.dig_score(txt)
                    lk, rationale = Prs.unify(obj)
                    if lk is not None:
                        score = max(0.01, min(0.99, lk))
                        await acc.push(score, rationale or "")
                        return {
                            "event_id": evt.get("event_id", "?"),
                            "prediction": score,
                            "reasoning": rationale or "",
                            "cost": spend,
                        }
                except Exception as exc:
                    print(f"[APP FAILED] {exc}, falling through to standard path...")

            # Standard path: primary engines with web search
            print(f"[HENRY] llm_flow: primary engines + web search")
            usr_msg = Tpl.usr(hdr, desc, closes, today)
            body = {
                "model": "gpt-5-pro",
                "input": [
                    {"role": "developer", "content": Tpl.sys()},
                    {"role": "user", "content": usr_msg},
                ],
                "run_id": CFG.rid,
            }

            used_openrouter = False
            try:
                d, cst = await Gw.oai_resp(cli, body, web=True)
                spend += cst
                txt = Prs.grab_txt(d)
                print(f"[OUTPUT]\n{txt}")
            except Exception as exc:
                used_openrouter = True
                print(f"[WEB FAILED] {exc}, pivoting to OpenRouter fallback...")
                queries = [
                    f"polymarket {hdr}",
                    f"{hdr} betting odds",
                    f"{hdr} latest news",
                ]
                parts = []
                for q in queries:
                    seg = await Gw.lookup(cli, q)
                    if seg:
                        parts.append(seg)
                if not parts:
                    # Desearch also failed — send to Sonnet without research
                    print("[WARN] Desearch returned nothing, querying Sonnet without research...")
                    fbk_messages = [
                        {"role": "system", "content": Tpl.sys_no_search()},
                        {"role": "user", "content": Tpl.usr_no_search(hdr, desc, closes, today)},
                    ]
                else:
                    intel = "\n\n".join(parts)
                    fbk_msg = Tpl.fbk(hdr, desc, closes, today, intel)
                    fbk_messages = [
                        {"role": "system", "content": Tpl.sys_with_research()},
                        {"role": "user", "content": fbk_msg},
                    ]
                try:
                    txt, cst = await Gw.openrouter(cli, fbk_messages)
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
                    rep = await Fix.recover_openrouter(cli, txt)
                else:
                    print("[WARN] Structured extraction failed, invoking repair...")
                    rep = await Fix.recover(cli, txt)
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

    @staticmethod
    async def assess(evt, gammaResult=None):
        acc = Acc()
        try:
            return await asyncio.wait_for(Eng._core(evt, acc, gammaResult=gammaResult), timeout=CFG.tlim)
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


if __name__ == "__main__":
    demo = {
        "event_id": "test-001",
        "title": "Will AnkiMobile Flashcards be the #3 Paid app in the US iPhone App Store on January 28, 2026?",
        "description": (
            "This market will resolve according to the iOS app ranked #3 in the United States"
            " on the iPhone Apple App Store's overall Top Charts under \"Paid Apps\", as of"
            " 23:59 UTC on the specified date (January 28, 2026).\n\nTo find the overall chart,"
            " open the US iOS App Store app, tap \"Apps\" at the bottom, scroll down to"
            " \"Top Paid Apps\" and tap \"See All\". Then under \"Paid Apps\" in the"
            " \"Top Charts\" section, you'll see the list that will be used as the resolution"
            " source.\n\nResolution source URL:"
            " https://apps.apple.com/us/iphone/charts/36?chart=top-paid\n"
        ),
        "cutoff": "2026-01-22T03:00:00Z",
        "metadata": {},
    }
    print("\n" + "=" * 60)
    r = agent_main(demo)
    print(f"\nOutcome: {r}")
