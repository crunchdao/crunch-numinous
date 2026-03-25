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
#                         HTTP REQUEST TRACKING
# =============================================================================

class HttpxManager:
    """
    Manager class for tracking all httpx requests (GET and POST).
    
    Counts total requests, method types, and stores URLs for each request.
    Supports both async and sync httpx calls.
    """
    
    def __init__(self):
        self.total_requests = 0
        self.get_count = 0
        self.post_count = 0
        self.request_log = []  # List of (method, url) tuples
        self._client = None
    
    async def get(self, url: str, **kwargs) -> httpx.Response:
        """Async GET request with tracking."""
        self.total_requests += 1
        self.get_count += 1
        self.request_log.append(("GET", url))
        
        if self._client is None:
            raise RuntimeError("HttpxManager not initialized with client")
        
        return await self._client.get(url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> httpx.Response:
        """Async POST request with tracking."""
        self.total_requests += 1
        self.post_count += 1
        self.request_log.append(("POST", url))
        
        if self._client is None:
            raise RuntimeError("HttpxManager not initialized with client")
        
        return await self._client.post(url, **kwargs)
    
    def post_sync(self, url: str, **kwargs) -> httpx.Response:
        """Synchronous POST request with tracking."""
        self.total_requests += 1
        self.post_count += 1
        self.request_log.append(("POST", url))
        return httpx.post(url, **kwargs)
    
    def set_client(self, client: httpx.AsyncClient):
        """Set the underlying httpx client for async operations."""
        self._client = client
    
    def get_stats(self) -> dict:
        """Get statistics about requests made."""
        return {
            "total_requests": self.total_requests,
            "get_count": self.get_count,
            "post_count": self.post_count,
            "request_log": self.request_log
        }
    
    def print_stats(self):
        """Print request statistics."""
        print(f"\n=== HttpxManager Statistics ===")
        print(f"Total requests: {self.total_requests}")
        print(f"GET requests: {self.get_count}")
        print(f"POST requests: {self.post_count}")
        print(f"\nRequest log ({len(self.request_log)} entries):")
        for method, url in self.request_log:
            print(f"  {method:4s} - {url}")
        print(f"================================\n")
    
    def reset(self):
        """Reset all counters and logs."""
        self.total_requests = 0
        self.get_count = 0
        self.post_count = 0
        self.request_log = []
        self._client = None


# Global HttpxManager instance
HTTP_MANAGER = HttpxManager()


# =============================================================================
#                         UTILITY FUNCTIONS
# =============================================================================

def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to maximum length with suffix."""
    if not text or len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def safe_float_parse(value, default: float = 0.0) -> float:
    """Safely parse float value with fallback."""
    try:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip().replace(",", ""))
        return default
    except (ValueError, TypeError, AttributeError):
        return default


def clamp_probability(value: float) -> float:
    """Clamp probability value to [0.01, 0.99] range."""
    return max(0.01, min(0.99, value))


def format_cost_dollars(amount: float, decimals: int = 4) -> str:
    """Format cost amount in dollars with specified decimals."""
    return f"${amount:.{decimals}f}"


def normalize_text(text: str) -> str:
    """Normalize text by collapsing whitespace and lowercasing."""
    if not text:
        return ""
    return " ".join(text.lower().split())


def extract_json_block(text: str, start_marker: str = "{", end_marker: str = "}") -> str:
    """Extract JSON block from text between markers."""
    if not text:
        return ""
    start_idx = text.find(start_marker)
    if start_idx < 0:
        return ""
    end_idx = text.rfind(end_marker)
    if end_idx < 0 or end_idx <= start_idx:
        return ""
    return text[start_idx:end_idx + 1]


def safe_dict_extract(dictionary: dict, key: str, default=None):
    """Safely extract value from dictionary with fallback."""
    if not isinstance(dictionary, dict):
        return default
    return dictionary.get(key, default)


def list_to_display_string(items: list, max_items: int = 5, separator: str = ", ") -> str:
    """Convert list to display string with optional truncation."""
    if not items or not isinstance(items, list):
        return ""
    display_items = items[:max_items] if max_items and len(items) > max_items else items
    result = separator.join(str(item) for item in display_items)
    if max_items and len(items) > max_items:
        result += f"{separator}... ({len(items) - max_items} more)"
    return result


def is_valid_numeric(value) -> bool:
    """Check if value can be converted to a valid number."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


# =============================================================================
#                    ADDITIONAL UTILITY FUNCTIONS
# =============================================================================

def calculate_average(values: list, default: float = 0.0) -> float:
    """Calculate average of numeric values with fallback."""
    if not values:
        return default
    valid_values = [v for v in values if is_valid_numeric(v)]
    if not valid_values:
        return default
    return sum(float(v) for v in valid_values) / len(valid_values)


def calculate_median(values: list, default: float = 0.0) -> float:
    """Calculate median of numeric values with fallback."""
    if not values:
        return default
    valid_values = sorted([float(v) for v in values if is_valid_numeric(v)])
    if not valid_values:
        return default
    n = len(valid_values)
    mid = n // 2
    if n % 2 == 0:
        return (valid_values[mid - 1] + valid_values[mid]) / 2.0
    return valid_values[mid]


def calculate_std_dev(values: list, default: float = 0.0) -> float:
    """Calculate standard deviation with fallback."""
    if not values or len(values) < 2:
        return default
    valid_values = [float(v) for v in values if is_valid_numeric(v)]
    if len(valid_values) < 2:
        return default
    mean = sum(valid_values) / len(valid_values)
    variance = sum((x - mean) ** 2 for x in valid_values) / len(valid_values)
    return variance ** 0.5


def weighted_average(values: list, weights: list, default: float = 0.0) -> float:
    """Calculate weighted average with fallback."""
    if not values or not weights or len(values) != len(weights):
        return default
    total_weight = sum(w for w in weights if is_valid_numeric(w))
    if total_weight == 0:
        return default
    weighted_sum = sum(float(v) * float(w) for v, w in zip(values, weights) 
                       if is_valid_numeric(v) and is_valid_numeric(w))
    return weighted_sum / total_weight


def round_to_decimals(value: float, decimals: int = 2) -> float:
    """Round to specified decimal places."""
    try:
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return 0.0


def round_to_significant(value: float, significant: int = 3) -> float:
    """Round to specified significant figures."""
    if value == 0:
        return 0.0
    try:
        from math import log10, floor
        return round(value, -int(floor(log10(abs(value)))) + (significant - 1))
    except (ValueError, TypeError, OverflowError):
        return value


def percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 0.0 if new_value == 0 else 100.0
    return ((new_value - old_value) / abs(old_value)) * 100.0


def moving_average(values: list, window: int = 3) -> list:
    """Calculate moving average with specified window size."""
    if not values or window <= 0:
        return []
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_values = values[start:i + 1]
        result.append(calculate_average(window_values))
    return result


def ensure_list(value) -> list:
    """Ensure value is a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def flatten_list(nested_list: list) -> list:
    """Flatten a nested list structure."""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def chunk_list(items: list, chunk_size: int) -> list:
    """Split list into chunks of specified size."""
    if not items or chunk_size <= 0:
        return []
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def deduplicate_list(items: list, key=None) -> list:
    """Remove duplicates from list while preserving order."""
    if key is None:
        seen = set()
        return [x for x in items if not (x in seen or seen.add(x))]
    seen = set()
    result = []
    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


def merge_dicts(*dicts, **kwargs) -> dict:
    """Merge multiple dictionaries with later values overwriting earlier ones."""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    result.update(kwargs)
    return result


def deep_get(dictionary: dict, *keys, default=None):
    """Safely get nested dictionary value with fallback."""
    current = dictionary
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def remove_none_values(dictionary: dict) -> dict:
    """Remove keys with None values from dictionary."""
    return {k: v for k, v in dictionary.items() if v is not None}


def filter_dict_by_keys(dictionary: dict, keys: list) -> dict:
    """Filter dictionary to only include specified keys."""
    return {k: v for k, v in dictionary.items() if k in keys}


# =============================================================================
#                         HELPER CLASSES
# =============================================================================

class TimeTracker:
    """Track execution time for operations."""
    
    def __init__(self):
        self.times = {}
        self.counts = {}
    
    def start(self, operation: str):
        """Start timing an operation."""
        if operation not in self.times:
            self.times[operation] = []
            self.counts[operation] = 0
        return time.time()
    
    def end(self, operation: str, start_time: float):
        """End timing an operation."""
        elapsed = time.time() - start_time
        if operation in self.times:
            self.times[operation].append(elapsed)
            self.counts[operation] += 1
    
    def get_stats(self, operation: str = None) -> dict:
        """Get timing statistics."""
        if operation:
            if operation not in self.times:
                return {}
            times = self.times[operation]
            return {
                "count": self.counts[operation],
                "total": sum(times),
                "average": calculate_average(times),
                "min": min(times) if times else 0,
                "max": max(times) if times else 0,
            }
        return {op: self.get_stats(op) for op in self.times.keys()}
    
    def print_stats(self):
        """Print timing statistics."""
        print("\n=== TimeTracker Statistics ===")
        for operation, stats in self.get_stats().items():
            print(f"{operation}:")
            print(f"  Count: {stats['count']}")
            print(f"  Total: {format_elapsed_time(stats['total'])}")
            print(f"  Avg:   {format_elapsed_time(stats['average'])}")
            print(f"  Min:   {format_elapsed_time(stats['min'])}")
            print(f"  Max:   {format_elapsed_time(stats['max'])}")
    
    def reset(self):
        """Reset all timing data."""
        self.times.clear()
        self.counts.clear()


class CostTracker:
    """Track API costs and usage."""
    
    def __init__(self):
        self.costs = defaultdict(float)
        self.calls = defaultdict(int)
    
    def add_cost(self, service: str, cost: float):
        """Add cost for a service."""
        self.costs[service] += cost
        self.calls[service] += 1
    
    def get_total_cost(self) -> float:
        """Get total cost across all services."""
        return sum(self.costs.values())
    
    def get_service_cost(self, service: str) -> float:
        """Get cost for specific service."""
        return self.costs.get(service, 0.0)
    
    def get_stats(self) -> dict:
        """Get cost statistics."""
        return {
            "total_cost": self.get_total_cost(),
            "by_service": dict(self.costs),
            "calls": dict(self.calls),
        }
    
    def print_stats(self):
        """Print cost statistics."""
        print("\n=== CostTracker Statistics ===")
        print(f"Total Cost: {format_cost_dollars(self.get_total_cost())}")
        print("\nBy Service:")
        for service, cost in self.costs.items():
            calls = self.calls[service]
            avg_cost = cost / calls if calls > 0 else 0
            print(f"  {service}:")
            print(f"    Calls: {calls}")
            print(f"    Total: {format_cost_dollars(cost)}")
            print(f"    Avg:   {format_cost_dollars(avg_cost)}")
    
    def reset(self):
        """Reset all cost tracking data."""
        self.costs.clear()
        self.calls.clear()


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_window: float):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def can_proceed(self) -> bool:
        """Check if call is allowed based on rate limit."""
        now = time.time()
        # Remove calls outside time window
        self.calls = [t for t in self.calls if now - t < self.time_window]
        return len(self.calls) < self.max_calls
    
    def record_call(self):
        """Record a call."""
        self.calls.append(time.time())
    
    def wait_time(self) -> float:
        """Get time to wait before next call is allowed."""
        if self.can_proceed():
            return 0.0
        now = time.time()
        oldest_call = min(self.calls)
        return max(0.0, self.time_window - (now - oldest_call))
    
    def reset(self):
        """Reset rate limiter."""
        self.calls.clear()


# Global instances
TIME_TRACKER = TimeTracker()
COST_TRACKER = CostTracker()


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
    url_preview = truncate_string(url, 80)
    
    # Start timing gamma API call
    gamma_start = TIME_TRACKER.start("gamma_api")
    
    for attempt in range(3):
        try:
            resp = HTTP_MANAGER.post_sync(f"{_GATEWAY_URL}/web/crawl",
                              json={"url": url, "run_id": _GATEWAY_RUN_ID}, timeout=30.0)
            if resp.status_code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(1.0 * (attempt + 1)); continue
            resp.raise_for_status()
            data = resp.json()
            cost = safe_float_parse(data.get("cost", _GATEWAY_COST_PER_CALL))
            if cost > 0:
                _gateway_total_cost += cost
            content = data.get("content", "")
            if not content:
                TIME_TRACKER.end("gamma_api", gamma_start)
                return None
            
            TIME_TRACKER.end("gamma_api", gamma_start)
            return json.loads(content) if isinstance(content, str) else content
        except Exception as e:
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1)); continue
            TIME_TRACKER.end("gamma_api", gamma_start)
            return None
    
    TIME_TRACKER.end("gamma_api", gamma_start)
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
    # Use normalize_text utility then remove punctuation
    normalized = normalize_text(text)
    return re.sub(r"[^\w\s]", "", normalized)

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
                json_block = extract_json_block(fencedContent)
                if json_block:
                    try:
                        return json.loads(json_block)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
    json_block = extract_json_block(textBlock)
    if json_block:
        try:
            return json.loads(json_block)
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
    if is_valid_numeric(numericAccumulator):
        return safe_float_parse(numericAccumulator)
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
    HTTP_MANAGER.set_client(httpSession)
    engine_short = truncate_string(engineName, 50)
    
    # Start timing this LLM call
    call_start = TIME_TRACKER.start("llm_openai")
    
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
        serverResponse = await HTTP_MANAGER.post(fullUrl, json=requestPayload)
        serverResponse.raise_for_status()
        return serverResponse.json()

    rawData = await withReattempts(performPost)
    extractedText = extractTextFromOpenAIReply(rawData)
    costIncurred = rawData.get("cost", 0.0)
    
    # Track timing and cost
    TIME_TRACKER.end("llm_openai", call_start)
    COST_TRACKER.add_cost("openai", costIncurred)
    
    return extractedText, costIncurred


async def invokeChutesEndpoint(httpSession, engineName, userPromptText, tokenCeiling=2000):
    HTTP_MANAGER.set_client(httpSession)
    engine_short = truncate_string(engineName, 50)
    
    # Start timing this LLM call
    call_start = TIME_TRACKER.start("llm_chutes")
    
    requestPayload = {
        "model": engineName,
        "messages": [{"role": "user", "content": userPromptText}],
        "max_tokens": tokenCeiling,
        "run_id": GATEWAY_SETTINGS["executionId"],
    }
    try:
        fullUrl = GATEWAY_SETTINGS["baseUrl"] + GATEWAY_SETTINGS["endpoints"]["chutes"]

        async def performPost():
            serverResponse = await HTTP_MANAGER.post(fullUrl, json=requestPayload)
            serverResponse.raise_for_status()
            return serverResponse.json()

        rawData = await withReattempts(performPost, maxTries=2)
        choicesList = rawData.get("choices", [])
        if not choicesList:
            TIME_TRACKER.end("llm_chutes", call_start)
            return None
        firstChoice = choicesList[0]
        messageContent = firstChoice.get("message", {})
        
        # Track timing
        TIME_TRACKER.end("llm_chutes", call_start)
        
        return messageContent.get("content", "")
    except Exception:
        TIME_TRACKER.end("llm_chutes", call_start)
        return None


async def invokeDesearchEndpoint(httpSession, searchPhrase):
    HTTP_MANAGER.set_client(httpSession)
    search_preview = truncate_string(searchPhrase, 60)
    
    # Start timing this search call
    search_start = TIME_TRACKER.start("desearch")
    
    requestPayload = {
        "query": searchPhrase,
        "model": "NOVA",
        "run_id": GATEWAY_SETTINGS["executionId"],
    }
    try:
        fullUrl = GATEWAY_SETTINGS["baseUrl"] + GATEWAY_SETTINGS["endpoints"]["desearch"]
        serverResponse = await HTTP_MANAGER.post(fullUrl, json=requestPayload)
        serverResponse.raise_for_status()
        result = serverResponse.json().get("results", [])
        
        # Track timing
        TIME_TRACKER.end("desearch", search_start)
        
        return result
    except Exception:
        TIME_TRACKER.end("desearch", search_start)
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
            engine_short = truncate_string(engineIdentifier, 50)
            print(f"[BARD] primary engine={engine_short} (web_search)")
            replyText, _ = await invokeOpenAIEndpoint(
                httpSession, engineIdentifier,
                conversationPayload, webLookupCapability,
            )
            if not replyText:
                print(f"[BARD] {engine_short} returned empty response")
                continue
            estimatedLikelihood, derivedRationale = interpretForecastResponse(replyText)
            print(f"[BARD] {engine_short} -> {estimatedLikelihood:.3f}")
            return estimatedLikelihood, derivedRationale
        except Exception as engineFailure:
            error_msg = truncate_string(str(engineFailure), 80)
            print(f"[ENGINE] {engine_short} encountered failure: {error_msg}")
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
            termsList = safe_dict_extract(parsedJson, "searchTerms") or safe_dict_extract(parsedJson, "queries")
            if termsList:
                return termsList[:4]
    abbreviatedTitle = truncate_string(questionRecord.get("title", ""), 80)
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
            sourceUrl = safe_dict_extract(entry, "url", "source")
            sourceTitle = safe_dict_extract(entry, "title", "")
            excerptRaw = safe_dict_extract(entry, "snippet", "")
            excerptTrimmed = truncate_string(excerptRaw, 300, "")
            formattedEntry = "[" + sourceUrl + "] " + sourceTitle + ": " + excerptTrimmed
            collectedExcerpts.append(formattedEntry)
    if not collectedExcerpts:
        return "No findings from search"
    return "\n\n".join(collectedExcerpts)


async def attemptFallbackEngines(httpSession, questionRecord):
    print(f"[BARD] fallback: desearch + chutes")
    searchTermList = await produceSearchTerms(httpSession, questionRecord)
    terms_display = list_to_display_string(searchTermList, max_items=4)
    print(f"[BARD] search terms: {terms_display}")
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
        engine_short = truncate_string(engineIdentifier, 50)
        engineReply = await invokeChutesEndpoint(
            httpSession, engineIdentifier, filledPrompt, tokenCeiling=1500
        )
        if engineReply is None:
            continue
        parsedJson = isolateJsonFromText(engineReply)
        if parsedJson is None:
            continue
        rawLikelihoodVal = safe_dict_extract(parsedJson, "likelihood")
        if rawLikelihoodVal is None:
            rawLikelihoodVal = safe_dict_extract(parsedJson, "probability")
        if rawLikelihoodVal is None:
            continue
        estimatedVal = constrainToValidRange(safe_float_parse(rawLikelihoodVal, 0.5))
        rationaleVal = safe_dict_extract(parsedJson, "rationale") or safe_dict_extract(parsedJson, "reasoning", "")
        print(f"[BARD] fallback {engine_short} -> {estimatedVal:.3f}")
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
            engine_short = truncate_string(engineIdentifier, 50)
            replyText, _ = await invokeOpenAIEndpoint(
                httpSession, engineIdentifier, conversationPayload,
            )
            if not replyText:
                continue
            estimatedLikelihood, derivedRationale = interpretForecastResponse(replyText)
            return estimatedLikelihood, derivedRationale
        except Exception as engineFailure:
            error_msg = truncate_string(str(engineFailure), 80)
            print(f"[ENGINE/APP] {engine_short} encountered failure: {error_msg}")
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
    # Reset HTTP tracking and timing for this run
    HTTP_MANAGER.reset()
    TIME_TRACKER.reset()
    COST_TRACKER.reset()
    
    # Start timing the entire agent run
    agent_start = TIME_TRACKER.start("agent_main")
    
    questionIdentifier = event_data.get("event_id", "?")
    title = event_data.get("title", "")
    title_preview = truncate_string(title, 80)
    print(f"[BARD] segment={classifyIntoSegment(event_data)} title={title_preview}")

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
    gamma_cost = format_cost_dollars(get_gateway_cost(), 4)
    print(f"[BARD] gamma={gammaStatus} cost={gamma_cost}")

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

    # End timing for agent run
    TIME_TRACKER.end("agent_main", agent_start)
    
    # Track gamma cost
    COST_TRACKER.add_cost("gamma_api", get_gateway_cost())

    # Print statistics
    HTTP_MANAGER.print_stats()
    TIME_TRACKER.print_stats()
    COST_TRACKER.print_stats()

    return {
        "event_id": outcome["event_id"],
        "prediction": adjustedLikelihood,
        "reasoning": rationaleStr,
    }
