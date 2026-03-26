import time
from threading import Lock


class TokenBucket:
    __slots__ = ("rate", "burst", "_tokens", "_last_refill", "_lock")

    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_refill = now
            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False


class RateLimiter:
    def __init__(self, rate: float, burst: int, max_buckets: int = 10_000):
        self.rate = rate
        self.burst = burst
        self._max_buckets = max_buckets
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_buckets:
                    oldest_key = next(iter(self._buckets))
                    del self._buckets[oldest_key]
                bucket = TokenBucket(self.rate, self.burst)
                self._buckets[key] = bucket
        return bucket.allow()
