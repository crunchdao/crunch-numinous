import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import aiohttp

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE_STATUSES = {429, 502, 503}
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BASE_DELAY = 1.0
_DEFAULT_MAX_DELAY = 30.0


async def with_retry(
    provider: str,
    call: Callable[[], Awaitable[T]],
    max_retries: int = _DEFAULT_MAX_RETRIES,
    base_delay: float = _DEFAULT_BASE_DELAY,
    max_delay: float = _DEFAULT_MAX_DELAY,
) -> T:
    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await call()
        except aiohttp.ClientResponseError as exc:
            last_exception = exc
            if exc.status not in _RETRYABLE_STATUSES or attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            retry_after = exc.headers.get("retry-after") if exc.headers else None
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
            logger.warning(
                "%s %d (attempt %d/%d), retrying in %.1fs",
                provider, exc.status, attempt + 1, max_retries + 1, delay,
            )
            await asyncio.sleep(delay)

    raise last_exception  # type: ignore[misc]
