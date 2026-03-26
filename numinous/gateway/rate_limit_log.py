import logging

import aiohttp

logger = logging.getLogger(__name__)

_RATE_LIMIT_HEADERS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
)


def log_rate_limit_headers(provider: str, response: aiohttp.ClientResponse) -> None:
    found = {h: response.headers[h] for h in _RATE_LIMIT_HEADERS if h in response.headers}
    if found:
        parts = " ".join("%s=%s" % (k, v) for k, v in found.items())
        logger.info("rate_limit provider=%s status=%d %s", provider, response.status, parts)
