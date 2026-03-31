import functools
import logging
from typing import Any, Callable

import aiohttp
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


async def raise_for_status(response: aiohttp.ClientResponse) -> None:
    try:
        response.raise_for_status()
    except aiohttp.ClientResponseError as error:
        response_body = await response.text()
        error.body = response_body

        raise error


def handle_provider_errors(provider: str) -> Callable[[Callable], Callable]:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except Exception as e:
                error_message = f"{provider} API error: {str(e)}"

                if isinstance(e, aiohttp.ClientResponseError):
                    response_body = getattr(e, "body", None)
                    if response_body:
                        error_message += f": {response_body}"
                    else:
                        error_message += ": (no response body)"

                    status_code = e.status
                else:
                    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

                logger.error(error_message)

                raise HTTPException(
                    status_code=status_code,
                    detail=error_message,
                )

        return wrapper

    return decorator
