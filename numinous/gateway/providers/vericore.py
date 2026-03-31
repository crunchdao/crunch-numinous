import aiohttp

from numinous.gateway.models.vericore import VericoreResponse
from numinous.gateway.rate_limit_log import log_rate_limit_headers
from numinous.gateway.retry import with_retry
from numinous.gateway.error_handler import raise_for_status


class VericoreClient:
    __api_key: str
    __base_url: str
    __timeout: aiohttp.ClientTimeout
    __headers: dict[str, str]

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Vericore API key is not set")

        self.__api_key = api_key
        self.__base_url = "https://api.verify.vericore.ai"
        self.__timeout = aiohttp.ClientTimeout(total=120)
        self.__headers = {
            "Authorization": f"api-key {self.__api_key}",
            "Content-Type": "application/json",
        }

    async def calculate_rating(
        self, statement: str, generate_preview: bool = False
    ) -> VericoreResponse:
        body = {
            "statement": statement,
            "generate_preview": str(generate_preview).lower(),
        }

        url = f"{self.__base_url}/calculate-rating/v2"

        async def _call() -> VericoreResponse:
            async with aiohttp.ClientSession(timeout=self.__timeout, headers=self.__headers) as session:
                async with session.post(url, json=body) as response:
                    log_rate_limit_headers("vericore", response)
                    await raise_for_status(response)
                    data = await response.json()
                    return VericoreResponse.model_validate(data)

        return await with_retry("vericore", _call)
