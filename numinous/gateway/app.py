import logging
import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, http_exception_handler, status

from numinous.gateway.rate_limit import RateLimiter

from numinous.gateway.cache import cached_gateway_call
from numinous.gateway.error_handler import handle_provider_errors
from numinous.gateway.providers.chutes import ChutesClient
from numinous.gateway.providers.desearch import DesearchClient
from numinous.gateway.providers.numinous_indicia import NuminousIndiciaClient
from numinous.gateway.providers.openai import OpenAIClient
from numinous.gateway.providers.openrouter import OpenRouterClient
from numinous.gateway.providers.perplexity import PerplexityClient
from numinous.gateway.providers.vericore import VericoreClient
from numinous.gateway.models import numinous_client as models
from numinous.gateway.models.chutes import ChuteStatus
from numinous.gateway.models.chutes import calculate_cost as calculate_chutes_cost
from numinous.gateway.models.desearch import DesearchEndpoint
from numinous.gateway.models.desearch import calculate_cost as calculate_desearch_cost
from numinous.gateway.models.numinous_indicia import (
    calculate_cost as calculate_numinous_indicia_cost,
)
from numinous.gateway.models.openai import calculate_cost as calculate_openai_cost
from numinous.gateway.models.openrouter import calculate_cost as calculate_openrouter_cost
from numinous.gateway.models.perplexity import calculate_cost as calculate_perplexity_cost
from numinous.gateway.models.vericore import calculate_cost as calculate_vericore_cost

logger = logging.getLogger(__name__)

def _load_allow_list(env_var: str) -> set[str] | None:
    raw = os.getenv(env_var)
    if not raw or raw.strip() == "*":
        return None
    return {m.strip() for m in raw.split(",") if m.strip()}


ALLOWED_MODELS: dict[str, set[str] | None] = {
    "chutes": _load_allow_list("ALLOWED_MODELS_CHUTES"),
    "openai": _load_allow_list("ALLOWED_MODELS_OPENAI"),
    "perplexity": _load_allow_list("ALLOWED_MODELS_PERPLEXITY"),
    "openrouter": _load_allow_list("ALLOWED_MODELS_OPENROUTER"),
    "desearch": _load_allow_list("ALLOWED_MODELS_DESEARCH"),
}

ALLOWED_ENDPOINTS: dict[str, set[str] | None] = {
    "desearch": _load_allow_list("ALLOWED_ENDPOINTS_DESEARCH"),
    "vericore": _load_allow_list("ALLOWED_ENDPOINTS_VERICORE"),
}


def _check_model_allowed(provider: str, model: str) -> None:
    allowed = ALLOWED_MODELS.get(provider)
    if allowed is None:
        return
    if model not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Model '%s' is not in the %s allow list" % (model, provider),
        )


def _check_endpoint_allowed(provider: str, endpoint: str) -> None:
    allowed = ALLOWED_ENDPOINTS.get(provider)
    if allowed is None:
        return
    if endpoint not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Endpoint '%s' is not in the %s allow list" % (endpoint, provider),
        )


RATE_LIMIT_PER_SECOND = float(os.getenv("RATE_LIMIT_PER_SECOND", "0"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "0"))
_rate_limiter: RateLimiter | None = None
if RATE_LIMIT_PER_SECOND > 0 and RATE_LIMIT_BURST > 0:
    _rate_limiter = RateLimiter(rate=RATE_LIMIT_PER_SECOND, burst=RATE_LIMIT_BURST)
    logger.info(
        "Rate limiting enabled: %s req/s, burst %s",
        RATE_LIMIT_PER_SECOND,
        RATE_LIMIT_BURST,
    )

RATE_LIMIT_EXEMPT_IPS: set[str] = set()
_exempt_raw = os.getenv("RATE_LIMIT_EXEMPT_IPS", "")
if _exempt_raw:
    RATE_LIMIT_EXEMPT_IPS = {ip.strip() for ip in _exempt_raw.split(",") if ip.strip()}

GATEWAY_DEBUG = os.getenv("GATEWAY_DEBUG", "").strip() == "1"

if GATEWAY_DEBUG:
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

app = FastAPI(title="Numinous API Gateway")
gateway_router = APIRouter(prefix="/api/gateway")


@app.middleware("http")
async def debug_logging_middleware(request: Request, call_next):
    if GATEWAY_DEBUG and request.url.path.startswith("/api/gateway/"):
        body = await request.body()
        headers = dict(request.headers)
        logger.debug(
            ">>> %s %s headers=%s body=%s",
            request.method,
            request.url.path,
            headers,
            body.decode("utf-8", errors="replace")[:2000] if body else "(empty)",
        )
        response = await call_next(request)
        logger.debug(
            "<<< %s %s status=%s",
            request.method,
            request.url.path,
            response.status_code,
        )
        return response
    return await call_next(request)


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    logger.error(
        "%s:%s %s %s -> %d: %s",
        request.client.host,
        request.client.port,
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )

    return await http_exception_handler(request, exc)  # still returns the default response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if _rate_limiter is not None and request.url.path.startswith("/api/gateway/"):
        client_ip = request.client.host if request.client else "unknown"
        if client_ip not in RATE_LIMIT_EXEMPT_IPS and not _rate_limiter.allow(client_ip):
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
            )
    return await call_next(request)


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "API Gateway"}


@app.get("/api/allowed")
async def list_allow_lists():
    return {
        "models": {
            provider: sorted(m) if m is not None else "*"
            for provider, m in ALLOWED_MODELS.items()
        },
        "endpoints": {
            provider: sorted(e) if e is not None else "*"
            for provider, e in ALLOWED_ENDPOINTS.items()
        },
    }


@gateway_router.post("/chutes/chat/completions", response_model=models.GatewayChutesCompletion)
@cached_gateway_call
@handle_provider_errors("Chutes")
async def chutes_chat_completion(request: models.ChutesInferenceRequest) -> models.ChutesCompletion:
    _check_model_allowed("chutes", request.model)
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="CHUTES_API_KEY not configured",
        )

    client = ChutesClient(api_key=api_key)
    messages = [msg.model_dump() for msg in request.messages]
    result = await client.chat_completion(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        tools=request.tools,
        tool_choice=request.tool_choice,
        **(request.model_extra or {}),
    )

    return models.GatewayChutesCompletion(
        **result.model_dump(), cost=calculate_chutes_cost(request.model, result)
    )


@gateway_router.get("/chutes/status", response_model=list[ChuteStatus])
@handle_provider_errors("Chutes")
async def get_chutes_status() -> list[ChuteStatus]:
    api_key = os.getenv("CHUTES_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="CHUTES_API_KEY not configured",
        )

    client = ChutesClient(api_key=api_key)
    return await client.get_chutes_status()


@gateway_router.post("/desearch/ai/search", response_model=models.GatewayDesearchAISearchResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_ai_search(
    request: models.DesearchAISearchRequest,
) -> models.GatewayDesearchAISearchResponse:
    _check_endpoint_allowed("desearch", "ai/search")
    _check_model_allowed("desearch", request.model)
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    result = await client.ai_search(
        prompt=request.prompt,
        model=request.model,
        tools=request.tools,
        date_filter=request.date_filter,
        result_type=request.result_type,
        system_message=request.system_message,
        count=request.count,
    )

    return models.GatewayDesearchAISearchResponse(
        **result.model_dump(),
        cost=calculate_desearch_cost(DesearchEndpoint.AI_SEARCH, request.model),
    )


@gateway_router.post("/desearch/ai/links", response_model=models.GatewayDesearchWebLinksResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_web_links_search(
    request: models.DesearchWebLinksRequest,
) -> models.GatewayDesearchWebLinksResponse:
    _check_endpoint_allowed("desearch", "ai/links")
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    result = await client.web_links_search(
        prompt=request.prompt, model=request.model, tools=request.tools, count=request.count
    )
    return models.GatewayDesearchWebLinksResponse(
        **result.model_dump(),
        cost=calculate_desearch_cost(DesearchEndpoint.AI_WEB_SEARCH, request.model),
    )


@gateway_router.post("/desearch/web/search", response_model=models.GatewayDesearchWebSearchResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_web_search(
    request: models.DesearchWebSearchRequest,
) -> models.GatewayDesearchWebSearchResponse:
    _check_endpoint_allowed("desearch", "web/search")
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    result = await client.web_search(
        query=request.query, num_results=request.num, start=request.start
    )
    return models.GatewayDesearchWebSearchResponse(
        **result.model_dump(),
        cost=calculate_desearch_cost(DesearchEndpoint.WEB_SEARCH),
    )


@gateway_router.post("/desearch/web/crawl", response_model=models.GatewayDesearchWebCrawlResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_web_crawl(
    request: models.DesearchWebCrawlRequest,
) -> models.GatewayDesearchWebCrawlResponse:
    _check_endpoint_allowed("desearch", "web/crawl")
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    result = await client.web_crawl(url=request.url)

    return models.GatewayDesearchWebCrawlResponse(
        **result.model_dump(),
        cost=calculate_desearch_cost(DesearchEndpoint.WEB_CRAWL),
    )


@gateway_router.post("/desearch/x/search", response_model=models.GatewayDesearchXSearchResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_x_search(
    request: models.DesearchXSearchRequest,
) -> models.GatewayDesearchXSearchResponse:
    _check_endpoint_allowed("desearch", "x/search")
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    result = await client.x_search(
        query=request.query,
        sort=request.sort,
        user=request.user,
        start_date=request.start_date,
        end_date=request.end_date,
        lang=request.lang,
        verified=request.verified,
        blue_verified=request.blue_verified,
        is_quote=request.is_quote,
        is_video=request.is_video,
        is_image=request.is_image,
        min_retweets=request.min_retweets,
        min_replies=request.min_replies,
        min_likes=request.min_likes,
        count=request.count,
    )

    return models.GatewayDesearchXSearchResponse(
        posts=result,
        cost=calculate_desearch_cost(DesearchEndpoint.X_SEARCH),
    )


@gateway_router.post("/desearch/x/post", response_model=models.GatewayDesearchXPostResponse)
@cached_gateway_call
@handle_provider_errors("Desearch")
async def desearch_x_post(
    request: models.DesearchXPostRequest,
) -> models.GatewayDesearchXPostResponse:
    _check_endpoint_allowed("desearch", "x/post")
    api_key = os.getenv("DESEARCH_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="DESEARCH_API_KEY not configured",
        )

    client = DesearchClient(api_key=api_key)
    result = await client.fetch_x_post(post_id=request.post_id)

    return models.GatewayDesearchXPostResponse(
        **result.model_dump(),
        cost=calculate_desearch_cost(DesearchEndpoint.FETCH_X_POST),
    )


@gateway_router.post("/openai/responses", response_model=models.GatewayOpenAIResponse)
@cached_gateway_call
@handle_provider_errors("OpenAI")
async def openai_create_response(
    request: models.OpenAIInferenceRequest,
) -> models.GatewayOpenAIResponse:
    _check_model_allowed("openai", request.model)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OPENAI_API_KEY not configured",
        )

    client = OpenAIClient(api_key=api_key)
    input_messages = [msg.model_dump(exclude_none=True) for msg in request.input]
    result = await client.create_response(
        model=request.model,
        input=input_messages,
        temperature=request.temperature,
        max_output_tokens=request.max_output_tokens,
        tools=request.tools,
        tool_choice=request.tool_choice,
        instructions=request.instructions,
        **(request.model_extra or {}),
    )

    return models.GatewayOpenAIResponse(
        **result.model_dump(), cost=calculate_openai_cost(request.model, result)
    )


@gateway_router.post("/perplexity/chat/completions")
@cached_gateway_call
@handle_provider_errors("Perplexity")
async def perplexity_chat_completion(request: models.PerplexityInferenceRequest):
    _check_model_allowed("perplexity", request.model)
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="PERPLEXITY_API_KEY not configured",
        )

    client = PerplexityClient(api_key=api_key)
    messages = [msg.model_dump() for msg in request.messages]
    result = await client.chat_completion(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        search_recency_filter=request.search_recency_filter,
        **(request.model_extra or {}),
    )

    return models.GatewayPerplexityCompletion(
        **result.model_dump(), cost=calculate_perplexity_cost(request.model, result)
    )


@gateway_router.post("/vericore/calculate-rating", response_model=models.GatewayVericoreResponse)
@cached_gateway_call
@handle_provider_errors("Vericore")
async def vericore_calculate_rating(
    request: models.VericoreCalculateRatingRequest,
) -> models.GatewayVericoreResponse:
    _check_endpoint_allowed("vericore", "calculate-rating")
    api_key = os.getenv("VERICORE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="VERICORE_API_KEY not configured",
        )

    client = VericoreClient(api_key=api_key)
    result = await client.calculate_rating(
        statement=request.statement,
        generate_preview=request.generate_preview,
    )

    return models.GatewayVericoreResponse(**result.model_dump(), cost=calculate_vericore_cost())


@gateway_router.post(
    "/openrouter/chat/completions", response_model=models.GatewayOpenRouterCompletion
)
@cached_gateway_call
@handle_provider_errors("OpenRouter")
async def openrouter_chat_completion(
    request: models.OpenRouterInferenceRequest,
) -> models.GatewayOpenRouterCompletion:
    _check_model_allowed("openrouter", request.model)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OPENROUTER_API_KEY not configured",
        )

    client = OpenRouterClient(api_key=api_key)
    messages = [msg.model_dump() for msg in request.messages]
    result = await client.chat_completion(
        model=request.model,
        messages=messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        tools=request.tools,
        tool_choice=request.tool_choice,
        **(request.model_extra or {}),
    )

    return models.GatewayOpenRouterCompletion(
        **result.model_dump(), cost=calculate_openrouter_cost(result)
    )


@gateway_router.post(
    "/numinous-indicia/x-osint",
    response_model=models.GatewayNuminousIndiciaSignalsResponse,
)
@cached_gateway_call
@handle_provider_errors("NuminousIndicia")
async def numinous_indicia_x_osint(
    request: models.NuminousIndiciaXOsintRequest,
) -> models.GatewayNuminousIndiciaSignalsResponse:
    client = NuminousIndiciaClient()
    signals = await client.x_osint(account=request.account, limit=request.limit)

    return models.GatewayNuminousIndiciaSignalsResponse(
        signals=signals, cost=float(calculate_numinous_indicia_cost())
    )


@gateway_router.post(
    "/numinous-indicia/liveuamap",
    response_model=models.GatewayNuminousIndiciaSignalsResponse,
)
@cached_gateway_call
@handle_provider_errors("NuminousIndicia")
async def numinous_indicia_liveuamap(
    request: models.NuminousIndiciaLiveuamapRequest,
) -> models.GatewayNuminousIndiciaSignalsResponse:
    client = NuminousIndiciaClient()
    signals = await client.liveuamap(region=request.region, limit=request.limit)

    return models.GatewayNuminousIndiciaSignalsResponse(
        signals=signals, cost=float(calculate_numinous_indicia_cost())
    )


app.include_router(gateway_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
