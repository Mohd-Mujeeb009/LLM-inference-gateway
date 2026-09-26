from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.config import Settings, get_settings
from app.core import (
    MockProvider,
    ProviderError,
    Router,
    SemanticCache,
    TokenBucketLimiter,
    UsageLedger,
)
from app.schemas import ChatRequest, ChatResponse, Choice, ChoiceMessage, Usage

REQUESTS = Counter("gateway_requests_total", "Gateway requests", ["model", "provider", "status"])
LATENCY = Histogram("gateway_request_latency_seconds", "Request latency", ["provider"])
CACHE_HITS = Counter("gateway_cache_hits_total", "Cache hits")
CACHE_MISSES = Counter("gateway_cache_misses_total", "Cache misses")
FALLBACKS = Counter("gateway_fallbacks_total", "Provider fallbacks", ["source", "destination"])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.router = Router(
        [
            MockProvider(
                "mock-primary", settings.primary_latency_ms, settings.primary_failure_rate
            ),
            MockProvider("mock-fallback", 10, 0),
        ]
    )
    app.state.limiter = TokenBucketLimiter(settings.requests_per_minute)
    app.state.cache = SemanticCache(settings.cache_ttl_seconds)
    app.state.usage = UsageLedger()
    yield


app = FastAPI(title="LLM Inference Gateway", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


async def authorize(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    key = authorization.removeprefix("Bearer ")
    if key not in settings.valid_api_keys:
        raise HTTPException(401, "invalid API key")
    return key


def token_count(text: str) -> int:
    return max(1, len(text.split()))


def response_payload(request: ChatRequest, answer: str) -> ChatResponse:
    prompt = sum(token_count(m.content) for m in request.messages)
    completion = token_count(answer)
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=[Choice(message=ChoiceMessage(content=answer))],
        usage=Usage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        ),
    )


@app.post("/v1/chat/completions")
async def chat(
    body: ChatRequest,
    request: Request,
    key: Annotated[str, Depends(authorize)],
) -> Response:
    allowed, remaining, retry_after = await request.app.state.limiter.check(key)
    if not allowed:
        return JSONResponse(
            {"error": {"message": "rate limit exceeded", "type": "rate_limit_error"}},
            429,
            headers={"Retry-After": str(retry_after), "X-RateLimit-Remaining": "0"},
        )
    common_headers = {"X-RateLimit-Remaining": str(remaining)}
    cached = request.app.state.cache.get(body)
    if cached is not None:
        CACHE_HITS.inc()
        payload = response_payload(body, cached)
        request.app.state.usage.add(
            model=body.model,
            provider="cache",
            prompt_tokens=payload.usage.prompt_tokens,
            completion_tokens=payload.usage.completion_tokens,
            cached=True,
        )
        headers = common_headers | {"X-Provider": "cache", "X-Cache": "HIT"}
        if body.stream:
            return StreamingResponse(
                stream_payload(payload), media_type="text/event-stream", headers=headers
            )
        return JSONResponse(payload.model_dump(), headers=headers)
    CACHE_MISSES.inc()
    started = time.perf_counter()
    try:
        answer, provider, fallback = await request.app.state.router.complete(body)
    except ProviderError as error:
        REQUESTS.labels(body.model, "none", "503").inc()
        raise HTTPException(503, str(error)) from error
    LATENCY.labels(provider).observe(time.perf_counter() - started)
    REQUESTS.labels(body.model, provider, "200").inc()
    if fallback:
        FALLBACKS.labels("mock-primary", provider).inc()
    request.app.state.cache.put(body, answer)
    payload = response_payload(body, answer)
    request.app.state.usage.add(
        model=body.model,
        provider=provider,
        prompt_tokens=payload.usage.prompt_tokens,
        completion_tokens=payload.usage.completion_tokens,
        cached=False,
    )
    headers = common_headers | {"X-Provider": provider, "X-Cache": "MISS"}
    if body.stream:
        return StreamingResponse(
            stream_payload(payload), media_type="text/event-stream", headers=headers
        )
    return JSONResponse(payload.model_dump(), headers=headers)


async def stream_payload(payload: ChatResponse) -> AsyncIterator[str]:
    for word in payload.choices[0].message.content.split():
        chunk = {
            "id": payload.id,
            "object": "chat.completion.chunk",
            "created": payload.created,
            "model": payload.model,
            "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/usage")
async def usage(request: Request, _: Annotated[str, Depends(authorize)]) -> dict[str, object]:
    return cast(UsageLedger, request.app.state.usage).summary()


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def ready() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
