import asyncio

from fastapi.testclient import TestClient

from app.core import (
    CircuitBreaker,
    CircuitState,
    MockProvider,
    Router,
    SemanticCache,
    TokenBucketLimiter,
)
from app.main import app
from app.schemas import ChatRequest, Message


def test_chat_cache_and_auth() -> None:
    with TestClient(app) as client:
        body = {"model": "llama-3.3-70b", "messages": [{"role": "user", "content": "hello"}]}
        assert client.post("/v1/chat/completions", json=body).status_code == 401
        first = client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer gw_demo_key"}
        )
        second = client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer gw_demo_key"}
        )
        assert first.status_code == 200 and first.headers["X-Cache"] == "MISS"
        assert second.status_code == 200 and second.headers["X-Cache"] == "HIT"


def test_openai_stream_contract() -> None:
    with TestClient(app) as client:
        body = {
            "model": "llama-3.3-70b",
            "messages": [{"role": "user", "content": "stream me"}],
            "stream": True,
        }
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json=body,
            headers={"Authorization": "Bearer gw_demo_key"},
        ) as response:
            text = "".join(response.iter_text())
        assert response.status_code == 200 and "data: [DONE]" in text


def test_breaker_transitions() -> None:
    breaker = CircuitBreaker(threshold=2, recovery_seconds=0)
    breaker.failure()
    breaker.failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.allow() and breaker.state is CircuitState.HALF_OPEN
    breaker.success()
    assert breaker.state is CircuitState.CLOSED


def test_fallback() -> None:
    async def run() -> None:
        router = Router([MockProvider("bad", 0, 1), MockProvider("good", 0, 0)])
        body = ChatRequest(model="m", messages=[Message(role="user", content="x")])
        answer, provider, fallback = await router.complete(body)
        assert answer and provider == "good" and fallback

    asyncio.run(run())


def test_limiter() -> None:
    async def run() -> None:
        limiter = TokenBucketLimiter(2, 60)
        assert (await limiter.check("k"))[0]
        assert (await limiter.check("k"))[0]
        assert not (await limiter.check("k"))[0]

    asyncio.run(run())


def test_temperature_skips_cache() -> None:
    cache = SemanticCache(60)
    body = ChatRequest(model="m", temperature=0.8, messages=[Message(role="user", content="x")])
    cache.put(body, "answer")
    assert cache.get(body) is None
