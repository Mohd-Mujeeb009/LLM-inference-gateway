from __future__ import annotations

import asyncio
import hashlib
import random
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypedDict

from app.schemas import ChatRequest


class ProviderError(RuntimeError):
    pass


class UsageEvent(TypedDict):
    at: int
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached: bool


class Provider(Protocol):
    name: str

    async def complete(self, request: ChatRequest) -> str: ...


class MockProvider:
    def __init__(self, name: str, latency_ms: int = 20, failure_rate: float = 0.0) -> None:
        self.name = name
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate

    async def complete(self, request: ChatRequest) -> str:
        await asyncio.sleep(self.latency_ms / 1000)
        if random.random() < self.failure_rate:
            raise ProviderError(f"{self.name} temporarily unavailable")
        prompt = request.messages[-1].content
        return f"[{self.name}] {prompt}"


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, threshold: int = 3, recovery_seconds: float = 10) -> None:
        self.threshold = threshold
        self.recovery_seconds = recovery_seconds
        self.failures = 0
        self.opened_at = 0.0
        self.state = CircuitState.CLOSED
        self._trial_active = False

    def allow(self) -> bool:
        if (
            self.state is CircuitState.OPEN
            and time.monotonic() - self.opened_at >= self.recovery_seconds
        ):
            self.state = CircuitState.HALF_OPEN
        if self.state is CircuitState.HALF_OPEN:
            if self._trial_active:
                return False
            self._trial_active = True
        return self.state is not CircuitState.OPEN

    def success(self) -> None:
        self.state, self.failures, self._trial_active = CircuitState.CLOSED, 0, False

    def failure(self) -> None:
        self._trial_active = False
        self.failures += 1
        if self.failures >= self.threshold or self.state is CircuitState.HALF_OPEN:
            self.state, self.opened_at = CircuitState.OPEN, time.monotonic()


class Router:
    def __init__(self, providers: list[Provider]) -> None:
        self.providers = providers
        self.breakers = {provider.name: CircuitBreaker() for provider in providers}

    async def complete(self, request: ChatRequest) -> tuple[str, str, bool]:
        last_error: Exception | None = None
        for index, provider in enumerate(self.providers):
            breaker = self.breakers[provider.name]
            if not breaker.allow():
                continue
            try:
                answer = await provider.complete(request)
                breaker.success()
                return answer, provider.name, index > 0
            except (ProviderError, TimeoutError) as error:
                last_error = error
                breaker.failure()
                if index + 1 < len(self.providers):
                    await asyncio.sleep(0.01 * (2**index))
        raise ProviderError("all providers unavailable") from last_error


@dataclass
class Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    def __init__(self, capacity: int, period_seconds: float = 60) -> None:
        self.capacity = capacity
        self.refill_rate = capacity / period_seconds
        self._buckets: dict[str, Bucket] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> tuple[bool, int, int]:
        async with self._lock:
            now = time.monotonic()
            bucket = self._buckets.setdefault(key, Bucket(float(self.capacity), now))
            bucket.tokens = min(
                self.capacity, bucket.tokens + (now - bucket.updated) * self.refill_rate
            )
            bucket.updated = now
            allowed = bucket.tokens >= 1
            if allowed:
                bucket.tokens -= 1
            wait = 0 if allowed else max(1, int((1 - bucket.tokens) / self.refill_rate) + 1)
            return allowed, int(bucket.tokens), wait


class SemanticCache:
    """Deterministic local cache seam; production swaps this for Redis vector search."""

    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, tuple[float, str]] = {}

    @staticmethod
    def key(request: ChatRequest) -> str:
        system = "|".join(m.content for m in request.messages if m.role == "system")
        user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        return hashlib.sha256(
            f"{request.model}|{system}|{user.strip().lower()}".encode()
        ).hexdigest()

    def get(self, request: ChatRequest) -> str | None:
        entry = self._entries.get(self.key(request))
        if not entry or entry[0] <= time.monotonic():
            return None
        return entry[1]

    def put(self, request: ChatRequest, value: str) -> None:
        if request.temperature <= 0.3:
            self._entries[self.key(request)] = (time.monotonic() + self.ttl_seconds, value)


class UsageLedger:
    def __init__(self) -> None:
        self.events: deque[UsageEvent] = deque(maxlen=10_000)

    def add(
        self, *, model: str, provider: str, prompt_tokens: int, completion_tokens: int, cached: bool
    ) -> None:
        self.events.append(
            {
                "at": int(time.time()),
                "model": model,
                "provider": provider,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cached": cached,
            }
        )

    def summary(self) -> dict[str, object]:
        return {
            "requests": len(self.events),
            "tokens": sum(int(e["total_tokens"]) for e in self.events),
            "cache_hits": sum(bool(e["cached"]) for e in self.events),
            "events": list(self.events),
        }
