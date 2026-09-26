# Architecture

The FastAPI edge authenticates bearer keys, applies a token bucket, checks a deterministic cache, and sends misses through an ordered provider router. Each provider owns a circuit breaker. Retryable failures move traffic to the next provider; successful low-temperature responses enter the cache. Usage and Prometheus measurements are recorded after completion.

The runnable local profile deliberately uses mock providers and in-memory seams, so it costs nothing. Production adapters implement the same provider protocol; Redis Lua replaces the local limiter, Redis vector search replaces the exact-key cache, and PostgreSQL replaces the in-process usage ledger.
