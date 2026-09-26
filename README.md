# LLM Inference Gateway

[![CI](https://github.com/Mohd-Mujeeb009/LLM-inference-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohd-Mujeeb009/LLM-inference-gateway/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Mohd-Mujeeb009/LLM-inference-gateway/actions/workflows/codeql.yml/badge.svg)](https://github.com/Mohd-Mujeeb009/LLM-inference-gateway/actions/workflows/codeql.yml)

A runnable OpenAI-compatible FastAPI gateway focused on reliability: authentication, token-bucket quotas, caching, ordered provider fallback, per-provider circuit breakers, SSE streaming, usage accounting, Prometheus metrics, Docker, and CI.

## Quick start

```bash
docker compose up --build
python scripts/demo.py
```

Or use the OpenAI SDK by setting `base_url="http://localhost:8000/v1"` and `api_key="gw_demo_key"`. Endpoints are `POST /v1/chat/completions`, `GET /v1/usage`, `GET /healthz`, `GET /readyz`, and `GET /metrics`.

## Local development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
ruff check .
mypy app
pytest -q
uvicorn app.main:app --reload
```

Every outbound-provider seam is async. Circuit breakers isolate unhealthy providers and permit one half-open probe after cooldown. Only retryable provider errors trigger fallback. Cache identity includes model, system prompt, and last user message; responses with temperature above `0.3` are never cached.

## Production extension points

The default profile is intentionally free and deterministic: two mock providers, an in-memory token bucket/cache, and an in-process usage ledger. The repository includes Redis Stack and PostgreSQL containers plus provider configuration; production adapters should replace the seams with Redis Lua, Redis vector search, SQLAlchemy/Alembic persistence, and secret-backed Groq/OpenAI/Azure clients. See `docs/architecture.md` and `docs/azure-deploy.md`.
