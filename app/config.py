from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")
    api_keys: str = "gw_demo_key"
    requests_per_minute: int = 100
    cache_ttl_seconds: int = 300
    cache_similarity_threshold: float = 0.92
    provider_timeout_seconds: float = 10.0
    primary_failure_rate: float = 0.0
    primary_latency_ms: int = 20

    @property
    def valid_api_keys(self) -> set[str]:
        return {key.strip() for key in self.api_keys.split(",") if key.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
