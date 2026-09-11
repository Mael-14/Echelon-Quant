from __future__ import annotations

from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="Echelon Quant")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)

    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="echelon_quant")
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="postgres")

    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_stream_market_events: str = Field(default="market-events")
    redis_stream_bot_events: str = Field(default="bot-events")
    redis_stream_order_events: str = Field(default="order-events")
    redis_stream_position_events: str = Field(default="position-events")
    redis_stream_risk_events: str = Field(default="risk-events")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
