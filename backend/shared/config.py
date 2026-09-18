from __future__ import annotations

from functools import lru_cache

from cryptography.fernet import Fernet
from pydantic import Field, computed_field, field_validator, model_validator
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
    # Deriv authentication (optional)
    deriv_token: str | None = Field(default=None)
    deriv_account_id: str | None = Field(default=None)
    deriv_app_id: int | None = Field(default=None)
    # Symmetric key (base64 urlsafe) used to encrypt stored Deriv tokens (Fernet).
    # Required outside development (see _validate_deriv_token_key below); if empty
    # in development, tokens are stored plaintext.
    deriv_token_key: str | None = Field(default=None)
    # Base URL api-gateway proxies bot lifecycle requests to.
    bot_service_url: str = Field(default="http://bot-service:8000")

    @field_validator("deriv_app_id", mode="before")
    @classmethod
    def _blank_app_id_is_unset(cls, value: object) -> object:
        # An empty env var (e.g. `DERIV_APP_ID=`) means "not configured", not the
        # literal string "" - which would otherwise fail int parsing.
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def _validate_deriv_token_key(self) -> "Settings":
        if self.deriv_token_key:
            try:
                Fernet(self.deriv_token_key.encode("utf-8"))
            except Exception as exc:
                raise ValueError(
                    "DERIV_TOKEN_KEY is set but is not a valid Fernet key (generate one with "
                    'python -c "from cryptography.fernet import Fernet; '
                    'print(Fernet.generate_key().decode())")'
                ) from exc
        elif self.environment.lower() != "development":
            raise ValueError(
                "DERIV_TOKEN_KEY must be set when ENVIRONMENT is not 'development' - "
                "Deriv tokens must not be stored in plaintext outside local development"
            )
        return self

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
        extra="ignore",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
