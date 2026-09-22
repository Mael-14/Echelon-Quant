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
    redis_stream_signal_events: str = Field(default="signal-events")
    # Deriv authentication (optional)
    deriv_token: str | None = Field(default=None)
    deriv_account_id: str | None = Field(default=None)
    deriv_app_id: int | None = Field(default=None)
    # Symmetric key (base64 urlsafe) used to encrypt stored Deriv tokens (Fernet).
    # Required outside development (see _validate_deriv_token_key below); if empty
    # in development, tokens are stored plaintext.
    deriv_token_key: str | None = Field(default=None)

    # Deriv legacy v3 API. This is a different API from the Options one that
    # deriv_client.py targets, not a different URL for it: options are
    # fixed-payout and fixed-expiry, with no stop-loss price, no take-profit and
    # no trailing. Multiplier contracts have all three, which is what a strategy
    # emitting a stop distance and an R-multiple target can actually be mapped
    # onto. Both clients are kept; see deriv_v3_client.py.
    deriv_v3_ws_url: str = Field(default="wss://ws.derivws.com/websockets/v3")
    # demo | real. Guarded rather than free text because the difference between
    # them is whether a mistake costs real money.
    deriv_trade_mode: str = Field(default="demo")
    deriv_multiplier: int = Field(default=100)
    deriv_stake: float = Field(default=10.0)
    deriv_currency: str = Field(default="USD")

    # analysis-service. The cascade is configuration rather than a constant, and
    # the artifact records the cascade it was trained on: a disagreement makes
    # the service degraded rather than silently serving the wrong dynamics.
    analysis_model_path: str = Field(default="artifacts/scalp_m1.joblib")
    analysis_exec_frame: str = Field(default="1m")
    analysis_macro_frames: str = Field(default="4h")
    analysis_macro_window: int = Field(default=600)
    # One whole UTC session at M1. `momentum` is distance from the session VWAP,
    # which re-anchors to a partial day if the window is shorter.
    analysis_exec_window: int = Field(default=1440)
    analysis_symbols: str = Field(default="frxEURUSD")
    # An estimated probability carries error, so accepting everything with EV
    # marginally above zero accepts a population whose true mean is about zero.
    analysis_min_ev_r: float = Field(default=0.0)

    # execution-service.
    execution_max_open_positions: int = Field(default=1)
    execution_daily_loss_limit_r: float = Field(default=5.0)
    execution_risk_pct: float = Field(default=0.01)
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

    @field_validator("deriv_trade_mode")
    @classmethod
    def _trade_mode_is_known(cls, value: str) -> str:
        # A typo here must not silently fall through to live trading, so the
        # check is an allowlist rather than a truthiness test on "demo".
        normalised = value.strip().lower()
        if normalised not in {"demo", "real"}:
            raise ValueError(f"DERIV_TRADE_MODE must be 'demo' or 'real', got {value!r}")
        return normalised

    @field_validator("deriv_multiplier")
    @classmethod
    def _multiplier_is_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"DERIV_MULTIPLIER must be positive, got {value}")
        return value

    @field_validator("deriv_stake")
    @classmethod
    def _stake_is_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError(f"DERIV_STAKE must be positive, got {value}")
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_demo_trading(self) -> bool:
        """Whether orders go to a demo account.

        Read this rather than comparing ``deriv_trade_mode`` at each call site:
        execution refuses to place an order unless it is true, and one site
        spelling the check differently is how that guard gets lost.
        """
        return self.deriv_trade_mode == "demo"

    @property
    def deriv_v3_url(self) -> str:
        """The v3 socket URL with ``app_id`` attached.

        Deriv takes the app id as a query parameter rather than a header, and
        rejects the connection outright without it.
        """
        if self.deriv_app_id is None:
            raise ValueError(
                "DERIV_APP_ID must be set to connect to the Deriv v3 API "
                "(register an app at https://api.deriv.com to obtain one)"
            )
        separator = "&" if "?" in self.deriv_v3_ws_url else "?"
        return f"{self.deriv_v3_ws_url}{separator}app_id={self.deriv_app_id}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
