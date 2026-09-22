"""The v3 settings guard the difference between demo and real money."""

from __future__ import annotations

import pytest

from backend.shared.config import Settings


def test_trade_mode_defaults_to_demo() -> None:
    """Nobody should reach a live account by forgetting to set something."""
    settings = Settings(environment="development")

    assert settings.deriv_trade_mode == "demo"
    assert settings.is_demo_trading is True


@pytest.mark.parametrize("value", ["DEMO", " demo ", "Demo"])
def test_trade_mode_is_normalised(value: str) -> None:
    assert Settings(environment="development", deriv_trade_mode=value).is_demo_trading is True


def test_real_mode_is_allowed_but_not_demo() -> None:
    settings = Settings(environment="development", deriv_trade_mode="real")

    assert settings.deriv_trade_mode == "real"
    assert settings.is_demo_trading is False


@pytest.mark.parametrize("value", ["live", "paper", "", "demo-ish"])
def test_an_unknown_trade_mode_is_rejected(value: str) -> None:
    """A typo must fail loudly, not fall through to whatever is not 'demo'."""
    with pytest.raises(ValueError, match="DERIV_TRADE_MODE"):
        Settings(environment="development", deriv_trade_mode=value)


@pytest.mark.parametrize("multiplier", [0, -1])
def test_multiplier_must_be_positive(multiplier: int) -> None:
    with pytest.raises(ValueError, match="DERIV_MULTIPLIER"):
        Settings(environment="development", deriv_multiplier=multiplier)


@pytest.mark.parametrize("stake", [0.0, -5.0])
def test_stake_must_be_positive(stake: float) -> None:
    with pytest.raises(ValueError, match="DERIV_STAKE"):
        Settings(environment="development", deriv_stake=stake)


def test_v3_url_carries_the_app_id() -> None:
    settings = Settings(environment="development", deriv_app_id=1234)

    assert settings.deriv_v3_url == "wss://ws.derivws.com/websockets/v3?app_id=1234"


def test_v3_url_without_an_app_id_says_what_to_do() -> None:
    settings = Settings(environment="development", deriv_app_id=None)

    with pytest.raises(ValueError, match="DERIV_APP_ID"):
        _ = settings.deriv_v3_url


def test_v3_url_appends_to_an_existing_query() -> None:
    settings = Settings(
        environment="development",
        deriv_app_id=7,
        deriv_v3_ws_url="wss://ws.derivws.com/websockets/v3?l=EN",
    )

    assert settings.deriv_v3_url.endswith("?l=EN&app_id=7")


def test_blank_app_id_env_var_is_still_unset() -> None:
    """DERIV_APP_ID= in a .env means "not configured", not the empty string."""
    assert Settings(environment="development", deriv_app_id="").deriv_app_id is None
