"""Central configuration loaded from environment variables (.env supported locally)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
except ImportError:  # Allows pure-domain tests to run before optional deps are installed.
    def load_dotenv() -> None:
        return None

load_dotenv()


#: The volatility indices the scalp profile runs on, and the whole of what the
#: Deriv venue trades under ADR-003.
#:
#: Measured on a day of real M1 history: R_50/75/100 and 1HZ50V/75V/100V
#: produced 42 trades a day between them, while R_10, R_25, 1HZ10V and 1HZ25V
#: produced *zero* -- their M1 ranges never clear the scalp technique's
#: volatility floor.
#:
#: Narrowed to R_75 alone by ADR-003. This costs roughly two thirds of that
#: measured trade flow and is deliberate: Deriv publishes 60 REST requests per
#: minute per token, and order flow -- a ``proposal`` per candidate plus a
#: ``buy`` and a ``sell`` per leg -- consumes that budget far faster than tick
#: subscriptions do. ``1HZ75V`` is a *distinct* synthetic with its own price
#: series, not a faster feed of this one, so it is not a free addition.
SCALP_MARKETS = ("R_75",)

#: Deriv's forex symbols, retained but **not subscribed**.
#:
#: The swing profile measured +0.120R over 521 backtested trades across these
#: nine. Live, they went 0 wins in 63 contracts: a Deriv multiplier contract
#: charges commission on notional, and on every one of these the entire stop
#: distance was worth less than the fee to open the position. ``MIN_STOP_TO_COST``
#: refuses them all, so subscribing them costs tick traffic and warm-up time for
#: candidates that can never reach an order. Forex moved to MT5 (ADR-003).
#:
#: Kept as a constant so a future Deriv-versus-MT5 comparison does not have to
#: reconstruct the notation.
DERIV_FOREX_MARKETS = (
    "frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxUSDCAD",
    "frxUSDCHF", "frxEURGBP", "frxEURJPY", "frxGBPJPY",
)

#: Deprecated alias. Prefer :data:`DERIV_FOREX_MARKETS`, which says which
#: broker's notation this is.
FOREX_MARKETS = DERIV_FOREX_MARKETS

#: What the MT5 venue trades, in MetaTrader's notation.
#:
#: Gold plus the two tightest-spread London majors and the two most liquid
#: Asian-session pairs, so both sessions are covered by instruments whose
#: five-minute ATR is comfortably larger than their spread. XAUUSD is not a
#: forex pair; it is here because its ATR-to-spread ratio is the best of the
#: set and it trades the London/New York overlap.
#:
#: These must match the symbol names the broker actually publishes -- Deriv MT5
#: suffixes some instruments -- so the adapter resolves each one against
#: ``symbols_get`` at connect time rather than trusting this tuple.
MT5_FOREX_MARKETS = ("XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD")


def _pairs_from_env(raw: str) -> list[str]:
    pairs = [p.strip() for p in raw.split(",") if p.strip()]
    # Keep every scalped market in the watchlist even when an existing
    # TRADING_PAIRS value predates them: a symbol the scalp profile routes to
    # but nobody subscribed would simply never trade.
    for symbol in SCALP_MARKETS:
        if symbol not in pairs:
            pairs.append(symbol)
    return pairs


def _symbols_from_env(name: str, default: tuple[str, ...]) -> list[str]:
    """A plain symbol list, with no force-appending.

    Unlike :func:`_pairs_from_env` this honours an empty value as "none",
    which is how the MT5 venue is switched off without code changes.
    """
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    return [s.strip() for s in raw.split(",") if s.strip()]


def _credential_from_env(name: str, default: str = "") -> str:
    """Normalize credentials pasted into Render without logging their values."""
    value = os.getenv(name, default).strip().strip('"').strip("'")
    if name == "DERIV_TOKEN" and value.lower().startswith("bearer "):
        value = value[7:].strip()
    return value


@dataclass(frozen=True)
class Settings:
    deriv_app_id: str = _credential_from_env("DERIV_APP_ID")
    deriv_token: str = _credential_from_env("DERIV_TOKEN", "YOUR_DERIV_TOKEN")
    deriv_account_id: str = _credential_from_env("DERIV_ACCOUNT_ID")

    telegram_bot_token: str = _credential_from_env(
        "TELEGRAM_BOT_TOKEN", os.getenv("Telegram_Bot_Token", "YOUR_BOT_TOKEN")
    )
    telegram_chat_id: str = _credential_from_env("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

    #: What the **Deriv** venue subscribes to. Forex is no longer in the
    #: default: it cannot clear the cost gate on multiplier contracts and now
    #: trades on MT5 instead (ADR-003).
    trading_pairs: list[str] = field(
        default_factory=lambda: _pairs_from_env(
            os.getenv("TRADING_PAIRS", ",".join(SCALP_MARKETS))
        )
    )

    max_risk_per_trade_pct: float = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "2.0"))
    simulated_balance: float = float(os.getenv("SIMULATED_BALANCE", "500.00"))

    demo_mode: bool = os.getenv("DEMO_MODE", "true").lower() == "true"
    sandbox_duration_days: int = int(os.getenv("SANDBOX_DURATION_DAYS", "7"))
    sandbox_state_path: str = os.getenv("SANDBOX_STATE_PATH", "state/sandbox.json")
    # Render Key Value connection string.  Without it, state lives in the
    # container filesystem and is lost on every deploy.
    state_store_url: str = _credential_from_env("STATE_STORE_URL") or _credential_from_env("REDIS_URL")
    shadow_enabled: bool = os.getenv("SHADOW_BOOK_ENABLED", "true").lower() == "true"
    shadow_interval_minutes: int = int(os.getenv("SHADOW_INTERVAL_MINUTES", "60"))
    market_memory_db_path: str = os.getenv("MARKET_MEMORY_DB_PATH", "state/market_memory.sqlite3")
    token_expires_at: str = os.getenv("DERIV_TOKEN_EXPIRES_AT", "")
    news_feed_url: str = os.getenv("NEWS_FEED_URL", "https://nfs.faireconomy.media/ff_calendar_thisweek.json")
    news_refresh_minutes: int = int(os.getenv("NEWS_REFRESH_MINUTES", "30"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    require_full_mtf: bool = os.getenv("REQUIRE_FULL_MTF", "true").lower() == "true"
    # Comma-separated technique names, or empty for every registered one.
    enabled_setups: str = os.getenv("ENABLED_SETUPS", "")
    # Symbols traded on the M1 scalp profile; everything else runs the swing
    # chain. The volatility indices are the default because they price
    # continuously and are not halted by the macro calendar.
    scalp_symbols: str = os.getenv("SCALP_SYMBOLS", ",".join(SCALP_MARKETS))
    # Risk per trade steps down as the balance grows: "floor:percent" pairs.
    risk_tiers: str = os.getenv("RISK_TIERS", "0:2,1000:1.5,10000:1,50000:0.75,100000:0.5")
    # Hard ceiling above the ladder, so a mis-typed tier cannot stake the account.
    risk_ceiling_pct: float = float(os.getenv("RISK_CEILING_PCT", "2.0"))
    # Bank half of any gain beyond this percentage of the working balance.
    profit_trigger_pct: float = float(os.getenv("PROFIT_TRIGGER_PCT", "20.0"))
    profit_reserve_fraction: float = float(os.getenv("PROFIT_RESERVE_FRACTION", "0.5"))
    # Realised loss over a rolling 24h window that halts trading outright. Per
    # trade risk bounds one position; nothing else bounds a losing streak.
    max_daily_drawdown_pct: float = float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "5.0"))
    drawdown_lockout_hours: float = float(os.getenv("DRAWDOWN_LOCKOUT_HOURS", "24.0"))
    # Cancel an entry when the quoted spread exceeds this multiple of the
    # symbol's recent average. All six traded symbols quote bid/ask, so a
    # missing spread means a broken feed and blocks by default.
    spread_max_multiple: float = float(os.getenv("SPREAD_MAX_MULTIPLE", "1.5"))
    spread_window: int = int(os.getenv("SPREAD_WINDOW", "20"))
    require_spread_data: bool = os.getenv("REQUIRE_SPREAD_DATA", "true").lower() == "true"
    use_authenticated_ws: bool = os.getenv("DERIV_USE_AUTHENTICATED_WS", "true").lower() == "true"
    deriv_currency: str = os.getenv("DERIV_CURRENCY", "USD")
    deriv_multiplier: int = int(os.getenv("DERIV_MULTIPLIER", "100"))
    # How long a position may stay open before it is sold at market, per
    # profile. The defaults are the measured optima (see
    # execution/exit_policy.py); 0 disables the clock for that profile, which
    # leaves only the stop and the target -- the behaviour that held scalps for
    # twenty-six minutes.
    # Smallest stake the broker will price for one multiplier contract. A
    # computed stake below it is refused outright rather than sent to be
    # rejected: at 2% a $5 account cannot fund one contract, which is a fact
    # about the account, not a transient error.
    deriv_min_stake: float = float(os.getenv("DERIV_MIN_STAKE", "1.0"))
    # Simultaneous open markets by balance, "floor:count". Rises with equity,
    # the opposite of the risk ladder: a larger account can carry more
    # positions going wrong at once. Total exposure is this times the risk
    # percentage, so raising both multiplies.
    concurrency_tiers: str = os.getenv("CONCURRENCY_TIERS", "0:1,50:2,100:3,500:4,1000:6,10000:8")
    # Hard cap above the ladder, so a mis-typed tier cannot open every market.
    # This is **per venue**: the concurrency ladder lives on VenueRuntime
    # because it is a function of one account's balance.
    max_concurrent_positions: int = int(os.getenv("MAX_CONCURRENT_POSITIONS", "8"))
    # Ceiling on open positions across *every* venue at once. Two accounts at
    # the per-venue cap is sixteen simultaneous positions, and gold, five
    # USD-quoted pairs and a volatility index are not independent bets. ADR-002
    # deferred this; ADR-003 decides it, because there are now two accounts.
    max_portfolio_positions: int = int(os.getenv("MAX_PORTFOLIO_POSITIONS", "6"))
    # A stop worth less than this many times the contract commission is not
    # a trade, it is a fee with a price attached. Measured over 118 live
    # contracts: every symbol below 1.2x won zero of 63; every symbol above 3x
    # won some. See docs/ARCHITECTURE.md.
    min_stop_to_cost: float = float(os.getenv("MIN_STOP_TO_COST", "3.0"))
    # How many contracts one signal may be split into. Each leg takes profit at
    # its own level; they share one stop, and the total stake is unchanged.
    max_order_legs: int = int(os.getenv("MAX_ORDER_LEGS", "3"))
    scalp_max_hold_seconds: float = float(os.getenv("SCALP_MAX_HOLD_SECONDS", "0") or 0)
    swing_max_hold_seconds: float = float(os.getenv("SWING_MAX_HOLD_SECONDS", "0") or 0)

    # --- MetaTrader 5 venue (ADR-003) -------------------------------------
    # Off by default. The MetaTrader5 package is a Windows-only binary wheel
    # driving a desktop terminal over localhost IPC, so the venue can only
    # exist where both are present; everywhere else -- CI, a Linux checkout,
    # a developer without the terminal -- it must stay absent rather than
    # fail at import.
    mt5_enabled: bool = os.getenv("MT5_ENABLED", "false").lower() == "true"
    mt5_symbols: list[str] = field(
        default_factory=lambda: _symbols_from_env("MT5_SYMBOLS", MT5_FOREX_MARKETS)
    )
    # Blank means "attach to whatever terminal is already logged in", which is
    # the normal desktop case. The EC2 deployment sets all four so a reboot
    # restores the session without a human at the console.
    mt5_login: int = int(os.getenv("MT5_LOGIN", "0") or 0)
    mt5_password: str = _credential_from_env("MT5_PASSWORD")
    mt5_server: str = _credential_from_env("MT5_SERVER")
    mt5_terminal_path: str = os.getenv("MT5_TERMINAL_PATH", "")
    # Maximum slippage tolerated on a market order, in points. MT5 rejects the
    # order outright rather than filling worse than this.
    mt5_deviation_points: int = int(os.getenv("MT5_DEVIATION_POINTS", "20"))
    # Stamped on every order so this agent's positions are distinguishable
    # from anything placed by hand in the same terminal. ``adopt_open_positions``
    # uses it to avoid adopting a manual trade.
    mt5_magic: int = int(os.getenv("MT5_MAGIC", "770075"))
    # Per-lot round-turn commission, in account currency. Deriv MT5 quotes
    # spread-only on these symbols, but a broker that charges commission makes
    # the cost gate wrong by exactly this much, so it is configurable rather
    # than assumed zero.
    mt5_commission_per_lot: float = float(os.getenv("MT5_COMMISSION_PER_LOT", "0.0"))
    # How often the tick poller asks the terminal for a new quote. MT5 exposes
    # no async stream, so this is the resolution of the whole price path.
    mt5_poll_interval_seconds: float = float(os.getenv("MT5_POLL_INTERVAL_SECONDS", "0.25"))

    @property
    def deriv_ws_uri(self) -> str:
        # Public market-data WebSocket; account authentication is upgraded to
        # the current PAT/OTP URL by DerivClient when account_id is configured.
        return "wss://api.derivws.com/trading/v1/options/ws/public"


settings = Settings()
