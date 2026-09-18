# Fix & Bot Development Plan

This plan has two phases: **Phase 1** hardens what already exists (removes duplication, adds persistence, closes safety gaps) so the platform is stable to build on. **Phase 2** builds the actual trading bot — the strategy → risk → execution → position loop that today only exists as empty service stubs and shared schemas.

Each item lists: what to do, why, the files it touches, and how to verify it.

**Status:** 1.1–1.5 and 1.7 are done. 1.6 (rate limits & safety guards) is still open — skipped ahead to 1.7 per direction; revisit 1.6 next.

---

## Phase 1 — Fix existing issues

### 1.1 Remove duplicated bot logic between `api-gateway` and `bot-service` — ✅ done
- **Problem**: `backend/apps/api-gateway/app/main.py` has its own inline `bots_store` dict and CRUD/start/stop/pause endpoints, duplicating `backend/services/bot-service/app/service.py`'s `BotManager` (which has a real state-machine with legal transitions). The two stores can drift.
- **Fix**: Strip bot CRUD out of api-gateway; have it proxy `/api/v1/bots*` requests to `bot-service` over HTTP (or keep the routes but make api-gateway a thin client calling bot-service). Keep the Deriv token/OAuth endpoints in api-gateway — those don't belong in bot-service.
- **Files**: `backend/apps/api-gateway/app/main.py`, `backend/services/bot-service/app/main.py`.
- **Verify**: `tests/test_bot_service_foundation.py` still passes; add an api-gateway test that hits its bot routes and asserts they reach bot-service (use an HTTP mock/fixture).

### 1.2 Persist bot state (remove in-memory-only stores) — ✅ done
- **Problem**: Both `bots_store` (api-gateway) and `BotManager._bots` (bot-service) are plain dicts — a restart loses every bot and its state.
- **Fix**: Add a `bots` table (config JSONB, status JSONB, timestamps) and back `BotManager` with Postgres via `asyncpg`, following the connection-pool pattern already used in `market-data-service/app/pipeline.py` (`_ensure_pg`). Load bots into memory on startup, write through on every transition. (`owner_id`/multi-tenancy dropped from scope — no auth/user concept exists in the codebase yet.)
- **Files**: `backend/services/bot-service/app/service.py` (added `load_bot`), new `backend/services/bot-service/app/store.py` (`BotStore`), `backend/services/bot-service/app/main.py` (startup/shutdown hooks + `_persist` after every mutation).
- **Verify**: `tests/test_bot_service_persistence.py` — save/load round-trip, upsert-not-duplicate, and a simulated restart against a fake in-memory Postgres table.

### 1.3 Replace runtime `CREATE TABLE IF NOT EXISTS` with Alembic migrations — ✅ done
- **Problem**: `deriv_tokens` (api-gateway), `market_ticks` (market-data-service), and `bots` (bot-service) were all created ad-hoc at startup via raw SQL. `alembic` was already a dependency but unused.
- **Fix**: `backend/alembic/` (async env.py resolving the DB URL from `Settings`, so it stays in sync with each service's own connection config) + one migration (`0001_initial_schema.py`) creating all three tables with the exact same `IF NOT EXISTS` DDL the services used to run themselves, so it's safe against dev DBs already populated by the old path. Removed the ad-hoc `_create_table`/`_create_deriv_tokens_table` calls and the now-unneeded api-gateway startup hook (which also sped up its tests: no more real DB connection attempt per test). `make migrate` / `make migrate-down` / `make migration m="..."` added; CI's typecheck job now also covers `backend/alembic`; a `migrate` one-off compose service (`profiles: ["tools"]`) runs it against the docker-compose stack.
- **Verify**: `alembic upgrade head --sql` (offline mode, no DB needed) renders the exact expected DDL; online path confirmed to reach a real connection attempt via SQLAlchemy's async engine. Full test suite green after removing the ad-hoc DDL.
- **Not done** (optional, flagged for later): CI has no Postgres service container, so migrations aren't actually exercised in CI yet — the plan doc's own wording made this conditional ("if CI gets a DB service container").

### 1.4 Enforce Deriv token encryption — ✅ done
- **Problem**: `DERIV_TOKEN_KEY` was optional; if unset, tokens were stored **plaintext**. Decrypt failures in `_decrypt_token`/`pipeline.py` silently returned ciphertext instead of raising.
- **Fix**: `Settings` now has a model validator: `DERIV_TOKEN_KEY` is required (and must be a syntactically valid Fernet key) whenever `ENVIRONMENT != "development"`; a malformed key raises regardless of environment. Extracted the previously-duplicated Fernet encrypt/decrypt logic (api-gateway and market-data-service each had their own copy) into a new shared `backend/shared/token_crypto.py` (`encrypt_token`/`decrypt_token`/`TokenDecryptionError`). `decrypt_token` now **raises** `TokenDecryptionError` on a corrupted/wrong-key token instead of silently returning unusable ciphertext — api-gateway's `/api/v1/deriv/accounts/{user_id}` turns that into a 409 "please re-authenticate"; market-data-service lets it propagate into the existing per-subscription `except Exception` handler, which already degrades to an unauthenticated public connection.
- **Files**: `backend/shared/config.py`, new `backend/shared/token_crypto.py`, `backend/apps/api-gateway/app/main.py`, `backend/services/market-data-service/app/pipeline.py`.
- **Verify**: `tests/test_token_encryption.py` (10 cases: dev-without-key allowed, non-dev-without-key raises, malformed-key-always-raises, encrypt/decrypt round-trip, corrupted-ciphertext raises, key-rotation raises) + `tests/test_api_gateway_token_decrypt.py` (409 on undecryptable stored token, 404 passthrough still works).

### 1.5 Central market-data fan-out (one socket per symbol, not per bot) — ✅ done
- **Problem**: `MarketDataPipeline._subscribe_and_consume` opened a distinct Deriv socket per `(user, symbol, bot)`, including a per-user DB token lookup + OTP auth for ticks — data that's actually public (see `docs/deriv-api.md`). Many bots watching the same symbol multiplied sockets unnecessarily.
- **Fix**: Replaced the per-subscription task model with per-*symbol* sharing: `_symbol_tasks`/`_symbol_subscribers` (refcounted, lock-guarded) + a new `_consume_symbol(symbol)` that opens one plain public Deriv connection per unique symbol and fans out via the existing Redis stream (consumers already filter/read by the `symbol` field in the payload). `subscribe_for_bot`/`unsubscribe_for_bot` now register/release a bot's interest in a symbol rather than starting/stopping a dedicated connection; the last subscriber leaving tears the shared connection down. The default symbols passed to `run()` get a permanent `"__default__"` sentinel subscriber so they're never torn down. All per-user OTP/token-lookup code was removed from the tick path entirely (it was solving a problem — account-scoped ticks — that doesn't exist; `account_id`/`app_id` stay in `subscribe_for_bot`'s signature for API compatibility but are unused).
- **Files**: `backend/services/market-data-service/app/pipeline.py` (also fixed a `redis.close()` → `aclose()` deprecation warning and removed an unused `Callable` import noticed along the way).
- **Verify**: `tests/test_market_data_fanout.py` (6 cases, `DerivClient` mocked to avoid real network) — two bots on the same symbol share one connection, different symbols get separate connections, unsubscribing one of two keeps the shared stream alive, unsubscribing the last one tears it down, a default `run()`-started symbol survives all bot unsubscribes, `subscribe_for_bot` is idempotent for a repeated key.

### 1.6 Rate limits & safety guards
- **Problem**: No per-user socket/subscription quotas; a bot bug or malicious user could open unbounded Deriv connections.
- **Fix**: Cap concurrent subscriptions per user/bot in `MarketDataPipeline.subscribe_for_bot`; return a 429/`HTTPException` when exceeded.
- **Files**: `backend/services/market-data-service/app/pipeline.py`, `backend/services/market-data-service/app/main.py`.
- **Verify**: test that the N+1th subscribe call for a user is rejected.

### 1.7 Observability — ✅ done
- **Problem**: Only `log.exception`/`log.info` calls scattered around (5 of 8 services had no logging at all); no metrics, no structured logging.
- **Fix**: New `backend/shared/observability.py` provides `configure_logging(service_name)` (routes the root logger through a single JSON-formatted handler; idempotent so re-importing a service's `main.py` in tests never duplicates handlers) and `metrics_response()` (renders `prometheus_client.generate_latest()` as a Prometheus-text `Response`). Every service's `main.py` now calls `configure_logging(...)` at import time and exposes `GET /metrics` alongside its existing `/health`. `market-data-service` additionally gets three counters/gauge wired into `pipeline.py`: `MARKET_DATA_TICKS_PROCESSED` (incremented per parsed tick in `_consume_symbol`), `MARKET_DATA_DERIV_RECONNECTS` (incremented on a successful `deriv.reconnect()`), and `MARKET_DATA_ACTIVE_SUBSCRIPTIONS` (a gauge set to `len(self._subscriptions)` on every subscribe/unsubscribe, and reset to 0 on pipeline shutdown).
- **Non-obvious gotcha**: the three metric objects live in `backend/shared/observability.py`, *not* in `pipeline.py` itself — `app.pipeline`/`app.main` get purged from `sys.modules` and re-imported by multiple test files (see `test_market_data_fanout.py`'s `_import_pipeline_module`), and re-running a `Counter(...)`/`Gauge(...)` call against `prometheus_client`'s process-global default `REGISTRY` on a second import raises `ValueError: Duplicated timeseries in CollectorRegistry`. `backend.shared.observability` is a normal dotted package that's never purged, so the metric objects are created exactly once per process and just imported (not re-created) elsewhere.
- **Files**: new `backend/shared/observability.py`; `backend/services/market-data-service/app/pipeline.py`; every service's `app/main.py` (`api-gateway`, `account-service`, `analysis-service`, `bot-service`, `execution-service`, `market-data-service`, `position-service`, `risk-service`, `strategy-ai-service`); `backend/requirements/base.txt` and `pyproject.toml` (added `prometheus_client`).
- **Verify**: `tests/test_observability.py` (JSON-log formatting round-trip, `GET /metrics` returns Prometheus text format via `account-service`, and the active-subscriptions gauge tracks subscribe/unsubscribe on a real `MarketDataPipeline`); full suite (`pytest tests/`) confirmed green with no registry-collision errors; `mypy`/`ruff` show no new issues beyond pre-existing ones in untouched code paths.

**Phase 1 exit criteria**: bot state survives a restart, there is one source of truth for bot lifecycle, schema changes go through migrations, tokens can't silently end up plaintext or corrupted, and the system won't open unbounded Deriv sockets.

---

## Phase 2 — Bot development (the actual trading loop)

Today `bot-service` only manages *lifecycle state* (CREATED/RUNNING/STOPPED/…). It doesn't yet generate signals, size positions, check risk, or place orders. The shared schemas (`Signal`, `Prediction`, `OrderRequest/Response`, `Position`, `RiskDecision` in `backend/shared/schemas/trading.py`) already define the contracts for this — Phase 2 is implementing the services that produce/consume them and wiring the event flow.

### 2.1 Define the event flow
Tick (market-data-service, `market-events` stream)
→ Signal (strategy-ai-service, reads ticks, emits `Signal`/`Prediction`)
→ Risk check (risk-service, reads `Signal` + account state, emits `RiskDecision`)
→ Order (execution-service, on approval, places order via Deriv, emits `OrderResponse`)
→ Position update (position-service, tracks `Position` from fills)
→ Bot status (bot-service, aggregates health/heartbeat from the above, can `EMERGENCY_STOP` on risk breach)

Use the Redis streams already declared in `backend/shared/config.py`/`backend/shared/events/redis_streams.py` (`market-events`, `bot-events`, `order-events`, `position-events`, `risk-events`) as the backbone — they're defined but currently only `market-events` is written to.

### 2.2 `account-service` — real account state
- Fetch and cache Deriv account balance/equity (`balance` WS request per `docs/deriv-api.md`), expose `Account` objects, publish balance changes if needed for risk checks.
- **Files**: `backend/services/account-service/app/main.py` (+ new `pipeline.py`/`client.py` following the market-data-service pattern), reuse `backend/shared/deriv_client.py`.

### 2.3 `strategy-ai-service` — signal generation
- Consume `market-events`, run a strategy (start with one simple rule-based strategy — e.g. moving-average crossover — before any ML), emit `Signal` to `bot-events` or a dedicated `signal-events` stream.
- Each bot's `BotConfig.strategy` field already names which strategy to run — use it to route.
- **Files**: `backend/services/strategy-ai-service/app/main.py` + new pipeline module.

### 2.4 `risk-service` — pre-trade risk checks
- Consume signals, apply the risk fields already on `BotConfig` (`max_positions`, `max_positions_per_symbol`, `min_lot`/`max_lot`, `risk_per_trade`, `max_daily_loss`, `max_drawdown`), emit `RiskDecision`.
- On breach of `max_daily_loss`/`max_drawdown`, call bot-service to trigger `EMERGENCY_STOP` (the state machine already supports this transition from any state).
- **Files**: `backend/services/risk-service/app/main.py` + new module.

### 2.5 `execution-service` — order placement
- On an approved `RiskDecision`, build a Deriv `proposal` → `buy` request (per `docs/deriv-api.md` contract lifecycle) via `DerivClient`, emit `OrderResponse`.
- Needs the user's OTP-authenticated socket — reuse the token-fetch/decrypt pattern from `market-data-service/app/pipeline.py`.
- **Files**: `backend/services/execution-service/app/main.py` + new module.

### 2.6 `position-service` — position tracking
- Consume order fills (`order-events`) and Deriv's `portfolio`/`proposal_open_contract` streams, maintain live `Position` records, publish `position-events` for risk-service to consume (drawdown calc needs live positions).
- **Files**: `backend/services/position-service/app/main.py` + new module.

### 2.7 Wire `bot-service` into the loop
- On `start_bot`: call market-data-service `/subscribe` for the bot's symbols (this HTTP call already exists per the integration-priority-plan and last commit — confirm it's actually invoked from `BotManager.start_bot`, not just available), and mark the bot as the owner of a strategy run.
- On `stop_bot`: call `/unsubscribe`, tear down any open positions per bot config (or leave open per a `close_on_stop` flag — decide with the user).
- Consume `risk-events` for `EMERGENCY_STOP` signals targeting this bot.
- **Files**: `backend/services/bot-service/app/service.py`, `main.py`.

### 2.8 End-to-end integration test
- One test (following the existing repo-level integration test pattern from the last commit) that: starts a bot with a mocked Deriv client, injects a fake tick, asserts a signal is produced, a risk decision is made, an order is "placed" (mocked), and a position is recorded.
- **Files**: `tests/test_end_to_end_bot_loop.py` (new).

**Phase 2 exit criteria**: a bot created via the API, once started, autonomously reacts to live/mocked ticks and produces trade decisions through to a recorded position — the full loop the shared schemas were designed for — with risk limits enforced and emergency-stop working.

---

## Suggested order of work
1. 1.1 → 1.2 → 1.3 (stop the duplication and data loss first — everything else is easier to build on a stable base)
2. 1.4, 1.6 (safety on the Deriv-facing paths before more traffic flows through them)
3. 1.5, 1.7 (efficiency/observability — valuable but not blocking)
4. 2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6 → 2.7 → 2.8 (build the loop one stage at a time, each stage independently testable against the previous stage's output)
