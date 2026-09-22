 Two real bugs in the toolkit, worth fixing before trusting any output

 - The "L1 / Lasso" model is actually L2. train_scalp_model.py:409,415 pass
   LogisticRegression(solver="liblinear", l1_ratio=1.0, ...) but leave penalty at its
   default "l2"; l1_ratio is ignored unless penalty="elasticnet". So the docstring's
   central premise ("Lasso is the point — a feature that contributes nothing has its weight
   driven to exactly zero") never happens, and the ZEROED verdict at line 440 can essentially
   never fire. Fix: penalty="l1".
 - Every script's --history default points at data/mt5_history.json, which does not
   exist (backtest_scalp.py:346, null_test_conditional.py:270, train_scalp_model.py:472).
   The only shipped file is data/deriv_history.json.

 ---

 Plan

 Phase 0 — Make the toolkit installable and runnable

 - Add fa-ml-toolkit/pyproject.toml exposing forex_agent as a package
   (requires-python = ">=3.11"), then pip install -e ./fa-ml-toolkit into the existing
   .venv. This removes the six sys.path.insert hacks and makes import forex_agent work in
   every container for free (the Dockerfiles already COPY . /app with PYTHONPATH=/app).
 - requirements.txt: numpy>=1.26, keep scikit-learn>=1.3, add joblib, un-comment
   httpx / websockets (both already in backend/requirements/base.txt), leave xgboost
   optional and lazily imported.
 - Delete asyncio_mode = auto from fa-ml-toolkit/pytest.ini — dead config, no async tests,
   and pytest-asyncio is not even a toolkit dependency.
 - Do not add fa-ml-toolkit/tests to the root testpaths. All nine services expose a
   top-level package named app and the existing tests depend on a careful sys.modules purge
   (see tests/test_market_data_service_api.py:30); widening collection will break it. Add a
   separate make test-ml target and a separate CI step instead.
 - Run the toolkit's 27 tests on numpy 2.x to confirm the pin relaxation is safe.

 Files: new fa-ml-toolkit/pyproject.toml; edit fa-ml-toolkit/requirements.txt,
 fa-ml-toolkit/pytest.ini, Makefile, .github/workflows/ci.yml.

 Phase 1 — Fix the toolkit and generalise it to H4

 - Fix the two bugs above (penalty="l1"; correct --history defaults).
 - Frame generalisation. build_dataset_v2 hardcodes M5 as the execution frame
   (frames.get("5m"), M5_WINDOW, and hold * 300 for the purge horizon at
   train_scalp_model.py:400 and search_combinations.py:180). Replace with --exec-frame /
   --macro-frames and derive seconds from the existing Timeframe.seconds property. For H4
   the cascade becomes macro = D1, micro = H4. Note encode_features refuses a non-coarser
   macro (features.py:186), so the 12-column FEATURES_V2 (two macro frames over one micro)
   collapses to a single D1→H4 cascade of 7 columns unless W1 bars are synthesised from D1.
   Recommendation: single D1→H4 cascade first; add W1 aggregation later only if the extra
   columns earn their keep.
 - Un-hardcode the cost model. SPREAD_ATR = 0.095 * 1.5 is duplicated at
   train_scalp_model.py:85 and backtest_scalp.py:65. The backtest exposes --spread-atr;
   the trainer does not, so there is currently no way to train under a non-M5 cost assumption.
   Add the flag, defaulting per-frame from the ADR-004 table (H4 → 0.017 * 1.5 = 0.0255).
 - Promote D1_WINDOW / H4_WINDOW / M5_WINDOW from module constants to CLI flags.

 Files: fa-ml-toolkit/scripts/train_scalp_model.py, backtest_scalp.py,
 null_test_conditional.py, search_combinations.py.

 Phase 2 — Deriv demo connectivity

 Getting credentials (walkthrough to run with you):
 1. Sign in at app.deriv.com — a demo account exists by default.
 2. Register an app at the developers dashboard to get an app_id.
 3. Settings → Security → API token, scope it read + trade (not admin/payments),
    with the demo (VRTC…) account selected. Copy the token.
 4. Put DERIV_APP_ID and DERIV_TOKEN into .env (currently both empty). Also generate
    DERIV_TOKEN_KEY with the Fernet one-liner already in .env.example — it is required
    whenever ENVIRONMENT != "development".

 New backend/shared/deriv_v3_client.py — a legacy-v3 client, added alongside the
 existing DerivClient rather than replacing it, so market-data-service and its tests do not
 regress. It needs the one thing DerivClient structurally lacks: req_id correlation.
 The current client's receive() returns whatever arrives next, so it cannot interleave a
 request/response with a live subscription — unavoidable once you are streaming ticks and
 placing orders on one socket.

 Methods: authorize, ping keepalive (<2 min), ticks_history (candles), ticks,
 contracts_for, proposal, buy, sell, proposal_open_contract, contract_update,
 portfolio, balance. Keep the websocket_factory seam that
 tests/test_deriv_client.py:35 already uses for fakes.

 Config (backend/shared/config.py, pydantic-settings, add fields + .env.example keys):
 DERIV_V3_WS_URL, DERIV_TRADE_MODE (demo | real, default demo),
 DERIV_MULTIPLIER, DERIV_STAKE, DERIV_CURRENCY.
 Do not import fa-ml-toolkit/forex_agent/config.py — it instantiates a module-level
 singleton at import time (config.py:236), raises ValueError during import on a malformed
 env var, and carries ~47 unused knobs (Telegram, OpenAI, MT5, risk ladders) that would fork
 this repo's configuration.

 Phase 3 — Candles, the missing input

 Add backend/shared/candles.py: fetch OHLC via v3 ticks_history
 (style: "candles", granularity, count, end) and keep rolling per-symbol windows.
 scripts/download_deriv_training_data.py already implements the exact paging logic
 (walks backwards on end = oldest - 1, PAGE = 1000) — reuse its shape.

 For H4, poll on frame close rather than aggregating ticks: far less code and no gap or
 dedup logic. Persist to a new market_candles table (alembic revision 0002, following the
 raw-op.execute style of 0001_initial_schema.py, since target_metadata = None and there
 are no ORM models) so the service warm-starts and research can export the same history.

 Reuse the existing unused MarketCandle schema at backend/shared/schemas/trading.py:83.

 Phase 4 — analysis-service becomes the ML service

 It is the least-integrated service in the repo: no Dockerfile, absent from docker-compose.yml
 and from the CI docker matrix, and the only service that does not use
 backend.shared.health. Fix all four, assign host port 8004.

 - Persist the model. Add joblib.dump({"model", "scaler", "feature_names", "encoding", "exec_frame", "macro_frames",
   "spread_atr"}, path) to the trainer, plus a loader here. The
   saved metadata matters: a model trained at M5 must not be silently served at H4.
 - Loop: on each H4 close → candles → encode_features(d1, h4) → encode_row_* →
   scaler.transform → predict_proba → expectancy.evaluate_edge(p_win, reward_r, cost_r)
   → if accepted, publish Signal + Prediction (both schemas already exist and are unused)
   to a signal-events Redis stream.
 - Fail closed. No artifact, or an artifact whose exec_frame disagrees with config →
   report degraded and emit nothing. This mirrors evaluate_edge's own design
   (expectancy.py:174), which catches ValueError and returns a rejecting verdict.
 - Endpoints: /health, /metrics, POST /api/v1/analysis/features,
   GET /api/v1/analysis/signal/{symbol}.
 - Define any new Prometheus metrics in backend/shared/observability.py, not in the
   service module — the existing comment at lines 45-51 explains that test re-imports otherwise
   raise "Duplicated timeseries in CollectorRegistry".

 Phase 5 — execution-service places demo trades

 Consume signal-events; map a toolkit signal onto a multiplier order. The key conversion:
 Deriv's limit_order.stop_loss / take_profit are amounts in account currency, not price
 levels, so:

 stop_loss_amount   = stake * multiplier * (stop_distance / entry_price)
 take_profit_amount = stop_loss_amount * reward_r          # preserves the R ratio
 contract_type      = MULTUP if side == BUY else MULTDOWN

 Stake from expectancy.KellySizing.scale_against(configured_pct, p_win, reward_r, cost_r) —
 it already exists, is quarter-Kelly with a 2% cap, and only ever reduces.

 Flow: contracts_for preflight → proposal → buy → subscribe proposal_open_contract →
 on each update call exit_policy.evaluate_exit(...) → contract_update to move the stop for
 breakeven/trailing, or sell on "time stop". Note evaluate_exit takes side as a bare
 str "BUY" (exit_policy.py:181) while the rest of the package uses models.Side; it works
 only because Side is a str enum. Normalise at the boundary.

 Guards: refuse to trade unless DERIV_TRADE_MODE == "demo" unless explicitly overridden;
 cap concurrent positions; daily-loss kill switch wired to the existing
 BotManager.emergency_stop_bot (bot-service/app/service.py:114, which currently has no HTTP
 route — add one).

 For v1, fold the risk check inline here rather than standing up risk-service; the
 RiskDecision schema exists when you want to split it out.

 Phase 6 — Tests

 Follow the repo's established patterns exactly: the websocket_factory/otp_resolver
 constructor seams with hand-rolled fakes (never unittest.mock, never the network), and the
 sys.modules purge dance for any service whose package is named app.

 New: tests/test_deriv_v3_client.py, tests/test_candles.py,
 tests/test_analysis_service_signal.py,
 tests/test_execution_multiplier_mapping.py (assert the stop/TP money conversion above), and
 tests/test_end_to_end_bot_loop.py — which docs/fix-and-bot-development-plan.md §2.8 already
 asks for and which does not exist.

 ---

 Verification

 # 0. Toolkit installs and its own tests pass on numpy 2.x / Py3.14
 pip install -e ./fa-ml-toolkit
 make test-ml                      # 27 tests

 # 1. Real credentials reach Deriv (demo), and candles come back
 python -m scripts.deriv_smoke --symbol frxEURUSD --frame 4h --count 5

 # 2. Download enough history for a D1 macro window (the bundled sample is too short)
 python fa-ml-toolkit/scripts/download_deriv_training_data.py \
     --count 40000 --raw data/deriv_h4.json

 # 3. Is there anything to find? Read the MDE column before the z column.
 python fa-ml-toolkit/scripts/null_test_conditional.py \
     --history data/deriv_h4.json --macro 1d --micro 4h

 # 4. Train and persist an artifact (this is new capability)
 python fa-ml-toolkit/scripts/train_scalp_model.py \
     --history data/deriv_h4.json --exec-frame 4h --spread-atr 0.0255 \
     --save-model artifacts/scalp_h4.joblib

 # 5. What an account would actually have done
 python fa-ml-toolkit/scripts/backtest_scalp.py --history data/deriv_h4.json

 # 6. Full stack
 docker compose up -d && docker compose --profile tools run --rm migrate
 curl localhost:8004/health
 curl localhost:8004/api/v1/analysis/signal/frxEURUSD

 # 7. End-to-end on demo, then confirm the contract exists on Deriv's side
 pytest tests/test_end_to_end_bot_loop.py

 Score everything in R after cost, never accuracy — a filter that lifts the win rate while
 raising the cost hurdle by more has improved nothing.

 Because H4 produces very few signals, add a --replay mode that feeds stored candles through
 the live path so the plumbing can be smoke-tested in minutes rather than days.

 ---

 Risks — read before trusting the demo results

 1. Expected value is the big one. The toolkit measured gross expectancy at ~zero and its
    own sequential backtest returned −0.0075R over 5,176 trades (t = −0.87). Multipliers add a
    commission and overnight swap on multi-day H4 holds, which is worse than the
    spread-only cost model ADR-004 analysed. Expect the demo balance to drift down. This
    exercise validates plumbing, not profitability — which is exactly why it belongs on demo.
 2. Multipliers may not be offered on frx* pairs in every jurisdiction or on demo. The
    contracts_for preflight in Phase 5 exists to catch this; have a fallback symbol ready.
     --count 40000 --raw data/deriv_h4.json

 # 3. Is there anything to find? Read the MDE column before the z column.
 python fa-ml-toolkit/scripts/null_test_conditional.py \
     --history data/deriv_h4.json --macro 1d --micro 4h

 # 4. Train and persist an artifact (this is new capability)
 python fa-ml-toolkit/scripts/train_scalp_model.py \
     --history data/deriv_h4.json --exec-frame 4h --spread-atr 0.0255 \
     --save-model artifacts/scalp_h4.joblib

 # 5. What an account would actually have done
 python fa-ml-toolkit/scripts/backtest_scalp.py --history data/deriv_h4.json

 # 6. Full stack
 docker compose up -d && docker compose --profile tools run --rm migrate
 curl localhost:8004/health
 curl localhost:8004/api/v1/analysis/signal/frxEURUSD

 # 7. End-to-end on demo, then confirm the contract exists on Deriv's side
 pytest tests/test_end_to_end_bot_loop.py

 Score everything in R after cost, never accuracy — a filter that lifts the win rate while
 raising the cost hurdle by more has improved nothing.

 Because H4 produces very few signals, add a --replay mode that feeds stored candles through
 the live path so the plumbing can be smoke-tested in minutes rather than days.

 ---

 Risks — read before trusting the demo results

 1. Expected value is the big one. The toolkit measured gross expectancy at ~zero and its
    own sequential backtest returned −0.0075R over 5,176 trades (t = −0.87). Multipliers add a
    commission and overnight swap on multi-day H4 holds, which is worse than the
    spread-only cost model ADR-004 analysed. Expect the demo balance to drift down. This
    exercise validates plumbing, not profitability — which is exactly why it belongs on demo.
 2. Multipliers may not be offered on frx* pairs in every jurisdiction or on demo. The
    contracts_for preflight in Phase 5 exists to catch this; have a fallback symbol ready.
  5. The bundled sample data is too short to train on — 261–367 daily bars against a 600-bar requirement, so every
     symbol gets skipped.

  I also found two real bugs: the "L1/Lasso" model is silently L2 (penalty left at default, so l1_ratio is ignored and
  the ZEROED verdict can never fire), and all three scripts default --history to a file that doesn't exist.

  One thing I want to be straight about, since you chose the full live loop: multipliers add commission and overnight
  swap on multi-day H4 holds, which is worse than the spread-only cost the ADR analysed. Expect thedemo balance to
  drift down. The plan is built to prove the plumbing works end-to-end — which is exactly the rightthing to do on
  demo — not to make money.

  Sources: Deriv legacy WebSocket docs (https://legacy-docs.deriv.com/docs/websockets) · Options API overview
  (https://developers.deriv.com/docs/intro/api-overview/) · deriv-api reference
  (https://github.com/deriv-com/deriv-api/blob/master/docs/DerivAPI.md) · API token guide
  (https://deriv.com/partners-help-center-questions/how-do-i-create-a-deriv-api-token)

● User approved Claude's plan
  ⎿  Plan saved to: ~\.claude\plans\so-understand-the-fa-ml-toolkit-dapper-bumblebee.md · /plan to edit
     Integrating fa-ml-toolkit into Echelon-Quant, live on a Deriv demo account

     Context

     fa-ml-toolkit/ was built standalone by a coworker and merged wholesale in one commit
     (35afd0c "ml files", via PR #2). It is tracked in git but wired into nothing: not in
     docker-compose.yml, not linted, not type-checked, not run in CI, and not importable from
     any service. Meanwhile analysis-service and strategy-ai-service are 19-line /health
     stubs waiting for exactly this code.

     The goal is to make the toolkit a real part of the platform and to be able to watch it trade
     end-to-end against a Deriv demo account: candles in → features → model → signal → risk →
     order → managed exit.

     Read this first: what the toolkit actually is

     It is not a signal generator. It is a measurement apparatus whose conclusion was that a
     directional model should not be built. fa-ml-toolkit/docs/ADR-004-no-directional-model.md
     records that five learning systems (2× GRU, logistic, ICT Q-table, decision tree) all landed
     on the base rate, and then explains why: for a driftless random walk with barriers at -1
     and +b, P(touch +b first) is exactly 1/(1+b). Measured on M15 majors the deviation was
     −0.010 to +0.003. The series is a martingale at intraday horizons, so EV = p·b − (1−p) − cost = −cost at every
     reward ratio. The ADR deleted the models, their trainers and their
     artifacts on purpose.

     What survives is genuinely valuable and worth integrating:

     ┌──────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────
     ┐
     │          module          │                                  what it gives you
     │
     ├──────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────
     ┤
     │ strategy/features.py     │ encode_features(macro, micro) → 5 dimensionless columns. 27 testspin the contract.
     │
     ├──────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────
     ┤
     │ strategy/barriers.py     │ triple-barrier labelling + the martingale null
     │
     ├──────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────
     ┤
     │ strategy/expectancy.py   │ evaluate_edge (fails closed), KellySizing, Brier, calibration
     │
     ├──────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────
     ┤
     │ execution/exit_policy.py │ breakeven / trailing / time-stop, pure functions
     │
     ├──────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────
     ┤
     │ strategy/indicators.py   │ ATR, RSI, MACD, swings, ADX, Donchian, Fibonacci
     │
     ├──────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────
     ┤
     │ strategy/ict.py          │ FVG, order blocks, sweeps — correct and tested, but imported by nothing
     │
     └──────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────
     ┘

     The ADR's actionable conclusion is that cost in R falls as 1/√time (M5 = 0.095R → H4 =
     0.017R), so the remedy is to move the execution frame up, not to train harder. That is
     why this plan is built on H4.

     Decisions taken (confirmed with you)

     1. Full live signal-to-trade loop on a Deriv demo account.
     2. Relax the numpy pin and run on the existing Python 3.14 venv.
     3. H4 primary execution frame, D1 structural.
     4. Demo account, walkthrough needed for app_id + token.

     ---

     The five blockers found during exploration

     These are the reasons the toolkit cannot simply be imported and called.

     1. There is no model, and no way to make one.
     scripts/train_scalp_model.py never persists anything — no joblib, no pickle, no artifact
     path anywhere in the repo. The fitted LogisticRegression and StandardScaler are local
     variables in train_and_report (lines 401, 415) that are garbage-collected on return. The
     entire output is stdout text. There is also no predict(), no loader, no service entrypoint.

     2. Nothing in the platform produces candles.
     encode_features needs multi-timeframe OHLC. backend/shared/deriv_client.py only does
     ticks; market-data-service emits raw ticks to the market-events Redis stream, which
     has no consumer anywhere in the repo. MarketCandle exists in
     backend/shared/schemas/trading.py:83 and is unused.

     3. The repo talks to the wrong Deriv API for this job. (the important one)
     backend/shared/deriv_client.py and docs/deriv-api.md target Deriv's Options API
     (wss://api.derivws.com/trading/v1/options/ws/*, OTP-URL auth). Options are fixed-payout,
     fixed-expiry contracts — there is no stop-loss price, no take-profit, no trailing. The toolkit
     emits a stop distance, an R-multiple target and managed exits, which only map onto
     multiplier contracts, and those live on the legacy v3 API:

     - wss://ws.derivws.com/websockets/v3?app_id={app_id} — simple authorize with a token
     - contract_type: MULTUP | MULTDOWN, with limit_order: {stop_loss, take_profit}
     - contract_update moves the stop on an open contract → this is what makes breakeven and
       trailing possible
     - session times out after 2 minutes idle, so a ping keepalive is mandatory

     This API is also far simpler to point at a demo account (authorize with a demo token; no OTP
     dance). Sources: legacy WebSocket docs (https://legacy-docs.deriv.com/docs/websockets),
     Options API overview (https://developers.deriv.com/docs/intro/api-overview/),
     deriv-api reference (https://github.com/deriv-com/deriv-api/blob/master/docs/DerivAPI.md).

     4. The dependencies cannot install as pinned.
     fa-ml-toolkit/requirements.txt pins numpy>=1.23,<2.0, which has no CPython 3.14 wheels.
     I grepped for numpy-1.x-only APIs (np.float_, np.NaN, np.product, …) and found none,
     so the cap is conservative rather than load-bearing. The toolkit also has no
     pyproject.toml/setup.py — every script bootstraps with sys.path.insert(...), so no
     service can import it.

     5. The bundled sample data cannot run two of the four documented steps.
     data/deriv_history.json has 261–367 daily bars per symbol, but null_test_conditional.py
     and train_scalp_model.py both require D1_WINDOW = 600. Every symbol is skipped and you get
     "not enough to train". Fresh history must be downloaded before anything can be trained.

     Two real bugs in the toolkit, worth fixing before trusting any output

     - The "L1 / Lasso" model is actually L2. train_scalp_model.py:409,415 pass
       LogisticRegression(solver="liblinear", l1_ratio=1.0, ...) but leave penalty at its
       default "l2"; l1_ratio is ignored unless penalty="elasticnet". So the docstring's
       central premise ("Lasso is the point — a feature that contributes nothing has its weight
       driven to exactly zero") never happens, and the ZEROED verdict at line 440 can essentially
       never fire. Fix: penalty="l1".
     - Every script's --history default points at data/mt5_history.json, which does not
       exist (backtest_scalp.py:346, null_test_conditional.py:270, train_scalp_model.py:472).
       The only shipped file is data/deriv_history.json.

     ---

     Plan

     Phase 0 — Make the toolkit installable and runnable

     - Add fa-ml-toolkit/pyproject.toml exposing forex_agent as a package
       (requires-python = ">=3.11"), then pip install -e ./fa-ml-toolkit into the existing
       .venv. This removes the six sys.path.insert hacks and makes import forex_agent work in
       every container for free (the Dockerfiles already COPY . /app with PYTHONPATH=/app).
     - requirements.txt: numpy>=1.26, keep scikit-learn>=1.3, add joblib, un-comment
       httpx / websockets (both already in backend/requirements/base.txt), leave xgboost
       optional and lazily imported.
     - Delete asyncio_mode = auto from fa-ml-toolkit/pytest.ini — dead config, no async tests,
       and pytest-asyncio is not even a toolkit dependency.
     - Do not add fa-ml-toolkit/tests to the root testpaths. All nine services expose a
       top-level package named app and the existing tests depend on a careful sys.modules purge
       (see tests/test_market_data_service_api.py:30); widening collection will break it. Add a
       separate make test-ml target and a separate CI step instead.
     - Run the toolkit's 27 tests on numpy 2.x to confirm the pin relaxation is safe.

     Files: new fa-ml-toolkit/pyproject.toml; edit fa-ml-toolkit/requirements.txt,
     fa-ml-toolkit/pytest.ini, Makefile, .github/workflows/ci.yml.

     Phase 1 — Fix the toolkit and generalise it to H4

     - Fix the two bugs above (penalty="l1"; correct --history defaults).
     - Frame generalisation. build_dataset_v2 hardcodes M5 as the execution frame
       (frames.get("5m"), M5_WINDOW, and hold * 300 for the purge horizon at
       train_scalp_model.py:400 and search_combinations.py:180). Replace with --exec-frame /
       --macro-frames and derive seconds from the existing Timeframe.seconds property. For H4
       the cascade becomes macro = D1, micro = H4. Note encode_features refuses a non-coarser
       macro (features.py:186), so the 12-column FEATURES_V2 (two macro frames over one micro)
       collapses to a single D1→H4 cascade of 7 columns unless W1 bars are synthesised from D1.
       Recommendation: single D1→H4 cascade first; add W1 aggregation later only if the extra
       columns earn their keep.
     - Un-hardcode the cost model. SPREAD_ATR = 0.095 * 1.5 is duplicated at
       train_scalp_model.py:85 and backtest_scalp.py:65. The backtest exposes --spread-atr;
       the trainer does not, so there is currently no way to train under a non-M5 cost assumption.
       Add the flag, defaulting per-frame from the ADR-004 table (H4 → 0.017 * 1.5 = 0.0255).
     - Promote D1_WINDOW / H4_WINDOW / M5_WINDOW from module constants to CLI flags.

     Files: fa-ml-toolkit/scripts/train_scalp_model.py, backtest_scalp.py,
     null_test_conditional.py, search_combinations.py.

     Phase 2 — Deriv demo connectivity

     Getting credentials (walkthrough to run with you):
     1. Sign in at app.deriv.com — a demo account exists by default.
     2. Register an app at the developers dashboard to get an app_id.
     3. Settings → Security → API token, scope it read + trade (not admin/payments),
        with the demo (VRTC…) account selected. Copy the token.
     4. Put DERIV_APP_ID and DERIV_TOKEN into .env (currently both empty). Also generate
        DERIV_TOKEN_KEY with the Fernet one-liner already in .env.example — it is required
        whenever ENVIRONMENT != "development".

     New backend/shared/deriv_v3_client.py — a legacy-v3 client, added alongside the
     existing DerivClient rather than replacing it, so market-data-service and its tests do not
     regress. It needs the one thing DerivClient structurally lacks: req_id correlation.
     The current client's receive() returns whatever arrives next, so it cannot interleave a
     request/response with a live subscription — unavoidable once you are streaming ticks and
     placing orders on one socket.

     Methods: authorize, ping keepalive (<2 min), ticks_history (candles), ticks,
     contracts_for, proposal, buy, sell, proposal_open_contract, contract_update,
     portfolio, balance. Keep the websocket_factory seam that
     tests/test_deriv_client.py:35 already uses for fakes.

     Config (backend/shared/config.py, pydantic-settings, add fields + .env.example keys):
     DERIV_V3_WS_URL, DERIV_TRADE_MODE (demo | real, default demo),
     DERIV_MULTIPLIER, DERIV_STAKE, DERIV_CURRENCY.
     Do not import fa-ml-toolkit/forex_agent/config.py — it instantiates a module-level
     singleton at import time (config.py:236), raises ValueError during import on a malformed
     env var, and carries ~47 unused knobs (Telegram, OpenAI, MT5, risk ladders) that would fork
     this repo's configuration.

     Phase 3 — Candles, the missing input

     Add backend/shared/candles.py: fetch OHLC via v3 ticks_history
     (style: "candles", granularity, count, end) and keep rolling per-symbol windows.
     scripts/download_deriv_training_data.py already implements the exact paging logic
     (walks backwards on end = oldest - 1, PAGE = 1000) — reuse its shape.

     For H4, poll on frame close rather than aggregating ticks: far less code and no gap or
     dedup logic. Persist to a new market_candles table (alembic revision 0002, following the
     raw-op.execute style of 0001_initial_schema.py, since target_metadata = None and there
     are no ORM models) so the service warm-starts and research can export the same history.

     Reuse the existing unused MarketCandle schema at backend/shared/schemas/trading.py:83.

     Phase 4 — analysis-service becomes the ML service

     It is the least-integrated service in the repo: no Dockerfile, absent from docker-compose.yml
     and from the CI docker matrix, and the only service that does not use
     backend.shared.health. Fix all four, assign host port 8004.

     - Persist the model. Add joblib.dump({"model", "scaler", "feature_names", "encoding", "exec_frame",
       "macro_frames", "spread_atr"}, path) to the trainer, plus a loader here. The
       saved metadata matters: a model trained at M5 must not be silently served at H4.
     - Loop: on each H4 close → candles → encode_features(d1, h4) → encode_row_* →
       scaler.transform → predict_proba → expectancy.evaluate_edge(p_win, reward_r, cost_r)
       → if accepted, publish Signal + Prediction (both schemas already exist and are unused)
       to a signal-events Redis stream.
     - Fail closed. No artifact, or an artifact whose exec_frame disagrees with config →
       report degraded and emit nothing. This mirrors evaluate_edge's own design
       (expectancy.py:174), which catches ValueError and returns a rejecting verdict.
     - Endpoints: /health, /metrics, POST /api/v1/analysis/features,
       GET /api/v1/analysis/signal/{symbol}.
     - Define any new Prometheus metrics in backend/shared/observability.py, not in the
       service module — the existing comment at lines 45-51 explains that test re-imports otherwise
       raise "Duplicated timeseries in CollectorRegistry".

     Phase 5 — execution-service places demo trades

     Consume signal-events; map a toolkit signal onto a multiplier order. The key conversion:
     Deriv's limit_order.stop_loss / take_profit are amounts in account currency, not price
     levels, so:

     stop_loss_amount   = stake * multiplier * (stop_distance / entry_price)
     take_profit_amount = stop_loss_amount * reward_r          # preserves the R ratio
     contract_type      = MULTUP if side == BUY else MULTDOWN

     Stake from expectancy.KellySizing.scale_against(configured_pct, p_win, reward_r, cost_r) —
     it already exists, is quarter-Kelly with a 2% cap, and only ever reduces.

     Flow: contracts_for preflight → proposal → buy → subscribe proposal_open_contract →
     on each update call exit_policy.evaluate_exit(...) → contract_update to move the stop for
     breakeven/trailing, or sell on "time stop". Note evaluate_exit takes side as a bare
     str "BUY" (exit_policy.py:181) while the rest of the package uses models.Side; it works
     only because Side is a str enum. Normalise at the boundary.

     Guards: refuse to trade unless DERIV_TRADE_MODE == "demo" unless explicitly overridden;
     cap concurrent positions; daily-loss kill switch wired to the existing
     BotManager.emergency_stop_bot (bot-service/app/service.py:114, which currently has no HTTP
     route — add one).

     For v1, fold the risk check inline here rather than standing up risk-service; the
     RiskDecision schema exists when you want to split it out.

     Phase 6 — Tests

     Follow the repo's established patterns exactly: the websocket_factory/otp_resolver
     constructor seams with hand-rolled fakes (never unittest.mock, never the network), and the
     sys.modules purge dance for any service whose package is named app.

     New: tests/test_deriv_v3_client.py, tests/test_candles.py,
     tests/test_analysis_service_signal.py,
     tests/test_execution_multiplier_mapping.py (assert the stop/TP money conversion above), and
     tests/test_end_to_end_bot_loop.py — which docs/fix-and-bot-development-plan.md §2.8 already
     asks for and which does not exist.

     ---

     Verification

     # 0. Toolkit installs and its own tests pass on numpy 2.x / Py3.14
     pip install -e ./fa-ml-toolkit
     make test-ml                      # 27 tests

     # 1. Real credentials reach Deriv (demo), and candles come back
     python -m scripts.deriv_smoke --symbol frxEURUSD --frame 4h --count 5

     # 2. Download enough history for a D1 macro window (the bundled sample is too short)
     python fa-ml-toolkit/scripts/download_deriv_training_data.py \
         --count 40000 --raw data/deriv_h4.json

     # 3. Is there anything to find? Read the MDE column before the z column.
     python fa-ml-toolkit/scripts/null_test_conditional.py \
         --history data/deriv_h4.json --macro 1d --micro 4h

     # 4. Train and persist an artifact (this is new capability)
     python fa-ml-toolkit/scripts/train_scalp_model.py \
         --history data/deriv_h4.json --exec-frame 4h --spread-atr 0.0255 \
         --save-model artifacts/scalp_h4.joblib

     # 5. What an account would actually have done
     python fa-ml-toolkit/scripts/backtest_scalp.py --history data/deriv_h4.json

     # 6. Full stack
     docker compose up -d && docker compose --profile tools run --rm migrate
     curl localhost:8004/health
     curl localhost:8004/api/v1/analysis/signal/frxEURUSD

     # 7. End-to-end on demo, then confirm the contract exists on Deriv's side
     pytest tests/test_end_to_end_bot_loop.py

     Score everything in R after cost, never accuracy — a filter that lifts the win rate while
     raising the cost hurdle by more has improved nothing.

     Because H4 produces very few signals, add a --replay mode that feeds stored candles through
     the live path so the plumbing can be smoke-tested in minutes rather than days.

     ---

     Risks — read before trusting the demo results

     1. Expected value is the big one. The toolkit measured gross expectancy at ~zero and its
        own sequential backtest returned −0.0075R over 5,176 trades (t = −0.87). Multipliers add a
        commission and overnight swap on multi-day H4 holds, which is worse than the
        spread-only cost model ADR-004 analysed. Expect the demo balance to drift down. This
        exercise validates plumbing, not profitability — which is exactly why it belongs on demo.
     2. Multipliers may not be offered on frx* pairs in every jurisdiction or on demo. The
        contracts_for preflight in Phase 5 exists to catch this; have a fallback symbol ready.
     3. Do not train on R_* / 1HZ*. The README is explicit that Deriv's volatility indices
        come from a cryptographically secure RNG and are provably unpredictable — yet R_75 and
        1HZ75V are in the bundled sample, SCALP_MARKETS = ("R_75",) in the toolkit config, and
        R_75 is a hardcoded default symbol in market-data-service/app/main.py:81. Exclude them
        from training.
     4. Python version divergence already exists and this plan does not fix it: .python-version
        and CI say 3.12, all Dockerfiles say python:3.12-slim, the local .venv is uv-made 3.14.
        Pin the analysis-service image to 3.12 and treat 3.14 as local-only.
     5. .dockerignore does not exclude fa-ml-toolkit/, so its 5 MB sample data is already
        being copied into all eight images. Add an exclusion for fa-ml-toolkit/data/.
     6. Broker-clock skew is documented but unhandled (README lines 154-158). Deriv epochs are
        true UTC so this is safe here, but it will bite if MT5 data is ever mixed in.