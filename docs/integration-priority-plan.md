# Integration & Productionization Plan — Market Data / Deriv Integration

This short plan lists prioritized tasks to make per-user Deriv integration, market-data fan-out, and bot wiring production-ready.

1. Secrets & token management (Highest)
   - What: Move per-user tokens to a secrets manager (Vault/KMS) or ensure `DERIV_TOKEN_KEY` is provisioned and rotated.
   - Outcome: Tokens not stored plaintext; centralized key management.
   - Effort: 1–2 days
   - Dependencies: infra (Vault/KMS), config rollout.

2. Market-data service control endpoints (High)
   - What: Add HTTP endpoints on `market-data-service` to start/stop per-bot subscriptions (`POST /api/v1/market/subscribe`, `DELETE /api/v1/market/subscribe`).
   - Outcome: Remote control for subscriptions; usable by `bot-service` and UI.
   - Effort: 0.5 day
   - Dependencies: existing `MarketDataPipeline.subscribe_for_bot` methods.

3. Bot → Market-data wiring (High)
   - What: When a bot starts, `bot-service` calls market-data subscribe endpoint for required symbols; stop on bot stop.
   - Outcome: Bots receive account-scoped ticks automatically.
   - Effort: 1 day
   - Dependencies: (2), stable API between services.

4. Persistent bot storage & ownership (High)
   - What: Persist `BotConfig`/`BotStatus` with `owner_id` in Postgres; remove in-memory store.
   - Outcome: Survives restarts and supports multi-tenancy.
   - Effort: 1–2 days + migrations
   - Dependencies: DB migrations, Alembic.

5. DB migrations & schema hygiene (Medium)
   - What: Replace runtime CREATE TABLE calls with Alembic migrations for `deriv_tokens`, `market_ticks`, and `bots` tables.
   - Outcome: Repeatable schema management.
   - Effort: 0.5–1 day
   - Dependencies: DB access and CI job.

6. OAuth UX & account selection (Medium)
   - What: Add OAuth redirect endpoint and a frontend helper to obtain `code`, call `/oauth/exchange`, then fetch accounts and let user choose `account_id`.
   - Outcome: Smooth user onboarding for OAuth users.
   - Effort: 1–2 days
   - Dependencies: Web frontend or simple static page.

7. Central market-data fan-out & dedupe (Medium)
   - What: Subscribe once per unique symbol centrally and fan-out ticks to Redis Streams/topics; consumers filter by bot/user.
   - Outcome: Fewer Deriv sockets and lower resource usage.
   - Effort: 2–3 days
   - Dependencies: Redis stream design, consumer code changes.

8. Rate-limits, quotas & safety (Medium)
   - What: Enforce per-user socket limits, subscription quotas, and backpressure handling.
   - Outcome: Protect platform from abuse/overload.
   - Effort: 1–2 days

9. Observability & testing (Medium)
   - What: Add metrics (Prometheus), structured logs, and integration tests that exercise token store → OTP → subscription flow.
   - Outcome: Easier debugging and confidence for releases.
   - Effort: 2–3 days

10. Deployment & infra (Lower)
    - What: Update Docker images, Compose/Kubernetes manifests, and runbook for secret provisioning and migration steps.
    - Outcome: Repeatable deploys with secret rotation.
    - Effort: 1–2 days

Next immediate action (recommended): implement task (2) — add HTTP subscribe/unsubscribe endpoints on `market-data-service` and test `bot-service` wiring. This unlocks end-to-end bot start/stop flows quickly.

If you want, I can start implementing item (2) now and open a PR with the changes.
