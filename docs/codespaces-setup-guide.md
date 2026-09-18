# Running Echelon Quant in GitHub Codespaces

This is a step-by-step guide to getting the full stack (Postgres, Redis, and every FastAPI
service) running inside a GitHub Codespace, how to fill in `.env`, and how the pieces actually
talk to each other under the hood.

---

## 1. Open the Codespace

1. On the repo's GitHub page: **Code → Codespaces → Create codespace on `feature/api-gateway`**
   (or whichever branch you're working from).
2. GitHub builds the codespace from `.devcontainer/devcontainer.json`. That file:
   - Uses the `mcr.microsoft.com/devcontainers/python:1-3.12-bullseye` base image.
   - Adds the `docker-in-docker` feature, so `docker` and `docker compose` work *inside* the
     codespace (the codespace itself is a container, and this lets it run more containers -
     Postgres, Redis, each service - alongside it).
   - Runs `pip install -r backend/requirements/dev.txt` automatically after creation
     (`postCreateCommand`), so `pytest`, `ruff`, `mypy` etc. are ready immediately.
   - Forwards ports `8000-8003` (the services with host ports) and `5432`/`6379`
     (Postgres/Redis), and labels them in the "Ports" tab so you can click through to a
     running service instead of guessing URLs.
3. Wait for "Setting up remote connection" / `postCreateCommand` to finish, then open a
   terminal in the codespace - everything below runs there.

You do **not** need Docker Desktop, a local Python install, or a local Postgres/Redis - the
codespace provides all of it.

---

## 2. Create and fill in `.env`

```bash
cp .env.example .env
```

`.env` is read by `backend/shared/config.py:Settings` (via `pydantic-settings`), and its keys
map 1:1 to `Settings` fields - every service imports the same `Settings` class, so one `.env`
configures all of them. Here's what each block means and what you actually need to change:

### Always fine to leave as-is (defaults match docker-compose's service names)

```env
ENVIRONMENT=development
APP_NAME=Echelon Quant
DEBUG=false

POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=echelon_quant
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_STREAM_MARKET_EVENTS=market-events
REDIS_STREAM_BOT_EVENTS=bot-events
REDIS_STREAM_ORDER_EVENTS=order-events
REDIS_STREAM_POSITION_EVENTS=position-events
REDIS_STREAM_RISK_EVENTS=risk-events

BOT_SERVICE_URL=http://bot-service:8000
```

`postgres` and `redis` here are **not placeholders** - they're the literal container/service
names docker-compose gives those containers on its internal network (see §4). Change
`POSTGRES_PASSWORD`/`POSTGRES_USER` only if you also change them in `docker-compose.yml`;
otherwise leave them matching.

> `ENVIRONMENT` matters for one safety check: `Settings` **requires** `DERIV_TOKEN_KEY` to be
> set (and refuses to boot without it) whenever `ENVIRONMENT != "development"`, so that Deriv
> tokens can never accidentally end up stored in plaintext outside local dev. Leave this as
> `development` for a Codespace unless you're deliberately testing that path.

### Deriv credentials - optional, only needed for authenticated trading

```env
DERIV_TOKEN=
DERIV_ACCOUNT_ID=
DERIV_APP_ID=
DERIV_TOKEN_KEY=
```

- **Leave all four blank** to run with the public, unauthenticated Deriv tick feed -
  `market-data-service` falls back to this automatically, and it's enough for local
  development, testing the API, and running the test suite.
- Fill these in only if you want the platform to place real/demo trades against your own
  Deriv account:
  - `DERIV_TOKEN` - an API token from your Deriv account (Deriv dashboard → API Token).
    Treat it like a password; never commit it.
  - `DERIV_ACCOUNT_ID` / `DERIV_APP_ID` - your Deriv account and registered app IDs (see
    `docs/deriv-api.md` for where these come from and the WebSocket contract).
  - `DERIV_TOKEN_KEY` - **not** a Deriv credential; it's a local Fernet symmetric key
    `api-gateway` uses to encrypt Deriv tokens before storing them in Postgres
    (`deriv_tokens` table). Generate one yourself:
    ```bash
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    ```
    Paste the output in as `DERIV_TOKEN_KEY=...`. If you leave `ENVIRONMENT=development` and
    leave this blank, tokens are stored in plaintext (dev-only convenience) - a warning built
    into `Settings`, not a Codespaces-specific thing. Never leave it blank outside development.

Nothing else in `.env.example` needs touching for a standard Codespaces run.

---

## 3. Bring the stack up

```bash
# 1. Start the database and cache first - everything else depends on these being healthy
docker compose up -d postgres redis

# 2. Apply schema migrations (services no longer create their own tables at startup -
#    see backend/alembic/versions/). This must run before step 3 on a fresh database.
docker compose --profile tools run --rm migrate

# 3. Start every application service
docker compose up -d

# 4. Confirm api-gateway is alive
curl http://localhost:8000/health
# -> {"status":"ok","service":"api-gateway","environment":"development"}
```

`docker compose --profile tools run --rm migrate` runs Alembic's `upgrade head` in a
throwaway container (defined in `docker-compose.yml` under the `tools` profile, so it doesn't
start automatically with `docker compose up`) and exits. Re-run it any time new migrations
land in `backend/alembic/versions/`. To roll one back: `docker compose run --rm migrate python -m alembic downgrade -1`.

### Check everything's healthy

```bash
docker compose ps                        # all should show "healthy" or "running"
curl http://localhost:8000/health        # api-gateway
curl http://localhost:8001/health        # account-service
curl http://localhost:8002/health        # bot-service
curl http://localhost:8003/health        # market-data-service
curl http://localhost:8003/metrics       # Prometheus metrics (see docs/fix-and-bot-development-plan.md 1.7)
```

Five other services (`analysis-service`, `execution-service`, `position-service`,
`risk-service`, `strategy-ai-service`) exist as code and have Dockerfiles, but **aren't wired
into `docker-compose.yml` yet** - they're stubs (`/health` only) with no assigned host port,
tracked in `docs/fix-and-bot-development-plan.md` Phase 2 as still-to-be-built. You can run one
manually with `docker compose build <name> && docker run --rm -p 8004:8000 --network echelon-quant_default <image>` if you need to poke at it before it's added to compose, but this isn't part of the normal workflow yet.

### Logs / stopping

```bash
docker compose logs -f bot-service      # tail one service's logs
docker compose down                      # stop everything, keep volumes (data persists)
docker compose down -v                   # stop and wipe Postgres/Redis data too
```

---

## 4. How Postgres and Redis connect to the services

**Everything lives on one Docker network, addressed by service name.** `docker-compose.yml`
declares `postgres`, `redis`, `api-gateway`, `account-service`, `bot-service`, and
`market-data-service` as services in the same file with no custom `networks:` section, so
Compose puts them all on one default bridge network and gives each container's *service name*
as its DNS hostname on that network. That's the entire trick behind `POSTGRES_HOST=postgres`
and `REDIS_HOST=redis` in `.env` - `postgres` isn't a placeholder, it resolves via Docker's
embedded DNS to whichever container is currently running the `postgres` service, on whatever
IP Docker assigned it this run. From your Codespace's own shell (outside any container), that
same hostname doesn't resolve - that's why `curl` above uses `localhost:<mapped-port>` instead.

**Startup ordering is enforced by `depends_on` + healthchecks**, not by any code:

```yaml
api-gateway:
  depends_on:
    postgres:
      condition: service_healthy
    redis:
      condition: service_healthy
```

Postgres and Redis each define a `healthcheck` (`pg_isready`, `redis-cli ping`), and every
application service's `depends_on` waits for `service_healthy` before Compose starts it - so
by the time `api-gateway`/`bot-service`/`market-data-service` boot, both dependencies are
already accepting connections. This is Compose-level sequencing only; it doesn't mean the
services can't handle Postgres or Redis going away *later* (see below).

**Each service builds its own connection from the same `Settings` object**, computed as
properties on `backend/shared/config.py:Settings`:

```python
@property
def database_url(self) -> str:
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"

@property
def redis_url(self) -> str:
    return f"redis://{host}:{port}/{db}"
```

- **Postgres**: services that touch the database (`api-gateway`, `bot-service`,
  `market-data-service`) open an `asyncpg` connection pool lazily on first use (e.g.
  `MarketDataPipeline._ensure_pg` / `api-gateway`'s `_ensure_db_pool`), not eagerly at import
  time. `market-data-service` and `bot-service` treat a failed Postgres connection as
  *non-fatal* - they log a warning and keep running in a degraded mode (ticks/bot mutations
  just aren't persisted until Postgres comes back); `api-gateway`'s token storage endpoints are
  the one place a DB failure is allowed to raise, since there's no meaningful fallback for
  "store this encrypted token nowhere."
- **Redis**: `market-data-service` opens a `redis.asyncio` client the same lazy way
  (`_ensure_redis`) and uses it as a **pub/fan-out backbone via Redis Streams**
  (`XADD`/consumer groups), not just a cache. Every parsed market tick gets pushed once to the
  `market-events` stream (`redis_stream_market_events` in `Settings`); this is what lets many
  bots watching the same symbol share **one** upstream Deriv WebSocket connection
  (`MarketDataPipeline`'s per-symbol fan-out, see `docs/fix-and-bot-development-plan.md` 1.5)
  instead of each bot needing its own. The other four declared streams
  (`bot-events`/`order-events`/`position-events`/`risk-events`) are the backbone for Phase 2's
  strategy → risk → execution → position event flow - defined in `Settings` today but not yet
  written to by any service (see `docs/fix-and-bot-development-plan.md` §2.1).
- **Schema migrations** (Alembic, `backend/alembic/`) resolve the same `Settings().database_url`
  independently (`backend/alembic/env.py:get_url()`), so the one-off `migrate` compose service
  talks to the exact same Postgres instance via the exact same `postgres` hostname - it just
  runs once and exits rather than staying up as a long-lived service.

In short: Postgres and Redis are two plain containers with no code of their own; every FastAPI
service is handed the same `.env`-driven `Settings`, resolves `postgres`/`redis` via Docker's
internal DNS, and opens its own lazy, independent connection/pool to each - there's no shared
connection-broker process in between.

---

## 5. Local run without Docker (optional)

Only if you specifically want to run a service directly against Codespaces' own Python instead
of in a container (Postgres/Redis still need to be reachable somehow - e.g. still via
`docker compose up -d postgres redis`, with `.env` pointing `POSTGRES_HOST`/`REDIS_HOST` at
`localhost` instead of the container names in that case):

```bash
make install-dev
make migrate
PYTHONPATH=. uvicorn app.main:app --app-dir backend/apps/api-gateway --reload
```

Run each other service the same way, pointing `--app-dir` at its own folder (e.g.
`backend/services/bot-service`) from the repo root, so `PYTHONPATH=.` resolves the shared
`backend.shared` package the same way each Dockerfile's `ENV PYTHONPATH=/app` does.

---

## 6. Quick troubleshooting

| Symptom | Likely cause |
|---|---|
| A service container exits immediately | Check `docker compose logs <service>` - usually a missing/invalid `.env` value, or migrations weren't run yet |
| `relation "bots" does not exist` (or similar) | You skipped step 2 (`docker compose --profile tools run --rm migrate`) |
| `api-gateway` boots but `DERIV_TOKEN_KEY` errors on non-dev environment | Set `ENVIRONMENT=development` for local/Codespaces use, or generate and set a real Fernet key |
| Can't reach `http://localhost:8000` from outside the codespace | Check the **Ports** tab - Codespaces ports default to private; set 8000 to "Public" or use the forwarded URL Codespaces gives you |
| Bots/ticks not persisting | Postgres may be unreachable - `bot-service`/`market-data-service` degrade silently by design (see §4); check `docker compose logs postgres` and `docker compose ps` |
| `market-data-service` logs `socket.gaierror: [Errno -3] Temporary failure in name resolution` connecting to `api.derivws.com` | Outbound DNS to the public internet is failing - not an app bug, and non-fatal (the service stays up, it just gets no ticks). First test from the Codespace terminal itself, *outside* any container: `curl -v https://api.derivws.com`. Fails there too → your GitHub org's Codespaces network firewall is blocking it; an org admin needs to allowlist `*.derivws.com` (Settings → Codespaces → network). Works on the host but not inside the container → it's the Docker-in-Docker bridge's DNS; add an explicit `dns:` block (e.g. `8.8.8.8`, `1.1.1.1`) to the affected service in `docker-compose.yml` and `docker compose up -d --force-recreate <service>`. If it used to work and the Codespace was just resumed from stopped, try `docker compose down && docker compose up -d` first - a stale bridge network is a common cause after a resume |
