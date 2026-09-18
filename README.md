# Echelon Quant

An automated trading platform for Deriv (synthetic indices / options), built as a set of
independent FastAPI microservices sharing one Postgres database and Redis instance.

## Services

| Service               | Port | Status                                           |
|------------------------|------|---------------------------------------------------|
| `api-gateway`          | 8000 | Public entrypoint - Deriv token/OAuth storage, proxies bot lifecycle requests to `bot-service` |
| `account-service`      | 8001 | Stub (`/health` only)                              |
| `bot-service`          | 8002 | Bot lifecycle state machine, persisted to Postgres |
| `market-data-service`  | 8003 | Deriv tick ingestion, fan-out to Redis + Postgres  |
| `analysis-service`     | -    | Stub (`/health` only)                              |
| `execution-service`    | -    | Stub (`/health` only)                              |
| `position-service`     | -    | Stub (`/health` only)                              |
| `risk-service`         | -    | Stub (`/health` only)                              |
| `strategy-ai-service`  | -    | Stub (`/health` only)                              |

See `docs/deriv-api.md` for the Deriv API integration notes and `docs/fix-and-bot-development-plan.md`
for the active roadmap.

## Running with Docker (recommended, incl. GitHub Codespaces)

This repo's dev container (`.devcontainer/devcontainer.json`) gives you a ready-to-use Docker
environment automatically when opened in GitHub Codespaces (or locally via VS Code's
"Reopen in Container"). Once the container is up:

```bash
# 1. Copy the example env file (defaults already match docker-compose's service names)
cp .env.example .env

# 2. Start the database and cache first
docker compose up -d postgres redis

# 3. Apply migrations (the app services no longer create their own tables at startup)
docker compose --profile tools run --rm migrate

# 4. Start everything else
docker compose up -d

# Check a service is up
curl http://localhost:8000/health
```

To re-run migrations after pulling new ones, or to roll one back:

```bash
docker compose --profile tools run --rm migrate                 # upgrade to head
docker compose run --rm migrate python -m alembic downgrade -1  # roll back one step
```

## Running locally without Docker

Requires Python 3.12+ and a reachable Postgres + Redis (update `.env` accordingly).

```bash
make install-dev        # installs backend/requirements/dev.txt
make migrate             # alembic upgrade head
PYTHONPATH=. uvicorn app.main:app --app-dir backend/apps/api-gateway --reload
```

Each service is run the same way, pointing `--app-dir` at its own directory
(e.g. `backend/services/bot-service`) - run from the repo root so `PYTHONPATH=.`
resolves the shared `backend.shared` package, same as `PYTHONPATH=/app` does in
each service's Dockerfile.

## Development

```bash
make test           # pytest
make lint            # ruff check
make format-check    # ruff format --check
make typecheck       # mypy
```

Database schema changes go through Alembic migrations in `backend/alembic/versions/` -
see `make migration m="add foo column"` to scaffold a new one. Do not add ad-hoc
`CREATE TABLE` calls to service startup code.
