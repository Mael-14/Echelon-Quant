PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev test lint format-check typecheck migrate migrate-down migration

install:
	$(PIP) install -r backend/requirements/base.txt

install-dev:
	$(PIP) install -r backend/requirements/dev.txt

test:
	pytest

lint:
	$(PYTHON) -m ruff check backend tests

format-check:
	$(PYTHON) -m ruff format --check backend tests

typecheck:
	$(PYTHON) -m mypy backend --ignore-missing-imports

# Apply all pending Alembic migrations (deriv_tokens, market_ticks, bots tables).
# Run this before starting any service against a fresh database.
migrate:
	$(PYTHON) -m alembic upgrade head

# Roll back the most recently applied migration.
migrate-down:
	$(PYTHON) -m alembic downgrade -1

# Scaffold a new migration file: make migration m="add foo column"
migration:
	$(PYTHON) -m alembic revision -m "$(m)"
