PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev install-ml test test-ml lint format-check typecheck migrate migrate-down migration

install:
	$(PIP) install -r backend/requirements/base.txt

install-dev:
	$(PIP) install -r backend/requirements/dev.txt

# The ML toolkit as an editable install, so `import forex_agent` resolves without
# any sys.path juggling. Its extras stay opt-in: xgboost and the downloaders are
# only needed by individual scripts.
install-ml:
	$(PIP) install -e './fa-ml-toolkit[dev]'

test:
	pytest

# fa-ml-toolkit/tests is deliberately NOT in the root testpaths. Every service
# exposes a top-level package named `app`, and the root suite depends on a
# careful sys.modules purge (tests/test_market_data_service_api.py) that widening
# collection would break. Run the toolkit from its own rootdir instead.
test-ml:
	cd fa-ml-toolkit && $(PYTHON) -m pytest -q

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
