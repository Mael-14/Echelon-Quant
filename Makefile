PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev test lint format-check typecheck

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
