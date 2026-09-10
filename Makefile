PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install install-dev test lint

install:
	$(PIP) install -r backend/requirements/base.txt

install-dev:
	$(PIP) install -r backend/requirements/dev.txt

test:
	pytest

lint:
	ruff check .
