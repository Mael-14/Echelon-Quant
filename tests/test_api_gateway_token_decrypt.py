from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient


def _import_api_gateway_main(*, purge_cache: bool):
    # Same pattern as tests/test_api_gateway_bot_proxy.py - see that file for why the
    # sys.modules purge is needed (module name collision with other services' `app` package).
    import importlib
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    service_dir = repo_root / "backend" / "apps" / "api-gateway"
    sys.path.insert(0, str(service_dir))

    if purge_cache:
        for module_name in list(sys.modules):
            if module_name == "app" or module_name.startswith("app."):
                del sys.modules[module_name]

    return importlib.import_module("app.main")


class _FakeConn:
    def __init__(self, row: dict | None) -> None:
        self._row = row

    async def fetchrow(self, query: str, *args):
        return self._row


class _FakeAcquireCtx:
    def __init__(self, row: dict | None) -> None:
        self._row = row

    async def __aenter__(self) -> _FakeConn:
        return _FakeConn(self._row)

    async def __aexit__(self, *exc_info) -> bool:
        return False


class _FakePool:
    def __init__(self, row: dict | None) -> None:
        self._row = row

    def acquire(self) -> _FakeAcquireCtx:
        return _FakeAcquireCtx(self._row)


@pytest.fixture
def main_module():
    return _import_api_gateway_main(purge_cache=True)


def test_list_deriv_accounts_returns_409_on_undecryptable_token(main_module) -> None:
    main_module.settings.deriv_token_key = Fernet.generate_key().decode()
    main_module._ensure_db_pool = AsyncMock(
        return_value=_FakePool({"token": "not-valid-ciphertext"})
    )

    with TestClient(main_module.app) as client:
        resp = client.get("/api/v1/deriv/accounts/u1")

    assert resp.status_code == 409
    assert "re-authenticate" in resp.json()["detail"]


def test_list_deriv_accounts_404_when_no_token_stored(main_module) -> None:
    main_module.settings.deriv_token_key = Fernet.generate_key().decode()
    main_module._ensure_db_pool = AsyncMock(return_value=_FakePool(None))

    with TestClient(main_module.app) as client:
        resp = client.get("/api/v1/deriv/accounts/u1")

    assert resp.status_code == 404
