from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from backend.shared.config import Settings
from backend.shared.token_crypto import TokenDecryptionError, decrypt_token, encrypt_token


def test_settings_allows_missing_key_in_development() -> None:
    settings = Settings(environment="development", deriv_token_key=None)
    assert settings.deriv_token_key is None


@pytest.mark.parametrize("environment", ["production", "staging", "Production"])
def test_settings_requires_key_outside_development(environment: str) -> None:
    with pytest.raises(ValueError, match="DERIV_TOKEN_KEY must be set"):
        Settings(environment=environment, deriv_token_key=None)


def test_settings_accepts_valid_key_outside_development() -> None:
    key = Fernet.generate_key().decode()
    settings = Settings(environment="production", deriv_token_key=key)
    assert settings.deriv_token_key == key


def test_settings_rejects_malformed_key_regardless_of_environment() -> None:
    with pytest.raises(ValueError, match="not a valid Fernet key"):
        Settings(environment="development", deriv_token_key="not-a-valid-fernet-key")


def test_encrypt_decrypt_round_trip_with_key() -> None:
    key = Fernet.generate_key().decode()
    settings = Settings(environment="production", deriv_token_key=key)

    stored = encrypt_token("plain-token", settings=settings)
    assert stored != "plain-token"
    assert decrypt_token(stored, settings=settings) == "plain-token"


def test_encrypt_is_plaintext_passthrough_without_key_in_development() -> None:
    settings = Settings(environment="development", deriv_token_key=None)

    stored = encrypt_token("plain-token", settings=settings)
    assert stored == "plain-token"
    assert decrypt_token(stored, settings=settings) == "plain-token"


def test_decrypt_raises_on_corrupted_ciphertext() -> None:
    key = Fernet.generate_key().decode()
    settings = Settings(environment="production", deriv_token_key=key)

    with pytest.raises(TokenDecryptionError):
        decrypt_token("this-is-not-valid-fernet-ciphertext", settings=settings)


def test_decrypt_raises_when_key_rotated() -> None:
    old_settings = Settings(
        environment="production", deriv_token_key=Fernet.generate_key().decode()
    )
    new_settings = Settings(
        environment="production", deriv_token_key=Fernet.generate_key().decode()
    )

    stored = encrypt_token("plain-token", settings=old_settings)

    with pytest.raises(TokenDecryptionError):
        decrypt_token(stored, settings=new_settings)
