from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from .config import Settings


class TokenDecryptionError(RuntimeError):
    """Raised when a stored token can't be decrypted with the configured key.

    Callers must treat this as "the token is unusable" (e.g. surface a re-auth
    prompt) rather than silently falling back to the raw stored value.
    """


def get_fernet(settings: Settings) -> Fernet | None:
    """Returns None only when no key is configured (allowed in development only -
    see Settings' validator). A key that's present but malformed raises, since
    that's a config bug that should fail loudly rather than silently disable
    encryption.
    """
    key = settings.deriv_token_key
    if not key:
        return None
    return Fernet(key.encode("utf-8") if isinstance(key, str) else key)


def encrypt_token(plain: str, *, settings: Settings) -> str:
    fernet = get_fernet(settings)
    if fernet is None:
        return plain
    return fernet.encrypt(plain.encode("utf-8")).decode("utf-8")


def decrypt_token(stored: str, *, settings: Settings) -> str:
    fernet = get_fernet(settings)
    if fernet is None:
        return stored
    try:
        return fernet.decrypt(stored.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise TokenDecryptionError(
            "Stored token could not be decrypted with the configured DERIV_TOKEN_KEY"
        ) from exc
