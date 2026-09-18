from __future__ import annotations

import asyncio
import json
import logging

import asyncpg

from backend.shared.config import Settings

from .service import BotRecord

log = logging.getLogger("bot-service-store")


class BotStore:
    def __init__(self, *, settings: Settings) -> None:
        self.settings = settings
        self._pg_pool: asyncpg.Pool | None = None
        self._pg_lock = asyncio.Lock()

    async def _ensure_pg(self) -> asyncpg.Pool | None:
        # Schema is managed by Alembic migrations (backend/alembic/versions/0001_initial_schema.py)
        # rather than created here - run `alembic upgrade head` before starting this service.
        if self._pg_pool is not None:
            return self._pg_pool
        async with self._pg_lock:
            if self._pg_pool is None:
                # Settings.database_url is SQLAlchemy-style 'postgresql+asyncpg://...'
                # asyncpg expects 'postgresql://...'
                dsn = self.settings.database_url.replace("+asyncpg", "")
                try:
                    self._pg_pool = await asyncpg.create_pool(dsn)
                except Exception:
                    # Could not connect to Postgres (not available in local/test environments).
                    # Log and continue without a DB connection; callers will treat bots as
                    # in-memory-only and retry the connection on the next call.
                    log.warning("Could not connect to Postgres at %s; continuing without DB", dsn)
                    self._pg_pool = None
        return self._pg_pool

    async def load_all(self) -> list[BotRecord]:
        pool = await self._ensure_pg()
        if pool is None:
            return []

        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT config, status FROM bots")

        records: list[BotRecord] = []
        for row in rows:
            try:
                records.append(
                    BotRecord(
                        config=json.loads(row["config"]),
                        status=json.loads(row["status"]),
                    )
                )
            except Exception:
                log.exception("Failed to decode persisted bot row; skipping")
        return records

    async def save(self, record: BotRecord) -> None:
        pool = await self._ensure_pg()
        if pool is None:
            return

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO bots(bot_id, config, status)
                VALUES($1, $2::jsonb, $3::jsonb)
                ON CONFLICT (bot_id) DO UPDATE
                SET config = EXCLUDED.config, status = EXCLUDED.status, updated_at = now()
                """,
                record.config.bot_id,
                record.config.model_dump_json(),
                record.status.model_dump_json(),
            )

    async def close(self) -> None:
        if self._pg_pool is not None:
            await self._pg_pool.close()
