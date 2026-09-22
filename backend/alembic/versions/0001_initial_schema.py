"""Initial schema: deriv_tokens, market_ticks, bots

Revision ID: 0001
Revises:
Create Date: 2026-09-17

These tables were previously created ad-hoc by each service on startup via
`CREATE TABLE IF NOT EXISTS` (api-gateway, market-data-service, bot-service
respectively). This migration captures the exact same DDL those services were
issuing, using `IF NOT EXISTS` so it's safe to run against a dev database that
already has the tables from the old ad-hoc path.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Owned by api-gateway (backend/apps/api-gateway/app/main.py)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS deriv_tokens (
            user_id TEXT PRIMARY KEY,
            token TEXT NOT NULL,
            account_id TEXT,
            app_id INTEGER,
            created_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )

    # Owned by market-data-service (backend/services/market-data-service/app/pipeline.py)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS market_ticks (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL,
            bid NUMERIC,
            ask NUMERIC,
            price NUMERIC,
            tick JSONB
        )
        """
    )

    # Owned by bot-service (backend/services/bot-service/app/store.py)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS bots (
            bot_id TEXT PRIMARY KEY,
            config JSONB NOT NULL,
            status JSONB NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now(),
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS bots")
    op.execute("DROP TABLE IF EXISTS market_ticks")
    op.execute("DROP TABLE IF EXISTS deriv_tokens")
