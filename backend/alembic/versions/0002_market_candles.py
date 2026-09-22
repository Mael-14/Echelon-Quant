"""market_candles: OHLC storage for the feature pipeline

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-22

`encode_features` needs multi-timeframe OHLC, and nothing in the platform
stored any: market_ticks holds raw ticks only. This table lets analysis-service
warm-start after a restart instead of re-downloading a full macro window, and
lets research export exactly the history the live path saw.

Follows 0001's raw `op.execute` style: `target_metadata` is None and there are
no ORM models, so there is nothing for autogenerate to work from.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # (symbol, timeframe, open_time) is the natural key: one bar per frame per
    # instant. It is UNIQUE rather than merely indexed because the writer
    # upserts on it -- the newest bar gets re-fetched and must correct itself
    # rather than duplicate.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS market_candles (
            id BIGSERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open_time TIMESTAMPTZ NOT NULL,
            close_time TIMESTAMPTZ NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume NUMERIC NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT market_candles_natural_key UNIQUE (symbol, timeframe, open_time)
        )
        """
    )

    # Every read is "the newest N bars for this symbol and frame", so the index
    # descends on open_time to serve that without a sort.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS market_candles_symbol_frame_time_idx
            ON market_candles (symbol, timeframe, open_time DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS market_candles_symbol_frame_time_idx")
    op.execute("DROP TABLE IF EXISTS market_candles")
