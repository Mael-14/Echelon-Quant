from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RedisStreamConfig:
    market_events: str = "market-events"
    bot_events: str = "bot-events"
    order_events: str = "order-events"
    position_events: str = "position-events"
    risk_events: str = "risk-events"


STREAMS = RedisStreamConfig()


@dataclass
class StreamMessage:
    stream: str
    data: dict
    message_id: str | None = None
    fields: dict | None = field(default_factory=dict)
