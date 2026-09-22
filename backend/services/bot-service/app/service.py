from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel

from backend.shared.schemas import BotConfig, BotLifecycleState, BotStatus


class BotRecord(BaseModel):
    config: BotConfig
    status: BotStatus


class BotManager:
    def __init__(self) -> None:
        self._bots: dict[str, BotRecord] = {}
        self._allowed_transitions: dict[BotLifecycleState, set[BotLifecycleState]] = {
            BotLifecycleState.CREATED: {
                BotLifecycleState.STARTING,
                BotLifecycleState.EMERGENCY_STOP,
            },
            BotLifecycleState.STARTING: {
                BotLifecycleState.RUNNING,
                BotLifecycleState.ERROR,
                BotLifecycleState.STOPPING,
                BotLifecycleState.EMERGENCY_STOP,
            },
            BotLifecycleState.RUNNING: {
                BotLifecycleState.PAUSED,
                BotLifecycleState.STOPPING,
                BotLifecycleState.ERROR,
                BotLifecycleState.EMERGENCY_STOP,
            },
            BotLifecycleState.PAUSED: {
                BotLifecycleState.RUNNING,
                BotLifecycleState.STOPPING,
                BotLifecycleState.ERROR,
                BotLifecycleState.EMERGENCY_STOP,
            },
            BotLifecycleState.STOPPING: {
                BotLifecycleState.STOPPED,
                BotLifecycleState.ERROR,
                BotLifecycleState.EMERGENCY_STOP,
            },
            BotLifecycleState.STOPPED: {
                BotLifecycleState.STARTING,
                BotLifecycleState.EMERGENCY_STOP,
            },
            BotLifecycleState.ERROR: {
                BotLifecycleState.STARTING,
                BotLifecycleState.STOPPING,
                BotLifecycleState.EMERGENCY_STOP,
            },
            BotLifecycleState.EMERGENCY_STOP: {BotLifecycleState.STOPPED},
        }

    def list_bots(self) -> list[BotRecord]:
        return list(self._bots.values())

    def create_bot(self, config: BotConfig) -> BotRecord:
        if config.bot_id in self._bots:
            raise ValueError(f"Bot '{config.bot_id}' already exists")

        record = BotRecord(config=config, status=BotStatus(bot_id=config.bot_id))
        self._bots[config.bot_id] = record
        return record

    def get_bot(self, bot_id: str) -> BotRecord | None:
        return self._bots.get(bot_id)

    def load_bot(self, record: BotRecord) -> None:
        """Hydrate a bot from persisted state without validating a state transition."""
        self._bots[record.config.bot_id] = record

    def update_bot(self, bot_id: str, config: BotConfig) -> BotRecord:
        current = self._bots.get(bot_id)
        if current is None:
            raise KeyError(f"Bot '{bot_id}' not found")
        if config.bot_id != bot_id:
            raise ValueError("Path id must match body bot_id")

        updated_status = current.status.model_copy(
            update={"last_heartbeat": datetime.now(timezone.utc)}
        )
        updated = BotRecord(config=config, status=updated_status)
        self._bots[bot_id] = updated
        return updated

    def transition(
        self, bot_id: str, target: BotLifecycleState, error: str | None = None
    ) -> BotStatus:
        current = self._bots.get(bot_id)
        if current is None:
            raise KeyError(f"Bot '{bot_id}' not found")

        source_state = current.status.state
        allowed_targets = self._allowed_transitions.get(source_state, set())
        if target not in allowed_targets:
            raise RuntimeError(
                f"Invalid transition for bot '{bot_id}': {source_state.value} -> {target.value}"
            )

        current.status = current.status.model_copy(
            update={
                "state": target,
                "error": error,
                "last_heartbeat": datetime.now(timezone.utc),
            }
        )
        return current.status

    def start_bot(self, bot_id: str) -> BotStatus:
        self.transition(bot_id, BotLifecycleState.STARTING)
        return self.transition(bot_id, BotLifecycleState.RUNNING)

    def stop_bot(self, bot_id: str) -> BotStatus:
        self.transition(bot_id, BotLifecycleState.STOPPING)
        return self.transition(bot_id, BotLifecycleState.STOPPED)

    def pause_bot(self, bot_id: str) -> BotStatus:
        return self.transition(bot_id, BotLifecycleState.PAUSED)

    def emergency_stop_bot(self, bot_id: str, reason: str) -> BotStatus:
        return self.transition(bot_id, BotLifecycleState.EMERGENCY_STOP, error=reason)
