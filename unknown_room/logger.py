from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

from unknown_room.actions import ActionRecord


class TickLogger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._records: list[dict] = []

    def log_tick(
        self,
        tick: int,
        sequence: list[int],
        action_records: list[ActionRecord],
        collective_welfare: float,
        living_count: int,
        extra: dict | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "tick": tick,
            "timestamp": time.time(),
            "sequence": sequence,
            "actions": [self._record_to_dict(r) for r in action_records],
            "collective_welfare": collective_welfare,
            "living_agents": living_count,
        }
        if extra:
            entry.update(extra)
        self._records.append(entry)

    def _record_to_dict(self, r: ActionRecord) -> dict:
        return {
            "tick": r.tick,
            "agent_id": r.agent_id,
            "action_type": r.action_type.value,
            "target_id": r.target_id,
            "success": r.success,
            "yield_amount": r.yield_amount,
            "zone_id": r.zone_id,
            "skip_reason": r.skip_reason,
        }

    def flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            json.dump(self._records, f, indent=2)

    def all_records(self) -> list[dict]:
        return list(self._records)
