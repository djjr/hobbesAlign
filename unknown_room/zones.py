from __future__ import annotations
from dataclasses import dataclass, field
from unknown_room.entities import ResourceType


@dataclass
class Zone:
    id: int
    entity_ids: list[int] = field(default_factory=list)


@dataclass
class JointPool:
    pool_id: int
    participant_ids: list[int]
    holdings: dict[ResourceType, float] = field(default_factory=dict)
    contributions: dict[int, float] = field(default_factory=dict)  # entity_id -> amount contributed

    def is_empty(self) -> bool:
        return all(v <= 0.0 for v in self.holdings.values())
