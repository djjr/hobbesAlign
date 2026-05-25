from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unknown_room.entities import ResourceType


class ActionType(Enum):
    SHUFFLE = "SHUFFLE"
    INTERACT = "INTERACT"
    GIVE = "GIVE"
    CLAIM_SHARE = "CLAIM_SHARE"
    CLAIM_ALL = "CLAIM_ALL"
    MOVE = "MOVE"
    DO_NOTHING = "DO_NOTHING"


@dataclass
class Action:
    agent_id: int
    action_type: ActionType
    target_id: int | None = None          # entity_id, or zone_id for MOVE
    resource_type: ResourceType | None = None
    amount: float | None = None
    exposed_indices: list[int] = field(default_factory=list)  # for SHUFFLE


@dataclass
class ActionRecord:
    tick: int
    agent_id: int
    action_type: ActionType
    target_id: int | None
    success: bool
    yield_amount: float | None
    zone_id: int
    skip_reason: str | None = None        # "engaged_as_target", "invalid", etc.
