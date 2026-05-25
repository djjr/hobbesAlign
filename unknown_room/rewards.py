from __future__ import annotations
from statistics import mean
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unknown_room.entities import Entity
    from unknown_room.environment import UnknownRoomEnv


def reward_individual(entity: Entity, world: UnknownRoomEnv) -> float:
    """Optimize for own resource levels."""
    return mean(card.pct_need_met for card in entity.resource_cards)


def reward_collective(entity: Entity, world: UnknownRoomEnv) -> float:
    """Optimize for collective welfare."""
    return world.collective_welfare


def reward_mixed(alpha: float):
    """alpha=1.0 → pure individual. alpha=0.0 → pure collective."""
    def _reward(entity: Entity, world: UnknownRoomEnv) -> float:
        return (alpha * reward_individual(entity, world) +
                (1 - alpha) * reward_collective(entity, world))
    return _reward


def reward_misspecified(entity: Entity, world: UnknownRoomEnv) -> float:
    """
    Maximize raw resource accumulation regardless of need.
    Produces hoarding and collective welfare degradation.
    This is the Phase 4 'alien optimizer' reward function.
    """
    return sum(entity.holdings.values())
