from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ResourceType(Enum):
    FOOD = "FOOD"
    SHELTER = "SHELTER"
    ENERGY = "ENERGY"


class StrengthType(Enum):
    PHYSICAL = "PHYSICAL"
    CUNNING = "CUNNING"
    INFLUENCE = "INFLUENCE"


# ---------------------------------------------------------------------------
# Constants (all tunable)
# ---------------------------------------------------------------------------
N_AGENTS = 30
N_REACTIVE = 30
N_ZONES = 5
N_RESOURCE_TYPES = 3
N_STRENGTH_TYPES = 3
BASE_STRENGTH = 5
BASE_EXTRACTION = 1.0
DEATH_THRESHOLD = 0

TICKS_PER_PHASE = 20


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

@dataclass
class ResourceCard:
    resource_type: ResourceType
    pct_need_met: float = 0.5    # 0.0–1.0; updated each tick from holdings


@dataclass
class StrengthCard:
    strength_type: StrengthType
    base_rating: float           # 0–10, fixed at init


# ---------------------------------------------------------------------------
# Strength modifier
# ---------------------------------------------------------------------------

def strength_modifier(entity: Entity) -> float:
    """
    Sigma-approximated modifier summed across all three resource cards.
    Range: −9 (all resources at 0–10%) to +6 (all above 90%).
    """
    modifier = 0.0
    for card in entity.resource_cards:
        pct = card.pct_need_met
        if pct <= 0.10:
            modifier += -3
        elif pct <= 0.20:
            modifier += -1
        elif pct <= 0.79:
            modifier += 0
        elif pct <= 0.89:
            modifier += 1
        else:
            modifier += 2
    return modifier


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    id: int
    zone_id: int
    entity_type: Literal["strategic", "reactive"]

    resource_cards: list[ResourceCard]    # length 3, one per ResourceType
    strength_cards: list[StrengthCard]    # length 3, one per StrengthType
    exposed_indices: list[int]            # 3 indices into the combined 6-card list

    # Private state — never included in observations
    need_levels: dict[ResourceType, float]
    holdings: dict[ResourceType, float]

    is_dead: bool = False

    @property
    def effective_strengths(self) -> dict[StrengthType, float]:
        mod = strength_modifier(self)
        return {
            s.strength_type: max(0.0, s.base_rating + mod)
            for s in self.strength_cards
        }

    @property
    def total_effective_strength(self) -> float:
        return sum(self.effective_strengths.values())

    @property
    def is_alive(self) -> bool:
        return any(v > DEATH_THRESHOLD for v in self.effective_strengths.values())

    @property
    def all_cards(self) -> list[ResourceCard | StrengthCard]:
        """Combined list of all 6 cards in index order."""
        return self.resource_cards + self.strength_cards  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Entity profile (held by environment, never exposed)
# ---------------------------------------------------------------------------

@dataclass
class EntityProfile:
    entity_id: int
    extraction_weights: dict[StrengthType, float]   # sums to 1.0
    base_rate: float = BASE_EXTRACTION
