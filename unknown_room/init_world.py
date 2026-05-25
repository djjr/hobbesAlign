from __future__ import annotations
import random
import numpy as np

from unknown_room.entities import (
    Entity, EntityProfile, ResourceCard, StrengthCard,
    ResourceType, StrengthType,
    N_AGENTS, N_REACTIVE, N_ZONES, BASE_STRENGTH, BASE_EXTRACTION,
)
from unknown_room.zones import Zone

ALL_RESOURCES = list(ResourceType)
ALL_STRENGTHS = list(StrengthType)


def init_world(
    n_agents: int = N_AGENTS,
    n_reactive: int = N_REACTIVE,
    n_zones: int = N_ZONES,
    rng: random.Random | None = None,
    np_rng: np.random.Generator | None = None,
) -> tuple[dict[int, Entity], dict[int, EntityProfile], dict[int, Zone]]:
    """
    Returns (entities, profiles, zones).
    entity ids 0..n_agents-1 are strategic; n_agents..n_agents+n_reactive-1 are reactive.
    """
    if rng is None:
        rng = random.Random()
    if np_rng is None:
        np_rng = np.random.default_rng()

    entities: dict[int, Entity] = {}
    profiles: dict[int, EntityProfile] = {}

    # --- Strategic agents ---
    for i in range(n_agents):
        # DESIGN QUESTION: need levels are random uniform [0.5, 1.5]
        need_levels = {r: rng.uniform(0.5, 1.5) for r in ALL_RESOURCES}
        # Starting holdings: mean need met ~50%
        holdings = {r: need_levels[r] * rng.uniform(0.25, 0.75) for r in ALL_RESOURCES}

        resource_cards = [
            ResourceCard(
                resource_type=r,
                pct_need_met=min(1.0, holdings[r] / need_levels[r]),
            )
            for r in ALL_RESOURCES
        ]
        strength_cards = [
            StrengthCard(strength_type=s, base_rating=float(BASE_STRENGTH))
            for s in ALL_STRENGTHS
        ]

        entities[i] = Entity(
            id=i,
            zone_id=i % n_zones,
            entity_type="strategic",
            resource_cards=resource_cards,
            strength_cards=strength_cards,
            exposed_indices=[0, 1, 2],  # first three cards by default
            need_levels=need_levels,
            holdings=holdings,
        )

        # DESIGN QUESTION: entity profiles are random Dirichlet at init
        weights_arr = np_rng.dirichlet(np.ones(len(ALL_STRENGTHS)))
        profiles[i] = EntityProfile(
            entity_id=i,
            extraction_weights={s: float(w) for s, w in zip(ALL_STRENGTHS, weights_arr)},
            base_rate=BASE_EXTRACTION,
        )

    # --- Reactive entities ---
    for j in range(n_reactive):
        eid = n_agents + j
        # Reactive entities have holdings but they are inexhaustible in Phase 1
        # (extraction produces yield without decrementing reactive holdings)
        need_levels = {r: 1.0 for r in ALL_RESOURCES}
        holdings = {r: 999.0 for r in ALL_RESOURCES}  # effectively unlimited

        resource_cards = [
            ResourceCard(resource_type=r, pct_need_met=1.0)
            for r in ALL_RESOURCES
        ]
        strength_cards = [
            StrengthCard(strength_type=s, base_rating=float(BASE_STRENGTH))
            for s in ALL_STRENGTHS
        ]

        entities[eid] = Entity(
            id=eid,
            zone_id=j % n_zones,
            entity_type="reactive",
            resource_cards=resource_cards,
            strength_cards=strength_cards,
            exposed_indices=[0, 1, 2],
            need_levels=need_levels,
            holdings=holdings,
        )

        weights_arr = np_rng.dirichlet(np.ones(len(ALL_STRENGTHS)))
        profiles[eid] = EntityProfile(
            entity_id=eid,
            extraction_weights={s: float(w) for s, w in zip(ALL_STRENGTHS, weights_arr)},
            base_rate=BASE_EXTRACTION,
        )

    # --- Zones ---
    zones: dict[int, Zone] = {z: Zone(id=z) for z in range(n_zones)}
    for entity in entities.values():
        zones[entity.zone_id].entity_ids.append(entity.id)

    return entities, profiles, zones
