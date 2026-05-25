from __future__ import annotations
import random
from collections import defaultdict
from typing import TYPE_CHECKING

from unknown_room.actions import Action, ActionRecord, ActionType
from unknown_room.entities import Entity, ResourceType, StrengthType
from unknown_room.zones import JointPool

if TYPE_CHECKING:
    from unknown_room.environment import UnknownRoomEnv


# ---------------------------------------------------------------------------
# Sequencing
# ---------------------------------------------------------------------------

def random_sequence(agent_ids: list[int], rng: random.Random) -> list[int]:
    """Redrawn randomly every tick. Isolated here for easy future swapping."""
    return rng.sample(agent_ids, len(agent_ids))


# ---------------------------------------------------------------------------
# Action logging (step 3)
# ---------------------------------------------------------------------------

def log_actions(
    actions: dict[int, Action],
    sequence: list[int],
    tick: int,
    entities: dict[int, Entity],
    zones: dict,
) -> tuple[list[Action], set[int], list[ActionRecord]]:
    """
    Walk the sequence. Skip any agent that has already been named as a target.
    Returns:
        logged   — actions that proceed to resolution
        engaged  — agent_ids that were targeted before their turn
        skipped_records — ActionRecords for skipped agents (for the log)
    """
    logged: list[Action] = []
    engaged: set[int] = set()
    skipped_records: list[ActionRecord] = []

    for agent_id in sequence:
        if agent_id not in actions:
            continue
        action = actions[agent_id]
        entity = entities[agent_id]

        if agent_id in engaged:
            skipped_records.append(ActionRecord(
                tick=tick,
                agent_id=agent_id,
                action_type=action.action_type,
                target_id=action.target_id,
                success=False,
                yield_amount=None,
                zone_id=entity.zone_id,
                skip_reason="engaged_as_target",
            ))
            continue

        logged.append(action)
        if action.target_id is not None and action.action_type == ActionType.INTERACT:
            engaged.add(action.target_id)

    return logged, engaged, skipped_records


# ---------------------------------------------------------------------------
# Group by target (step 4)
# ---------------------------------------------------------------------------

def group_by_target(logged: list[Action]) -> dict[int, list[Action]]:
    """Returns target_id -> list of INTERACT actions directed at that target."""
    groups: dict[int, list[Action]] = defaultdict(list)
    for action in logged:
        if action.target_id is not None and action.action_type == ActionType.INTERACT:
            groups[action.target_id].append(action)
    return dict(groups)


# ---------------------------------------------------------------------------
# Outcome function components
# ---------------------------------------------------------------------------

def combined_strengths(entities_list: list[Entity]) -> dict[StrengthType, float]:
    """Sum effective strengths across all initiating entities."""
    combined: dict[StrengthType, float] = defaultdict(float)
    for entity in entities_list:
        for stype, val in entity.effective_strengths.items():
            combined[stype] += val
    return dict(combined)


def match_score(
    strengths: dict[StrengthType, float],
    weights: dict[StrengthType, float],
) -> float:
    """
    Dot product of normalized initiator strengths against target profile weights.
    Returns value in [0, 1].
    """
    total = sum(strengths.values()) or 1.0
    return sum(
        (strengths.get(stype, 0.0) / total) * weight
        for stype, weight in weights.items()
    )


def resistance(entity: Entity) -> float:
    """Reactive: 0. Strategic: total effective strength."""
    if entity.entity_type == "reactive":
        return 0.0
    return entity.total_effective_strength


# ---------------------------------------------------------------------------
# Resolve a single INTERACT coalition (one or more initiators vs one target)
# ---------------------------------------------------------------------------

def resolve_interaction(
    initiator_ids: list[int],
    target_id: int,
    tick: int,
    entities: dict[int, Entity],
    profiles: dict,
    pools: dict[int, JointPool],
    next_pool_id: list[int],   # mutable int wrapper for incrementing
    rng: random.Random,
) -> tuple[list[ActionRecord], JointPool | None]:
    """
    Returns (records, new_pool_or_None).
    new_pool is created only if coalition succeeds and yield > 0.
    Solo interactions deliver yield directly to the single initiator.
    """
    target = entities[target_id]
    initiators = [entities[i] for i in initiator_ids]
    profile = profiles[target_id]

    comb = combined_strengths(initiators)
    ms = match_score(comb, profile.extraction_weights)
    net = sum(comb.values()) - resistance(target)

    records = []
    new_pool = None

    if net <= 0:
        for iid in initiator_ids:
            records.append(ActionRecord(
                tick=tick,
                agent_id=iid,
                action_type=ActionType.INTERACT,
                target_id=target_id,
                success=False,
                yield_amount=0.0,
                zone_id=entities[iid].zone_id,
            ))
        return records, None

    yield_amount = profile.base_rate * ms * net

    if len(initiator_ids) == 1:
        # Solo: yield goes directly to the initiator
        iid = initiator_ids[0]
        # Distribute yield across resources proportionally to deficit
        _credit_entity(entities[iid], yield_amount)
        records.append(ActionRecord(
            tick=tick,
            agent_id=iid,
            action_type=ActionType.INTERACT,
            target_id=target_id,
            success=True,
            yield_amount=yield_amount,
            zone_id=entities[iid].zone_id,
        ))
    else:
        # Coalition: yield enters a joint pool
        pid = next_pool_id[0]
        next_pool_id[0] += 1
        new_pool = JointPool(
            pool_id=pid,
            participant_ids=list(initiator_ids),
            holdings={r: yield_amount / len(ResourceType) for r in ResourceType},
        )
        # Track each initiator's proportional contribution
        contrib_each = yield_amount / len(initiator_ids)
        for iid in initiator_ids:
            new_pool.contributions[iid] = contrib_each
            records.append(ActionRecord(
                tick=tick,
                agent_id=iid,
                action_type=ActionType.INTERACT,
                target_id=target_id,
                success=True,
                yield_amount=yield_amount,
                zone_id=entities[iid].zone_id,
            ))

    return records, new_pool


def _credit_entity(entity: Entity, yield_amount: float) -> None:
    """
    Distribute yield across the entity's three resource holdings.
    Allocated proportionally to current deficit (largest deficit gets most).
    Falls back to equal split if no deficit.
    """
    deficits = {
        r: max(0.0, entity.need_levels[r] - entity.holdings[r])
        for r in ResourceType
    }
    total_deficit = sum(deficits.values())
    if total_deficit == 0.0:
        share = yield_amount / len(ResourceType)
        for r in ResourceType:
            entity.holdings[r] += share
    else:
        for r in ResourceType:
            entity.holdings[r] += yield_amount * (deficits[r] / total_deficit)


# ---------------------------------------------------------------------------
# Resolve non-INTERACT actions
# ---------------------------------------------------------------------------

def resolve_give(action: Action, entities: dict[int, Entity], tick: int) -> ActionRecord:
    giver = entities[action.agent_id]
    receiver = entities[action.target_id]
    rtype = action.resource_type
    amount = min(action.amount or 0.0, giver.holdings.get(rtype, 0.0))

    giver.holdings[rtype] = giver.holdings.get(rtype, 0.0) - amount
    receiver.holdings[rtype] = receiver.holdings.get(rtype, 0.0) + amount

    return ActionRecord(
        tick=tick,
        agent_id=action.agent_id,
        action_type=ActionType.GIVE,
        target_id=action.target_id,
        success=True,
        yield_amount=amount,
        zone_id=giver.zone_id,
    )


def resolve_claim_share(
    action: Action,
    pool: JointPool,
    entities: dict[int, Entity],
    tick: int,
) -> ActionRecord:
    """Equal share claim. Always succeeds. Removes claimant from pool."""
    n = len(pool.participant_ids)
    entity = entities[action.agent_id]

    for r in ResourceType:
        share = pool.holdings.get(r, 0.0) / n
        entity.holdings[r] = entity.holdings.get(r, 0.0) + share
        pool.holdings[r] = pool.holdings.get(r, 0.0) - share

    pool.participant_ids.remove(action.agent_id)

    return ActionRecord(
        tick=tick,
        agent_id=action.agent_id,
        action_type=ActionType.CLAIM_SHARE,
        target_id=pool.pool_id,
        success=True,
        yield_amount=sum(pool.holdings.values()) / max(n, 1),
        zone_id=entity.zone_id,
    )


def resolve_contested_claim(
    pool: JointPool,
    claimants: list[int],
    entities: dict[int, Entity],
    tick: int,
    rng: random.Random,
) -> list[ActionRecord]:
    """
    All CLAIM_ALL participants compete. Winner takes entire pool.
    Tiebreak: highest proportional contribution, then random.
    """
    records = []

    if len(claimants) == 1:
        winner_id = claimants[0]
    else:
        winner_id = max(
            claimants,
            key=lambda eid: (
                entities[eid].total_effective_strength,
                pool.contributions.get(eid, 0.0),
                rng.random(),   # final tiebreak
            ),
        )

    winner = entities[winner_id]
    total_yield = sum(pool.holdings.values())  # capture before zeroing
    for r in ResourceType:
        winner.holdings[r] = winner.holdings.get(r, 0.0) + pool.holdings.get(r, 0.0)
        pool.holdings[r] = 0.0

    for cid in claimants:
        records.append(ActionRecord(
            tick=tick,
            agent_id=cid,
            action_type=ActionType.CLAIM_ALL,
            target_id=pool.pool_id,
            success=(cid == winner_id),
            yield_amount=total_yield if cid == winner_id else 0.0,
            zone_id=entities[cid].zone_id,
        ))

    pool.participant_ids.clear()
    return records


def resolve_move(
    action: Action,
    entities: dict[int, Entity],
    zones: dict[int, object],
    tick: int,
) -> ActionRecord:
    entity = entities[action.agent_id]
    old_zone = zones[entity.zone_id]
    new_zone_id = action.target_id
    new_zone = zones[new_zone_id]

    old_zone.entity_ids.remove(entity.id)
    entity.zone_id = new_zone_id
    new_zone.entity_ids.append(entity.id)

    return ActionRecord(
        tick=tick,
        agent_id=action.agent_id,
        action_type=ActionType.MOVE,
        target_id=new_zone_id,
        success=True,
        yield_amount=None,
        zone_id=new_zone_id,
    )


def resolve_shuffle(
    action: Action,
    entities: dict[int, Entity],
    tick: int,
) -> ActionRecord:
    entity = entities[action.agent_id]
    indices = action.exposed_indices
    if len(indices) == 3 and all(0 <= i <= 5 for i in indices):
        entity.exposed_indices = list(indices)
        success = True
    else:
        success = False

    return ActionRecord(
        tick=tick,
        agent_id=action.agent_id,
        action_type=ActionType.SHUFFLE,
        target_id=None,
        success=success,
        yield_amount=None,
        zone_id=entity.zone_id,
    )


def resolve_do_nothing(
    action: Action,
    entities: dict[int, Entity],
    tick: int,
) -> ActionRecord:
    return ActionRecord(
        tick=tick,
        agent_id=action.agent_id,
        action_type=ActionType.DO_NOTHING,
        target_id=None,
        success=True,
        yield_amount=None,
        zone_id=entities[action.agent_id].zone_id,
    )
