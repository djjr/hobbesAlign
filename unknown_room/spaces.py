"""
Observation and action space definitions for RL consumption.

Observation vector layout (OBS_SIZE floats, all in [0, 1]):
  [0:3]   resource pct_need_met  (FOOD, SHELTER, ENERGY)
  [3:6]   effective strengths normalised to [0,1]  (PHYSICAL, CUNNING, INFLUENCE)
  [6:9]   holdings normalised by 2×need_level
  [9:15]  exposed-card flags (binary, one per card index 0-5)
  [15:16] in_pool flag
  [16:19] pool holdings normalised
  [19:20] pool n_participants / N_AGENTS
  [20:160] zone entity slots  (MAX_ZONE_SLOTS=20, 7 floats each)
            per slot: [present, type0, type1, type2, val0, val1, val2]
            type_i = 1 if card i is a resource card, 0 if strength card
            val_i  = pct_need_met or base_rating/10
  [160:205] event log  (5 events × 9 floats)
            per event: action_type one-hot (7) + success (1) + is_self (1)
  [205:206] collective_welfare
  [206:207] tick / ticks_per_phase
  [207:208] zone_id / n_zones

Action space (flat Discrete, ACTION_SIZE slots):
  0            DO_NOTHING
  1            SHUFFLE
  2..6         MOVE to zone 0-4
  7..66        INTERACT with entity 0-59
  67..126      GIVE to entity 0-59  (gives 25% of most-abundant resource)
  127          CLAIM_SHARE
  128          CLAIM_ALL
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unknown_room.environment import UnknownRoomEnv

from unknown_room.actions import Action, ActionType
from unknown_room.entities import ResourceType, StrengthType, ResourceCard

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

OBS_RESOURCE_START  = 0
OBS_STRENGTH_START  = 3
OBS_HOLDINGS_START  = 6
OBS_EXPOSED_START   = 9
OBS_POOL_START      = 15
OBS_ZONE_START      = 20
OBS_LOG_START       = 160
OBS_GLOBAL_START    = 205
OBS_SIZE            = 208

MAX_ZONE_SLOTS   = 20        # padded zone entity slots
EVENT_LOG_SLOTS  = 5

# Action layout
_N_ZONES   = 5
_N_TOTAL   = 60              # N_AGENTS + N_REACTIVE default

ACT_DO_NOTHING   = 0
ACT_SHUFFLE      = 1
ACT_MOVE_BASE    = 2                               # 2..6
ACT_INTERACT_BASE = ACT_MOVE_BASE + _N_ZONES       # 7..66
ACT_GIVE_BASE    = ACT_INTERACT_BASE + _N_TOTAL    # 67..126
ACT_CLAIM_SHARE  = ACT_GIVE_BASE + _N_TOTAL        # 127
ACT_CLAIM_ALL    = ACT_CLAIM_SHARE + 1             # 128
ACTION_SIZE      = ACT_CLAIM_ALL + 1               # 129

ALL_ACTION_TYPES = [at.value for at in ActionType]

# ---------------------------------------------------------------------------
# Observation → numpy array
# ---------------------------------------------------------------------------

def obs_to_array(obs: dict, env: UnknownRoomEnv, agent_id: int) -> np.ndarray:
    """Flatten the dict observation returned by UnknownRoomEnv into a float32 vector."""
    vec = np.zeros(OBS_SIZE, dtype=np.float32)

    # Own resource cards [0:3]
    for i, card in enumerate(obs["own_resource_cards"]):
        vec[OBS_RESOURCE_START + i] = card["pct_need_met"]

    # Own effective strengths [3:6]  — max possible is BASE_STRENGTH + 6 = 11, use 16 for headroom
    for i, card in enumerate(obs["own_strength_cards"]):
        vec[OBS_STRENGTH_START + i] = card["effective_rating"] / 16.0

    # Own holdings [6:9]  — normalise by 2× need so 100% need-met → ~0.5
    agent = env.entities[agent_id]
    for i, rtype in enumerate(ResourceType):
        need  = max(agent.need_levels.get(rtype, 1.0), 1e-6)
        held  = obs["own_holdings"].get(rtype.value, 0.0)
        vec[OBS_HOLDINGS_START + i] = min(1.0, held / (2.0 * need))

    # Exposed card flags [9:15]
    for idx in obs["own_exposed_indices"]:
        if 0 <= idx < 6:
            vec[OBS_EXPOSED_START + idx] = 1.0

    # Pool state [15:20]
    pool = obs.get("own_pool")
    if pool:
        vec[OBS_POOL_START] = 1.0
        for i, rtype in enumerate(ResourceType):
            vec[OBS_POOL_START + 1 + i] = min(1.0, pool["holdings"].get(rtype.value, 0.0) / 10.0)
        vec[OBS_POOL_START + 4] = len(pool["participant_ids"]) / max(env.n_agents, 1)

    # Zone entity slots [20:160]
    for slot, entity_obs in enumerate(obs["zone_entities"][:MAX_ZONE_SLOTS]):
        base = OBS_ZONE_START + slot * 7
        vec[base] = 1.0  # present
        for ci, card in enumerate(entity_obs["exposed_cards"][:3]):
            is_resource = "pct_need_met" in card
            vec[base + 1 + ci] = 1.0 if is_resource else 0.0
            vec[base + 4 + ci] = card.get("pct_need_met", card.get("base_rating", 0.0) / 10.0)

    # Event log [160:205]
    for slot, event in enumerate(obs.get("zone_event_log", [])[:EVENT_LOG_SLOTS]):
        base = OBS_LOG_START + slot * 9
        atype = event.get("action_type", "DO_NOTHING")
        if atype in ALL_ACTION_TYPES:
            vec[base + ALL_ACTION_TYPES.index(atype)] = 1.0
        vec[base + 7] = 1.0 if event.get("success") else 0.0
        vec[base + 8] = 1.0 if event.get("agent_id") == agent_id else 0.0

    # Global [205:208]
    vec[OBS_GLOBAL_START]     = obs.get("collective_welfare", 0.0)
    vec[OBS_GLOBAL_START + 1] = obs.get("tick", 0) / max(env.ticks_per_phase, 1)
    vec[OBS_GLOBAL_START + 2] = obs.get("zone_id", 0) / max(env.n_zones, 1)

    return vec


# ---------------------------------------------------------------------------
# Action mask
# ---------------------------------------------------------------------------

def action_mask(agent_id: int, env: UnknownRoomEnv) -> np.ndarray:
    """Binary mask over ACTION_SIZE slots; 1 = valid."""
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    valid = env.valid_actions(agent_id)
    entity = env.entities[agent_id]
    zone = env.zones[entity.zone_id]
    zone_entity_ids = [eid for eid in zone.entity_ids if eid != agent_id]

    if ActionType.DO_NOTHING in valid:
        mask[ACT_DO_NOTHING] = 1.0
    if ActionType.SHUFFLE in valid:
        mask[ACT_SHUFFLE] = 1.0
    if ActionType.MOVE in valid:
        for z in range(env.n_zones):
            if z != entity.zone_id:
                mask[ACT_MOVE_BASE + z] = 1.0
    if ActionType.INTERACT in valid:
        for eid in zone_entity_ids:
            if 0 <= eid < _N_TOTAL:
                mask[ACT_INTERACT_BASE + eid] = 1.0
    if ActionType.GIVE in valid:
        for eid in zone_entity_ids:
            if 0 <= eid < _N_TOTAL:
                mask[ACT_GIVE_BASE + eid] = 1.0
    if ActionType.CLAIM_SHARE in valid:
        mask[ACT_CLAIM_SHARE] = 1.0
    if ActionType.CLAIM_ALL in valid:
        mask[ACT_CLAIM_ALL] = 1.0

    return mask


# ---------------------------------------------------------------------------
# Action index → Action dataclass
# ---------------------------------------------------------------------------

def decode_action(idx: int, agent_id: int, env: UnknownRoomEnv) -> Action:
    """Convert a flat action index back to an Action for the environment."""
    entity = env.entities[agent_id]

    if idx == ACT_DO_NOTHING:
        return Action(agent_id=agent_id, action_type=ActionType.DO_NOTHING)

    if idx == ACT_SHUFFLE:
        import random
        new_exposed = random.sample(range(6), 3)
        return Action(agent_id=agent_id, action_type=ActionType.SHUFFLE,
                      exposed_indices=new_exposed)

    if ACT_MOVE_BASE <= idx < ACT_MOVE_BASE + _N_ZONES:
        zone_id = idx - ACT_MOVE_BASE
        return Action(agent_id=agent_id, action_type=ActionType.MOVE, target_id=zone_id)

    if ACT_INTERACT_BASE <= idx < ACT_INTERACT_BASE + _N_TOTAL:
        target_id = idx - ACT_INTERACT_BASE
        return Action(agent_id=agent_id, action_type=ActionType.INTERACT, target_id=target_id)

    if ACT_GIVE_BASE <= idx < ACT_GIVE_BASE + _N_TOTAL:
        target_id = idx - ACT_GIVE_BASE
        # Give 25% of the most-abundant resource (highest holdings / need ratio)
        best_r = max(ResourceType,
                     key=lambda r: entity.holdings.get(r, 0.0) /
                                   max(entity.need_levels.get(r, 1.0), 1e-6))
        amount = entity.holdings.get(best_r, 0.0) * 0.25
        return Action(agent_id=agent_id, action_type=ActionType.GIVE,
                      target_id=target_id, resource_type=best_r, amount=amount)

    if idx == ACT_CLAIM_SHARE:
        return Action(agent_id=agent_id, action_type=ActionType.CLAIM_SHARE)

    if idx == ACT_CLAIM_ALL:
        return Action(agent_id=agent_id, action_type=ActionType.CLAIM_ALL)

    return Action(agent_id=agent_id, action_type=ActionType.DO_NOTHING)
