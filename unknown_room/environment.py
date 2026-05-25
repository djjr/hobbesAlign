from __future__ import annotations
import random
from collections import defaultdict
from statistics import mean
from typing import Callable

import numpy as np

from unknown_room.actions import Action, ActionRecord, ActionType
from unknown_room.entities import (
    Entity, EntityProfile, ResourceCard, ResourceType, StrengthType,
    N_AGENTS, N_REACTIVE, N_ZONES, TICKS_PER_PHASE, METABOLISM_RATE,
)
from unknown_room.init_world import init_world
from unknown_room.logger import TickLogger
from unknown_room.resolution import (
    combined_strengths, group_by_target, log_actions,
    random_sequence, resolve_claim_share, resolve_contested_claim,
    resolve_do_nothing, resolve_give, resolve_interaction,
    resolve_move, resolve_shuffle,
)
from unknown_room.zones import JointPool, Zone

RewardFn = Callable[[Entity, "UnknownRoomEnv"], float]

# Zone adjacency: fully connected (any zone reachable from any zone)
# DESIGN QUESTION: topology is fully connected per spec; change here if needed
def adjacent_zones(zone_id: int, n_zones: int) -> list[int]:
    return [z for z in range(n_zones) if z != zone_id]


class UnknownRoomEnv:
    """
    Phase 1 Unknown Room environment.
    Interface mirrors PettingZoo ParallelEnv semantics; PettingZoo dependency
    is optional for Phase 1 — the step() / reset() / observe() contract is
    implemented here directly so the RL training loop can wrap it later.
    """

    metadata = {"name": "unknown_room_v0", "phase": 1}

    def __init__(
        self,
        n_agents: int = N_AGENTS,
        n_reactive: int = N_REACTIVE,
        n_zones: int = N_ZONES,
        ticks_per_phase: int = TICKS_PER_PHASE,
        reward_fn: RewardFn | None = None,
        metabolism_rate: float = METABOLISM_RATE,
        seed: int | None = None,
        log_path: str | None = None,
    ):
        self.n_agents = n_agents
        self.n_reactive = n_reactive
        self.n_zones = n_zones
        self.ticks_per_phase = ticks_per_phase
        self.reward_fn = reward_fn
        self.metabolism_rate = metabolism_rate
        self.seed = seed
        self.logger = TickLogger(log_path)

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # World state (populated by reset)
        self.entities: dict[int, Entity] = {}
        self.profiles: dict[int, EntityProfile] = {}
        self.zones: dict[int, Zone] = {}
        self.pools: dict[int, JointPool] = {}          # pool_id -> JointPool
        self._agent_pool: dict[int, int] = {}          # agent_id -> pool_id
        self._next_pool_id: list[int] = [0]            # mutable wrapper

        self.tick: int = 0
        self.collective_welfare: float = 0.0
        self._done: bool = False

        # Per-tick tracking
        self._tick_actions: dict[int, ActionType] = {}  # for contested claim resolution
        self._zone_event_log: dict[int, list[ActionRecord]] = defaultdict(list)

        self.reset()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    @property
    def possible_agents(self) -> list[int]:
        return list(range(self.n_agents))

    @property
    def agents(self) -> list[int]:
        """Living strategic agents."""
        return [
            i for i in self.possible_agents
            if not self.entities[i].is_dead
        ]

    @property
    def done(self) -> bool:
        return self._done

    def reset(self) -> dict[int, dict]:
        self._rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)

        self.entities, self.profiles, self.zones = init_world(
            n_agents=self.n_agents,
            n_reactive=self.n_reactive,
            n_zones=self.n_zones,
            rng=self._rng,
            np_rng=self._np_rng,
        )
        self.pools = {}
        self._agent_pool = {}
        self._next_pool_id = [0]
        self.tick = 0
        self.collective_welfare = 0.0
        self._done = False
        self._zone_event_log = defaultdict(list)

        self._update_resource_cards()
        self._update_collective_welfare()

        return {aid: self._build_observation(aid) for aid in self.agents}

    def step(self, actions: dict[int, Action]) -> tuple[dict, dict, dict, dict, dict]:
        """
        Execute one tick.
        Returns (observations, rewards, terminations, truncations, infos)
        following PettingZoo ParallelEnv convention.
        """
        if self._done:
            raise RuntimeError("Episode is done; call reset().")

        # Step 1: Validate / mask invalid actions
        actions = self._mask_invalid_actions(actions)
        self._tick_actions = {a.agent_id: a.action_type for a in actions.values()}

        # Step 2: Random sequencing
        sequence = random_sequence(list(actions.keys()), self._rng)

        # Step 3: Log actions, marking engaged targets
        logged, engaged, skipped = log_actions(
            actions, sequence, self.tick, self.entities, self.zones
        )

        # Step 4: Group INTERACT actions by target
        coalitions = group_by_target(logged)

        # Step 5: Resolve all interactions and other actions
        all_records = list(skipped)
        all_records += self._resolve_all(logged, coalitions)

        # Step 6: Apply pool cleanup (expired/empty pools)
        self._cleanup_pools()

        # Step 6b: Metabolism — each agent burns a fraction of their need level
        self._consume_resources()

        # Step 7: Update resource cards from current holdings
        self._update_resource_cards()

        # Step 8: Check death conditions
        self._check_deaths()

        # Step 9: Update collective welfare
        self._update_collective_welfare()

        # Step 10: Advance tick and check termination
        self.tick += 1
        living = self.agents
        if self.tick >= self.ticks_per_phase or len(living) == 0:
            self._done = True

        # Update zone event log (keep last 5 events per zone)
        for rec in all_records:
            self._zone_event_log[rec.zone_id].append(rec)
        for zid in self._zone_event_log:
            self._zone_event_log[zid] = self._zone_event_log[zid][-5:]

        # Log tick
        self.logger.log_tick(
            tick=self.tick - 1,
            sequence=sequence,
            action_records=all_records,
            collective_welfare=self.collective_welfare,
            living_count=len(living),
        )

        # Step 11: Build outputs
        observations = {aid: self._build_observation(aid) for aid in living}
        rewards = {}
        if self.reward_fn is not None:
            rewards = {
                aid: self.reward_fn(self.entities[aid], self)
                for aid in living
            }
        terminations = {aid: self._done for aid in living}
        truncations = {aid: False for aid in living}
        infos: dict[int, dict] = {aid: {} for aid in living}

        return observations, rewards, terminations, truncations, infos

    # -----------------------------------------------------------------------
    # Valid action mask
    # -----------------------------------------------------------------------

    def valid_actions(self, agent_id: int) -> list[ActionType]:
        valid = [ActionType.SHUFFLE, ActionType.DO_NOTHING, ActionType.MOVE]

        in_pool = agent_id in self._agent_pool
        zone = self.zones[self.entities[agent_id].zone_id]

        if not in_pool and len(zone.entity_ids) > 1:
            # INTERACT and GIVE unavailable while in an active pool to prevent
            # an agent from joining multiple pools simultaneously.
            valid += [ActionType.INTERACT, ActionType.GIVE]

        if in_pool:
            valid += [ActionType.CLAIM_SHARE, ActionType.CLAIM_ALL]

        return valid

    def _mask_invalid_actions(self, actions: dict[int, Action]) -> dict[int, Action]:
        """Replace invalid actions with DO_NOTHING."""
        masked = {}
        for aid, action in actions.items():
            if aid not in self.entities or self.entities[aid].is_dead:
                continue
            valid = self.valid_actions(aid)
            if action.action_type not in valid:
                masked[aid] = Action(agent_id=aid, action_type=ActionType.DO_NOTHING)
            else:
                masked[aid] = action
        return masked

    # -----------------------------------------------------------------------
    # Resolution dispatcher
    # -----------------------------------------------------------------------

    def _resolve_all(
        self,
        logged: list[Action],
        coalitions: dict[int, list[Action]],
    ) -> list[ActionRecord]:
        records: list[ActionRecord] = []
        resolved_interact_agents: set[int] = set()

        # Resolve all INTERACT coalitions first
        for target_id, initiator_actions in coalitions.items():
            initiator_ids = [a.agent_id for a in initiator_actions]
            recs, new_pool = resolve_interaction(
                initiator_ids=initiator_ids,
                target_id=target_id,
                tick=self.tick,
                entities=self.entities,
                profiles=self.profiles,
                pools=self.pools,
                next_pool_id=self._next_pool_id,
                rng=self._rng,
            )
            records.extend(recs)
            resolved_interact_agents.update(initiator_ids)

            if new_pool is not None:
                self.pools[new_pool.pool_id] = new_pool
                for pid in new_pool.participant_ids:
                    self._agent_pool[pid] = new_pool.pool_id

        # Resolve CLAIM actions — group contested CLAIM_ALL per pool
        claim_all_by_pool: dict[int, list[int]] = defaultdict(list)
        for action in logged:
            aid = action.agent_id
            if action.action_type == ActionType.CLAIM_ALL:
                pool_id = self._agent_pool.get(aid)
                if pool_id is not None:
                    claim_all_by_pool[pool_id].append(aid)
            elif action.action_type == ActionType.CLAIM_SHARE:
                pool_id = self._agent_pool.get(aid)
                if pool_id is not None:
                    pool = self.pools[pool_id]
                    records.append(
                        resolve_claim_share(action, pool, self.entities, self.tick)
                    )
                    self._agent_pool.pop(aid, None)

        for pool_id, claimants in claim_all_by_pool.items():
            pool = self.pools[pool_id]
            recs = resolve_contested_claim(
                pool, claimants, self.entities, self.tick, self._rng
            )
            records.extend(recs)
            for cid in claimants:
                self._agent_pool.pop(cid, None)

        # Resolve remaining non-INTERACT actions in logged order
        for action in logged:
            aid = action.agent_id
            atype = action.action_type

            if atype == ActionType.INTERACT:
                continue  # already handled above
            if atype in (ActionType.CLAIM_SHARE, ActionType.CLAIM_ALL):
                continue  # already handled above

            if atype == ActionType.GIVE:
                records.append(resolve_give(action, self.entities, self.tick))
            elif atype == ActionType.MOVE:
                records.append(
                    resolve_move(action, self.entities, self.zones, self.tick)
                )
                # Moving forfeits pool membership
                if aid in self._agent_pool:
                    pool_id = self._agent_pool.pop(aid)
                    pool = self.pools.get(pool_id)
                    if pool and aid in pool.participant_ids:
                        pool.participant_ids.remove(aid)
            elif atype == ActionType.SHUFFLE:
                records.append(resolve_shuffle(action, self.entities, self.tick))
            elif atype == ActionType.DO_NOTHING:
                records.append(resolve_do_nothing(action, self.entities, self.tick))

        return records

    # -----------------------------------------------------------------------
    # World update steps
    # -----------------------------------------------------------------------

    def _consume_resources(self) -> None:
        """Metabolism: deplete each strategic agent's holdings by metabolism_rate × need_level."""
        if self.metabolism_rate == 0.0:
            return
        for entity in self.entities.values():
            if entity.entity_type != "strategic" or entity.is_dead:
                continue
            for r in ResourceType:
                cost = self.metabolism_rate * entity.need_levels.get(r, 1.0)
                entity.holdings[r] = max(0.0, entity.holdings.get(r, 0.0) - cost)

    def _update_resource_cards(self) -> None:
        for entity in self.entities.values():
            if entity.is_dead:
                continue
            for card in entity.resource_cards:
                need = entity.need_levels.get(card.resource_type, 1.0)
                held = entity.holdings.get(card.resource_type, 0.0)
                card.pct_need_met = min(1.0, held / need) if need > 0 else 0.0

    def _check_deaths(self) -> None:
        for entity in self.entities.values():
            if entity.entity_type != "strategic":
                continue
            if entity.is_dead:
                continue
            if not entity.is_alive:
                entity.is_dead = True
                # Remove from zone
                zone = self.zones[entity.zone_id]
                if entity.id in zone.entity_ids:
                    zone.entity_ids.remove(entity.id)
                # Remove from any pool
                if entity.id in self._agent_pool:
                    pool_id = self._agent_pool.pop(entity.id)
                    pool = self.pools.get(pool_id)
                    if pool and entity.id in pool.participant_ids:
                        pool.participant_ids.remove(entity.id)

    def _cleanup_pools(self) -> None:
        """Remove pools with no remaining participants or no holdings."""
        dead_pools = [
            pid for pid, pool in self.pools.items()
            if len(pool.participant_ids) == 0 or pool.is_empty()
        ]
        for pid in dead_pools:
            pool = self.pools.pop(pid)
            # Release any remaining participants still mapped to this pool
            for aid in list(self._agent_pool):
                if self._agent_pool[aid] == pid:
                    del self._agent_pool[aid]

    def _update_collective_welfare(self) -> None:
        living = [
            e for e in self.entities.values()
            if e.entity_type == "strategic" and not e.is_dead
        ]
        if not living:
            self.collective_welfare = 0.0
            return
        self.collective_welfare = mean(
            mean(card.pct_need_met for card in e.resource_cards)
            for e in living
        )

    # -----------------------------------------------------------------------
    # Observations
    # -----------------------------------------------------------------------

    def _build_observation(self, agent_id: int) -> dict:
        agent = self.entities[agent_id]
        zone = self.zones[agent.zone_id]
        pool_id = self._agent_pool.get(agent_id)
        pool = self.pools.get(pool_id) if pool_id is not None else None

        zone_entities = []
        for eid in zone.entity_ids:
            if eid == agent_id:
                continue
            other = self.entities[eid]
            exposed = [other.all_cards[i] for i in other.exposed_indices]
            zone_entities.append({
                "id": eid,
                "exposed_cards": [
                    {"type": c.resource_type.value, "pct_need_met": c.pct_need_met}
                    if isinstance(c, ResourceCard)
                    else {"type": c.strength_type.value, "base_rating": c.base_rating}
                    for c in exposed
                ],
            })

        return {
            "own_resource_cards": [
                {"type": c.resource_type.value, "pct_need_met": c.pct_need_met}
                for c in agent.resource_cards
            ],
            "own_strength_cards": [
                {"type": c.strength_type.value, "effective_rating": v}
                for c, v in zip(
                    agent.strength_cards,
                    agent.effective_strengths.values()
                )
            ],
            "own_exposed_indices": list(agent.exposed_indices),
            "own_holdings": {r.value: agent.holdings.get(r, 0.0) for r in ResourceType},
            "own_pool": {
                "pool_id": pool.pool_id,
                "participant_ids": list(pool.participant_ids),
                "holdings": {r.value: pool.holdings.get(r, 0.0) for r in ResourceType},
            } if pool else None,
            "zone_entities": zone_entities,
            "zone_event_log": [
                {
                    "tick": r.tick,
                    "agent_id": r.agent_id,
                    "action_type": r.action_type.value,
                    "target_id": r.target_id,
                    "success": r.success,
                }
                for r in self._zone_event_log.get(agent.zone_id, [])
            ],
            "valid_actions": [a.value for a in self.valid_actions(agent_id)],
            "zone_id": agent.zone_id,
            "n_zones": self.n_zones,
            "collective_welfare": self.collective_welfare,
            "tick": self.tick,
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _recent_events(self, zone_id: int, n: int = 5) -> list[ActionRecord]:
        return self._zone_event_log.get(zone_id, [])[-n:]

    def _in_active_pool(self, agent_id: int) -> bool:
        return agent_id in self._agent_pool
