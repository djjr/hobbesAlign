from __future__ import annotations
import random

from unknown_room.actions import Action, ActionType
from unknown_room.entities import ResourceType
from unknown_room.policies.base import AgentPolicy


class RandomAgent(AgentPolicy):
    """
    Baseline policy: choose uniformly at random from valid actions.
    All target and parameter choices are also random.
    """

    def __init__(self, agent_id: int, seed: int | None = None):
        super().__init__(agent_id)
        self._rng = random.Random(seed)

    def act(self, observation: dict, valid_actions: list[ActionType]) -> Action:
        action_type = self._rng.choice(valid_actions)
        target_id = None
        resource_type = None
        amount = None
        exposed_indices: list[int] = []

        zone_entity_ids = [e["id"] for e in observation.get("zone_entities", [])]

        if action_type == ActionType.INTERACT:
            if zone_entity_ids:
                target_id = self._rng.choice(zone_entity_ids)

        elif action_type == ActionType.GIVE:
            if zone_entity_ids:
                target_id = self._rng.choice(zone_entity_ids)
                resource_type = self._rng.choice(list(ResourceType))
                own_holdings = observation.get("own_holdings", {})
                available = own_holdings.get(resource_type.value, 0.0)
                amount = self._rng.uniform(0.0, available) if available > 0 else 0.0

        elif action_type == ActionType.MOVE:
            own_zone = observation.get("zone_id", 0)
            # Zone topology is fully connected; any other zone is reachable.
            # n_zones inferred from observation; defaults to 5 if not present.
            n_zones = observation.get("n_zones", 5)
            options = [z for z in range(n_zones) if z != own_zone]
            target_id = self._rng.choice(options) if options else 0

        elif action_type == ActionType.SHUFFLE:
            all_indices = list(range(6))
            exposed_indices = self._rng.sample(all_indices, 3)

        # CLAIM_SHARE, CLAIM_ALL, DO_NOTHING need no extra fields

        return Action(
            agent_id=self.agent_id,
            action_type=action_type,
            target_id=target_id,
            resource_type=resource_type,
            amount=amount,
            exposed_indices=exposed_indices,
        )
