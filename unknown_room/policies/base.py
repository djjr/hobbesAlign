from __future__ import annotations
from abc import ABC, abstractmethod

from unknown_room.actions import Action, ActionType


class AgentPolicy(ABC):

    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    @abstractmethod
    def act(self, observation: dict, valid_actions: list[ActionType]) -> Action:
        """
        Given partial observation and valid action types, return one Action.
        """
        ...

    def update(self, observation: dict, action: Action, reward: float) -> None:
        """Called after each tick. No-op for non-learning agents."""
        pass
