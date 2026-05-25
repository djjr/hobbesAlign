"""
PettingZoo ParallelEnv wrapper around UnknownRoomEnv.

Converts our internal environment to the standard PettingZoo interface so any
PettingZoo-compatible RL library can consume it. The wrapper handles:
  - String agent IDs (PettingZoo convention) vs int agent IDs (our env)
  - Observation flattening to numpy arrays
  - Action decoding from flat int to Action dataclass
  - Action masks surfaced in info dicts
"""
from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium import spaces as gym_spaces
from pettingzoo import ParallelEnv

from unknown_room import spaces as ur_spaces
from unknown_room.environment import UnknownRoomEnv
from unknown_room.rewards import reward_individual


class UnknownRoomPZEnv(ParallelEnv):
    """PettingZoo ParallelEnv wrapping UnknownRoomEnv."""

    metadata = {"name": "unknown_room_v0", "render_modes": []}

    def __init__(self, env_kwargs: dict | None = None):
        super().__init__()
        env_kwargs = env_kwargs or {}
        self._env = UnknownRoomEnv(**env_kwargs)

        self.possible_agents = [f"agent_{i}" for i in self._env.possible_agents]
        self.agents = list(self.possible_agents)
        self._name_to_id = {f"agent_{i}": i for i in self._env.possible_agents}
        self._id_to_name = {i: f"agent_{i}" for i in self._env.possible_agents}

        self._obs_space = gym_spaces.Box(
            low=0.0, high=1.0,
            shape=(ur_spaces.OBS_SIZE,),
            dtype=np.float32,
        )
        self._act_space = gym_spaces.Discrete(ur_spaces.ACTION_SIZE)

    # ------------------------------------------------------------------
    # Required PettingZoo properties
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> gym_spaces.Space:
        return self._obs_space

    def action_space(self, agent: str) -> gym_spaces.Space:
        return self._act_space

    # ------------------------------------------------------------------
    # PettingZoo interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self._env.seed = seed
        raw_obs = self._env.reset()
        self.agents = [self._id_to_name[aid] for aid in self._env.agents]

        obs   = self._wrap_obs(raw_obs)
        infos = self._build_infos()
        return obs, infos

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[dict, dict, dict, dict, dict]:
        int_actions = {
            self._name_to_id[name]: ur_spaces.decode_action(
                int(act), self._name_to_id[name], self._env
            )
            for name, act in actions.items()
        }

        raw_obs, rewards, terminations, truncations, _ = self._env.step(int_actions)
        self.agents = [self._id_to_name[aid] for aid in self._env.agents]

        obs   = self._wrap_obs(raw_obs)
        rews  = {self._id_to_name[aid]: r for aid, r in rewards.items()}
        terms = {self._id_to_name[aid]: t for aid, t in terminations.items()}
        trunc = {self._id_to_name[aid]: t for aid, t in truncations.items()}
        infos = self._build_infos()

        return obs, rews, terms, trunc, infos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wrap_obs(self, raw_obs: dict[int, dict]) -> dict[str, np.ndarray]:
        return {
            self._id_to_name[aid]: ur_spaces.obs_to_array(obs, self._env, aid)
            for aid, obs in raw_obs.items()
        }

    def _build_infos(self) -> dict[str, dict]:
        return {
            name: {"action_mask": ur_spaces.action_mask(self._name_to_id[name], self._env)}
            for name in self.agents
        }

    @property
    def collective_welfare(self) -> float:
        return self._env.collective_welfare

    @property
    def done(self) -> bool:
        return self._env.done
