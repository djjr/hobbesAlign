"""
Shared MLP actor-critic policy used by the PPO training loop.

All agents share one network (parameter sharing). This is standard for
homogeneous MARL and dramatically reduces sample complexity.

Architecture:
  obs → shared trunk (2 hidden layers) → actor head (logits) + critic head (value)

Action masking: invalid logits are set to -1e9 before softmax so they get
zero probability. The mask is stored in the rollout buffer alongside obs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from unknown_room.spaces import OBS_SIZE, ACTION_SIZE


class MLPPolicy(nn.Module):

    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()

        layers = []
        in_dim = obs_size
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_size), nn.Tanh()]
            in_dim = hidden_size

        self.trunk  = nn.Sequential(*layers)
        self.actor  = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: torch.Tensor,           # (batch, OBS_SIZE)
        mask: torch.Tensor,          # (batch, ACTION_SIZE)  float, 1=valid 0=invalid
    ) -> tuple[Categorical, torch.Tensor]:
        """Return (action distribution, value estimate)."""
        h = self.trunk(obs)
        logits = self.actor(h)
        logits = logits + (1.0 - mask) * (-1e9)   # mask invalid actions
        dist   = Categorical(logits=logits)
        value  = self.critic(h).squeeze(-1)        # (batch,)
        return dist, value

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or argmax) an action. Returns (action, log_prob, value)."""
        dist, value = self.forward(obs, mask)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), value
