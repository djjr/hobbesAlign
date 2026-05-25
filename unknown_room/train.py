"""
PPO training loop for Unknown Room (Phase 1).

All agents share a single policy network (parameter sharing).
Each episode contributes transitions from all living agents.
After each episode, PPO updates the shared policy.

Usage:
    python -m unknown_room.train --reward individual --episodes 500
    python -m unknown_room.train --reward collective --episodes 500 --out runs/trained_collective
    python -m unknown_room.train --reward misspecified --episodes 500
"""
from __future__ import annotations
import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from unknown_room.environment import UnknownRoomEnv
from unknown_room.policies.mlp_policy import MLPPolicy
from unknown_room import spaces as ur_spaces
from unknown_room import rewards as reward_module

REWARD_FNS = {
    "individual":   reward_module.reward_individual,
    "collective":   reward_module.reward_collective,
    "misspecified": reward_module.reward_misspecified,
    "mixed_05":     reward_module.reward_mixed(0.5),
}


# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    # Architecture
    hidden_size:  int   = 128
    n_layers:     int   = 2

    # PPO
    lr:           float = 3e-4
    gamma:        float = 0.99
    gae_lambda:   float = 0.95
    clip_eps:     float = 0.2
    entropy_coef: float = 0.01
    value_coef:   float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs:   int   = 4
    minibatch_size: int = 64

    # Training
    episodes:     int   = 500
    ticks:        int   = 40
    seed:         int   = 0


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    obs:       list = field(default_factory=list)
    actions:   list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values:    list = field(default_factory=list)
    rewards:   list = field(default_factory=list)
    masks:     list = field(default_factory=list)   # action masks
    dones:     list = field(default_factory=list)

    def add(self, obs, action, log_prob, value, reward, mask, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# GAE advantage computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,   # (T,)
    values:  torch.Tensor,   # (T,)
    dones:   torch.Tensor,   # (T,)  1 = terminal
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (advantages, returns), both shape (T,)."""
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_val = 0.0 if dones[t] else (values[t + 1] if t + 1 < T else 0.0)
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    policy: MLPPolicy,
    optimizer: optim.Optimizer,
    rollout: Rollout,
    cfg: PPOConfig,
    device: torch.device,
):
    obs_t      = torch.FloatTensor(np.array(rollout.obs)).to(device)
    actions_t  = torch.LongTensor(np.array(rollout.actions)).to(device)
    old_lp_t   = torch.FloatTensor(np.array(rollout.log_probs)).to(device)
    values_t   = torch.FloatTensor(np.array(rollout.values)).to(device)
    rewards_t  = torch.FloatTensor(np.array(rollout.rewards)).to(device)
    masks_t    = torch.FloatTensor(np.array(rollout.masks)).to(device)
    dones_t    = torch.FloatTensor(np.array(rollout.dones)).to(device)

    advantages, returns = compute_gae(
        rewards_t.cpu(), values_t.cpu(), dones_t.cpu(),
        cfg.gamma, cfg.gae_lambda
    )
    advantages = advantages.to(device)
    returns    = returns.to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    T = len(rollout)
    indices = np.arange(T)
    total_loss = 0.0

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, T, cfg.minibatch_size):
            mb = indices[start: start + cfg.minibatch_size]
            if len(mb) == 0:
                continue

            dist, values_pred = policy(obs_t[mb], masks_t[mb])
            log_probs = dist.log_prob(actions_t[mb])
            entropy   = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_lp_t[mb])
            adv   = advantages[mb]

            actor_loss = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv,
            ).mean()
            value_loss  = F.mse_loss(values_pred, returns[mb])
            loss = actor_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

    return total_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    reward_name: str = "individual",
    cfg: PPOConfig | None = None,
    out_dir: str | None = None,
    verbose: bool = True,
) -> dict:
    if cfg is None:
        cfg = PPOConfig()

    device = torch.device("cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    reward_fn = REWARD_FNS[reward_name]
    env = UnknownRoomEnv(
        ticks_per_phase=cfg.ticks,
        reward_fn=reward_fn,
        seed=cfg.seed,
    )

    policy    = MLPPolicy(hidden_size=cfg.hidden_size, n_layers=cfg.n_layers).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    welfare_history = []
    survival_history = []

    for episode in range(cfg.episodes):
        agent_rng = random.Random(cfg.seed + episode)
        rollout = Rollout()

        raw_obs = env.reset()
        # Build per-agent obs arrays and masks
        agent_obs  = {
            aid: ur_spaces.obs_to_array(obs, env, aid)
            for aid, obs in raw_obs.items()
        }
        agent_mask = {
            aid: ur_spaces.action_mask(aid, env)
            for aid in env.agents
        }

        while not env.done:
            actions_int = {}
            step_data = {}

            for aid in env.agents:
                obs_t  = torch.FloatTensor(agent_obs[aid]).unsqueeze(0).to(device)
                mask_t = torch.FloatTensor(agent_mask[aid]).unsqueeze(0).to(device)

                with torch.no_grad():
                    action, log_prob, value = policy.act(obs_t, mask_t)

                actions_int[aid] = action.item()
                step_data[aid] = (
                    agent_obs[aid],
                    action.item(),
                    log_prob.item(),
                    value.item(),
                    agent_mask[aid],
                )

            # Decode action indices to Action objects
            decoded = {
                aid: ur_spaces.decode_action(idx, aid, env)
                for aid, idx in actions_int.items()
            }

            raw_obs, rewards, terminations, truncations, _ = env.step(decoded)

            # Record transitions
            for aid, (obs_arr, act, lp, val, mask_arr) in step_data.items():
                done = terminations.get(aid, True) or truncations.get(aid, False)
                rew  = rewards.get(aid, 0.0)
                rollout.add(obs_arr, act, lp, val, rew, mask_arr, float(done))

            # Update obs/masks for next step
            agent_obs = {
                aid: ur_spaces.obs_to_array(obs, env, aid)
                for aid, obs in raw_obs.items()
            }
            agent_mask = {aid: ur_spaces.action_mask(aid, env) for aid in env.agents}

        # PPO update after episode
        if len(rollout) > 0:
            ppo_update(policy, optimizer, rollout, cfg, device)

        welfare_history.append(env.collective_welfare)
        survival_history.append(len(env.agents))

        if verbose and (episode % 50 == 0 or episode == cfg.episodes - 1):
            recent_welfare = np.mean(welfare_history[-20:])
            print(
                f"Episode {episode:4d}/{cfg.episodes} | "
                f"welfare={env.collective_welfare:.3f} | "
                f"avg20={recent_welfare:.3f} | "
                f"living={len(env.agents)} | "
                f"transitions={len(rollout)}"
            )

    # Save outputs
    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), out_path / "policy.pt")
        with open(out_path / "training_log.json", "w") as f:
            json.dump({
                "reward": reward_name,
                "welfare_history": welfare_history,
                "survival_history": survival_history,
                "config": cfg.__dict__,
            }, f, indent=2)
        print(f"\nSaved policy and log to {out_dir}/")

    return {
        "reward": reward_name,
        "final_welfare": welfare_history[-1],
        "mean_welfare_last20": float(np.mean(welfare_history[-20:])),
        "welfare_history": welfare_history,
        "survival_history": survival_history,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Unknown Room agents with PPO.")
    parser.add_argument("--reward", default="individual", choices=list(REWARD_FNS))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", default=None, help="Output directory for policy + log.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cfg = PPOConfig(
        episodes=args.episodes,
        ticks=args.ticks,
        seed=args.seed,
        hidden_size=args.hidden,
        lr=args.lr,
    )
    train(reward_name=args.reward, cfg=cfg, out_dir=args.out, verbose=not args.quiet)


if __name__ == "__main__":
    main()
