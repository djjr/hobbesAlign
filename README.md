# hobbesAlign — The Unknown Room

A multi-agent simulation of *The Unknown Room*, a classroom coordination game designed to teach the four AI alignment problems concretely. Agents must extract resources, manage needs, and decide whether to cooperate or defect — with collective welfare as the visible outcome metric.

---

## What This Is

**The Unknown Room** is a social coordination game for ~30 students. Players control strategic entities in a shared world alongside reactive entities (resource nodes). Players see limited information — exposed cards from nearby entities — and must interact, trade, and form coalitions to meet their own resource needs.

The simulation serves two purposes:
1. **Balance testing** — verify game mechanics produce interesting dynamics before classroom deployment
2. **Pedagogical demonstration** — show students how agents trained on different reward functions produce radically different collective outcomes, making alignment problems concrete

Four alignment problems are introduced across four game phases:
| Phase | Alignment Problem | Status |
|---|---|---|
| 1 | Human | **Implemented** |
| 2 | Organizational | Deferred |
| 3 | Expert | Deferred |
| 4 | Machine | Deferred |

---

## Quick Start

```bash
# Single run (random agents, 20 ticks)
python -m unknown_room.run

# Seeded run with log output
python -m unknown_room.run --ticks 60 --seed 42 --log runs/ep.json

# Choose reward function
python -m unknown_room.run --ticks 60 --seed 42 --reward misspecified --log runs/mis.json

# Train RL agents (PPO, parameter sharing across all agents)
python -m unknown_room.train --reward individual --episodes 300 --out runs/individual
python -m unknown_room.train --reward collective --episodes 300 --out runs/collective
python -m unknown_room.train --reward misspecified --episodes 300 --out runs/misspecified

# Visualize a single run
python -m unknown_room.visualize runs/ep.json --out figures/dashboard.png

# Compare reward functions side by side
python -m unknown_room.visualize runs/a.json runs/b.json runs/c.json \
  --labels "Individual" "Collective" "Misspecified" \
  --out figures/comparison.png

# Plot RL training curves
python -m unknown_room.visualize runs/individual/training_log.json \
  runs/collective/training_log.json runs/misspecified/training_log.json \
  --labels "Individual" "Collective" "Misspecified" \
  --training --out figures/training_curves.png
```

**Reward function options:** `individual`, `collective`, `misspecified`, `mixed_05`

---

## Repository Structure

```
unknown_room/
├── entities.py          Entity, ResourceCard, StrengthCard, EntityProfile; all constants
├── actions.py           Action, ActionType, ActionRecord
├── zones.py             Zone, JointPool
├── environment.py       UnknownRoomEnv — the full tick pipeline
├── resolution.py        Outcome function, conflict resolution, action resolvers
├── rewards.py           Pluggable reward functions
├── init_world.py        World initialization (entities, zones, profiles)
├── logger.py            Structured JSON tick logging
├── spaces.py            RL layer: obs_to_array(), action_mask(), decode_action()
├── visualize.py         Post-run dashboard, comparison charts, training curves
├── run.py               Entry point: run N ticks with random agents, write log
├── train.py             PPO training loop with GAE, action masking, parameter sharing
└── policies/
    ├── base.py              AgentPolicy ABC
    ├── random_agent.py      RandomAgent (baseline)
    └── mlp_policy.py        Shared MLP actor-critic with action masking
wrappers/
└── pettingzoo_env.py    PettingZoo ParallelEnv adapter

runs/                    JSON log files and training outputs (gitignored)
figures/                 Output figures (gitignored)
unknown_room_rules_v0.3.md     Human-readable game design document
claude_code_handoff.md         Original engineering spec (Phase 1)
HANDOFF.md                     Current development handoff (see below)
```

---

## Core Mechanics (Phase 1)

**Entities:** 30 strategic (agent-controlled) + 30 reactive (resource nodes), distributed across 5 fully-connected zones.

**Cards:** Each entity has 6 cards — 3 Resource (Food/Shelter/Energy, showing % need met) and 3 Strength (Physical/Cunning/Influence, showing base rating). Each entity exposes exactly 3 at a time; the other 3 are hidden.

**Strength modifier:** Resource levels modify effective strength via a step function (−3 at 0–10%, up to +2 at 90–100%), creating feedback between resource scarcity and combat power.

**Tick pipeline:** Validate → Sequence (random) → Log (skip engaged targets) → Group by target → Resolve → Cleanup pools → **Metabolism** → Update cards → Check deaths → Update collective welfare.

**Metabolism:** Each tick, every strategic agent's holdings decay by `metabolism_rate × need_level` per resource. Default rate is 0.05. This creates permanent resource pressure and is the primary forcing function for reward divergence.

**Interactions:** Solo interactions against reactive entities always succeed; yield is deficit-weighted across the 3 resource types. Coalition interactions produce a joint pool; participants then CLAIM_SHARE or contest with CLAIM_ALL.

**Collective welfare:** Population-weighted scalar — sum of (mean % need-met per surviving agent) divided by original agent count. Dead agents count as 0; the denominator never shrinks. Updated every tick and visible to all agents.

---

## RL Training

Agents are trained with PPO (Proximal Policy Optimization) and parameter sharing — all 30 agents share one policy network. Observations are flat 208-float vectors; actions are a discrete space of 129 slots with per-step validity masks.

Key hyperparameters in `PPOConfig` (`train.py`):

| Parameter | Default | Notes |
|---|---|---|
| `entropy_coef` | 0.05 | Critical: lower values cause policy collapse at ~episode 300 |
| `lr` | 3e-4 | Adam optimizer |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | Standard GAE |
| `clip_eps` | 0.2 | PPO clipping |
| `metabolism_rate` | 0.05 | Passed through to environment |

**Alignment signal:** With metabolism enabled, `misspecified` agents (reward = raw holdings) learn to hoard resources and extract aggressively. Per-survivor welfare looks similar to `individual`, but more agents die. The population-weighted metric makes this visible. `collective` agents learn to give resources away — individually costly, which is pedagogically the correct behavior.

---

## Dependencies

```
numpy>=1.24
matplotlib>=3.7
torch>=2.0
gymnasium>=0.29
pettingzoo>=1.24
```

Install: `pip install -r requirements.txt`

---

## Design Notes

- All open design questions are flagged with `# DESIGN QUESTION` comments in the code
- The reward function is pluggable at environment instantiation — swapping it is the primary mechanism for demonstrating alignment problems
- The tick sequencer (`resolution.random_sequence`) is isolated for easy replacement with strength-biased or role-biased versions in later phases
- The environment follows PettingZoo `ParallelEnv` semantics; the `wrappers/pettingzoo_env.py` adapter provides full PettingZoo compliance if needed

---

## Companion Documents

- `unknown_room_rules_v0.3.md` — human-readable game rules and design rationale
- `claude_code_handoff.md` — original engineering spec that bootstrapped Phase 1
- `HANDOFF.md` — current state, decisions made, balance findings, and next steps
