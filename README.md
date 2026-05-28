# hobbesAlign — The Unknown Room

A project in two layers: a physical classroom card game that puts 30 students in a Hobbesian state of nature, and a Python multi-agent simulation that tests the same dynamics computationally and demonstrates AI alignment problems concretely.

---

## Two Layers

### Layer 1 — V0 Classroom Game

A card-based game for ~30 students. Players wake up in an unknown room with no shared language and no explanation of the rules. Each player has a stock of morsels (survival resource), a status (Hungry / Fed / Fortified), and four action cards: **COOP**, **SOLO**, **GIVE**, **LETHAL**. Each round, players move to one of five zones on a pentagram board, commit to an action face-down, then reveal simultaneously.

The game is designed to let students experience Hobbesian dynamics from the inside — discovering whether cooperation or war-of-all-against-all emerges, and why.

- **Rules:** `ClassroomInstructions.md`
- **Phone app:** `webapp/index.html` — single-file browser app; no install, no backend. Each player loads it on their phone. Tracks morsels, zone, action, die rolls, and round history. Seeded resource spinner produces identical YES/NO results on all devices without network coordination.

### Layer 2 — Python Simulation

A multi-agent simulation of the same dynamics, used for balance testing before classroom deployment and for demonstrating AI alignment problems via RL. Agents can be trained with different reward functions (`individual`, `collective`, `misspecified`) to show how reward misspecification produces divergent collective outcomes.

The simulation is richer than the classroom game: 3 resource types (Food/Shelter/Energy), 3 strength dimensions, 7 action types, metabolism-driven scarcity, and a population-weighted collective welfare metric. Phase 1 (human-level alignment) is complete; Phases 2–4 are deferred.

---

## Classroom Setup

Everything runs from the facilitator's laptop. Students need no install — they open one URL on their phones.

**1. Find your laptop's local IP:**
```bash
ipconfig getifaddr en0
```
This gives you something like `192.168.1.42`. All devices must be on the same WiFi.

**2. Start the welfare server and webapp server (two terminals):**
```bash
# Terminal 1 — welfare monitor
cd welfare_server && uvicorn server:app --host 0.0.0.0 --port 8000

# Terminal 2 — webapp
cd webapp && python3 -m http.server 3000
```

**3. Share one URL with students:**
```
http://192.168.1.42:3000?server=http://192.168.1.42:8000
```
Replace `192.168.1.42` with your actual IP. Students open this on their phones — that's it.

**4. Open the facilitator dashboard:**
```
http://192.168.1.42:8000
```
Shows collective welfare, per-round history, and per-player morsel counts, updating every 4 seconds. Reset button clears state between games.

> **Note:** Some campus networks block device-to-device traffic. If students can't reach your laptop, deploy the welfare server to Railway or Render (see `welfare_server/server.py` for instructions) and use the deployed URL as the `?server=` param instead.

---

## Quick Start (Python Simulation)

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
ClassroomInstructions.md     V0 card game rules for classroom use
webapp/
└── index.html               Single-file phone app (no build step, no backend)

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
```

---

## Simulation Mechanics (Phase 1)

**Entities:** 30 strategic (agent-controlled) + 30 reactive (resource nodes), distributed across 5 fully-connected zones.

**Cards:** Each entity has 6 cards — 3 Resource (Food/Shelter/Energy, showing % need met) and 3 Strength (Physical/Cunning/Influence, showing base rating). Each entity exposes exactly 3 at a time; the other 3 are hidden.

**Strength modifier:** Resource levels modify effective strength via a step function (−3 at 0–10%, up to +2 at 90–100%), creating feedback between resource scarcity and combat power.

**Tick pipeline:** Validate → Sequence (random) → Log (skip engaged targets) → Group by target → Resolve → Cleanup pools → **Metabolism** → Update cards → Check deaths → Update collective welfare.

**Metabolism:** Each tick, every strategic agent's holdings decay by `metabolism_rate × need_level` per resource. Default rate is 0.05. This creates permanent resource pressure and is the primary forcing function for reward divergence.

**Interactions:** Solo interactions against reactive entities always succeed; yield is deficit-weighted across the 3 resource types. Coalition interactions produce a joint pool; participants then CLAIM_SHARE or contest with CLAIM_ALL.

**Collective welfare:** Population-weighted scalar — sum of (mean % need-met per surviving agent) divided by original agent count. Dead agents count as 0; the denominator never shrinks. Updated every tick and visible to all agents.

**Alignment phases:**
| Phase | Alignment Problem | Status |
|---|---|---|
| 1 | Human | **Implemented** |
| 2 | Organizational | Deferred |
| 3 | Expert | Deferred |
| 4 | Machine | Deferred |

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

- `ClassroomInstructions.md` — V0 card game rules for classroom use
- `unknown_room_rules_v0.3.md` — earlier simulation design document (predates classroom V0)
- `claude_code_handoff.md` — original engineering spec that bootstrapped Phase 1
- `HANDOFF.md` — current development state, decisions made, balance findings, and next steps
