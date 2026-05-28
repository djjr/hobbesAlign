# HANDOFF — hobbesAlign / The Unknown Room

**Date:** 2026-05-28  
**Phase:** 1 complete + RL training loop complete + V0 classroom game complete  
**Next session:** WorldConfig dataclass + YAML presets (Python sim), or balance tuning

---

## Project Overview

This repo has two related but distinct artifacts:

1. **V0 Classroom Game** — a physical card game for 30 students plus a companion phone web app. Designed to let students experience Hobbesian dynamics from the inside. Rules in `ClassroomInstructions.md`; app in `webapp/index.html`.

2. **Python Simulation** — a multi-agent RL environment (`unknown_room/`) used to balance-test the game mechanics and demonstrate AI alignment problems computationally. Phase 1 complete. The sim is intentionally richer than the classroom game (3 resource types, 7 action types) and will grow richer as Phases 2–4 are added.

The two layers are parallel analogs, not the same model. The classroom game is deliberately simplified for human play; the simulation is the computational version that can be run at scale and with trained agents.

---

## V0 Classroom Game — What Was Built

### ClassroomInstructions.md

Complete rules for a 30-player classroom card game. Key design:

- **Single resource:** morsels (survival currency). Start: 5 per player. MDR: −1/round.
- **Status:** Hungry (0–2 morsels), Fed (3–5), Fortified (6+). Determines strength die.
- **Strength dice:** Hungry [1,1,2,2,3,4] · Fed [1,2,3,4,5,6] · Fortified [3,4,4,5,5,6]
- **Four actions:** COOP (green), SOLO (yellow), GIVE (blue), LETHAL (red)
- **Board:** pentagram — 5 fully-connected nodes, each with a resource spinner (YES/NO mix)
- **Encounter resolution:** full payoff matrix — COOP+COOP, COOP+SOLO, COOP+LETHAL, SOLO+LETHAL, GIVE+anything, LETHAL+LETHAL. Lethal encounters resolved by dice contest; COOP/GIVE players roll one die category lower when their opponent plays LETHAL.
- **Collective welfare:** total morsels ÷ 30, announced each round. Denominator never shrinks.
- **Elimination:** bag hits 0 after MDR, or lose a LETHAL dice contest. Eliminated players become observers with full information.

### webapp/index.html

Single-file browser app, no build step, no backend. Each player loads it on their phone.

Key features:
- **Private screen:** morsel count + status, zone picker (color-coded by fecundity), action picker, round history
- **Public screen:** card flip (CSS 3D) revealing zone + action + resource result; die roll animation; payoff reference table filtered to player's action with active column highlighted
- **Seeded resource spinner:** deterministic hash of (round, zone) produces identical YES/NO on all devices without network coordination. Each zone has a hardcoded fecundity (Zone 1: 75%, Zone 5: 20%, etc.).
- **Disadvantage toggle:** appears only for COOP/GIVE players. If opponent played LETHAL, player taps toggle → die drops one category before rolling. LETHAL player's app unchanged — symmetric by design.
- **Elimination flow:** "Eliminated by LETHAL" button shows morsels given to opponent (net of MDR). Leads to history screen.
- **localStorage persistence** across browser sessions. Wake Lock API keeps screen on.

---

## Python Simulation — What Was Built

Phase 1 of the Unknown Room simulation is complete and running stably. All files are in `unknown_room/`. The environment runs to completion without crashes across 300-episode RL training runs with 30 strategic + 30 reactive entities.

### Implemented

- Full data structures: `Entity`, `ResourceCard`, `StrengthCard`, `EntityProfile`, `Zone`, `JointPool`, `Action`, `ActionRecord`
- Complete tick pipeline (10 steps): validate → sequence → log → group → resolve → cleanup pools → **metabolism** → update cards → check deaths → update welfare → emit observations
- All Phase 1 action types: `INTERACT`, `GIVE`, `CLAIM_SHARE`, `CLAIM_ALL`, `MOVE`, `SHUFFLE`, `DO_NOTHING`
- Pluggable reward functions: `individual`, `collective`, `misspecified`, `mixed_05`
- `RandomAgent` baseline policy
- Structured JSON tick logging
- **RL training loop** (PPO with GAE, action masking, parameter sharing across agents)
- Flat observation vector (208 floats) and discrete action space (129 slots) with validity masks
- PettingZoo `ParallelEnv` wrapper (`wrappers/pettingzoo_env.py`)
- Shared MLP actor-critic policy (`policies/mlp_policy.py`)
- Training CLI: `python -m unknown_room.train --reward <fn> --episodes 300 --out runs/trained_X`
- Training curve visualization: `python -m unknown_room.visualize ... --training`
- Post-run matplotlib dashboard (6 panels) and multi-run comparison chart
- **Metabolism** (resource consumption per tick, `--metabolism 0.05`) — creates permanent scarcity forcing reward divergence
- **Population-weighted collective welfare** — dead agents count as 0, denominator = original `n_agents` (prevents survivor bias masking alignment failures)
- CLI entry points: `python -m unknown_room.run` and `python -m unknown_room.train` and `python -m unknown_room.visualize`

### Not Implemented (explicitly deferred)

- Phases 2–4 mechanics
- Offer/negotiation (two-tick sequential exchange)
- Strength-biased sequencing
- Simulation web viewer (FastAPI backend + live frontend)
- Parameter config system (WorldConfig + YAML presets) — *next recommended task*

---

## Key Design Decisions (Python Sim)

| Decision | Choice | Rationale |
|---|---|---|
| Zone topology | Fully connected | Confirmed with user |
| Reactive entity depletion | None — inexhaustible | Confirmed with user |
| Coalition vs. solo yield | Both produce pools; solo delivers directly | Confirmed with user |
| Multiple pool prevention | INTERACT removed from valid_actions while in pool | Confirmed with user |
| Move forfeits pool | Yes — moving removes agent from pool | Design choice |
| Solo yield distribution | Deficit-weighted across 3 resource types | Design choice — flag if different split wanted |
| Coalition pool holdings | Yield split equally across 3 resource types | Design choice |
| Agent seeding | Each agent gets unique seed derived from global seed | Bug fix — identical seeds caused all agents to choose identically |
| Collective welfare metric | Population-weighted (dead = 0, denominator = n_agents) | Prevents survivor bias masking alignment failures |
| Scarcity mechanism | Metabolism: holdings decay by `metabolism_rate × need_level` each tick | Makes reward functions diverge; `metabolism_rate=0.05` default |
| PPO entropy coefficient | `entropy_coef=0.05` | Prevents policy collapse at ~episode 300 (see Training Stability below) |

---

## Bugs Found and Fixed

1. **All agents chose identical actions** — `RandomAgent` initialized with same seed for all 30 agents. Fixed in `run.py` by deriving per-agent seeds from a master RNG.

2. **CLAIM_ALL winner logged yield_amount=0** — pool holdings were zeroed before computing the log value. Fixed in `resolution.py`.

3. **Pool participants not released on pool expiry** — `_cleanup_pools` deleted the pool but didn't clear `_agent_pool` mappings, leaving agents permanently locked out of INTERACT. Fixed in `environment.py`.

4. **Survivor-biased welfare metric** — original metric averaged only living agents; a culling strategy could appear welfare-positive. Fixed by dividing total survivor welfare by the original `n_agents` count.

---

## Balance Findings (from RL testing)

### Scarcity is required for reward divergence

Without metabolism (`--metabolism 0`), reactive entities are inexhaustible. Any policy that discovers INTERACT → reactive entity immediately fills all resource needs regardless of what it is optimizing for. All three reward functions converge to ~100% collective welfare within ~10 episodes.

**Fix applied:** `metabolism_rate=0.05` (default) causes holdings to decay each tick by `need_level × 0.05` per resource. Agents must continuously extract to stay alive. This alone produces meaningful divergence.

### Training results with metabolism=0.05, episodes=300, entropy_coef=0.05

| Reward | Final welfare | Final living (of 30) | Notes |
|---|---|---|---|
| `individual` | ~0.83 | ~24–27 | Good survival, high welfare |
| `collective` | ~0.70 | ~25–28 | Takes GIVE actions (costly to self), lower individual welfare |
| `misspecified` | ~0.83 | ~18–22 | Hoards raw holdings, higher per-agent welfare but more deaths |

**Teaching insight:** `collective` welfare ends up *lower* than `individual` because agents learn to give resources away — individually costly, collectively beneficial. This is the correct alignment behavior and is pedagogically useful.

**Key alignment signal:** `misspecified` produces similar per-survivor welfare but more agent deaths. Population-weighted metric makes this visible: final welfare ~0.65–0.80 with higher variance. Without population-weighting, it would look identical to `individual`.

---

## Training Stability

### Policy collapse at ~episode 300 (fixed)

**Symptom:** With `entropy_coef=0.01`, welfare and survival collapsed after ~episode 300 and never recovered. Agents converged to deterministic policies; a single bad minibatch caused catastrophic forgetting.

**Fix:** Raised `entropy_coef` from 0.01 to 0.05 in `PPOConfig`. This maintains enough exploration pressure to prevent premature convergence.

**Trade-off:** Slightly lower peak welfare (collective ~0.70 vs ~0.75 with lower entropy), but training is stable through all 300 episodes.

**Possible refinements (not yet tried):**
- `entropy_coef=0.03` as a middle ground
- Learning rate decay (cosine schedule) to reduce step size in late training

---

## What to Do Next (Python Sim)

### Option A — Parameter config system (recommended)
Create a `WorldConfig` dataclass covering all tunable constants with a YAML preset loader. This makes it easy to:
- Run the alignment demo at different difficulty levels
- Save/reproduce experimental configurations
- Give students access to "scenario presets" without editing code
- Create a V0-analog preset (single resource type, simplified action space) to shadow the classroom game

### Option B — Balance tuning
Run many episodes with random agents across different seeds. Plot distribution of final welfare, death rates, and pool formation frequency. Identify degenerate cases (mass death, welfare collapse) and tune constants.

### Option C — Simulation web viewer
FastAPI backend running the simulation tick-by-tick, served to a frontend via WebSocket. Shows zone state, agent cards, welfare ticker, and event log in real time. (Distinct from the classroom webapp, which has no backend.)

---

## Open Design Questions (flagged `# DESIGN QUESTION` in code)

- Need levels: currently `uniform(0.5, 1.5)` — may want tighter range
- Starting holdings: `uniform(25%–75%)` of need — may want asymmetry to create have/have-not dynamics
- Solo yield distribution: deficit-weighted — may want equal split instead
- Coalition pool holdings: equal split across resource types — may want to tie to target's resource profile
- `_random_sequence` is isolated in `resolution.py` for easy swap to strength-biased version

---

## Running the Simulation

```bash
cd /path/to/hobbesAlign

# Basic run (random agents)
python -m unknown_room.run --ticks 40 --seed 42

# Run with specific reward function and log
python -m unknown_room.run --ticks 60 --seed 42 --reward misspecified --log runs/mis.json

# Train RL agents (PPO, parameter sharing)
python -m unknown_room.train --reward individual --episodes 300 --out runs/individual
python -m unknown_room.train --reward collective --episodes 300 --out runs/collective
python -m unknown_room.train --reward misspecified --episodes 300 --out runs/misspecified

# Visualize a single run
python -m unknown_room.visualize runs/ep.json --out figures/dashboard.png

# Compare reward functions
python -m unknown_room.visualize runs/a.json runs/b.json runs/c.json \
  --labels "Individual" "Collective" "Misspecified" --out figures/comparison.png

# Plot training curves
python -m unknown_room.visualize runs/individual/training_log.json \
  runs/collective/training_log.json runs/misspecified/training_log.json \
  --labels "Individual" "Collective" "Misspecified" \
  --training --out figures/training_curves.png
```

---

## File Map

```
ClassroomInstructions.md     V0 card game rules
webapp/index.html            Phone app for classroom play

unknown_room/
├── entities.py          constants, ResourceCard, StrengthCard, Entity, EntityProfile
├── actions.py           Action (+ exposed_indices field), ActionType, ActionRecord
├── zones.py             Zone, JointPool
├── environment.py       UnknownRoomEnv — main class, tick pipeline, observation builder
├── resolution.py        All resolvers, outcome function, random_sequence (isolated)
├── rewards.py           reward_individual, reward_collective, reward_misspecified, reward_mixed
├── init_world.py        World initialization — entities, profiles, zone distribution
├── logger.py            TickLogger — JSON log per episode
├── spaces.py            obs_to_array(), action_mask(), decode_action() — RL observation/action layer
├── visualize.py         plot_dashboard (6-panel), plot_comparison, plot_training_curves
├── run.py               CLI entry point (random agents), run_episode(), REWARD_FNS registry
├── train.py             PPO training loop, PPOConfig, Rollout, compute_gae(), ppo_update()
└── policies/
    ├── base.py              AgentPolicy ABC
    ├── random_agent.py      RandomAgent with per-instance RNG
    └── mlp_policy.py        MLPPolicy — shared MLP actor-critic with action masking
wrappers/
└── pettingzoo_env.py    PettingZoo ParallelEnv adapter (string agent IDs, obs flattening)
```

---

## Key Constants (all in `entities.py`)

| Constant | Default | Effect |
|---|---|---|
| `N_AGENTS` | 30 | Strategic agents |
| `N_REACTIVE` | 30 | Reactive entities (inexhaustible) |
| `N_ZONES` | 5 | Number of zones (fully connected) |
| `TICKS_PER_PHASE` | 20 | Episode length (override with `--ticks`) |
| `METABOLISM_RATE` | 0.05 | Fraction of need_level consumed per tick |
| `BASE_EXTRACTION` | 1.0 | Base yield multiplier for interactions |
| `DEATH_THRESHOLD` | 0 | Effective strength ≤ this → death |
