# HANDOFF — hobbesAlign / The Unknown Room

**Date:** 2026-05-25  
**Phase:** 1 complete + RL training loop complete  
**Next session:** Parameter config system, or web app scaffolding

---

## What Was Built

Phase 1 of the Unknown Room simulation is complete and running stably. All files are in `unknown_room/`. The environment runs to completion without crashes across 100+ tick episodes with 30 strategic + 30 reactive entities.

### Implemented

- Full data structures: `Entity`, `ResourceCard`, `StrengthCard`, `EntityProfile`, `Zone`, `JointPool`, `Action`, `ActionRecord`
- Complete tick pipeline (10 steps): validate → sequence → log → group → resolve → cleanup pools → update cards → check deaths → update welfare → emit observations
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
- CLI entry points: `python -m unknown_room.run` and `python -m unknown_room.visualize`

### Not Implemented (explicitly deferred)

- RL training loop
- PettingZoo wrapper (environment follows the interface but doesn't depend on the library)
- Phases 2–4 mechanics
- Offer/negotiation (two-tick sequential exchange)
- Strength-biased sequencing
- Web application

---

## Key Design Decisions Made

These were either unspecified or ambiguous in the original handoff and were resolved:

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

---

## Bugs Found and Fixed

1. **All agents chose identical actions** — `RandomAgent` was initialized with the same seed for all 30 agents. Fixed in `run.py` by deriving per-agent seeds from a master RNG.

2. **CLAIM_ALL winner logged yield_amount=0** — pool holdings were zeroed before computing the log value. Fixed in `resolution.py`.

3. **Pool participants not released on pool expiry** — `_cleanup_pools` deleted the pool but didn't clear `_agent_pool` mappings, leaving agents permanently locked out of INTERACT. Fixed in `environment.py`.

---

## Balance Findings (from RL testing)

**Finding: Phase 1 has no meaningful scarcity, so all reward functions converge.**

Running PPO under `individual`, `collective`, and `misspecified` reward functions all produce ~100% collective welfare within ~10 episodes. The hoard policy (`misspecified`) shows occasional late-episode welfare collapse as it learns to extract from other strategic entities, but no sustained divergence.

**Root cause:** Reactive entities are inexhaustible. Any policy that discovers INTERACT → reactive entity immediately fills all its resource needs, regardless of what it is optimizing for. Accumulating raw holdings (misspecified) incidentally satisfies pct_need_met, so the alignment problem is invisible in Phase 1.

**What will make reward functions diverge:**

| Mechanism | Effort | Notes |
|---|---|---|
| **Need consumption per tick** | Low — one line in `_update_resource_cards` | Holdings decay by `need_level × consumption_rate` each tick; agents must continuously extract. This alone should produce divergence. |
| **Reactive entity depletion** | Low — track reactive holdings; don't credit if depleted | Creates genuine scarcity and competition between agents |
| **Longer episodes** | Zero — just increase `--ticks` | May expose late-game hoarding effects |

**Decision:** Noted as design finding. Implement need consumption or depletion when ready to demonstrate alignment divergence. Both are single-parameter additions flagged `# DESIGN QUESTION` style.

---

## What to Do Next

### Option A — RL training loop (recommended)
The environment is ready. Next steps:
1. Wrap `UnknownRoomEnv` with PettingZoo's `ParallelEnv` (thin adapter, ~50 lines)
2. Flatten observations into numpy arrays (required by most RL libraries)
3. Define observation and action spaces formally
4. Wire up a training loop using RLlib or PettingZoo + stable-baselines3
5. Train agents under each reward function and regenerate the comparison chart — this is the pedagogical payoff

### Option B — Web application
The web app should drive off the simulation engine. Suggested stack: FastAPI backend running the simulation tick-by-tick, served to a React/Svelte frontend via WebSocket. The frontend shows zone state, agent cards, welfare ticker, and event log in real time. Students would control agents via the web UI instead of code.

### Option C — Balance tuning
Run many episodes with random agents across different seeds. Plot distribution of final welfare, death rates, and pool formation frequency. Identify degenerate cases (mass death, welfare collapse) and tune constants.

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

# Basic run
python -m unknown_room.run --ticks 40 --seed 42

# Run with specific reward function and log
python -m unknown_room.run --ticks 60 --seed 42 --reward misspecified --log runs/mis.json

# Visualize
python -m unknown_room.visualize runs/ep.json --out figures/dashboard.png

# Compare reward functions
python -m unknown_room.visualize runs/a.json runs/b.json runs/c.json \
  --labels "Individual" "Collective" "Misspecified" --out figures/comparison.png
```

**Note:** The comparison chart currently shows identical lines because random agents don't use reward signals. It will diverge once trained RL agents are introduced.

---

## File Map

```
unknown_room/
├── entities.py      constants, ResourceCard, StrengthCard, Entity, EntityProfile
├── actions.py       Action (+ exposed_indices field added vs. spec), ActionType, ActionRecord
├── zones.py         Zone, JointPool
├── environment.py   UnknownRoomEnv — main class, tick pipeline, observation builder
├── resolution.py    All resolvers, outcome function, random_sequence (isolated)
├── rewards.py       reward_individual, reward_collective, reward_misspecified, reward_mixed
├── init_world.py    World initialization — entities, profiles, zone distribution
├── logger.py        TickLogger — JSON log per episode
├── visualize.py     plot_dashboard (6-panel), plot_comparison (welfare overlay)
├── run.py           CLI entry point, run_episode(), REWARD_FNS registry
└── policies/
    ├── base.py          AgentPolicy ABC
    └── random_agent.py  RandomAgent
```
