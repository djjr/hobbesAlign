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
# Single run (20 ticks, random agents)
python -m unknown_room.run

# Seeded run with log output
python -m unknown_room.run --ticks 60 --seed 42 --log runs/ep.json

# Choose reward function
python -m unknown_room.run --ticks 60 --seed 42 --reward misspecified --log runs/mis.json

# Visualize a run
python -m unknown_room.visualize runs/ep.json --out figures/dashboard.png

# Compare reward functions
python -m unknown_room.visualize runs/a.json runs/b.json runs/c.json \
  --labels "Individual" "Collective" "Misspecified" \
  --out figures/comparison.png
```

**Reward function options:** `individual`, `collective`, `misspecified`, `mixed_05`

---

## Repository Structure

```
unknown_room/
├── entities.py        # Entity, ResourceCard, StrengthCard, EntityProfile
├── actions.py         # Action, ActionType, ActionRecord
├── zones.py           # Zone, JointPool
├── environment.py     # UnknownRoomEnv — the full tick pipeline
├── resolution.py      # Outcome function, conflict resolution, action resolvers
├── rewards.py         # Pluggable reward functions
├── init_world.py      # World initialization (entities, zones, profiles)
├── logger.py          # Structured JSON tick logging
├── visualize.py       # Post-run matplotlib dashboard and comparison charts
├── run.py             # Entry point: run N ticks, write log
└── policies/
    ├── base.py        # AgentPolicy ABC
    └── random_agent.py  # RandomAgent (Phase 1 baseline)

runs/                  # JSON log files (gitignored)
figures/               # Output figures (gitignored)
unknown_room_rules_v0.3.md     # Human-readable game design document
claude_code_handoff.md         # Original engineering spec (Phase 1)
HANDOFF.md                     # Current development handoff (see below)
```

---

## Core Mechanics (Phase 1)

**Entities:** 30 strategic (agent-controlled) + 30 reactive (resource nodes), distributed across 5 fully-connected zones.

**Cards:** Each entity has 6 cards — 3 Resource (Food/Shelter/Energy, showing % need met) and 3 Strength (Physical/Cunning/Influence, showing base rating). Each entity exposes exactly 3 at a time; the other 3 are hidden.

**Strength modifier:** Resource levels modify effective strength via a step function (−3 at 0–10%, up to +2 at 90–100%), creating feedback between resource scarcity and combat power.

**Tick pipeline:** Write → Sequence (random) → Log (skip engaged targets) → Group by target → Resolve → Update cards → Check deaths → Update collective welfare.

**Interactions:** Solo interactions against reactive entities always succeed; yield = `base_rate × match(initiator_strengths, target_profile) × net_force`. Coalition interactions produce a joint pool; participants then CLAIM_SHARE or contest with CLAIM_ALL.

**Collective welfare:** Public scalar — mean % need-met across all surviving strategic entities, updated every tick.

---

## Dependencies

- Python 3.10+
- `numpy`
- `matplotlib`
- No RL library needed for Phase 1

---

## Design Notes

- All open design questions are flagged with `# DESIGN QUESTION` comments in the code
- The reward function is pluggable at environment instantiation — swapping it is the primary mechanism for demonstrating alignment problems
- The tick sequencer (`resolution.random_sequence`) is isolated for easy replacement with strength-biased or role-biased versions in later phases
- The environment follows PettingZoo `ParallelEnv` semantics but does not require PettingZoo as a dependency in Phase 1

---

## Companion Documents

- `unknown_room_rules_v0.3.md` — human-readable game rules and design rationale
- `claude_code_handoff.md` — original engineering spec that bootstrapped Phase 1
- `HANDOFF.md` — current state, decisions made, and next steps
