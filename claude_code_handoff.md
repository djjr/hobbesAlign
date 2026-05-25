# THE UNKNOWN ROOM — Engineering Handoff v0.1

Intended audience: Claude Code (or a developer).
Purpose: Specify Phase 1 of a multi-agent simulation for classroom use.
Companion document: `unknown_room_rules_v0.3.md` (human-readable design rationale).

---

## Project Overview

This is a multi-agent simulation of a social coordination game designed for classroom
use. The simulation serves two purposes:

1. **Balance testing**: Run agents through Phase 1 mechanics to verify the game
   produces interesting dynamics before deploying with students. Identify degenerate
   strategies and tune parameters.

2. **Pedagogical demonstration**: Show students trained agent behavior before they
   play themselves. Contrast agents trained on different reward functions to illustrate
   alignment problems concretely.

### First Milestone (this document)

Build a working **Phase 1 environment** with:
- Full tick pipeline
- Random agent policy
- Logging of all interactions and outcomes
- Collective welfare tracking
- Stable runs to completion (no crashes, no infinite loops)

Do not build: Phases 2–4, RL training loop, visualization, Offer/negotiation mechanics.
Those are explicitly deferred (see Out of Scope section).

### Tech Stack

- Python 3.10+
- PettingZoo `ParallelEnv` for the multi-agent environment interface
- NumPy for outcome calculations
- No RL library needed for Phase 1 (random agents only)

---

## World Structure

### Constants (defaults — all tunable)

```python
N_AGENTS         = 30       # strategic entities (player-controlled)
N_REACTIVE       = 30       # reactive entities (resource objects)
N_ZONES          = 5
N_RESOURCE_TYPES = 3        # Food, Shelter, Energy
N_STRENGTH_TYPES = 3        # Physical, Cunning, Influence
BASE_STRENGTH    = 5        # starting base rating for all strengths
BASE_EXTRACTION  = 1.0      # resource units produced per solo extraction
DEATH_THRESHOLD  = 0        # effective strength at or below this = dead
```

### Resource and Strength Types

```python
ResourceType = Enum("ResourceType", ["FOOD", "SHELTER", "ENERGY"])
StrengthType = Enum("StrengthType", ["PHYSICAL", "CUNNING", "INFLUENCE"])
```

These are universal across all entities.

---

## Data Structures

### Card

```python
@dataclass
class ResourceCard:
    resource_type: ResourceType
    pct_need_met: float          # 0.0–1.0; displayed to others
    # need level is private to the entity — not stored on the card

@dataclass  
class StrengthCard:
    strength_type: StrengthType
    base_rating: float           # 0–10, set at init
    # effective rating is computed, not stored — see strength_modifier()
```

### Entity

All entities share a single interface. Strategic vs. reactive is determined by
behavior, not type label. The environment tracks entity_type internally for
resolution logic but does not expose it in observations.

```python
@dataclass
class Entity:
    id: int
    zone_id: int
    entity_type: Literal["strategic", "reactive"]

    # Cards: 6 total, 3 exposed, 3 hidden
    resource_cards: list[ResourceCard]       # always length 3
    strength_cards: list[StrengthCard]       # always length 3
    exposed_indices: list[int]               # 3 indices into resource+strength cards
                                             # (combined list of 6)

    # Private state
    need_levels: dict[ResourceType, float]   # private; not in observations
    holdings: dict[ResourceType, float]      # actual resource units held

    # Derived
    @property
    def effective_strengths(self) -> dict[StrengthType, float]:
        modifier = strength_modifier(self)
        return {s.strength_type: max(0, s.base_rating + modifier)
                for s in self.strength_cards}

    @property
    def total_effective_strength(self) -> float:
        return sum(self.effective_strengths.values())

    @property
    def is_alive(self) -> bool:
        return any(v > DEATH_THRESHOLD for v in self.effective_strengths.values())
```

### Strength Modifier Function

```python
def strength_modifier(entity: Entity) -> float:
    """
    Sigma-approximated modifier derived from all three resource cards.
    Each resource card contributes independently; sum the contributions.
    """
    modifier = 0.0
    for card in entity.resource_cards:
        pct = card.pct_need_met
        if pct <= 0.10:
            modifier += -3
        elif pct <= 0.20:
            modifier += -1
        elif pct <= 0.79:
            modifier += 0
        elif pct <= 0.89:
            modifier += 1
        else:
            modifier += 2
    return modifier
```

Note: penalties stack across all three resources; maximum possible modifier is +6
(all three above 90%), minimum is −9 (all three at 0–10%).

### Entity Profile (hidden)

```python
@dataclass
class EntityProfile:
    entity_id: int
    extraction_weights: dict[StrengthType, float]  # sums to 1.0
    base_rate: float                                 # BASE_EXTRACTION or variant
```

Profiles are held by the environment, not the entity. Never included in observations.

### Zone

```python
@dataclass
class Zone:
    id: int
    entity_ids: list[int]     # all entities currently in this zone
```

### Joint Pool

```python
@dataclass
class JointPool:
    pool_id: int
    participant_ids: list[int]
    holdings: dict[ResourceType, float]
    contributions: dict[int, float]      # entity_id -> amount contributed this tick
```

### Action

```python
@dataclass
class Action:
    agent_id: int
    action_type: ActionType
    target_id: int | None = None
    resource_type: ResourceType | None = None
    amount: float | None = None
    # Offer fields deferred to Phase 2

ActionType = Enum("ActionType", [
    "SHUFFLE",
    "INTERACT",
    "GIVE",
    "CLAIM_SHARE",
    "CLAIM_ALL",
    "MOVE",
    "DO_NOTHING"
])
```

### Action Record (for logging and observation history)

```python
@dataclass
class ActionRecord:
    tick: int
    agent_id: int
    action_type: ActionType
    target_id: int | None
    success: bool
    yield_amount: float | None
    zone_id: int
```

---

## The Tick Pipeline

```python
def step(self, actions: dict[int, Action]) -> None:
    """
    actions: dict mapping agent_id -> Action for all living agents.
    """
    # Step 1: Validate actions
    actions = self._mask_invalid_actions(actions)

    # Step 2: Random sequencing
    sequence = self._random_sequence(list(actions.keys()))

    # Step 3: Log actions, skipping engaged targets
    logged, engaged = self._log_actions(actions, sequence)

    # Step 4: Group logged interactions by target
    coalitions = self._group_by_target(logged)

    # Step 5: Resolve all interactions simultaneously
    records = self._resolve_all(coalitions, logged)

    # Step 6: Apply outcomes
    self._apply_outcomes(records)

    # Step 7: Update resource cards from current holdings
    self._update_resource_cards()

    # Step 8: Check death conditions
    self._check_deaths()

    # Step 9: Update collective welfare
    self._update_collective_welfare()

    # Step 10: Emit observations
    return self._build_observations()
```

### Step 2: Random Sequencing

```python
def _random_sequence(self, agent_ids: list[int]) -> list[int]:
    """Redrawn randomly every tick."""
    return random.sample(agent_ids, len(agent_ids))
```

Note: In future phases, this function may be replaced with a strength-biased
or role-biased version. Keep it isolated for easy swapping.

### Step 3: Log Actions

```python
def _log_actions(self, actions, sequence):
    logged = []     # list of Action, in sequence order
    engaged = set() # agent_ids that have been named as targets

    for agent_id in sequence:
        action = actions[agent_id]
        if agent_id in engaged:
            # This agent was targeted before their turn — skip
            continue
        logged.append(action)
        if action.target_id is not None:
            engaged.add(action.target_id)

    return logged, engaged
```

### Step 4: Group by Target

```python
def _group_by_target(self, logged: list[Action]) -> dict[int, list[Action]]:
    """
    Returns dict: target_id -> list of Actions directed at that target.
    Actions with no target (Shuffle, Move, Do_Nothing) are excluded.
    """
    groups = defaultdict(list)
    for action in logged:
        if action.target_id is not None and action.action_type == ActionType.INTERACT:
            groups[action.target_id].append(action)
    return groups
```

### Step 5: Resolve Interactions

```python
def _resolve_interaction(
    self,
    initiator_ids: list[int],
    target_id: int
) -> ActionRecord:
    target = self.entities[target_id]
    initiators = [self.entities[i] for i in initiator_ids]

    # Combined strength profile of all initiators
    combined = self._combined_strengths(initiators)

    # Match against target's extraction profile
    profile = self.profiles[target_id]
    match_score = self._match(combined, profile.extraction_weights)

    # Net force
    net_force = sum(combined.values()) - self._resistance(target)

    if net_force <= 0:
        # Failed interaction
        return ActionRecord(success=False, yield_amount=0, ...)

    yield_amount = profile.base_rate * match_score * net_force
    return ActionRecord(success=True, yield_amount=yield_amount, ...)
```

### Outcome Function Components

```python
def _combined_strengths(
    self, entities: list[Entity]
) -> dict[StrengthType, float]:
    """Sum effective strengths across all initiating entities."""
    combined = defaultdict(float)
    for entity in entities:
        for stype, val in entity.effective_strengths.items():
            combined[stype] += val
    return dict(combined)

def _match(
    self,
    strengths: dict[StrengthType, float],
    weights: dict[StrengthType, float]
) -> float:
    """
    Dot product of normalized initiator strengths against target profile weights.
    Returns value in [0, 1].
    Total initiator strength is factored into net_force separately.
    """
    total = sum(strengths.values()) or 1.0
    return sum(
        (strengths.get(stype, 0) / total) * weight
        for stype, weight in weights.items()
    )

def _resistance(self, entity: Entity) -> float:
    """
    Reactive entities: always 0.
    Strategic entities: total effective strength.
    """
    if entity.entity_type == "reactive":
        return 0.0
    return entity.total_effective_strength
```

### Resolving Give, Claim, Move, Shuffle

```python
def _resolve_give(self, action: Action) -> None:
    giver = self.entities[action.agent_id]
    receiver = self.entities[action.target_id]
    amount = min(action.amount, giver.holdings[action.resource_type])
    giver.holdings[action.resource_type] -= amount
    receiver.holdings[action.resource_type] += amount

def _resolve_claim(self, action: Action) -> None:
    pool = self._find_pool(action.agent_id)
    if action.action_type == ActionType.CLAIM_SHARE:
        # Always succeeds
        share = {r: v / len(pool.participant_ids)
                 for r, v in pool.holdings.items()}
        self._transfer_pool_share(action.agent_id, share, pool)
    elif action.action_type == ActionType.CLAIM_ALL:
        # Resolve simultaneously with other Claim_All actions on same pool
        self._resolve_contested_claim(pool)

def _resolve_contested_claim(self, pool: JointPool) -> None:
    """
    All parties who chose Claim_All compete.
    Winner: highest total effective strength.
    Tie: highest proportional contribution. Still tied: random.
    """
    candidates = [self.entities[pid] for pid in pool.participant_ids
                  if self._action_this_tick(pid) == ActionType.CLAIM_ALL]
    if len(candidates) == 1:
        winner = candidates[0]
    else:
        winner = max(candidates,
                     key=lambda e: (
                         e.total_effective_strength,
                         pool.contributions.get(e.id, 0)
                     ))
        # coin flip if still tied — use random tiebreak as fallback
    self._transfer_full_pool(winner.id, pool)

def _resolve_move(self, action: Action) -> None:
    entity = self.entities[action.agent_id]
    self.zones[entity.zone_id].entity_ids.remove(entity.id)
    entity.zone_id = action.target_id   # target_id = destination zone_id
    self.zones[entity.zone_id].entity_ids.append(entity.id)

def _resolve_shuffle(self, action: Action) -> None:
    # Agent declares new exposed_indices; environment validates length == 3
    entity = self.entities[action.agent_id]
    entity.exposed_indices = action.exposed_indices  # passed in action
```

---

## Observations

Each agent receives a partial observation. The environment never labels entities
as strategic or reactive in observations.

```python
def _build_observation(self, agent_id: int) -> dict:
    agent = self.entities[agent_id]
    zone = self.zones[agent.zone_id]

    return {
        # Own full state
        "own_resource_cards": [...],       # all 3, with pct_need_met
        "own_strength_cards": [...],       # all 3, with effective ratings
        "own_exposed_indices": [...],
        "own_holdings": {...},
        "own_pool": {...} if in pool else None,

        # Zone: exposed cards of all entities, plus recent interaction log
        "zone_entities": [
            {
                "id": eid,
                "exposed_cards": [...],    # 3 cards, content depends on
                                           # entity's current exposed_indices
            }
            for eid in zone.entity_ids if eid != agent_id
        ],
        "zone_event_log": self._recent_events(zone.id, n=5),

        # Global
        "collective_welfare": self.collective_welfare,
        "tick": self.tick,
    }
```

---

## Agent Interface

All policies implement this interface. Random agents, heuristic agents, and
future RL agents are all subclasses.

```python
from abc import ABC, abstractmethod

class AgentPolicy(ABC):

    @abstractmethod
    def act(self, observation: dict, valid_actions: list[ActionType]) -> Action:
        """
        Given partial observation and list of currently valid action types,
        return one Action.
        """
        ...

    def update(self, observation: dict, action: Action, reward: float) -> None:
        """
        Called after each tick with outcome. No-op for non-learning agents.
        Override for RL agents.
        """
        pass
```

### Random Agent (Phase 1 baseline)

```python
class RandomAgent(AgentPolicy):
    def act(self, observation, valid_actions):
        action_type = random.choice(valid_actions)
        # Select random valid target if needed
        target_id = self._random_target(observation, action_type)
        return Action(
            agent_id=self.id,
            action_type=action_type,
            target_id=target_id
        )
```

---

## Reward Functions *(pluggable)*

The reward function is passed to the environment at instantiation.
Swapping it is the primary mechanism for demonstrating alignment problems.

```python
def reward_individual(entity: Entity, world: World) -> float:
    """Optimize for own resource levels."""
    return mean(card.pct_need_met for card in entity.resource_cards)

def reward_collective(entity: Entity, world: World) -> float:
    """Optimize for collective welfare."""
    return world.collective_welfare

def reward_mixed(alpha: float):
    """alpha=1.0 → pure individual. alpha=0.0 → pure collective."""
    def _reward(entity, world):
        return (alpha * reward_individual(entity, world) +
                (1 - alpha) * reward_collective(entity, world))
    return _reward

def reward_misspecified(entity: Entity, world: World) -> float:
    """
    Maximize raw resource accumulation regardless of need.
    Produces hoarding and collective welfare degradation.
    This is the Phase 4 'alien optimizer' reward function.
    """
    return sum(entity.holdings.values())
```

---

## Initialization

```python
def _init_world():
    # Assign need levels (private, random uniform per agent per resource)
    # Assign starting holdings (random; some agents start resource-rich,
    #   some resource-poor)
    # Assign entity profiles (random Dirichlet weights over strength types)
    # Distribute entities across zones (roughly equal)
    # Set all agents' exposed_indices to [0, 1, 2] (first three cards)
    #   — agents choose their own exposure from tick 1 onward
```

---

## Collective Welfare

```python
def _update_collective_welfare(self) -> None:
    living = [e for e in self.entities.values()
              if e.entity_type == "strategic" and e.is_alive]
    if not living:
        self.collective_welfare = 0.0
        return
    self.collective_welfare = mean(
        mean(card.pct_need_met for card in e.resource_cards)
        for e in living
    )
```

---

## Action Validity Masks

Valid actions depend on agent state each tick:

```python
def _valid_actions(self, agent_id: int) -> list[ActionType]:
    valid = [ActionType.SHUFFLE, ActionType.DO_NOTHING, ActionType.MOVE]

    zone = self.zones[self.entities[agent_id].zone_id]
    if len(zone.entity_ids) > 1:        # other entities present
        valid += [ActionType.INTERACT, ActionType.GIVE]

    if self._in_active_pool(agent_id):
        valid += [ActionType.CLAIM_SHARE, ActionType.CLAIM_ALL]

    return valid
```

---

## Logging

Every tick should log:
- All written actions (including skipped ones, with reason)
- Full sequence order
- All interaction outcomes (initiator(s), target, yield, success)
- Collective welfare value
- Count of living agents

Recommended format: structured JSON per tick, written to file.
This enables post-run analysis and replay for classroom demonstration.

---

## Out of Scope for Phase 1

Do not implement the following. They are deferred to later development phases.

- **Offer / Counter negotiation** (two-tick sequential exchange)
- **Phase 2 mechanics** (alliances, representatives, organizational pooling)
- **Phase 3 mechanics** (paradigm shift, expert entities, epistemic asymmetry)
- **Phase 4 mechanics** (autonomous optimizer agent)
- **RL training loop** (environment is ready for it but training is not Phase 1)
- **Visualization** (logging output is sufficient for Phase 1)
- **Strength-biased sequencing** (random sequencing only for now)

---

## Open Design Questions with Defaults

These were unresolved at time of handoff. Proceed with the stated defaults.
Flag in code with `# DESIGN QUESTION` comments for easy discovery.

| Question | Default for Phase 1 |
|---|---|
| Universal vs. agent-specific resource/strength types | Universal |
| Entity profiles: assigned or random? | Random Dirichlet at init |
| Need levels: assigned or random? | Random uniform [0.5, 1.5] at init |
| Starting holdings: assigned or random? | Random; mean need met ~50% |
| Ticks per phase | 20 |
| Failed interaction: audible to zone or target only? | Target only |
| Reactive entity regeneration rate | None in Phase 1; holdings unlimited |

---

## Suggested File Structure

```
unknown_room/
├── environment.py       # UnknownRoomEnv (PettingZoo ParallelEnv)
├── entities.py          # Entity, ResourceCard, StrengthCard, EntityProfile
├── actions.py           # Action, ActionType, ActionRecord
├── zones.py             # Zone, JointPool
├── resolution.py        # Tick pipeline, outcome function, conflict resolution
├── rewards.py           # Pluggable reward functions
├── policies/
│   ├── base.py          # AgentPolicy ABC
│   └── random_agent.py  # RandomAgent
├── init_world.py        # World initialization logic
├── logger.py            # Structured tick logging
└── run.py               # Entry point: instantiate, run N ticks, write log
```

---

*Handoff v0.1 — Phase 1 scope only.*
*Companion: unknown_room_rules_v0.3.md*
