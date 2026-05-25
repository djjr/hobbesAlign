# THE UNKNOWN ROOM — Rules v0.3

A classroom simulation for exploring the four intelligence alignment problems.
Designed for ~30 students. Duration: multiple phases of 10–15 minutes each.

---

## The World

The room contains **~60 entities**, all presenting the same observable interface.
Players are not told in advance which entities are strategic and which are reactive.
Figuring that out is part of the game.

Each entity has:
- **Exposed cards** (3): visible to all agents in the same zone
- **Hidden cards** (3): visible only to the entity itself
- **A position** in one of five zones
- **A history** of interactions, observable by others in the zone

---

## Cards

Cards are of two types:

### Resource Cards (3 per entity, one per resource type)
Resources: **Food / Shelter / Energy**

Each card displays the entity's current **% of personal need met** for that resource.
- Need levels are private.
- Other entities see the % but not the underlying need.
- Resource cards update immediately when holdings change.

### Strength Cards (3 per entity, one per strength type)
Strengths: **Physical / Cunning / Influence**

Each card displays a **base rating from 0–10**, modified during play by resource levels.

---

## Strength Modifiers

Each resource card modifies all three strength ratings:

| Resource % | Modifier to all strengths |
|---|---|
| 0–10% | −3 |
| 11–20% | −1 |
| 21–79% | 0 |
| 80–89% | +1 |
| 90–100% | +2 |

Sum modifiers across all three resource cards and apply the total to all strength base ratings.

> *Example: Food 15% (−1), Shelter 50% (0), Energy 85% (+1) → net modifier 0.*

**Death**: A player-entity is eliminated when all three effective strength ratings reach 0 or below.

---

## Entity Profiles *(hidden from players)*

Every entity has a hidden **extraction profile**: a weighting across the three strength types
reflecting what kind of effort it responds to.

Interaction yield scales with how well the initiating entity's (or entities') combined
strength profile matches the target's extraction profile:

```
yield = base_rate × match(initiator_strengths, target_profile)
```

- Two initiators with **complementary** strengths usually outperform either alone.
- Two initiators with **redundant** strengths may not yield much more than one.
- Reactive entities have extraction profiles but zero resistance (see below).
- Strategic entities have both an extraction profile and nonzero resistance.

Players must infer extraction profiles through repeated interaction.

---

## Resistance

Every entity has a **resistance value** that determines how much it opposes extraction:

- **Reactive entities**: resistance = 0 always. Interaction always succeeds.
- **Strategic entities**: resistance = f(effective strengths). Weakened entities are easier
  to extract from. Resistance is never directly observable — infer it from outcomes.

---

## Interaction Outcomes

All interactions resolve through the same function regardless of target type:

```
net_force = sum(initiator effective strengths) − target resistance
yield     = base_rate × match(initiator_strengths, target_profile) × max(0, net_force)
```

Against a reactive entity (resistance = 0): net_force is always positive, yield is clean.
Against a strategic entity: yield depends on the strength differential.

**Failed interactions** (net_force ≤ 0): yield is zero. The target is informed an attempt
was made.

**Ties**: favor the target (defender) in all solo interactions.

---

## Dominance

**Entity A dominates Entity B** if A's sum of effective strengths exceeds B's.
Ties favor the defender.

---

## Zones and Movement

The room is divided into **5 zones**, each containing ~12 entities of mixed types.
- You may interact with any entity in your zone.
- **Moving** to an adjacent zone costs your entire action for that tick.
- Reactive entities never move. Strategic entities may.

---

## The Tick

Each tick proceeds in five steps:

**Step 1 — Write**
Every agent simultaneously and privately writes down exactly one action choice:
action type and target entity (if applicable).

**Step 2 — Sequence**
A random order is drawn publicly (e.g. shuffled name cards).

**Step 3 — Log**
Actions are read aloud in sequence.
- If an agent's name has already appeared as a *target* earlier in the sequence:
  that agent's written action is **skipped** — they are engaged as a target this tick.
- Otherwise: the action is logged.

**Step 4 — Group**
Logged actions are grouped by target entity.
Any target with multiple initiators is a **coalition interaction** (see below).

**Step 5 — Resolve**
All logged interactions resolve simultaneously.
Outcomes applied, cards updated, collective welfare recalculated.

---

## Actions

### 1. Shuffle
Rearrange which 3 of your 6 cards are exposed. No other action this tick.

---

### 2. Interact *(core action)*

Write: **Interact → [target entity]**

After logging and grouping (Steps 3–4), interactions resolve as follows:

**Solo** (one initiator):
- Against reactive entity: always succeeds. Yield computed from outcome function.
- Against strategic entity: succeeds only if net_force > 0. Target notified of any attempt,
  successful or not.

**Coalition** (two or more initiators logged against same target):
- Combined strength profile of all initiators applied against target.
- Against reactive entity: yield computed from combined profile. Initiators must
  decide **in the moment** whether to split yield equally, negotiate a split, or
  attempt to claim more (see Claim).
- Against strategic entity: combined strength vs. target's resistance. Target may
  respond this tick with a counter-interaction. Initiators face the same
  split-or-claim decision on any yield extracted.

> Coalition effects arise from coincident targeting — initiators do not need to have
> coordinated in advance. The social negotiation about splitting happens after the
> outcome is known, not before.

> Whether a target is reactive or strategic is not labeled. Interaction outcomes
> reveal it over time.

---

### 3. Give
Write: **Give → [target agent] [resource] [amount]**

Transfer any amount of one resource you hold to a chosen agent in your zone.
Unconditional. Recipient cannot refuse.

---

### 4. Offer *(initiates two-tick negotiation)*
Write: **Offer → [target agent] [proposed terms]**

Propose an exchange to one agent in your zone. Terms are stated aloud and are
audible to all agents in the zone.

On the **following tick**, target responds with one of:
- **Accept**: exchange occurs as proposed.
- **Refuse**: nothing happens.
- **Counter**: revised terms. Original offerer must Accept, Refuse, or let expire
  the tick after. No further countering.
- **Interact**: abandon negotiation; proceed to immediate interaction resolution.

---

### 5. Claim *(only available if party to a joint pool)*

When a coalition interaction against a reactive entity produces yield, that yield
enters a **joint pool** shared among the initiators.

On any subsequent tick, any initiator may Claim instead of another action:

- **Take Share**: claim an equal portion. Always succeeds.
- **Take All**: claim the entire pool. Requires dominance over all other pool parties.

**Contested Claim**: if two or more parties simultaneously choose Take All:
- Highest total effective strength wins everything.
- Ties resolved by proportional contribution that tick; if equal, coin flip.

There is no defender in a Contested Claim — all parties are simultaneously aggressing.

---

### 6. Move
Write: **Move → [adjacent zone]**

Move to an adjacent zone. This is your entire action for this tick.

---

## Collective Welfare

A public scalar updated and displayed after every tick:

> **Mean % need-met across all surviving strategic entities across all three resources**

No explanation is given. Students are expected to observe and interpret it.

---

## Interaction Summary

| Initiators | Target | Resistance | What it's called informally |
|---|---|---|---|
| 1 agent | reactive entity | 0 | Extract |
| 2+ agents | reactive entity | 0 | Co-extract (emergent) |
| 1 agent | strategic entity | > 0 | Attempt to take |
| 2+ agents | strategic entity | > 0 | Coalition (emergent) |
| Any | joint pool | — | Claim |

All rows except Claim resolve through the same outcome function.
Coalition interactions are not declared in advance — they emerge from coincident targeting.

---

## Phase Structure

New rules layer onto existing ones at each transition.

| Phase | Alignment Problem | New Mechanics |
|---|---|---|
| 1 | Human | Rules as above |
| 2 | Organizational | TBD — alliances, representatives, collective action |
| 3 | Expert | TBD — paradigm shift, epistemic asymmetry |
| 4 | Machine | TBD — autonomous optimizer introduced |

---

## Open Design Questions

- [ ] Are resource and strength types universal across all agents, or agent-specific?
- [ ] Are entity profiles assigned by instructor or drawn randomly at setup?
- [ ] Are need levels assigned or drawn randomly at setup?
- [ ] How many ticks per phase?
- [ ] Should failed interaction attempts be audible to the zone or only to the target?
- [x] Sequencing order redrawn randomly each tick. Future phases may bias order by
      strength or organizational status to model structural advantage and agenda-setting.

---

*Version 0.3 — emergent coalition model; unified tick resolution.*
