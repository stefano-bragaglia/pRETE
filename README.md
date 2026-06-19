# pRETE (Python RETE)

A pure-Python implementation of the Rete algorithm for production rule systems.

![pRETE logo](images/pRETE-logo-small.png)

## Background

Implements the algorithm from:
- Forgy, C. L. (1982). Rete: A fast algorithm for the many pattern/many object pattern match problem. *Artificial Intelligence*, 19(1), 17–37.
- Doorenbos, R. B. (1995). *Production system techniques for large rule bases* (CMU-CS-95-113). Carnegie Mellon University.

Working memory elements are represented as `(id, attribute, value)` triples per Doorenbos §2.1.

## Install

```bash
pip install -e .[dev]
```

## Usage

### Core concepts

| Term | What it is |
|------|-----------|
| **WME** | A `(id, attribute, value)` triple — one fact in working memory |
| **Condition** | A triple of field tests; fields may be constants, `?variables`, or `WILDCARD` |
| **Production** | A rule: a list of `Condition`s (LHS) plus a Python callable (RHS) |
| **ReteNetwork** | The compiled network; call `add_wme` / `remove_wme` / `add_production` |
| **InferenceEngine** | Wraps `ReteNetwork` with a select-and-fire loop (`run()`) |

Variable names start with `?`.  A variable binds to a field value on its first
match and must equal that value in every subsequent condition that reuses it.
`WILDCARD` matches any value without binding.

---

### Quick start — basic fact matching

```python
from rete import WME, Condition, Production, ReteNetwork

net = ReteNetwork()

net.add_production(Production(
    lhs=[
        Condition("?x", "on", "?y"),      # x is on y
        Condition("?y", "on", "table"),   # y is on the table
    ],
    rhs=lambda token: print(
        f"{token.wmes[0].id} is on {token.wmes[0].value}, which is on the table"
    ),
))

net.add_wme(WME("A", "on", "B"))
net.add_wme(WME("B", "on", "table"))
net.add_wme(WME("C", "on", "table"))

# One match: A→B→table  (C→table has nothing on top of it)
for inst in net.conflict_set:
    inst.production.rhs(inst.token)
# A is on B, which is on the table
```

---

### Negated conditions

Pass `negated=True` to exclude facts that satisfy the condition.  The negated
condition does not contribute a WME to the token.

```python
from rete import WME, Condition, Production, ReteNetwork

net = ReteNetwork()

net.add_production(Production(
    lhs=[
        Condition("?x", "on",    "table"),
        Condition("?x", "color", "blue", negated=True),  # NOT blue
    ],
    rhs=lambda token: print(f"{token.wmes[0].id} is on the table and not blue"),
))

net.add_wme(WME("A", "on",    "table"))
net.add_wme(WME("A", "color", "red"))    # not blue → fires
net.add_wme(WME("B", "on",    "table"))
net.add_wme(WME("B", "color", "blue"))   # blue     → suppressed
# A is on the table and not blue
```

---

### Negated conjunctive conditions (NCC)

`NccGroup` wraps multiple conditions that must **not** jointly match.

```python
from rete import WME, Condition, NccGroup, Production, ReteNetwork

net = ReteNetwork()

net.add_production(Production(
    lhs=[
        Condition("?x", "type", "order"),
        NccGroup(conditions=(              # fires only when BOTH conditions below are absent
            Condition("?x", "status",   "paid"),
            Condition("?x", "shipped",  "yes"),
        )),
    ],
    rhs=lambda token: print(f"Order {token.wmes[0].id} is neither paid nor shipped"),
))

net.add_wme(WME("order-1", "type",    "order"))
net.add_wme(WME("order-2", "type",    "order"))
net.add_wme(WME("order-2", "status",  "paid"))
net.add_wme(WME("order-2", "shipped", "yes"))
# Only order-1 fires (order-2 has both a paid and a shipped fact)
```

---

### Retraction

Removing a WME automatically retracts every match that depended on it.

```python
w = WME("A", "on", "table")
net.add_wme(w)
# ... rule fires ...
net.remove_wme(w)
# conflict set entry for w is gone
```

---

### Inference engine — select-and-fire loop

`InferenceEngine` wraps `ReteNetwork` and fires one instantiation per cycle
until the conflict set is empty (or a step cap is reached).

```python
from rete.engine import InferenceEngine
from rete import WME, Condition, Production

engine = InferenceEngine()

collected = []

engine.add_production(Production(
    lhs=[Condition("?x", "on", "table")],
    rhs=lambda token: collected.append(token.wmes[0].id),
))

engine.add_wme(WME("A", "on", "table"))
engine.add_wme(WME("B", "on", "table"))

fired = engine.run()
print(f"Fired {fired} rule(s), matched: {sorted(collected)}")
# Fired 2 rule(s), matched: ['A', 'B']
```

The default conflict-resolution strategy is **recency** (last-added wins).
Pass a custom callable to `InferenceEngine(strategy=...)` to change it; a
built-in FIFO strategy is available as `InferenceEngine.fifo_strategy`.

---

## Bundled examples

```bash
python src/examples/blocks_world.py   # Doorenbos §2.1 — three-pattern join
python src/examples/negation.py       # Doorenbos §2.7 — negated condition
python src/examples/sharing.py        # Doorenbos §2.3 — two productions sharing a beta node
```

## Dev

```bash
xenon --max-absolute A --max-modules A --max-average A src/ tests/
ruff check src/ tests/
pytest --cov
```

## History
- ***v1.0.0:*** supports triples