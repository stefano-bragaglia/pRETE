# pRETE (Python RETE)

A pure-Python implementation of the Rete algorithm for production rule systems,
matching over arbitrary Python objects (POPOs — Plain Old Python Objects).

> **v2.0.0 migration:** the `(id, attribute, value)` triple model has been
> replaced by Drools-style pattern matching over `@dataclass` (or any Python)
> objects.  `WME`, `Condition`, and `WILDCARD` are removed; use `Fact`,
> `Pattern`, and `JoinSpec` instead.  See [CHANGELOG.md](CHANGELOG.md).

![pRETE logo](images/pRETE-logo-small.png)

## Background

Implements the algorithm from:
- Forgy, C. L. (1982). Rete: A fast algorithm for the many pattern/many object pattern match problem. *Artificial Intelligence*, 19(1), 17–37.
- Doorenbos, R. B. (1995). *Production system techniques for large rule bases* (CMU-CS-95-113). Carnegie Mellon University.

v1.x represented working memory elements as `(id, attribute, value)` triples per
Doorenbos §2.1.  v2.0 replaces triples with Drools-style POPOs: any Python
object may be a fact; patterns match by type then by callable field tests.

## Install

```bash
pip install -e .[dev]
```

## Usage

### Core concepts

| Term | What it is |
|------|-----------|
| **`Fact(obj)`** | Wraps any Python object as a working-memory element |
| **`Pattern(type_, alpha_tests, join_tests, bindings, negated)`** | Matches facts by type then by callable field tests |
| **`JoinSpec(attr_of_fact, var_name)`** | Declares a cross-fact variable reference inside a `Pattern` |
| **`Production(lhs, rhs)`** | A rule: a list of `Pattern`s / `NccGroup`s (LHS) plus a Python callable (RHS) |
| **`ReteNetwork`** | The compiled network; call `add_fact` / `remove_fact` / `add_production` |
| **`InferenceEngine`** | Wraps `ReteNetwork` with a select-and-fire loop (`run()`); adds `update_fact` |

Variable names start with `$`.  A variable binds to an object field on its first
match (`bindings` tuple in `Pattern`) and must equal that value in every
subsequent condition that references it (`join_tests` / `JoinSpec`).

---

### Quick start — basic fact matching

```python
from dataclasses import dataclass
from rete import Fact, Pattern, JoinSpec, Production, ReteNetwork

@dataclass
class On:
    upper: str
    lower: str

def on_table(obj: On) -> bool:
    return obj.lower == "table"

net = ReteNetwork()

net.add_production(Production(
    lhs=[
        Pattern(On, bindings=(("$lower", "lower"),)),            # bind $lower = On.lower
        Pattern(On, alpha_tests=(on_table,),                     # alpha: lower == "table"
                    join_tests=(JoinSpec("upper", "$lower"),)),  # beta:  upper == $lower
    ],
    rhs=lambda token: print(token.bindings["$lower"], "is above table"),
))

net.add_fact(Fact(On("A", "B")))
net.add_fact(Fact(On("B", "table")))
net.add_fact(Fact(On("C", "table")))

# One match: A→B→table
for inst in net.conflict_set:
    inst.production.rhs(inst.token)
# B is above table
```

---

### Negated conditions

Pass `negated=True` to exclude facts that satisfy the pattern.

```python
@dataclass
class Color:
    block: str
    color: str

def is_blue(obj: Color) -> bool:
    return obj.color == "blue"

net.add_production(Production(
    lhs=[
        Pattern(On, alpha_tests=(on_table,), bindings=(("$block", "upper"),)),
        Pattern(Color, alpha_tests=(is_blue,),
                join_tests=(JoinSpec("block", "$block"),), negated=True),
    ],
    rhs=lambda token: print(token.bindings["$block"], "is on the table and not blue"),
))
```

---

### Negated conjunctive conditions (NCC)

`NccGroup` wraps multiple patterns that must **not** jointly match.

```python
from rete import NccGroup
```

See `src/examples/fraud_detection.py` for a full NCC example.

---

### Retraction

Removing a `Fact` automatically retracts every match that depended on it.

```python
f = Fact(On("A", "table"))
net.add_fact(f)
# ... rule fires ...
net.remove_fact(f)
# conflict set entry for f is gone
```

---

### Mutation — `update_fact`

POPOs are mutable; use `update_fact` to keep the network in sync after
modifying an object (equivalent to Drools `modify`):

```python
engine.update_fact(fact)   # retract + re-assert; object identity preserved
```

---

### Inference engine — select-and-fire loop

`InferenceEngine` wraps `ReteNetwork` and fires one instantiation per cycle
until the conflict set is empty (or a step cap is reached).

```python
from rete.engine import InferenceEngine
from rete import Fact, Pattern, Production

engine = InferenceEngine()

collected = []

engine.add_production(Production(
    lhs=[Pattern(On, alpha_tests=(on_table,), bindings=(("$upper", "upper"),))],
    rhs=lambda token: collected.append(token.bindings["$upper"]),
))

engine.add_fact(Fact(On("A", "table")))
engine.add_fact(Fact(On("B", "table")))

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
# Doorenbos classics (rewritten with @dataclass facts)
python src/examples/blocks_world.py      # §2.1 — three-pattern join
python src/examples/negation.py          # §2.7 — negated condition
python src/examples/sharing.py           # §2.3 — two productions sharing a beta node

# New non-trivial examples (v2.0)
python src/examples/loan_application.py  # multi-rule, update_fact, cross-fact binding
python src/examples/temperature_alarm.py # pure alpha test; RHS inserts new facts
python src/examples/family_tree.py       # transitive inference via engine.run() chaining
python src/examples/fraud_detection.py   # NCC group; retraction round-trip
```

## Dev

```bash
xenon --max-absolute A --max-modules A --max-average A src/ tests/
ruff check src/ tests/
pytest --cov
```

## History
- ***v2.0.0:*** Drools-style POPO matching — `Fact`, `Pattern`, `JoinSpec`; `update_fact`; MRO dispatch
- ***v1.0.1:*** incremental fixes
- ***v1.0.0:*** supports triples (`WME`, `Condition`)