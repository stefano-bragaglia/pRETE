# pRETE (Python RETE)

A pure-Python implementation of the Rete algorithm for production rule systems,
matching over arbitrary Python objects (POPOs — Plain Old Python Objects).

> **v2.0.0 breaking change:** the `(id, attribute, value)` triple model has
> been replaced by Drools-style pattern matching over `@dataclass` objects.
> `WME`, `Condition`, and `WILDCARD` are removed; use `Fact`, `Pattern`, and
> `JoinSpec` instead.  See [CHANGELOG.md](CHANGELOG.md).

> **v2.1.0 — pRETE Rule Language (PRL):** rules can now be written in `.prl`
> text files (a Python-flavoured subset of Drools Rule Language) and loaded
> directly into the engine via `load_prl()`.  The RETE engine itself is
> unchanged.

![pRETE logo](images/pRETE-logo-small.png)

## Background

Implements the algorithm from:
- Forgy, C. L. (1982). Rete: A fast algorithm for the many pattern/many object
  pattern match problem. *Artificial Intelligence*, 19(1), 17–37.
- Doorenbos, R. B. (1995). *Production system techniques for large rule bases*
  (CMU-CS-95-113). Carnegie Mellon University.

v1.x represented working memory as `(id, attribute, value)` triples per
Doorenbos §2.1.  v2.0 follows the Drools model: any Python object may be a
fact; patterns match by type then by callable field tests; variable bindings
are named and carried in the token.

---

## Install

```bash
pip install -e .[dev]
```

---

## Core concepts

| Term | What it is |
|------|-----------|
| `Fact(obj)` | Wraps any Python object as a working-memory element (identity semantics — two `Fact`s wrapping equal objects are distinct) |
| `Pattern(type_, alpha_tests, join_tests, bindings, negated)` | Matches facts by `isinstance` check then by callable field tests |
| `JoinSpec(attr_of_fact, var_name)` | Compile-time cross-fact constraint declared inside a `Pattern`; resolved at join time |
| `Production(lhs, rhs)` | A rule: a list of `Pattern`s / `NccGroup`s and a Python callable that receives the matched `Token` |
| `Token` | An immutable sequence of matched `Fact`s plus a `bindings: dict[str, Any]` of named variable values |
| `ReteNetwork` | The compiled network; call `add_fact` / `remove_fact` / `add_production` |
| `InferenceEngine` | Wraps `ReteNetwork` with a select-and-fire loop; adds `update_fact` |

Variable names start with `$`.  A variable binds to an object attribute on its
first match (`Pattern.bindings`) and must equal that value in every subsequent
condition that references it (`Pattern.join_tests` / `JoinSpec`).

---

## Quick start

### Single-pattern rule

```python
from dataclasses import dataclass
from rete import Fact, Pattern, Production, ReteNetwork

@dataclass
class Temperature:
    sensor: str
    value: float

def too_hot(obj: Temperature) -> bool:
    return obj.value >= 80.0

net = ReteNetwork()
alarms = []

net.add_production(Production(
    lhs=[Pattern(Temperature, alpha_tests=(too_hot,),
                 bindings=(("$sensor", "sensor"),))],
    rhs=lambda token: alarms.append(token.bindings["$sensor"]),
))

net.add_fact(Fact(Temperature("T1", 60.0)))
net.add_fact(Fact(Temperature("T2", 95.0)))

for inst in net.conflict_set:
    inst.production.rhs(inst.token)

print(alarms)   # ['T2']
```

### Cross-fact binding

Use `bindings` to capture a variable and `JoinSpec` to require it in a later
pattern.

```python
from dataclasses import dataclass
from rete import Fact, JoinSpec, Pattern, Production, ReteNetwork

@dataclass
class Color:
    block: str
    color: str

@dataclass
class Size:
    block: str
    size: str

def is_red(obj: Color) -> bool:   return obj.color == "red"
def is_large(obj: Size) -> bool:  return obj.size == "large"

net = ReteNetwork()

net.add_production(Production(
    lhs=[
        # bind $block = Color.block
        Pattern(Color, alpha_tests=(is_red,), bindings=(("$block", "block"),)),
        # require Size.block == $block
        Pattern(Size,  alpha_tests=(is_large,),
                join_tests=(JoinSpec("block", "$block"),)),
    ],
    rhs=lambda token: print(f"Block {token.bindings['$block']} is red and large"),
))

net.add_fact(Fact(Color("B1", "red")))
net.add_fact(Fact(Size("B1", "large")))
net.add_fact(Fact(Color("B2", "red")))    # B2 has no matching Size → no match

for inst in net.conflict_set:
    inst.production.rhs(inst.token)
# Block B1 is red and large
```

> **Alpha sharing note:** two `Pattern`s that pass the **same function object**
> in `alpha_tests` share one alpha memory.  Always use stable, module-level
> functions — not inline lambdas — when sharing matters.

### Negated conditions

`negated=True` makes the pattern a blocking condition: the rule fires only when
**no** fact satisfies it.

```python
@dataclass
class Marker:
    block: str
    key: str

def is_broken(obj: Marker) -> bool:
    return obj.key == "broken"

net.add_production(Production(
    lhs=[
        Pattern(Color, alpha_tests=(is_red,), bindings=(("$block", "block"),)),
        Pattern(Marker, alpha_tests=(is_broken,),
                join_tests=(JoinSpec("block", "$block"),), negated=True),
    ],
    rhs=lambda token: print(f"{token.bindings['$block']} is red and not broken"),
))
```

### Negated conjunctive conditions (NCC)

`NccGroup` wraps several patterns that must **not** jointly match.

```python
from rete import NccGroup

net.add_production(Production(
    lhs=[
        Pattern(Color, alpha_tests=(is_red,), bindings=(("$block", "block"),)),
        NccGroup(conditions=(
            Pattern(Marker, alpha_tests=(is_broken,),
                    join_tests=(JoinSpec("block", "$block"),)),
        )),
    ],
    rhs=lambda token: print("match"),
))
```

See `src/examples/fraud_detection.py` for a full NCC round-trip example.

### Retraction

Removing a `Fact` automatically retracts every match that depended on it.

```python
f = Fact(Temperature("T3", 90.0))
net.add_fact(f)
# ... conflict set has a new entry ...
net.remove_fact(f)
# conflict set entry is gone
```

### Mutation — `update_fact`

POPOs are mutable.  Mutate an attribute in place, then call `update_fact` to
resync the network (equivalent to Drools `modify`).  Object identity is
preserved across the retract / re-assert cycle.

```python
engine = InferenceEngine()
# ... add productions and facts ...
fact.obj.approved = False   # mutate in place
engine.update_fact(fact)    # retract → re-assert
engine.run()
```

### Inference engine — select-and-fire loop

`InferenceEngine` wraps `ReteNetwork` with a `run()` loop.

```python
from rete import Fact, InferenceEngine, Pattern, Production

@dataclass
class Item:
    name: str

engine = InferenceEngine()
found = []

engine.add_production(Production(
    lhs=[Pattern(Item, bindings=(("$name", "name"),))],
    rhs=lambda token: found.append(token.bindings["$name"]),
))

engine.add_fact(Fact(Item("apple")))
engine.add_fact(Fact(Item("banana")))

fired = engine.run()
print(f"Fired {fired} rule(s): {sorted(found)}")
# Fired 2 rule(s): ['apple', 'banana']
```

The default conflict-resolution strategy is **recency** (last-added wins).
`InferenceEngine.fifo_strategy` is also available; pass any callable as
`InferenceEngine(strategy=...)` for a custom policy.

---

## MRO dispatch

The alpha network dispatches by `type(fact.obj).__mro__`, so a `Dog` fact
reaches a `Pattern(type_=Animal)` automatically.  No explicit registration
needed.

---

## pRETE Rule Language (PRL)

PRL is a text notation for writing rules without touching Python — a strict
subset of [Drools Rule Language](https://docs.drools.org) adapted for pRETE.
Rules live in `.prl` files; `load_prl()` compiles them into `Production`
objects and hands them to the engine.

### What PRL supports

| Construct | Example |
|---|---|
| Fact-type declaration | `declare Temperature value: double end` |
| OOPath pattern | `/Temperature[value >= 80]` |
| Traditional pattern | `Temperature(value >= 80)` |
| Fact binding | `$t: /Temperature[value >= 80]` |
| Field binding | `$v: value` inside `[…]` |
| Cross-fact join | `field == $bound_var` |
| Single negation | `not /Temperature[value < 0]` |
| Conjunctive negation (NCC) | `not ( Pattern1() Pattern2() )` |
| Rule salience | `salience 10` |
| RHS helpers | `insert(obj)`, `retract(obj)`, `update(obj)` |

### Quick start

```
// temperature_alarm.prl
declare Temperature
  sensor: String
  value:  double
end

declare Alert
  message: String
end

rule "Too Hot"
  salience 10
  when
    $t: /Temperature[value >= 80]
  then
    insert(Alert("Sensor " + t.sensor + " too hot"))
end
```

```python
from pathlib import Path
from rete import Fact, InferenceEngine
from rete.prl import load_prl

engine = InferenceEngine()
types, productions = load_prl(
    Path("temperature_alarm.prl").read_text(), engine=engine
)
for p in productions:
    engine.add_production(p)

Temperature = types["Temperature"]
engine.add_fact(Fact(Temperature(sensor="S1", value=95.0)))
engine.run()
```

The grammar is documented in [`reference/prl-grammar.ebnf`](reference/prl-grammar.ebnf).

---

## Bundled examples

```bash
# Doorenbos classics (v2.0 rewrite with @dataclass facts)
python src/examples/blocks_world.py      # §2.1 — three-pattern join
python src/examples/negation.py          # §2.7 — negated condition
python src/examples/sharing.py           # §2.3 — two productions sharing a beta node

# Non-trivial examples (v2.0)
python src/examples/loan_application.py  # update_fact; cross-fact binding; _approved guard
python src/examples/temperature_alarm.py # pure alpha test; RHS inserts new facts
python src/examples/family_tree.py       # transitive inference via forward chaining
python src/examples/fraud_detection.py   # NccGroup; retraction round-trip
```

---

## Dev

```bash
hatch run check   # xenon (complexity A) + ruff + pytest --cov (fail-under 80)
```

Individual tools:
```bash
xenon --max-absolute A --max-modules A --max-average A src/ tests/
ruff check src/ tests/
pytest --cov
```

---

## History
- **v2.1.0** — PRL parser: `load_prl()`, `.prl` files, lexer / AST / compiler pipeline
- **v2.0.0** — Drools-style POPO matching: `Fact`, `Pattern`, `JoinSpec`; `update_fact`; MRO dispatch; named variable bindings on `Token`
- **v1.0.1** — incremental fixes
- **v1.0.0** — triple WME model (`WME`, `Condition`)
