# pRETE (Python RETE)

A pure-Python implementation of the Rete algorithm for production rule systems,
matching over arbitrary Python objects (POPOs — Plain Old Python Objects).

![pRETE logo](https://raw.githubusercontent.com/stefano-bragaglia/pRETE/main/images/pRETE-logo-small-wide.png)

[![CI](https://github.com/stefano-bragaglia/pRETE/actions/workflows/ci.yml/badge.svg)](https://github.com/stefano-bragaglia/pRETE/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/prete)](https://pypi.org/project/prete/)
[![Python versions](https://img.shields.io/pypi/pyversions/prete)](https://pypi.org/project/prete/)
[![License](https://img.shields.io/github/license/stefano-bragaglia/pRETE)](LICENSE)

> **v2.0.0 breaking change:** the `(id, attribute, value)` triple model has
> been replaced by Drools-style pattern matching over `@dataclass` objects.
> `WME`, `Condition`, and `WILDCARD` are removed; use `Fact`, `Pattern`, and
> `JoinSpec` instead.  See [CHANGELOG.md](CHANGELOG.md).

> **v2.1.0 — pRETE Rule Language (PRL):** rules can now be written in `.prl`
> text files (a Python-flavoured subset of Drools Rule Language) and loaded
> directly into the engine via `load_prl()`.  The RETE engine itself is
> unchanged.

> **v2.5.0 — PRL Extra Features:** ten new PRL language constructs — type
> inheritance (`extends`), identity keys (`@key`), positional/named constraint
> shorthand, `@no-loop` tag, `import` / `from … import`, `or` disjunction,
> `forall`, `exists`, CEP event semantics (`@role`, `@timestamp`, `@expires`),
> and `accumulate` with built-in aggregation functions.  New engine nodes:
> `ExistsNode`, `AccumulateNode`; logical clock for CEP.

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
pip install prete
```

For development:

```bash
pip install -e ".[dev]"
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

See `src/examples/programmatic/fraud_detection.py` for a full NCC round-trip example.

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

#### Core (v2.1.0)

| Construct | Example |
|---|---|
| Fact-type declaration | `declare Temperature value: double end` |
| OOPath pattern | `/Temperature[value >= 80]` |
| Traditional pattern | `Temperature(value >= 80)` |
| Fact binding | `$t: Temperature(value >= 80)` |
| Field binding | `$v: value` inside a pattern |
| Cross-fact join | `field == $bound_var` |
| Single negation | `not Temperature(value < 0)` |
| Conjunctive negation (NCC) | `not ( Pattern1() Pattern2() )` |
| Rule salience | `salience 10` |
| RHS helpers | `insert(obj)`, `retract(obj)`, `update(obj)` |

#### Extra features (v2.5.0)

| Construct | Example |
|---|---|
| Type inheritance | `declare Dog extends Animal` |
| Identity key | `@key` before a field in `declare` — custom `__eq__`/`__hash__` |
| Positional constraints | `Point(0, 0)` — values matched left-to-right by declaration order |
| Named constraints | `Point(y=0)` — any subset, any order |
| `@no-loop` tag | `@no-loop` before `rule` — prevents self-re-activation |
| Python imports | `from myapp.models import Customer` at top of `.prl` file |
| `or` disjunction | `PatternA() or PatternB()` — compiler expands to N productions |
| `forall` | `forall(Order(status=="pending"), Approval(orderId==$o.id))` |
| `exists` | `exists Invoice(overdue == true)` — fires once per left context |
| `@role(event)` / `@timestamp` / `@expires` | CEP — events expire automatically after `advance_clock(t)` |
| `accumulate` | `accumulate(Order($a: amount); $total: sum($a); $total > 1000)` |

#### Field defaults and generics (v2.5.3)

| Construct | Example |
|---|---|
| Field default | `stage: str = null`, `history: list[str] = []` — mirrors Python's `@dataclass`; mutable defaults never alias across instances |
| Bracket generics | `list[str]`, `dict[str, int]` — replaces the old, erased Java-diamond form (`List<String>`) |

### Quick start

```
// temperature_alarm.prl
declare Temperature
  sensor: str
  value:  float
end

declare Alert
  message: str
end

rule "Too Hot"
  salience 10
  when
    $t: /Temperature[value >= 80]
  then
    insert(Alert("Sensor " + t.obj.sensor + " too hot"))
end
```

```python
from pathlib import Path
from rete import Fact, InferenceEngine, load_prl

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

### v2.5.0 feature examples

**Type inheritance** — a rule on the parent type fires for child facts:

```
declare Animal  name: String  end
declare Dog extends Animal  breed: String  end

rule "greet animal"
when
  $a: Animal()
then
  greet(a)
end
```

**`exists`** — fires once per account regardless of how many overdue invoices exist:

```
rule "alert account"
when
  $acc: Account()
  exists Invoice(accountId == $acc.id, overdue == true)
then
  alert(acc)
end
```

**`accumulate`** — aggregate and constrain in the LHS:

```
rule "flag high spend"
when
  accumulate(
    Order($amount: amount);
    $total: sum($amount);
    $total > 1000
  )
then
  results.append(total)
end
```

**CEP** — events expire automatically after the logical clock advances:

```
@role(event)
@expires(30s)
declare StockTick
  @timestamp
  ts: float
  symbol: String
  price: float
end
```

```python
engine.add_fact(Fact(StockTick(ts=0.0, symbol="ACME", price=42.0)))
engine.advance_clock(31.0)   # tick expired — retracted before next run()
engine.run()
```

The grammar is documented in [`reference/prl-grammar.ebnf`](reference/prl-grammar.ebnf).

**Editor support:** [prl-highlight](https://github.com/stefano-bragaglia/prl-highlight) provides `.prl` syntax highlighting for PyCharm.

---

## Bundled examples

Examples are split into two folders:

```bash
# src/examples/programmatic/ — pure Python, no .prl files
python src/examples/programmatic/blocks_world.py      # §2.1 — three-pattern join
python src/examples/programmatic/negation.py          # §2.7 — negated condition
python src/examples/programmatic/sharing.py           # §2.3 — two productions sharing a beta node
python src/examples/programmatic/loan_application.py  # update_fact; cross-fact binding
python src/examples/programmatic/temperature_alarm.py # alpha test; RHS inserts new facts
python src/examples/programmatic/family_tree.py       # transitive inference
python src/examples/programmatic/fraud_detection.py   # NccGroup; retraction round-trip

# src/examples/declarative/ — load rules from .prl files in declarative/prl/
python src/examples/declarative/blocks_world_prl.py      # PRL equivalent of blocks_world
python src/examples/declarative/negation_prl.py
python src/examples/declarative/sharing_prl.py
python src/examples/declarative/loan_application_prl.py
python src/examples/declarative/temperature_alarm_prl.py
python src/examples/declarative/family_tree_prl.py
python src/examples/declarative/fraud_detection_prl.py
python src/examples/declarative/inheritance_prl.py       # ES-1: extends
python src/examples/declarative/identity_key_prl.py      # ES-3: @key
python src/examples/declarative/compact_patterns_prl.py  # ES-4: positional/named constraints
python src/examples/declarative/self_modify_prl.py       # ES-2: @no-loop
python src/examples/declarative/imported_types_prl.py    # ES-5: import
python src/examples/declarative/disjunction_prl.py       # ES-6: or
python src/examples/declarative/universal_prl.py         # ES-6: forall
python src/examples/declarative/existence_check_prl.py   # ES-7: exists
python src/examples/declarative/event_stream_prl.py      # ES-8: CEP
python src/examples/declarative/aggregation_prl.py       # ES-9: accumulate
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
- **v2.5.2** — improving documentation and adding links to sister projects
- **v2.5.1** — test suite expanded to 99%+ coverage
- **v2.5.0** — ten PRL language extensions: `extends`, `@key`, positional/named constraints, `@no-loop` tag, `import`, `or`/`forall`, `exists`, CEP (`@role`/`@timestamp`/`@expires`), `accumulate`; new `ExistsNode` and `AccumulateNode` beta nodes; logical clock; examples reorganised into `declarative/` and `programmatic/`
- **v2.1.0** — PRL parser: `load_prl()`, `.prl` files, lexer / AST / compiler pipeline
- **v2.0.0** — Drools-style POPO matching: `Fact`, `Pattern`, `JoinSpec`; `update_fact`; MRO dispatch; named variable bindings on `Token`
- **v1.0.1** — incremental fixes
- **v1.0.0** — triple WME model (`WME`, `Condition`)
