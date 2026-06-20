# PRL Parser — Step 7: PRL Versions of the Examples

## Overview

Create a `.prl` source file and a Python driver for each of the seven bundled
examples, then add a `TestPrlExamples` class to `tests/test_examples.py`.

No changes to `src/rete/` or any existing file except `tests/test_examples.py`.

---

## Files

| File | Action |
|---|---|
| `src/examples/prl/blocks_world.prl` | **CREATE** |
| `src/examples/prl/negation.prl` | **CREATE** |
| `src/examples/prl/sharing.prl` | **CREATE** |
| `src/examples/prl/loan_application.prl` | **CREATE** |
| `src/examples/prl/temperature_alarm.prl` | **CREATE** |
| `src/examples/prl/family_tree.prl` | **CREATE** |
| `src/examples/prl/fraud_detection.prl` | **CREATE** |
| `src/examples/blocks_world_prl.py` | **CREATE** |
| `src/examples/negation_prl.py` | **CREATE** |
| `src/examples/sharing_prl.py` | **CREATE** |
| `src/examples/loan_application_prl.py` | **CREATE** |
| `src/examples/temperature_alarm_prl.py` | **CREATE** |
| `src/examples/family_tree_prl.py` | **CREATE** |
| `src/examples/fraud_detection_prl.py` | **CREATE** |
| `tests/test_examples.py` | **UPDATE** — add `TestPrlExamples` class |

---

## PRL source files

### `blocks_world.prl`

```
// Doorenbos §2.1 — canonical blocks-world example.
// Expected match: x=B1, y=B2, z=B3.
declare On
  upper: str
  lower: str
end

declare LeftOf
  left: str
  right: str
end

declare Color
  block: str
  color: str
end

rule "find-stack-of-two-blocks-to-the-left-of-a-red-block"
  when
    On($x: upper, $y: lower)
    LeftOf(left == $y, $z: right)
    Color(block == $z, color == "red")
  then
    results.append((x, y, z))
end
```

### `negation.prl`

```
// Doorenbos §2.7 — negative conditions.
// Expected match: x=B3.
declare On
  upper: str
  lower: str
end

declare Color
  block: str
  color: str
end

rule "find-blocks-on-table-not-blue"
  when
    On(lower == "table", $x: upper)
    not Color(block == $x, color == "blue")
  then
    results.append(x)
end
```

### `sharing.prl`

```
// Doorenbos §2.3 — two productions; C1/C2 are logically identical.
// P1: tests z is red → x=B1,y=B2,z=B3
// P2: tests y is red → x=B1,y=B3,z=B4
declare On
  upper: str
  lower: str
end

declare LeftOf
  left: str
  right: str
end

declare Color
  block: str
  color: str
end

rule "find-stack-left-of-red-z"
  when
    On($x: upper, $y: lower)
    LeftOf(left == $y, $z: right)
    Color(block == $z, color == "red")
  then
    results.append((x, y, z))
end

rule "find-stack-left-of-red-y"
  when
    On($x: upper, $y: lower)
    LeftOf(left == $y, $z: right)
    Color(block == $y, color == "red")
  then
    results.append((x, y, z))
end
```

### `loan_application.prl`

```
// Loan approval — underage and bankruptcy rules.
declare Applicant
  name: str
  age: int
end

declare LoanApplication
  applicant: str
  approved: bool
  explanation: str
end

declare Bankruptcy
  name: str
  year: int
end

rule "underage"
  when
    Applicant(age < 21, $name: name)
    $loan: LoanApplication(approved == True, applicant == $name)
  then
    loan.obj.approved = False
    loan.obj.explanation = "Underage"
    update(loan)
end

rule "bankruptcy"
  when
    $loan: LoanApplication(approved == True, $name: applicant)
    Bankruptcy(name == $name, year > 1990)
  then
    loan.obj.approved = False
    loan.obj.explanation = "Bankruptcy"
    update(loan)
end
```

### `temperature_alarm.prl`

```
// Temperature monitoring — inserts Alert facts.
declare Temperature
  sensor: str
  value: float
end

declare Alert
  severity: str
  message: str
end

rule "too-hot"
  when
    /Temperature[value >= 80, $sensor: sensor, $value: value]
  then
    alerts.append(Alert("HIGH", f"Sensor {sensor}: {value}°C"))
end
```

`alerts` is an injected side-effect channel (not an Alert Fact in WM).
The test driver passes `types={"alerts": alerts}` so the list is available
in the exec namespace.

### `family_tree.prl`

```
// Transitive ancestor inference.
declare Parent
  parent: str
  child: str
end

declare Ancestor
  ancestor: str
  descendant: str
end

rule "direct"
  when
    Parent($p: parent, $c: child)
  then
    if (p, c) not in known:
        known.add((p, c))
        insert(Ancestor(p, c))
end

rule "transitive"
  when
    Parent($p: parent, $c: child)
    Ancestor(ancestor == $c, $d: descendant)
  then
    if (p, d) not in known:
        known.add((p, d))
        insert(Ancestor(p, d))
end
```

`known` is a `set[tuple[str, str]]` injected via `types={"known": known}`.
`Ancestor` is declared in this file; the compiler makes it available in the
exec namespace automatically.

### `fraud_detection.prl`

```
// Unauthorized transaction detection via NCC.
declare Transaction
  id: str
  account_id: str
  amount: float
end

declare Account
  id: str
  limit: float
end

declare Authorization
  transaction_id: str
end

rule "unauthorized"
  when
    Transaction($tx_id: id, $account_id: account_id)
    Account(id == $account_id)
    not ( Authorization(transaction_id == $tx_id) )
  then
    fired.append(tx_id)
end
```

---

## Driver pattern (all seven)

```python
"""<docstring matching the original>"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "<name>.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _make_facts(types: dict) -> list[Fact]:
    """Return the working-memory facts for this example."""
    ...  # one line per Fact


def _run() -> <return_type>:
    """Return the observable output — called by tests."""
    ...


def main() -> None:
    """Run with printed output and embedded assertions."""
    result = _run()
    ...  # print + assert


if __name__ == "__main__":
    main()
```

`_setup` complexity: 1 + 1(for) = 2. `_run` complexity: ≤ 3 in all cases.

---

## Driver specifics

### `blocks_world_prl.py`

```python
def _run() -> list[tuple[str, str, str]]:
    results: list = []
    engine, types = _setup({"results": results})
    for f in _make_facts(types):
        engine.add_fact(f)
    engine.run()
    return results
```

`_make_facts` returns the nine `On` / `LeftOf` / `Color` facts w1–w9.
Complexity of `_run`: 1 + 1(for) = 2.
Expected return: `[("B1", "B2", "B3")]`.

### `negation_prl.py`

```python
def _run() -> list[str]:
    results: list = []
    engine, types = _setup({"results": results})
    for f in _make_facts(types):
        engine.add_fact(f)
    engine.run()
    return results
```

Expected return: `["B3"]`.

### `sharing_prl.py`

```python
def _run() -> list[tuple[str, str, str]]:
    results: list = []
    engine, types = _setup({"results": results})
    for f in _make_facts(types):
        engine.add_fact(f)
    engine.run()
    return results
```

Expected return (as a set): `{("B1", "B2", "B3"), ("B1", "B3", "B4")}`.

### `loan_application_prl.py`

The PRL-compiled `LoanApplication` has no default values — all three
positional arguments are required.

```python
def _run() -> tuple:
    engine, types = _setup()
    Loan = types["LoanApplication"]
    alice_loan = Fact(Loan(applicant="Alice", approved=True, explanation=""))
    bob_loan   = Fact(Loan(applicant="Bob",   approved=True, explanation=""))
    carol_loan = Fact(Loan(applicant="Carol", approved=True, explanation=""))
    for f in _make_facts(types, alice_loan, bob_loan, carol_loan):
        engine.add_fact(f)
    engine.run()
    return alice_loan.obj, bob_loan.obj, carol_loan.obj
```

`_make_facts(types, alice_loan, bob_loan, carol_loan)` returns
the seven-element list (Applicant × 3, Loan × 3, Bankruptcy × 1).

Complexity of `_run`: 1 + 1(for) = 2.
Expected: `alice.approved=False`, `alice.explanation="Underage"`,
`bob.approved=True`, `carol.approved=False`, `carol.explanation="Bankruptcy"`.

### `temperature_alarm_prl.py`

```python
def _run() -> list:
    alerts: list = []
    engine, types = _setup({"alerts": alerts})
    engine.add_fact(Fact(types["Temperature"](sensor="T1", value=60.0)))
    engine.add_fact(Fact(types["Temperature"](sensor="T2", value=95.0)))
    engine.run()
    return alerts
```

Complexity: 1. Expected: `len == 1`, `"T2" in alerts[0].message`.

### `family_tree_prl.py`

```python
def _run() -> set[tuple[str, str]]:
    known: set[tuple[str, str]] = set()
    engine, types = _setup({"known": known})
    for f in _make_facts(types):
        engine.add_fact(f)
    engine.run()
    return known
```

`_make_facts` returns `[Fact(Parent("A","B")), Fact(Parent("B","C")), Fact(Parent("C","D"))]`.
Expected: `{("A","B"),("A","C"),("A","D"),("B","C"),("B","D"),("C","D")}`.

### `fraud_detection_prl.py`

Three-phase round-trip test.

```python
def _run() -> list[str]:
    fired: list[str] = []
    engine, types = _setup({"fired": fired})
    tx   = Fact(types["Transaction"](id="tx1", account_id="acc1", amount=500.0))
    acct = Fact(types["Account"](id="acc1", limit=1000.0))
    engine.add_fact(tx)
    engine.add_fact(acct)
    engine.run()                          # phase 1: fired=["tx1"]
    auth = Fact(types["Authorization"](transaction_id="tx1"))
    engine.add_fact(auth)
    engine.run()                          # phase 2: no re-fire
    engine.remove_fact(auth)
    engine.run()                          # phase 3: fired=["tx1","tx1"]
    return fired
```

Complexity: 1. Expected: `["tx1", "tx1"]`.

---

## `tests/test_examples.py` — new `TestPrlExamples` class

```python
class TestPrlExamples:
    """PRL drivers produce the same results as the Python originals."""

    def test_blocks_world_prl(self) -> None:
        from examples.blocks_world_prl import _run
        assert _run() == [("B1", "B2", "B3")]

    def test_negation_prl(self) -> None:
        from examples.negation_prl import _run
        assert _run() == ["B3"]

    def test_sharing_prl(self) -> None:
        from examples.sharing_prl import _run
        assert set(_run()) == {("B1", "B2", "B3"), ("B1", "B3", "B4")}

    def test_loan_application_prl(self) -> None:
        from examples.loan_application_prl import _run
        alice, bob, carol = _run()
        assert (alice.approved, alice.explanation) == (False, "Underage")
        assert bob.approved is True
        assert (carol.approved, carol.explanation) == (False, "Bankruptcy")

    def test_temperature_alarm_prl(self) -> None:
        from examples.temperature_alarm_prl import _run
        alerts = _run()
        assert len(alerts) == 1
        assert "T2" in alerts[0].message

    def test_family_tree_prl(self) -> None:
        from examples.family_tree_prl import _run
        assert _run() == {
            ("A", "B"), ("A", "C"), ("A", "D"),
            ("B", "C"), ("B", "D"), ("C", "D"),
        }

    def test_fraud_detection_prl(self) -> None:
        from examples.fraud_detection_prl import _run
        assert _run() == ["tx1", "tx1"]
```

All test methods: complexity 1 + 1–3 asserts ≤ 4. ✓

---

## Criticalities

### 1. PRL `declare` does not support default values

`make_dataclass` creates a class with no default values.
`LoanApplication(applicant, approved, explanation)` requires all three
arguments. The driver must **not** write `Loan("Alice")` (which worked in
the original because of `approved=True, explanation=""` defaults) — write
`Loan(applicant="Alice", approved=True, explanation="")` instead.

### 2. `True` / `False` in LHS constraints use Python capitalization

`LoanApplication(approved == True, …)` works because `True` is now a PRL
keyword (added in Step 6).  The original `drl.py` prototype accepted `true`
(Drools-style lowercase); the PRL stack accepts both, but the `.prl` source
files should use `True` / `False` consistently.

### 3. `$loan` is a Fact wrapper — `.obj` required for attribute access

In `loan_application.prl`:
```
$loan: LoanApplication(…)
then
    loan.obj.approved = False   # correct
    loan.approved = False       # AttributeError at runtime
```
Field bindings (`$name: applicant`) expose the raw attribute value directly;
fact bindings (`$loan: LoanApplication(…)`) expose the `Fact` wrapper.
Don't conflate them in any then-block.

### 4. `insert(obj)` takes the POJO, not a Fact

In `family_tree.prl`:
```
insert(Ancestor(p, c))   # correct — insert wraps it in Fact internally
insert(Fact(Ancestor(p, c)))  # double-wrapping → AttributeError in alpha test
```
The `_engine_helpers` `insert` implementation calls `engine.add_fact(Fact(obj))`.
Passing an already-wrapped `Fact` to `insert` would try to wrap a `Fact` in
another `Fact` — the alpha test would then call `isinstance(Fact(...), Ancestor)`
which is `False`, so the rule would never fire.

### 5. Alpha-memory sharing is not preserved in `sharing.prl`

The two rules in `sharing.prl` have logically identical C1 and C2, so the
original `sharing.py` shares a beta-memory node between them.  The PRL
compiler generates a fresh lambda object per rule per constraint; different
`id()` values → different alpha keys → separate alpha memories.  **The output
is identical** (both matches still fire); only the network topology differs.
This is a known limitation documented in Step 4's plan.  Do not claim
sharing in the driver's docstring.

### 6. `family_tree.prl` then-blocks have multi-line Python

```
then
    if (p, c) not in known:
        known.add((p, c))
        insert(Ancestor(p, c))
end
```

The lexer captures lines verbatim until a bare `end` line.  The `if` block
uses 4-space indentation inside the then-block; `textwrap.dedent` removes
the common leading whitespace.  Mixing tabs and spaces in the then-block
will break `dedent` silently — use spaces only.

The `if` branch adds complexity to the exec'd code, but since it lives
inside a string constant, xenon does not analyse it.  The `_make_rhs_closure`
function's own complexity is unchanged.

### 7. `fraud_detection_prl._run()` keeps a reference to the `Authorization` Fact

`engine.remove_fact(auth)` requires the same `Fact` object that was added.
`auth` must be constructed in `_run()`, passed to `engine.add_fact(auth)`,
and kept alive until `engine.remove_fact(auth)`.  Do not re-create the
`Authorization` object for the remove call — it would have a different
identity and the removal would silently fail (wrong fact).

### 8. `temperature_alarm.prl` — `°` character in then-block

The then-block source:
```python
alerts.append(Alert("HIGH", f"Sensor {sensor}: {value}°C"))
```
The `°` (U+00B0) is inside a Python string literal in the RAWBLOCK.  The
PRL lexer captures the RAWBLOCK verbatim (byte-level copy, no tokenisation);
`exec` receives valid UTF-8.  No issue, but the `.prl` file must be saved
as UTF-8 (the Python default).
