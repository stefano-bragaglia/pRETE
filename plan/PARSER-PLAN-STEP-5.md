# PRL Parser — Step 5: End-to-End Integration Tests

## Overview

Add `tests/test_prl.py` — a full-pipeline integration test suite that exercises
`load_prl()` against a live `InferenceEngine`.  This step has **no new
implementation**: all tests exercise code already delivered in Steps 1–4.

The test suite mirrors the coverage of the pre-existing `tests/test_drl.py`
(the regex-based prototype) but uses the real PRL pipeline and cleaner
conventions.

---

## Files

| File | Action | Notes |
|---|---|---|
| `tests/test_prl.py` | **CREATE** | Integration tests; all should pass on first commit |

No `src/` files are modified.

---

## Module layout

```python
"""End-to-end integration tests for the PRL pipeline (``load_prl`` → engine)."""
from __future__ import annotations

from dataclasses import fields as dc_fields

import pytest

from rete.engine import InferenceEngine
from rete.fact import Fact
from rete.prl import load_prl


def _setup(
    prl: str,
    ctx: dict | None = None,
) -> tuple[InferenceEngine, dict]:
    """Compile *prl*, wire all productions into a new engine, return both."""
    engine = InferenceEngine()
    types, prods = load_prl(prl, types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types
```

`_setup` complexity: 2 (one `for` loop).

The `ctx` parameter is the combined namespace seed: user-defined types AND
communication channels (e.g. `{"fired": fired_list}`).  The returned `types`
dict contains all declared types plus every key from `ctx`.

---

## Test classes

### `TestDeclare`

| Test | What it checks |
|---|---|
| `test_class_generated` | `"Temp" in types`; `types["Temp"].__name__ == "Temp"` |
| `test_java_type_mapping` | `dc_fields(types["Bean"])` has `float` / `str` / `int` field types |
| `test_instance_creation` | `types["Temp"](value=42.0).value == 42.0` |
| `test_cross_declare_reference` | second `declare` uses first as a field type; instance nests correctly |

### `TestOoPath`

| Test | What it checks |
|---|---|
| `test_fires_for_matching_value` | `/Temp[value >= 80]` → `engine.run() == 1` when Temp(95) in WM |
| `test_does_not_fire_for_non_matching_value` | same pattern → 0 steps for Temp(60) |
| `test_no_constraint_matches_any_instance` | `/Temp` → fires for each Temp fact |
| `test_fact_binding_exposed_in_rhs` | `$t: /Temp[…]` → `t` is a `Fact`; `t.obj.value` is correct |

### `TestTraditional`

| Test | What it checks |
|---|---|
| `test_fires_for_matching_value` | `Temp(value > 50)` → fires for Temp(100) |
| `test_does_not_fire_for_non_matching_value` | 0 steps for Temp(10) |

### `TestBindingsAndJoins`

| Test | What it checks |
|---|---|
| `test_field_binding_value_in_rhs` | `$s: sensor` → `s` available in then-block |
| `test_cross_fact_join_fires` | `$name: name` in P1; `applicant == $name` in P2; fires for matching pair |
| `test_cross_fact_join_suppressed_for_mismatch` | non-matching key pair → 0 firings |

### `TestFactBinding`

| Test | What it checks |
|---|---|
| `test_fact_wrapper_accessible_in_rhs` | `$t: /Temp` → `t` is a `Fact` instance |
| `test_mutation_via_fact_binding` | `app.obj.approved = False` mutates the wrapped object |

### `TestNegation`

| Test | What it checks |
|---|---|
| `test_fires_when_negated_fact_absent` | `not Color(color == "blue")` fires when only red Color exists |
| `test_suppressed_when_negated_fact_present` | same rule does not fire when blue Color exists |

### `TestNcc`

| Test | What it checks |
|---|---|
| `test_fires_when_ncc_unsatisfied` | `not ( A(v==1) B(v==2) )` fires when A present but not B |
| `test_suppressed_when_ncc_satisfied` | rule does not fire when both A(v==1) and B(v==2) present |

### `TestRhsHelpers`

| Test | What it checks |
|---|---|
| `test_insert_triggers_second_rule` | `insert(Alert("HIGH"))` → second rule fires when Alert matched |
| `test_retract_helper_fires` | `retract(t)` → rule fires once, then conflict set is empty |
| `test_update_triggers_re_evaluation` | `app.obj.approved = False; update(app)` → mutation confirmed; no infinite loop |

### `TestComments`

| Test | What it checks |
|---|---|
| `test_line_comment_stripped` | `// comment` and inline `// comment` do not break parsing |
| `test_block_comment_stripped` | `/* block comment */` stripped |

### `TestEndToEnd`

| Test | What it checks |
|---|---|
| `test_temperature_alarm` | Two-declare program; `$t: /Temperature[value >= 80]`; RHS appends to `alerts`; exactly one alert fired for sensor reading 95°C; "S2" in message |

---

## PRL source conventions used in tests

### Fact binding convention

`$t: /TypeName[…]` → in the then-block, `t` is the **`Fact` wrapper**.
Access the underlying object via `t.obj.field`.  Mutation: `t.obj.field = value`.
Engine helpers: `retract(t)` and `update(t)` both expect the `Fact` wrapper.

This differs from `test_drl.py` (which may expose the POJO directly) and
from Drools DRL (Java getters/setters).  The convention is intentional and
consistent with how Step 4's `_make_rhs_closure` injects fact bindings.

### Side-effect capture

Since then-blocks run in `exec`, the cleanest way to observe a firing is to
pass a `list` via `ctx` and `append` to it in the then-block:

```python
fired: list = []
engine, types = _setup(prl_src, ctx={"fired": fired})
engine.run()
assert fired == [True]
```

The `ctx` dict merges with the exec namespace via `dict(types)` in
`_make_rhs_closure`, so `fired` is visible inside the then-block.

### NCC tests use an explicit `Trigger` fact

To avoid relying on engine behaviour for pure-NCC LHS productions (verified
safe but not exercised by existing tests), NCC tests include one positive
condition (`Trigger()`) before the `not ( … )` group.  This matches the
convention in `test_integration.py` and makes the test intent explicit.

A marker `declare Trigger\nend` with no fields creates a zero-field dataclass.
`types["Trigger"]()` constructs an instance.

---

## Criticalities

### 1. `$t` in then-block exposes the `Fact` wrapper, not `.obj`

`_make_rhs_closure` injects `token.facts[idx]` under the stripped name.
That is a `Fact` object.  Accessing the underlying data requires `t.obj.field`,
not `t.field`.  Tests that compare field values or mutate them must use `.obj`.

Writing `t.value = 99` in a then-block raises `AttributeError` at runtime
(since `Fact` has no `value` attribute); `t.obj.value = 99` is correct.

### 2. `update(app)` must be called after mutation to trigger re-evaluation

`app.obj.approved = False` mutates the fact in place.  Without `update(app)`,
the engine does not know the fact changed and will not re-match.  Tests that
verify mutation side-effects don't need `update` (the mutation is visible
directly on `fact.obj`), but tests that verify re-matching behaviour do.

### 3. The `update` helper must NOT cause an infinite loop

A rule that matches `App(approved == True)`, mutates `approved = False`, and
calls `update(app)` will NOT re-fire because the updated fact no longer
satisfies `approved == True`.  All `update` tests must be designed so the
post-mutation fact does not re-match the LHS.

### 4. `_setup` reuses the `ctx` dict as both the types seed AND namespace

When `ctx={"fired": fired}` is passed, `fired` becomes part of the resolved
`types` dict returned by `_setup`.  The then-block sees `fired` as a bare
name because `_make_rhs_closure` calls `ns.update(dict(types))`.  Do NOT
declare a PRL type with the same name as a `ctx` key — the type entry would
shadow the communication channel.

### 5. Cross-declare field-type references require declaration order

A `declare Outer` that references `Inner` as a field type must appear
**after** `declare Inner` in the PRL source.  `load_prl` processes declares
in source order.  Placing `Outer` before `Inner` causes `_java_type("Inner",
{})` to fall back to `typing.Any` (wrong) rather than the compiled `Inner`
class.  Test sources must respect declaration order.

### 6. `engine.run()` return value is the number of firings

Use `assert engine.run() == N` to verify exactly N rules fired.  This is more
robust than checking `len(fired)` when order matters, and avoids the need for
complex WM-state inspection.

### 7. `then\npass\nend` is a valid empty-effect rule for negative tests

Rules where the assertion is "did NOT fire" use `pass` as the then-block.
After `_strip_dollars` and `textwrap.dedent`, the code is `"pass\n"`;
`exec("pass\n", ns)` succeeds silently.  Always use this pattern rather than
an empty block, since an empty `rhs_src` also works but `pass` makes the
no-op intent explicit.
