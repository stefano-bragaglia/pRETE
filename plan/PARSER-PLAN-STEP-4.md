# PRL Parser — Step 4: Compiler (AST → RETE Objects)

## Overview

Implement the compiler that converts a `ProgramNode` (from Step 3) into the
types and `Production` objects that the RETE network understands.  The public
entry point is a single function:

```python
from rete.prl import load_prl

types, productions = load_prl(source_text, types={}, engine=engine)
for p in productions:
    engine.add_production(p)
```

The compiler is the bridge between the PRL parse tree (AST nodes) and the live
RETE engine (alpha/beta networks).  It has three sub-tasks:

- **4a — Type resolution:** `DeclareDecl` → Python dataclass via `make_dataclass`
- **4b — LHS compilation:** `PatternNode / NccPatternGroup` → `Pattern / NccGroup`
- **4c — RHS compilation:** `rhs_src` (raw Python text) → `Callable[[Token], None]`

---

## Files

| File | Action | Notes |
|---|---|---|
| `src/rete/prl.py` | **CREATE** | Compiler + `load_prl()` public function |
| `tests/test_prl_compiler.py` | **CREATE** | Unit tests; committed before code per workflow |

No existing files are modified in this step.

---

## Constants

```python
import operator, re, textwrap
from dataclasses import make_dataclass
from typing import Any

_JAVA_TO_PY: dict[str, type] = {
    "int": int, "long": int, "short": int, "byte": int,
    "double": float, "float": float,
    "boolean": bool,
    "String": str, "char": str,
    "Object": object,
}

_OPS: dict[str, Any] = {
    "==": operator.eq, "!=": operator.ne,
    "<":  operator.lt, "<=": operator.le,
    ">":  operator.gt, ">=": operator.ge,
}

_VAR_RE = re.compile(r'\$([A-Za-z_]\w*)')
```

---

## Objects / Functions

All functions are module-level in `prl.py`.  Only `load_prl` is public
(`__all__ = ["load_prl"]`).  Tests import private helpers directly by name.

### Public entry point

```python
def load_prl(
    text: str,
    types: dict[str, type] | None = None,
    engine=None,
) -> tuple[dict[str, type], list[Production]]:
    ...
```

Orchestrates: `tokenize → parse → _compile_declares → [_compile_rule, ...]`.

Complexity budget: 3 (one `for`, one comprehension `for`).

---

### 4a — Type resolution

#### `_compile_declare(decl, types) -> type`

Calls `make_dataclass(decl.name, fields)` where `fields` is a list of
`(field_name, python_type)` tuples built from `decl.fields`.  Python type is
resolved via `_java_type`.

Complexity: 2 (one comprehension `for`).

#### `_java_type(name, types) -> type`

Checks `types` dict first (user-declared types may reference each other), then
`_JAVA_TO_PY`, then falls back to `typing.Any`.

Complexity: 2 (one `if`).

#### `_resolve_type(name, types) -> type`

Looks up `name` in `types`; raises `NameError` if absent (user forgot to
`declare` the type).

Complexity: 2 (one `if`).

---

### 4b — LHS compilation

#### `_compile_lhs(lhs_nodes, types) -> (list[Pattern | NccGroup], dict[str, int])`

Iterates over `lhs_nodes` (tuple of `PatternNode | NccPatternGroup`), building:
- `conditions`: the `Pattern | NccGroup` list for `Production.lhs`
- `fact_bindings`: `{fact_var_str: index}` for use by `_compile_rhs`

Complexity: 3 (one `for`, one `if`).

#### `_compile_pattern(node, idx, types, fact_bindings) -> Pattern`

1. Resolves `node.type_name` via `_resolve_type`.
2. If `node.fact_var` is set, records `fact_bindings[node.fact_var] = idx`.
3. Calls `_compile_constraints(node.constraints)` for the three component lists.
4. Returns `Pattern(type_, alpha_tests, join_tests, bindings, node.negated)`.

Complexity: 2 (one `if`).

#### `_compile_constraints(constraints) -> (tuple, tuple, tuple)`

Walks `constraints` (BindConstraint | CompareConstraint), accumulating three lists:
`alpha_tests`, `join_tests`, `bindings`.  Delegates each item to
`_apply_constraint`.  Returns the three lists converted to `tuple`.

Complexity: 2 (one `for`).

#### `_apply_constraint(c, alpha_tests, join_tests, bindings) -> None`

Dispatches on `isinstance(c, BindConstraint)`:

- `BindConstraint(var, field)` → append `(var, field)` to `bindings`.
- `CompareConstraint(field, op, rhs)`:
  - `isinstance(rhs, str) and rhs.startswith("$")` → `join_tests.append(JoinSpec(field, rhs))`
  - else → `alpha_tests.append(_make_alpha_test(field, op, rhs))`

Complexity: 4 (one `if`, one `if+and`).

#### `_make_alpha_test(field, op, rhs) -> Callable`

Returns a closure `lambda obj: _OPS[op](_getattr_path(obj, field), rhs)`.
The `_OPS` lookup is bound at compile time (not at call time).

Complexity: 1.

#### `_getattr_path(obj, path) -> Any`

Resolves a dotted field path (`"address.city"`) via iterative `getattr`.
Single-segment paths are the common case and have no overhead.

Complexity: 2 (one `for`).

#### `_compile_ncc(node, types) -> NccGroup`

Compiles each `PatternNode` in `node.patterns` via `_compile_pattern` (passing
`-1` as index and an empty `fact_bindings` dict — NCC-internal patterns cannot
be referenced from the outer RHS).  Returns `NccGroup(tuple(patterns))`.

Complexity: 2 (one comprehension `for`).

---

### 4c — RHS compilation

#### `_compile_rhs(rhs_src, fact_bindings, types, engine) -> Callable[[Token], None]`

1. `_strip_dollars(rhs_src)` — replace `$var` → `var` throughout.
2. `textwrap.dedent(...)` — strip uniform leading whitespace.
3. Build `ns_base = dict(types)`.
4. If `engine is not None`, update `ns_base` with `_engine_helpers(engine)`.
5. Return `_make_rhs_closure(code, fact_bindings, ns_base)`.

Complexity: 2 (one `if`).

#### `_make_rhs_closure(code, fact_bindings, ns_base) -> Callable[[Token], None]`

Returns the inner `rhs(token)` closure.  The closure:

1. Starts a fresh namespace: `ns = dict(ns_base)`.
2. Injects field bindings: `{k.lstrip("$"): v for k, v in token.bindings.items()}`.
3. Injects fact bindings: for each `(var, idx)` in `fact_bindings`, sets
   `ns[var.lstrip("$")] = token.facts[idx]`.
4. Executes: `exec(code, ns)`.

Outer function complexity: 1.  Inner `rhs` closure complexity: 3 (one
comprehension `for`, one loop `for`).

#### `_strip_dollars(src) -> str`

`_VAR_RE.sub(r'\1', src)` — single-line, complexity 1.

#### `_engine_helpers(engine) -> dict[str, Callable]`

Returns `{"insert": ..., "retract": ..., "update": ...}` where each helper
delegates to the corresponding `engine` method:
- `insert(obj)` → `engine.add_fact(Fact(obj))`
- `retract(fact)` → `engine.remove_fact(fact)`
- `update(fact)` → `engine.update_fact(fact)`

Complexity: 1 (no branches in the outer function; each inner `def` is trivially simple).

---

## Method table summary

| Function | Complexity | Notes |
|---|---|---|
| `load_prl` | 3 | public; calls tokenize + parse + compile |
| `_compile_declare` | 2 | make_dataclass |
| `_java_type` | 2 | dict lookup with Any fallback |
| `_resolve_type` | 2 | NameError guard |
| `_compile_lhs` | 3 | for + if dispatch |
| `_compile_pattern` | 2 | single if for fact_var |
| `_compile_constraints` | 2 | for loop |
| `_apply_constraint` | 4 | isinstance + str.startswith |
| `_make_alpha_test` | 1 | lambda factory |
| `_getattr_path` | 2 | for loop over dotted parts |
| `_compile_ncc` | 2 | comprehension |
| `_compile_rhs` | 2 | single if for engine |
| `_make_rhs_closure` | 1 | returns inner closure |
| `_strip_dollars` | 1 | regex sub |
| `_engine_helpers` | 1 | dict of three closures |
| `rhs` (inner closure) | 3 | comprehension + for |

All ≤ 5. ✓

---

## Tests — `tests/test_prl_compiler.py`

Committed **before** the implementation.  Tests import private helpers directly
(e.g., `from rete.prl import _compile_declare, _compile_lhs, _compile_rhs`).
The `Token` import is aliased as `ReteToken` to avoid shadowing any local name.

| Class | Cases |
|---|---|
| `TestJavaType` | `double` → `float`; `String` → `str`; `boolean` → `bool`; `int` / `long` → `int`; `Object` → `object`; unknown → `Any`; user type takes priority over `_JAVA_TO_PY` |
| `TestCompileDeclare` | No fields → empty dataclass; one field; two fields; field type is the actual Python type (not a string); instances can be created from the generated class |
| `TestResolveType` | Known type returns the type; unknown type raises `NameError` |
| `TestCompilePattern` | Pattern `type_` matches declared class; alpha test fires for matching value; alpha test does not fire for non-matching value; bind constraint produces binding tuple; compare constraint produces alpha test; join constraint (`$var` rhs) produces `JoinSpec`; `negated=True` on `PatternNode` propagates to `Pattern.negated` |
| `TestCompileNcc` | One-pattern NCC → `NccGroup` with one `Pattern`; two-pattern NCC → two `Pattern` instances |
| `TestCompileLhs` | Single `PatternNode` → `conditions` of length 1; `fact_var` on a node populates `fact_bindings`; `NccPatternGroup` → `NccGroup` in result; mixed list (pattern + NCC) → two items in `conditions` |
| `TestStripDollars` | `$foo` → `foo`; `$a.field` → `a.field`; no `$` → unchanged; `$` in a string literal is also stripped (accepted limitation — PRL then-blocks rarely quote `$` names) |
| `TestCompileRhs` | Empty `rhs_src` → callable that accepts a token without error; field binding accessible by stripped name in exec; fact binding injects `token.facts[idx]` at stripped name; `insert` / `retract` / `update` present when `engine` given; when `engine=None` the helpers are absent from the namespace |
| `TestLoadPrl` | Declare + rule → `types` dict contains the new class; `productions` list has one `Production`; `Production.lhs` is non-empty; types passed in via `types=` are available to the program |

Keep each test method to ≤ 4 `assert` statements.

### Helper setup needed in the test file

```python
from dataclasses import dataclass, fields as dc_fields
from rete.fact import Fact
from rete.fact import Token as ReteToken
from rete.condition import JoinSpec, NccGroup, Pattern
from rete.prl_ast import (
    BindConstraint, CompareConstraint,
    DeclareDecl, FieldDecl, NccPatternGroup, PatternNode, ProgramNode,
)
from rete.prl import (
    _compile_declare, _compile_lhs, _compile_rhs, _java_type,
    _resolve_type, _strip_dollars,
    load_prl,
)

@dataclass
class _Temp:
    """Minimal fact type for compiler tests."""
    value: float

@dataclass
class _Sensor:
    id: str
    value: float

_TYPES = {"_Temp": _Temp, "_Sensor": _Sensor}
```

---

## Criticalities

### 1. `_make_alpha_test` captures `rhs` by closure — beware the late-binding trap

Python closures capture variables by reference, not by value.  A naïve loop:

```python
for c in constraints:
    tests.append(lambda obj: op_fn(getattr(obj, c.field), c.rhs))
```

… captures `c` by reference; all lambdas see the **last** value of `c`.  Always
use default-argument capture:

```python
lambda obj, _field=c.field, _fn=op_fn, _rhs=c.rhs: _fn(getattr(obj, _field), _rhs)
```

OR extract a factory function (`_make_alpha_test`) so each call creates a
distinct closure frame.  The plan uses the factory approach — `_make_alpha_test`
already creates a fresh frame per call, so the loop does not capture `c`
directly.  Do not inline the lambda back into `_apply_constraint`.

### 2. `BindConstraint.var` carries the `$` prefix; `Pattern.bindings` must keep it

The engine stores extracted values as `token.bindings["$v"] = value`.  The RHS
closure then injects `{"v": value}` (stripped) into the exec namespace.  If the
compiler strips `$` from `var` before storing in `Pattern.bindings`, the engine
writes `token.bindings["v"]` but the RHS closure strips nothing → the variable
is double-accessible (once with and once without `$`).  Keep the full `"$v"` in
`Pattern.bindings`; strip only in the RHS namespace.

### 3. `_compile_ncc` passes an empty `fact_bindings` dict

Patterns inside a `not ( … )` group are matched against working memory
independently of the outer LHS.  Their fact-level bindings (`$fact: /Type[…]`)
cannot be referenced from the outer RHS (the engine does not expose NCC-internal
token contents).  Pass `{}` as `fact_bindings` for NCC sub-patterns and discard
the result.  If a user writes `$x: /T[]` inside an NCC, the binding is silently
ignored (acceptable — the grammar allows it but the engine cannot use it).

### 4. `exec(code, ns)` uses a single namespace dict (no separate globals/locals)

Calling `exec(code, globals_dict, locals_dict)` with two separate dicts causes
assignments in the exec'd code to land in `locals_dict`, but name lookups (for
functions defined at the top of the then-block) check `globals_dict` first.
This means a `def foo():` defined in one statement is not visible in the next
`foo()` call.  Use a single dict for both: `exec(code, ns)` (passing `ns` as
globals only, so locals defaults to the same dict).

### 5. `textwrap.dedent` needs consistent indentation

The RAWBLOCK from the lexer preserves the user's indentation relative to the
rule body.  `textwrap.dedent` removes the **common leading whitespace** across
all non-empty lines.  If the user mixes tabs and spaces, `dedent` will not
remove anything.  This is a PRL authoring concern, not an implementation bug;
document it in `load_prl`'s docstring as a known limitation.

### 6. `_strip_dollars` replaces `$` in string literals too

The regex `\$([A-Za-z_]\w*)` does not distinguish `$var` in code from `$var`
inside a string literal (e.g., `"$foo"`).  In practice, PRL then-blocks rarely
quote variable names, so this is acceptable.  If it becomes a problem in Step 7
(examples), add a `# ponytail:` note at that call site.

### 7. Alpha-memory sharing is not preserved across rules with the same constraint

`Pattern.alpha_key()` uses `id(fn)` for each alpha test.  Two patterns with
logically identical constraints (e.g., `value >= 80`) produce **different**
lambda objects → different `id()` values → no shared alpha memory.  This is
a performance limitation of the compiled approach, not a correctness bug.
Alpha sharing requires module-level named functions (as the existing examples
do).  Deferred to an optimisation pass in a later step.

### 8. `make_dataclass` produces mutable dataclasses by default

`make_dataclass(name, fields)` does not pass `frozen=True`.  The resulting
class is mutable, which matches Drools POJO semantics (the RHS can mutate
`loan.approved = False`).  Do not add `frozen=True` — the engine's
`update_fact` assumes the object can be mutated in place.

### 9. Salience is carried on `RuleDecl` but not yet on `Production`

The current `Production` dataclass has `lhs` and `rhs` only — no `salience`
field.  The compiler must NOT silently discard the parsed salience value.
Pass it as a `# ponytail:` comment in `_compile_rule` noting where it will be
wired when the engine is extended:

```python
# ponytail: rule.salience parsed but Production has no salience field yet
```

This keeps the intent visible without modifying engine internals in this step.

### 10. `engine=None` is a valid call for pure-reasoning rules

`load_prl(text)` (no engine) should still compile successfully.  The helpers
(`insert`, `retract`, `update`) are simply absent from the exec namespace;
any then-block that calls them will raise `NameError` at runtime (acceptable —
document in docstring).  Do not inject `None`-returning stubs.
