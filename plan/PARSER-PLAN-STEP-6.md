# PRL Parser — Step 6: Package Integration

## Overview

Wire `load_prl` into the public package API: re-export it from
`src/rete/__init__.py` and fix the PRL quick-start section in `README.md`
(which still shows Java type names and wrong fact-access syntax from before
the Python-naming switch).

No new implementation code.  One new test line.

---

## Files

| File | Action |
|---|---|
| `src/rete/__init__.py` | Add import + `__all__` entry |
| `README.md` | Fix PRL quick-start: Python names, `t.obj.sensor`, import path |
| `tests/test_prl.py` | Add one smoke test for top-level import |

---

## `src/rete/__init__.py`

Add after the existing imports:

```python
from rete.prl import load_prl
```

Add `"load_prl"` to `__all__` in alphabetical order — between `"JoinTest"`
and `"LeftNode"`.

No other changes.  `rete.prl` uses only direct submodule imports
(`from rete.condition import …`, `from rete.fact import …`), so there is
no circular-import risk.

---

## `README.md` — PRL quick-start fix

The PRL quick-start block at line ~277 has three errors introduced before the
Python-naming switch:

### 1. Java type names in `declare` blocks

Current:
```
declare Temperature
  sensor: String
  value:  double
end

declare Alert
  message: String
end
```

Fixed:
```
declare Temperature
  sensor: str
  value:  float
end

declare Alert
  message: str
end
```

### 2. Wrong fact-access syntax in then-block

Current:
```
insert(Alert("Sensor " + t.sensor + " too hot"))
```

Fixed:
```
insert(Alert("Sensor " + t.obj.sensor + " too hot"))
```

`$t` in the then-block is a `Fact` wrapper; the underlying object is at
`.obj`.  Writing `t.sensor` raises `AttributeError` at runtime.

### 3. Import path for `load_prl`

Current (works but bypasses the public API):
```python
from rete.prl import load_prl
```

After Step 6, `load_prl` is re-exported from `rete`:
```python
from rete import Fact, InferenceEngine, load_prl
```

Update the quick-start snippet to use the top-level import.

---

## `tests/test_prl.py`

Add a single test at module level (or in a new `TestPublicApi` class) that
verifies `load_prl` is importable from the top-level `rete` package:

```python
class TestPublicApi:
    def test_load_prl_importable_from_rete(self) -> None:
        from rete import load_prl as _lp
        assert callable(_lp)
```

Complexity: 1 + 1(assert) = 2 ✓.

---

## Criticalities

### 1. No circular import

`rete/__init__.py` currently imports from `rete.alpha`, `rete.beta`,
`rete.condition`, `rete.engine`, `rete.fact`, `rete.network`.  Adding
`from rete.prl import load_prl` appends one more submodule import.

`rete.prl` in turn imports `from rete.condition import …` and
`from rete.fact import …`.  Because those submodules appear **before**
`rete.prl` in `__init__.py`, they are already in `sys.modules` by the time
`rete.prl` is imported.  No circular dependency.

The risk to watch: if `rete.prl` (or any transitive import) ever does
`from rete import …` (package-level, not submodule-level), it would hit the
partially-initialised `rete` package and may raise `ImportError`.  Verify
with `hatch run python -c "import rete; print(rete.load_prl)"` before
committing.

### 2. `__all__` ordering

`__all__` in `__init__.py` is alphabetically sorted.  `"load_prl"` sorts
between `"JoinTest"` and `"LeftNode"`.  Inserting it elsewhere would break
the ordering convention (checked by ruff rule `RUF022` if enabled, and
readable by humans).

### 3. `t.obj.sensor` vs `t.sensor` in README

The README quick-start is user-facing documentation.  An incorrect snippet
(`t.sensor`) will mislead users into writing broken rules.  The fix is
mandatory, not cosmetic.

### 4. `insert(Alert("Sensor " + t.obj.sensor + " too hot"))` — Fact wrapper vs POJO

The `insert` helper takes a **POJO** (not a `Fact`): `insert(Alert("…"))`
wraps the `Alert` object in `Fact` internally (see `_engine_helpers`).
The `t` variable (from `$t: /Temperature[…]`) is a `Fact` wrapper;
attribute access goes through `.obj`.  Both behaviours are correct and must
be shown consistently in the README.
