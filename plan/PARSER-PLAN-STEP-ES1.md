# PRL Extra Features — Step ES-1: Type Inheritance (`extends`)

## Branch

`step/ES1-extends` cut from `prl-extras`.

---

## Files touched

| File | Change |
|---|---|
| `src/rete/prl_lexer.py` | Add `"extends"` to `KEYWORDS` set |
| `src/rete/prl_ast.py` | Add `extends: str \| None = None` field to `DeclareDecl` |
| `src/rete/prl_parser.py` | After parsing declare name, peek for `KW("extends")` and consume the following `IDENT` as the parent type name |
| `src/rete/prl.py` | `compile_declare`: topological sort of `DeclareDecl` nodes; pass `bases=(parent_type,)` to `make_dataclass` for child types |
| `src/rete/alpha.py` | Verify (and fix if needed) that alpha dispatch uses `isinstance(fact.obj, type_)` — not `type(fact.obj) is type_` |
| `tests/test_prl_parser.py` | New cases: `extends` parsed; missing parent name raises; `extends` in child `DeclareDecl` sets field correctly |
| `tests/test_prl_compiler.py` | `issubclass(Dog, Animal)` after compile; child-type fields accessible; circular inheritance raises `TypeError`; out-of-order declaration (child before parent) works |
| `tests/test_prl.py` | Integration: `Animal()` pattern fires for `Dog` facts; `Dog()` pattern does not fire for `Cat` facts |

---

## Objects modified

### `DeclareDecl` (`prl_ast.py`)

Add one optional field:

```python
@dataclass(frozen=True)
class DeclareDecl:
    name: str
    fields: tuple[FieldDecl, ...]
    extends: str | None = None      # new
```

No visitor or traversal code added — the compiler walks `DeclareDecl.extends` directly.

### `compile_declare` / `load_prl` (`prl.py`)

- Before iterating `program.declares`, sort them via a topological sort
  (Kahn's algorithm or `graphlib.TopologicalSorter` from stdlib) so parent
  types are always compiled before child types.
- When `decl.extends` is set, resolve the parent from `types`; raise
  `NameError` if absent.
- Pass `bases=(parent_type,)` to `make_dataclass`.
- Detect cycles in the inheritance graph before sorting; raise `TypeError`
  with the cycle listed.

`graphlib.TopologicalSorter` (Python 3.9+) is stdlib — use it directly.

### `AlphaMemory` (`alpha.py`)

Currently dispatches via `type(fact.obj)`.  Must dispatch via `isinstance`
so a `Dog` fact activates both the `Dog` and `Animal` alpha memories.
This may be a one-line change in `RootNode.activate` or `AlphaMemory.activate`
— verify before claiming the engine is unchanged; fix in this step if needed.

---

## Tests

### `tests/test_prl_parser.py` — new cases

```
parse("declare Dog extends Animal\n    name: String\nend")
  → DeclareDecl(name="Dog", fields=(...), extends="Animal")

parse("declare Dog extends\nend")
  → ParseError (missing parent name)
```

### `tests/test_prl_compiler.py` — new cases

```
compile: Animal + Dog extends Animal
  → issubclass(Dog, Animal) is True
  → Dog has both Animal's fields and its own

compile: circular (A extends B, B extends A)
  → TypeError

compile: child declared before parent in file
  → compiles without error (topo sort handles ordering)
```

### `tests/test_prl.py` — new cases

```python
# Pattern(Animal) fires for Dog fact
# Pattern(Dog) does NOT fire for Cat fact
```

---

## Criticalities

### 1. `isinstance` vs exact-type alpha dispatch

The engine's `RootNode` or `AlphaMemory` may index alpha memories by exact
type.  If so, inserting a `Dog` fact will never activate the `Animal` alpha
memory, silently breaking parent-type patterns.

**Check first:** grep `alpha.py` for `type(` and `isinstance`.  If the
dispatch is `type(fact.obj) == type_` or `type(fact.obj) is type_`, change
it to `isinstance(fact.obj, type_)`.  This is the most likely silent failure
mode for this entire step.

### 2. Topological sort — out-of-order declarations

A `.prl` file may declare `Dog extends Animal` before `Animal`.  The compiler
must sort before calling `make_dataclass`, not rely on file order.

Use `graphlib.TopologicalSorter`:

```python
from graphlib import TopologicalSorter, CycleError

ts = TopologicalSorter()
for decl in program.declares:
    ts.add(decl.name, decl.extends) if decl.extends else ts.add(decl.name)
try:
    order = list(ts.static_order())
except CycleError as e:
    raise TypeError(f"Circular inheritance: {e}") from e
```

`CycleError` maps cleanly to `TypeError` with a meaningful message.

### 3. Dataclass field ordering in the MRO

Python dataclasses require that inherited fields with defaults do not precede
fields without defaults.  For example:

```python
@dataclass
class Animal:
    name: str          # no default

@dataclass
class Dog(Animal):
    breed: str         # no default — OK
```

But:

```python
@dataclass
class Animal:
    name: str = ""     # has default

@dataclass
class Dog(Animal):
    breed: str         # no default — TypeError from Python
```

The compiler must validate the merged field list (parent fields + child
fields in MRO order) **before** calling `make_dataclass` and raise a clear
`SyntaxError` if the ordering is violated.  This avoids a confusing
`TypeError` from deep inside `dataclasses`.

### 4. `make_dataclass` with `bases`

`make_dataclass(name, fields, bases=(parent,))` only lists the **child's own
fields** in `fields`; parent fields are inherited.  Do not pass the full
merged list — that would duplicate fields and likely trigger a `TypeError`
from `dataclasses`.

### 5. `issubclass` in tests vs `is` comparisons in the engine

After ES-1, two code paths may compare types:

- Pattern matching: `isinstance(fact.obj, type_)` — correct after the fix
  in criticality 1.
- `NegativeJoinNode` / `BetaMemory` bookkeeping: these iterate over stored
  `Fact` objects by identity (`id(fact)`), not by type — unaffected by
  inheritance.

No other engine components do type comparisons.  Confirm by grepping
`beta.py` and `network.py` for `type(`.
