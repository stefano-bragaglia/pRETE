# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.5.3] — 2026-07-07

**Extending API change — `declare` field defaults and Python-bracket generics.**

### Added
- `declare` block fields may carry an optional default value
  (`field: type = value`), mirroring Python's own `@dataclass`: scalar
  literals (`None`/`null`, `str`, `int`, `float`, `bool`) and container
  literals (`[]`/`{}`, empty or with literal elements). Mutable defaults
  compile transparently to `dataclasses.field(default_factory=...)`, so
  every instance gets its own list/dict rather than one shared, aliased
  default. Field-ordering violations (a non-defaulted field following a
  defaulted one, including across `extends` and externally-supplied
  `types=` parents) surface as the same `TypeError`
  `dataclasses`/`make_dataclass` already raises natively — no separate
  validation was needed.
- `declare` block field types now use Python-bracket generics
  (`list[str]`, `dict[str, int]`, arbitrarily nested) in place of the old,
  erased Java-diamond form. The parsed parameter is preserved as the
  compiled field's real type annotation instead of being discarded.

### Changed
- `src/rete/prl_ast.py` — new `ContainerLiteral` node; `FieldDecl` gains
  `has_default` / `default`.
- `src/rete/prl_parser.py` — `_parse_type_ref` / new `_parse_type_params`
  replace the old diamond-erasing `_skip_generic`; `_parse_field` parses
  an optional `= value` clause via new `_parse_optional_default` /
  `_parse_default_value` / `_parse_list_literal` / `_parse_dict_literal`.
- `src/rete/prl.py` — `_java_type` resolves bracket-generic expressions
  recursively (new `_resolve_base_type`, `_split_generic`,
  `_split_top_level_commas`); `_compile_declare`'s field list now goes
  through new `_field_spec` / `_default_field` to honour a default.

### Removed
- Java-style diamond generics (`List<String>`) in `declare` blocks are no
  longer accepted — always accepted-but-erased since v2.1.0, so no
  runtime behavior anyone could depend on changes, only the accepted
  surface syntax. Existing `.prl` files using this form must be
  rewritten to bracket generics (`list[str]`).

### Fixed
- Removed a stale `omit = ["src/rete/drl.py"]` entry from
  `pyproject.toml`'s coverage config — that file (the regex-based DRL
  prototype superseded by `prl.py`) no longer exists anywhere in the
  tree.

### Notes
- Set/tuple literal defaults (`{1, 2}`, `(1, 2)`) are not supported — no
  concrete use case needed them, and `{}` is unambiguously an empty dict
  in this grammar, matching Python itself.
- Addresses [pRETE#1](https://github.com/stefano-bragaglia/pRETE/issues/1).

---

## [2.5.2] — 2026-07-07

**Documentation.**

### Added
- Link to [prl-highlight](https://github.com/stefano-bragaglia/prl-highlight),
  the PyCharm `.prl` syntax-highlighting plugin, in the PRL section of the README.

### Changed
- README logo swapped for the wide variant.
- `pyproject.toml` classifiers extended to cover Python 3.14.

---

## [2.5.1] — 2026-06-21

**Test coverage.**

### Changed
- Test suite expanded from 59 missed lines to 11 (99.17% coverage); remaining
  11 misses are intentional (error-raising guards and structural dead code).

---

## [2.5.0] — 2026-06-21

**Extending API change — PRL extra features.**

### Added
- `extends` keyword in `declare` blocks — type inheritance via Python MRO;
  parent-type patterns fire for child-type facts via `isinstance`.
- `@key` field tag — generates identity-aware `__eq__` / `__hash__` on declared
  types using only the annotated key fields.
- Positional and named constraint shorthand — `TypeName(v1, v2)` and
  `TypeName(field=v)` in patterns, resolved against declaration order at
  compile time.
- `@no-loop` rule tag — alias for the existing `no-loop: true` attribute;
  prevents a rule from re-activating on its own WM modifications.
- `import` / `from … import` / `from … import … as` — self-contained `.prl`
  files that declare their own type dependencies; `types` dict in `load_prl`
  becomes optional.
- `or` disjunction — single PRL rule with K LHS branches compiles to K
  `Production` objects sharing the same RHS closure.
- `forall(P, Q)` — universal quantification, rewritten by the compiler to
  `NccGroup([P, negated_Q])`.
- `exists Pattern(…)` — new `ExistsNode` in `beta.py`; fires once per
  matching left context regardless of right-fact count.
- `@role(event)`, `@timestamp`, `@duration`, `@expires` — synchronous
  logical-clock CEP model; events are auto-retracted when
  `timestamp + expires < logical_clock`.
- `accumulate(inner; $result: fn; constraint)` — new `AccumulateNode` in
  `beta.py` with incremental per-left-token state for `sum`, `count`, `min`,
  `max`, `collectList`.
- New example `.prl` files and companion Python drivers for all extra features
  (`src/examples/prl/inheritance.prl`, `identity_key.prl`,
  `compact_patterns.prl`, `self_modify.prl`, `imported_types.prl`,
  `disjunction.prl`, `universal.prl`, `existence_check.prl`,
  `event_stream.prl`, `aggregation.prl`).

### Changed
- `src/rete/prl_ast.py` — `DeclareDecl` gains `extends` and `tags` fields;
  `FieldDecl` and `RuleDecl` gain `tags`; new AST nodes: `Tag`, `ImportDecl`,
  `OrGroup`, `ForallNode`, `AccumulateNode` (AST).
- `src/rete/prl_lexer.py` — new keywords: `extends`, `import`, `from`, `as`,
  `or`, `forall`, `exists`, `accumulate`; new `AT` token type.
- `src/rete/prl_parser.py` — tag parsing, `extends`, all import forms,
  `or` / `forall` / `exists` / `accumulate` conditions.
- `src/rete/prl.py` — compiler handles all new AST nodes; topological sort
  for `extends`; `__prl_meta__` dict on generated types for tag semantics.
- `src/rete/beta.py` — `ExistsNode`, `AccumulateNode`.
- `src/rete/condition.py` — `Pattern.exists`, `AccumulateSpec`.
- `src/rete/network.py` — wires `ExistsNode` / `AccumulateNode`.
- `src/rete/fact.py` — `Fact.timestamp` field for CEP.
- `src/rete/engine.py` — `logical_clock`, `advance_clock()`, `_expire_events()`.

---

## [2.1.0] — 2026-06-20

**Extending API change — pRETE Rule Language (PRL) parser added.**

### Added
- `rete.prl.load_prl(text, types, engine)` — parses PRL source into
  `(dict[str, type], list[Production])` ready for the RETE engine.
- `src/rete/prl_lexer.py` — tokenizer for PRL source text.
- `src/rete/prl_ast.py` — frozen AST dataclasses (`FieldDecl`, `DeclareDecl`,
  `BindConstraint`, `CompareConstraint`, `PatternNode`, `NccPatternGroup`,
  `RuleDecl`, `ProgramNode`).
- `src/rete/prl_parser.py` — hand-written recursive-descent parser.
- `src/rete/prl.py` — compiler (AST → `Pattern` / `NccGroup` / `Production`)
  and `load_prl()` entry point.
- `.prl` program files for all seven bundled examples
  (`src/examples/prl/*.prl`).
- PRL-powered companion drivers for all seven examples
  (`src/examples/*_prl.py`).
- Per-step unit tests: `test_prl_lexer.py`, `test_prl_ast.py`,
  `test_prl_parser.py`, `test_prl_compiler.py`.
- Full-pipeline integration tests: `test_prl.py`.

### Changed
- `src/rete/__init__.py` — `load_prl` added to public re-exports and `__all__`.

### Notes
- The engine, network, alpha, and beta modules are **unchanged**.
- PRL is a strict subset of Drools Rule Language (DRL 8.x); see
  `reference/prl-grammar.ebnf` for the full grammar.
- `salience` is parsed and preserved on `RuleDecl` (default 0) but not yet
  wired into `InferenceEngine`'s conflict-set strategy.
- The regex-based `drl.py` prototype is superseded by `prl.py` and will be
  removed in a future release.

---

## [2.0.0] — 2026-06-19

**Breaking API change — triple model replaced by Plain Old Python Objects (POPOs).**

### Added
- `Fact(obj)` — wraps any Python object as a working-memory element.
- `Pattern(type_, alpha_tests, join_tests, bindings, negated)` — replaces
  `Condition`; matches by type then by callable field tests.
- `JoinSpec(attr_of_fact, var_name)` — compile-time cross-fact variable
  reference declared in `Pattern.join_tests`.
- `Token.bindings: dict[str, Any]` — named variable bindings carried by
  each partial match (replaces positional `token.wmes[i]` access).
- `InferenceEngine.update_fact(fact)` — Drools-style `modify`: retracts
  and re-asserts a mutated fact so the network stays in sync.
- New examples: `loan_application.py`, `temperature_alarm.py`,
  `family_tree.py`, `fraud_detection.py`.

### Removed
- `WME(id, attribute, value)` — replaced by `Fact(obj)`.
- `Condition(id_test, attribute_test, value_test)` — replaced by `Pattern`.
- `WILDCARD` constant — no longer needed; omit the field from `alpha_tests`.
- `add_wme` / `remove_wme` on `ReteNetwork` and `InferenceEngine` —
  replaced by `add_fact` / `remove_fact`.
- `AlphaNode` trie — replaced by a single predicate per `AlphaMemory` and
  type-indexed dispatch in `RootNode`.

### Changed
- `Token.wmes` → `Token.facts`.
- Alpha network now dispatches by MRO, so a `Dog` fact activates a
  `Pattern(type_=Animal)`.
- Alpha memory sharing is keyed by `(type_, tuple(id(fn) for fn in alpha_tests))`;
  stable function references (not inline lambdas) are required for sharing.

---

## [1.0.1] — prior

Incremental fixes to the v1.0.0 triple-based engine.

## [1.0.0] — prior

Initial release: triple `(id, attribute, value)` WME model, full Doorenbos
algorithm with right/left unlinking and NCC support.
