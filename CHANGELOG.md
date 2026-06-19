# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
