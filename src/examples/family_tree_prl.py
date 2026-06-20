"""Transitive ancestor inference loaded from PRL.

Rules: direct parent → Ancestor; transitive closure via insert.
The ``known`` set is injected as dedup guard; PRL cannot express
set-membership natively.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "family_tree.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _make_facts(types: dict) -> list[Fact]:
    """Return the three Parent facts A→B, B→C, C→D."""
    Parent = types["Parent"]
    return [
        Fact(Parent(parent="A", child="B")),
        Fact(Parent(parent="B", child="C")),
        Fact(Parent(parent="C", child="D")),
    ]


def _run() -> set[tuple[str, str]]:
    """Return the full ancestor closure — used by tests."""
    known: set[tuple[str, str]] = set()
    engine, types = _setup({"known": known})
    for f in _make_facts(types):
        engine.add_fact(f)
    engine.run()
    return known


def main() -> None:
    """Run with printed output and embedded assertions."""
    known = _run()
    for ancestor, descendant in sorted(known):
        print(f"  {ancestor} → {descendant}")
    assert known == {
        ("A", "B"), ("A", "C"), ("A", "D"),
        ("B", "C"), ("B", "D"),
        ("C", "D"),
    }


if __name__ == "__main__":
    main()
