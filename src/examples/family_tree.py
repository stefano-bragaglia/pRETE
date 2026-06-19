"""Transitive ancestor inference via forward chaining.

Rules
-----
* *Direct*:     Parent(parent=$p, child=$c)
                → assert Ancestor($p, $c) if not already known.
* *Transitive*: Parent(parent=$p, child=$c) ∧ Ancestor(ancestor=$c, descendant=$d)
                → assert Ancestor($p, $d) if not already known.

The ``known`` set is the mandatory dedup guard: without it the transitive
rule loops forever, producing duplicate Ancestor facts on every cycle.

Given Parent(A,B), Parent(B,C), Parent(C,D) the expected closure is the
six pairs: (A,B), (A,C), (A,D), (B,C), (B,D), (C,D).
"""
from __future__ import annotations

from dataclasses import dataclass

from rete.condition import JoinSpec, Pattern, Production
from rete.engine import InferenceEngine
from rete.fact import Fact


@dataclass
class Parent:
    """Direct parent-child relationship."""

    parent: str
    child: str


@dataclass
class Ancestor:
    """Derived ancestor-descendant relationship."""

    ancestor: str
    descendant: str


def build_engine(known: set[tuple[str, str]]) -> InferenceEngine:
    """Return an engine with the two ancestor rules.

    :param known: mutable set shared between the engine and the caller;
                  tracks ``(ancestor, descendant)`` pairs already asserted.
    """
    engine = InferenceEngine()

    def _add_ancestor(p: str, d: str) -> None:
        if (p, d) not in known:
            known.add((p, d))
            engine.add_fact(Fact(Ancestor(p, d)))

    # Rule 1 — Direct ancestor
    engine.add_production(Production(
        lhs=[Pattern(Parent, bindings=(("$p", "parent"), ("$c", "child")))],
        rhs=lambda t: _add_ancestor(t.bindings["$p"], t.bindings["$c"]),
    ))

    # Rule 2 — Transitive ancestor
    engine.add_production(Production(
        lhs=[
            Pattern(Parent, bindings=(("$p", "parent"), ("$c", "child"))),
            Pattern(Ancestor, join_tests=(JoinSpec("ancestor", "$c"),),
                    bindings=(("$d", "descendant"),)),
        ],
        rhs=lambda t: _add_ancestor(t.bindings["$p"], t.bindings["$d"]),
    ))

    return engine


def main() -> None:
    """Run the family-tree example and print the derived ancestor closure."""
    known: set[tuple[str, str]] = set()
    engine = build_engine(known)

    parents = [
        Fact(Parent("A", "B")),
        Fact(Parent("B", "C")),
        Fact(Parent("C", "D")),
    ]

    print("Asserting parent facts:")
    for f in parents:
        print(f"  {f.obj}")
        engine.add_fact(f)

    engine.run()

    print("\nDerived ancestors:")
    for ancestor, descendant in sorted(known):
        print(f"  {ancestor} → {descendant}")

    assert known == {
        ("A", "B"), ("A", "C"), ("A", "D"),
        ("B", "C"), ("B", "D"),
        ("C", "D"),
    }


if __name__ == "__main__":
    main()
