"""Doorenbos §2.7 — negative conditions (NegativeJoinNode).

Production:  find-blocks-on-table-not-blue
  C1:  (?x ^on table)         ← positive: x is on the table
  -C2: (?x ^color blue)       ← negated:  x must NOT be blue

Working memory w1–w9 from the thesis.
  B2 is on table (w4) AND is blue (w6) → excluded by -C2
  B3 is on table (w8) AND is red  (w9) → passes -C2 → fires

Expected single match: w8  →  x=B3
"""
from __future__ import annotations

from dataclasses import dataclass

from rete import Fact, Pattern, Production, ReteNetwork
from rete.condition import JoinSpec


@dataclass
class On:
    """Spatial relation: *upper* rests on *lower*."""

    upper: str
    lower: str


@dataclass
class Color:
    """Colour property of a block."""

    block: str
    color: str


def _on_table(obj: On) -> bool:
    """Alpha test: block is on the table."""
    return obj.lower == "table"


def _is_blue(obj: Color) -> bool:
    """Alpha test: block is blue."""
    return obj.color == "blue"


def _make_facts() -> list[Fact]:
    """Return working-memory facts w1–w9 (Doorenbos §2.1)."""
    return [
        Fact(On("B1", "B2")),      # w1
        Fact(On("B1", "B3")),      # w2
        Fact(Color("B1", "red")),  # w3
        Fact(On("B2", "table")),   # w4
        Fact(Color("B2", "blue")), # w5  (was w6 in blocks_world; reordered for clarity)
        Fact(Color("B3", "red")),  # w6
        Fact(On("B3", "table")),   # w7
    ]


def _build_network() -> ReteNetwork:
    """Compile the negation production into a fresh network."""
    net = ReteNetwork()
    net.add_production(Production(
        lhs=[
            Pattern(On, alpha_tests=(_on_table,), bindings=(("$x", "upper"),)),
            Pattern(Color, alpha_tests=(_is_blue,),
                    join_tests=(JoinSpec("block", "$x"),),
                    negated=True),
        ],
        rhs=lambda t: None,
    ))
    return net


def _run() -> list[str]:
    """Return ``["B3"]`` for the matching block — used by tests."""
    net = _build_network()
    for f in _make_facts():
        net.add_fact(f)
    return [i.token.bindings["$x"] for i in net.conflict_set]


def main() -> None:
    """Run the example with printed output and embedded assertions."""
    net = _build_network()
    facts = _make_facts()

    print("Adding facts:")
    for f in facts:
        print(f"  {f.obj}")
        net.add_fact(f)

    print(f"\nConflict set: {len(net.conflict_set)} instantiation(s)")
    for inst in net.conflict_set:
        x = inst.token.bindings["$x"]
        print(f"  fired: x={x} is on the table and not blue")

    assert len(net.conflict_set) == 1
    assert net.conflict_set[0].token.bindings["$x"] == "B3"


if __name__ == "__main__":
    main()
