"""Doorenbos §2.1 — canonical blocks-world example (Figure 2.2b).

Production:  find-stack-of-two-blocks-to-the-left-of-a-red-block
  C1: (?x ^on ?y)
  C2: (?y ^left-of ?z)
  C3: (?z ^color red)

Working memory w1–w9 from the thesis.
Expected single match: w1 ^ w5 ^ w9  →  x=B1, y=B2, z=B3
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
class LeftOf:
    """Spatial relation: *left* is to the left of *right*."""

    left: str
    right: str


@dataclass
class Color:
    """Colour property of a block."""

    block: str
    color: str


def _is_red(obj: Color) -> bool:
    """Alpha test: block is red."""
    return obj.color == "red"


def _make_facts() -> list[Fact]:
    """Return working-memory facts w1–w9 (Doorenbos §2.1)."""
    return [
        Fact(On("B1", "B2")),      # w1
        Fact(On("B1", "B3")),      # w2
        Fact(Color("B1", "red")),  # w3
        Fact(On("B2", "table")),   # w4
        Fact(LeftOf("B2", "B3")),  # w5
        Fact(Color("B2", "blue")), # w6
        Fact(LeftOf("B3", "B4")), # w7
        Fact(On("B3", "table")),   # w8
        Fact(Color("B3", "red")),  # w9
    ]


def _build_network() -> ReteNetwork:
    """Compile the blocks-world production into a fresh network."""
    net = ReteNetwork()
    net.add_production(Production(
        lhs=[
            Pattern(On, bindings=(("$x", "upper"), ("$y", "lower"))),
            Pattern(LeftOf, join_tests=(JoinSpec("left", "$y"),),
                    bindings=(("$z", "right"),)),
            Pattern(Color, alpha_tests=(_is_red,),
                    join_tests=(JoinSpec("block", "$z"),)),
        ],
        rhs=lambda t: None,
    ))
    return net


def _run() -> list[tuple[str, str, str]]:
    """Return ``[(x, y, z)]`` for each match — used by tests."""
    net = _build_network()
    for f in _make_facts():
        net.add_fact(f)
    return [
        (i.token.bindings["$x"], i.token.bindings["$y"], i.token.bindings["$z"])
        for i in net.conflict_set
    ]


def main() -> None:
    """Run the example with printed output and embedded assertions."""
    net = _build_network()
    facts = _make_facts()

    print("Adding facts w1–w9:")
    for i, f in enumerate(facts, 1):
        print(f"  w{i}: {f.obj}")
        net.add_fact(f)

    print(f"\nConflict set: {len(net.conflict_set)} instantiation(s)")
    for inst in net.conflict_set:
        b = inst.token.bindings
        print(f"  fired: x={b['$x']}, y={b['$y']}, z={b['$z']}")

    assert len(net.conflict_set) == 1
    b = net.conflict_set[0].token.bindings
    assert (b["$x"], b["$y"], b["$z"]) == ("B1", "B2", "B3")


if __name__ == "__main__":
    main()
