"""Doorenbos §2.3 — node-sharing between two productions.

Two productions with identical C1^C2 share one beta memory node:

  P1 (find-stack-of-two-blocks-to-the-left-of-a-red-block):
    C1: (?x ^on ?y)
    C2: (?y ^left-of ?z)
    C3: (?z ^color red)          ← tests z

  P2 (slightly-modified-version):
    C1: (?x ^on ?y)              ← same as P1
    C2: (?y ^left-of ?z)         ← same as P1  → shared beta memory
    C3': (?y ^color red)         ← tests y, not z

Working memory w1–w9 from the thesis.
Expected:
  P1 fires for w1^w5^w9  →  x=B1, y=B2, z=B3
  P2 fires for w2^w7^w9  →  x=B1, y=B3, z=B4
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
        Fact(LeftOf("B2", "B3")), # w5
        Fact(Color("B2", "blue")), # w6
        Fact(LeftOf("B3", "B4")), # w7
        Fact(On("B3", "table")),   # w8
        Fact(Color("B3", "red")),  # w9
    ]


def _shared_c1() -> Pattern:
    """C1 pattern shared by P1 and P2."""
    return Pattern(On, bindings=(("$x", "upper"), ("$y", "lower")))


def _shared_c2() -> Pattern:
    """C2 pattern shared by P1 and P2."""
    return Pattern(
        LeftOf,
        join_tests=(JoinSpec("left", "$y"),),
        bindings=(("$z", "right"),),
    )


def _build_network() -> ReteNetwork:
    """Compile both productions; C1 and C2 join nodes are shared."""
    net = ReteNetwork()

    # P1: C3 tests z
    net.add_production(Production(
        lhs=[
            _shared_c1(),
            _shared_c2(),
            Pattern(Color, alpha_tests=(_is_red,),
                    join_tests=(JoinSpec("block", "$z"),)),
        ],
        rhs=lambda t: None,
    ))

    # P2: C3' tests y (not z) — different JoinSpec → separate join node
    net.add_production(Production(
        lhs=[
            _shared_c1(),
            _shared_c2(),
            Pattern(Color, alpha_tests=(_is_red,),
                    join_tests=(JoinSpec("block", "$y"),)),
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

    bindings = {
        (i.token.bindings["$x"], i.token.bindings["$y"], i.token.bindings["$z"])
        for i in net.conflict_set
    }
    assert bindings == {("B1", "B2", "B3"), ("B1", "B3", "B4")}


if __name__ == "__main__":
    main()
