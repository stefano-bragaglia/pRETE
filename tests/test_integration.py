"""Integration tests: full Rete stack from Fact assertion through rule firing.

:see: Forgy §1.1, Doorenbos §2.5, §2.7, §2.8
"""
from dataclasses import dataclass

from rete.condition import JoinSpec, NccGroup, Pattern, Production
from rete.engine import InferenceEngine
from rete.fact import Fact, Token


# ---------------------------------------------------------------------------
# Test dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Color:
    block: str
    color: str


@dataclass
class Size:
    block: str
    size: str


@dataclass
class Marker:
    """Generic presence marker: (block, key) pair signals a named property."""

    block: str
    key: str


# ---------------------------------------------------------------------------
# Module-level alpha-test functions (stable ids for alpha-memory sharing)
# ---------------------------------------------------------------------------


def _is_red(obj: Color) -> bool:
    return obj.color == "red"


def _is_large(obj: Size) -> bool:
    return obj.size == "large"


def _is_broken(obj: Marker) -> bool:
    return obj.key == "broken"


def _marker_a(obj: Marker) -> bool:
    return obj.key == "a"


def _marker_b(obj: Marker) -> bool:
    return obj.key == "b"


def _marker_c(obj: Marker) -> bool:
    return obj.key == "c"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine() -> InferenceEngine:
    return InferenceEngine()


# ---------------------------------------------------------------------------
# Forward chaining
# ---------------------------------------------------------------------------


def test_simple_forward_chain():
    """A→B→C: asserting fact A eventually fires rule C via B."""
    engine = _engine()
    log = []

    def rhs_a(t: Token) -> None:
        engine.add_fact(Fact(Marker("x", "b")))

    def rhs_b(t: Token) -> None:
        engine.add_fact(Fact(Marker("x", "c")))

    engine.add_production(Production(
        lhs=[Pattern(Marker, alpha_tests=(_marker_a,))],
        rhs=rhs_a,
    ))
    engine.add_production(Production(
        lhs=[Pattern(Marker, alpha_tests=(_marker_b,))],
        rhs=rhs_b,
    ))
    engine.add_production(Production(
        lhs=[Pattern(Marker, alpha_tests=(_marker_c,))],
        rhs=lambda t: log.append("C"),
    ))
    engine.add_fact(Fact(Marker("x", "a")))
    steps = engine.run()
    assert steps == 3
    assert log == ["C"]


def test_retraction_cancels_match():
    """Retracting a Fact removes its instantiation before run()."""
    engine = _engine()
    fired = []
    engine.add_production(Production(
        lhs=[Pattern(Color, alpha_tests=(_is_red,))],
        rhs=lambda t: fired.append(1),
    ))
    fact = Fact(Color("b1", "red"))
    engine.add_fact(fact)
    engine.remove_fact(fact)
    engine.run()
    assert fired == []


def test_retraction_during_run():
    """RHS removes the triggering Fact; the instantiation is already gone before
    retraction, so no double-retract error."""
    engine = _engine()
    fact = Fact(Color("b1", "red"))

    def rhs(t: Token) -> None:
        engine.remove_fact(fact)

    engine.add_production(Production(
        lhs=[Pattern(Color, alpha_tests=(_is_red,))],
        rhs=rhs,
    ))
    engine.add_fact(fact)
    steps = engine.run()
    assert steps == 1
    assert engine.network.conflict_set == []


def test_variable_binding_across_conditions():
    """Two-condition rule: $block matches same block field in both facts."""
    engine = _engine()
    matched = []
    p0 = Pattern(Color, alpha_tests=(_is_red,), bindings=(("$block", "block"),))
    spec = JoinSpec("block", "$block")
    p1 = Pattern(Size, alpha_tests=(_is_large,), join_tests=(spec,))
    engine.add_production(Production(lhs=[p0, p1], rhs=lambda t: matched.append(t)))
    engine.add_fact(Fact(Color("b1", "red")))
    engine.add_fact(Fact(Size("b1", "large")))
    engine.add_fact(Fact(Color("b2", "red")))  # b2 has no matching Size → no match
    engine.run()
    assert len(matched) == 1
    assert matched[0].bindings["$block"] == "b1"


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------


def test_negation_integration():
    """Negative condition: rule fires only when the blocking Fact is absent."""
    engine = _engine()
    fired = []
    engine.add_production(Production(
        lhs=[
            Pattern(Color, alpha_tests=(_is_red,)),
            Pattern(Marker, alpha_tests=(_is_broken,), negated=True),
        ],
        rhs=lambda t: fired.append(1),
    ))
    engine.add_fact(Fact(Color("b1", "red")))
    engine.run()
    assert fired == [1]


def test_negation_blocked_by_wme():
    """Negative condition: rule does not fire when the blocking Fact is present."""
    engine = _engine()
    fired = []
    engine.add_production(Production(
        lhs=[
            Pattern(Color, alpha_tests=(_is_red,)),
            Pattern(Marker, alpha_tests=(_is_broken,), negated=True),
        ],
        rhs=lambda t: fired.append(1),
    ))
    engine.add_fact(Fact(Color("b1", "red")))
    engine.add_fact(Fact(Marker("b1", "broken")))
    engine.run()
    assert fired == []


# ---------------------------------------------------------------------------
# NCC (conjunctive negation)
# ---------------------------------------------------------------------------


def test_ncc_integration():
    """NCC rule fires only when the negated subnetwork has no match."""
    engine = _engine()
    fired = []
    engine.add_production(Production(
        lhs=[
            Pattern(Color, alpha_tests=(_is_red,)),
            NccGroup(conditions=(Pattern(Marker, alpha_tests=(_is_broken,)),)),
        ],
        rhs=lambda t: fired.append(1),
    ))
    engine.add_fact(Fact(Color("b1", "red")))
    engine.run()
    assert fired == [1]


def test_ncc_blocked_by_subnetwork_match():
    """NCC rule does not fire when the negated subnetwork matches."""
    engine = _engine()
    fired = []
    engine.add_production(Production(
        lhs=[
            Pattern(Color, alpha_tests=(_is_red,)),
            NccGroup(conditions=(Pattern(Marker, alpha_tests=(_is_broken,)),)),
        ],
        rhs=lambda t: fired.append(1),
    ))
    engine.add_fact(Fact(Color("b1", "red")))
    engine.add_fact(Fact(Marker("b1", "broken")))
    engine.run()
    assert fired == []


def test_ncc_retraction_round_trip():
    """Retracting the blocking Fact makes the NCC rule re-fire."""
    engine = _engine()
    fired = []
    engine.add_production(Production(
        lhs=[
            Pattern(Color, alpha_tests=(_is_red,)),
            NccGroup(conditions=(Pattern(Marker, alpha_tests=(_is_broken,)),)),
        ],
        rhs=lambda t: fired.append(1),
    ))
    engine.add_fact(Fact(Color("b1", "red")))
    blocker = Fact(Marker("b1", "broken"))
    engine.add_fact(blocker)
    engine.run()
    assert fired == []

    engine.remove_fact(blocker)
    engine.run()
    assert fired == [1]


# ---------------------------------------------------------------------------
# Multiple productions sharing nodes
# ---------------------------------------------------------------------------


def test_shared_alpha_memory():
    """Two productions sharing the same alpha memory both fire."""
    engine = _engine()
    log = []
    p = Pattern(Color, alpha_tests=(_is_red,))
    engine.add_production(Production(lhs=[p], rhs=lambda t: log.append("P1")))
    engine.add_production(Production(lhs=[p], rhs=lambda t: log.append("P2")))
    engine.add_fact(Fact(Color("b1", "red")))
    engine.run()
    assert sorted(log) == ["P1", "P2"]


# ---------------------------------------------------------------------------
# update_fact
# ---------------------------------------------------------------------------


def test_update_fact_integration():
    """End-to-end: mutate fact obj in place, update_fact re-evaluates the network."""
    engine = _engine()
    fired = []
    engine.add_production(Production(
        lhs=[Pattern(Color, alpha_tests=(_is_red,))],
        rhs=lambda t: fired.append(t.facts[-1].obj.block),
    ))
    fact = Fact(Color("b1", "blue"))
    engine.add_fact(fact)
    engine.run()
    assert fired == []

    fact.obj.color = "red"
    engine.update_fact(fact)
    engine.run()
    assert fired == ["b1"]


# ---------------------------------------------------------------------------
# Cross-fact binding — non-matching pair is silent
# ---------------------------------------------------------------------------


def test_cross_fact_binding_non_matching_pair_silent():
    """A Size for a different block does not satisfy the JoinSpec — no match."""
    engine = _engine()
    matched = []
    p0 = Pattern(Color, alpha_tests=(_is_red,), bindings=(("$block", "block"),))
    spec = JoinSpec("block", "$block")
    p1 = Pattern(Size, alpha_tests=(_is_large,), join_tests=(spec,))
    engine.add_production(Production(lhs=[p0, p1], rhs=lambda t: matched.append(t)))
    engine.add_fact(Fact(Color("b1", "red")))
    engine.add_fact(Fact(Size("b2", "large")))  # b2 ≠ b1 → no match
    engine.run()
    assert matched == []
