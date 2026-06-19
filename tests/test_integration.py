"""Integration tests: full Rete stack from WME assertion through rule firing.

:see: Forgy §1.1, Doorenbos §2.5, §2.7, §2.8
"""
from rete.condition import Condition, NccGroup, Production
from rete.engine import InferenceEngine
from rete.fact import Token, WME


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
        engine.add_wme(WME("b", "exists", "true"))

    def rhs_b(t: Token) -> None:
        engine.add_wme(WME("c", "exists", "true"))

    engine.add_production(Production(lhs=[Condition("a", "exists", "true")], rhs=rhs_a))
    engine.add_production(Production(lhs=[Condition("b", "exists", "true")], rhs=rhs_b))
    engine.add_production(Production(
        lhs=[Condition("c", "exists", "true")],
        rhs=lambda t: log.append("C"),
    ))

    engine.add_wme(WME("a", "exists", "true"))
    steps = engine.run()
    assert steps == 3
    assert log == ["C"]


def test_retraction_cancels_match():
    """Retracting a WME removes its instantiation before run()."""
    engine = _engine()
    fired = []
    engine.add_production(Production(
        lhs=[Condition("b1", "color", "red")],
        rhs=lambda t: fired.append(1),
    ))
    wme = WME("b1", "color", "red")
    engine.add_wme(wme)
    engine.remove_wme(wme)
    engine.run()
    assert fired == []


def test_retraction_during_run():
    """RHS removes the triggering WME; the instantiation is already gone before
    retraction, so no double-retract error."""
    engine = _engine()

    wme = WME("b1", "color", "red")

    def rhs(t: Token) -> None:
        engine.remove_wme(wme)

    engine.add_production(Production(lhs=[Condition("b1", "color", "red")], rhs=rhs))
    engine.add_wme(wme)
    steps = engine.run()
    assert steps == 1
    assert engine.network.conflict_set == []


def test_variable_binding_across_conditions():
    """Two-condition rule: ?x matches same id in both WMEs."""
    engine = _engine()
    matched = []
    engine.add_production(Production(
        lhs=[
            Condition("?x", "color", "red"),
            Condition("?x", "size", "large"),
        ],
        rhs=lambda t: matched.append(t),
    ))
    engine.add_wme(WME("b1", "color", "red"))
    engine.add_wme(WME("b1", "size", "large"))
    engine.add_wme(WME("b2", "color", "red"))  # b2 has no size → should not match
    engine.run()
    assert len(matched) == 1
    assert matched[0].wmes[0].id == "b1"


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------


def test_negation_integration():
    """Negative condition: rule fires only when the blocking WME is absent."""
    engine = _engine()
    fired = []
    engine.add_production(Production(
        lhs=[
            Condition("b1", "color", "red"),
            Condition("b1", "broken", "true", negated=True),
        ],
        rhs=lambda t: fired.append(1),
    ))
    engine.add_wme(WME("b1", "color", "red"))
    engine.run()
    assert fired == [1]


def test_negation_blocked_by_wme():
    """Negative condition: rule does not fire when the blocking WME is present."""
    engine = _engine()
    fired = []
    engine.add_production(Production(
        lhs=[
            Condition("b1", "color", "red"),
            Condition("b1", "broken", "true", negated=True),
        ],
        rhs=lambda t: fired.append(1),
    ))
    engine.add_wme(WME("b1", "color", "red"))
    engine.add_wme(WME("b1", "broken", "true"))
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
            Condition("b1", "color", "red"),
            NccGroup(conditions=(Condition("b1", "broken", "true"),)),
        ],
        rhs=lambda t: fired.append(1),
    ))
    engine.add_wme(WME("b1", "color", "red"))
    engine.run()
    assert fired == [1]


def test_ncc_blocked_by_subnetwork_match():
    """NCC rule does not fire when the negated subnetwork matches."""
    engine = _engine()
    fired = []
    engine.add_production(Production(
        lhs=[
            Condition("b1", "color", "red"),
            NccGroup(conditions=(Condition("b1", "broken", "true"),)),
        ],
        rhs=lambda t: fired.append(1),
    ))
    engine.add_wme(WME("b1", "color", "red"))
    engine.add_wme(WME("b1", "broken", "true"))
    engine.run()
    assert fired == []


# ---------------------------------------------------------------------------
# Multiple productions sharing nodes
# ---------------------------------------------------------------------------


def test_shared_alpha_memory():
    """Two productions sharing the same alpha memory both fire."""
    engine = _engine()
    log = []
    cond = Condition("b1", "color", "red")
    engine.add_production(Production(lhs=[cond], rhs=lambda t: log.append("P1")))
    engine.add_production(Production(lhs=[cond], rhs=lambda t: log.append("P2")))
    engine.add_wme(WME("b1", "color", "red"))
    engine.run()
    assert sorted(log) == ["P1", "P2"]
