"""Unit tests for engine.py — InferenceEngine.

:see: Forgy §1.1
"""
from rete.beta import Instantiation, PNode
from rete.condition import Condition, Production
from rete.engine import InferenceEngine
from rete.wme import Token, WME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prod(lhs: list[Condition], rhs=None) -> Production:
    return Production(lhs=lhs, rhs=rhs or (lambda t: None))


# ---------------------------------------------------------------------------
# run() — basic behaviour
# ---------------------------------------------------------------------------


def test_run_empty_conflict_set():
    engine = InferenceEngine()
    assert engine.run() == 0


def test_run_fires_single_instantiation():
    fired = []
    engine = InferenceEngine()
    engine.add_production(Production(
        lhs=[Condition("b1", "color", "red")],
        rhs=lambda t: fired.append(t),
    ))
    engine.add_wme(WME("b1", "color", "red"))
    steps = engine.run()
    assert steps == 1
    assert len(fired) == 1
    assert engine.network.conflict_set == []


def test_run_forward_chaining():
    """Rule A's RHS adds a WME that triggers rule B."""
    engine = InferenceEngine()
    log = []

    def rhs_a(token: Token) -> None:
        engine.add_wme(WME("b2", "size", "large"))

    engine.add_production(Production(
        lhs=[Condition("b1", "color", "red")],
        rhs=rhs_a,
    ))
    engine.add_production(Production(
        lhs=[Condition("b2", "size", "large")],
        rhs=lambda t: log.append("B"),
    ))
    engine.add_wme(WME("b1", "color", "red"))
    steps = engine.run()
    assert steps == 2
    assert log == ["B"]
    assert engine.network.conflict_set == []


def test_run_max_steps():
    """max_steps stops the loop early."""
    engine = InferenceEngine()
    log = []

    def rhs(t: Token) -> None:
        log.append(1)
        engine.add_wme(WME("x", "y", str(len(log))))  # keeps adding new matches

    engine.add_production(Production(lhs=[Condition("x", "y", "?v")], rhs=rhs))
    engine.add_wme(WME("x", "y", "seed"))
    steps = engine.run(max_steps=3)
    assert steps == 3


def test_run_returns_step_count():
    engine = InferenceEngine()
    fired = []
    engine.add_production(Production(
        lhs=[Condition("a", "b", "c")],
        rhs=lambda t: fired.append(1),
    ))
    engine.add_wme(WME("a", "b", "c"))
    engine.add_wme(WME("a", "b", "c"))  # second WME, second match on same condition
    # Two distinct WMEs both match → two instantiations
    assert engine.run() == 2


def test_run_does_not_refire_same_instantiation():
    """Fired instantiation is removed before RHS, so it cannot be re-selected."""
    engine = InferenceEngine()
    fired = []
    engine.add_production(Production(
        lhs=[Condition("b1", "color", "red")],
        rhs=lambda t: fired.append(1),
    ))
    engine.add_wme(WME("b1", "color", "red"))
    engine.run()
    assert len(fired) == 1  # fired exactly once


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------


def test_fifo_strategy():
    cs = [
        Instantiation(production=_prod([]), token=Token()),
        Instantiation(production=_prod([]), token=Token()),
    ]
    first = cs[0]
    assert InferenceEngine.fifo_strategy(cs) is first


def test_recency_strategy():
    cs = [
        Instantiation(production=_prod([]), token=Token()),
        Instantiation(production=_prod([]), token=Token()),
    ]
    last = cs[-1]
    assert InferenceEngine.recency_strategy(cs) is last


def test_custom_strategy():
    """Engine accepts any callable as strategy."""
    chosen = Instantiation(production=_prod([]), token=Token())
    engine = InferenceEngine(strategy=lambda cs: chosen)
    # We can set strategy; engine wraps it without error.
    assert engine.strategy([chosen]) is chosen


# ---------------------------------------------------------------------------
# Passthrough delegates
# ---------------------------------------------------------------------------


def test_delegate_add_wme():
    engine = InferenceEngine()
    wme = WME("a", "b", "c")
    engine.add_wme(wme)
    # No production → nothing to assert, but no error either.


def test_delegate_remove_wme():
    engine = InferenceEngine()
    wme = WME("a", "b", "c")
    engine.add_wme(wme)
    engine.remove_wme(wme)


def test_delegate_add_production_returns_pnode():
    engine = InferenceEngine()
    pn = engine.add_production(_prod([Condition("a", "b", "c")]))
    assert isinstance(pn, PNode)


def test_delegate_remove_production():
    engine = InferenceEngine()
    pn = engine.add_production(_prod([Condition("a", "b", "c")]))
    engine.add_wme(WME("a", "b", "c"))
    engine.remove_production(pn)
    assert engine.network.conflict_set == []


# ---------------------------------------------------------------------------
# Instantiation.__eq__ correctness for conflict_set.remove
# ---------------------------------------------------------------------------


def test_instantiation_equality_distinct_tokens():
    """Two Instantiations with identical WME tuples compare equal — verify remove
    picks the right one and does not raise."""
    wme = WME("x", "y", "z")
    p = _prod([])
    t1 = Token(wmes=(wme,))
    t2 = Token(wmes=(wme,))  # distinct object, same contents
    i1 = Instantiation(production=p, token=t1)
    i2 = Instantiation(production=p, token=t2)
    cs = [i1, i2]
    cs.remove(i1)
    assert i2 in cs
