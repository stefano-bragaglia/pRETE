"""Unit tests for engine.py — InferenceEngine.

:see: Forgy §1.1
"""
from dataclasses import dataclass

from rete.beta import Instantiation, PNode
from rete.condition import Pattern, Production
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
class Counter:
    value: int


@dataclass
class LoanApp:
    applicant: str
    approved: bool = True


# ---------------------------------------------------------------------------
# Module-level alpha-test functions (stable ids for alpha-memory sharing)
# ---------------------------------------------------------------------------


def _is_red(obj: Color) -> bool:
    return obj.color == "red"


def _is_large(obj: Size) -> bool:
    return obj.size == "large"


def _is_approved(obj: LoanApp) -> bool:
    return obj.approved


def _is_denied(obj: LoanApp) -> bool:
    return not obj.approved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prod(lhs: list, rhs=None) -> Production:
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
        lhs=[Pattern(Color, alpha_tests=(_is_red,))],
        rhs=lambda t: fired.append(t),
    ))
    engine.add_fact(Fact(Color("b1", "red")))
    steps = engine.run()
    assert steps == 1
    assert len(fired) == 1
    assert engine.network.conflict_set == []


def test_run_forward_chaining():
    """Rule A's RHS adds a Fact that triggers rule B."""
    engine = InferenceEngine()
    log = []

    def rhs_a(token: Token) -> None:
        engine.add_fact(Fact(Size("b2", "large")))

    engine.add_production(Production(
        lhs=[Pattern(Color, alpha_tests=(_is_red,))],
        rhs=rhs_a,
    ))
    engine.add_production(Production(
        lhs=[Pattern(Size, alpha_tests=(_is_large,))],
        rhs=lambda t: log.append("B"),
    ))
    engine.add_fact(Fact(Color("b1", "red")))
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
        engine.add_fact(Fact(Counter(len(log))))  # each Fact is a distinct identity

    engine.add_production(Production(lhs=[Pattern(Counter)], rhs=rhs))
    engine.add_fact(Fact(Counter(0)))  # seed
    steps = engine.run(max_steps=3)
    assert steps == 3


def test_run_returns_step_count():
    engine = InferenceEngine()
    fired = []
    engine.add_production(Production(
        lhs=[Pattern(Color)],
        rhs=lambda t: fired.append(1),
    ))
    engine.add_fact(Fact(Color("a", "b")))
    engine.add_fact(Fact(Color("a", "b")))  # distinct identity → second match
    assert engine.run() == 2


def test_run_does_not_refire_same_instantiation():
    """Fired instantiation is removed before RHS, so it cannot be re-selected."""
    engine = InferenceEngine()
    fired = []
    engine.add_production(Production(
        lhs=[Pattern(Color, alpha_tests=(_is_red,))],
        rhs=lambda t: fired.append(1),
    ))
    engine.add_fact(Fact(Color("b1", "red")))
    engine.run()
    assert len(fired) == 1


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
    assert engine.strategy([chosen]) is chosen


# ---------------------------------------------------------------------------
# Passthrough delegates
# ---------------------------------------------------------------------------


def test_delegate_add_fact():
    engine = InferenceEngine()
    engine.add_fact(Fact(Color("a", "b")))  # no error


def test_delegate_remove_fact():
    engine = InferenceEngine()
    f = Fact(Color("a", "b"))
    engine.add_fact(f)
    engine.remove_fact(f)  # no error


def test_delegate_add_production_returns_pnode():
    engine = InferenceEngine()
    pn = engine.add_production(_prod([Pattern(Color)]))
    assert isinstance(pn, PNode)


def test_delegate_remove_production():
    engine = InferenceEngine()
    pn = engine.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    engine.add_fact(Fact(Color("a", "red")))
    engine.remove_production(pn)
    assert engine.network.conflict_set == []


# ---------------------------------------------------------------------------
# Instantiation.__eq__ correctness for conflict_set.remove
# ---------------------------------------------------------------------------


def test_instantiation_equality_distinct_tokens():
    """Two Instantiations with identical Fact tuples compare equal; remove picks
    the first and leaves the second."""
    fact = Fact(Color("x", "y"))
    p = _prod([])
    t1 = Token(facts=(fact,))
    t2 = Token(facts=(fact,))  # distinct Token object, same contents
    i1 = Instantiation(production=p, token=t1)
    i2 = Instantiation(production=p, token=t2)
    cs = [i1, i2]
    cs.remove(i1)
    assert i2 in cs


# ---------------------------------------------------------------------------
# update_fact
# ---------------------------------------------------------------------------


def test_update_fact_new_match_after_mutation():
    """Mutate obj so it now matches a rule; update_fact makes it visible."""
    engine = InferenceEngine()
    denied = []
    engine.add_production(Production(
        lhs=[Pattern(LoanApp, alpha_tests=(_is_denied,))],
        rhs=lambda t: denied.append(t.facts[-1].obj.applicant),
    ))
    app = Fact(LoanApp("Alice", True))
    engine.add_fact(app)
    engine.run()
    assert denied == []

    app.obj.approved = False
    engine.update_fact(app)
    engine.run()
    assert denied == ["Alice"]


def test_update_fact_stale_match_removed():
    """Mutate obj so it no longer matches; old match is gone after update_fact."""
    engine = InferenceEngine()
    approved = []
    engine.add_production(Production(
        lhs=[Pattern(LoanApp, alpha_tests=(_is_approved,))],
        rhs=lambda t: approved.append(t.facts[-1].obj.applicant),
    ))
    app = Fact(LoanApp("Bob", True))
    engine.add_fact(app)
    engine.run()
    assert approved == ["Bob"]

    app.obj.approved = False
    engine.update_fact(app)
    engine.run()
    assert len(approved) == 1  # no new firing after mutation


def test_update_fact_called_from_rhs():
    """update_fact inside RHS: approved rule fires once, then denied rule fires
    once; no double-fire, no missing fire, no infinite loop."""
    engine = InferenceEngine()
    fired = []

    def approve_to_deny(token: Token) -> None:
        fact = token.facts[-1]
        fact.obj.approved = False
        engine.update_fact(fact)

    engine.add_production(Production(
        lhs=[Pattern(LoanApp, alpha_tests=(_is_approved,))],
        rhs=approve_to_deny,
    ))
    engine.add_production(Production(
        lhs=[Pattern(LoanApp, alpha_tests=(_is_denied,))],
        rhs=lambda t: fired.append(1),
    ))
    app = Fact(LoanApp("Carol", True))
    engine.add_fact(app)
    engine.run(max_steps=10)
    assert len(fired) == 1


def test_update_fact_preserves_object_identity():
    """The same Fact object ends up in the token after update_fact."""
    engine = InferenceEngine()
    seen = []
    engine.add_production(Production(
        lhs=[Pattern(LoanApp, alpha_tests=(_is_approved,))],
        rhs=lambda t: seen.append(t.facts[-1]),
    ))
    app = Fact(LoanApp("Dave", True))
    engine.add_fact(app)
    engine.update_fact(app)
    engine.run()
    assert len(seen) == 1
    assert seen[0] is app


def test_token_bindings_accessible_in_rhs():
    """token.bindings populated by pattern.bindings is accessible in RHS."""
    engine = InferenceEngine()
    captured = []
    p = Pattern(Color, bindings=(("$color", "color"),))
    engine.add_production(Production(
        lhs=[p],
        rhs=lambda t: captured.append(t.bindings["$color"]),
    ))
    engine.add_fact(Fact(Color("b1", "red")))
    engine.run()
    assert captured == ["red"]


# ---------------------------------------------------------------------------
# CEP — logical clock and event expiry
# ---------------------------------------------------------------------------


@dataclass
class _Reading:
    sensor: str
    ts: float


_READING_META = {
    "role": "event",
    "timestamp_field": "ts",
    "duration_field": None,
    "expires_delta": 30.0,
}


@dataclass
class _Plain:
    value: int


class TestCep:
    def setup_method(self):
        _Reading.__prl_meta__ = _READING_META

    def teardown_method(self):
        if hasattr(_Reading, "__prl_meta__"):
            del _Reading.__prl_meta__

    def test_advance_clock_sets_logical_clock(self):
        engine = InferenceEngine()
        engine.advance_clock(42.0)
        assert engine.logical_clock == 42.0

    def test_advance_clock_is_absolute(self):
        engine = InferenceEngine()
        engine.advance_clock(10.0)
        engine.advance_clock(5.0)
        assert engine.logical_clock == 5.0

    def test_add_fact_stamps_timestamp(self):
        engine = InferenceEngine()
        f = Fact(_Reading("s1", 100.0))
        engine.add_fact(f)
        assert f.timestamp == 100.0

    def test_add_fact_no_meta_no_stamp(self):
        engine = InferenceEngine()
        f = Fact(_Plain(1))
        engine.add_fact(f)
        assert f.timestamp is None

    def test_expire_events_removes_stale(self):
        engine = InferenceEngine()
        engine.add_production(Production(lhs=[Pattern(_Reading)], rhs=lambda t: None))
        f = Fact(_Reading("s1", 0.0))
        engine.add_fact(f)
        engine.advance_clock(31.0)
        engine.run()
        assert f not in engine.network.root._facts

    def test_expire_events_keeps_fresh(self):
        engine = InferenceEngine()
        engine.add_production(Production(lhs=[Pattern(_Reading)], rhs=lambda t: None))
        f = Fact(_Reading("s1", 0.0))
        engine.add_fact(f)
        engine.advance_clock(29.0)
        engine.run()
        assert f in engine.network.root._facts

    def test_non_event_fact_never_expires(self):
        engine = InferenceEngine()
        engine.add_production(Production(lhs=[Pattern(_Plain)], rhs=lambda t: None))
        f = Fact(_Plain(1))
        engine.add_fact(f)
        engine.advance_clock(9999.0)
        engine.run()
        assert f in engine.network.root._facts

    def test_expiry_triggers_negative_rules(self):
        engine = InferenceEngine()
        fired = []
        engine.add_production(Production(
            lhs=[Pattern(_Reading, negated=True)],
            rhs=lambda t: fired.append(1),
        ))
        f = Fact(_Reading("s1", 0.0))
        engine.add_fact(f)
        engine.advance_clock(31.0)
        engine.run()
        assert fired  # negated rule fires after expiry removes the event

    def test_expire_multiple_stale(self):
        engine = InferenceEngine()
        engine.add_production(Production(lhs=[Pattern(_Reading)], rhs=lambda t: None))
        f1 = Fact(_Reading("s1", 0.0))
        f2 = Fact(_Reading("s2", 5.0))
        engine.add_fact(f1)
        engine.add_fact(f2)
        engine.advance_clock(40.0)
        engine.run()
        assert f1 not in engine.network.root._facts
        assert f2 not in engine.network.root._facts

    def test_is_stale_no_expires_delta(self):
        # line 160: _is_stale returns False when expires_delta is absent from meta
        _Reading.__prl_meta__ = {"role": "event", "timestamp_field": "ts"}
        engine = InferenceEngine()
        f = Fact(_Reading("s1", 0.0))
        engine.add_fact(f)
        engine.advance_clock(9999.0)
        engine.run()
        assert f in engine.network.root._facts

    def test_is_stale_no_timestamp_field(self):
        # line 162: _is_stale returns False when fact.timestamp is None
        _Reading.__prl_meta__ = {"role": "event", "expires_delta": 30.0}
        engine = InferenceEngine()
        f = Fact(_Reading("s1", 0.0))
        engine.add_fact(f)  # no timestamp_field → fact.timestamp stays None
        engine.advance_clock(9999.0)
        engine.run()
        assert f in engine.network.root._facts


# ---------------------------------------------------------------------------
# no_loop — _changed_by_other
# ---------------------------------------------------------------------------


@dataclass
class _Trigger:
    value: int


@dataclass
class _Response:
    value: int


class TestNoLoop:
    def test_no_loop_preserves_other_production_entries(self):
        # line 33: _changed_by_other returns True for entries from other productions
        engine = InferenceEngine()
        fired_b = []

        def rhs_a(token: Token) -> None:
            engine.add_fact(Fact(_Response(1)))

        engine.add_production(Production(
            lhs=[Pattern(_Trigger)],
            rhs=rhs_a,
            no_loop=True,
        ))
        engine.add_production(Production(
            lhs=[Pattern(_Response)],
            rhs=lambda t: fired_b.append(1),
        ))
        engine.add_fact(Fact(_Trigger(1)))
        engine.run()
        assert fired_b == [1]
