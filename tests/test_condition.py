from rete.condition import WILDCARD, Condition, Production, matches
from rete.wme import Token, WME


def test_constant_matches_exact():
    assert matches(Condition("b1", "color", "red"), WME("b1", "color", "red"))


def test_constant_rejects_mismatch():
    assert not matches(Condition("b1", "color", "red"), WME("b1", "color", "blue"))


def test_wildcard_matches_any():
    assert matches(Condition(WILDCARD, WILDCARD, WILDCARD), WME("x", "y", "z"))


def test_variable_matches_any():
    assert matches(Condition("?x", "?attr", "?val"), WME("b1", "color", "red"))


def test_condition_all_constants_match():
    assert matches(Condition("b1", "color", "red"), WME("b1", "color", "red"))


def test_condition_all_constants_reject():
    assert not matches(Condition("b1", "color", "red"), WME("b2", "color", "red"))


def test_condition_mixed_tests():
    assert matches(Condition("b1", WILDCARD, "?v"), WME("b1", "color", "red"))


def test_production_stores_lhs_rhs():
    cond = Condition("b1", "color", "red")
    p = Production(lhs=[cond], rhs=lambda t: None)
    assert p.lhs == [cond]
    assert callable(p.rhs)


def test_production_rhs_invocable():
    fired: list[Token] = []
    p = Production(lhs=[], rhs=fired.append)
    t = Token()
    p.rhs(t)
    assert fired == [t]
