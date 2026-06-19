from rete.condition import WILDCARD, Condition, Production
from rete.wme import Token, WME


def test_constant_matches_exact():
    assert Condition("b1", "color", "red").matches(WME("b1", "color", "red"))


def test_constant_rejects_mismatch():
    assert not Condition("b1", "color", "red").matches(WME("b1", "color", "blue"))


def test_wildcard_matches_any():
    assert Condition(WILDCARD, WILDCARD, WILDCARD).matches(WME("x", "y", "z"))


def test_variable_matches_any():
    assert Condition("?x", "?attr", "?val").matches(WME("b1", "color", "red"))


def test_condition_all_constants_match():
    assert Condition("b1", "color", "red").matches(WME("b1", "color", "red"))


def test_condition_all_constants_reject():
    assert not Condition("b1", "color", "red").matches(WME("b2", "color", "red"))


def test_condition_mixed_tests():
    assert Condition("b1", WILDCARD, "?v").matches(WME("b1", "color", "red"))


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
