"""Tests for Pattern, JoinSpec, NccGroup, and Production (condition.py)."""
from dataclasses import dataclass, FrozenInstanceError

from rete.condition import JoinSpec, NccGroup, Pattern, Production
from rete.fact import Fact, Token


@dataclass
class Block:
    """Minimal POPO used as a test fact object."""

    color: str


@dataclass
class On:
    """Minimal POPO: upper block resting on lower."""

    upper: str
    lower: str


# ---------------------------------------------------------------------------
# Pattern.matches
# ---------------------------------------------------------------------------


def test_pattern_type_match():
    assert Pattern(type_=Block).matches(Fact(Block("red")))


def test_pattern_type_mismatch():
    assert not Pattern(type_=Block).matches(Fact(On("a", "b")))


def test_pattern_alpha_test_passes():
    def is_red(obj: Block) -> bool:
        return obj.color == "red"

    p = Pattern(type_=Block, alpha_tests=(is_red,))
    assert p.matches(Fact(Block("red")))
    assert not p.matches(Fact(Block("blue")))


def test_pattern_alpha_test_fails():
    def always_false(obj: Block) -> bool:
        return False

    p = Pattern(type_=Block, alpha_tests=(always_false,))
    assert not p.matches(Fact(Block("red")))


def test_pattern_all_alpha_tests_must_pass():
    def is_red(obj: Block) -> bool:
        return obj.color == "red"

    def always_false(obj: Block) -> bool:
        return False

    p = Pattern(type_=Block, alpha_tests=(is_red, always_false))
    assert not p.matches(Fact(Block("red")))


def test_pattern_no_alpha_tests_type_only():
    p = Pattern(type_=Block)
    assert p.matches(Fact(Block("any")))
    assert not p.matches(Fact(On("a", "b")))


def test_pattern_type_check_before_alpha_tests():
    """isinstance fails fast; alpha_tests must not run on wrong type."""
    calls: list[bool] = []

    def probe(obj: Block) -> bool:
        calls.append(True)
        return True

    p = Pattern(type_=Block, alpha_tests=(probe,))
    assert not p.matches(Fact(On("a", "b")))
    assert calls == []  # probe never called


# ---------------------------------------------------------------------------
# Pattern.extract_bindings
# ---------------------------------------------------------------------------


def test_pattern_extract_bindings_single():
    p = Pattern(type_=Block, bindings=(("$color", "color"),))
    assert p.extract_bindings(Fact(Block("red"))) == {"$color": "red"}


def test_pattern_extract_bindings_empty():
    p = Pattern(type_=Block)
    assert p.extract_bindings(Fact(Block("red"))) == {}


def test_pattern_extract_bindings_multiple():
    p = Pattern(type_=On, bindings=(("$upper", "upper"), ("$lower", "lower")))
    result = p.extract_bindings(Fact(On("A", "B")))
    assert result == {"$upper": "A", "$lower": "B"}


# ---------------------------------------------------------------------------
# Pattern.alpha_key
# ---------------------------------------------------------------------------


def test_pattern_alpha_key_includes_type():
    p = Pattern(type_=Block)
    assert p.alpha_key()[0] is Block


def test_pattern_alpha_key_same_function_shared():
    def is_red(obj: Block) -> bool:
        return obj.color == "red"

    p1 = Pattern(type_=Block, alpha_tests=(is_red,))
    p2 = Pattern(type_=Block, alpha_tests=(is_red,))
    assert p1.alpha_key() == p2.alpha_key()


def test_pattern_alpha_key_distinct_lambdas():
    # Two distinct lambda objects → different ids → different alpha keys.
    # Stable, named function references are required for alpha memory sharing.
    p1 = Pattern(type_=Block, alpha_tests=(lambda obj: True,))
    p2 = Pattern(type_=Block, alpha_tests=(lambda obj: True,))
    assert p1.alpha_key() != p2.alpha_key()


def test_pattern_alpha_key_different_types():
    p1 = Pattern(type_=Block)
    p2 = Pattern(type_=On)
    assert p1.alpha_key() != p2.alpha_key()


def test_pattern_alpha_key_excludes_bindings():
    """Patterns that differ only in bindings share an alpha memory."""
    def fn(obj: Block) -> bool:
        return True

    p1 = Pattern(type_=Block, alpha_tests=(fn,), bindings=(("$a", "color"),))
    p2 = Pattern(type_=Block, alpha_tests=(fn,), bindings=(("$b", "color"),))
    assert p1.alpha_key() == p2.alpha_key()


# ---------------------------------------------------------------------------
# Pattern fields and frozen
# ---------------------------------------------------------------------------


def test_pattern_negated_default_false():
    assert not Pattern(type_=Block).negated


def test_pattern_negated_true():
    assert Pattern(type_=Block, negated=True).negated


def test_pattern_frozen():
    p = Pattern(type_=Block)
    try:
        p.negated = True  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except FrozenInstanceError:
        pass


# ---------------------------------------------------------------------------
# JoinSpec
# ---------------------------------------------------------------------------


def test_joinspec_fields():
    js = JoinSpec(attr_of_fact="upper", var_name="$lower")
    assert js.attr_of_fact == "upper"
    assert js.var_name == "$lower"


def test_joinspec_frozen():
    js = JoinSpec("upper", "$lower")
    try:
        js.var_name = "changed"  # type: ignore[misc]
        raise AssertionError("expected FrozenInstanceError")
    except FrozenInstanceError:
        pass


# ---------------------------------------------------------------------------
# NccGroup
# ---------------------------------------------------------------------------


def test_ncc_group_with_patterns():
    p1 = Pattern(type_=Block)
    p2 = Pattern(type_=On)
    g = NccGroup(conditions=(p1, p2))
    assert len(g.conditions) == 2
    assert g.conditions[0] is p1
    assert g.conditions[1] is p2


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------


def test_production_with_pattern():
    p = Pattern(type_=Block)
    prod = Production(lhs=[p], rhs=lambda t: None)
    assert prod.lhs == [p]
    assert callable(prod.rhs)


def test_production_rhs_invocable():
    fired: list[Token] = []
    prod = Production(lhs=[], rhs=fired.append)
    t = Token()
    prod.rhs(t)
    assert fired == [t]
