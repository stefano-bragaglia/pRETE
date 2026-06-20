"""Unit tests for the PRL compiler (``src/rete/prl.py``).

Tests are committed before the implementation and expected to fail with
``ImportError`` until ``prl.py`` is created.  Private helpers are imported
directly for focused unit testing.
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields

import pytest

from rete.condition import JoinSpec, NccGroup, Pattern
from rete.fact import Fact
from rete.fact import Token as ReteToken
from rete.prl_ast import (
    BindConstraint,
    CompareConstraint,
    DeclareDecl,
    FieldDecl,
    NccPatternGroup,
    PatternNode,
)
from rete.prl import (
    _compile_declare,
    _compile_lhs,
    _compile_rhs,
    _java_type,
    _resolve_type,
    _strip_dollars,
    load_prl,
)


# ---------------------------------------------------------------------------
# Minimal fact types for compiler tests
# ---------------------------------------------------------------------------

@dataclass
class _Temp:
    value: float


@dataclass
class _Sensor:
    id: str
    value: float


_TYPES: dict[str, type] = {"_Temp": _Temp, "_Sensor": _Sensor}


# ===========================================================================
# Java type mapping
# ===========================================================================

class TestJavaType:
    """``_java_type`` maps PRL type names to Python types."""

    def test_float_maps_to_float(self) -> None:
        assert _java_type("float", {}) is float

    def test_str_maps_to_str(self) -> None:
        assert _java_type("str", {}) is str

    def test_bool_maps_to_bool(self) -> None:
        assert _java_type("bool", {}) is bool

    def test_int_maps_to_int(self) -> None:
        assert _java_type("int", {}) is int

    def test_java_aliases_still_work(self) -> None:
        assert _java_type("double", {}) is float
        assert _java_type("String", {}) is str
        assert _java_type("boolean", {}) is bool

    def test_unknown_falls_back_to_any(self) -> None:
        from typing import Any
        assert _java_type("Blob", {}) is Any

    def test_user_type_takes_priority(self) -> None:
        assert _java_type("_Temp", _TYPES) is _Temp


# ===========================================================================
# Declare compilation
# ===========================================================================

class TestCompileDeclare:
    """``_compile_declare`` produces a usable Python dataclass."""

    def test_no_fields(self) -> None:
        cls = _compile_declare(DeclareDecl("Marker", ()), {})
        assert cls.__name__ == "Marker"
        assert len(dc_fields(cls)) == 0

    def test_one_field_type(self) -> None:
        decl = DeclareDecl("Temp", (FieldDecl("value", "float"),))
        cls = _compile_declare(decl, {})
        assert dc_fields(cls)[0].type is float

    def test_two_fields_names(self) -> None:
        decl = DeclareDecl(
            "Temp",
            (FieldDecl("sensor", "str"), FieldDecl("value", "float")),
        )
        cls = _compile_declare(decl, {})
        names = [f.name for f in dc_fields(cls)]
        assert names == ["sensor", "value"]

    def test_instance_creation(self) -> None:
        decl = DeclareDecl("Temp", (FieldDecl("value", "float"),))
        cls = _compile_declare(decl, {})
        obj = cls(value=42.0)
        assert obj.value == 42.0


# ===========================================================================
# Type resolution
# ===========================================================================

class TestResolveType:
    """``_resolve_type`` guards against undeclared types."""

    def test_known_type_returned(self) -> None:
        assert _resolve_type("_Temp", _TYPES) is _Temp

    def test_unknown_type_raises_name_error(self) -> None:
        with pytest.raises(NameError):
            _resolve_type("Ghost", {})


# ===========================================================================
# Pattern compilation
# ===========================================================================

class TestCompilePattern:
    """``_compile_lhs`` produces correct ``Pattern`` objects."""

    def test_pattern_type_matches_declared_class(self) -> None:
        node = PatternNode("_Temp", None, (), False)
        conditions, _ = _compile_lhs((node,), _TYPES)
        assert isinstance(conditions[0], Pattern)
        assert conditions[0].type_ is _Temp

    def test_alpha_test_fires_for_matching_value(self) -> None:
        c = CompareConstraint("value", ">=", 80.0)
        node = PatternNode("_Temp", None, (c,), False)
        conditions, _ = _compile_lhs((node,), _TYPES)
        pat = conditions[0]
        assert isinstance(pat, Pattern)
        assert pat.matches(Fact(_Temp(95.0))) is True
        assert pat.matches(Fact(_Temp(60.0))) is False

    def test_bind_constraint_produces_binding_tuple(self) -> None:
        c = BindConstraint("$v", "value")
        node = PatternNode("_Temp", None, (c,), False)
        conditions, _ = _compile_lhs((node,), _TYPES)
        pat = conditions[0]
        assert isinstance(pat, Pattern)
        assert ("$v", "value") in pat.bindings

    def test_join_constraint_produces_join_spec(self) -> None:
        c = CompareConstraint("value", "==", "$other")
        node = PatternNode("_Temp", None, (c,), False)
        conditions, _ = _compile_lhs((node,), _TYPES)
        pat = conditions[0]
        assert isinstance(pat, Pattern)
        assert len(pat.join_tests) == 1
        assert isinstance(pat.join_tests[0], JoinSpec)
        assert pat.join_tests[0].var_name == "$other"

    def test_negated_flag_propagates(self) -> None:
        node = PatternNode("_Temp", None, (), True)
        conditions, _ = _compile_lhs((node,), _TYPES)
        assert conditions[0].negated is True

    def test_fact_var_populates_fact_bindings(self) -> None:
        node = PatternNode("_Temp", "$t", (), False)
        _, fact_bindings = _compile_lhs((node,), _TYPES)
        assert "$t" in fact_bindings
        assert fact_bindings["$t"] == 0


# ===========================================================================
# NCC compilation
# ===========================================================================

class TestCompileNcc:
    """``NccPatternGroup`` compiles to ``NccGroup``."""

    def test_one_pattern_ncc(self) -> None:
        inner = PatternNode("_Temp", None, (), False)
        node = NccPatternGroup((inner,))
        conditions, _ = _compile_lhs((node,), _TYPES)
        assert isinstance(conditions[0], NccGroup)
        assert len(conditions[0].conditions) == 1

    def test_two_pattern_ncc(self) -> None:
        p1 = PatternNode("_Temp", None, (), False)
        p2 = PatternNode("_Sensor", None, (), False)
        node = NccPatternGroup((p1, p2))
        conditions, _ = _compile_lhs((node,), _TYPES)
        ncc = conditions[0]
        assert isinstance(ncc, NccGroup)
        assert ncc.conditions[0].type_ is _Temp
        assert ncc.conditions[1].type_ is _Sensor


# ===========================================================================
# LHS (compile_lhs composition)
# ===========================================================================

class TestCompileLhs:
    """``_compile_lhs`` combines patterns and NCCs in order."""

    def test_single_pattern_length(self) -> None:
        node = PatternNode("_Temp", None, (), False)
        conditions, _ = _compile_lhs((node,), _TYPES)
        assert len(conditions) == 1

    def test_two_patterns_length(self) -> None:
        n1 = PatternNode("_Temp", None, (), False)
        n2 = PatternNode("_Sensor", None, (), False)
        conditions, _ = _compile_lhs((n1, n2), _TYPES)
        assert len(conditions) == 2

    def test_mixed_pattern_and_ncc(self) -> None:
        pat = PatternNode("_Temp", None, (), False)
        ncc = NccPatternGroup((PatternNode("_Sensor", None, (), False),))
        conditions, _ = _compile_lhs((pat, ncc), _TYPES)
        assert isinstance(conditions[0], Pattern)
        assert isinstance(conditions[1], NccGroup)

    def test_fact_binding_index_tracks_position(self) -> None:
        n1 = PatternNode("_Temp", "$a", (), False)
        n2 = PatternNode("_Sensor", "$b", (), False)
        _, fact_bindings = _compile_lhs((n1, n2), _TYPES)
        assert fact_bindings["$a"] == 0
        assert fact_bindings["$b"] == 1


# ===========================================================================
# Dollar stripping
# ===========================================================================

class TestStripDollars:
    """``_strip_dollars`` removes ``$`` prefixes from variable names."""

    def test_simple_var(self) -> None:
        assert _strip_dollars("$foo") == "foo"

    def test_dotted_access(self) -> None:
        assert _strip_dollars("$loan.obj.approved") == "loan.obj.approved"

    def test_no_dollar_unchanged(self) -> None:
        assert _strip_dollars("approved = False") == "approved = False"

    def test_multiple_vars(self) -> None:
        result = _strip_dollars("$a + $b")
        assert result == "a + b"


# ===========================================================================
# RHS compilation
# ===========================================================================

class TestCompileRhs:
    """``_compile_rhs`` produces a callable that executes the then-block."""

    def test_empty_rhs_callable(self) -> None:
        rhs = _compile_rhs("", {}, {}, None)
        tok = ReteToken(facts=(), bindings={})
        rhs(tok)   # must not raise

    def test_field_binding_accessible_by_stripped_name(self) -> None:
        captured: list[str] = []
        rhs = _compile_rhs(
            "captured.append(sensor)",
            {},
            {"captured": captured},
            None,
        )
        tok = ReteToken(facts=(), bindings={"$sensor": "T1"})
        rhs(tok)
        assert captured == ["T1"]

    def test_fact_binding_injects_fact(self) -> None:
        captured: list[object] = []
        fact = Fact(_Temp(99.0))
        rhs = _compile_rhs(
            "captured.append(t)",
            {"$t": 0},
            {"captured": captured},
            None,
        )
        tok = ReteToken(facts=(fact,), bindings={})
        rhs(tok)
        assert captured == [fact]

    def test_helpers_absent_when_engine_is_none(self) -> None:
        rhs = _compile_rhs("insert", {}, {}, None)
        tok = ReteToken(facts=(), bindings={})
        with pytest.raises(NameError):
            rhs(tok)

    def test_helpers_present_when_engine_given(self) -> None:
        from unittest.mock import MagicMock
        engine = MagicMock()
        captured: list[str] = []
        rhs = _compile_rhs(
            "captured.append(type(insert).__name__)",
            {},
            {"captured": captured},
            engine,
        )
        tok = ReteToken(facts=(), bindings={})
        rhs(tok)
        assert captured == ["function"]


# ===========================================================================
# load_prl end-to-end
# ===========================================================================

class TestLoadPrl:
    """``load_prl`` assembles the full pipeline."""

    def test_empty_source(self) -> None:
        types, prods = load_prl("")
        assert types == {}
        assert prods == []

    def test_declare_populates_types(self) -> None:
        types, _ = load_prl("declare Foo\n  x: int\nend")
        assert "Foo" in types
        assert len(dc_fields(types["Foo"])) == 1

    def test_rule_produces_one_production(self) -> None:
        src = (
            "declare _T\n  v: float\nend\n"
            'rule "r" when\n  _T(v > 0)\nthen\npass\nend'
        )
        _, prods = load_prl(src)
        assert len(prods) == 1

    def test_production_lhs_populated(self) -> None:
        src = (
            "declare _T\n  v: float\nend\n"
            'rule "r" when\n  _T(v > 0)\nthen\npass\nend'
        )
        _, prods = load_prl(src)
        assert len(prods[0].lhs) == 1

    def test_external_types_available(self) -> None:
        src = 'rule "r" when\n  _Temp(value > 0)\nthen\npass\nend'
        _, prods = load_prl(src, types=_TYPES)
        assert len(prods) == 1

    def test_unknown_type_raises(self) -> None:
        src = 'rule "r" when\n  Ghost(x > 0)\nthen\npass\nend'
        with pytest.raises(NameError):
            load_prl(src)
