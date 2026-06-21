"""Unit tests for the PRL compiler (``src/rete/prl.py``).

Tests are committed before the implementation and expected to fail with
``ImportError`` until ``prl.py`` is created.  Private helpers are imported
directly for focused unit testing.
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dc_fields

import pytest

from rete.condition import AccumulateSpec, JoinSpec, NccGroup, Pattern
from rete.fact import Fact
from rete.fact import Token as ReteToken
from rete.prl_ast import (
    AccumulateExpr,
    BindConstraint,
    CompareConstraint,
    DeclareDecl,
    FieldDecl,
    NamedConstraint,
    NccPatternGroup,
    PatternNode,
    PositionalConstraint,
    Tag,
)
from rete.prl import (
    _ACCUMULATE_FNS,
    _compile_accumulate,
    _compile_declare,
    _compile_lhs,
    _compile_pattern,
    _compile_rhs,
    _has_tag,
    _java_type,
    _parse_time_offset,
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
# Declare compilation — extends
# ===========================================================================

class TestCompileDeclareExtends:
    """``_compile_declare`` handles ``extends`` to produce subclasses."""

    def test_child_is_subclass(self) -> None:
        parent = _compile_declare(
            DeclareDecl("Animal", (FieldDecl("name", "str"),)), {}
        )
        child = _compile_declare(
            DeclareDecl("Dog", (FieldDecl("breed", "str"),), extends="Animal"),
            {"Animal": parent},
        )
        assert issubclass(child, parent)

    def test_child_has_parent_and_own_fields(self) -> None:
        parent = _compile_declare(
            DeclareDecl("Animal", (FieldDecl("name", "str"),)), {}
        )
        child = _compile_declare(
            DeclareDecl("Dog", (FieldDecl("breed", "str"),), extends="Animal"),
            {"Animal": parent},
        )
        obj = child(name="Rex", breed="Lab")
        assert obj.name == "Rex"
        assert obj.breed == "Lab"

    def test_out_of_order_compiles(self) -> None:
        src = (
            "declare Dog extends Animal\n  breed: str\nend\n"
            "declare Animal\n  name: str\nend"
        )
        types, _ = load_prl(src)
        assert issubclass(types["Dog"], types["Animal"])

    def test_circular_inheritance_raises(self) -> None:
        src = "declare A extends B\nend\ndeclare B extends A\nend"
        with pytest.raises(TypeError):
            load_prl(src)

    def test_unknown_parent_raises(self) -> None:
        with pytest.raises(NameError):
            _compile_declare(
                DeclareDecl("Dog", (), extends="Ghost"),
                {},
            )


# ===========================================================================
# Declare compilation — @key (ES-3)
# ===========================================================================

class TestCompileDeclareKey:
    """``@key`` field tags inject key-only equality and hashing."""

    def test_no_key_uses_full_equality(self) -> None:
        decl = DeclareDecl("T", (FieldDecl("a", "int"), FieldDecl("b", "int")))
        cls = _compile_declare(decl, {})
        assert cls(a=1, b=2) != cls(a=1, b=99)

    def test_single_key_equal_when_key_matches(self) -> None:
        decl = DeclareDecl("C", (
            FieldDecl("id", "int", tags=(Tag("key"),)),
            FieldDecl("name", "str"),
        ))
        cls = _compile_declare(decl, {})
        assert cls(id=1, name="Alice") == cls(id=1, name="Bob")

    def test_single_key_unequal_when_key_differs(self) -> None:
        decl = DeclareDecl("C", (
            FieldDecl("id", "int", tags=(Tag("key"),)),
            FieldDecl("name", "str"),
        ))
        cls = _compile_declare(decl, {})
        assert cls(id=1, name="Alice") != cls(id=2, name="Alice")

    def test_single_key_hash_consistency(self) -> None:
        decl = DeclareDecl("C", (
            FieldDecl("id", "int", tags=(Tag("key"),)),
            FieldDecl("name", "str"),
        ))
        cls = _compile_declare(decl, {})
        assert hash(cls(id=1, name="Alice")) == hash(cls(id=1, name="Bob"))

    def test_composite_key_all_match(self) -> None:
        decl = DeclareDecl("O", (
            FieldDecl("cid", "int", tags=(Tag("key"),)),
            FieldDecl("oid", "int", tags=(Tag("key"),)),
            FieldDecl("amount", "float"),
        ))
        cls = _compile_declare(decl, {})
        assert cls(cid=1, oid=99, amount=10.0) == cls(cid=1, oid=99, amount=99.9)

    def test_composite_key_partial_mismatch(self) -> None:
        decl = DeclareDecl("O", (
            FieldDecl("cid", "int", tags=(Tag("key"),)),
            FieldDecl("oid", "int", tags=(Tag("key"),)),
            FieldDecl("amount", "float"),
        ))
        cls = _compile_declare(decl, {})
        assert cls(cid=1, oid=1, amount=5.0) != cls(cid=1, oid=2, amount=5.0)

    def test_type_mismatch_returns_not_implemented(self) -> None:
        decl = DeclareDecl("C", (FieldDecl("id", "int", tags=(Tag("key"),)),))
        cls = _compile_declare(decl, {})
        assert cls(id=1).__eq__("not-a-customer") is NotImplemented

    def test_usable_as_dict_key(self) -> None:
        decl = DeclareDecl("C", (
            FieldDecl("id", "int", tags=(Tag("key"),)),
            FieldDecl("name", "str"),
        ))
        cls = _compile_declare(decl, {})
        d = {cls(id=1, name="Alice"): "first"}
        assert d[cls(id=1, name="Bob")] == "first"


# ===========================================================================
# Shorthand constraint compilation (ES-4)
# ===========================================================================

class TestCompileShorthand:
    """Positional and named constraints expand to correct alpha tests."""

    def _point_types(self) -> dict[str, type]:
        decl = DeclareDecl("Point", (FieldDecl("x", "int"), FieldDecl("y", "int")))
        return {"Point": _compile_declare(decl, {})}

    def test_positional_first_field(self) -> None:
        types = self._point_types()
        node = PatternNode("Point", None, (PositionalConstraint(0),), False)
        p = _compile_pattern(node, 0, types, {})
        assert p.alpha_tests[0](types["Point"](x=0, y=9))
        assert not p.alpha_tests[0](types["Point"](x=1, y=0))

    def test_positional_second_field(self) -> None:
        types = self._point_types()
        node = PatternNode("Point", None, (
            PositionalConstraint(99), PositionalConstraint(0),
        ), False)
        p = _compile_pattern(node, 0, types, {})
        assert p.alpha_tests[1](types["Point"](x=0, y=0))
        assert not p.alpha_tests[1](types["Point"](x=0, y=1))

    def test_named_resolves_to_correct_field(self) -> None:
        types = self._point_types()
        node = PatternNode("Point", None, (NamedConstraint("y", 5),), False)
        p = _compile_pattern(node, 0, types, {})
        assert p.alpha_tests[0](types["Point"](x=0, y=5))
        assert not p.alpha_tests[0](types["Point"](x=5, y=0))

    def test_too_many_positionals_raises(self) -> None:
        types = self._point_types()
        node = PatternNode("Point", None, (
            PositionalConstraint(0),
            PositionalConstraint(0),
            PositionalConstraint(0),
        ), False)
        with pytest.raises(SyntaxError):
            _compile_pattern(node, 0, types, {})

    def test_collision_raises(self) -> None:
        types = self._point_types()
        node = PatternNode("Point", None, (
            PositionalConstraint(0),
            NamedConstraint("x", 0),
        ), False)
        with pytest.raises(SyntaxError):
            _compile_pattern(node, 0, types, {})

    def test_positional_with_extends(self) -> None:
        base = DeclareDecl("A", (FieldDecl("a", "int"),))
        child = DeclareDecl("B", (FieldDecl("b", "int"),), extends="A")
        types = {"A": _compile_declare(base, {})}
        types["B"] = _compile_declare(child, types)
        node = PatternNode("B", None, (PositionalConstraint(7),), False)
        p = _compile_pattern(node, 0, types, {})
        assert p.alpha_tests[0](types["B"](a=7, b=0))
        assert not p.alpha_tests[0](types["B"](a=0, b=7))


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

    def test_no_loop_attribute_sets_production_flag(self) -> None:
        src = (
            "declare _T\n  v: float\nend\n"
            'rule "r"\n  no-loop\n  when\n  _T(v > 0)\nthen\npass\nend'
        )
        _, prods = load_prl(src)
        assert prods[0].no_loop is True

    def test_no_loop_tag_sets_production_flag(self) -> None:
        src = (
            "declare _T\n  v: float\nend\n"
            '@no-loop\nrule "r" when\n  _T(v > 0)\nthen\npass\nend'
        )
        _, prods = load_prl(src)
        assert prods[0].no_loop is True

    def test_no_loop_false_by_default(self) -> None:
        src = (
            "declare _T\n  v: float\nend\n"
            'rule "r" when\n  _T(v > 0)\nthen\npass\nend'
        )
        _, prods = load_prl(src)
        assert prods[0].no_loop is False


# ===========================================================================
# Import resolution (ES-5)
# ===========================================================================

class TestCompileImport:
    """Import declarations resolve to the correct Python class."""

    def test_from_import_single(self) -> None:
        types, _ = load_prl("from rete.fact import Fact")
        from rete.fact import Fact
        assert types["Fact"] is Fact

    def test_import_class(self) -> None:
        types, _ = load_prl("import rete.fact.Fact")
        from rete.fact import Fact
        assert types["Fact"] is Fact

    def test_from_import_with_alias(self) -> None:
        types, _ = load_prl("from rete.fact import Fact as F")
        from rete.fact import Fact
        assert types["F"] is Fact

    def test_from_import_multiple(self) -> None:
        types, _ = load_prl("from rete.fact import Fact, Token")
        from rete.fact import Fact, Token
        assert types["Fact"] is Fact
        assert types["Token"] is Token

    def test_unknown_module_raises(self) -> None:
        with pytest.raises(ImportError):
            load_prl("import no_such_module.NoClass")

    def test_unknown_attr_raises(self) -> None:
        with pytest.raises(ImportError):
            load_prl("from rete.fact import NoSuchClass")

    def test_import_available_for_extends(self) -> None:
        """Imported type is available as parent in ``extends``."""
        import sys
        import types as _types_mod
        from dataclasses import make_dataclass
        Base = make_dataclass("_ES5Base", [("score", int)])
        mod = _types_mod.ModuleType("_es5_compiler_mod")
        mod._ES5Base = Base  # type: ignore[attr-defined]
        sys.modules["_es5_compiler_mod"] = mod
        try:
            src = (
                "from _es5_compiler_mod import _ES5Base\n"
                "declare Child extends _ES5Base\n  name: str\nend\n"
            )
            resolved, _ = load_prl(src)
            assert issubclass(resolved["Child"], Base)
        finally:
            del sys.modules["_es5_compiler_mod"]


# ===========================================================================
# or disjunction compilation (ES-6)
# ===========================================================================

class TestCompileOr:
    """``or`` rule compiles to K productions sharing the same RHS."""

    @staticmethod
    def _types() -> dict:
        from dataclasses import make_dataclass
        return {
            "A": make_dataclass("A", [("v", int)]),
            "B": make_dataclass("B", [("v", int)]),
        }

    def test_two_branch_or_produces_two_productions(self) -> None:
        src = 'rule "r"\nwhen\n  A() or\n  B()\nthen\nend'
        _, prods = load_prl(src, types=self._types())
        assert len(prods) == 2

    def test_three_branch_or_produces_three_productions(self) -> None:
        src = 'rule "r"\nwhen\n  A() or\n  B() or\n  A()\nthen\nend'
        _, prods = load_prl(src, types=self._types())
        assert len(prods) == 3

    def test_non_or_rule_produces_one_production(self) -> None:
        src = 'rule "r"\nwhen\n  A()\nthen\nend'
        _, prods = load_prl(src, types=self._types())
        assert len(prods) == 1

    def test_mismatched_branch_vars_raises(self) -> None:
        """All or-branches must bind the same variable set."""
        src = 'rule "r"\nwhen\n  $x: A() or\n  $y: B()\nthen\nend'
        with pytest.raises(SyntaxError, match="branch"):
            load_prl(src, types=self._types())

    def test_or_first_branch_lhs_correct_type(self) -> None:
        src = 'rule "r"\nwhen\n  A() or\n  B()\nthen\nend'
        types = self._types()
        _, prods = load_prl(src, types=types)
        assert prods[0].lhs[0].type_ is types["A"]
        assert prods[1].lhs[0].type_ is types["B"]

    def test_matching_fact_vars_do_not_raise(self) -> None:
        src = 'rule "r"\nwhen\n  $x: A() or\n  $x: B()\nthen\nend'
        _, prods = load_prl(src, types=self._types())
        assert len(prods) == 2

    def test_or_no_loop_propagates_to_all_branches(self) -> None:
        src = 'rule "r"\n  no-loop\nwhen\n  A() or\n  B()\nthen\nend'
        _, prods = load_prl(src, types=self._types())
        assert all(p.no_loop for p in prods)


# ===========================================================================
# forall compilation (ES-6)
# ===========================================================================

class TestCompileForall:
    """``forall(P, Q)`` compiles to an NccGroup."""

    @staticmethod
    def _types() -> dict:
        from dataclasses import make_dataclass
        return {
            "Order": make_dataclass("Order", [("status", str)]),
            "Approval": make_dataclass("Approval", [("ref", str)]),
        }

    def test_forall_produces_ncc_group(self) -> None:
        src = 'rule "r"\nwhen\n  forall(Order(), Approval())\nthen\nend'
        _, prods = load_prl(src, types=self._types())
        assert len(prods) == 1
        assert isinstance(prods[0].lhs[0], NccGroup)

    def test_forall_ncc_has_two_patterns(self) -> None:
        src = 'rule "r"\nwhen\n  forall(Order(), Approval())\nthen\nend'
        _, prods = load_prl(src, types=self._types())
        ncc = prods[0].lhs[0]
        assert len(ncc.conditions) == 2

    def test_forall_second_pattern_is_negated(self) -> None:
        src = 'rule "r"\nwhen\n  forall(Order(), Approval())\nthen\nend'
        _, prods = load_prl(src, types=self._types())
        ncc = prods[0].lhs[0]
        assert ncc.conditions[1].negated is True

    def test_forall_first_pattern_not_negated(self) -> None:
        src = 'rule "r"\nwhen\n  forall(Order(), Approval())\nthen\nend'
        _, prods = load_prl(src, types=self._types())
        ncc = prods[0].lhs[0]
        assert ncc.conditions[0].negated is False

    def test_forall_condition_types_correct(self) -> None:
        src = 'rule "r"\nwhen\n  forall(Order(), Approval())\nthen\nend'
        types = self._types()
        _, prods = load_prl(src, types=types)
        ncc = prods[0].lhs[0]
        assert ncc.conditions[0].type_ is types["Order"]
        assert ncc.conditions[1].type_ is types["Approval"]


# ===========================================================================
# CEP metadata
# ===========================================================================


def _make_decl(
    name: str = "Ev",
    fields: tuple = (),
    tags: tuple = (),
) -> DeclareDecl:
    return DeclareDecl(name=name, fields=fields, tags=tags)


def _field(name: str, *tag_names: str) -> FieldDecl:
    return FieldDecl(
        name=name, type_name="float", tags=tuple(Tag(n) for n in tag_names)
    )


class TestCepMeta:
    def test_role_event_sets_prl_meta(self) -> None:
        decl = _make_decl(tags=(Tag("role", "event"),))
        cls = _compile_declare(decl, {})
        assert cls.__prl_meta__["role"] == "event"

    def test_plain_declare_no_prl_meta(self) -> None:
        decl = _make_decl()
        cls = _compile_declare(decl, {})
        assert not hasattr(cls, "__prl_meta__")

    def test_timestamp_field_stored(self) -> None:
        decl = _make_decl(
            fields=(_field("ts", "timestamp"),),
            tags=(Tag("role", "event"),),
        )
        cls = _compile_declare(decl, {})
        assert cls.__prl_meta__["timestamp_field"] == "ts"

    def test_no_timestamp_field_is_none(self) -> None:
        decl = _make_decl(tags=(Tag("role", "event"),))
        cls = _compile_declare(decl, {})
        assert cls.__prl_meta__["timestamp_field"] is None

    def test_expires_delta_seconds(self) -> None:
        decl = _make_decl(tags=(Tag("role", "event"), Tag("expires", "30s")))
        cls = _compile_declare(decl, {})
        assert cls.__prl_meta__["expires_delta"] == 30.0

    def test_expires_delta_minutes(self) -> None:
        decl = _make_decl(tags=(Tag("role", "event"), Tag("expires", "5m")))
        cls = _compile_declare(decl, {})
        assert cls.__prl_meta__["expires_delta"] == 300.0

    def test_expires_delta_composite(self) -> None:
        decl = _make_decl(tags=(Tag("role", "event"), Tag("expires", "1h30m")))
        cls = _compile_declare(decl, {})
        assert cls.__prl_meta__["expires_delta"] == 5400.0

    def test_duration_field_stored(self) -> None:
        decl = _make_decl(
            fields=(_field("dur", "duration"),),
            tags=(Tag("role", "event"),),
        )
        cls = _compile_declare(decl, {})
        assert cls.__prl_meta__["duration_field"] == "dur"

    def test_no_expires_is_none(self) -> None:
        decl = _make_decl(tags=(Tag("role", "event"),))
        cls = _compile_declare(decl, {})
        assert cls.__prl_meta__["expires_delta"] is None

    def test_expires_without_timestamp_fact_stays(self) -> None:
        """@expires with no @timestamp → expires_delta set but timestamp_field None."""
        decl = _make_decl(tags=(Tag("role", "event"), Tag("expires", "10s")))
        cls = _compile_declare(decl, {})
        assert cls.__prl_meta__["expires_delta"] == 10.0
        assert cls.__prl_meta__["timestamp_field"] is None


class TestParseTimeOffset:
    def test_seconds(self) -> None:
        assert _parse_time_offset("10s") == 10.0

    def test_minutes(self) -> None:
        assert _parse_time_offset("5m") == 300.0

    def test_hours(self) -> None:
        assert _parse_time_offset("1h") == 3600.0

    def test_composite(self) -> None:
        assert _parse_time_offset("1h30m") == 5400.0

    def test_empty_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            _parse_time_offset("")

    def test_no_unit_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError):
            _parse_time_offset("30")


class TestHasTag:
    def test_tag_present(self) -> None:
        fd = _field("f", "key")
        assert _has_tag(fd, "key") is True

    def test_tag_absent(self) -> None:
        fd = _field("f")
        assert _has_tag(fd, "key") is False

    def test_other_tag_not_matched(self) -> None:
        fd = _field("f", "timestamp")
        assert _has_tag(fd, "key") is False


# ===========================================================================
# Accumulate compiler helpers
# ===========================================================================


@dataclass
class _Order:
    amount: float


def _acc_expr(
    type_name="Order",
    constraints=None,
    result_var="$total",
    function="sum",
    bind_var="$amount",
    constraint=None,
) -> AccumulateExpr:
    inner = PatternNode(
        type_name=type_name,
        constraints=tuple(constraints or [BindConstraint("$amount", "amount")]),
        exists=False,
        fact_var=None,
        negated=False,
    )
    return AccumulateExpr(
        inner=inner,
        result_var=result_var,
        function=function,
        bind_var=bind_var,
        constraint=constraint,
    )


_ACC_TYPES = {"Order": _Order}


class TestAccumulateFns:
    def test_sum_fn(self) -> None:
        assert _ACCUMULATE_FNS["sum"]([1, 2, 3]) == 6

    def test_count_fn(self) -> None:
        assert _ACCUMULATE_FNS["count"]([1, 2, 3]) == 3

    def test_min_fn(self) -> None:
        assert _ACCUMULATE_FNS["min"]([3, 1, 2]) == 1

    def test_max_fn(self) -> None:
        assert _ACCUMULATE_FNS["max"]([3, 1, 2]) == 3

    def test_collect_list_fn(self) -> None:
        assert _ACCUMULATE_FNS["collectList"]([1, 2]) == [1, 2]

    def test_min_empty_returns_none(self) -> None:
        assert _ACCUMULATE_FNS["min"]([]) is None

    def test_max_empty_returns_none(self) -> None:
        assert _ACCUMULATE_FNS["max"]([]) is None


class TestAccumulateCompiler:
    def test_returns_accumulate_spec(self) -> None:
        spec = _compile_accumulate(_acc_expr(), _ACC_TYPES)
        assert isinstance(spec, AccumulateSpec)

    def test_inner_pattern_type(self) -> None:
        spec = _compile_accumulate(_acc_expr(), _ACC_TYPES)
        assert spec.inner.type_ is _Order

    def test_bind_attr_resolved(self) -> None:
        spec = _compile_accumulate(_acc_expr(), _ACC_TYPES)
        assert spec.bind_attr == "amount"

    def test_result_var_stored(self) -> None:
        spec = _compile_accumulate(_acc_expr(), _ACC_TYPES)
        assert spec.result_var == "$total"

    def test_count_bind_attr_none(self) -> None:
        spec = _compile_accumulate(
            _acc_expr(constraints=[], bind_var=None, function="count"), _ACC_TYPES
        )
        assert spec.bind_attr is None

    def test_constraint_none_when_absent(self) -> None:
        spec = _compile_accumulate(_acc_expr(), _ACC_TYPES)
        assert spec.constraint is None

    def test_constraint_compiled(self) -> None:
        acc = _acc_expr(constraint=CompareConstraint("$total", ">", 100))
        spec = _compile_accumulate(acc, _ACC_TYPES)
        assert spec.constraint is not None
        assert spec.constraint(150) is True
        assert spec.constraint(50) is False

    def test_unknown_bind_var_raises(self) -> None:
        acc = _acc_expr(bind_var="$missing")
        with pytest.raises(NameError):
            _compile_accumulate(acc, _ACC_TYPES)
