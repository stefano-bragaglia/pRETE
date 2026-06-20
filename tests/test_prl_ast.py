"""Unit tests for PRL AST dataclasses (``src/rete/prl_ast.py``).

Tests are committed before the implementation and are expected to fail
with an ``ImportError`` until ``prl_ast.py`` is created.
"""
from __future__ import annotations

import pytest

from rete.prl_ast import (
    BindConstraint,
    CompareConstraint,
    DeclareDecl,
    FieldDecl,
    NccPatternGroup,
    PatternNode,
    ProgramNode,
    RuleDecl,
)


# ===========================================================================
# FieldDecl
# ===========================================================================

class TestFieldDecl:
    """``FieldDecl`` is a frozen, equality-comparable, hashable dataclass."""

    def test_construction(self) -> None:
        fd = FieldDecl("value", "double")
        assert fd.name == "value"
        assert fd.type_name == "double"

    def test_frozen(self) -> None:
        fd = FieldDecl("name", "String")
        with pytest.raises(AttributeError):
            fd.name = "other"  # type: ignore[misc]

    def test_structural_equality(self) -> None:
        assert FieldDecl("age", "int") == FieldDecl("age", "int")
        assert FieldDecl("age", "int") != FieldDecl("age", "long")

    def test_hashable(self) -> None:
        fd = FieldDecl("age", "int")
        assert hash(fd) == hash(FieldDecl("age", "int"))


# ===========================================================================
# DeclareDecl
# ===========================================================================

class TestDeclareDecl:
    """``DeclareDecl`` wraps a name and a tuple of ``FieldDecl`` instances."""

    def test_construction_no_fields(self) -> None:
        dd = DeclareDecl("Marker", ())
        assert dd.name == "Marker"
        assert dd.fields == ()

    def test_construction_with_fields(self) -> None:
        fields = (FieldDecl("value", "double"),)
        dd = DeclareDecl("Temp", fields)
        assert len(dd.fields) == 1

    def test_frozen(self) -> None:
        dd = DeclareDecl("Marker", ())
        with pytest.raises(AttributeError):
            dd.name = "Other"  # type: ignore[misc]

    def test_structural_equality(self) -> None:
        fields = (FieldDecl("v", "int"),)
        assert DeclareDecl("T", fields) == DeclareDecl("T", fields)
        assert DeclareDecl("T", fields) != DeclareDecl("X", fields)


# ===========================================================================
# BindConstraint
# ===========================================================================

class TestBindConstraint:
    """``BindConstraint`` represents ``$var: field_path`` inside a pattern."""

    def test_construction(self) -> None:
        bc = BindConstraint("$v", "value")
        assert bc.var == "$v"
        assert bc.field == "value"

    def test_frozen(self) -> None:
        bc = BindConstraint("$v", "value")
        with pytest.raises(AttributeError):
            bc.var = "$x"  # type: ignore[misc]

    def test_structural_equality(self) -> None:
        assert BindConstraint("$v", "value") == BindConstraint("$v", "value")
        assert BindConstraint("$v", "value") != BindConstraint("$x", "value")

    def test_hashable(self) -> None:
        bc = BindConstraint("$v", "value")
        assert hash(bc) == hash(BindConstraint("$v", "value"))


# ===========================================================================
# CompareConstraint
# ===========================================================================

class TestCompareConstraint:
    """``CompareConstraint`` stores field, operator, and a typed rhs value."""

    def test_rhs_string(self) -> None:
        cc = CompareConstraint("name", "==", "Alice")
        assert cc.rhs == "Alice"

    def test_rhs_int(self) -> None:
        cc = CompareConstraint("age", ">", 18)
        assert cc.rhs == 18

    def test_rhs_float(self) -> None:
        cc = CompareConstraint("value", ">=", 3.14)
        assert cc.rhs == 3.14

    def test_rhs_bool(self) -> None:
        cc = CompareConstraint("active", "==", True)
        assert cc.rhs is True

    def test_rhs_none(self) -> None:
        cc = CompareConstraint("owner", "==", None)
        assert cc.rhs is None

    def test_rhs_variable_reference(self) -> None:
        cc = CompareConstraint("block", "==", "$b")
        assert cc.rhs == "$b"

    def test_frozen(self) -> None:
        cc = CompareConstraint("age", ">", 18)
        with pytest.raises(AttributeError):
            cc.op = "<"  # type: ignore[misc]

    def test_structural_equality(self) -> None:
        a = CompareConstraint("age", ">", 18)
        b = CompareConstraint("age", ">", 18)
        assert a == b
        assert a != CompareConstraint("age", "<", 18)

    def test_hashable(self) -> None:
        cc = CompareConstraint("age", ">", 18)
        assert hash(cc) == hash(CompareConstraint("age", ">", 18))


# ===========================================================================
# PatternNode
# ===========================================================================

class TestPatternNode:
    """``PatternNode`` represents a single positive or negated pattern."""

    def test_minimal_construction(self) -> None:
        pn = PatternNode("Temp", None, (), False)
        assert pn.type_name == "Temp"
        assert pn.fact_var is None
        assert pn.constraints == ()
        assert pn.negated is False

    def test_with_fact_var(self) -> None:
        pn = PatternNode("Temp", "$t", (), False)
        assert pn.fact_var == "$t"

    def test_with_constraints(self) -> None:
        c = CompareConstraint("value", ">", 80)
        pn = PatternNode("Temp", None, (c,), False)
        assert len(pn.constraints) == 1

    def test_negated_flag(self) -> None:
        pn = PatternNode("Temp", None, (), True)
        assert pn.negated is True

    def test_frozen(self) -> None:
        pn = PatternNode("Temp", None, (), False)
        with pytest.raises(AttributeError):
            pn.type_name = "Other"  # type: ignore[misc]

    def test_structural_equality(self) -> None:
        a = PatternNode("Temp", None, (), False)
        b = PatternNode("Temp", None, (), False)
        assert a == b
        assert a != PatternNode("Temp", None, (), True)


# ===========================================================================
# NccPatternGroup
# ===========================================================================

class TestNccPatternGroup:
    """``NccPatternGroup`` holds patterns that must not jointly match."""

    def test_single_pattern(self) -> None:
        p = PatternNode("Marker", None, (), False)
        grp = NccPatternGroup((p,))
        assert len(grp.patterns) == 1

    def test_two_patterns(self) -> None:
        p1 = PatternNode("A", None, (), False)
        p2 = PatternNode("B", None, (), False)
        grp = NccPatternGroup((p1, p2))
        assert len(grp.patterns) == 2
        assert grp.patterns[0] is p1

    def test_frozen(self) -> None:
        grp = NccPatternGroup((PatternNode("X", None, (), False),))
        with pytest.raises(AttributeError):
            grp.patterns = ()  # type: ignore[misc]

    def test_structural_equality(self) -> None:
        p = PatternNode("T", None, (), False)
        assert NccPatternGroup((p,)) == NccPatternGroup((p,))
        assert NccPatternGroup((p,)) != NccPatternGroup(())


# ===========================================================================
# RuleDecl
# ===========================================================================

class TestRuleDecl:
    """``RuleDecl`` has three optional fields with sensible defaults."""

    def test_defaults(self) -> None:
        rd = RuleDecl("my-rule")
        assert rd.salience == 0
        assert rd.lhs == ()
        assert rd.rhs_src == ""

    def test_explicit_salience(self) -> None:
        rd = RuleDecl("r", salience=10)
        assert rd.salience == 10

    def test_explicit_lhs(self) -> None:
        p = PatternNode("T", None, (), False)
        rd = RuleDecl("r", lhs=(p,))
        assert len(rd.lhs) == 1

    def test_explicit_rhs_src(self) -> None:
        rd = RuleDecl("r", rhs_src="print('fired')\n")
        assert "fired" in rd.rhs_src

    def test_frozen(self) -> None:
        rd = RuleDecl("r")
        with pytest.raises(AttributeError):
            rd.name = "other"  # type: ignore[misc]

    def test_structural_equality(self) -> None:
        assert RuleDecl("r") == RuleDecl("r")
        assert RuleDecl("r") != RuleDecl("r", salience=5)


# ===========================================================================
# ProgramNode
# ===========================================================================

class TestProgramNode:
    """``ProgramNode`` is the parse-tree root; both fields are required."""

    def test_empty_program(self) -> None:
        pn = ProgramNode((), ())
        assert pn.declares == ()
        assert pn.rules == ()

    def test_with_declare(self) -> None:
        dd = DeclareDecl("T", ())
        pn = ProgramNode((dd,), ())
        assert len(pn.declares) == 1

    def test_with_rule(self) -> None:
        rd = RuleDecl("r")
        pn = ProgramNode((), (rd,))
        assert len(pn.rules) == 1

    def test_frozen(self) -> None:
        pn = ProgramNode((), ())
        with pytest.raises(AttributeError):
            pn.declares = (DeclareDecl("X", ()),)  # type: ignore[misc]

    def test_structural_equality(self) -> None:
        assert ProgramNode((), ()) == ProgramNode((), ())
        rd = RuleDecl("r")
        assert ProgramNode((), (rd,)) != ProgramNode((), ())
