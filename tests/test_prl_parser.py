"""Unit tests for the PRL recursive-descent parser (``src/rete/prl_parser.py``).

All tests call ``parse(tokenize(text))`` and assert on the resulting AST
structure.  They are committed before the implementation and expected to fail
with ``ImportError`` until ``prl_parser.py`` is created.
"""
from __future__ import annotations

import pytest

from rete.prl_ast import (
    BindConstraint,
    CompareConstraint,
    ForallNode,
    NamedConstraint,
    NccPatternGroup,
    OrGroup,
    PatternNode,
    PositionalConstraint,
    ProgramNode,
    RuleDecl,
    Tag,
)
from rete.prl_lexer import tokenize
from rete.prl_parser import parse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(src: str) -> ProgramNode:
    """Tokenize and parse *src*; return the root ProgramNode."""
    return parse(tokenize(src))


def _first_rule(src: str) -> RuleDecl:
    """Return the first RuleDecl from *src*."""
    return _parse(src).rules[0]


def _rule_with_lhs(lhs_src: str) -> RuleDecl:
    """Parse a minimal rule whose LHS is *lhs_src* and RHS is ``pass``."""
    return _first_rule(f'rule "r" when\n{lhs_src}\nthen\npass\nend')


def _lhs_cond(lhs_src: str) -> PatternNode | NccPatternGroup:
    """Return the first condition from a rule with the given LHS source."""
    return _rule_with_lhs(lhs_src).lhs[0]


def _first_constraint(pattern_src: str) -> BindConstraint | CompareConstraint:
    """Return the first constraint parsed from a single-pattern LHS."""
    pat = _lhs_cond(pattern_src)
    return pat.constraints[0]  # type: ignore[union-attr]


# ===========================================================================
# Program-level
# ===========================================================================

class TestParseProgram:
    """Top-level parse produces the correct ProgramNode."""

    def test_empty_source(self) -> None:
        prog = _parse("")
        assert prog.declares == ()
        assert prog.rules == ()

    def test_package_silently_consumed(self) -> None:
        prog = _parse("package org.example;")
        assert prog.declares == ()
        assert prog.rules == ()

    def test_declare_and_rule_both_present(self) -> None:
        src = "declare Marker\nend\n" + 'rule "r" when\nthen\npass\nend'
        prog = _parse(src)
        assert len(prog.declares) == 1
        assert len(prog.rules) == 1


# ===========================================================================
# Package declaration
# ===========================================================================

class TestParsePackage:
    """Package declarations are consumed without producing a node."""

    def test_with_semicolon(self) -> None:
        assert _parse("package org;").rules == ()

    def test_without_semicolon(self) -> None:
        assert _parse("package org").rules == ()

    def test_dotted_name(self) -> None:
        assert _parse("package org.mortgages.rules").rules == ()


# ===========================================================================
# Declare declarations
# ===========================================================================

class TestParseDeclare:
    """``declare`` blocks become DeclareDecl nodes."""

    def test_no_fields(self) -> None:
        dd = _parse("declare Marker\nend").declares[0]
        assert dd.name == "Marker"
        assert dd.fields == ()
        assert dd.extends is None

    def test_one_field(self) -> None:
        fd = _parse("declare Temp\n  value: float\nend").declares[0].fields[0]
        assert fd.name == "value"
        assert fd.type_name == "float"

    def test_two_fields_count(self) -> None:
        fields = _parse(
            "declare Temp\n  sensor: str\n  value: float\nend"
        ).declares[0].fields
        assert len(fields) == 2
        assert fields[0].name == "sensor"

    def test_type_name_stored_verbatim(self) -> None:
        dd = _parse("declare Temp\n  value: float\nend").declares[0]
        assert dd.fields[0].type_name == "float"


# ===========================================================================
# Rule declaration
# ===========================================================================

class TestParseRule:
    """Rule declarations produce correct RuleDecl nodes."""

    def test_name_stripped_of_quotes(self) -> None:
        rd = _first_rule('rule "too-hot" when\nthen\npass\nend')
        assert rd.name == "too-hot"

    def test_default_salience_is_zero(self) -> None:
        rd = _first_rule('rule "r" when\nthen\npass\nend')
        assert rd.salience == 0

    def test_empty_lhs_is_empty_tuple(self) -> None:
        rd = _first_rule('rule "r" when\nthen\npass\nend')
        assert rd.lhs == ()

    def test_rhs_src_preserved_verbatim(self) -> None:
        rd = _first_rule('rule "r" when\nthen\n  x = 1\nend')
        assert "x = 1" in rd.rhs_src


# ===========================================================================
# Rule attributes
# ===========================================================================

class TestParseRuleAttrs:
    """``salience`` and ``no-loop`` are parsed correctly."""

    def test_salience(self) -> None:
        rd = _first_rule('rule "r"\n  salience 10\n  when\nthen\npass\nend')
        assert rd.salience == 10

    def test_negative_salience(self) -> None:
        rd = _first_rule('rule "r"\n  salience -5\n  when\nthen\npass\nend')
        assert rd.salience == -5

    def test_no_loop_does_not_change_salience(self) -> None:
        rd = _first_rule('rule "r"\n  no-loop\n  when\nthen\npass\nend')
        assert rd.salience == 0
        assert rd.no_loop is True

    def test_no_loop_with_explicit_true(self) -> None:
        rd = _first_rule('rule "r"\n  no-loop true\n  when\nthen\npass\nend')
        assert rd.salience == 0
        assert rd.no_loop is True

    def test_salience_and_no_loop_together(self) -> None:
        rd = _first_rule(
            'rule "r"\n  salience 5\n  no-loop\n  when\nthen\npass\nend'
        )
        assert rd.salience == 5
        assert rd.no_loop is True

    def test_no_loop_false_by_default(self) -> None:
        rd = _first_rule('rule "r" when\nthen\npass\nend')
        assert rd.no_loop is False


# ===========================================================================
# LHS (when-block)
# ===========================================================================

class TestParseLhs:
    """The when-block parses into a tuple of conditions."""

    def test_empty_when_block(self) -> None:
        rd = _first_rule('rule "r" when\nthen\npass\nend')
        assert rd.lhs == ()

    def test_one_oopath_condition(self) -> None:
        assert len(_rule_with_lhs("/Temp[value > 80]").lhs) == 1

    def test_one_traditional_condition(self) -> None:
        assert len(_rule_with_lhs("Temp(value > 80)").lhs) == 1

    def test_two_conditions(self) -> None:
        assert len(_rule_with_lhs("/A[]\n  /B[]").lhs) == 2


# ===========================================================================
# OOPath patterns
# ===========================================================================

class TestParseOoPath:
    """``/TypeName[constraints]`` patterns parse correctly."""

    def test_no_constraints(self) -> None:
        pat = _lhs_cond("/Temp")
        assert isinstance(pat, PatternNode)
        assert pat.type_name == "Temp"
        assert pat.constraints == ()

    def test_with_one_constraint(self) -> None:
        pat = _lhs_cond("/Temp[value > 80]")
        assert isinstance(pat, PatternNode)
        assert len(pat.constraints) == 1

    def test_with_fact_binding(self) -> None:
        pat = _lhs_cond("$t: /Temp[value > 80]")
        assert isinstance(pat, PatternNode)
        assert pat.fact_var == "$t"

    def test_bare_identifier_fact_binding(self) -> None:
        pat = _lhs_cond("t: /Temp[value > 80]")
        assert isinstance(pat, PatternNode)
        assert pat.fact_var == "t"

    def test_not_negated_by_default(self) -> None:
        pat = _lhs_cond("/Temp")
        assert isinstance(pat, PatternNode)
        assert pat.negated is False


# ===========================================================================
# Traditional patterns
# ===========================================================================

class TestParseTraditional:
    """``TypeName(constraints)`` patterns parse correctly."""

    def test_no_constraints(self) -> None:
        pat = _lhs_cond("Temp()")
        assert isinstance(pat, PatternNode)
        assert pat.type_name == "Temp"
        assert pat.constraints == ()

    def test_with_one_constraint(self) -> None:
        pat = _lhs_cond("Temp(value > 80)")
        assert isinstance(pat, PatternNode)
        assert len(pat.constraints) == 1

    def test_with_fact_binding(self) -> None:
        pat = _lhs_cond("$t: Temp(value > 80)")
        assert isinstance(pat, PatternNode)
        assert pat.fact_var == "$t"


# ===========================================================================
# Negation
# ===========================================================================

class TestParseNegation:
    """``not pattern`` sets ``negated=True`` on the PatternNode."""

    def test_negated_oopath(self) -> None:
        pat = _lhs_cond("not /Temp[value < 0]")
        assert isinstance(pat, PatternNode)
        assert pat.negated is True
        assert pat.type_name == "Temp"

    def test_negated_traditional(self) -> None:
        pat = _lhs_cond("not Temp(value < 0)")
        assert isinstance(pat, PatternNode)
        assert pat.negated is True


# ===========================================================================
# Negated conjunctive conditions (NCC)
# ===========================================================================

class TestParseNcc:
    """``not ( pattern+ )`` produces an NccPatternGroup."""

    def test_ncc_two_patterns(self) -> None:
        cond = _lhs_cond("not ( /A[] /B[] )")
        assert isinstance(cond, NccPatternGroup)
        assert len(cond.patterns) == 2

    def test_ncc_pattern_type_names(self) -> None:
        cond = _lhs_cond("not ( /A[] /B[] )")
        assert isinstance(cond, NccPatternGroup)
        assert cond.patterns[0].type_name == "A"
        assert cond.patterns[1].type_name == "B"

    def test_empty_ncc_raises(self) -> None:
        with pytest.raises(SyntaxError):
            _lhs_cond("not ()")


# ===========================================================================
# Constraints
# ===========================================================================

class TestParseConstraints:
    """Bind and compare constraints inside patterns."""

    def test_bind_constraint_fields(self) -> None:
        c = _first_constraint("/T[$v: value]")
        assert isinstance(c, BindConstraint)
        assert c.var == "$v"
        assert c.field == "value"

    def test_compare_int(self) -> None:
        c = _first_constraint("/T[value > 80]")
        assert isinstance(c, CompareConstraint)
        assert c.op == ">"
        assert c.rhs == 80

    def test_compare_float(self) -> None:
        c = _first_constraint("/T[value >= 3.14]")
        assert isinstance(c, CompareConstraint)
        assert c.rhs == pytest.approx(3.14)

    def test_compare_string(self) -> None:
        c = _first_constraint('/T[name == "Alice"]')
        assert isinstance(c, CompareConstraint)
        assert c.rhs == "Alice"

    def test_compare_bool_true(self) -> None:
        c = _first_constraint("/T[active == True]")
        assert isinstance(c, CompareConstraint)
        assert c.rhs is True

    def test_compare_none(self) -> None:
        c = _first_constraint("/T[owner == None]")
        assert isinstance(c, CompareConstraint)
        assert c.rhs is None

    def test_compare_variable_reference(self) -> None:
        c = _first_constraint("/T[block == $b]")
        assert isinstance(c, CompareConstraint)
        assert c.rhs == "$b"

    def test_compare_negative_int(self) -> None:
        c = _first_constraint("/T[value > -1]")
        assert isinstance(c, CompareConstraint)
        assert c.rhs == -1

    def test_two_constraints(self) -> None:
        pat = _lhs_cond("/T[a > 1, b < 2]")
        assert isinstance(pat, PatternNode)
        assert len(pat.constraints) == 2

    def test_dotted_field_path(self) -> None:
        c = _first_constraint('/T[address.city == "NYC"]')
        assert isinstance(c, CompareConstraint)
        assert c.field == "address.city"


# ===========================================================================
# Error cases
# ===========================================================================

# ===========================================================================
# extends
# ===========================================================================

class TestParseDeclareExtends:
    """``extends`` clause in declare blocks is parsed into ``DeclareDecl.extends``."""

    def test_extends_sets_field(self) -> None:
        dd = _parse("declare Dog extends Animal\nend").declares[0]
        assert dd.extends == "Animal"

    def test_extends_name_stored(self) -> None:
        dd = _parse("declare Dog extends Animal\n  breed: str\nend").declares[0]
        assert dd.name == "Dog"
        assert dd.extends == "Animal"

    def test_no_extends_is_none(self) -> None:
        dd = _parse("declare Animal\n  name: str\nend").declares[0]
        assert dd.extends is None

    def test_extends_missing_parent_raises(self) -> None:
        with pytest.raises(SyntaxError):
            _parse("declare Dog extends\nend")


# ===========================================================================
# Error cases
# ===========================================================================

class TestParseErrors:
    """Malformed PRL raises SyntaxError."""

    def test_unknown_rule_attribute(self) -> None:
        with pytest.raises(SyntaxError):
            _first_rule('rule "r"\n  enabled\n  when\nthen\npass\nend')

    def test_unrecognised_top_level_token(self) -> None:
        with pytest.raises(SyntaxError):
            _parse("42")


# ===========================================================================
# Tag parsing (ES-2)
# ===========================================================================

class TestParseTags:
    """Tags (``@name`` / ``@name(value)``) attach to declare, field, and rule nodes."""

    def test_tag_no_value_on_declare(self) -> None:
        dd = _parse("@timestamp\ndeclare E\nend").declares[0]
        assert dd.tags == (Tag("timestamp"),)

    def test_tag_with_value_on_declare(self) -> None:
        dd = _parse("@role(event)\ndeclare E\nend").declares[0]
        assert dd.tags == (Tag("role", "event"),)

    def test_multiple_tags_on_declare(self) -> None:
        dd = _parse("@role(event)\n@expires(30s)\ndeclare E\nend").declares[0]
        assert dd.tags == (Tag("role", "event"), Tag("expires", "30s"))

    def test_no_tags_gives_empty_tuple_on_declare(self) -> None:
        dd = _parse("declare D\n  x: int\nend").declares[0]
        assert dd.tags == ()

    def test_field_tag_no_value(self) -> None:
        fd = _parse("declare D\n  @key\n  id: int\nend").declares[0].fields[0]
        assert fd.name == "id"
        assert fd.tags == (Tag("key"),)

    def test_field_tag_with_value(self) -> None:
        fd = _parse("declare D\n  @custom(foo)\n  x: int\nend").declares[0].fields[0]
        assert fd.tags == (Tag("custom", "foo"),)

    def test_field_without_tag_has_empty_tags(self) -> None:
        fd = _parse("declare D\n  x: int\nend").declares[0].fields[0]
        assert fd.tags == ()

    def test_multiple_fields_with_mixed_tags(self) -> None:
        src = "declare D\n  @key\n  id: int\n  name: str\nend"
        fields = _parse(src).declares[0].fields
        assert fields[0].tags == (Tag("key"),)
        assert fields[1].tags == ()

    def test_no_loop_tag_on_rule_stored_in_tags(self) -> None:
        rd = _first_rule('@no-loop\nrule "r" when\nthen\npass\nend')
        assert Tag("no-loop") in rd.tags

    def test_no_loop_tag_does_not_set_no_loop_field(self) -> None:
        """Parser stores @no-loop in tags; the compiler sets Production.no_loop."""
        rd = _first_rule('@no-loop\nrule "r" when\nthen\npass\nend')
        assert rd.no_loop is False  # attribute form only; compiler combines both

    def test_unknown_tag_stored_on_rule(self) -> None:
        rd = _first_rule('@future_feature\nrule "r" when\nthen\npass\nend')
        assert any(t.name == "future_feature" for t in rd.tags)

    def test_no_loop_kw_tag_name_parsed(self) -> None:
        """``@no-loop`` tag name is lexed as KW; parser must accept it."""
        rd = _first_rule('@no-loop\nrule "r" when\nthen\npass\nend')
        assert rd.tags[0].name == "no-loop"


# ===========================================================================
# Shorthand constraints (ES-4)
# ===========================================================================

class TestParseShorthandConstraints:
    """Positional and named-keyword constraint forms."""

    def test_positional_single_int(self) -> None:
        c = _first_constraint("Point(0)")
        assert isinstance(c, PositionalConstraint)
        assert c.value == 0

    def test_positional_two_values(self) -> None:
        pat = _lhs_cond("Point(0, 0)")
        assert len(pat.constraints) == 2
        assert all(isinstance(c, PositionalConstraint) for c in pat.constraints)

    def test_positional_string_value(self) -> None:
        c = _first_constraint('Order("open")')
        assert isinstance(c, PositionalConstraint)
        assert c.value == "open"

    def test_positional_variable_reference(self) -> None:
        c = _first_constraint("Point($x)")
        assert isinstance(c, PositionalConstraint)
        assert c.value == "$x"

    def test_named_single(self) -> None:
        c = _first_constraint("Point(y=0)")
        assert isinstance(c, NamedConstraint)
        assert c.field == "y"
        assert c.value == 0

    def test_named_string_value(self) -> None:
        c = _first_constraint('Order(status="open")')
        assert isinstance(c, NamedConstraint)
        assert c.value == "open"

    def test_mixed_positional_and_named(self) -> None:
        pat = _lhs_cond("Point(0, y=1)")
        assert isinstance(pat.constraints[0], PositionalConstraint)
        assert isinstance(pat.constraints[1], NamedConstraint)

    def test_named_and_bind_coexist(self) -> None:
        pat = _lhs_cond("Point($v: x, y=1)")
        assert isinstance(pat.constraints[0], BindConstraint)
        assert isinstance(pat.constraints[1], NamedConstraint)

    def test_named_and_compare_coexist(self) -> None:
        pat = _lhs_cond('Order(status="open", amount > 100)')
        assert isinstance(pat.constraints[0], NamedConstraint)
        assert isinstance(pat.constraints[1], CompareConstraint)

    def test_positional_negative_number(self) -> None:
        c = _first_constraint("Point(-1)")
        assert isinstance(c, PositionalConstraint)
        assert c.value == -1

    def test_positional_bool(self) -> None:
        c = _first_constraint("Flag(true)")
        assert isinstance(c, PositionalConstraint)
        assert c.value is True


# ===========================================================================
# Import declarations (ES-5)
# ===========================================================================

class TestParseImport:
    """``import`` and ``from … import`` at the top level."""

    def test_import_class(self) -> None:
        prog = _parse("import rete.fact.Fact")
        assert len(prog.imports) == 1
        assert prog.imports[0].names == (("rete.fact.Fact", "Fact"),)

    def test_import_class_with_alias(self) -> None:
        prog = _parse("import rete.fact.Fact as F")
        assert prog.imports[0].names == (("rete.fact.Fact", "F"),)

    def test_from_import_single(self) -> None:
        prog = _parse("from rete.fact import Fact")
        assert prog.imports[0].names == (("rete.fact.Fact", "Fact"),)

    def test_from_import_with_alias(self) -> None:
        prog = _parse("from rete.fact import Fact as F")
        assert prog.imports[0].names == (("rete.fact.Fact", "F"),)

    def test_from_import_multiple(self) -> None:
        prog = _parse("from rete.fact import Fact, Token")
        assert prog.imports[0].names == (
            ("rete.fact.Fact", "Fact"),
            ("rete.fact.Token", "Token"),
        )

    def test_from_import_multiple_with_aliases(self) -> None:
        prog = _parse("from rete.fact import Fact as F, Token as T")
        assert prog.imports[0].names == (
            ("rete.fact.Fact", "F"),
            ("rete.fact.Token", "T"),
        )

    def test_import_before_declare(self) -> None:
        src = "import rete.fact.Fact\ndeclare Marker\nend"
        prog = _parse(src)
        assert len(prog.imports) == 1
        assert len(prog.declares) == 1

    def test_no_imports_gives_empty_tuple(self) -> None:
        prog = _parse("declare Marker\nend")
        assert prog.imports == ()

    def test_multiple_import_stmts(self) -> None:
        src = "import rete.fact.Fact\nfrom rete.fact import Token"
        prog = _parse(src)
        assert len(prog.imports) == 2


# ===========================================================================
# or disjunction (ES-6)
# ===========================================================================

class TestParseOr:
    """``or`` keyword splits the LHS into an ``OrGroup``."""

    def test_two_branch_or_yields_or_group(self) -> None:
        src = 'rule "r"\nwhen\n  A() or\n  B()\nthen\nend'
        og = _parse(src).rules[0].lhs[0]
        assert isinstance(og, OrGroup)

    def test_two_branch_or_branch_count(self) -> None:
        src = 'rule "r"\nwhen\n  A() or\n  B()\nthen\nend'
        og = _parse(src).rules[0].lhs[0]
        assert len(og.branches) == 2

    def test_two_branch_or_type_names(self) -> None:
        src = 'rule "r"\nwhen\n  A() or\n  B()\nthen\nend'
        og = _parse(src).rules[0].lhs[0]
        assert og.branches[0][0].type_name == "A"
        assert og.branches[1][0].type_name == "B"

    def test_three_branch_or(self) -> None:
        src = 'rule "r"\nwhen\n  A() or\n  B() or\n  C()\nthen\nend'
        prog = _parse(src)
        og = prog.rules[0].lhs[0]
        assert isinstance(og, OrGroup)
        assert len(og.branches) == 3

    def test_multi_condition_branch(self) -> None:
        """Each branch may contain multiple conditions."""
        src = 'rule "r"\nwhen\n  A()\n  B() or\n  C()\nthen\nend'
        prog = _parse(src)
        og = prog.rules[0].lhs[0]
        assert isinstance(og, OrGroup)
        assert len(og.branches[0]) == 2
        assert len(og.branches[1]) == 1

    def test_no_or_gives_flat_tuple(self) -> None:
        src = 'rule "r"\nwhen\n  A()\nthen\nend'
        prog = _parse(src)
        lhs = prog.rules[0].lhs
        assert not any(isinstance(n, OrGroup) for n in lhs)
        assert lhs[0].type_name == "A"

    def test_or_with_fact_binding(self) -> None:
        src = 'rule "r"\nwhen\n  $x: A() or\n  $x: B()\nthen\nend'
        prog = _parse(src)
        og = prog.rules[0].lhs[0]
        assert og.branches[0][0].fact_var == "$x"
        assert og.branches[1][0].fact_var == "$x"

    def test_or_with_constraints(self) -> None:
        src = 'rule "r"\nwhen\n  A(x == 1) or\n  B(y == 2)\nthen\nend'
        prog = _parse(src)
        og = prog.rules[0].lhs[0]
        assert isinstance(og, OrGroup)
        assert len(og.branches[0][0].constraints) == 1

    def test_or_branches_are_tuples(self) -> None:
        src = 'rule "r"\nwhen\n  A() or\n  B()\nthen\nend'
        og = _parse(src).rules[0].lhs[0]
        assert isinstance(og.branches, tuple)
        assert all(isinstance(b, tuple) for b in og.branches)


# ===========================================================================
# forall (ES-6)
# ===========================================================================

class TestParseForall:
    """``forall(P, Q)`` produces a ``ForallNode``."""

    def test_basic_forall(self) -> None:
        src = 'rule "r"\nwhen\n  forall(Order(), Approval())\nthen\nend'
        prog = _parse(src)
        assert len(prog.rules[0].lhs) == 1
        fn = prog.rules[0].lhs[0]
        assert isinstance(fn, ForallNode)
        assert fn.pattern.type_name == "Order"
        assert fn.condition.type_name == "Approval"

    def test_forall_with_constraint_on_P(self) -> None:
        src = (
            'rule "r"\nwhen\n'
            '  forall(\n'
            '    Order(status == "pending"),\n'
            '    Approval()\n'
            '  )\nthen\nend'
        )
        prog = _parse(src)
        fn = prog.rules[0].lhs[0]
        assert isinstance(fn, ForallNode)
        c = fn.pattern.constraints[0]
        assert isinstance(c, CompareConstraint)
        assert c.op == "=="

    def test_forall_with_fact_binding_in_P(self) -> None:
        src = 'rule "r"\nwhen\n  forall($o: Order(), Approval())\nthen\nend'
        prog = _parse(src)
        fn = prog.rules[0].lhs[0]
        assert fn.pattern.fact_var == "$o"

    def test_forall_condition_has_no_fact_var_by_default(self) -> None:
        src = 'rule "r"\nwhen\n  forall(Order(), Approval())\nthen\nend'
        prog = _parse(src)
        fn = prog.rules[0].lhs[0]
        assert fn.condition.fact_var is None

    def test_missing_comma_raises(self) -> None:
        src = 'rule "r"\nwhen\n  forall(Order() Approval())\nthen\nend'
        with pytest.raises(SyntaxError):
            _parse(src)

    def test_forall_is_single_lhs_node(self) -> None:
        src = 'rule "r"\nwhen\n  forall(Order(), Approval())\nthen\nend'
        lhs = _parse(src).rules[0].lhs
        assert len(lhs) == 1
        assert isinstance(lhs[0], ForallNode)
