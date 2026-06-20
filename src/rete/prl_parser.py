"""Recursive-descent parser for pRETE Rule Language (PRL).

Converts a flat list of :class:`~rete.prl_lexer.Tok` instances produced by
:func:`~rete.prl_lexer.tokenize` into a :class:`~rete.prl_ast.ProgramNode`.
"""
from __future__ import annotations

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
from rete.prl_lexer import Tok

__all__ = ["parse"]

_NUM_PARSERS: dict[str, type] = {"INT": int, "FLOAT": float}

_BOOL_LITERALS: dict[str, bool | None] = {
    "true": True,
    "false": False,
    "null": None,
    "None": None,
}


class Parser:
    """Token-stream cursor with one private method per grammar production."""

    def __init__(self, toks: list[Tok]) -> None:
        self._toks = toks
        self._pos = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self) -> ProgramNode:
        """Parse the full token stream and return the root AST node."""
        return self._parse_program()

    # ------------------------------------------------------------------
    # Top-level structure
    # ------------------------------------------------------------------

    def _parse_program(self) -> ProgramNode:
        declares: list[DeclareDecl] = []
        rules: list[RuleDecl] = []
        while not self._at_end():
            self._parse_top_level(declares, rules)
        return ProgramNode(tuple(declares), tuple(rules))

    def _parse_top_level(
        self,
        declares: list[DeclareDecl],
        rules: list[RuleDecl],
    ) -> None:
        if self._peek_kw("package"):
            self._skip_package()
        elif self._peek_kw("declare"):
            declares.append(self._parse_declare())
        elif self._peek_kw("rule"):
            rules.append(self._parse_rule())
        else:
            t = self._peek()
            raise SyntaxError(
                f"Expected 'package', 'declare', or 'rule' at line "
                f"{getattr(t, 'line', '?')}, got {getattr(t, 'value', t)!r}"
            )

    def _skip_package(self) -> None:
        self._advance()  # consume 'package'
        self._parse_field_path()  # consume qualified name
        if self._peek_punct(";"):
            self._advance()

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def _parse_declare(self) -> DeclareDecl:
        self._expect("KW", "declare")
        name = self._expect("IDENT").value
        fields: list[FieldDecl] = []
        while not self._peek_kw("end"):
            fields.append(self._parse_field())
        self._expect("KW", "end")
        return DeclareDecl(name, tuple(fields))

    def _parse_field(self) -> FieldDecl:
        name = self._expect("IDENT").value
        self._expect("PUNCT", ":")
        type_name = self._parse_type_ref()
        return FieldDecl(name, type_name)

    def _parse_type_ref(self) -> str:
        name = self._expect("IDENT").value
        if self._peek_op("<"):
            self._skip_generic()
        return name

    def _skip_generic(self) -> None:
        self._advance()  # consume '<'
        depth = 1
        while depth > 0:
            t = self._advance()
            if t.value == "<":
                depth += 1
            elif t.value == ">":
                depth -= 1

    # ------------------------------------------------------------------
    # Rules and attributes
    # ------------------------------------------------------------------

    def _parse_rule(self) -> RuleDecl:
        self._expect("KW", "rule")
        name = self._expect("STRING").value[1:-1]
        salience = self._parse_rule_attrs()
        self._expect("KW", "when")
        lhs = self._parse_lhs()
        self._expect("KW", "then")
        rhs_src = self._expect("RAWBLOCK").value
        self._expect("KW", "end")
        return RuleDecl(name, salience, lhs, rhs_src)

    def _parse_rule_attrs(self) -> int:
        salience = 0
        while not self._peek_kw("when"):
            salience = self._parse_one_attr(salience)
        return salience

    def _parse_one_attr(self, salience: int) -> int:
        if self._peek_kw("salience"):
            self._advance()
            return self._parse_int()
        if self._peek_kw("no-loop"):
            self._advance()
            self._try_bool()
            return salience
        t = self._peek()
        raise SyntaxError(
            f"Unknown rule attribute at line {getattr(t, 'line', '?')}: "
            f"{getattr(t, 'value', t)!r}"
        )

    def _try_bool(self) -> None:
        t = self._peek()
        if t and t.kind == "KW" and t.value in ("true", "false"):
            self._advance()

    # ------------------------------------------------------------------
    # LHS and conditions
    # ------------------------------------------------------------------

    def _parse_lhs(self) -> tuple[PatternNode | NccPatternGroup, ...]:
        conds: list[PatternNode | NccPatternGroup] = []
        while not self._peek_kw("then"):
            conds.append(self._parse_condition())
        return tuple(conds)

    def _parse_condition(self) -> PatternNode | NccPatternGroup:
        if self._peek_kw("not"):
            return self._parse_negated()
        return self._parse_pattern()

    def _parse_negated(self) -> PatternNode | NccPatternGroup:
        self._expect("KW", "not")
        if self._peek_punct("("):
            return self._parse_ncc()
        return self._parse_pattern(negated=True)

    def _parse_ncc(self) -> NccPatternGroup:
        self._expect("PUNCT", "(")
        patterns: list[PatternNode] = []
        while not self._peek_punct(")"):
            patterns.append(self._parse_pattern())
        if not patterns:
            raise SyntaxError("NCC group requires at least one pattern")
        self._expect("PUNCT", ")")
        return NccPatternGroup(tuple(patterns))

    # ------------------------------------------------------------------
    # Patterns
    # ------------------------------------------------------------------

    def _parse_pattern(self, negated: bool = False) -> PatternNode:
        fact_var = self._try_fact_binding()
        if self._peek_punct("/"):
            return self._parse_oopath(fact_var, negated)
        return self._parse_traditional(fact_var, negated)

    def _try_fact_binding(self) -> str | None:
        t = self._peek()
        if t is None or t.kind not in ("VAR", "IDENT"):
            return None
        nxt = self._peek_at(1)
        if nxt is None or not self._is_colon(nxt):
            return None
        self._advance()
        self._advance()
        return t.value

    def _is_colon(self, t: Tok) -> bool:
        return t.kind == "PUNCT" and t.value == ":"

    def _parse_oopath(
        self, fact_var: str | None, negated: bool
    ) -> PatternNode:
        self._expect("PUNCT", "/")
        type_name = self._expect("IDENT").value
        constraints: tuple[BindConstraint | CompareConstraint, ...] = ()
        if self._peek_punct("["):
            self._advance()
            constraints = self._parse_constraints("]")
            self._expect("PUNCT", "]")
        return PatternNode(type_name, fact_var, constraints, negated)

    def _parse_traditional(
        self, fact_var: str | None, negated: bool
    ) -> PatternNode:
        type_name = self._expect("IDENT").value
        self._expect("PUNCT", "(")
        constraints: tuple[BindConstraint | CompareConstraint, ...] = ()
        if not self._peek_punct(")"):
            constraints = self._parse_constraints(")")
        self._expect("PUNCT", ")")
        return PatternNode(type_name, fact_var, constraints, negated)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _parse_constraints(
        self, close: str
    ) -> tuple[BindConstraint | CompareConstraint, ...]:
        clist: list[BindConstraint | CompareConstraint] = []
        while not self._peek_punct(close):
            clist.append(self._parse_constraint())
            if self._peek_punct(","):
                self._advance()
        return tuple(clist)

    def _parse_constraint(self) -> BindConstraint | CompareConstraint:
        if self._peek_kind("VAR"):
            return self._parse_bind()
        return self._parse_compare()

    def _parse_bind(self) -> BindConstraint:
        var = self._expect("VAR").value
        self._expect("PUNCT", ":")
        field = self._parse_field_path()
        return BindConstraint(var, field)

    def _parse_compare(self) -> CompareConstraint:
        field = self._parse_field_path()
        op = self._expect("OP").value
        rhs = self._parse_value()
        return CompareConstraint(field, op, rhs)

    def _parse_field_path(self) -> str:
        parts = [self._expect("IDENT").value]
        while self._peek_punct("."):
            self._advance()
            parts.append(self._expect("IDENT").value)
        return ".".join(parts)

    # ------------------------------------------------------------------
    # Values
    # ------------------------------------------------------------------

    def _parse_value(self) -> str | int | float | bool | None:
        t = self._peek()
        if t is None:
            raise SyntaxError("Expected a value but reached end of input")
        if t.kind == "VAR":
            return self._advance().value
        if t.kind == "STRING":
            return self._advance().value[1:-1]
        return self._parse_literal()

    def _parse_literal(self) -> int | float | bool | None:
        t = self._peek()
        if t and t.kind == "KW" and t.value in _BOOL_LITERALS:
            self._advance()
            return _BOOL_LITERALS[t.value]
        return self._parse_number()

    def _parse_number(self) -> int | float:
        neg = self._peek_punct("-")
        if neg:
            self._advance()
        return self._parse_unsigned(neg)

    def _parse_unsigned(self, neg: bool) -> int | float:
        t = self._peek()
        if t is None or t.kind not in _NUM_PARSERS:
            raise SyntaxError(
                f"Expected a number at line {getattr(t, 'line', '?')}, "
                f"got {getattr(t, 'kind', t)!r}"
            )
        self._advance()
        val = _NUM_PARSERS[t.kind](t.value)
        return -val if neg else val

    def _parse_int(self) -> int:
        neg = self._peek_punct("-")
        if neg:
            self._advance()
        tok = self._expect("INT")
        val = int(tok.value)
        return -val if neg else val

    # ------------------------------------------------------------------
    # Token utilities
    # ------------------------------------------------------------------

    def _at_end(self) -> bool:
        return self._pos >= len(self._toks)

    def _peek(self) -> Tok | None:
        return self._toks[self._pos] if not self._at_end() else None

    def _peek_at(self, offset: int) -> Tok | None:
        idx = self._pos + offset
        return self._toks[idx] if idx < len(self._toks) else None

    def _peek_kind(self, kind: str) -> bool:
        t = self._peek()
        return t is not None and t.kind == kind

    def _peek_kw(self, kw: str) -> bool:
        t = self._peek()
        return t is not None and t.kind == "KW" and t.value == kw

    def _peek_punct(self, ch: str) -> bool:
        t = self._peek()
        return t is not None and t.kind == "PUNCT" and t.value == ch

    def _peek_op(self, op: str) -> bool:
        t = self._peek()
        return t is not None and t.kind == "OP" and t.value == op

    def _advance(self) -> Tok:
        t = self._toks[self._pos]
        self._pos += 1
        return t

    def _expect(self, kind: str, value: str | None = None) -> Tok:
        t = self._peek()
        if t is None:
            raise SyntaxError(
                f"Expected {kind!r} but reached end of input"
            )
        if t.kind != kind:
            raise SyntaxError(
                f"Expected {kind!r} at line {t.line}, "
                f"got {t.kind!r} {t.value!r}"
            )
        if value is not None and t.value != value:
            raise SyntaxError(
                f"Expected {value!r} at line {t.line}, got {t.value!r}"
            )
        return self._advance()


def parse(toks: list[Tok]) -> ProgramNode:
    """Parse a PRL token list into a :class:`~rete.prl_ast.ProgramNode`.

    :param toks: token list produced by :func:`~rete.prl_lexer.tokenize`.
    :returns: root AST node.
    :raises SyntaxError: on any unexpected token or missing delimiter.
    """
    return Parser(toks).parse()
