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
    ForallNode,
    ImportDecl,
    NamedConstraint,
    NccPatternGroup,
    OrGroup,
    PatternNode,
    PositionalConstraint,
    ProgramNode,
    RuleDecl,
    Tag,
)
from rete.prl_lexer import Tok

__all__ = ["parse"]

_NUM_PARSERS: dict[str, type] = {"INT": int, "FLOAT": float}

_BOOL_LITERALS: dict[str, bool | None] = {
    "True": True, "False": False,
    "true": True, "false": False,
    "None": None, "null": None,
}

_VALUE_KW: frozenset[str] = frozenset(_BOOL_LITERALS)

_VALUE_KINDS: frozenset[str] = frozenset({"INT", "FLOAT", "STRING"})


def _is_value_token(t: Tok | None) -> bool:
    """Return True if *t* starts a value literal (not a field path)."""
    if t is None:
        return False
    if t.kind in _VALUE_KINDS:
        return True
    return _is_minus_or_bool_kw(t)


def _is_minus_or_bool_kw(t: Tok) -> bool:
    """Return True if *t* is a unary minus or a boolean/null keyword."""
    if t.kind == "PUNCT" and t.value == "-":
        return True
    return t.kind == "KW" and t.value in _VALUE_KW


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
        imports: list[ImportDecl] = []
        declares: list[DeclareDecl] = []
        rules: list[RuleDecl] = []
        while not self._at_end():
            self._parse_top_level(imports, declares, rules)
        return ProgramNode(tuple(declares), tuple(rules), tuple(imports))

    def _parse_top_level(
        self,
        imports: list[ImportDecl],
        declares: list[DeclareDecl],
        rules: list[RuleDecl],
    ) -> None:
        tags = self._parse_tags()
        if self._peek_kw("package"):
            self._skip_package()
        elif self._peek_kw("declare"):
            declares.append(self._parse_declare(tags))
        elif self._peek_kw("rule"):
            rules.append(self._parse_rule(tags))
        elif self._try_import(imports):
            pass
        else:
            t = self._peek()
            raise SyntaxError(
                f"Expected 'package', 'declare', 'rule', 'import', or 'from' "
                f"at line {getattr(t, 'line', '?')}, got {getattr(t, 'value', t)!r}"
            )

    def _try_import(self, imports: list[ImportDecl]) -> bool:
        """Consume one import statement if present; return True when found."""
        if self._peek_kw("import"):
            imports.append(self._parse_import_stmt())
            return True
        if self._peek_kw("from"):
            imports.append(self._parse_from_stmt())
            return True
        return False

    def _parse_import_stmt(self) -> ImportDecl:
        """Parse ``import a.b.ClassName [as Alias]``."""
        self._expect("KW", "import")
        qualified = self._parse_field_path()
        alias = self._parse_optional_as() or qualified.rpartition(".")[2]
        return ImportDecl(((qualified, alias),))

    def _parse_from_stmt(self) -> ImportDecl:
        """Parse ``from a.b import Name [as Alias] [, ...]``."""
        self._expect("KW", "from")
        module = self._parse_field_path()
        self._expect("KW", "import")
        return ImportDecl(self._parse_import_names(module))

    def _parse_import_names(self, module: str) -> tuple[tuple[str, str], ...]:
        names = [self._parse_one_import_name(module)]
        while self._peek_punct(","):
            self._advance()
            names.append(self._parse_one_import_name(module))
        return tuple(names)

    def _parse_one_import_name(self, module: str) -> tuple[str, str]:
        name = self._expect("IDENT").value
        alias = self._parse_optional_as() or name
        return (f"{module}.{name}", alias)

    def _parse_optional_as(self) -> str | None:
        """Consume ``as IDENT`` if present and return the alias; else None."""
        if self._peek_kw("as"):
            self._advance()
            return self._expect("IDENT").value
        return None

    def _skip_package(self) -> None:
        self._advance()  # consume 'package'
        self._parse_field_path()  # consume qualified name
        if self._peek_punct(";"):
            self._advance()

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def _parse_declare(self, tags: tuple[Tag, ...] = ()) -> DeclareDecl:
        self._expect("KW", "declare")
        name = self._expect("IDENT").value
        extends: str | None = None
        if self._peek_kw("extends"):
            self._advance()
            extends = self._expect("IDENT").value
        fields: list[FieldDecl] = []
        while not self._peek_kw("end"):
            field_tags = self._parse_tags()
            if self._peek_kw("end"):
                break  # stray tags before 'end' — store-and-ignore policy
            fields.append(self._parse_field(field_tags))
        self._expect("KW", "end")
        return DeclareDecl(name, tuple(fields), extends, tags)

    def _parse_field(self, tags: tuple[Tag, ...] = ()) -> FieldDecl:
        name = self._expect("IDENT").value
        self._expect("PUNCT", ":")
        type_name = self._parse_type_ref()
        return FieldDecl(name, type_name, tags)

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

    def _parse_rule(self, tags: tuple[Tag, ...] = ()) -> RuleDecl:
        self._expect("KW", "rule")
        name = self._expect("STRING").value[1:-1]
        salience, no_loop = self._parse_rule_attrs()
        self._expect("KW", "when")
        lhs = self._parse_lhs()
        self._expect("KW", "then")
        rhs_src = self._expect("RAWBLOCK").value
        self._expect("KW", "end")
        return RuleDecl(name, salience, no_loop, lhs, rhs_src, tags)

    def _parse_rule_attrs(self) -> tuple[int, bool]:
        salience = 0
        no_loop = False
        while not self._peek_kw("when"):
            salience, no_loop = self._parse_one_attr(salience, no_loop)
        return salience, no_loop

    def _parse_one_attr(self, salience: int, no_loop: bool) -> tuple[int, bool]:
        if self._peek_kw("salience"):
            self._advance()
            return self._parse_int(), no_loop
        if self._peek_kw("no-loop"):
            self._advance()
            self._try_bool()
            return salience, True
        t = self._peek()
        raise SyntaxError(
            f"Unknown rule attribute at line {getattr(t, 'line', '?')}: "
            f"{getattr(t, 'value', t)!r}"
        )

    def _try_bool(self) -> None:
        t = self._peek()
        if t and t.kind == "KW" and t.value in ("True", "False", "true", "false"):
            self._advance()

    # ------------------------------------------------------------------
    # LHS and conditions
    # ------------------------------------------------------------------

    def _parse_lhs(self) -> tuple:
        """Parse the LHS condition list, handling ``or`` branches.

        Returns a flat tuple of conditions when no ``or`` is found,
        or a single-element tuple containing an :class:`OrGroup` otherwise.
        """
        branches: list[tuple] = []
        current: list = []
        while not self._peek_kw("then"):
            current.append(self._parse_condition())
            if self._peek_kw("or"):
                self._advance()
                branches.append(tuple(current))
                current = []
        if not branches:
            return tuple(current)
        branches.append(tuple(current))
        return (OrGroup(tuple(branches)),)

    def _parse_condition(self) -> PatternNode | NccPatternGroup | ForallNode:
        if self._peek_kw("not"):
            return self._parse_negated()
        if self._peek_kw("forall"):
            return self._parse_forall()
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

    def _parse_forall(self) -> ForallNode:
        """Parse ``forall(P, Q)`` — two comma-separated patterns in parens."""
        self._expect("KW", "forall")
        self._expect("PUNCT", "(")
        pattern = self._parse_pattern()
        self._expect("PUNCT", ",")
        condition = self._parse_pattern()
        self._expect("PUNCT", ")")
        return ForallNode(pattern, condition)

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

    def _parse_constraint(self):
        if self._is_positional_start():
            return self._parse_positional()
        if self._is_named_start():
            return self._parse_named()
        if self._peek_kind("VAR"):
            return self._parse_bind()
        return self._parse_compare()

    def _is_positional_start(self) -> bool:
        """Return True if the next token starts a positional constraint."""
        if self._peek_kind("VAR"):
            return not self._peek2_punct(":")
        return _is_value_token(self._peek())

    def _is_named_start(self) -> bool:
        """Return True if the next tokens form ``IDENT =`` (named constraint)."""
        return self._peek_kind("IDENT") and self._peek2_eq()

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

    def _parse_positional(self) -> PositionalConstraint:
        return PositionalConstraint(self._parse_value())

    def _parse_named(self) -> NamedConstraint:
        field = self._expect("IDENT").value
        self._expect("OP", "=")
        return NamedConstraint(field, self._parse_value())

    def _peek2_eq(self) -> bool:
        t = self._toks[self._pos + 1] if self._pos + 1 < len(self._toks) else None
        return t is not None and t.kind == "OP" and t.value == "="

    def _peek2_punct(self, v: str) -> bool:
        t = self._toks[self._pos + 1] if self._pos + 1 < len(self._toks) else None
        return t is not None and t.kind == "PUNCT" and t.value == v

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
    # Tag parsing
    # ------------------------------------------------------------------

    def _parse_tags(self) -> tuple[Tag, ...]:
        """Collect zero or more ``@name`` / ``@name(value)`` annotations."""
        tags: list[Tag] = []
        while self._peek_kind("AT"):
            tags.append(self._parse_one_tag())
        return tuple(tags)

    def _parse_one_tag(self) -> Tag:
        """Parse a single tag; tag name may be IDENT or KW (e.g. ``no-loop``)."""
        self._expect("AT")
        t = self._peek()
        if t is None or t.kind not in ("IDENT", "KW"):
            raise SyntaxError(
                f"Expected tag name after '@' at line {getattr(t, 'line', '?')}"
            )
        name = self._advance().value
        value: str | None = None
        if self._peek_punct("("):
            self._advance()
            value = self._collect_tag_value()
            self._expect("PUNCT", ")")
        return Tag(name, value)

    def _collect_tag_value(self) -> str:
        """Consume tokens up to ``)``, joining their text."""
        parts: list[str] = []
        while not self._peek_punct(")"):
            t = self._peek()
            if t is None:
                raise SyntaxError("Unterminated tag value: missing ')'")
            parts.append(self._advance().value)
        return "".join(parts)

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
