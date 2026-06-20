"""Frozen AST dataclasses for the pRETE Rule Language (PRL).

One dataclass per grammar production.  These are pure data containers:
no validation, no visitor infrastructure.  The parser (``prl_parser``)
constructs them; the compiler (``prl``) reads them.

All multi-value fields are ``tuple`` so that frozen instances are hashable.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FieldDecl",
    "DeclareDecl",
    "BindConstraint",
    "CompareConstraint",
    "PatternNode",
    "NccPatternGroup",
    "RuleDecl",
    "ProgramNode",
]


@dataclass(frozen=True)
class FieldDecl:
    """One field inside a ``declare`` block.

    :param name: field identifier, e.g. ``"value"``.
    :param type_name: base type name after generic erasure, e.g. ``"double"``.
        Generic parameters (``List<Integer>``) are stripped by the parser;
        the compiler maps this name via ``_JAVA_TO_PY``.
    """

    name: str
    type_name: str


@dataclass(frozen=True)
class DeclareDecl:
    """A complete fact-type declaration.

    :param name: class name, e.g. ``"Temperature"``.
    :param fields: ordered tuple of :class:`FieldDecl` instances; may be empty.
    :param extends: parent type name for inheritance, or ``None``.
    """

    name: str
    fields: tuple[FieldDecl, ...]
    extends: str | None = None


@dataclass(frozen=True)
class BindConstraint:
    """A variable-binding constraint: ``$var: field_path``.

    :param var: dollar-prefixed variable name, e.g. ``"$v"``.
    :param field: dotted field path, e.g. ``"address.city"``.
    """

    var: str
    field: str


@dataclass(frozen=True)
class CompareConstraint:
    """A comparison constraint: ``field_path op value_expr``.

    :param field: dotted field path, e.g. ``"value"``.
    :param op: comparison operator — one of ``== != < <= > >=``.
    :param rhs: right-hand side value.  A ``str`` starting with ``$`` is a
        variable reference (join test); any other ``str`` is a string literal.
        ``int``, ``float``, ``bool``, or ``None`` are parsed literal values.
    """

    field: str
    op: str
    rhs: str | int | float | bool | None


@dataclass(frozen=True)
class PatternNode:
    """A single positive or negated pattern in the LHS.

    :param type_name: fact type to match, e.g. ``"Temperature"``.
    :param fact_var: dollar-prefixed fact binding if present (``"$t"``),
        ``None`` otherwise.  Always provided explicitly by the parser.
    :param constraints: ordered tuple of :class:`BindConstraint` or
        :class:`CompareConstraint`; may be empty.
    :param negated: ``True`` for ``not pattern`` (single-pattern negation).
    """

    type_name: str
    fact_var: str | None
    constraints: tuple[BindConstraint | CompareConstraint, ...]
    negated: bool


@dataclass(frozen=True)
class NccPatternGroup:
    """A negated conjunctive condition group — ``not ( pattern+ )``.

    Compiles to :class:`~rete.condition.NccGroup` (Doorenbos §2.8).

    Named ``NccPatternGroup`` to avoid collisions with ``beta.NccNode``
    (a beta-network node) and ``condition.NccGroup`` (the compiled object).

    :param patterns: non-empty tuple of :class:`PatternNode` instances.
        PRL does not support nested NCCs, so no ``NccPatternGroup`` here.
    """

    patterns: tuple[PatternNode, ...]


@dataclass(frozen=True)
class RuleDecl:
    """A complete rule declaration.

    :param name: rule name string (the quoted literal, e.g. ``"too-hot"``).
    :param salience: conflict-resolution priority; defaults to ``0`` when the
        ``salience`` attribute is absent.  Stored for future engine wiring;
        not yet used by :class:`~rete.engine.InferenceEngine`.
    :param lhs: ordered tuple of :class:`PatternNode` or
        :class:`NccPatternGroup`; empty tuple means unconditional.
    :param rhs_src: verbatim Python source of the then-block (may be empty).
    """

    name: str
    salience: int = 0
    lhs: tuple[PatternNode | NccPatternGroup, ...] = ()
    rhs_src: str = ""


@dataclass(frozen=True)
class ProgramNode:
    """The parse-tree root — the entire PRL program.

    Both fields are required (no defaults); the parser always provides them
    explicitly, even as empty tuples.

    :param declares: ordered tuple of :class:`DeclareDecl` nodes.
    :param rules: ordered tuple of :class:`RuleDecl` nodes.
    """

    declares: tuple[DeclareDecl, ...]
    rules: tuple[RuleDecl, ...]
