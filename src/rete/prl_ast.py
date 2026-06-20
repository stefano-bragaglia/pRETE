"""Frozen AST dataclasses for the pRETE Rule Language (PRL).

One dataclass per grammar production.  These are pure data containers:
no validation, no visitor infrastructure.  The parser (``prl_parser``)
constructs them; the compiler (``prl``) reads them.

All multi-value fields are ``tuple`` so that frozen instances are hashable.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Tag",
    "FieldDecl",
    "DeclareDecl",
    "ImportDecl",
    "BindConstraint",
    "CompareConstraint",
    "PositionalConstraint",
    "NamedConstraint",
    "PatternNode",
    "NccPatternGroup",
    "OrGroup",
    "ForallNode",
    "RuleDecl",
    "ProgramNode",
]


@dataclass(frozen=True)
class Tag:
    """A single annotation tag — ``@name`` or ``@name(value)``.

    :param name: tag identifier, e.g. ``"key"``, ``"role"``, ``"no-loop"``.
    :param value: raw text between parentheses if present, ``None`` otherwise.
    """

    name: str
    value: str | None = None


@dataclass(frozen=True)
class FieldDecl:
    """One field inside a ``declare`` block.

    :param name: field identifier, e.g. ``"value"``.
    :param type_name: base type name after generic erasure, e.g. ``"double"``.
        Generic parameters (``List<Integer>``) are stripped by the parser;
        the compiler maps this name via ``_JAVA_TO_PY``.
    :param tags: zero or more :class:`Tag` annotations preceding this field.
    """

    name: str
    type_name: str
    tags: tuple[Tag, ...] = ()


@dataclass(frozen=True)
class DeclareDecl:
    """A complete fact-type declaration.

    :param name: class name, e.g. ``"Temperature"``.
    :param fields: ordered tuple of :class:`FieldDecl` instances; may be empty.
    :param extends: parent type name for inheritance, or ``None``.
    :param tags: zero or more :class:`Tag` annotations preceding this declare.
    """

    name: str
    fields: tuple[FieldDecl, ...]
    extends: str | None = None
    tags: tuple[Tag, ...] = ()


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
class PositionalConstraint:
    """A bare value constraint — ``TypeName(v1, v2)`` positional form.

    The compiler maps this to the *n*-th declared field (0-based count of
    ``PositionalConstraint`` instances in the pattern, left-to-right).

    :param value: parsed value (same range as :attr:`CompareConstraint.rhs`).
    """

    value: str | int | float | bool | None


@dataclass(frozen=True)
class NamedConstraint:
    """A keyword-style constraint — ``field=value`` inside a pattern.

    Semantically equivalent to ``CompareConstraint(field, "==", value)``.

    :param field: top-level field name (no dots).
    :param value: parsed value (same range as :attr:`CompareConstraint.rhs`).
    """

    field: str
    value: str | int | float | bool | None


@dataclass(frozen=True)
class PatternNode:
    """A single positive or negated pattern in the LHS.

    :param type_name: fact type to match, e.g. ``"Temperature"``.
    :param fact_var: dollar-prefixed fact binding if present (``"$t"``),
        ``None`` otherwise.  Always provided explicitly by the parser.
    :param constraints: ordered tuple of constraint nodes; may be empty.
    :param negated: ``True`` for ``not pattern`` (single-pattern negation).
    :param exists: ``True`` for ``exists pattern`` (existential check).
    """

    type_name: str
    fact_var: str | None
    constraints: tuple[
        BindConstraint | CompareConstraint
        | PositionalConstraint | NamedConstraint, ...
    ]
    negated: bool
    exists: bool = False


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
class OrGroup:
    """Disjunctive LHS — K mutually alternative condition sequences.

    The compiler splits one ``RuleDecl`` containing an ``OrGroup`` into K
    ``Production`` objects, each with the same RHS closure but a distinct
    LHS branch.  All branches must bind the same set of ``$var`` names.

    :param branches: each element is a full LHS sequence (a tuple of
        ``PatternNode | NccPatternGroup``); at least two branches expected.
    """

    branches: tuple[tuple[PatternNode | NccPatternGroup, ...], ...]


@dataclass(frozen=True)
class ForallNode:
    """Universal-quantification shorthand — ``forall(P, Q)``.

    Fires when no fact matches P without a corresponding Q-fact.
    Compiles to ``NccGroup([P_pattern, Q_negated_pattern])`` so the rule
    fires when there is zero (P-without-Q) evidence in working memory.

    :param pattern: the universally-quantified pattern P.
    :param condition: the condition Q that must hold for every P.
    """

    pattern: PatternNode
    condition: PatternNode


@dataclass(frozen=True)
class RuleDecl:
    """A complete rule declaration.

    :param name: rule name string (the quoted literal, e.g. ``"too-hot"``).
    :param salience: conflict-resolution priority; defaults to ``0`` when the
        ``salience`` attribute is absent.  Stored for future engine wiring;
        not yet used by :class:`~rete.engine.InferenceEngine`.
    :param no_loop: ``True`` when the ``no-loop`` rule attribute is set.
        The ``@no-loop`` tag (stored in ``tags``) is combined by the compiler.
    :param lhs: ordered tuple of :class:`PatternNode` or
        :class:`NccPatternGroup`; empty tuple means unconditional.
    :param rhs_src: verbatim Python source of the then-block (may be empty).
    :param tags: zero or more :class:`Tag` annotations preceding this rule.
    """

    name: str
    salience: int = 0
    no_loop: bool = False
    lhs: tuple[PatternNode | NccPatternGroup | ForallNode | OrGroup, ...] = ()
    rhs_src: str = ""
    tags: tuple[Tag, ...] = ()


@dataclass(frozen=True)
class ImportDecl:
    """One import statement — one or more ``(qualified_path, local_alias)`` pairs.

    The compiler splits each *qualified_path* at its last ``.`` to obtain the
    module and attribute names for ``importlib``.

    :param names: tuple of ``(full_dotted_path, local_alias)`` entries.
    """

    names: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ProgramNode:
    """The parse-tree root — the entire PRL program.

    :param declares: ordered tuple of :class:`DeclareDecl` nodes.
    :param rules: ordered tuple of :class:`RuleDecl` nodes.
    :param imports: ordered tuple of :class:`ImportDecl` nodes; defaults to
        ``()`` to preserve backward-compatible positional construction.
    """

    declares: tuple[DeclareDecl, ...]
    rules: tuple[RuleDecl, ...]
    imports: tuple[ImportDecl, ...] = ()
