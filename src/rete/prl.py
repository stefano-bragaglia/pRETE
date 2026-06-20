"""PRL compiler: converts a PRL parse tree into RETE ``Production`` objects.

Public entry point::

    from rete.prl import load_prl

    types, productions = load_prl(source_text, types={}, engine=engine)
    for p in productions:
        engine.add_production(p)

The compiler has three sub-tasks:

- **Type resolution** — ``DeclareDecl`` → Python dataclass via ``make_dataclass``
- **LHS compilation** — ``PatternNode | NccPatternGroup`` → ``Pattern | NccGroup``
- **RHS compilation** — raw Python source → ``Callable[[Token], None]`` via ``exec``

:see: plan/PARSER-PLAN-STEP-4.md
"""
from __future__ import annotations

import operator
import re
import textwrap
from dataclasses import make_dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Any, Callable

from rete.condition import JoinSpec, NccGroup, Pattern, Production
from rete.fact import Fact, Token
from rete.prl_ast import (
    BindConstraint,
    CompareConstraint,
    DeclareDecl,
    NccPatternGroup,
    PatternNode,
)
from rete.prl_lexer import tokenize
from rete.prl_parser import parse

__all__ = ["load_prl"]

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_JAVA_TO_PY: dict[str, type] = {
    "str": str, "float": float, "bool": bool,
    "int": int, "long": int, "short": int, "byte": int,
    "double": float, "boolean": bool,
    "String": str, "char": str,
    "Object": object,
}

_OPS: dict[str, Any] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

_VAR_RE = re.compile(r"\$([A-Za-z_]\w*)")

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_prl(
    text: str,
    types: dict[str, type] | None = None,
    engine: Any = None,
) -> tuple[dict[str, type], list[Production]]:
    """Parse and compile PRL source into RETE types and productions.

    :param text: PRL source text.
    :param types: optional pre-existing type mapping (e.g. Python dataclasses
        declared outside PRL).  The compiler adds declared types to a copy
        of this dict.
    :param engine: optional :class:`~rete.engine.InferenceEngine`; if given,
        then-block helpers ``insert``, ``retract``, and ``update`` are
        injected into the RHS exec namespace.  If ``None``, any then-block
        that calls these helpers raises ``NameError`` at runtime.
    :returns: ``(resolved_types, productions)`` where *resolved_types* is the
        updated type mapping and *productions* is a list of ``Production``
        objects ready for ``engine.add_production``.
    :raises NameError: if a pattern references an undeclared fact type.

    .. note::
        ``textwrap.dedent`` requires consistent indentation in then-blocks.
        Mixed tabs and spaces will prevent dedenting and may cause
        ``IndentationError`` at exec time.
    """
    program = parse(tokenize(text))
    resolved: dict[str, type] = dict(types or {})
    for decl in _topo_sort_declares(program.declares):
        resolved[decl.name] = _compile_declare(decl, resolved)
    productions = [
        _compile_rule(r, resolved, engine) for r in program.rules
    ]
    return resolved, productions


# ---------------------------------------------------------------------------
# 4a — Type resolution
# ---------------------------------------------------------------------------


def _inheritance_graph(
    declares: tuple[DeclareDecl, ...],
) -> dict[str, set[str]]:
    """Build a ``{name: {parent}}`` adjacency dict for topological sorting.

    :param declares: raw declaration tuple from the parser.
    """
    return {
        d.name: {d.extends} if d.extends else set()
        for d in declares
    }


def _topo_sort_declares(
    declares: tuple[DeclareDecl, ...],
) -> list[DeclareDecl]:
    """Return *declares* sorted so parent types precede child types.

    :param declares: raw declaration tuple from the parser.
    :returns: topologically ordered list.
    :raises TypeError: on circular inheritance.
    """
    by_name = {d.name: d for d in declares}
    ts: TopologicalSorter[str] = TopologicalSorter(_inheritance_graph(declares))
    try:
        order = list(ts.static_order())
    except CycleError as exc:
        raise TypeError(f"Circular inheritance: {exc}") from exc
    return [by_name[n] for n in order if n in by_name]


def _compile_declare(decl: DeclareDecl, types: dict[str, type]) -> type:
    """Convert a ``DeclareDecl`` into a Python dataclass.

    :param decl: the declaration AST node.
    :param types: current type mapping (may include previously declared types).
    :returns: a new mutable dataclass class (not frozen — supports ``update``).
    """
    fields = [(fd.name, _java_type(fd.type_name, types)) for fd in decl.fields]
    if decl.extends:
        parent = _resolve_type(decl.extends, types)
        return make_dataclass(decl.name, fields, bases=(parent,))
    return make_dataclass(decl.name, fields)


def _java_type(name: str, types: dict[str, type]) -> type:
    """Resolve a PRL type name to a Python type.

    User-declared types take priority over ``_JAVA_TO_PY``.
    Unknown names fall back to ``typing.Any``.

    :param name: type name string from a ``FieldDecl``.
    :param types: current type mapping.
    """
    if name in types:
        return types[name]
    return _JAVA_TO_PY.get(name, Any)


def _resolve_type(name: str, types: dict[str, type]) -> type:
    """Look up *name* in *types*; raise ``NameError`` if absent.

    :param name: fact type name from a ``PatternNode``.
    :param types: compiled type mapping.
    :raises NameError: if *name* has not been declared.
    """
    if name not in types:
        raise NameError(
            f"Unknown fact type {name!r}. "
            "Declare it with 'declare' or pass it via types=."
        )
    return types[name]


# ---------------------------------------------------------------------------
# 4b — LHS compilation
# ---------------------------------------------------------------------------


def _compile_rule(
    rule: Any,
    types: dict[str, type],
    engine: Any,
) -> Production:
    """Compile one ``RuleDecl`` into a ``Production``.

    :param rule: the ``RuleDecl`` AST node.
    :param types: resolved type mapping.
    :param engine: optional engine for RHS helper injection.

    .. note::
        ``rule.salience`` is parsed but ``Production`` has no salience field
        yet; it is silently carried here until the engine is extended.
    """
    # ponytail: rule.salience parsed but Production has no salience field yet
    conditions, fact_bindings = _compile_lhs(rule.lhs, types)
    rhs = _compile_rhs(rule.rhs_src, fact_bindings, types, engine)
    return Production(lhs=conditions, rhs=rhs)


def _compile_lhs(
    lhs_nodes: tuple[PatternNode | NccPatternGroup, ...],
    types: dict[str, type],
) -> tuple[list[Pattern | NccGroup], dict[str, int]]:
    """Compile the LHS tuple into a conditions list and a fact-binding index.

    :param lhs_nodes: ordered tuple of ``PatternNode | NccPatternGroup``.
    :param types: resolved type mapping.
    :returns: ``(conditions, fact_bindings)`` where *fact_bindings* maps each
        fact-variable string (e.g. ``"$t"``) to its index in *conditions*.
    """
    conditions: list[Pattern | NccGroup] = []
    fact_bindings: dict[str, int] = {}
    for idx, node in enumerate(lhs_nodes):
        if isinstance(node, NccPatternGroup):
            conditions.append(_compile_ncc(node, types))
        else:
            conditions.append(_compile_pattern(node, idx, types, fact_bindings))
    return conditions, fact_bindings


def _compile_pattern(
    node: PatternNode,
    idx: int,
    types: dict[str, type],
    fact_bindings: dict[str, int],
) -> Pattern:
    """Compile one ``PatternNode`` into a ``Pattern``.

    :param node: the pattern AST node.
    :param idx: position of this pattern in the LHS (for fact-binding index).
    :param types: resolved type mapping.
    :param fact_bindings: mutated in place when ``node.fact_var`` is set.
    """
    type_ = _resolve_type(node.type_name, types)
    if node.fact_var:
        fact_bindings[node.fact_var] = idx
    alpha_tests, join_tests, bindings = _compile_constraints(node.constraints)
    return Pattern(type_, alpha_tests, join_tests, bindings, node.negated)


def _compile_constraints(
    constraints: tuple[BindConstraint | CompareConstraint, ...],
) -> tuple[tuple, tuple, tuple]:
    """Walk *constraints* and sort them into alpha tests, join specs, bindings.

    :param constraints: the constraint tuple from a ``PatternNode``.
    :returns: ``(alpha_tests, join_tests, bindings)`` as tuples.
    """
    alpha_tests: list[Callable] = []
    join_tests: list[JoinSpec] = []
    bindings: list[tuple[str, str]] = []
    for c in constraints:
        _apply_constraint(c, alpha_tests, join_tests, bindings)
    return tuple(alpha_tests), tuple(join_tests), tuple(bindings)


def _apply_constraint(
    c: BindConstraint | CompareConstraint,
    alpha_tests: list[Callable],
    join_tests: list[JoinSpec],
    bindings: list[tuple[str, str]],
) -> None:
    """Route one constraint to the appropriate bucket.

    :param c: the constraint AST node.
    :param alpha_tests: accumulator for alpha-stage callables.
    :param join_tests: accumulator for cross-fact ``JoinSpec`` objects.
    :param bindings: accumulator for ``(var, field)`` binding pairs.
    """
    if isinstance(c, BindConstraint):
        bindings.append((c.var, c.field))
        return
    if isinstance(c.rhs, str) and c.rhs.startswith("$"):
        join_tests.append(JoinSpec(c.field, c.rhs))
    else:
        alpha_tests.append(_make_alpha_test(c.field, c.op, c.rhs))


def _make_alpha_test(field: str, op: str, rhs: Any) -> Callable:
    """Return a single-argument callable that tests one field value.

    Uses a factory function (not an inline lambda) to avoid the
    late-binding closure trap when called from a loop.

    :param field: dotted field path, e.g. ``"address.city"``.
    :param op: comparison operator string, e.g. ``">"``.
    :param rhs: right-hand side value to compare against.
    """
    fn = _OPS[op]
    return lambda obj, _f=field, _fn=fn, _r=rhs: _fn(_getattr_path(obj, _f), _r)


def _getattr_path(obj: Any, path: str) -> Any:
    """Resolve a dotted attribute path on *obj*.

    :param obj: the root object.
    :param path: dot-separated attribute chain, e.g. ``"address.city"``.
    """
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj


def _compile_ncc(node: NccPatternGroup, types: dict[str, type]) -> NccGroup:
    """Compile an ``NccPatternGroup`` into an ``NccGroup``.

    Fact-level bindings inside the NCC group are not propagated to the outer
    RHS — the engine does not expose NCC-internal token contents.

    :param node: the NCC AST node.
    :param types: resolved type mapping.
    """
    patterns = tuple(
        _compile_pattern(p, -1, types, {}) for p in node.patterns
    )
    return NccGroup(patterns)


# ---------------------------------------------------------------------------
# 4c — RHS compilation
# ---------------------------------------------------------------------------


def _compile_rhs(
    rhs_src: str,
    fact_bindings: dict[str, int],
    types: dict[str, type],
    engine: Any,
) -> Callable[[Token], None]:
    """Build the RHS callable for one rule.

    :param rhs_src: raw Python source from the then-block (may be empty).
    :param fact_bindings: ``{fact_var: lhs_index}`` from LHS compilation.
    :param types: resolved type mapping (injected as names in the namespace).
    :param engine: optional engine; if given, helpers are added to namespace.
    :returns: ``(token) -> None`` closure that executes the then-block.
    """
    code = textwrap.dedent(_strip_dollars(rhs_src))
    ns_base = dict(types)
    if engine is not None:
        ns_base.update(_engine_helpers(engine))
    return _make_rhs_closure(code, fact_bindings, ns_base)


def _make_rhs_closure(
    code: str,
    fact_bindings: dict[str, int],
    ns_base: dict,
) -> Callable[[Token], None]:
    """Return the RHS closure that executes *code* against a matched token.

    :param code: dedented, dollar-stripped Python source.
    :param fact_bindings: ``{fact_var: lhs_index}`` map.
    :param ns_base: base namespace dict (types + optional helpers).
    """
    def rhs(token: Token) -> None:
        ns = dict(ns_base)
        ns.update({k.lstrip("$"): v for k, v in token.bindings.items()})
        for var, idx in fact_bindings.items():
            ns[var.lstrip("$")] = token.facts[idx]
        exec(code, ns)  # noqa: S102

    return rhs


def _strip_dollars(src: str) -> str:
    """Replace every ``$identifier`` in *src* with the bare identifier.

    :param src: PRL then-block source text.

    .. note::
        The regex does not distinguish ``$var`` in code from ``$var`` inside a
        string literal.  This is an accepted limitation for PRL then-blocks.
    """
    return _VAR_RE.sub(r"\1", src)


def _engine_helpers(engine: Any) -> dict[str, Callable]:
    """Return a dict of WM-mutation helpers bound to *engine*.

    :param engine: a live :class:`~rete.engine.InferenceEngine`.
    :returns: ``{"insert": ..., "retract": ..., "update": ...}``.
    """
    def insert(obj: Any) -> None:
        engine.add_fact(Fact(obj))

    def retract(fact: Fact) -> None:
        engine.remove_fact(fact)

    def update(fact: Fact) -> None:
        engine.update_fact(fact)

    return {"insert": insert, "retract": retract, "update": update}
