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

import importlib
import operator
import re
import textwrap
from dataclasses import make_dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Any, Callable

from rete.condition import AccumulateSpec, JoinSpec, NccGroup, Pattern, Production
from rete.fact import Fact, Token
from rete.prl_ast import (
    AccumulateExpr,
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

_ACCUMULATE_FNS: dict[str, Any] = {
    "count": lambda vals: len(vals),
    "sum": lambda vals: sum(vals),
    "min": lambda vals: min(vals) if vals else None,
    "max": lambda vals: max(vals) if vals else None,
    "collectList": lambda vals: list(vals),
}

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
    for imp in program.imports:
        _resolve_import(imp, resolved)
    for decl in _topo_sort_declares(program.declares):
        resolved[decl.name] = _compile_declare(decl, resolved)
    return resolved, _flatten_rules(program.rules, resolved, engine)


# ---------------------------------------------------------------------------
# 4a — Import resolution
# ---------------------------------------------------------------------------


def _resolve_import(imp: ImportDecl, types: dict[str, type]) -> None:
    """Inject all names from *imp* into *types*.

    :param imp: the :class:`ImportDecl` AST node.
    :param types: the type registry to mutate in place.
    """
    for qualified, alias in imp.names:
        types[alias] = _import_name(qualified)


def _import_name(qualified: str) -> type:
    """Resolve a fully-qualified dotted name to a Python object via importlib.

    Splits at the last ``.``: everything before is the module; the last
    component is the attribute fetched from that module.

    :param qualified: full dotted path, e.g. ``"rete.fact.Fact"``.
    :raises ImportError: if the module or attribute cannot be resolved.
    """
    module, _, attr = qualified.rpartition(".")
    if not module:
        raise ImportError(
            f"Cannot import bare name {qualified!r}; "
            "use 'import module.ClassName' or 'from module import ClassName'"
        )
    # ponytail: 'import a.b as mm' resolves to getattr(module_a, 'b'), not
    # the module a.b itself. Full module-alias support needs dotted type names
    # in patterns and is deferred past ES-5.
    try:
        return getattr(importlib.import_module(module), attr)
    except ModuleNotFoundError as exc:
        raise ImportError(f"Cannot import {qualified!r}: {exc}") from exc
    except AttributeError:
        raise ImportError(
            f"Module {module!r} has no attribute {attr!r}"
        ) from None


# ---------------------------------------------------------------------------
# 4b — Type resolution
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

    When any field carries ``@key``, the generated class uses ``eq=False``
    so that a manually injected key-only ``__eq__`` / ``__hash__`` can be
    assigned without being blocked by the dataclass-generated ``__hash__ = None``.

    :param decl: the declaration AST node.
    :param types: current type mapping (may include previously declared types).
    :returns: a new mutable dataclass class (not frozen — supports ``update``).
    """
    fields = [(fd.name, _java_type(fd.type_name, types)) for fd in decl.fields]
    key_names = _key_fields(decl)
    kwargs: dict = {"eq": False} if key_names else {}
    if decl.extends:
        parent = _resolve_type(decl.extends, types)
        cls = make_dataclass(decl.name, fields, bases=(parent,), **kwargs)
    else:
        cls = make_dataclass(decl.name, fields, **kwargs)
    if key_names:
        _inject_key_eq(cls, key_names)
    _attach_event_meta(cls, decl)
    return cls


def _key_fields(decl: DeclareDecl) -> tuple[str, ...]:
    """Return the names of fields in *decl* that carry an ``@key`` tag.

    Only inspects fields declared directly in *decl*, not inherited fields.

    :param decl: the declaration AST node.
    """
    # ponytail: @key on inherited fields not supported; local fields only
    return tuple(fd.name for fd in decl.fields if _has_key_tag(fd))


def _has_tag(fd: FieldDecl, tag_name: str) -> bool:
    """Return True iff *fd* has at least one tag with *tag_name*.

    :param fd: a field declaration node.
    :param tag_name: the tag name to look for.
    """
    return any(t.name == tag_name for t in fd.tags)


def _has_key_tag(fd: FieldDecl) -> bool:
    """Return True iff *fd* has at least one tag named ``"key"``.

    :param fd: a field declaration node.
    """
    return _has_tag(fd, "key")


def _parse_time_offset(s: str) -> float:
    """Convert a time string like ``"10s"``, ``"5m"``, ``"1h30m"`` to seconds.

    :param s: time string using h/m/s units.
    :raises ValueError: if *s* is empty, has no recognised units, or does not match.
    """
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", s)
    if not m or not any(m.groups()):
        raise ValueError(f"Invalid time offset: {s!r}")
    h, mins, secs = (int(g) if g else 0 for g in m.groups())
    return float(h * 3600 + mins * 60 + secs)


def _is_event_role(decl: DeclareDecl) -> bool:
    """Return True iff *decl* has ``@role(event)``."""
    return any(t.name == "role" and t.value == "event" for t in decl.tags)


def _field_with_tag(decl: DeclareDecl, tag_name: str) -> str | None:
    """Return the first field name in *decl* carrying *tag_name*, or None."""
    return next((fd.name for fd in decl.fields if _has_tag(fd, tag_name)), None)


def _decl_tag_value(decl: DeclareDecl, tag_name: str) -> str | None:
    """Return the value of the first declare-level tag named *tag_name*, or None."""
    return next((t.value for t in decl.tags if t.name == tag_name), None)


def _attach_event_meta(cls: type, decl: DeclareDecl) -> None:
    """Attach ``__prl_meta__`` to *cls* if *decl* carries ``@role(event)``.

    :param cls: the freshly created dataclass to annotate.
    :param decl: the declaration AST node.
    """
    if not _is_event_role(decl):
        return
    expires_val = _decl_tag_value(decl, "expires")
    cls.__prl_meta__ = {
        "role": "event",
        "timestamp_field": _field_with_tag(decl, "timestamp"),
        # ponytail: duration_field unused; interval-overlap matching deferred
        "duration_field": _field_with_tag(decl, "duration"),
        "expires_delta": _parse_time_offset(expires_val) if expires_val else None,
    }


def _inject_key_eq(cls: type, key_names: tuple[str, ...]) -> None:
    """Assign key-field-only ``__eq__`` and ``__hash__`` to *cls*.

    Called only when ``eq=False`` was passed to ``make_dataclass``, so
    ``__hash__`` is still the identity-based ``object.__hash__`` and
    can be overridden by direct assignment.

    :param cls: the freshly created dataclass to patch.
    :param key_names: ordered tuple of field names that form the identity key.
    """
    def __eq__(self: object, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented  # type: ignore[return-value]
        return all(getattr(self, f) == getattr(other, f) for f in key_names)

    def __hash__(self: object) -> int:
        return hash(tuple(getattr(self, f) for f in key_names))

    cls.__eq__ = __eq__  # type: ignore[method-assign]
    cls.__hash__ = __hash__  # type: ignore[method-assign]


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


def _flatten_rules(
    rules: tuple,
    types: dict[str, type],
    engine: Any,
) -> list[Production]:
    """Compile all rules, flattening or-rules into multiple Productions."""
    result: list[Production] = []
    for r in rules:
        result.extend(_compile_rule(r, types, engine))
    return result


def _compile_rule(
    rule: Any,
    types: dict[str, type],
    engine: Any,
) -> list[Production]:
    """Compile one ``RuleDecl`` into one or more ``Production`` objects.

    A rule with no ``or`` returns a single-element list.  A rule whose LHS
    is an :class:`~rete.prl_ast.OrGroup` returns one ``Production`` per branch.

    :param rule: the ``RuleDecl`` AST node.
    :param types: resolved type mapping.
    :param engine: optional engine for RHS helper injection.

    .. note::
        ``rule.salience`` is parsed but ``Production`` has no salience field
        yet; it is silently carried here until the engine is extended.
    """
    # ponytail: rule.salience parsed but Production has no salience field yet
    no_loop = rule.no_loop or any(t.name == "no-loop" for t in rule.tags)
    lhs = rule.lhs
    if len(lhs) == 1 and isinstance(lhs[0], OrGroup):
        return _compile_or_rule(rule, lhs[0], types, engine, no_loop)
    conditions, fact_bindings = _compile_lhs(lhs, types)
    rhs = _compile_rhs(rule.rhs_src, fact_bindings, types, engine)
    return [Production(lhs=conditions, rhs=rhs, no_loop=no_loop)]


def _compile_or_rule(
    rule: Any,
    or_group: OrGroup,
    types: dict[str, type],
    engine: Any,
    no_loop: bool,
) -> list[Production]:
    """Expand one or-rule into one Production per branch.

    Each branch compiles independently with its own LHS and fact-binding
    index, but all share the same raw RHS source.

    :param rule: the original ``RuleDecl`` (for ``rhs_src``).
    :param or_group: the parsed disjunction.
    :param types: resolved type mapping.
    :param engine: optional engine for RHS helper injection.
    :param no_loop: combined ``no-loop`` flag from attribute + tag.

    .. note::
        Branch naming (``"rule_name [or 0]"``) is a documentation convention
        only — ``Production`` has no ``name`` field in this version.

    # ponytail: Production.name not added here; deferred until an engine
    # feature (e.g. query/debug) actually needs it.
    """
    _check_or_var_scopes(or_group.branches)
    prods = []
    for branch in or_group.branches:
        conditions, fact_bindings = _compile_lhs(branch, types)
        rhs = _compile_rhs(rule.rhs_src, fact_bindings, types, engine)
        prods.append(Production(lhs=conditions, rhs=rhs, no_loop=no_loop))
    return prods


def _node_vars(node: PatternNode) -> set[str]:
    """Return all ``$var`` names bound by one ``PatternNode``.

    Collects both the ``fact_var`` prefix (``$x: Type()``) and any
    :class:`~rete.prl_ast.BindConstraint` vars (``$v: field``).

    :param node: a ``PatternNode`` from the parser.
    """
    bound = {node.fact_var} if node.fact_var else set()
    bound.update(c.var for c in node.constraints if isinstance(c, BindConstraint))
    return bound


def _branch_vars(branch: tuple) -> frozenset[str]:
    """Return all ``$var`` names bound across one or-branch.

    :param branch: a tuple of ``PatternNode | NccPatternGroup`` nodes.
    """
    bound: set[str] = set()
    for node in branch:
        if isinstance(node, PatternNode):
            bound |= _node_vars(node)
    return frozenset(bound)


def _check_or_var_scopes(branches: tuple) -> None:
    """Raise ``SyntaxError`` if or-branches bind different variable sets.

    All branches must bind exactly the same set of ``$var`` names so that
    the shared RHS closure does not raise ``NameError`` at runtime.

    :param branches: all branches of an ``OrGroup``.
    :raises SyntaxError: on variable-set mismatch between any two branches.
    """
    sets = [_branch_vars(b) for b in branches]
    ref = sets[0]
    for i, s in enumerate(sets[1:], 1):
        if s != ref:
            raise SyntaxError(
                f"'or' branch {i} binds {sorted(s)!r} "
                f"but branch 0 binds {sorted(ref)!r}; "
                "all branches must bind the same variable names"
            )


def _compile_lhs(
    lhs_nodes: tuple,
    types: dict[str, type],
) -> tuple[list[Pattern | NccGroup], dict[str, int]]:
    """Compile the LHS tuple into a conditions list and a fact-binding index.

    :param lhs_nodes: ordered tuple of ``PatternNode | NccPatternGroup | ForallNode``.
    :param types: resolved type mapping.
    :returns: ``(conditions, fact_bindings)`` where *fact_bindings* maps each
        fact-variable string (e.g. ``"$t"``) to its index in *conditions*.
    """
    conditions: list[Pattern | NccGroup | AccumulateSpec] = []
    fact_bindings: dict[str, int] = {}
    for idx, node in enumerate(lhs_nodes):
        if isinstance(node, NccPatternGroup):
            conditions.append(_compile_ncc(node, types))
        elif isinstance(node, ForallNode):
            conditions.append(_compile_forall(node, idx, types, fact_bindings))
        elif isinstance(node, AccumulateExpr):
            conditions.append(_compile_accumulate(node, types))
        else:
            conditions.append(_compile_pattern(node, idx, types, fact_bindings))
    return conditions, fact_bindings


def _compile_forall(
    node: ForallNode,
    idx: int,
    types: dict[str, type],
    fact_bindings: dict[str, int],
) -> NccGroup:
    """Compile ``forall(P, Q)`` to ``NccGroup([P, Q_negated])``.

    The semantics of ``forall(P, Q)`` are "for every P there is a Q", which
    is logically equivalent to ``NOT(P AND NOT Q)``.  We compile this as an
    NCC subnetwork ``[P_pattern, Q_negated_pattern]`` — the rule fires when
    no token matches P without a matching Q.

    :param node: the ``ForallNode`` AST node.
    :param idx: position of this node in the outer LHS (for join-spec offsets).
    :param types: resolved type mapping.
    :param fact_bindings: mutable dict updated with ``P``'s fact_var if set.
    """
    p_pattern = _compile_pattern(node.pattern, idx, types, fact_bindings)
    q_negated = _compile_pattern(
        PatternNode(
            node.condition.type_name,
            node.condition.fact_var,
            node.condition.constraints,
            negated=True,
        ),
        idx + 1,
        types,
        fact_bindings,
    )
    return NccGroup((p_pattern, q_negated))


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
    constraints = _resolve_shorthand(node.constraints, type_)
    alpha_tests, join_tests, bindings = _compile_constraints(constraints)
    return Pattern(type_, alpha_tests, join_tests, bindings, node.negated, node.exists)


def _resolve_shorthand(
    constraints: tuple,
    type_: type,
) -> tuple[BindConstraint | CompareConstraint, ...]:
    """Expand :class:`PositionalConstraint` and :class:`NamedConstraint` nodes.

    Positional constraints are mapped to fields in declaration order;
    named constraints are mapped to the named field — both become
    :class:`CompareConstraint` (``==``).  Raises :class:`SyntaxError` on
    out-of-bounds positionals or positional/named collision.

    :param constraints: raw constraint tuple from the parser.
    :param type_: the compiled fact type (provides field order).
    """
    fields = _field_names(type_)
    out: list[BindConstraint | CompareConstraint] = []
    pos_idx = 0
    pos_fields: list[str] = []
    named_fields: set[str] = set()
    for c in constraints:
        if isinstance(c, PositionalConstraint):
            _check_pos_in_bounds(pos_idx, fields, type_)
            pos_fields.append(fields[pos_idx])
            out.append(CompareConstraint(fields[pos_idx], "==", c.value))
            pos_idx += 1
        elif isinstance(c, NamedConstraint):
            named_fields.add(c.field)
            out.append(CompareConstraint(c.field, "==", c.value))
        else:
            out.append(c)
    _check_collision(pos_fields, named_fields, type_)
    return tuple(out)


def _field_names(type_: type) -> list[str]:
    """Return field names in declaration order, parent fields first.

    :param type_: a dataclass type (from ``make_dataclass``).
    """
    return list(type_.__dataclass_fields__)


def _check_pos_in_bounds(idx: int, fields: list[str], type_: type) -> None:
    """Raise :class:`SyntaxError` if positional index *idx* exceeds field count.

    :param idx: current positional slot (0-based).
    :param fields: ordered field name list for the type.
    :param type_: type name for the error message.
    """
    if idx >= len(fields):
        raise SyntaxError(
            f"{type_.__name__} has {len(fields)} field(s); "
            "too many positional constraints"
        )


def _check_collision(
    pos_fields: list[str],
    named_fields: set[str],
    type_: type,
) -> None:
    """Raise :class:`SyntaxError` if a field appears in both positional and named.

    :param pos_fields: field names claimed by positional constraints.
    :param named_fields: field names claimed by named constraints.
    :param type_: type name for the error message.
    """
    overlap = set(pos_fields) & named_fields
    if overlap:
        raise SyntaxError(
            f"{type_.__name__}: field(s) {sorted(overlap)!r} appear "
            "in both positional and named constraints"
        )


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


def _compile_accumulate(
    node: AccumulateExpr, types: dict[str, type]
) -> AccumulateSpec:
    """Compile an :class:`AccumulateExpr` to an :class:`AccumulateSpec`.

    :param node: the accumulate AST node.
    :param types: resolved type mapping.
    :raises KeyError: if the function name is not a known built-in.
    :raises NameError: if *bind_var* is not bound in the inner pattern.
    """
    inner_pat = _compile_pattern(node.inner, -1, types, {})
    fn = _ACCUMULATE_FNS[node.function]
    bind_attr = _resolve_bind_attr(inner_pat, node.bind_var)
    constraint = _compile_acc_constraint(node.constraint)
    return AccumulateSpec(
        inner=inner_pat,
        fn=fn,
        bind_attr=bind_attr,
        result_var=node.result_var,
        constraint=constraint,
    )


def _resolve_bind_attr(inner: Pattern, bind_var: str | None) -> str | None:
    """Return the object attribute name for *bind_var* in the inner pattern.

    :param inner: the compiled inner pattern.
    :param bind_var: dollar-prefixed variable (``"$amount"``); ``None`` for count.
    :raises NameError: if *bind_var* is not bound in *inner*.
    """
    if bind_var is None:
        return None
    for var, attr in inner.bindings:
        if var == bind_var:
            return attr
    raise NameError(
        f"accumulate bind variable {bind_var!r} is not bound in the inner pattern"
    )


def _compile_acc_constraint(
    constraint: Any,
) -> Callable[[Any], bool] | None:
    """Compile an optional accumulate constraint to a callable.

    :param constraint: a :class:`~rete.prl_ast.CompareConstraint` or ``None``.
    :returns: ``lambda value: op(value, rhs)`` or ``None``.
    """
    if constraint is None:
        return None
    fn = _OPS[constraint.op]
    rhs = constraint.rhs
    return lambda v, _fn=fn, _r=rhs: _fn(v, _r)


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
