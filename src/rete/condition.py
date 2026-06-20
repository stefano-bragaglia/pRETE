"""Pattern and Production types for the Rete algorithm LHS.

Alpha tests (``alpha_tests``) are self-contained callables evaluated per-fact.
Join tests (``join_tests`` / :class:`JoinSpec`) reference variables already bound
in the token and are evaluated at join time.  Keeping the two separate preserves
the alpha/beta network boundary that is RETE's core optimisation.

.. note::
    Alpha memory sharing requires **stable function references** in
    ``alpha_tests``.  Two :class:`Pattern` objects pass the same function object
    → same :meth:`Pattern.alpha_key` → shared alpha memory.  Two objects with
    logically-identical but distinct lambdas will **not** share.  Use named,
    module-level functions when sharing matters.

:see: Doorenbos §2.1, §2.2, §2.8
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rete.fact import Fact, Token


@dataclass(frozen=True)
class JoinSpec:
    """Compile-time cross-fact variable reference declared inside a :class:`Pattern`.

    Distinct from ``JoinTest`` (``beta.py``): ``JoinSpec`` is user-declared;
    ``JoinTest`` is the network-runtime object derived from it during compilation.

    :param attr_of_fact: attribute on the new Fact's object to read
    :param var_name: variable already bound in the token to compare against
    :see: UPDATE_PLAN §Step 2
    """

    attr_of_fact: str
    var_name: str


@dataclass(frozen=True)
class Pattern:
    """A type-and-predicate pattern for one position in a production LHS.

    Matches a :class:`~rete.fact.Fact` iff the wrapped object passes the
    ``isinstance`` check for ``type_`` **and** every callable in ``alpha_tests``
    returns ``True``.  Cross-fact constraints are declared as :class:`JoinSpec`
    entries in ``join_tests`` and resolved at join time.

    :param type_: the object type this pattern matches (``isinstance`` check)
    :param alpha_tests: callables ``(obj) → bool``; all must pass (alpha stage)
    :param join_tests: :class:`JoinSpec` references for cross-fact constraints
    :param bindings: ``(var_name, attr_name)`` pairs to extract on a match
    :param negated: if ``True``, production fires only when no match exists
    :see: UPDATE_PLAN §Step 2
    """

    type_: type
    alpha_tests: tuple[Callable[[Any], bool], ...] = ()
    join_tests: tuple[JoinSpec, ...] = ()
    bindings: tuple[tuple[str, str], ...] = ()
    negated: bool = False

    def matches(self, fact: Fact) -> bool:
        """Return ``True`` iff *fact* passes the type check and all alpha tests.

        :param fact: the working-memory element to test
        """
        return isinstance(fact.obj, self.type_) and all(
            t(fact.obj) for t in self.alpha_tests
        )

    def extract_bindings(self, fact: Fact) -> dict[str, Any]:
        """Return a ``{var_name: value}`` dict extracted from *fact*.

        :param fact: the matched working-memory element
        """
        return {var: getattr(fact.obj, attr) for var, attr in self.bindings}

    def alpha_key(self) -> tuple:
        """Return a hashable key identifying the alpha memory for this pattern.

        Key is ``(type_, tuple(id(fn) for fn in alpha_tests))``.  Two patterns
        with the same ``type_`` and the **same function objects** share one alpha
        memory; patterns with distinct function objects (even logically equivalent
        lambdas) do not.

        :see: UPDATE_PLAN §Key design decisions §1
        """
        return (self.type_, tuple(id(t) for t in self.alpha_tests))



@dataclass(frozen=True)
class NccGroup:
    """A group of patterns to be conjunctively negated (Doorenbos §2.8).

    The production fires only when **no** joint match of all patterns in the
    group exists in working memory.

    :see: Doorenbos §2.8
    """

    conditions: tuple[Pattern, ...]


@dataclass
class Production:
    """A production rule: an LHS list of patterns and a callable RHS.

    The RHS receives the matched :class:`~rete.fact.Token`; variable bindings
    are available via ``token.bindings``.

    :param no_loop: when ``True``, prevents self-reactivation — any new
        conflict-set entries for this production that are created during its
        own RHS execution are removed before the next cycle.

    :see: Doorenbos §2.1
    """

    lhs: list[Pattern | NccGroup]
    rhs: Callable[[Token], None]
    no_loop: bool = False
