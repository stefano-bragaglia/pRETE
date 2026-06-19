"""Condition and Production types for the Rete algorithm LHS.

:see: Doorenbos §2.1, §2.2, §2.8
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rete.wme import Token, WME

# Sentinel: match any field value without binding (Doorenbos §2.2).
# Variables use the '?' prefix convention; binding is resolved in join nodes (Phase 3).
WILDCARD: object = object()


@dataclass(frozen=True)
class Condition:
    """A triple of field tests for one pattern in a production's LHS.

    Each field is a string constant, a ``'?'``-prefixed variable name, or
    ``WILDCARD``. Variable binding is resolved by join nodes (Phase 3).
    When ``negated`` is ``True`` the condition must *not* be satisfied by
    any WME in working memory for the production to fire (Doorenbos §2.7).

    :see: Doorenbos §2.1, §2.7
    """

    id_test: object
    attribute_test: object
    value_test: object
    negated: bool = False

    @staticmethod
    def _is_constant(test: object) -> bool:
        """Return ``True`` iff *test* is a plain constant (not wildcard, not variable).

        :param test: a condition field test value
        """
        return isinstance(test, str) and not test.startswith("?")

    @staticmethod
    def _field_matches(test: object, value: str) -> bool:
        """Test a single field against a candidate value.

        :param test: a string constant, ``'?'``-prefixed variable, or ``WILDCARD``
        :param value: the WME field value to test against
        :returns: ``True`` if the test passes
        """
        return (
            test is WILDCARD
            or (isinstance(test, str) and test.startswith("?"))
            or test == value
        )

    def alpha_tests(self) -> list[tuple[str, str]]:
        """Return ``(field, symbol)`` pairs for each constant field in this condition.

        :see: Doorenbos §2.2
        """
        all_tests = [
            ("id", self.id_test),
            ("attribute", self.attribute_test),
            ("value", self.value_test),
        ]
        return [(f, s) for f, s in all_tests if self._is_constant(s)]  # type: ignore[return-value]

    def matches(self, wme: WME) -> bool:
        """Return ``True`` iff all constant tests in this condition pass for ``wme``.

        :param wme: the working memory element to test
        :see: Doorenbos §2.2
        """
        return (
            self._field_matches(self.id_test, wme.id)
            and self._field_matches(self.attribute_test, wme.attribute)
            and self._field_matches(self.value_test, wme.value)
        )


@dataclass(frozen=True)
class NccGroup:
    """A group of conditions to be conjunctively negated (Doorenbos §2.8).

    The production fires only when **no** joint match of all conditions in
    the group exists in working memory.

    :see: Doorenbos §2.8
    """

    conditions: tuple[Condition, ...]


@dataclass
class Production:
    """A production rule: an LHS list of conditions and a callable RHS.

    The RHS receives the matched :class:`Token`; variable bindings are
    derived by pairing the token's WMEs with the LHS conditions.

    :see: Doorenbos §2.1
    """

    lhs: list[Condition | NccGroup]
    rhs: Callable[[Token], None]
