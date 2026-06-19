from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rete.wme import Token, WME

# Sentinel: match any field value without binding (Doorenbos §2.2).
# Variables use the '?' prefix convention; binding is resolved in join nodes (Phase 3).
WILDCARD: object = object()


def _field_matches(test: object, value: str) -> bool:
    return (
        test is WILDCARD
        or (isinstance(test, str) and test.startswith("?"))
        or test == value
    )


def matches(condition: Condition, wme: WME) -> bool:
    """True iff all constant tests in condition pass for wme (Doorenbos §2.2)."""
    return (
        _field_matches(condition.id_test, wme.id)
        and _field_matches(condition.attribute_test, wme.attribute)
        and _field_matches(condition.value_test, wme.value)
    )


@dataclass(frozen=True)
class Condition:
    """Triple of field tests for one LHS pattern (Doorenbos §2.1)."""

    id_test: object
    attribute_test: object
    value_test: object


@dataclass
class Production:
    """A production: an LHS list of Conditions and an RHS callable (Doorenbos §2.1)."""

    lhs: list[Condition]
    rhs: Callable[[Token], None]
