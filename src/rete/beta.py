"""Beta network for the Rete algorithm: inter-element joins.

:see: Forgy §2.2.2, Doorenbos §2.4
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rete.alpha import AlphaMemory
from rete.condition import Condition
from rete.wme import Token, WME


@dataclass(frozen=True)
class JoinTest:
    """One variable-consistency constraint between the new WME and a token WME.

    Passes iff ``getattr(wme, field_of_wme) == getattr(token.wmes[condition_index],
    field_of_token_wme)``.

    :see: Doorenbos §2.4
    """

    field_of_wme: str
    condition_index: int
    field_of_token_wme: str

    @classmethod
    def extract(
        cls,
        condition: Condition,
        earlier_conditions: list[Condition],
    ) -> list[JoinTest]:
        """Extract variable-consistency tests for *condition*.

        For each ``'?'``-prefixed variable in *condition*, find its bindings
        in *earlier_conditions* and emit one :class:`JoinTest` per match.

        :param condition: the condition being compiled
        :param earlier_conditions: all preceding conditions in the LHS (ordered)
        :returns: list of :class:`JoinTest`; empty if no shared variables
        :see: Doorenbos §2.4
        """
        tests: list[JoinTest] = []
        for fname, val in [
            ("id", condition.id_test),
            ("attribute", condition.attribute_test),
            ("value", condition.value_test),
        ]:
            if isinstance(val, str) and val.startswith("?"):
                tests.extend(cls._tests_for_variable(val, fname, earlier_conditions))
        return tests

    @staticmethod
    def _variable_binding(var: str, condition: Condition) -> str | None:
        """Return the field name where *var* appears in *condition*, or ``None``.

        :param var: a ``'?'``-prefixed variable name
        :param condition: the condition to search
        """
        for fname, val in [
            ("id", condition.id_test),
            ("attribute", condition.attribute_test),
            ("value", condition.value_test),
        ]:
            if val == var:
                return fname
        return None

    @staticmethod
    def _tests_for_variable(
        var: str,
        field_of_wme: str,
        earlier: list[Condition],
    ) -> list[JoinTest]:
        """Emit one :class:`JoinTest` per earlier condition that binds *var*.

        :param var: the variable to look up
        :param field_of_wme: the field in the new WME where *var* appears
        :param earlier: conditions preceding the current one in the LHS
        """
        tests: list[JoinTest] = []
        for i, cond in enumerate(earlier):
            f = JoinTest._variable_binding(var, cond)
            if f is not None:
                tests.append(JoinTest(field_of_wme, i, f))
        return tests


@dataclass
class BetaMemory:
    """Stores partial-match tokens arriving from the left.

    Each join node uses one as its left input (except the first, which uses
    :class:`DummyTopNode`).

    :see: Doorenbos §2.4
    """

    items: list[Token] = field(default_factory=list)
    successors: list = field(default_factory=list, repr=False)

    def left_activate(self, token: Token) -> None:
        """Store *token* and notify downstream join nodes.

        :param token: the new partial match
        """
        self.items.append(token)
        for s in self.successors:
            s.left_activate(token)

    def left_retract(self, token: Token) -> None:
        """Remove *token* and notify downstream join nodes.

        :param token: the partial match being retracted
        """
        self.items.remove(token)
        for s in self.successors:
            s.left_retract(token)


@dataclass
class DummyTopNode:
    """Provides the single empty token that seeds the first join node.

    Its content is fixed; it is never activated or retracted.

    :see: Doorenbos §2.4
    """

    # ponytail: immutable tuple signals that this content never changes.
    items: tuple[Token, ...] = field(default_factory=lambda: (Token(),))


@dataclass
class JoinNode:
    """Inter-element join node with two inputs (left = beta, right = alpha).

    On activation, iterates the opposite memory and emits extended tokens for
    each pair that passes all variable-consistency tests.  On retraction,
    propagates removal for any previously emitted token that depended on the
    retracted element.

    :see: Doorenbos §2.4
    """

    children: list = field(default_factory=list, repr=False)
    alpha_memory: AlphaMemory = field(default_factory=AlphaMemory)
    beta_memory: BetaMemory | DummyTopNode = field(default_factory=DummyTopNode)
    tests: list[JoinTest] = field(default_factory=list)

    def right_activate(self, wme: WME) -> None:
        """Handle a new WME arriving in the alpha memory (right input).

        :param wme: the WME that just entered the alpha memory
        """
        for token in self.beta_memory.items:
            if self._passes_tests(token, wme):
                self._extend_and_emit(token, wme)

    def left_activate(self, token: Token) -> None:
        """Handle a new token arriving in the beta memory (left input).

        :param token: the partial match that just entered the beta memory
        """
        for wme in self.alpha_memory.items:
            if self._passes_tests(token, wme):
                self._extend_and_emit(token, wme)

    def right_retract(self, wme: WME) -> None:
        """Handle removal of a WME from the alpha memory.

        :param wme: the WME being retracted
        """
        # ponytail: O(n) scan per child; upgrade to wme.beta_tokens in Phase 6.
        for child in self.children:
            for extended in list(child.items):
                if self._is_derived_from_wme(extended, wme):
                    child.left_retract(extended)

    def left_retract(self, token: Token) -> None:
        """Handle removal of a token from the beta memory.

        :param token: the partial match being retracted
        """
        # ponytail: O(n) scan per child; upgrade to wme.beta_tokens in Phase 6.
        for child in self.children:
            for extended in list(child.items):
                if extended.wmes[:-1] == token.wmes:
                    child.left_retract(extended)

    def _passes_tests(self, token: Token, wme: WME) -> bool:
        """Return ``True`` iff *wme* is consistent with *token* for all join tests.

        :param token: existing partial match
        :param wme: candidate WME to join
        """
        for test in self.tests:
            wme_val = getattr(wme, test.field_of_wme)
            tok_val = getattr(token.wmes[test.condition_index], test.field_of_token_wme)
            if wme_val != tok_val:
                return False
        return True

    def _extend_and_emit(self, token: Token, wme: WME) -> None:
        """Create an extended token and send it downstream.

        :param token: existing partial match
        :param wme: WME to append
        """
        extended = Token(wmes=token.wmes + (wme,))
        for child in self.children:
            child.left_activate(extended)

    @staticmethod
    def _is_derived_from_wme(token: Token, wme: WME) -> bool:
        """Return ``True`` iff *token*'s last WME is *wme* (by identity).

        :param token: candidate extended token
        :param wme: the WME being retracted
        """
        return bool(token.wmes) and token.wmes[-1] is wme


