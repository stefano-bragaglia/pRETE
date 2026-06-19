"""Beta network for the Rete algorithm: inter-element joins.

:see: Forgy §2.2.2, Doorenbos §2.4, §2.8
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rete.alpha import AlphaMemory
from rete.condition import Condition, Production
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
    parent_join: JoinNode | NegativeJoinNode | NccNode | None = field(
        default=None, repr=False
    )

    def left_activate(self, token: Token) -> None:
        """Store *token* and notify downstream join nodes.

        :param token: the new partial match
        """
        self.items.append(token)
        if token.wmes:
            token.wmes[-1].beta_tokens.append((token, self))
        for s in self.successors:
            s.left_activate(token)

    def left_retract(self, token: Token) -> None:
        """Remove *token* and notify downstream join nodes.

        The guard on ``token not in self.items`` makes this idempotent:
        cascading retraction paths (e.g. NJN / NCC propagating unextended
        tokens) can reach the same node twice via ``beta_tokens``.

        :param token: the partial match being retracted
        """
        if token not in self.items:
            return
        if token.wmes:
            token.wmes[-1].beta_tokens.remove((token, self))
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
        for token, mem in list(wme.beta_tokens):
            mem.left_retract(token)

    def left_retract(self, token: Token) -> None:
        """Handle removal of a token from the beta memory.

        :param token: the partial match being retracted
        """
        # ponytail: O(n) scan per child; upgrade to wme.beta_tokens in Phase 6.
        for child in self.children:
            for extended in list(child.items):
                if extended.wmes[:-1] == token.wmes:
                    child.left_retract(extended)

    def update_child(self, new_child: BetaMemory | PNode) -> None:
        """Initialise *new_child* with all matches already held by this node.

        Called immediately after attaching a new downstream child so it
        catches up with WMEs already in the network.

        :param new_child: a newly created :class:`BetaMemory` or :class:`PNode`
        :see: Doorenbos §2.6 ``update-new-node-with-matches-from-above``
        """
        for token in self.beta_memory.items:
            for wme in self.alpha_memory.items:
                if self._passes_tests(token, wme):
                    new_child.left_activate(Token(wmes=token.wmes + (wme,)))

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


@dataclass
class NegativeToken:
    """Pairs a left-input token with its count of blocking right WMEs.

    :see: Doorenbos §2.7
    """

    token: Token
    count: int = 0


@dataclass
class NegativeJoinNode:
    """Negated join node: propagates a token downstream only while count is zero.

    On WME add the count of matching tokens is incremented (possibly retracting
    tokens); on WME remove the count is decremented (possibly re-asserting them).

    :see: Forgy §2.3, Doorenbos §2.7
    """

    children: list = field(default_factory=list, repr=False)
    alpha_memory: AlphaMemory = field(default_factory=AlphaMemory)
    left_input: BetaMemory | DummyTopNode = field(default_factory=DummyTopNode)
    tests: list[JoinTest] = field(default_factory=list)
    items: list[NegativeToken] = field(default_factory=list)

    def left_activate(self, token: Token) -> None:
        """Handle a new token from the left input.

        :param token: the partial match entering from the left
        """
        count = self._count_matches(token)
        neg_tok = NegativeToken(token=token, count=count)
        self.items.append(neg_tok)
        if count == 0:
            for child in self.children:
                child.left_activate(token)

    def right_activate(self, wme: WME) -> None:
        """Handle a new WME arriving in the alpha memory (right input).

        :param wme: the WME that just entered the alpha memory
        """
        for neg_tok in self.items:
            if self._passes_tests(neg_tok.token, wme):
                if neg_tok.count == 0:
                    for child in self.children:
                        child.left_retract(neg_tok.token)
                neg_tok.count += 1

    def right_retract(self, wme: WME) -> None:
        """Handle removal of a WME from the alpha memory.

        :param wme: the WME being retracted
        """
        for neg_tok in self.items:
            if self._passes_tests(neg_tok.token, wme):
                neg_tok.count -= 1
                if neg_tok.count == 0:
                    for child in self.children:
                        child.left_activate(neg_tok.token)

    def left_retract(self, token: Token) -> None:
        """Handle removal of a token from the left input.

        :param token: the partial match being retracted
        """
        neg_tok = next(nt for nt in self.items if nt.token is token)
        if neg_tok.count == 0:
            for child in self.children:
                child.left_retract(token)
        self.items.remove(neg_tok)

    def update_child(self, child: object) -> None:
        """Seed *child* with all currently propagated tokens (count == 0).

        :param child: a newly attached downstream node
        :see: Doorenbos §2.6 ``update-new-node-with-matches-from-above``
        """
        for neg_tok in self.items:
            if neg_tok.count == 0:
                child.left_activate(neg_tok.token)

    def _count_matches(self, token: Token) -> int:
        """Count WMEs in the alpha memory that pass all join tests for *token*.

        :param token: the left-input token to test against
        """
        return sum(
            1 for wme in self.alpha_memory.items if self._passes_tests(token, wme)
        )

    def _passes_tests(self, token: Token, wme: WME) -> bool:
        """Return ``True`` iff *wme* is consistent with *token* for all join tests.

        :param token: existing partial match
        :param wme: candidate WME to test
        """
        for test in self.tests:
            wme_val = getattr(wme, test.field_of_wme)
            tok_val = getattr(token.wmes[test.condition_index], test.field_of_token_wme)
            if wme_val != tok_val:
                return False
        return True

    def _initialize_from(self, tokens: object) -> None:
        """Seed this node by left-activating each token in *tokens*.

        :param tokens: an iterable of :class:`Token` objects
        """
        for t in tokens:
            self.left_activate(t)


@dataclass
class NccToken:
    """Left token paired with its count of subnetwork matches and their result tokens.

    ``results`` holds partner tokens that extend this owner, enabling direct
    cleanup in :meth:`NccNode.left_retract` without a full partner scan.

    :see: Doorenbos §2.8
    """

    token: Token
    count: int = 0
    results: list[Token] = field(default_factory=list)


@dataclass
class NccPartnerNode:
    """Sink at the end of the NCC subnetwork; delegates to :class:`NccNode`.

    Stores received result tokens in ``items`` and forwards count bookkeeping
    to ``ncc_node``.  ``sub_last_join`` is set by
    :meth:`ReteNetwork._build_ncc` immediately after wiring, and is used by
    :meth:`ReteNetwork._gc_ncc_node` to unlink the subnetwork.

    :see: Doorenbos §2.8
    """

    items: list[Token] = field(default_factory=list)
    ncc_node: NccNode | None = field(default=None, repr=False)
    sub_last_join: JoinNode | None = field(default=None, repr=False)

    def left_activate(self, token: Token) -> None:
        """Forward *token* to the NCC node, buffering if no owner token yet.

        Registers in ``token.wmes[-1].beta_tokens`` so that
        :meth:`JoinNode.right_retract` can reach this node when the
        subnetwork WME is later removed.

        :param token: the result token from the subnetwork
        """
        self.items.append(token)
        if token.wmes:
            token.wmes[-1].beta_tokens.append((token, self))
        owner_wmes = token.wmes[: self.ncc_node.owner_length]
        ncc_tok = self._find_ncc_token(owner_wmes)
        if ncc_tok:
            ncc_tok.results.append(token)
            if ncc_tok.count == 0:
                for child in self.ncc_node.children:
                    child.left_retract(ncc_tok.token)
            ncc_tok.count += 1
        else:
            self.ncc_node.new_result_buffer.append(token)

    def left_retract(self, token: Token) -> None:
        """Decrement the owner's count; re-assert downstream if count reaches zero.

        :param token: the result token being retracted from the subnetwork
        """
        self.items.remove(token)
        if token.wmes:
            token.wmes[-1].beta_tokens.remove((token, self))
        owner_wmes = token.wmes[: self.ncc_node.owner_length]
        ncc_tok = self._find_ncc_token(owner_wmes)
        if ncc_tok:
            ncc_tok.results.remove(token)
            ncc_tok.count -= 1
            if ncc_tok.count == 0:
                for child in self.ncc_node.children:
                    child.left_activate(ncc_tok.token)
        else:
            self.ncc_node.new_result_buffer.remove(token)

    def _find_ncc_token(self, owner_wmes: tuple) -> NccToken | None:
        """Return the :class:`NccToken` matching *owner_wmes*, or ``None``.

        :param owner_wmes: prefix slice of the result token's WMEs
        """
        return next(
            (nt for nt in self.ncc_node.items if nt.token.wmes == owner_wmes),
            None,
        )


@dataclass
class NccNode:
    """Consumer on the main join chain for a negated conjunctive condition group.

    Uses count-based blocking (like :class:`NegativeJoinNode`) but receives
    results from :class:`NccPartnerNode` instead of directly from an alpha
    memory.  ``new_result_buffer`` holds partner results that arrived before
    the corresponding :class:`NccToken` was created (initialisation ordering).

    :see: Doorenbos §2.8
    """

    children: list = field(default_factory=list, repr=False)
    partner: NccPartnerNode | None = field(default=None, repr=False)
    items: list[NccToken] = field(default_factory=list)
    new_result_buffer: list[Token] = field(default_factory=list)
    owner_length: int = 0
    left_input: BetaMemory | DummyTopNode = field(default_factory=DummyTopNode)

    def left_activate(self, token: Token) -> None:
        """Create an :class:`NccToken`, draining any buffered results for *token*.

        :param token: the left token arriving from the main chain
        """
        results = self._drain_buffer(token)
        ncc_tok = NccToken(token=token, count=len(results), results=results)
        self.items.append(ncc_tok)
        if ncc_tok.count == 0:
            for child in self.children:
                child.left_activate(token)

    def left_retract(self, token: Token) -> None:
        """Remove the :class:`NccToken` for *token* and retract children if needed.

        :param token: the left token being retracted
        """
        ncc_tok = self._find_ncc_token(token)
        self._retract_partner_results(ncc_tok)
        self.items.remove(ncc_tok)
        if ncc_tok.count == 0:
            for child in self.children:
                child.left_retract(token)

    def update_child(self, child: object) -> None:
        """Seed *child* with all currently propagated tokens (count == 0).

        :param child: a newly attached downstream node
        :see: Doorenbos §2.6 ``update-new-node-with-matches-from-above``
        """
        for ncc_tok in self.items:
            if ncc_tok.count == 0:
                child.left_activate(ncc_tok.token)

    def _drain_buffer(self, token: Token) -> list[Token]:
        """Partition ``new_result_buffer`` by *token*; return the matching results.

        :param token: the left token whose owner WMEs to match against
        """
        keep, drain = [], []
        for r in self.new_result_buffer:
            if r.wmes[: self.owner_length] == token.wmes:
                drain.append(r)
            else:
                keep.append(r)
        self.new_result_buffer[:] = keep
        return drain

    def _find_ncc_token(self, token: Token) -> NccToken:
        """Return the :class:`NccToken` whose ``.token`` is *token* (identity check).

        :param token: the left token to look up
        """
        return next(nt for nt in self.items if nt.token is token)

    def _retract_partner_results(self, ncc_tok: NccToken) -> None:
        """Remove all result tokens in *ncc_tok* from the partner's ``items``.

        :param ncc_tok: the NCC token whose results are being cleaned up
        """
        for result in list(ncc_tok.results):
            self.partner.items.remove(result)
        ncc_tok.results.clear()


@dataclass
class Instantiation:
    """A production–token pair representing one full match in the conflict set.

    :see: Doorenbos §2.1
    """

    production: Production
    token: Token


@dataclass
class PNode:
    """Terminal node: records full matches in the shared conflict set.

    Sits at the end of a join chain (Doorenbos §2.4).  A token arriving via
    ``left_activate`` represents a complete production match; it is stored in
    ``items`` (so ``JoinNode`` retract scans can reach it) and added to the
    shared ``conflict_set`` as an :class:`Instantiation`.

    :see: Doorenbos §2.4, §2.6
    """

    production: Production
    conflict_set: list[Instantiation]
    items: list[Token] = field(default_factory=list)
    parent_join: JoinNode | NegativeJoinNode | NccNode | None = field(
        default=None, repr=False
    )

    def left_activate(self, token: Token) -> None:
        """Record a full match.

        :param token: the complete match token arriving from the last join node
        """
        self.items.append(token)
        if token.wmes:
            token.wmes[-1].beta_tokens.append((token, self))
        self.conflict_set.append(Instantiation(self.production, token))

    def left_retract(self, token: Token) -> None:
        """Remove a previously recorded match.

        Idempotent: a no-op if *token* is no longer present (see
        :meth:`BetaMemory.left_retract` for why this guard is needed).

        :param token: the complete match token being retracted
        """
        if token not in self.items:
            return
        if token.wmes:
            token.wmes[-1].beta_tokens.remove((token, self))
        self.items.remove(token)
        # Guard: instantiation may have been removed already by the engine loop.
        inst = Instantiation(self.production, token)
        if inst in self.conflict_set:
            self.conflict_set.remove(inst)


