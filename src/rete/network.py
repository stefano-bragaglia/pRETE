"""Rete network: production add / remove and alpha/beta network construction.

:see: Doorenbos §2.6, Appendix A, §2.8
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rete.alpha import AlphaMemory, RootNode
from rete.beta import (
    BetaMemory,
    DummyTopNode,
    ExistsNode,
    Instantiation,
    JoinNode,
    JoinTest,
    NccNode,
    NccPartnerNode,
    NegativeJoinNode,
    PNode,
)
from rete.condition import NccGroup, Pattern, Production
from rete.fact import Fact


@dataclass
class ReteNetwork:
    """A compiled Rete network with a shared conflict set.

    All three public fields have factory defaults, so ``ReteNetwork()``
    requires no arguments.

    :see: Doorenbos §2.6
    """

    root: RootNode = field(default_factory=RootNode)
    dummy_top: DummyTopNode = field(default_factory=DummyTopNode)
    conflict_set: list[Instantiation] = field(default_factory=list)

    def add_fact(self, fact: Fact) -> None:
        """Feed *fact* into the alpha network, triggering all matching join nodes.

        :param fact: the Fact to assert
        :see: Doorenbos §2.5 ``add-wme``
        """
        self.root.activate(fact)

    def remove_fact(self, fact: Fact) -> None:
        """Retract *fact* from all alpha memories and propagate through the network.

        :param fact: the Fact to retract
        :see: Doorenbos §2.5 ``remove-wme``
        """
        self.root.deactivate(fact)
        for am in list(fact.alpha_memories):
            am.deactivate(fact)

    def add_production(self, production: Production) -> PNode:
        """Compile *production* into the network and return its terminal PNode.

        Shares existing alpha memories, join nodes, and beta memories wherever
        pattern keys are identical.  Initialises any newly created nodes with
        existing Fact matches via :meth:`JoinNode.update_child`.

        :param production: the production to add
        :returns: the newly created :class:`PNode`
        :see: Doorenbos §2.6 ``add-production``
        """
        last = self._build_join_chain(production.lhs)
        pn = PNode(
            production=production, conflict_set=self.conflict_set, parent_join=last
        )
        last.children.append(pn)
        last.update_child(pn)
        return pn

    def remove_production(self, p_node: PNode) -> None:
        """Retract all live matches and unlink *p_node* from the network.

        Retracts every existing instantiation, removes the PNode from its
        parent join node, then garbage-collects any nodes that have no
        remaining subscribers.

        :param p_node: the :class:`PNode` returned by :meth:`add_production`
        :see: Doorenbos §2.6 ``remove-production``, Appendix A
        """
        for token in list(p_node.items):
            p_node.left_retract(token)
        jn = p_node.parent_join
        jn.children.remove(p_node)
        p_node.parent_join = None
        if isinstance(jn, NegativeJoinNode):
            self._gc_negative_join_node(jn)
        elif isinstance(jn, ExistsNode):
            self._gc_exists_node(jn)
        elif isinstance(jn, NccNode):
            self._gc_ncc_node(jn)
        else:
            self._gc_join_node(jn)

    # ------------------------------------------------------------------
    # Network construction helpers (Doorenbos §2.6, §2.8)
    # ------------------------------------------------------------------

    def _build_join_chain(
        self, lhs: list[Pattern | NccGroup]
    ) -> JoinNode | NegativeJoinNode | ExistsNode | NccNode:
        """Build or share the join-node chain for *lhs*.

        :param lhs: ordered list of patterns / NCC groups from the production LHS
        :returns: the last node in the compiled chain
        :see: Doorenbos §2.6 ``build-or-share-network-for-conditions``, §2.8
        """
        left: BetaMemory | DummyTopNode = self.dummy_top
        earlier: list[Pattern] = []
        last = None
        for i, item in enumerate(lhs):
            if isinstance(item, NccGroup):
                last = self._build_ncc(left, earlier, item)
            else:
                last = self._process_condition(item, left, earlier)
                earlier.append(item)
            if i < len(lhs) - 1:
                left = self._build_or_share_beta_memory(last)
        return last

    def _process_condition(
        self,
        pattern: Pattern,
        left: BetaMemory | DummyTopNode,
        earlier: list[Pattern],
    ) -> JoinNode | NegativeJoinNode | ExistsNode:
        """Compile one :class:`Pattern` into a join node, sharing if possible.

        :param pattern: the pattern to compile
        :param left: current left input
        :param earlier: patterns already compiled (for JoinTest derivation)
        """
        am = self.root.build_or_share_alpha_memory(pattern)
        tests = JoinTest.extract(pattern, earlier)
        if pattern.negated:
            return self._build_or_share_negative_join_node(left, am, tests)
        if pattern.exists:
            return self._build_or_share_exists_node(left, am, tests)
        return self._build_or_share_join_node(left, am, tests, pattern)

    def _build_ncc(
        self,
        left: BetaMemory | DummyTopNode,
        earlier: list[Pattern],
        group: NccGroup,
    ) -> NccNode:
        """Build an NCC node and its subnetwork for *group*.

        Partner is seeded first so any existing matches land in
        ``new_result_buffer``; the NCC node then drains that buffer as it
        processes left tokens, giving correct counts from the start.

        :param left: current left input on the main chain
        :param earlier: patterns already compiled on the main chain
        :param group: the NCC group to compile
        :see: Doorenbos §2.8
        """
        ncc_node = NccNode(owner_length=len(earlier), left_input=left)
        partner = NccPartnerNode(ncc_node=ncc_node)
        ncc_node.partner = partner
        sub_last = self._build_ncc_subnetwork(left, earlier, group)
        sub_last.children.append(partner)
        partner.sub_last_join = sub_last
        if isinstance(left, BetaMemory):
            left.successors.append(ncc_node)
        sub_last.update_child(partner)
        for token in left.items:
            ncc_node.left_activate(token)
        return ncc_node

    def _build_ncc_subnetwork(
        self,
        left: BetaMemory | DummyTopNode,
        earlier: list[Pattern],
        group: NccGroup,
    ) -> JoinNode | NegativeJoinNode:
        """Build the join-node chain for the NCC *group*'s subnetwork.

        Uses :meth:`_process_condition` so that negated patterns inside an
        NCC (needed by ``forall`` compilation) produce :class:`NegativeJoinNode`
        instead of :class:`JoinNode`.

        :param left: the same left input as the NCC node (parallel branch)
        :param earlier: patterns from the main chain (variables accessible in group)
        :param group: the NCC group supplying the subnetwork patterns
        :returns: the last join node in the subnetwork
        """
        sub_left: BetaMemory | DummyTopNode = left
        sub_earlier = list(earlier)
        sub_last: JoinNode | NegativeJoinNode | None = None
        for i, pattern in enumerate(group.conditions):
            sub_last = self._process_condition(pattern, sub_left, sub_earlier)
            sub_earlier.append(pattern)
            if i < len(group.conditions) - 1:
                sub_left = self._build_or_share_beta_memory(sub_last)
        return sub_last

    def _build_or_share_join_node(
        self,
        left: BetaMemory | DummyTopNode,
        am: AlphaMemory,
        tests: list[JoinTest],
        pattern: Pattern,
    ) -> JoinNode:
        """Return a matching existing join node or create and wire a new one.

        Sharing key: ``left`` by identity, ``tests`` by value,
        ``pattern.alpha_key()`` and ``pattern.bindings`` by value.
        The bindings component prevents aliasing two patterns that share an
        alpha memory (same type + alpha_tests) but extract different variables.

        :param left: the left input — a :class:`BetaMemory` or :class:`DummyTopNode`
        :param am: the alpha memory supplying Facts (right input)
        :param tests: variable-consistency tests for this join
        :param pattern: the :class:`Pattern` being compiled (supplies bindings)
        :see: Doorenbos §2.6 ``build-or-share-join-node``
        """
        for jn in am.successors:
            if self._jn_matches(jn, left, tests, pattern):
                return jn
        jn = JoinNode(
            children=[], alpha_memory=am, left_input=left, tests=tests, pattern=pattern
        )
        self._init_join_links(jn, left, am)
        return jn

    def _build_or_share_negative_join_node(
        self,
        left: BetaMemory | DummyTopNode,
        am: AlphaMemory,
        tests: list[JoinTest],
    ) -> NegativeJoinNode:
        """Return a matching existing NJN or create and wire a new one.

        Sharing key: ``left`` by identity, ``tests`` by value.
        NJN has no ``pattern`` field — negated conditions only block, they
        never contribute variable bindings to the token.

        :param left: the left input — a :class:`BetaMemory` or :class:`DummyTopNode`
        :param am: the alpha memory supplying Facts (right input)
        :param tests: variable-consistency tests for this join
        :see: Doorenbos §2.7
        """
        for njn in am.successors:
            if self._njn_matches(njn, left, tests):
                return njn
        njn = NegativeJoinNode(
            children=[], alpha_memory=am, left_input=left, tests=tests
        )
        self._init_njn_links(njn, left, am)
        njn._initialize_from(left.items)
        return njn

    def _init_join_links(
        self,
        jn: JoinNode,
        left: BetaMemory | DummyTopNode,
        am: AlphaMemory,
    ) -> None:
        """Set initial link state for a newly built :class:`JoinNode`.

        Right-unlink (preferred) when beta is empty: node stays in bm.successors
        so a future left activation can re-link it to am.  Left-unlink only when
        beta is non-empty and alpha is empty: node stays in am.successors.
        The two states are mutually exclusive.

        :param jn: the newly created join node
        :param left: its left input
        :param am: its alpha memory (right input)
        :see: Doorenbos Ch. 4–5
        """
        if isinstance(left, BetaMemory) and not left.items:
            jn.right_unlinked = True
            left.successors.append(jn)     # stay in bm so left_activate re-links
        elif not am.items:
            jn.left_unlinked = True
            am.successors.append(jn)        # stay in am so right_activate re-links
        else:
            am.successors.append(jn)
            if isinstance(left, BetaMemory):
                left.successors.append(jn)

    def _init_njn_links(
        self,
        njn: NegativeJoinNode,
        left: BetaMemory | DummyTopNode,
        am: AlphaMemory,
    ) -> None:
        """Set initial link state for a newly built :class:`NegativeJoinNode`.

        NJN only supports right unlinking (left unlinking would prevent propagation
        of tokens with count=0 when alpha is empty).

        :param njn: the newly created negative join node
        :param left: its left input
        :param am: its alpha memory (right input)
        :see: Doorenbos Ch. 4
        """
        if isinstance(left, BetaMemory) and not left.items:
            njn.right_unlinked = True
            left.successors.append(njn)    # stay in bm so left_activate re-links
        else:
            am.successors.append(njn)
            if isinstance(left, BetaMemory):
                left.successors.append(njn)

    def _build_or_share_exists_node(
        self,
        left: BetaMemory | DummyTopNode,
        am: AlphaMemory,
        tests: list[JoinTest],
    ) -> ExistsNode:
        """Return a matching existing ExistsNode or create and wire a new one.

        Sharing key: ``left`` by identity, ``tests`` by value.

        :param left: the left input — a :class:`BetaMemory` or :class:`DummyTopNode`
        :param am: the alpha memory supplying Facts (right input)
        :param tests: variable-consistency tests for this join
        """
        for en in am.successors:
            if self._en_matches(en, left, tests):
                return en
        en = ExistsNode(
            children=[], alpha_memory=am, left_input=left, tests=tests
        )
        self._init_exists_node_links(en, left, am)
        en._initialize_from(left.items)
        return en

    def _init_exists_node_links(
        self,
        en: ExistsNode,
        left: BetaMemory | DummyTopNode,
        am: AlphaMemory,
    ) -> None:
        """Set initial link state for a newly built :class:`ExistsNode`.

        Like :class:`NegativeJoinNode`, only right unlinking applies —
        left-unlinking would prevent receiving future right activations needed
        to trigger the 0→1 emit transition.

        :param en: the newly created ExistsNode
        :param left: its left input
        :param am: its alpha memory (right input)
        """
        if isinstance(left, BetaMemory) and not left.items:
            en.right_unlinked = True
            left.successors.append(en)
        else:
            am.successors.append(en)
            if isinstance(left, BetaMemory):
                left.successors.append(en)

    @staticmethod
    def _join_key(
        left: BetaMemory | DummyTopNode,
        tests: list[JoinTest],
        pattern: Pattern,
    ) -> tuple:
        """Return a hashable sharing key for a join node.

        Uses ``id(left)`` for identity, ``pattern.alpha_key()`` for the alpha
        memory slot, and ``pattern.bindings`` to distinguish patterns that share
        an alpha memory but extract different variables.

        :param left: left input node
        :param tests: join tests
        :param pattern: the pattern being compiled
        """
        return (id(left), tests, pattern.alpha_key(), pattern.bindings)

    @staticmethod
    def _jn_matches(
        jn: object,
        left: BetaMemory | DummyTopNode,
        tests: list[JoinTest],
        pattern: Pattern,
    ) -> bool:
        """Return ``True`` iff *jn* is a shareable :class:`JoinNode` for *pattern*.

        :param jn: candidate node from ``am.successors``
        :param left: expected left input
        :param tests: expected join tests
        :param pattern: the pattern being compiled
        """
        return (
            isinstance(jn, JoinNode)
            and jn.pattern is not None
            and ReteNetwork._join_key(jn.left_input, jn.tests, jn.pattern)
            == ReteNetwork._join_key(left, tests, pattern)
        )

    @staticmethod
    def _njn_matches(
        njn: object,
        left: BetaMemory | DummyTopNode,
        tests: list[JoinTest],
    ) -> bool:
        """Return ``True`` iff *njn* is a matching :class:`NegativeJoinNode`.

        :param njn: candidate node from ``am.successors``
        :param left: expected left input
        :param tests: expected join tests
        """
        return (
            isinstance(njn, NegativeJoinNode)
            and njn.left_input is left
            and njn.tests == tests
        )

    @staticmethod
    def _en_matches(
        en: object,
        left: BetaMemory | DummyTopNode,
        tests: list[JoinTest],
    ) -> bool:
        """Return ``True`` iff *en* is a matching :class:`ExistsNode`.

        :param en: candidate node from ``am.successors``
        :param left: expected left input
        :param tests: expected join tests
        """
        return (
            isinstance(en, ExistsNode)
            and en.left_input is left
            and en.tests == tests
        )

    def _build_or_share_beta_memory(
        self, parent_join: JoinNode | NegativeJoinNode | ExistsNode | NccNode
    ) -> BetaMemory:
        """Return the existing BetaMemory child of *parent_join* or create one.

        :param parent_join: the upstream join or NCC node
        :see: Doorenbos §2.6 ``build-or-share-beta-memory``
        """
        for child in parent_join.children:
            if isinstance(child, BetaMemory):
                return child
        bm = BetaMemory(parent_join=parent_join)
        parent_join.children.append(bm)
        parent_join.update_child(bm)
        return bm

    # ------------------------------------------------------------------
    # Garbage collection (Doorenbos Appendix A)
    # ------------------------------------------------------------------

    def _gc_join_node(self, jn: JoinNode) -> None:
        """Remove *jn* from the network if it has no remaining children.

        Guards against double-removal when UL has already unlinked the node.

        :param jn: the join node to potentially garbage-collect
        :see: Doorenbos Appendix A ``delete-node-and-any-unused-ancestors``
        """
        if jn.children:
            return
        if not jn.right_unlinked:
            jn.alpha_memory.successors.remove(jn)
        if isinstance(jn.left_input, BetaMemory):
            if not jn.left_unlinked:
                jn.left_input.successors.remove(jn)
            self._gc_beta_memory(jn.left_input)

    def _gc_negative_join_node(self, njn: NegativeJoinNode) -> None:
        """Remove *njn* from the network if it has no remaining children.

        Guards against double-removal when UL has already right-unlinked the node.

        :param njn: the negative join node to potentially garbage-collect
        :see: Doorenbos Appendix A ``delete-node-and-any-unused-ancestors``
        """
        if njn.children:
            return
        if not njn.right_unlinked:
            njn.alpha_memory.successors.remove(njn)
        if isinstance(njn.left_input, BetaMemory):
            njn.left_input.successors.remove(njn)
            self._gc_beta_memory(njn.left_input)

    def _gc_exists_node(self, en: ExistsNode) -> None:
        """Remove *en* from the network if it has no remaining children.

        :param en: the ExistsNode to potentially garbage-collect
        :see: Doorenbos Appendix A ``delete-node-and-any-unused-ancestors``
        """
        if en.children:
            return
        if not en.right_unlinked:
            en.alpha_memory.successors.remove(en)
        if isinstance(en.left_input, BetaMemory):
            en.left_input.successors.remove(en)
            self._gc_beta_memory(en.left_input)

    def _gc_ncc_node(self, ncc: NccNode) -> None:
        """Remove *ncc* from the network if it has no remaining children.

        Also unlinks the partner from the subnetwork and GCs the subnetwork.

        :param ncc: the NCC node to potentially garbage-collect
        """
        if ncc.children:
            return
        if isinstance(ncc.left_input, BetaMemory):
            ncc.left_input.successors.remove(ncc)
            self._gc_beta_memory(ncc.left_input)
        sub_last = ncc.partner.sub_last_join
        sub_last.children.remove(ncc.partner)
        if isinstance(sub_last, NegativeJoinNode):
            self._gc_negative_join_node(sub_last)
        else:
            self._gc_join_node(sub_last)

    def _gc_beta_memory(self, bm: BetaMemory) -> None:
        """Remove *bm* from the network if it has no remaining successors.

        :param bm: the beta memory to potentially garbage-collect
        :see: Doorenbos Appendix A ``delete-node-and-any-unused-ancestors``
        """
        # ponytail: alpha memory GC (clearing orphaned alpha nodes) deferred;
        # an empty alpha_memory.successors is a memory leak, not a correctness bug.
        if bm.successors or bm.parent_join is None:
            return
        bm.parent_join.children.remove(bm)
        if isinstance(bm.parent_join, NegativeJoinNode):
            self._gc_negative_join_node(bm.parent_join)
        elif isinstance(bm.parent_join, NccNode):
            self._gc_ncc_node(bm.parent_join)
        else:
            self._gc_join_node(bm.parent_join)
