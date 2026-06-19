"""Rete network: production add / remove and alpha/beta network construction.

:see: Doorenbos §2.6, Appendix A, §2.8
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rete.alpha import AlphaMemory, RootNode
from rete.beta import (
    BetaMemory,
    DummyTopNode,
    Instantiation,
    JoinNode,
    JoinTest,
    NccNode,
    NccPartnerNode,
    NegativeJoinNode,
    PNode,
)
from rete.condition import Condition, NccGroup, Production
from rete.wme import WME


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

    def add_wme(self, wme: WME) -> None:
        """Feed *wme* into the alpha network, triggering all matching join nodes.

        :param wme: the WME to assert
        :see: Doorenbos §2.5 ``add-wme``
        """
        self.root.activate(wme)

    def remove_wme(self, wme: WME) -> None:
        """Retract *wme* from all alpha memories and propagate through the network.

        :param wme: the WME to retract
        :see: Doorenbos §2.5 ``remove-wme``
        """
        for am in list(wme.alpha_memories):
            am.deactivate(wme)

    def add_production(self, production: Production) -> PNode:
        """Compile *production* into the network and return its terminal PNode.

        Shares existing alpha nodes, alpha memories, join nodes, and beta
        memories wherever tests are identical.  Initialises any newly created
        nodes with existing WME matches via :meth:`JoinNode.update_child`.

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
        elif isinstance(jn, NccNode):
            self._gc_ncc_node(jn)
        else:
            self._gc_join_node(jn)

    # ------------------------------------------------------------------
    # Network construction helpers (Doorenbos §2.6, §2.8)
    # ------------------------------------------------------------------

    def _build_join_chain(
        self, lhs: list[Condition | NccGroup]
    ) -> JoinNode | NegativeJoinNode | NccNode:
        """Build or share the join-node chain for *lhs*.

        :param lhs: ordered list of conditions / NCC groups from the production LHS
        :returns: the last node in the compiled chain
        :see: Doorenbos §2.6 ``build-or-share-network-for-conditions``, §2.8
        """
        left: BetaMemory | DummyTopNode = self.dummy_top
        earlier: list[Condition] = []
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
        cond: Condition,
        left: BetaMemory | DummyTopNode,
        earlier: list[Condition],
    ) -> JoinNode | NegativeJoinNode:
        """Compile one :class:`Condition` into a join node, sharing if possible.

        :param cond: the condition to compile
        :param left: current left input
        :param earlier: conditions already compiled (for variable tests)
        """
        am = self.root.build_or_share_alpha_memory(cond)
        tests = JoinTest.extract(cond, earlier)
        if cond.negated:
            return self._build_or_share_negative_join_node(left, am, tests)
        return self._build_or_share_join_node(left, am, tests)

    def _build_ncc(
        self,
        left: BetaMemory | DummyTopNode,
        earlier: list[Condition],
        group: NccGroup,
    ) -> NccNode:
        """Build an NCC node and its subnetwork for *group*.

        Partner is seeded first so any existing matches land in
        ``new_result_buffer``; the NCC node then drains that buffer as it
        processes left tokens, giving correct counts from the start.

        :param left: current left input on the main chain
        :param earlier: conditions already compiled on the main chain
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
        earlier: list[Condition],
        group: NccGroup,
    ) -> JoinNode:
        """Build the positive join-node chain for the NCC *group*'s subnetwork.

        :param left: the same left input as the NCC node (parallel branch)
        :param earlier: conditions from the main chain (variables accessible in group)
        :param group: the NCC group supplying the subnetwork conditions
        :returns: the last join node in the subnetwork
        """
        sub_left: BetaMemory | DummyTopNode = left
        sub_earlier = list(earlier)
        sub_last: JoinNode | None = None
        for i, cond in enumerate(group.conditions):
            am = self.root.build_or_share_alpha_memory(cond)
            tests = JoinTest.extract(cond, sub_earlier)
            sub_last = self._build_or_share_join_node(sub_left, am, tests)
            sub_earlier.append(cond)
            if i < len(group.conditions) - 1:
                sub_left = self._build_or_share_beta_memory(sub_last)
        return sub_last

    def _build_or_share_join_node(
        self,
        left: BetaMemory | DummyTopNode,
        am: AlphaMemory,
        tests: list[JoinTest],
    ) -> JoinNode:
        """Return a matching existing join node or create and wire a new one.

        Sharing key: ``left`` by identity, ``tests`` by value.

        :param left: the left input — a :class:`BetaMemory` or :class:`DummyTopNode`
        :param am: the alpha memory supplying WMEs (right input)
        :param tests: variable-consistency tests for this join
        :see: Doorenbos §2.6 ``build-or-share-join-node``
        """
        for jn in am.successors:
            if (
                isinstance(jn, JoinNode)
                and jn.beta_memory is left
                and jn.tests == tests
            ):
                return jn
        jn = JoinNode(children=[], alpha_memory=am, beta_memory=left, tests=tests)
        am.successors.append(jn)
        if isinstance(left, BetaMemory):
            left.successors.append(jn)
        return jn

    def _build_or_share_negative_join_node(
        self,
        left: BetaMemory | DummyTopNode,
        am: AlphaMemory,
        tests: list[JoinTest],
    ) -> NegativeJoinNode:
        """Return a matching existing NJN or create and wire a new one.

        Sharing key: ``left`` by identity, ``tests`` by value.

        :param left: the left input — a :class:`BetaMemory` or :class:`DummyTopNode`
        :param am: the alpha memory supplying WMEs (right input)
        :param tests: variable-consistency tests for this join
        :see: Doorenbos §2.7
        """
        for njn in am.successors:
            if self._njn_matches(njn, left, tests):
                return njn
        njn = NegativeJoinNode(
            children=[], alpha_memory=am, left_input=left, tests=tests
        )
        am.successors.append(njn)
        if isinstance(left, BetaMemory):
            left.successors.append(njn)
        njn._initialize_from(left.items)
        return njn

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

    def _build_or_share_beta_memory(
        self, parent_join: JoinNode | NegativeJoinNode | NccNode
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

        :param jn: the join node to potentially garbage-collect
        :see: Doorenbos Appendix A ``delete-node-and-any-unused-ancestors``
        """
        if jn.children:
            return
        jn.alpha_memory.successors.remove(jn)
        if isinstance(jn.beta_memory, BetaMemory):
            jn.beta_memory.successors.remove(jn)
            self._gc_beta_memory(jn.beta_memory)

    def _gc_negative_join_node(self, njn: NegativeJoinNode) -> None:
        """Remove *njn* from the network if it has no remaining children.

        :param njn: the negative join node to potentially garbage-collect
        :see: Doorenbos Appendix A ``delete-node-and-any-unused-ancestors``
        """
        if njn.children:
            return
        njn.alpha_memory.successors.remove(njn)
        if isinstance(njn.left_input, BetaMemory):
            njn.left_input.successors.remove(njn)
            self._gc_beta_memory(njn.left_input)

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
        partner = ncc.partner
        if partner and partner.sub_last_join:
            partner.sub_last_join.children.remove(partner)
            self._gc_join_node(partner.sub_last_join)

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
