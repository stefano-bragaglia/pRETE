"""Rete network: production add / remove and alpha/beta network construction.

:see: Doorenbos §2.6, Appendix A
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
    NegativeJoinNode,
    PNode,
)
from rete.condition import Condition, Production
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
        jn = self._build_join_chain(production.lhs)
        pn = PNode(
            production=production, conflict_set=self.conflict_set, parent_join=jn
        )
        jn.children.append(pn)
        jn.update_child(pn)
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
        else:
            self._gc_join_node(jn)

    # ------------------------------------------------------------------
    # Network construction helpers (Doorenbos §2.6)
    # ------------------------------------------------------------------

    def _build_join_chain(
        self, lhs: list[Condition]
    ) -> JoinNode | NegativeJoinNode:
        """Build or share the join-node chain for *lhs*.

        :param lhs: ordered list of conditions from the production LHS
        :returns: the last join node (positive or negative) in the compiled chain
        :see: Doorenbos §2.6 ``build-or-share-network-for-conditions``
        """
        left: BetaMemory | DummyTopNode = self.dummy_top
        earlier: list[Condition] = []
        last = None
        for i, cond in enumerate(lhs):
            am = self.root.build_or_share_alpha_memory(cond)
            tests = JoinTest.extract(cond, earlier)
            if cond.negated:
                last = self._build_or_share_negative_join_node(left, am, tests)
            else:
                last = self._build_or_share_join_node(left, am, tests)
            earlier.append(cond)
            if i < len(lhs) - 1:
                left = self._build_or_share_beta_memory(last)
        return last

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
        self, parent_join: JoinNode | NegativeJoinNode
    ) -> BetaMemory:
        """Return the existing BetaMemory child of *parent_join* or create one.

        :param parent_join: the upstream join node
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
        else:
            self._gc_join_node(bm.parent_join)
