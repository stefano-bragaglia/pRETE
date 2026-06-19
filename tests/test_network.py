"""Unit tests for network.py — ReteNetwork.

:see: Doorenbos §2.6, Appendix A
"""
from rete.beta import BetaMemory, JoinNode, PNode
from rete.condition import Condition, Production
from rete.network import ReteNetwork
from rete.wme import WME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prod(lhs: list[Condition]) -> Production:
    return Production(lhs=lhs, rhs=lambda t: None)


# ---------------------------------------------------------------------------
# add_production — structure
# ---------------------------------------------------------------------------


def test_add_production_returns_pnode():
    net = ReteNetwork()
    pn = net.add_production(_prod([Condition("b1", "color", "red")]))
    assert isinstance(pn, PNode)


def test_add_production_pnode_in_join_children():
    net = ReteNetwork()
    pn = net.add_production(_prod([Condition("b1", "color", "red")]))
    assert pn in pn.parent_join.children


def test_add_production_join_in_alpha_successors():
    net = ReteNetwork()
    pn = net.add_production(_prod([Condition("b1", "color", "red")]))
    jn = pn.parent_join
    assert jn in jn.alpha_memory.successors


def test_add_production_two_conditions_creates_beta_memory():
    net = ReteNetwork()
    pn = net.add_production(
        _prod([Condition("b1", "color", "red"), Condition("b1", "size", "large")])
    )
    assert isinstance(pn.parent_join.beta_memory, BetaMemory)


def test_add_production_pnode_parent_join_set():
    net = ReteNetwork()
    pn = net.add_production(_prod([Condition("b1", "color", "red")]))
    assert isinstance(pn.parent_join, JoinNode)


def test_add_production_conflict_set_empty_on_empty_wm():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red")]))
    assert net.conflict_set == []


# ---------------------------------------------------------------------------
# add_production — sharing (Doorenbos §2.6)
# ---------------------------------------------------------------------------


def test_add_two_productions_share_alpha_memory():
    net = ReteNetwork()
    c = Condition("b1", "color", "red")
    pn1 = net.add_production(_prod([c]))
    pn2 = net.add_production(_prod([c]))
    assert pn1.parent_join.alpha_memory is pn2.parent_join.alpha_memory


def test_add_two_productions_share_join_node():
    net = ReteNetwork()
    c = Condition("b1", "color", "red")
    pn1 = net.add_production(_prod([c]))
    pn2 = net.add_production(_prod([c]))
    assert pn1.parent_join is pn2.parent_join


def test_add_two_productions_share_beta_memory():
    net = ReteNetwork()
    c0 = Condition("b1", "color", "red")
    pn1 = net.add_production(_prod([c0, Condition("b1", "size", "large")]))
    pn2 = net.add_production(_prod([c0, Condition("b2", "weight", "heavy")]))
    assert pn1.parent_join.beta_memory is pn2.parent_join.beta_memory


def test_add_two_productions_independent_pnodes():
    net = ReteNetwork()
    c = Condition("b1", "color", "red")
    pn1 = net.add_production(_prod([c]))
    pn2 = net.add_production(_prod([c]))
    assert pn1 is not pn2


# ---------------------------------------------------------------------------
# add_production — replay (Doorenbos §2.6 update-new-node-with-matches-from-above)
# ---------------------------------------------------------------------------


def test_add_production_replays_matching_wme():
    # Add p1 to create network structure, activate a WME, then add p2: it replays.
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red")]))
    w = WME("b1", "color", "red")
    net.root.activate(w)
    assert len(net.conflict_set) == 1
    net.add_production(_prod([Condition("b1", "color", "red")]))
    assert len(net.conflict_set) == 2


def test_add_production_no_replay_on_non_matching_wme():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red")]))
    net.root.activate(WME("b1", "color", "blue"))  # wrong value
    net.add_production(_prod([Condition("b1", "color", "red")]))
    assert net.conflict_set == []


def test_add_production_replays_two_condition_match():
    net = ReteNetwork()
    c0 = Condition("?x", "color", "red")
    c1 = Condition("?x", "size", "large")
    net.add_production(_prod([c0, c1]))
    w1, w2 = WME("b1", "color", "red"), WME("b1", "size", "large")
    net.root.activate(w1)
    net.root.activate(w2)
    assert len(net.conflict_set) == 1
    net.add_production(_prod([c0, c1]))
    assert len(net.conflict_set) == 2


# ---------------------------------------------------------------------------
# remove_production (Doorenbos §2.6 / Appendix A)
# ---------------------------------------------------------------------------


def test_remove_production_pnode_removed_from_join():
    net = ReteNetwork()
    pn = net.add_production(_prod([Condition("b1", "color", "red")]))
    jn = pn.parent_join
    net.remove_production(pn)
    assert pn not in jn.children


def test_remove_production_clears_conflict_set():
    net = ReteNetwork()
    pn = net.add_production(_prod([Condition("b1", "color", "red")]))
    net.root.activate(WME("b1", "color", "red"))
    assert len(net.conflict_set) == 1
    net.remove_production(pn)
    assert net.conflict_set == []


def test_remove_production_gc_orphan_join_node():
    net = ReteNetwork()
    pn = net.add_production(_prod([Condition("b1", "color", "red")]))
    jn, am = pn.parent_join, pn.parent_join.alpha_memory
    net.remove_production(pn)
    assert jn not in am.successors


def test_remove_production_gc_orphan_beta_memory():
    net = ReteNetwork()
    pn = net.add_production(
        _prod([Condition("b1", "color", "red"), Condition("b1", "size", "large")])
    )
    jn2 = pn.parent_join
    bm = jn2.beta_memory
    assert isinstance(bm, BetaMemory)
    jn1 = bm.parent_join
    net.remove_production(pn)
    assert bm not in jn1.children


def test_remove_production_preserves_shared_join_node():
    net = ReteNetwork()
    c = Condition("b1", "color", "red")
    pn1 = net.add_production(_prod([c]))
    net.add_production(_prod([c]))
    jn = pn1.parent_join
    net.remove_production(pn1)
    assert jn in jn.alpha_memory.successors


def test_remove_production_preserves_shared_beta_memory():
    net = ReteNetwork()
    c0 = Condition("b1", "color", "red")
    pn1 = net.add_production(_prod([c0, Condition("b1", "size", "large")]))
    net.add_production(_prod([c0, Condition("b2", "weight", "heavy")]))
    bm = pn1.parent_join.beta_memory
    assert isinstance(bm, BetaMemory)
    net.remove_production(pn1)
    assert bm in bm.parent_join.children


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def test_integration_add_production_then_activate_wme():
    net = ReteNetwork()
    p = _prod([Condition("b1", "color", "red")])
    net.add_production(p)
    w = WME("b1", "color", "red")
    net.root.activate(w)
    assert len(net.conflict_set) == 1
    assert net.conflict_set[0].token.wmes == (w,)
    assert net.conflict_set[0].production is p


def test_integration_activate_wme_then_add_production():
    # Pre-build alpha network so root.activate routes the WME into the memory.
    net = ReteNetwork()
    net.root.build_or_share_alpha_memory(Condition("b1", "color", "red"))
    w = WME("b1", "color", "red")
    net.root.activate(w)
    p = _prod([Condition("b1", "color", "red")])
    net.add_production(p)
    assert len(net.conflict_set) == 1
    assert net.conflict_set[0].token.wmes == (w,)


def test_integration_two_conditions_activation():
    net = ReteNetwork()
    p = _prod([Condition("?x", "color", "red"), Condition("?x", "size", "large")])
    net.add_production(p)
    w1, w2 = WME("b1", "color", "red"), WME("b1", "size", "large")
    net.root.activate(w1)
    net.root.activate(w2)
    assert len(net.conflict_set) == 1
    assert net.conflict_set[0].token.wmes == (w1, w2)


def test_integration_remove_then_activate():
    net = ReteNetwork()
    pn = net.add_production(_prod([Condition("b1", "color", "red")]))
    net.remove_production(pn)
    net.root.activate(WME("b1", "color", "red"))
    assert net.conflict_set == []


# ---------------------------------------------------------------------------
# Phase 6 — add_wme / remove_wme (Doorenbos §2.5)
# ---------------------------------------------------------------------------


def test_add_wme_creates_match():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red")]))
    net.add_wme(WME("b1", "color", "red"))
    assert len(net.conflict_set) == 1


def test_add_wme_no_match():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red")]))
    net.add_wme(WME("b1", "color", "blue"))
    assert net.conflict_set == []


def test_add_wme_populates_alpha_memories():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red")]))
    w = WME("b1", "color", "red")
    net.add_wme(w)
    assert w.alpha_memories


def test_add_wme_populates_beta_tokens():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red")]))
    w = WME("b1", "color", "red")
    net.add_wme(w)
    assert w.beta_tokens


def test_add_wme_two_conditions():
    net = ReteNetwork()
    net.add_production(
        _prod([Condition("?x", "color", "red"), Condition("?x", "size", "large")])
    )
    w1, w2 = WME("b1", "color", "red"), WME("b1", "size", "large")
    net.add_wme(w1)
    net.add_wme(w2)
    assert len(net.conflict_set) == 1


def test_remove_wme_clears_conflict_set():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red")]))
    w = WME("b1", "color", "red")
    net.add_wme(w)
    assert len(net.conflict_set) == 1
    net.remove_wme(w)
    assert net.conflict_set == []


def test_remove_wme_clears_alpha_memories():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red")]))
    w = WME("b1", "color", "red")
    net.add_wme(w)
    net.remove_wme(w)
    assert w.alpha_memories == []


def test_remove_wme_clears_beta_tokens():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red")]))
    w = WME("b1", "color", "red")
    net.add_wme(w)
    net.remove_wme(w)
    assert w.beta_tokens == []


def test_remove_wme_partial_two_condition_match():
    net = ReteNetwork()
    net.add_production(
        _prod([Condition("?x", "color", "red"), Condition("?x", "size", "large")])
    )
    w1, w2 = WME("b1", "color", "red"), WME("b1", "size", "large")
    net.add_wme(w1)
    net.add_wme(w2)
    assert len(net.conflict_set) == 1
    net.remove_wme(w1)
    assert net.conflict_set == []


def test_remove_wme_second_of_two_conditions():
    net = ReteNetwork()
    net.add_production(
        _prod([Condition("?x", "color", "red"), Condition("?x", "size", "large")])
    )
    w1, w2 = WME("b1", "color", "red"), WME("b1", "size", "large")
    net.add_wme(w1)
    net.add_wme(w2)
    net.remove_wme(w2)
    assert net.conflict_set == []


def test_remove_wme_shared_production():
    net = ReteNetwork()
    c = Condition("b1", "color", "red")
    net.add_production(_prod([c]))
    net.add_production(_prod([c]))
    w = WME("b1", "color", "red")
    net.add_wme(w)
    assert len(net.conflict_set) == 2
    net.remove_wme(w)
    assert net.conflict_set == []


def test_remove_wme_idempotent_after_retract():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red")]))
    w = WME("b1", "color", "blue")
    net.add_wme(w)
    net.remove_wme(w)
    assert net.conflict_set == []


# ---------------------------------------------------------------------------
# Phase 7 — negated conditions (Doorenbos §2.7)
# ---------------------------------------------------------------------------


def test_negated_only_condition_fires_with_no_wme():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red", negated=True)]))
    assert len(net.conflict_set) == 1


def test_negated_only_condition_blocked_by_wme():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red", negated=True)]))
    net.add_wme(WME("b1", "color", "red"))
    assert net.conflict_set == []


def test_negated_condition_blocking_then_retracted():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red", negated=True)]))
    w = WME("b1", "color", "red")
    net.add_wme(w)
    assert net.conflict_set == []
    net.remove_wme(w)
    assert len(net.conflict_set) == 1


def test_positive_then_negated_condition_fires():
    net = ReteNetwork()
    net.add_production(
        _prod([
            Condition("b1", "color", "red"),
            Condition("b1", "size", "large", negated=True),
        ])
    )
    net.add_wme(WME("b1", "color", "red"))
    assert len(net.conflict_set) == 1


def test_positive_then_negated_condition_blocked():
    net = ReteNetwork()
    net.add_production(
        _prod([
            Condition("b1", "color", "red"),
            Condition("b1", "size", "large", negated=True),
        ])
    )
    net.add_wme(WME("b1", "color", "red"))
    net.add_wme(WME("b1", "size", "large"))
    assert net.conflict_set == []


def test_positive_then_negated_remove_blocker():
    net = ReteNetwork()
    net.add_production(
        _prod([
            Condition("b1", "color", "red"),
            Condition("b1", "size", "large", negated=True),
        ])
    )
    net.add_wme(WME("b1", "color", "red"))
    w_blocker = WME("b1", "size", "large")
    net.add_wme(w_blocker)
    assert net.conflict_set == []
    net.remove_wme(w_blocker)
    assert len(net.conflict_set) == 1


def test_multiple_blocking_wmes():
    net = ReteNetwork()
    net.add_production(_prod([Condition("b1", "color", "red", negated=True)]))
    w1 = WME("b1", "color", "red")
    w2 = WME("b1", "color", "red")
    net.add_wme(w1)
    net.add_wme(w2)
    assert net.conflict_set == []
    net.remove_wme(w1)
    assert net.conflict_set == []
    net.remove_wme(w2)
    assert len(net.conflict_set) == 1


def test_negated_condition_gc_on_remove_production():
    net = ReteNetwork()
    pn = net.add_production(_prod([Condition("b1", "color", "red", negated=True)]))
    net.remove_production(pn)
    assert net.conflict_set == []


def test_negated_shares_alpha_memory():
    net = ReteNetwork()
    pn_pos = net.add_production(_prod([Condition("b1", "color", "red")]))
    pn_neg = net.add_production(_prod([Condition("b1", "color", "red", negated=True)]))
    assert pn_pos.parent_join.alpha_memory is pn_neg.parent_join.alpha_memory
