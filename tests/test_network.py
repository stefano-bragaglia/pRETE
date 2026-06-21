"""Unit tests for network.py — ReteNetwork.

:see: Doorenbos §2.6, Appendix A, §2.8
"""
from dataclasses import dataclass

from rete.beta import (
    AccumulateNode,
    BetaMemory,
    ExistsNode,
    JoinNode,
    NccNode,
    NegativeJoinNode,
    PNode,
)
from rete.condition import AccumulateSpec, JoinSpec, NccGroup, Pattern, Production
from rete.fact import Fact
from rete.network import ReteNetwork


# ---------------------------------------------------------------------------
# Test dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Color:
    """Fact type for block colour tests."""

    block: str
    color: str


@dataclass
class Size:
    """Fact type for block size tests."""

    block: str
    size: str


@dataclass
class Weight:
    """Fact type for block weight tests."""

    block: str
    weight: str


@dataclass
class On:
    """Fact type for stacking / ancestry tests."""

    upper: str
    lower: str


# ---------------------------------------------------------------------------
# Module-level alpha-test functions (stable ids required for alpha_key sharing)
# ---------------------------------------------------------------------------


def _is_red(obj: Color) -> bool:
    """Alpha test: Color.color == 'red'."""
    return obj.color == "red"


def _is_large(obj: Size) -> bool:
    """Alpha test: Size.size == 'large'."""
    return obj.size == "large"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prod(lhs: list) -> Production:
    return Production(lhs=lhs, rhs=lambda t: None)


# ---------------------------------------------------------------------------
# add_production — structure
# ---------------------------------------------------------------------------


def test_add_production_returns_pnode():
    net = ReteNetwork()
    pn = net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    assert isinstance(pn, PNode)


def test_add_production_pnode_in_join_children():
    net = ReteNetwork()
    pn = net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    assert pn in pn.parent_join.children


def test_add_production_join_in_alpha_successors():
    net = ReteNetwork()
    pn = net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    jn = pn.parent_join
    assert jn in jn.alpha_memory.successors


def test_add_production_two_conditions_creates_beta_memory():
    net = ReteNetwork()
    p0 = Pattern(Color, bindings=(("$block", "block"),))
    p1 = Pattern(Size, join_tests=(JoinSpec("block", "$block"),))
    pn = net.add_production(_prod([p0, p1]))
    assert isinstance(pn.parent_join.left_input, BetaMemory)


def test_add_production_pnode_parent_join_set():
    net = ReteNetwork()
    pn = net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    assert isinstance(pn.parent_join, JoinNode)


def test_add_production_conflict_set_empty_on_empty_wm():
    net = ReteNetwork()
    net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    assert net.conflict_set == []


# ---------------------------------------------------------------------------
# add_production — sharing (Doorenbos §2.6)
# ---------------------------------------------------------------------------


def test_add_two_productions_share_alpha_memory():
    net = ReteNetwork()
    p = Pattern(Color, alpha_tests=(_is_red,))
    pn1 = net.add_production(_prod([p]))
    pn2 = net.add_production(_prod([p]))
    assert pn1.parent_join.alpha_memory is pn2.parent_join.alpha_memory


def test_add_two_productions_share_join_node():
    net = ReteNetwork()
    p = Pattern(Color, alpha_tests=(_is_red,))
    pn1 = net.add_production(_prod([p]))
    pn2 = net.add_production(_prod([p]))
    assert pn1.parent_join is pn2.parent_join


def test_add_two_productions_share_beta_memory():
    net = ReteNetwork()
    p0 = Pattern(Color, bindings=(("$block", "block"),))
    p1 = Pattern(Size, join_tests=(JoinSpec("block", "$block"),))
    p2 = Pattern(Weight, join_tests=(JoinSpec("block", "$block"),))
    pn1 = net.add_production(_prod([p0, p1]))
    pn2 = net.add_production(_prod([p0, p2]))
    assert pn1.parent_join.left_input is pn2.parent_join.left_input


def test_add_two_productions_independent_pnodes():
    net = ReteNetwork()
    p = Pattern(Color, alpha_tests=(_is_red,))
    pn1 = net.add_production(_prod([p]))
    pn2 = net.add_production(_prod([p]))
    assert pn1 is not pn2


def test_add_two_productions_different_bindings_dont_share_join():
    """Patterns with same alpha_key but different bindings get separate JoinNodes."""
    net = ReteNetwork()
    p1 = Pattern(Color, bindings=(("$block", "block"),))
    p2 = Pattern(Color, bindings=(("$color", "color"),))
    pn1 = net.add_production(_prod([p1]))
    pn2 = net.add_production(_prod([p2]))
    assert pn1.parent_join is not pn2.parent_join


# ---------------------------------------------------------------------------
# add_production — replay (Doorenbos §2.6 update-new-node-with-matches-from-above)
# ---------------------------------------------------------------------------


def test_add_production_replays_matching_fact():
    net = ReteNetwork()
    p = Pattern(Color, alpha_tests=(_is_red,))
    net.add_production(_prod([p]))
    net.add_fact(Fact(Color("b1", "red")))
    assert len(net.conflict_set) == 1
    net.add_production(_prod([p]))
    assert len(net.conflict_set) == 2


def test_add_production_no_replay_on_non_matching_fact():
    net = ReteNetwork()
    p = Pattern(Color, alpha_tests=(_is_red,))
    net.add_production(_prod([p]))
    net.add_fact(Fact(Color("b1", "blue")))  # blue ≠ red → no match
    net.add_production(_prod([p]))
    assert net.conflict_set == []


def test_add_production_replays_two_condition_match():
    net = ReteNetwork()
    p0 = Pattern(Color, bindings=(("$block", "block"),))
    p1 = Pattern(Size, join_tests=(JoinSpec("block", "$block"),))
    net.add_production(_prod([p0, p1]))
    net.add_fact(Fact(Color("b1", "red")))
    net.add_fact(Fact(Size("b1", "large")))
    assert len(net.conflict_set) == 1
    net.add_production(_prod([p0, p1]))
    assert len(net.conflict_set) == 2


# ---------------------------------------------------------------------------
# remove_production (Doorenbos §2.6 / Appendix A)
# ---------------------------------------------------------------------------


def test_remove_production_pnode_removed_from_join():
    net = ReteNetwork()
    pn = net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    jn = pn.parent_join
    net.remove_production(pn)
    assert pn not in jn.children


def test_remove_production_clears_conflict_set():
    net = ReteNetwork()
    pn = net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    net.add_fact(Fact(Color("b1", "red")))
    assert len(net.conflict_set) == 1
    net.remove_production(pn)
    assert net.conflict_set == []


def test_remove_production_gc_orphan_join_node():
    net = ReteNetwork()
    pn = net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    jn, am = pn.parent_join, pn.parent_join.alpha_memory
    net.remove_production(pn)
    assert jn not in am.successors


def test_remove_production_gc_orphan_beta_memory():
    net = ReteNetwork()
    p0 = Pattern(Color, bindings=(("$block", "block"),))
    p1 = Pattern(Size, join_tests=(JoinSpec("block", "$block"),))
    pn = net.add_production(_prod([p0, p1]))
    jn2 = pn.parent_join
    bm = jn2.left_input
    assert isinstance(bm, BetaMemory)
    jn1 = bm.parent_join
    net.remove_production(pn)
    assert bm not in jn1.children


def test_remove_production_preserves_shared_join_node():
    net = ReteNetwork()
    p = Pattern(Color, alpha_tests=(_is_red,))
    pn1 = net.add_production(_prod([p]))
    net.add_production(_prod([p]))
    jn = pn1.parent_join
    net.remove_production(pn1)
    assert jn in jn.alpha_memory.successors


def test_remove_production_preserves_shared_beta_memory():
    net = ReteNetwork()
    p0 = Pattern(Color, bindings=(("$block", "block"),))
    p1 = Pattern(Size, join_tests=(JoinSpec("block", "$block"),))
    p2 = Pattern(Weight, join_tests=(JoinSpec("block", "$block"),))
    pn1 = net.add_production(_prod([p0, p1]))
    net.add_production(_prod([p0, p2]))
    bm = pn1.parent_join.left_input
    assert isinstance(bm, BetaMemory)
    net.remove_production(pn1)
    assert bm in bm.parent_join.children


# ---------------------------------------------------------------------------
# Integration — add_fact / root.activate
# ---------------------------------------------------------------------------


def test_integration_add_production_then_add_fact():
    net = ReteNetwork()
    p = _prod([Pattern(Color, alpha_tests=(_is_red,))])
    net.add_production(p)
    f = Fact(Color("b1", "red"))
    net.add_fact(f)
    assert len(net.conflict_set) == 1
    assert net.conflict_set[0].token.facts == (f,)
    assert net.conflict_set[0].production is p


def test_integration_add_fact_then_add_production():
    net = ReteNetwork()
    # Pre-build alpha memory so the fact is stored before add_production.
    net.root.build_or_share_alpha_memory(Pattern(Color, alpha_tests=(_is_red,)))
    f = Fact(Color("b1", "red"))
    net.add_fact(f)
    p = _prod([Pattern(Color, alpha_tests=(_is_red,))])
    net.add_production(p)
    assert len(net.conflict_set) == 1
    assert net.conflict_set[0].token.facts == (f,)


def test_integration_remove_then_add_fact():
    net = ReteNetwork()
    pn = net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    net.remove_production(pn)
    net.add_fact(Fact(Color("b1", "red")))
    assert net.conflict_set == []


# ---------------------------------------------------------------------------
# add_fact / remove_fact (Doorenbos §2.5)
# ---------------------------------------------------------------------------


def test_add_fact_creates_match():
    net = ReteNetwork()
    net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    net.add_fact(Fact(Color("b1", "red")))
    assert len(net.conflict_set) == 1


def test_add_fact_no_match():
    net = ReteNetwork()
    net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    net.add_fact(Fact(Color("b1", "blue")))
    assert net.conflict_set == []


def test_add_fact_populates_alpha_memories():
    net = ReteNetwork()
    net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    f = Fact(Color("b1", "red"))
    net.add_fact(f)
    assert f.alpha_memories


def test_add_fact_populates_beta_tokens():
    net = ReteNetwork()
    net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    f = Fact(Color("b1", "red"))
    net.add_fact(f)
    assert f.beta_tokens


def test_add_fact_two_conditions():
    net = ReteNetwork()
    p0 = Pattern(Color, bindings=(("$block", "block"),))
    p1 = Pattern(Size, join_tests=(JoinSpec("block", "$block"),))
    net.add_production(_prod([p0, p1]))
    net.add_fact(Fact(Color("b1", "red")))
    net.add_fact(Fact(Size("b1", "large")))
    assert len(net.conflict_set) == 1


def test_remove_fact_clears_conflict_set():
    net = ReteNetwork()
    net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    f = Fact(Color("b1", "red"))
    net.add_fact(f)
    assert len(net.conflict_set) == 1
    net.remove_fact(f)
    assert net.conflict_set == []


def test_remove_fact_clears_alpha_memories():
    net = ReteNetwork()
    net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    f = Fact(Color("b1", "red"))
    net.add_fact(f)
    net.remove_fact(f)
    assert f.alpha_memories == []


def test_remove_fact_clears_beta_tokens():
    net = ReteNetwork()
    net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    f = Fact(Color("b1", "red"))
    net.add_fact(f)
    net.remove_fact(f)
    assert f.beta_tokens == []


def test_remove_fact_partial_two_condition_match():
    net = ReteNetwork()
    p0 = Pattern(Color, bindings=(("$block", "block"),))
    p1 = Pattern(Size, join_tests=(JoinSpec("block", "$block"),))
    net.add_production(_prod([p0, p1]))
    f1, f2 = Fact(Color("b1", "red")), Fact(Size("b1", "large"))
    net.add_fact(f1)
    net.add_fact(f2)
    assert len(net.conflict_set) == 1
    net.remove_fact(f1)
    assert net.conflict_set == []


def test_remove_fact_second_of_two_conditions():
    net = ReteNetwork()
    p0 = Pattern(Color, bindings=(("$block", "block"),))
    p1 = Pattern(Size, join_tests=(JoinSpec("block", "$block"),))
    net.add_production(_prod([p0, p1]))
    f1, f2 = Fact(Color("b1", "red")), Fact(Size("b1", "large"))
    net.add_fact(f1)
    net.add_fact(f2)
    net.remove_fact(f2)
    assert net.conflict_set == []


def test_remove_fact_shared_production():
    net = ReteNetwork()
    p = Pattern(Color, alpha_tests=(_is_red,))
    net.add_production(_prod([p]))
    net.add_production(_prod([p]))
    f = Fact(Color("b1", "red"))
    net.add_fact(f)
    assert len(net.conflict_set) == 2
    net.remove_fact(f)
    assert net.conflict_set == []


def test_remove_fact_idempotent_after_no_match():
    net = ReteNetwork()
    net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    f = Fact(Color("b1", "blue"))
    net.add_fact(f)
    net.remove_fact(f)
    assert net.conflict_set == []


# ---------------------------------------------------------------------------
# Cross-fact binding (JoinSpec / JoinTest)
# ---------------------------------------------------------------------------


def test_cross_fact_binding_join_fires():
    net = ReteNetwork()
    p0 = Pattern(On, bindings=(("$lower", "lower"),))
    p1 = Pattern(On, join_tests=(JoinSpec("upper", "$lower"),))
    net.add_production(_prod([p0, p1]))
    f1 = Fact(On("b1", "b2"))  # lower=b2 → binds $lower=b2
    f2 = Fact(On("b2", "b3"))  # upper=b2 == $lower=b2 → match
    net.add_fact(f1)
    net.add_fact(f2)
    assert len(net.conflict_set) == 1
    assert net.conflict_set[0].token.bindings == {"$lower": "b2"}


def test_cross_fact_binding_no_match():
    net = ReteNetwork()
    p0 = Pattern(On, bindings=(("$lower", "lower"),))
    p1 = Pattern(On, join_tests=(JoinSpec("upper", "$lower"),))
    net.add_production(_prod([p0, p1]))
    f1 = Fact(On("b1", "b2"))  # binds $lower=b2
    f2 = Fact(On("b3", "b4"))  # upper=b3 ≠ $lower=b2 → no match for f1 token
    net.add_fact(f1)
    net.add_fact(f2)
    assert net.conflict_set == []


def test_cross_fact_binding_retract_removes_match():
    net = ReteNetwork()
    p0 = Pattern(On, bindings=(("$lower", "lower"),))
    p1 = Pattern(On, join_tests=(JoinSpec("upper", "$lower"),))
    net.add_production(_prod([p0, p1]))
    f1 = Fact(On("b1", "b2"))
    f2 = Fact(On("b2", "b3"))
    net.add_fact(f1)
    net.add_fact(f2)
    assert len(net.conflict_set) == 1
    net.remove_fact(f1)
    assert net.conflict_set == []


# ---------------------------------------------------------------------------
# Negated conditions (Doorenbos §2.7)
# ---------------------------------------------------------------------------


def test_negated_only_condition_fires_with_no_fact():
    net = ReteNetwork()
    net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,), negated=True)]))
    assert len(net.conflict_set) == 1


def test_negated_only_condition_blocked_by_fact():
    net = ReteNetwork()
    net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,), negated=True)]))
    net.add_fact(Fact(Color("b1", "red")))
    assert net.conflict_set == []


def test_negated_condition_blocking_then_retracted():
    net = ReteNetwork()
    net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,), negated=True)]))
    f = Fact(Color("b1", "red"))
    net.add_fact(f)
    assert net.conflict_set == []
    net.remove_fact(f)
    assert len(net.conflict_set) == 1


def test_positive_then_negated_condition_fires():
    net = ReteNetwork()
    net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        Pattern(Size, alpha_tests=(_is_large,), negated=True),
    ]))
    net.add_fact(Fact(Color("b1", "red")))
    assert len(net.conflict_set) == 1


def test_positive_then_negated_condition_blocked():
    net = ReteNetwork()
    net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        Pattern(Size, alpha_tests=(_is_large,), negated=True),
    ]))
    net.add_fact(Fact(Color("b1", "red")))
    net.add_fact(Fact(Size("b1", "large")))
    assert net.conflict_set == []


def test_positive_then_negated_remove_blocker():
    net = ReteNetwork()
    net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        Pattern(Size, alpha_tests=(_is_large,), negated=True),
    ]))
    net.add_fact(Fact(Color("b1", "red")))
    f_block = Fact(Size("b1", "large"))
    net.add_fact(f_block)
    assert net.conflict_set == []
    net.remove_fact(f_block)
    assert len(net.conflict_set) == 1


def test_multiple_blocking_facts():
    net = ReteNetwork()
    net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,), negated=True)]))
    f1 = Fact(Color("b1", "red"))
    f2 = Fact(Color("b2", "red"))
    net.add_fact(f1)
    net.add_fact(f2)
    assert net.conflict_set == []
    net.remove_fact(f1)
    assert net.conflict_set == []
    net.remove_fact(f2)
    assert len(net.conflict_set) == 1


def test_negated_condition_gc_on_remove_production():
    net = ReteNetwork()
    p = Pattern(Color, alpha_tests=(_is_red,), negated=True)
    pn = net.add_production(_prod([p]))
    net.remove_production(pn)
    assert net.conflict_set == []


def test_negated_shares_alpha_memory():
    net = ReteNetwork()
    pn_pos = net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,))]))
    p_neg = Pattern(Color, alpha_tests=(_is_red,), negated=True)
    pn_neg = net.add_production(_prod([p_neg]))
    assert pn_pos.parent_join.alpha_memory is pn_neg.parent_join.alpha_memory


# ---------------------------------------------------------------------------
# Bug regression — double-retraction when removing a positive fact
# ---------------------------------------------------------------------------


def test_remove_positive_fact_with_active_match():
    """Removing the positive fact of a [pos, neg] match must not raise."""
    net = ReteNetwork()
    net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        Pattern(Size, alpha_tests=(_is_large,), negated=True),
    ]))
    f = Fact(Color("b1", "red"))
    net.add_fact(f)
    assert len(net.conflict_set) == 1
    net.remove_fact(f)
    assert net.conflict_set == []


# ---------------------------------------------------------------------------
# Conjunctive negations (Doorenbos §2.8)
# ---------------------------------------------------------------------------


def test_ncc_no_match_fires():
    net = ReteNetwork()
    net.add_production(_prod([NccGroup((Pattern(Color, alpha_tests=(_is_red,)),))]))
    assert len(net.conflict_set) == 1


def test_ncc_blocked_by_match():
    net = ReteNetwork()
    net.add_production(_prod([NccGroup((Pattern(Color, alpha_tests=(_is_red,)),))]))
    net.add_fact(Fact(Color("b1", "red")))
    assert net.conflict_set == []


def test_ncc_unblocked_on_retraction():
    net = ReteNetwork()
    net.add_production(_prod([NccGroup((Pattern(Color, alpha_tests=(_is_red,)),))]))
    f = Fact(Color("b1", "red"))
    net.add_fact(f)
    assert net.conflict_set == []
    net.remove_fact(f)
    assert len(net.conflict_set) == 1


def test_ncc_two_conditions_both_absent_fires():
    net = ReteNetwork()
    net.add_production(_prod([NccGroup((
        Pattern(Color, alpha_tests=(_is_red,)),
        Pattern(Size, alpha_tests=(_is_large,)),
    ))]))
    assert len(net.conflict_set) == 1


def test_ncc_two_conditions_partial_match_fires():
    net = ReteNetwork()
    net.add_production(_prod([NccGroup((
        Pattern(Color, alpha_tests=(_is_red,)),
        Pattern(Size, alpha_tests=(_is_large,)),
    ))]))
    net.add_fact(Fact(Color("b1", "red")))
    assert len(net.conflict_set) == 1


def test_ncc_two_conditions_full_match_blocked():
    net = ReteNetwork()
    net.add_production(_prod([NccGroup((
        Pattern(Color, alpha_tests=(_is_red,)),
        Pattern(Size, alpha_tests=(_is_large,)),
    ))]))
    net.add_fact(Fact(Color("b1", "red")))
    net.add_fact(Fact(Size("b1", "large")))
    assert net.conflict_set == []


def test_ncc_positive_then_ncc_fires():
    net = ReteNetwork()
    net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        NccGroup((Pattern(Size, alpha_tests=(_is_large,)),)),
    ]))
    net.add_fact(Fact(Color("b1", "red")))
    assert len(net.conflict_set) == 1


def test_ncc_positive_then_ncc_blocked():
    net = ReteNetwork()
    net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        NccGroup((Pattern(Size, alpha_tests=(_is_large,)),)),
    ]))
    net.add_fact(Fact(Color("b1", "red")))
    net.add_fact(Fact(Size("b1", "large")))
    assert net.conflict_set == []


def test_ncc_positive_then_ncc_unblocked():
    net = ReteNetwork()
    net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        NccGroup((Pattern(Size, alpha_tests=(_is_large,)),)),
    ]))
    net.add_fact(Fact(Color("b1", "red")))
    f_block = Fact(Size("b1", "large"))
    net.add_fact(f_block)
    assert net.conflict_set == []
    net.remove_fact(f_block)
    assert len(net.conflict_set) == 1


def test_ncc_retroactive_retraction():
    net = ReteNetwork()
    net.add_production(_prod([NccGroup((Pattern(Color, alpha_tests=(_is_red,)),))]))
    assert len(net.conflict_set) == 1
    net.add_fact(Fact(Color("b1", "red")))
    assert net.conflict_set == []


def test_ncc_initialization_with_existing_facts():
    net = ReteNetwork()
    net.root.build_or_share_alpha_memory(Pattern(Color, alpha_tests=(_is_red,)))
    net.add_fact(Fact(Color("b1", "red")))
    net.add_production(_prod([NccGroup((Pattern(Color, alpha_tests=(_is_red,)),))]))
    assert net.conflict_set == []


def test_ncc_gc_on_remove_production():
    net = ReteNetwork()
    g = NccGroup((Pattern(Color, alpha_tests=(_is_red,)),))
    pn = net.add_production(_prod([g]))
    net.remove_production(pn)
    assert net.conflict_set == []


def test_ncc_pnode_parent_join_is_ncc_node():
    net = ReteNetwork()
    g = NccGroup((Pattern(Color, alpha_tests=(_is_red,)),))
    pn = net.add_production(_prod([g]))
    assert isinstance(pn.parent_join, NccNode)


# ---------------------------------------------------------------------------
# Coverage: sharing and init-link paths
# ---------------------------------------------------------------------------


def test_share_negative_join_node():
    # line 249: _build_or_share_negative_join_node returns existing NJN
    net = ReteNetwork()
    p = Pattern(Color, alpha_tests=(_is_red,), negated=True)
    pn1 = net.add_production(_prod([p]))
    pn2 = net.add_production(_prod([p]))
    assert pn1.parent_join is pn2.parent_join
    assert isinstance(pn1.parent_join, NegativeJoinNode)


def test_init_join_links_beta_memory_non_empty():
    # line 284: else branch of _init_join_links when left IS non-empty BetaMemory
    # Requires BOTH alpha memories to be non-empty at construction time so the
    # second JoinNode takes the else branch (not the left-unlink elif).
    net = ReteNetwork()
    net.root.build_or_share_alpha_memory(Pattern(Color, alpha_tests=(_is_red,)))
    net.root.build_or_share_alpha_memory(Pattern(Size, alpha_tests=(_is_large,)))
    net.add_fact(Fact(Color("b1", "red")))
    net.add_fact(Fact(Size("b1", "large")))
    net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        Pattern(Size, alpha_tests=(_is_large,)),
    ]))
    assert len(net.conflict_set) == 1


def test_init_njn_links_beta_memory_non_empty():
    # line 308: else branch of _init_njn_links when left IS non-empty BetaMemory
    net = ReteNetwork()
    net.root.build_or_share_alpha_memory(Pattern(Color, alpha_tests=(_is_red,)))
    net.add_fact(Fact(Color("b1", "red")))
    pn = net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        Pattern(Size, alpha_tests=(_is_large,), negated=True),
    ]))
    # Color matched → NJN left has token; no Size → negation fires.
    assert len(net.conflict_set) == 1
    assert isinstance(pn.parent_join, NegativeJoinNode)


def test_share_exists_node():
    # lines 325-326 + 427: _build_or_share_exists_node returns existing ExistsNode
    net = ReteNetwork()
    p = Pattern(Color, alpha_tests=(_is_red,), exists=True)
    pn1 = net.add_production(_prod([p]))
    pn2 = net.add_production(_prod([p]))
    assert pn1.parent_join is pn2.parent_join
    assert isinstance(pn1.parent_join, ExistsNode)


def test_init_exists_node_links_beta_memory_non_empty():
    # lines 354-356: else branch of _init_exists_node_links with non-empty BetaMemory
    net = ReteNetwork()
    net.root.build_or_share_alpha_memory(Pattern(Color, alpha_tests=(_is_red,)))
    net.add_fact(Fact(Color("b1", "red")))
    pn = net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        Pattern(Size, alpha_tests=(_is_large,), exists=True),
    ]))
    assert isinstance(pn.parent_join, ExistsNode)


def test_init_accumulate_links_right_unlinked():
    # lines 538-539: _init_accumulate_links right-unlinking path
    # [Color, accumulate(Size)] with no facts → BetaMemory for Color is empty
    spec = AccumulateSpec(
        inner=Pattern(Size, alpha_tests=(_is_large,)),
        fn=lambda vals: len(vals),
        bind_attr=None,
        result_var="$n",
    )
    net = ReteNetwork()
    pn = net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,)), spec]))
    assert isinstance(pn.parent_join, AccumulateNode)
    assert pn.parent_join.right_unlinked


def test_init_accumulate_links_beta_memory_non_empty():
    # line 543: else branch of _init_accumulate_links with non-empty BetaMemory
    spec = AccumulateSpec(
        inner=Pattern(Size, alpha_tests=(_is_large,)),
        fn=lambda vals: len(vals),
        bind_attr=None,
        result_var="$n",
    )
    net = ReteNetwork()
    net.root.build_or_share_alpha_memory(Pattern(Color, alpha_tests=(_is_red,)))
    net.add_fact(Fact(Color("b1", "red")))
    pn = net.add_production(_prod([Pattern(Color, alpha_tests=(_is_red,)), spec]))
    assert isinstance(pn.parent_join, AccumulateNode)
    assert not pn.parent_join.right_unlinked


# ---------------------------------------------------------------------------
# Coverage: GC paths
# ---------------------------------------------------------------------------


def test_gc_negative_join_node_early_return():
    # line 480: _gc_negative_join_node returns when NJN still has children
    net = ReteNetwork()
    p = Pattern(Color, alpha_tests=(_is_red,), negated=True)
    pn1 = net.add_production(_prod([p]))
    pn2 = net.add_production(_prod([p]))
    shared_njn = pn1.parent_join  # capture before removal clears it
    net.remove_production(pn1)
    assert pn2.parent_join is shared_njn  # NJN still alive (not GC'd)


def test_gc_negative_join_node_with_beta_memory_left():
    # lines 484-485: _gc_negative_join_node when NJN left_input IS BetaMemory
    net = ReteNetwork()
    pn = net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        Pattern(Size, alpha_tests=(_is_large,), negated=True),
    ]))
    net.remove_production(pn)
    assert net.conflict_set == []


def test_gc_exists_node_early_return():
    # lines 493-494: _gc_exists_node returns when ExistsNode still has children
    net = ReteNetwork()
    p = Pattern(Color, alpha_tests=(_is_red,), exists=True)
    pn1 = net.add_production(_prod([p]))
    pn2 = net.add_production(_prod([p]))
    shared_en = pn1.parent_join
    net.remove_production(pn1)
    assert pn2.parent_join is shared_en


def test_gc_exists_node_removes_from_network():
    # lines 495-499: _gc_exists_node full path when left_input IS BetaMemory
    # Add Color facts first so the ExistsNode is NOT right-unlinked → line 496 hit.
    net = ReteNetwork()
    net.root.build_or_share_alpha_memory(Pattern(Color, alpha_tests=(_is_red,)))
    net.add_fact(Fact(Color("b1", "red")))
    pn = net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        Pattern(Size, alpha_tests=(_is_large,), exists=True),
    ]))
    net.remove_production(pn)
    assert net.conflict_set == []


def test_gc_accumulate_node_right_linked():
    # lines 552-553: removes from alpha memory when not right_unlinked
    # (Facts exist at construction → AccumulateNode is right-linked)
    spec = AccumulateSpec(
        inner=Pattern(Size, alpha_tests=(_is_large,)),
        fn=lambda vals: len(vals),
        bind_attr=None,
        result_var="$n",
    )
    net = ReteNetwork()
    net.root.build_or_share_alpha_memory(spec.inner)
    net.add_fact(Fact(Size("b1", "large")))
    pn = net.add_production(_prod([spec]))
    assert not pn.parent_join.right_unlinked
    net.remove_production(pn)
    assert net.conflict_set == []


def test_gc_accumulate_node_removes_from_network():
    # lines 552-556: _gc_accumulate_node when left_input IS BetaMemory
    spec = AccumulateSpec(
        inner=Pattern(Size, alpha_tests=(_is_large,)),
        fn=lambda vals: len(vals),
        bind_attr=None,
        result_var="$n",
    )
    net = ReteNetwork()
    pn = net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        spec,
    ]))
    net.add_fact(Fact(Size("b1", "large")))
    net.remove_production(pn)
    assert net.conflict_set == []


def test_gc_ncc_node_with_beta_memory_left():
    # lines 568-569: _gc_ncc_node when NCC left_input IS BetaMemory
    net = ReteNetwork()
    pn = net.add_production(_prod([
        Pattern(Color, alpha_tests=(_is_red,)),
        NccGroup((Pattern(Size, alpha_tests=(_is_large,)),)),
    ]))
    net.remove_production(pn)
    assert net.conflict_set == []


def test_gc_ncc_node_negated_inner_sub_last():
    # line 573: _gc_ncc_node when sub_last IS NegativeJoinNode
    net = ReteNetwork()
    # NCC whose only inner pattern is negated → sub_last is a NegativeJoinNode
    pn = net.add_production(_prod([
        NccGroup((Pattern(Color, alpha_tests=(_is_red,), negated=True),)),
    ]))
    net.remove_production(pn)
    assert net.conflict_set == []
