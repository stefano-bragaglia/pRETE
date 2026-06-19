"""Unit tests for network.py — ReteNetwork.

:see: Doorenbos §2.6, Appendix A, §2.8
"""
from dataclasses import dataclass

from rete.beta import BetaMemory, JoinNode, NccNode, PNode
from rete.condition import JoinSpec, NccGroup, Pattern, Production
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
