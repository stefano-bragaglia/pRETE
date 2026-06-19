"""Unit tests for beta.py — JoinTest, BetaMemory, DummyTopNode, JoinNode, PNode.

:see: Doorenbos §2.4, §2.6
"""
from rete.alpha import AlphaMemory
from rete.beta import (
    BetaMemory,
    DummyTopNode,
    Instantiation,
    JoinNode,
    JoinTest,
    PNode,
)
from rete.condition import WILDCARD, Condition, Production
from rete.wme import Token, WME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Recorder:
    """Minimal successor stub recording left_activate / left_retract calls."""

    def __init__(self):
        self.activated: list[Token] = []
        self.retracted: list[Token] = []

    def left_activate(self, token: Token) -> None:
        self.activated.append(token)

    def left_retract(self, token: Token) -> None:
        self.retracted.append(token)


def _make_join(tests=None, left=None):
    """Return ``(JoinNode, AlphaMemory, downstream BetaMemory)``."""
    am = AlphaMemory()
    beta = left if left is not None else BetaMemory()
    child = BetaMemory()
    jn = JoinNode(
        children=[child], alpha_memory=am, beta_memory=beta, tests=tests or []
    )
    return jn, am, child


# ---------------------------------------------------------------------------
# DummyTopNode
# ---------------------------------------------------------------------------


def test_dummy_top_node_has_one_empty_token():
    top = DummyTopNode()
    assert len(top.items) == 1
    assert top.items[0] == Token()


# ---------------------------------------------------------------------------
# BetaMemory
# ---------------------------------------------------------------------------


def test_beta_memory_left_activate_stores_token():
    bm = BetaMemory()
    w = WME("b1", "color", "red")
    t = Token(wmes=(w,))
    bm.left_activate(t)
    assert t in bm.items


def test_beta_memory_left_activate_notifies_successor():
    rec = _Recorder()
    bm = BetaMemory(successors=[rec])
    t = Token(wmes=(WME("b1", "color", "red"),))
    bm.left_activate(t)
    assert rec.activated == [t]


def test_beta_memory_left_retract_removes_token():
    bm = BetaMemory()
    t = Token(wmes=(WME("b1", "color", "red"),))
    bm.left_activate(t)
    bm.left_retract(t)
    assert t not in bm.items


def test_beta_memory_left_retract_notifies_successor():
    rec = _Recorder()
    bm = BetaMemory(successors=[rec])
    t = Token(wmes=(WME("b1", "color", "red"),))
    bm.left_activate(t)
    bm.left_retract(t)
    assert rec.retracted == [t]


def test_beta_memory_multiple_tokens():
    bm = BetaMemory()
    t1 = Token(wmes=(WME("b1", "color", "red"),))
    t2 = Token(wmes=(WME("b2", "color", "blue"),))
    bm.left_activate(t1)
    bm.left_activate(t2)
    bm.left_retract(t1)
    assert t1 not in bm.items
    assert t2 in bm.items


# ---------------------------------------------------------------------------
# JoinTest / extract_join_tests
# ---------------------------------------------------------------------------


def test_extract_join_tests_no_variables():
    cond = Condition("b1", "color", "red")
    assert JoinTest.extract(cond, []) == []


def test_extract_join_tests_wildcard():
    cond = Condition(WILDCARD, "color", WILDCARD)
    assert JoinTest.extract(cond, [Condition("b1", "size", "large")]) == []


def test_extract_join_tests_unbound_variable():
    cond = Condition("?x", "color", "red")
    assert JoinTest.extract(cond, []) == []


def test_extract_join_tests_single_match_same_field():
    earlier = [Condition("?x", "color", "red")]
    cond = Condition("?x", "size", "large")
    tests = JoinTest.extract(cond, earlier)
    assert tests == [
        JoinTest(field_of_wme="id", condition_index=0, field_of_token_wme="id")
    ]


def test_extract_join_tests_single_match_different_fields():
    # ?v first appears as value_test; new condition uses it as id_test
    earlier = [Condition("b1", "color", "?v")]
    cond = Condition("?v", "size", "large")
    tests = JoinTest.extract(cond, earlier)
    assert tests == [
        JoinTest(field_of_wme="id", condition_index=0, field_of_token_wme="value")
    ]


def test_extract_join_tests_two_occurrences():
    # ?x appears in two earlier conditions
    earlier = [
        Condition("?x", "color", "red"),
        Condition("?x", "size", "large"),
    ]
    cond = Condition("?x", "weight", "heavy")
    tests = JoinTest.extract(cond, earlier)
    assert tests == [
        JoinTest(field_of_wme="id", condition_index=0, field_of_token_wme="id"),
        JoinTest(field_of_wme="id", condition_index=1, field_of_token_wme="id"),
    ]


def test_extract_join_tests_partial_earlier_match():
    # ?x in one earlier condition but not the other → only one JoinTest
    earlier = [
        Condition("?x", "color", "red"),
        Condition("b2", "size", "large"),  # ?x absent here
    ]
    cond = Condition("?x", "weight", "heavy")
    tests = JoinTest.extract(cond, earlier)
    assert tests == [
        JoinTest(field_of_wme="id", condition_index=0, field_of_token_wme="id")
    ]


def test_extract_join_tests_two_variables():
    # Two distinct variables each bound in an earlier condition
    earlier = [Condition("?x", "?y", "red")]
    cond = Condition("?x", "?y", "blue")
    tests = JoinTest.extract(cond, earlier)
    assert JoinTest("id", 0, "id") in tests
    assert JoinTest("attribute", 0, "attribute") in tests


# ---------------------------------------------------------------------------
# JoinNode — right_activate / left_activate
# ---------------------------------------------------------------------------


def test_join_node_right_activate_empty_beta():
    jn, am, child = _make_join()  # left is empty BetaMemory
    jn.right_activate(WME("b1", "color", "red"))
    assert child.items == []


def test_join_node_right_activate_with_dummy_top():
    w = WME("b1", "color", "red")
    jn, am, child = _make_join(left=DummyTopNode())
    jn.right_activate(w)
    assert len(child.items) == 1
    assert child.items[0].wmes == (w,)


def test_join_node_right_activate_emits_extended_token():
    w0 = WME("b0", "on", "table")
    w = WME("b1", "color", "red")
    bm = BetaMemory(items=[Token(wmes=(w0,))])
    jn, am, child = _make_join(left=bm)
    jn.right_activate(w)
    assert len(child.items) == 1
    assert child.items[0].wmes == (w0, w)


def test_join_node_right_activate_extended_token_contains_wme():
    w = WME("b1", "color", "red")
    jn, am, child = _make_join(left=DummyTopNode())
    jn.right_activate(w)
    assert child.items[0].wmes[-1] is w


def test_join_node_left_activate_empty_alpha():
    jn, am, child = _make_join()
    jn.left_activate(Token())
    assert child.items == []


def test_join_node_left_activate_emits_extended_token():
    w = WME("b1", "color", "red")
    jn, am, child = _make_join()
    am.items.append(w)
    jn.left_activate(Token())
    assert len(child.items) == 1
    assert child.items[0].wmes == (w,)


def test_join_node_consistency_check_pass():
    w_prev = WME("block1", "color", "red")
    w_new = WME("block1", "size", "large")
    bm = BetaMemory(items=[Token(wmes=(w_prev,))])
    test = JoinTest(field_of_wme="id", condition_index=0, field_of_token_wme="id")
    jn, am, child = _make_join(tests=[test], left=bm)
    jn.right_activate(w_new)
    assert len(child.items) == 1


def test_join_node_consistency_check_fail():
    w_prev = WME("block1", "color", "red")
    w_new = WME("block2", "size", "large")  # different id
    bm = BetaMemory(items=[Token(wmes=(w_prev,))])
    test = JoinTest(field_of_wme="id", condition_index=0, field_of_token_wme="id")
    jn, am, child = _make_join(tests=[test], left=bm)
    jn.right_activate(w_new)
    assert child.items == []


def test_join_node_right_activate_multiple_tokens():
    w0a = WME("b0", "on", "table")
    w0b = WME("b2", "on", "floor")
    w = WME("b1", "color", "red")
    bm = BetaMemory(items=[Token(wmes=(w0a,)), Token(wmes=(w0b,))])
    jn, am, child = _make_join(left=bm)
    jn.right_activate(w)
    assert len(child.items) == 2


def test_join_node_left_activate_multiple_wmes():
    w1 = WME("b1", "color", "red")
    w2 = WME("b2", "color", "blue")
    jn, am, child = _make_join()
    am.items.extend([w1, w2])
    jn.left_activate(Token())
    assert len(child.items) == 2


# ---------------------------------------------------------------------------
# JoinNode — retraction
# ---------------------------------------------------------------------------


def test_join_node_right_retract_removes_derived_tokens():
    w = WME("b1", "color", "red")
    jn, am, child = _make_join(left=DummyTopNode())
    jn.right_activate(w)
    assert len(child.items) == 1
    jn.right_retract(w)
    assert child.items == []


def test_join_node_right_retract_no_match():
    w1 = WME("b1", "color", "red")
    w2 = WME("b2", "color", "blue")
    jn, am, child = _make_join(left=DummyTopNode())
    jn.right_activate(w1)
    jn.right_retract(w2)  # w2 was never emitted
    assert len(child.items) == 1


def test_join_node_left_retract_removes_derived_tokens():
    w0 = WME("b0", "on", "table")
    w = WME("b1", "color", "red")
    t = Token(wmes=(w0,))
    bm = BetaMemory(items=[t])
    jn, am, child = _make_join(left=bm)
    jn.right_activate(w)
    assert len(child.items) == 1
    jn.left_retract(t)
    assert child.items == []


def test_join_node_left_retract_no_match():
    w0 = WME("b0", "on", "table")
    w3 = WME("b3", "on", "shelf")
    w = WME("b1", "color", "red")
    t1 = Token(wmes=(w0,))
    t2 = Token(wmes=(w3,))
    bm = BetaMemory(items=[t1])
    jn, am, child = _make_join(left=bm)
    jn.right_activate(w)
    jn.left_retract(t2)  # t2 was never in bm; nothing derived from it
    assert len(child.items) == 1


# ---------------------------------------------------------------------------
# Chain integration
# ---------------------------------------------------------------------------


def test_two_join_nodes_chain():
    # DummyTop → JoinNode1(am1) → BetaMemory1 → JoinNode2(am2) → BetaMemory2
    am1 = AlphaMemory()
    am2 = AlphaMemory()
    bm1 = BetaMemory()
    bm2 = BetaMemory()
    top = DummyTopNode()

    jn2 = JoinNode(children=[bm2], alpha_memory=am2, beta_memory=bm1, tests=[])
    bm1.successors = [jn2]
    jn1 = JoinNode(children=[bm1], alpha_memory=am1, beta_memory=top, tests=[])
    am1.successors = [jn1]
    am2.successors = [jn2]

    w1 = WME("b1", "color", "red")
    w2 = WME("b1", "size", "large")
    am1.activate(w1)   # Token((w1,)) in bm1; jn2.left_activate sees empty am2
    am2.activate(w2)   # jn2 joins w2 with Token((w1,)) → Token((w1, w2))

    assert len(bm2.items) == 1
    assert bm2.items[0].wmes == (w1, w2)


# ---------------------------------------------------------------------------
# PNode
# ---------------------------------------------------------------------------


def _make_production() -> Production:
    return Production(lhs=[], rhs=lambda t: None)


def test_pnode_left_activate_stores_token_in_items():
    pn = PNode(production=_make_production(), conflict_set=[])
    t = Token(wmes=(WME("b1", "color", "red"),))
    pn.left_activate(t)
    assert t in pn.items


def test_pnode_left_activate_adds_instantiation_to_conflict_set():
    p = _make_production()
    cs: list[Instantiation] = []
    pn = PNode(production=p, conflict_set=cs)
    t = Token(wmes=(WME("b1", "color", "red"),))
    pn.left_activate(t)
    assert cs == [Instantiation(p, t)]


def test_pnode_left_retract_removes_token_from_items():
    pn = PNode(production=_make_production(), conflict_set=[])
    t = Token(wmes=(WME("b1", "color", "red"),))
    pn.left_activate(t)
    pn.left_retract(t)
    assert t not in pn.items


def test_pnode_left_retract_removes_instantiation_from_conflict_set():
    p = _make_production()
    cs: list[Instantiation] = []
    pn = PNode(production=p, conflict_set=cs)
    t = Token(wmes=(WME("b1", "color", "red"),))
    pn.left_activate(t)
    pn.left_retract(t)
    assert cs == []


def test_pnode_multiple_tokens_coexist():
    p = _make_production()
    cs: list[Instantiation] = []
    pn = PNode(production=p, conflict_set=cs)
    t1 = Token(wmes=(WME("b1", "color", "red"),))
    t2 = Token(wmes=(WME("b2", "color", "blue"),))
    pn.left_activate(t1)
    pn.left_activate(t2)
    assert len(pn.items) == 2
    assert len(cs) == 2


def test_pnode_left_retract_removes_only_target():
    p = _make_production()
    cs: list[Instantiation] = []
    pn = PNode(production=p, conflict_set=cs)
    t1 = Token(wmes=(WME("b1", "color", "red"),))
    t2 = Token(wmes=(WME("b2", "color", "blue"),))
    pn.left_activate(t1)
    pn.left_activate(t2)
    pn.left_retract(t1)
    assert t1 not in pn.items
    assert t2 in pn.items
    assert cs == [Instantiation(p, t2)]


def test_join_node_with_pnode_child_right_activate():
    p = _make_production()
    cs: list[Instantiation] = []
    pn = PNode(production=p, conflict_set=cs)
    am = AlphaMemory()
    jn = JoinNode(children=[pn], alpha_memory=am, beta_memory=DummyTopNode(), tests=[])
    w = WME("b1", "color", "red")
    jn.right_activate(w)
    assert len(cs) == 1
    assert cs[0].token.wmes == (w,)
    assert cs[0].production is p


def test_join_node_with_pnode_child_right_retract():
    p = _make_production()
    cs: list[Instantiation] = []
    pn = PNode(production=p, conflict_set=cs)
    am = AlphaMemory()
    jn = JoinNode(children=[pn], alpha_memory=am, beta_memory=DummyTopNode(), tests=[])
    w = WME("b1", "color", "red")
    jn.right_activate(w)
    jn.right_retract(w)
    assert cs == []
    assert pn.items == []


# ---------------------------------------------------------------------------
# Chain integration
# ---------------------------------------------------------------------------


def test_chain_retract_wme_cascades():
    am1 = AlphaMemory()
    am2 = AlphaMemory()
    bm1 = BetaMemory()
    bm2 = BetaMemory()
    top = DummyTopNode()

    jn2 = JoinNode(children=[bm2], alpha_memory=am2, beta_memory=bm1, tests=[])
    bm1.successors = [jn2]
    jn1 = JoinNode(children=[bm1], alpha_memory=am1, beta_memory=top, tests=[])
    am1.successors = [jn1]
    am2.successors = [jn2]

    w1 = WME("b1", "color", "red")
    w2 = WME("b1", "size", "large")
    am1.activate(w1)
    am2.activate(w2)
    assert len(bm2.items) == 1

    # Retract w1 → bm1 loses Token((w1,)) → jn2.left_retract → bm2 loses Token((w1, w2))
    am1.deactivate(w1)
    assert bm1.items == []
    assert bm2.items == []
