"""Unit tests for beta.py — positive and negative join nodes.

:see: Doorenbos §2.4, §2.6, §2.7, §2.8
"""
from rete.alpha import AlphaMemory
from rete.beta import (
    BetaMemory,
    DummyTopNode,
    Instantiation,
    JoinNode,
    JoinTest,
    NccNode,
    NccPartnerNode,
    NccToken,
    NegativeJoinNode,
    NegativeToken,
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
        children=[child], alpha_memory=am, left_input=beta, tests=tests or []
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

    jn2 = JoinNode(children=[bm2], alpha_memory=am2, left_input=bm1, tests=[])
    bm1.successors = [jn2]
    jn1 = JoinNode(children=[bm1], alpha_memory=am1, left_input=top, tests=[])
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
    jn = JoinNode(children=[pn], alpha_memory=am, left_input=DummyTopNode(), tests=[])
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
    jn = JoinNode(children=[pn], alpha_memory=am, left_input=DummyTopNode(), tests=[])
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

    jn2 = JoinNode(children=[bm2], alpha_memory=am2, left_input=bm1, tests=[])
    bm1.successors = [jn2]
    jn1 = JoinNode(children=[bm1], alpha_memory=am1, left_input=top, tests=[])
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


# ---------------------------------------------------------------------------
# NegativeToken
# ---------------------------------------------------------------------------


def test_negative_token_default_count():
    nt = NegativeToken(token=Token())
    assert nt.count == 0


# ---------------------------------------------------------------------------
# NegativeJoinNode helpers
# ---------------------------------------------------------------------------


def _make_njn(tests=None, left=None):
    """Return ``(NegativeJoinNode, AlphaMemory, _Recorder)``."""
    am = AlphaMemory()
    rec = _Recorder()
    njn = NegativeJoinNode(
        children=[rec],
        alpha_memory=am,
        left_input=left if left is not None else DummyTopNode(),
        tests=tests or [],
    )
    return njn, am, rec


# ---------------------------------------------------------------------------
# NegativeJoinNode — left_activate
# ---------------------------------------------------------------------------


def test_njn_left_activate_empty_alpha_propagates():
    njn, _am, rec = _make_njn()
    t = Token()
    njn.left_activate(t)
    assert rec.activated == [t]
    assert njn.items[0].count == 0


def test_njn_left_activate_matching_wme_blocks():
    njn, am, rec = _make_njn()
    am.items.append(WME("b1", "color", "red"))
    t = Token()
    njn.left_activate(t)
    assert rec.activated == []
    assert njn.items[0].count == 1


def test_njn_left_activate_nonmatching_wme_propagates():
    # WME in alpha but join test fails → count stays 0 → propagates.
    w_prev = WME("block1", "color", "red")
    w_right = WME("block2", "size", "large")
    bm = BetaMemory(items=[Token(wmes=(w_prev,))])
    test = JoinTest(field_of_wme="id", condition_index=0, field_of_token_wme="id")
    njn, am, rec = _make_njn(tests=[test], left=bm)
    am.items.append(w_right)  # id "block2" ≠ token id "block1" → test fails
    t = Token(wmes=(w_prev,))
    njn.left_activate(t)
    assert rec.activated == [t]
    assert njn.items[0].count == 0


# ---------------------------------------------------------------------------
# NegativeJoinNode — right_activate
# ---------------------------------------------------------------------------


def test_njn_right_activate_retracts_when_count_zero():
    njn, am, rec = _make_njn()
    t = Token()
    njn.left_activate(t)   # count 0 → propagated
    assert rec.activated == [t]
    w = WME("b1", "color", "red")
    njn.right_activate(w)
    assert rec.retracted == [t]
    assert njn.items[0].count == 1


def test_njn_right_activate_no_retract_when_already_blocked():
    njn, am, rec = _make_njn()
    am.items.append(WME("b1", "color", "red"))
    t = Token()
    njn.left_activate(t)   # count 1 → blocked
    rec.retracted.clear()
    w2 = WME("b2", "size", "large")
    njn.right_activate(w2)  # count now 2 → no retract
    assert rec.retracted == []
    assert njn.items[0].count == 2


def test_njn_right_activate_increments_count():
    njn, am, rec = _make_njn()
    t = Token()
    njn.left_activate(t)
    w = WME("b1", "color", "red")
    njn.right_activate(w)
    assert njn.items[0].count == 1


# ---------------------------------------------------------------------------
# NegativeJoinNode — right_retract
# ---------------------------------------------------------------------------


def test_njn_right_retract_asserts_when_count_reaches_zero():
    njn, am, rec = _make_njn()
    w = WME("b1", "color", "red")
    am.items.append(w)
    t = Token()
    njn.left_activate(t)   # count 1 → blocked
    rec.activated.clear()
    njn.right_retract(w)   # count → 0 → assert
    assert rec.activated == [t]
    assert njn.items[0].count == 0


def test_njn_right_retract_no_assert_when_count_stays_positive():
    njn, am, rec = _make_njn()
    w1 = WME("b1", "color", "red")
    w2 = WME("b2", "size", "large")
    am.items.extend([w1, w2])
    t = Token()
    njn.left_activate(t)   # count 2 → blocked
    rec.activated.clear()
    njn.right_retract(w1)  # count → 1 → still blocked
    assert rec.activated == []
    assert njn.items[0].count == 1


# ---------------------------------------------------------------------------
# NegativeJoinNode — left_retract
# ---------------------------------------------------------------------------


def test_njn_left_retract_propagated_token():
    njn, am, rec = _make_njn()
    t = Token()
    njn.left_activate(t)   # count 0 → propagated
    assert rec.activated == [t]
    njn.left_retract(t)
    assert rec.retracted == [t]
    assert njn.items == []


def test_njn_left_retract_blocked_token():
    njn, am, rec = _make_njn()
    am.items.append(WME("b1", "color", "red"))
    t = Token()
    njn.left_activate(t)   # count 1 → blocked
    njn.left_retract(t)
    assert rec.retracted == []
    assert njn.items == []


# ---------------------------------------------------------------------------
# NegativeJoinNode — update_child
# ---------------------------------------------------------------------------


def test_njn_update_child_only_sends_propagated():
    njn, am, _rec = _make_njn()
    # seed one propagated (count 0) and one blocked (count 1) item directly
    t0 = Token()
    t1 = Token(wmes=(WME("b1", "color", "red"),))
    njn.items.append(NegativeToken(token=t0, count=0))
    njn.items.append(NegativeToken(token=t1, count=1))
    new_child = _Recorder()
    njn.update_child(new_child)
    assert new_child.activated == [t0]
    assert new_child.retracted == []


# ---------------------------------------------------------------------------
# Bug regression — double-retraction via unextended token (Phase 8 fix)
# ---------------------------------------------------------------------------


def test_beta_memory_left_retract_idempotent():
    """left_retract called twice must not raise (double-retraction guard)."""
    bm = BetaMemory()
    t = Token(wmes=(WME("b1", "color", "red"),))
    bm.left_activate(t)
    bm.left_retract(t)
    bm.left_retract(t)  # must be a no-op


def test_pnode_left_retract_idempotent():
    """left_retract called twice must not raise (double-retraction guard)."""
    pn = PNode(production=_make_production(), conflict_set=[])
    t = Token(wmes=(WME("b1", "color", "red"),))
    pn.left_activate(t)
    pn.left_retract(t)
    pn.left_retract(t)  # must be a no-op


# ---------------------------------------------------------------------------
# NCC helpers
# ---------------------------------------------------------------------------


def _make_ncc_pair(owner_length: int = 0):
    """Return ``(NccNode, NccPartnerNode, _Recorder)`` wired together."""
    rec = _Recorder()
    ncc = NccNode(children=[rec], owner_length=owner_length)
    partner = NccPartnerNode(ncc_node=ncc)
    ncc.partner = partner
    return ncc, partner, rec


# ---------------------------------------------------------------------------
# NccToken
# ---------------------------------------------------------------------------


def test_ncc_token_defaults():
    nt = NccToken(token=Token())
    assert nt.count == 0
    assert nt.results == []


# ---------------------------------------------------------------------------
# NccPartnerNode — left_activate
# ---------------------------------------------------------------------------


def test_ncc_partner_activate_buffers_when_no_ncc_token():
    ncc, partner, _rec = _make_ncc_pair(owner_length=0)
    result = Token(wmes=(WME("x", "a", "v"),))
    partner.left_activate(result)
    assert result in ncc.new_result_buffer
    assert result in partner.items


def test_ncc_partner_activate_increments_count():
    ncc, partner, _rec = _make_ncc_pair(owner_length=0)
    base = Token()
    ncc_tok = NccToken(token=base, count=0)
    ncc.items.append(ncc_tok)
    result = Token(wmes=(WME("x", "a", "v"),))
    partner.left_activate(result)
    assert ncc_tok.count == 1
    assert result in ncc_tok.results


def test_ncc_partner_activate_retracts_when_count_was_zero():
    ncc, partner, rec = _make_ncc_pair(owner_length=0)
    base = Token()
    ncc_tok = NccToken(token=base, count=0)
    ncc.items.append(ncc_tok)
    result = Token(wmes=(WME("x", "a", "v"),))
    partner.left_activate(result)
    assert rec.retracted == [base]
    assert ncc_tok.count == 1


def test_ncc_partner_activate_no_retract_when_already_blocked():
    ncc, partner, rec = _make_ncc_pair(owner_length=0)
    base = Token()
    ncc_tok = NccToken(token=base, count=1)
    ncc.items.append(ncc_tok)
    result = Token(wmes=(WME("x", "a", "v"),))
    partner.left_activate(result)
    assert rec.retracted == []
    assert ncc_tok.count == 2


# ---------------------------------------------------------------------------
# NccPartnerNode — left_retract
# ---------------------------------------------------------------------------


def test_ncc_partner_retract_from_buffer():
    ncc, partner, _rec = _make_ncc_pair(owner_length=0)
    result = Token(wmes=(WME("x", "a", "v"),))
    partner.left_activate(result)  # lands in buffer
    partner.left_retract(result)
    assert result not in ncc.new_result_buffer
    assert result not in partner.items


def test_ncc_partner_retract_decrements_count():
    ncc, partner, _rec = _make_ncc_pair(owner_length=0)
    base = Token()
    ncc.items.append(NccToken(token=base, count=0))
    result = Token(wmes=(WME("x", "a", "v"),))
    partner.left_activate(result)  # registers in beta_tokens; count: 0→1
    partner.left_retract(result)
    assert ncc.items[0].count == 0
    assert result not in ncc.items[0].results


def test_ncc_partner_retract_asserts_when_count_reaches_zero():
    ncc, partner, rec = _make_ncc_pair(owner_length=0)
    base = Token()
    ncc.items.append(NccToken(token=base, count=0))
    result = Token(wmes=(WME("x", "a", "v"),))
    partner.left_activate(result)  # count: 0→1, triggers retract on rec
    rec.retracted.clear()          # reset: only test left_retract effect
    partner.left_retract(result)   # count: 1→0 → re-assert base
    assert rec.activated == [base]


def test_ncc_partner_retract_no_assert_count_stays_positive():
    ncc, partner, rec = _make_ncc_pair(owner_length=0)
    base = Token()
    ncc.items.append(NccToken(token=base, count=0))
    result1 = Token(wmes=(WME("x", "a", "v"),))
    result2 = Token(wmes=(WME("y", "b", "w"),))
    partner.left_activate(result1)  # count: 0→1, retracts base
    partner.left_activate(result2)  # count: 1→2 (blocked, no retract)
    rec.retracted.clear()
    partner.left_retract(result1)   # count: 2→1, still blocked
    assert rec.activated == []
    assert ncc.items[0].count == 1


# ---------------------------------------------------------------------------
# NccNode — left_activate
# ---------------------------------------------------------------------------


def test_ncc_node_activate_empty_buffer_propagates():
    ncc, _partner, rec = _make_ncc_pair(owner_length=0)
    t = Token()
    ncc.left_activate(t)
    assert rec.activated == [t]
    assert ncc.items[0].count == 0


def test_ncc_node_activate_drains_matching_buffer():
    ncc, _partner, rec = _make_ncc_pair(owner_length=0)
    result = Token(wmes=(WME("x", "a", "v"),))
    ncc.new_result_buffer.append(result)
    t = Token()
    ncc.left_activate(t)
    assert rec.activated == []
    assert ncc.items[0].count == 1
    assert ncc.new_result_buffer == []


def test_ncc_node_activate_ignores_nonmatching_buffer():
    w_main = WME("b1", "color", "red")
    w_other = WME("b2", "color", "blue")
    ncc, _partner, rec = _make_ncc_pair(owner_length=1)
    # result whose first WME is w_other — does not match left token (w_main,)
    result = Token(wmes=(w_other, WME("x", "y", "z")))
    ncc.new_result_buffer.append(result)
    t = Token(wmes=(w_main,))
    ncc.left_activate(t)
    assert rec.activated == [t]
    assert ncc.items[0].count == 0
    assert result in ncc.new_result_buffer  # untouched


# ---------------------------------------------------------------------------
# NccNode — left_retract
# ---------------------------------------------------------------------------


def test_ncc_node_retract_propagated_token():
    ncc, _partner, rec = _make_ncc_pair(owner_length=0)
    t = Token()
    ncc.left_activate(t)  # count=0 → propagated
    ncc.left_retract(t)
    assert rec.retracted == [t]
    assert ncc.items == []


def test_ncc_node_retract_blocked_token():
    ncc, partner, rec = _make_ncc_pair(owner_length=0)
    t = Token()
    ncc.left_activate(t)                        # count=0 → propagated
    result = Token(wmes=(WME("x", "a", "v"),))
    partner.left_activate(result)               # count: 0→1, retracts from rec
    rec.retracted.clear()                       # reset: only test left_retract
    ncc.left_retract(t)                         # count=1 → children NOT called
    assert rec.retracted == []
    assert ncc.items == []


def test_ncc_node_retract_clears_partner_items():
    ncc, partner, _rec = _make_ncc_pair(owner_length=0)
    t = Token()
    result = Token(wmes=(WME("x", "a", "v"),))
    partner.items.append(result)
    ncc_tok = NccToken(token=t, count=1, results=[result])
    ncc.items.append(ncc_tok)
    ncc.left_retract(t)
    assert result not in partner.items
    assert ncc_tok.results == []


# ---------------------------------------------------------------------------
# NccNode — update_child
# ---------------------------------------------------------------------------


def test_ncc_node_update_child_only_propagated():
    ncc, _partner, _rec = _make_ncc_pair(owner_length=0)
    t0 = Token()
    t1 = Token(wmes=(WME("b1", "color", "red"),))
    ncc.items.append(NccToken(token=t0, count=0))
    ncc.items.append(NccToken(token=t1, count=1))
    new_child = _Recorder()
    ncc.update_child(new_child)
    assert new_child.activated == [t0]
    assert new_child.retracted == []
