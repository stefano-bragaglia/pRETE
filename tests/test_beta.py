"""Unit tests for beta.py — positive and negative join nodes.

:see: Doorenbos §2.4, §2.6, §2.7, §2.8
"""
from dataclasses import dataclass

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
from rete.condition import JoinSpec, Pattern, Production
from rete.fact import Fact, Token


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Recorder:
    """Minimal successor stub recording left_activate / left_retract calls."""

    def __init__(self):
        self.activated: list[Token] = []
        self.retracted: list[Token] = []

    def left_activate(self, token: Token) -> None:
        """Record activation."""
        self.activated.append(token)

    def left_retract(self, token: Token) -> None:
        """Record retraction."""
        self.retracted.append(token)


def _any_am() -> AlphaMemory:
    """Return an AlphaMemory that accepts any Fact (used in isolated node tests)."""
    return AlphaMemory(type_=object, predicate=lambda _: True)


def _color_pattern(**kw) -> Pattern:
    """Return a Pattern for Color with optional bindings/join_tests."""
    return Pattern(Color, **kw)


def _make_join(tests=None, left=None, pattern=None):
    """Return ``(JoinNode, AlphaMemory, downstream BetaMemory)``."""
    am = _any_am()
    beta = left if left is not None else BetaMemory()
    child = BetaMemory()
    jn = JoinNode(
        children=[child],
        alpha_memory=am,
        left_input=beta,
        tests=tests or [],
        pattern=pattern,
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
    f = Fact(Color("b1", "red"))
    t = Token(facts=(f,))
    bm.left_activate(t)
    assert t in bm.items


def test_beta_memory_left_activate_notifies_successor():
    rec = _Recorder()
    bm = BetaMemory(successors=[rec])
    f = Fact(Color("b1", "red"))
    t = Token(facts=(f,))
    bm.left_activate(t)
    assert rec.activated == [t]


def test_beta_memory_left_retract_removes_token():
    bm = BetaMemory()
    f = Fact(Color("b1", "red"))
    t = Token(facts=(f,))
    bm.left_activate(t)
    bm.left_retract(t)
    assert t not in bm.items


def test_beta_memory_left_retract_notifies_successor():
    rec = _Recorder()
    bm = BetaMemory(successors=[rec])
    f = Fact(Color("b1", "red"))
    t = Token(facts=(f,))
    bm.left_activate(t)
    bm.left_retract(t)
    assert rec.retracted == [t]


def test_beta_memory_multiple_tokens():
    bm = BetaMemory()
    f1 = Fact(Color("b1", "red"))
    f2 = Fact(Color("b2", "blue"))
    t1 = Token(facts=(f1,))
    t2 = Token(facts=(f2,))
    bm.left_activate(t1)
    bm.left_activate(t2)
    bm.left_retract(t1)
    assert t1 not in bm.items
    assert t2 in bm.items


# ---------------------------------------------------------------------------
# JoinTest
# ---------------------------------------------------------------------------


def test_join_test_extract_empty_join_specs():
    p = Pattern(Color)
    assert JoinTest.extract(p, []) == []


def test_join_test_extract_single_spec():
    spec = JoinSpec("block", "$block")
    p = Pattern(Color, join_tests=(spec,))
    tests = JoinTest.extract(p, [])
    assert tests == [JoinTest("block", "$block")]


def test_join_test_extract_multiple_specs():
    specs = (JoinSpec("block", "$block"), JoinSpec("color", "$color"))
    p = Pattern(Color, join_tests=specs)
    tests = JoinTest.extract(p, [])
    assert tests == [JoinTest("block", "$block"), JoinTest("color", "$color")]


def test_join_test_extract_ignores_earlier_arg():
    # earlier is passed by network.py for compat; JoinTest.extract ignores it
    spec = JoinSpec("block", "$b")
    p = Pattern(Color, join_tests=(spec,))
    earlier = [Pattern(Size)]
    assert JoinTest.extract(p, earlier) == [JoinTest("block", "$b")]


def test_join_test_passes_matching():
    test = JoinTest("block", "$block")
    f = Fact(Size("b1", "large"))
    t = Token(facts=(), bindings={"$block": "b1"})
    jn, _am, _child = _make_join(tests=[test])
    assert jn._passes_tests(t, f)


def test_join_test_fails_mismatch():
    test = JoinTest("block", "$block")
    f = Fact(Size("b2", "large"))
    t = Token(facts=(), bindings={"$block": "b1"})
    jn, _am, _child = _make_join(tests=[test])
    assert not jn._passes_tests(t, f)


def test_join_test_passes_when_no_tests():
    jn, _am, _child = _make_join()
    assert jn._passes_tests(Token(), Fact(Color("b1", "red")))


# ---------------------------------------------------------------------------
# JoinNode — right_activate / left_activate
# ---------------------------------------------------------------------------


def test_join_node_right_activate_empty_beta():
    jn, am, child = _make_join()  # left is empty BetaMemory
    jn.right_activate(Fact(Color("b1", "red")))
    assert child.items == []


def test_join_node_right_activate_with_dummy_top():
    f = Fact(Color("b1", "red"))
    jn, am, child = _make_join(left=DummyTopNode())
    jn.right_activate(f)
    assert len(child.items) == 1
    assert child.items[0].facts == (f,)


def test_join_node_right_activate_emits_extended_token():
    f0 = Fact(Color("b0", "green"))
    f = Fact(Color("b1", "red"))
    bm = BetaMemory(items=[Token(facts=(f0,))])
    jn, am, child = _make_join(left=bm)
    jn.right_activate(f)
    assert len(child.items) == 1
    assert child.items[0].facts == (f0, f)


def test_join_node_right_activate_extended_token_contains_fact():
    f = Fact(Color("b1", "red"))
    jn, am, child = _make_join(left=DummyTopNode())
    jn.right_activate(f)
    assert child.items[0].facts[-1] is f


def test_join_node_left_activate_empty_alpha():
    jn, am, child = _make_join()
    jn.left_activate(Token())
    assert child.items == []


def test_join_node_left_activate_emits_extended_token():
    f = Fact(Color("b1", "red"))
    jn, am, child = _make_join()
    am.items.append(f)
    jn.left_activate(Token())
    assert len(child.items) == 1
    assert child.items[0].facts == (f,)


def test_join_node_consistency_check_pass():
    f_prev = Fact(Color("block1", "red"))
    f_new = Fact(Size("block1", "large"))
    bm = BetaMemory(items=[Token(facts=(f_prev,), bindings={"$block": "block1"})])
    test = JoinTest("block", "$block")
    jn, am, child = _make_join(tests=[test], left=bm)
    jn.right_activate(f_new)
    assert len(child.items) == 1


def test_join_node_consistency_check_fail():
    f_prev = Fact(Color("block1", "red"))
    f_new = Fact(Size("block2", "large"))  # different block
    bm = BetaMemory(items=[Token(facts=(f_prev,), bindings={"$block": "block1"})])
    test = JoinTest("block", "$block")
    jn, am, child = _make_join(tests=[test], left=bm)
    jn.right_activate(f_new)
    assert child.items == []


def test_join_node_right_activate_multiple_tokens():
    f0a = Fact(Color("b0", "green"))
    f0b = Fact(Color("b2", "blue"))
    f = Fact(Size("b1", "large"))
    bm = BetaMemory(items=[Token(facts=(f0a,)), Token(facts=(f0b,))])
    jn, am, child = _make_join(left=bm)
    jn.right_activate(f)
    assert len(child.items) == 2


def test_join_node_left_activate_multiple_facts():
    f1 = Fact(Color("b1", "red"))
    f2 = Fact(Color("b2", "blue"))
    jn, am, child = _make_join()
    am.items.extend([f1, f2])
    jn.left_activate(Token())
    assert len(child.items) == 2


# ---------------------------------------------------------------------------
# Binding merge
# ---------------------------------------------------------------------------


def test_join_node_merges_bindings_on_extend():
    p = Pattern(Color, bindings=(("$color", "color"),))
    f = Fact(Color("b1", "red"))
    jn, am, child = _make_join(left=DummyTopNode(), pattern=p)
    jn.right_activate(f)
    assert child.items[0].bindings == {"$color": "red"}


def test_join_node_merges_parent_bindings():
    p = Pattern(Size, bindings=(("$size", "size"),))
    f_parent = Fact(Color("b1", "red"))
    f_new = Fact(Size("b1", "large"))
    parent_token = Token(facts=(f_parent,), bindings={"$color": "red"})
    bm = BetaMemory(items=[parent_token])
    jn, am, child = _make_join(left=bm, pattern=p)
    jn.right_activate(f_new)
    assert child.items[0].bindings == {"$color": "red", "$size": "large"}


def test_update_child_emits_bindings():
    p = Pattern(Color, bindings=(("$color", "color"),))
    f = Fact(Color("b1", "red"))
    am = _any_am()
    am.items.append(f)
    top = DummyTopNode()
    child = BetaMemory()
    jn = JoinNode(
        children=[],
        alpha_memory=am,
        left_input=top,
        tests=[],
        pattern=p,
    )
    jn.update_child(child)
    assert len(child.items) == 1
    assert child.items[0].bindings == {"$color": "red"}


# ---------------------------------------------------------------------------
# JoinNode — retraction
# ---------------------------------------------------------------------------


def test_join_node_right_retract_removes_derived_tokens():
    f = Fact(Color("b1", "red"))
    jn, am, child = _make_join(left=DummyTopNode())
    jn.right_activate(f)
    assert len(child.items) == 1
    jn.right_retract(f)
    assert child.items == []


def test_join_node_right_retract_no_match():
    f1 = Fact(Color("b1", "red"))
    f2 = Fact(Color("b2", "blue"))
    jn, am, child = _make_join(left=DummyTopNode())
    jn.right_activate(f1)
    jn.right_retract(f2)  # f2 was never emitted
    assert len(child.items) == 1


def test_join_node_left_retract_removes_derived_tokens():
    f0 = Fact(Color("b0", "green"))
    f = Fact(Color("b1", "red"))
    t = Token(facts=(f0,))
    bm = BetaMemory(items=[t])
    jn, am, child = _make_join(left=bm)
    jn.right_activate(f)
    assert len(child.items) == 1
    jn.left_retract(t)
    assert child.items == []


def test_join_node_left_retract_no_match():
    f0 = Fact(Color("b0", "green"))
    f3 = Fact(Color("b3", "yellow"))
    f = Fact(Color("b1", "red"))
    t1 = Token(facts=(f0,))
    t2 = Token(facts=(f3,))
    bm = BetaMemory(items=[t1])
    jn, am, child = _make_join(left=bm)
    jn.right_activate(f)
    jn.left_retract(t2)  # t2 was never in bm; nothing derived from it
    assert len(child.items) == 1


# ---------------------------------------------------------------------------
# Chain integration
# ---------------------------------------------------------------------------


def test_two_join_nodes_chain():
    am1 = _any_am()
    am2 = _any_am()
    bm1 = BetaMemory()
    bm2 = BetaMemory()
    top = DummyTopNode()

    jn2 = JoinNode(children=[bm2], alpha_memory=am2, left_input=bm1, tests=[])
    bm1.successors = [jn2]
    jn1 = JoinNode(children=[bm1], alpha_memory=am1, left_input=top, tests=[])
    am1.successors = [jn1]
    am2.successors = [jn2]

    f1 = Fact(Color("b1", "red"))
    f2 = Fact(Size("b1", "large"))
    am1.activate(f1)
    am2.activate(f2)

    assert len(bm2.items) == 1
    assert bm2.items[0].facts == (f1, f2)


# ---------------------------------------------------------------------------
# PNode
# ---------------------------------------------------------------------------


def _make_production() -> Production:
    return Production(lhs=[], rhs=lambda t: None)


def test_pnode_left_activate_stores_token_in_items():
    pn = PNode(production=_make_production(), conflict_set=[])
    f = Fact(Color("b1", "red"))
    t = Token(facts=(f,))
    pn.left_activate(t)
    assert t in pn.items


def test_pnode_left_activate_adds_instantiation_to_conflict_set():
    p = _make_production()
    cs: list[Instantiation] = []
    pn = PNode(production=p, conflict_set=cs)
    f = Fact(Color("b1", "red"))
    t = Token(facts=(f,))
    pn.left_activate(t)
    assert cs == [Instantiation(p, t)]


def test_pnode_left_retract_removes_token_from_items():
    pn = PNode(production=_make_production(), conflict_set=[])
    f = Fact(Color("b1", "red"))
    t = Token(facts=(f,))
    pn.left_activate(t)
    pn.left_retract(t)
    assert t not in pn.items


def test_pnode_left_retract_removes_instantiation_from_conflict_set():
    p = _make_production()
    cs: list[Instantiation] = []
    pn = PNode(production=p, conflict_set=cs)
    f = Fact(Color("b1", "red"))
    t = Token(facts=(f,))
    pn.left_activate(t)
    pn.left_retract(t)
    assert cs == []


def test_pnode_multiple_tokens_coexist():
    p = _make_production()
    cs: list[Instantiation] = []
    pn = PNode(production=p, conflict_set=cs)
    f1 = Fact(Color("b1", "red"))
    f2 = Fact(Color("b2", "blue"))
    t1 = Token(facts=(f1,))
    t2 = Token(facts=(f2,))
    pn.left_activate(t1)
    pn.left_activate(t2)
    assert len(pn.items) == 2
    assert len(cs) == 2


def test_pnode_left_retract_removes_only_target():
    p = _make_production()
    cs: list[Instantiation] = []
    pn = PNode(production=p, conflict_set=cs)
    f1 = Fact(Color("b1", "red"))
    f2 = Fact(Color("b2", "blue"))
    t1 = Token(facts=(f1,))
    t2 = Token(facts=(f2,))
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
    am = _any_am()
    jn = JoinNode(children=[pn], alpha_memory=am, left_input=DummyTopNode(), tests=[])
    f = Fact(Color("b1", "red"))
    jn.right_activate(f)
    assert len(cs) == 1
    assert cs[0].token.facts == (f,)
    assert cs[0].production is p


def test_join_node_with_pnode_child_right_retract():
    p = _make_production()
    cs: list[Instantiation] = []
    pn = PNode(production=p, conflict_set=cs)
    am = _any_am()
    jn = JoinNode(children=[pn], alpha_memory=am, left_input=DummyTopNode(), tests=[])
    f = Fact(Color("b1", "red"))
    jn.right_activate(f)
    jn.right_retract(f)
    assert cs == []
    assert pn.items == []


# ---------------------------------------------------------------------------
# Chain retraction
# ---------------------------------------------------------------------------


def test_chain_retract_fact_cascades():
    am1 = _any_am()
    am2 = _any_am()
    bm1 = BetaMemory()
    bm2 = BetaMemory()
    top = DummyTopNode()

    jn2 = JoinNode(children=[bm2], alpha_memory=am2, left_input=bm1, tests=[])
    bm1.successors = [jn2]
    jn1 = JoinNode(children=[bm1], alpha_memory=am1, left_input=top, tests=[])
    am1.successors = [jn1]
    am2.successors = [jn2]

    f1 = Fact(Color("b1", "red"))
    f2 = Fact(Size("b1", "large"))
    am1.activate(f1)
    am2.activate(f2)
    assert len(bm2.items) == 1

    am1.deactivate(f1)
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
    am = _any_am()
    rec = _Recorder()
    njn = NegativeJoinNode(
        children=[rec],
        alpha_memory=am,
        left_input=left if left is not None else DummyTopNode(),
        tests=tests or [],
    )
    am.successors.append(njn)
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


def test_njn_left_activate_matching_fact_blocks():
    njn, am, rec = _make_njn()
    am.items.append(Fact(Color("b1", "red")))
    t = Token()
    njn.left_activate(t)
    assert rec.activated == []
    assert njn.items[0].count == 1


def test_njn_left_activate_nonmatching_fact_propagates():
    f_prev = Fact(Color("block1", "red"))
    f_right = Fact(Size("block2", "large"))
    bm = BetaMemory(items=[Token(facts=(f_prev,), bindings={"$block": "block1"})])
    test = JoinTest("block", "$block")
    njn, am, rec = _make_njn(tests=[test], left=bm)
    am.items.append(f_right)  # block "block2" ≠ "$block" "block1" → test fails
    t = Token(facts=(f_prev,), bindings={"$block": "block1"})
    njn.left_activate(t)
    assert rec.activated == [t]
    assert njn.items[0].count == 0


# ---------------------------------------------------------------------------
# NegativeJoinNode — right_activate
# ---------------------------------------------------------------------------


def test_njn_right_activate_retracts_when_count_zero():
    njn, am, rec = _make_njn()
    t = Token()
    njn.left_activate(t)
    assert rec.activated == [t]
    f = Fact(Color("b1", "red"))
    njn.right_activate(f)
    assert rec.retracted == [t]
    assert njn.items[0].count == 1


def test_njn_right_activate_no_retract_when_already_blocked():
    njn, am, rec = _make_njn()
    am.items.append(Fact(Color("b1", "red")))
    t = Token()
    njn.left_activate(t)  # count 1 → blocked
    rec.retracted.clear()
    f2 = Fact(Size("b2", "large"))
    njn.right_activate(f2)
    assert rec.retracted == []
    assert njn.items[0].count == 2


def test_njn_right_activate_increments_count():
    njn, am, rec = _make_njn()
    t = Token()
    njn.left_activate(t)
    f = Fact(Color("b1", "red"))
    njn.right_activate(f)
    assert njn.items[0].count == 1


# ---------------------------------------------------------------------------
# NegativeJoinNode — right_retract
# ---------------------------------------------------------------------------


def test_njn_right_retract_asserts_when_count_reaches_zero():
    njn, am, rec = _make_njn()
    f = Fact(Color("b1", "red"))
    am.items.append(f)
    t = Token()
    njn.left_activate(t)  # count 1 → blocked
    rec.activated.clear()
    njn.right_retract(f)  # count → 0 → assert
    assert rec.activated == [t]
    assert njn.items[0].count == 0


def test_njn_right_retract_no_assert_when_count_stays_positive():
    njn, am, rec = _make_njn()
    f1 = Fact(Color("b1", "red"))
    f2 = Fact(Size("b2", "large"))
    am.items.extend([f1, f2])
    t = Token()
    njn.left_activate(t)  # count 2 → blocked
    rec.activated.clear()
    njn.right_retract(f1)  # count → 1 → still blocked
    assert rec.activated == []
    assert njn.items[0].count == 1


# ---------------------------------------------------------------------------
# NegativeJoinNode — left_retract
# ---------------------------------------------------------------------------


def test_njn_left_retract_propagated_token():
    njn, am, rec = _make_njn()
    t = Token()
    njn.left_activate(t)
    assert rec.activated == [t]
    njn.left_retract(t)
    assert rec.retracted == [t]
    assert njn.items == []


def test_njn_left_retract_blocked_token():
    njn, am, rec = _make_njn()
    am.items.append(Fact(Color("b1", "red")))
    t = Token()
    njn.left_activate(t)  # count 1 → blocked
    njn.left_retract(t)
    assert rec.retracted == []
    assert njn.items == []


# ---------------------------------------------------------------------------
# NegativeJoinNode — update_child
# ---------------------------------------------------------------------------


def test_njn_update_child_only_sends_propagated():
    njn, am, _rec = _make_njn()
    f = Fact(Color("b1", "red"))
    t0 = Token()
    t1 = Token(facts=(f,))
    njn.items.append(NegativeToken(token=t0, count=0))
    njn.items.append(NegativeToken(token=t1, count=1))
    new_child = _Recorder()
    njn.update_child(new_child)
    assert new_child.activated == [t0]
    assert new_child.retracted == []


# ---------------------------------------------------------------------------
# Bug regression — double-retraction via unextended token
# ---------------------------------------------------------------------------


def test_beta_memory_left_retract_idempotent():
    bm = BetaMemory()
    f = Fact(Color("b1", "red"))
    t = Token(facts=(f,))
    bm.left_activate(t)
    bm.left_retract(t)
    bm.left_retract(t)  # must be a no-op


def test_pnode_left_retract_idempotent():
    pn = PNode(production=_make_production(), conflict_set=[])
    f = Fact(Color("b1", "red"))
    t = Token(facts=(f,))
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
    f = Fact(Color("x", "blue"))
    result = Token(facts=(f,))
    partner.left_activate(result)
    assert result in ncc.new_result_buffer
    assert result in partner.items


def test_ncc_partner_activate_increments_count():
    ncc, partner, _rec = _make_ncc_pair(owner_length=0)
    base = Token()
    ncc_tok = NccToken(token=base, count=0)
    ncc.items.append(ncc_tok)
    f = Fact(Color("x", "blue"))
    result = Token(facts=(f,))
    partner.left_activate(result)
    assert ncc_tok.count == 1
    assert result in ncc_tok.results


def test_ncc_partner_activate_retracts_when_count_was_zero():
    ncc, partner, rec = _make_ncc_pair(owner_length=0)
    base = Token()
    ncc_tok = NccToken(token=base, count=0)
    ncc.items.append(ncc_tok)
    f = Fact(Color("x", "blue"))
    result = Token(facts=(f,))
    partner.left_activate(result)
    assert rec.retracted == [base]
    assert ncc_tok.count == 1


def test_ncc_partner_activate_no_retract_when_already_blocked():
    ncc, partner, rec = _make_ncc_pair(owner_length=0)
    base = Token()
    ncc_tok = NccToken(token=base, count=1)
    ncc.items.append(ncc_tok)
    f = Fact(Color("x", "blue"))
    result = Token(facts=(f,))
    partner.left_activate(result)
    assert rec.retracted == []
    assert ncc_tok.count == 2


# ---------------------------------------------------------------------------
# NccPartnerNode — left_retract
# ---------------------------------------------------------------------------


def test_ncc_partner_retract_from_buffer():
    ncc, partner, _rec = _make_ncc_pair(owner_length=0)
    f = Fact(Color("x", "blue"))
    result = Token(facts=(f,))
    partner.left_activate(result)
    partner.left_retract(result)
    assert result not in ncc.new_result_buffer
    assert result not in partner.items


def test_ncc_partner_retract_decrements_count():
    ncc, partner, _rec = _make_ncc_pair(owner_length=0)
    base = Token()
    ncc.items.append(NccToken(token=base, count=0))
    f = Fact(Color("x", "blue"))
    result = Token(facts=(f,))
    partner.left_activate(result)  # count: 0→1
    partner.left_retract(result)
    assert ncc.items[0].count == 0
    assert result not in ncc.items[0].results


def test_ncc_partner_retract_asserts_when_count_reaches_zero():
    ncc, partner, rec = _make_ncc_pair(owner_length=0)
    base = Token()
    ncc.items.append(NccToken(token=base, count=0))
    f = Fact(Color("x", "blue"))
    result = Token(facts=(f,))
    partner.left_activate(result)  # count: 0→1, triggers retract
    rec.retracted.clear()
    partner.left_retract(result)   # count: 1→0 → re-assert base
    assert rec.activated == [base]


def test_ncc_partner_retract_no_assert_count_stays_positive():
    ncc, partner, rec = _make_ncc_pair(owner_length=0)
    base = Token()
    ncc.items.append(NccToken(token=base, count=0))
    f1 = Fact(Color("x", "blue"))
    f2 = Fact(Color("y", "green"))
    result1 = Token(facts=(f1,))
    result2 = Token(facts=(f2,))
    partner.left_activate(result1)  # count: 0→1, retracts base
    partner.left_activate(result2)  # count: 1→2
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
    f = Fact(Color("x", "blue"))
    result = Token(facts=(f,))
    ncc.new_result_buffer.append(result)
    t = Token()
    ncc.left_activate(t)
    assert rec.activated == []
    assert ncc.items[0].count == 1
    assert ncc.new_result_buffer == []


def test_ncc_node_activate_ignores_nonmatching_buffer():
    f_main = Fact(Color("b1", "red"))
    f_other = Fact(Color("b2", "blue"))
    ncc, _partner, rec = _make_ncc_pair(owner_length=1)
    # result whose first fact is f_other — doesn't match left token (f_main,)
    result = Token(facts=(f_other, Fact(Size("x", "large"))))
    ncc.new_result_buffer.append(result)
    t = Token(facts=(f_main,))
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
    f = Fact(Color("x", "blue"))
    result = Token(facts=(f,))
    partner.left_activate(result)               # count: 0→1, retracts from rec
    rec.retracted.clear()
    ncc.left_retract(t)                         # count=1 → children NOT called
    assert rec.retracted == []
    assert ncc.items == []


def test_ncc_node_retract_clears_partner_items():
    ncc, partner, _rec = _make_ncc_pair(owner_length=0)
    t = Token()
    f = Fact(Color("x", "blue"))
    result = Token(facts=(f,))
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
    f = Fact(Color("b1", "red"))
    t0 = Token()
    t1 = Token(facts=(f,))
    ncc.items.append(NccToken(token=t0, count=0))
    ncc.items.append(NccToken(token=t1, count=1))
    new_child = _Recorder()
    ncc.update_child(new_child)
    assert new_child.activated == [t0]
    assert new_child.retracted == []


# ---------------------------------------------------------------------------
# NCC — owner_length prefix matching uses Fact identity
# ---------------------------------------------------------------------------


def test_ncc_owner_prefix_uses_fact_identity():
    """Facts in result tokens must be the same objects as in owner token."""
    f_owner = Fact(Color("b1", "red"))
    f_sub = Fact(Size("b1", "large"))
    ncc, partner, rec = _make_ncc_pair(owner_length=1)

    # owner token carries f_owner
    owner_token = Token(facts=(f_owner,))
    ncc.left_activate(owner_token)  # count=0 → propagated
    assert rec.activated == [owner_token]
    rec.activated.clear()

    # result token's first fact IS f_owner (same object)
    result = Token(facts=(f_owner, f_sub))
    partner.left_activate(result)   # prefix (f_owner,) matches → count: 0→1 → retract
    assert rec.retracted == [owner_token]
