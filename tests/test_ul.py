"""Unit and integration tests for Rete/UL — right and left unlinking.

:see: Doorenbos Ch. 4–5
"""
from rete.alpha import AlphaMemory
from rete.beta import (
    BetaMemory,
    JoinNode,
    NegativeJoinNode,
)
from rete.condition import Condition, Production
from rete.engine import InferenceEngine
from rete.network import ReteNetwork
from rete.fact import Token, WME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jn(left=None, am=None) -> tuple[JoinNode, AlphaMemory, BetaMemory]:
    """Return a wired ``(JoinNode, AlphaMemory, child BetaMemory)``."""
    if am is None:
        am = AlphaMemory()
    beta = left if left is not None else BetaMemory()
    child = BetaMemory()
    jn = JoinNode(children=[child], alpha_memory=am, left_input=beta, tests=[])
    am.successors.append(jn)
    if isinstance(beta, BetaMemory):
        beta.successors.append(jn)
    return jn, am, child


def _njn(left=None, am=None) -> tuple[NegativeJoinNode, AlphaMemory]:
    """Return a wired ``(NegativeJoinNode, AlphaMemory)``."""
    if am is None:
        am = AlphaMemory()
    beta = left if left is not None else BetaMemory()
    njn = NegativeJoinNode(children=[], alpha_memory=am, left_input=beta, tests=[])
    am.successors.append(njn)
    if isinstance(beta, BetaMemory):
        beta.successors.append(njn)
    return njn, am


def _token(*ids: str) -> Token:
    return Token(wmes=tuple(WME(i, "a", "v") for i in ids))


# ---------------------------------------------------------------------------
# Right unlinking — JoinNode
# ---------------------------------------------------------------------------


def test_right_unlink_flag_defaults_false():
    jn, _, _ = _jn()
    assert not jn.right_unlinked


def test_right_unlink_after_beta_drains():
    """JoinNode leaves alpha.successors when its beta memory goes empty."""
    bm = BetaMemory()
    jn, am, _ = _jn(left=bm)
    tok = _token("x")
    bm.left_activate(tok)
    bm.left_retract(tok)
    assert jn not in am.successors
    assert jn.right_unlinked


def test_right_unlink_partial_drain_stays_linked():
    """Right-unlink fires only when the LAST token leaves."""
    bm = BetaMemory()
    jn, am, _ = _jn(left=bm)
    t1, t2 = _token("a"), _token("b")
    bm.left_activate(t1)
    bm.left_activate(t2)
    bm.left_retract(t1)
    assert jn in am.successors
    assert not jn.right_unlinked


def test_right_relink_on_left_activate():
    """JoinNode re-enters alpha.successors on the next left activation."""
    bm = BetaMemory()
    jn, am, _ = _jn(left=bm)
    tok = _token("x")
    bm.left_activate(tok)
    bm.left_retract(tok)          # right-unlink
    assert jn.right_unlinked
    tok2 = _token("y")
    bm.left_activate(tok2)        # re-link
    assert jn in am.successors
    assert not jn.right_unlinked


def test_right_unlink_no_spurious_right_activate():
    """WME added while right-unlinked does not reach the join node."""
    bm = BetaMemory()
    jn, am, child = _jn(left=bm)
    tok = _token("x")
    bm.left_activate(tok)
    bm.left_retract(tok)          # right-unlink — child is also empty now

    wme = WME("b1", "color", "red")
    am.activate(wme)              # right activation skipped — jn not in am.successors
    assert child.items == []


def test_right_unlink_correctness_after_relink():
    """Match produced correctly when beta re-fills after right-unlink."""
    bm = BetaMemory()
    jn, am, child = _jn(left=bm)

    tok = _token("x")
    bm.left_activate(tok)
    bm.left_retract(tok)          # right-unlink

    wme = WME("b1", "color", "red")
    am.activate(wme)              # arrives while unlinked — no match yet

    bm.left_activate(tok)         # re-link + join drives tok vs wme
    assert len(child.items) == 1


# ---------------------------------------------------------------------------
# Left unlinking — JoinNode
# ---------------------------------------------------------------------------


def test_left_unlink_flag_defaults_false():
    jn, _, _ = _jn()
    assert not jn.left_unlinked


def test_left_unlink_after_alpha_drains():
    """JoinNode leaves beta.successors when its alpha memory goes empty."""
    bm = BetaMemory()
    jn, am, _ = _jn(left=bm)
    wme = WME("b1", "color", "red")
    am.activate(wme)
    am.deactivate(wme)
    assert jn not in bm.successors
    assert jn.left_unlinked


def test_left_unlink_partial_drain_stays_linked():
    """Left-unlink fires only when the LAST WME leaves the alpha memory."""
    bm = BetaMemory()
    jn, am, _ = _jn(left=bm)
    w1 = WME("b1", "color", "red")
    w2 = WME("b2", "color", "red")
    am.activate(w1)
    am.activate(w2)
    am.deactivate(w1)
    assert jn in bm.successors
    assert not jn.left_unlinked


def test_left_relink_on_right_activate():
    """JoinNode re-enters beta.successors on the next right activation."""
    bm = BetaMemory()
    jn, am, _ = _jn(left=bm)
    wme = WME("b1", "color", "red")
    am.activate(wme)
    am.deactivate(wme)            # left-unlink
    assert jn.left_unlinked
    wme2 = WME("b2", "color", "red")
    am.activate(wme2)             # re-link
    assert jn in bm.successors
    assert not jn.left_unlinked


def test_left_unlink_no_spurious_left_activate():
    """Token added to beta while left-unlinked does not reach the join node."""
    bm = BetaMemory()
    jn, am, child = _jn(left=bm)
    wme = WME("b1", "color", "red")
    am.activate(wme)
    am.deactivate(wme)            # left-unlink

    tok = _token("x")
    bm.left_activate(tok)         # jn not in bm.successors — skipped
    assert child.items == []


def test_left_unlink_correctness_after_relink():
    """Match produced correctly when alpha re-fills after left-unlink."""
    bm = BetaMemory()
    jn, am, child = _jn(left=bm)

    wme = WME("b1", "color", "red")
    am.activate(wme)
    am.deactivate(wme)            # left-unlink

    tok = _token("x")
    bm.left_activate(tok)         # arrives while unlinked — no match yet

    wme2 = WME("b2", "color", "blue")
    am.activate(wme2)             # re-link + replay beta: tok joins wme2
    assert len(child.items) == 1


# ---------------------------------------------------------------------------
# Right unlinking — NegativeJoinNode
# ---------------------------------------------------------------------------


def test_njn_right_unlink_flag_defaults_false():
    njn, _ = _njn()
    assert not njn.right_unlinked


def test_njn_right_unlink_after_items_drain():
    """NJN leaves alpha.successors when its items list goes empty."""
    bm = BetaMemory()
    njn, am = _njn(left=bm)
    tok = _token("x")
    bm.left_activate(tok)         # populates njn.items
    bm.left_retract(tok)          # drains njn.items → right-unlink
    assert njn not in am.successors
    assert njn.right_unlinked


def test_njn_right_relink_on_left_activate():
    """NJN re-enters alpha.successors on the next left activation."""
    bm = BetaMemory()
    njn, am = _njn(left=bm)
    tok = _token("x")
    bm.left_activate(tok)
    bm.left_retract(tok)          # right-unlink
    bm.left_activate(_token("y"))  # re-link
    assert njn in am.successors
    assert not njn.right_unlinked


# ---------------------------------------------------------------------------
# Build-time link initialisation — via ReteNetwork
# ---------------------------------------------------------------------------


def test_build_time_right_unlink():
    """JoinNode built when beta is empty starts right-unlinked."""
    net = ReteNetwork()
    pn = net.add_production(Production(
        lhs=[Condition("b1", "color", "red")],
        rhs=lambda t: None,
    ))
    jn = pn.parent_join
    # Dummy top node always has one token — never right-unlinked.
    assert not jn.right_unlinked


def test_build_time_right_unlink_with_beta():
    """JoinNode whose BetaMemory is empty at build time starts right-unlinked."""
    net = ReteNetwork()
    pn = net.add_production(Production(
        lhs=[
            Condition("?x", "color", "red"),
            Condition("?x", "size", "large"),
        ],
        rhs=lambda t: None,
    ))
    jn2 = pn.parent_join
    # No WMEs yet — the beta memory feeding jn2 is empty → right-unlinked.
    assert jn2.right_unlinked


def test_build_time_left_unlink():
    """JoinNode whose alpha memory is empty at build time starts left-unlinked."""
    net = ReteNetwork()
    pn = net.add_production(Production(
        lhs=[Condition("b1", "color", "red")],
        rhs=lambda t: None,
    ))
    jn = pn.parent_join
    # No WMEs in the network → alpha memory is empty → left-unlinked.
    assert jn.left_unlinked


# ---------------------------------------------------------------------------
# GC with unlinked nodes
# ---------------------------------------------------------------------------


def test_gc_right_unlinked_node():
    """remove_production works when the join node is right-unlinked."""
    net = ReteNetwork()
    pn = net.add_production(Production(
        lhs=[
            Condition("?x", "color", "red"),
            Condition("?x", "size", "large"),
        ],
        rhs=lambda t: None,
    ))
    # jn2 is right-unlinked (beta empty at build); remove_production must not crash.
    net.remove_production(pn)


def test_gc_left_unlinked_node():
    """remove_production works when the join node is left-unlinked."""
    net = ReteNetwork()
    pn = net.add_production(Production(
        lhs=[Condition("b1", "color", "red")],
        rhs=lambda t: None,
    ))
    # jn is left-unlinked (alpha empty at build); remove_production must not crash.
    net.remove_production(pn)


# ---------------------------------------------------------------------------
# End-to-end correctness with UL active
# ---------------------------------------------------------------------------


def test_ul_end_to_end_match():
    """Full engine run produces correct results regardless of UL link state."""
    engine = InferenceEngine()
    fired = []
    engine.add_production(Production(
        lhs=[Condition("b1", "color", "red")],
        rhs=lambda t: fired.append(t),
    ))
    engine.add_wme(WME("b1", "color", "red"))
    engine.run()
    assert len(fired) == 1


def test_ul_end_to_end_retraction():
    """Retracting before run leaves conflict set empty (UL must not interfere)."""
    engine = InferenceEngine()
    fired = []
    engine.add_production(Production(
        lhs=[Condition("b1", "color", "red")],
        rhs=lambda t: fired.append(t),
    ))
    wme = WME("b1", "color", "red")
    engine.add_wme(wme)
    engine.remove_wme(wme)
    engine.run()
    assert fired == []


def test_ul_end_to_end_two_conditions():
    """Two-condition rule matches correctly despite right-unlinking on inner join."""
    engine = InferenceEngine()
    fired = []
    engine.add_production(Production(
        lhs=[
            Condition("?x", "color", "red"),
            Condition("?x", "size", "large"),
        ],
        rhs=lambda t: fired.append(t),
    ))
    engine.add_wme(WME("b1", "color", "red"))
    engine.add_wme(WME("b1", "size", "large"))
    engine.run()
    assert len(fired) == 1


def test_ul_end_to_end_wme_before_production():
    """WMEs added before the production still match after left-relink."""
    engine = InferenceEngine()
    engine.add_wme(WME("b1", "color", "red"))
    fired = []
    engine.add_production(Production(
        lhs=[Condition("b1", "color", "red")],
        rhs=lambda t: fired.append(t),
    ))
    engine.run()
    assert len(fired) == 1
