"""Unit tests for alpha.py — AlphaMemory, AlphaNode, RootNode, build helpers.

:see: Doorenbos §2.2
"""
from rete.alpha import (
    AlphaMemory,
    AlphaNode,
    RootNode,
    build_or_share_alpha_node,
)
from rete.condition import WILDCARD, Condition
from rete.wme import WME


# ---------------------------------------------------------------------------
# AlphaMemory
# ---------------------------------------------------------------------------


def test_alpha_memory_activate_stores_wme():
    mem = AlphaMemory()
    w = WME("b1", "color", "red")
    mem.activate(w)
    assert w in mem.items


def test_alpha_memory_activate_sets_back_pointer():
    mem = AlphaMemory()
    w = WME("b1", "color", "red")
    mem.activate(w)
    assert mem in w.alpha_memories


def test_alpha_memory_deactivate_removes_wme():
    mem = AlphaMemory()
    w = WME("b1", "color", "red")
    mem.activate(w)
    mem.deactivate(w)
    assert w not in mem.items


def test_alpha_memory_deactivate_clears_back_pointer():
    mem = AlphaMemory()
    w = WME("b1", "color", "red")
    mem.activate(w)
    mem.deactivate(w)
    assert mem not in w.alpha_memories


# ---------------------------------------------------------------------------
# AlphaNode
# ---------------------------------------------------------------------------


def test_alpha_node_matching_wme_reaches_memory():
    mem = AlphaMemory()
    node = AlphaNode(field="attribute", symbol="color", output_memory=mem)
    w = WME("b1", "color", "red")
    node.activate(w)
    assert w in mem.items


def test_alpha_node_nonmatching_wme_stops():
    mem = AlphaMemory()
    node = AlphaNode(field="attribute", symbol="color", output_memory=mem)
    w = WME("b1", "size", "large")
    node.activate(w)
    assert w not in mem.items


def test_alpha_node_propagates_to_child():
    mem = AlphaMemory()
    child = AlphaNode(field="value", symbol="red", output_memory=mem)
    parent = AlphaNode(field="attribute", symbol="color", children=[child])
    w = WME("b1", "color", "red")
    parent.activate(w)
    assert w in mem.items


def test_alpha_node_nonmatching_does_not_reach_child():
    mem = AlphaMemory()
    child = AlphaNode(field="value", symbol="red", output_memory=mem)
    parent = AlphaNode(field="attribute", symbol="color", children=[child])
    w = WME("b1", "size", "large")
    parent.activate(w)
    assert w not in mem.items


def test_alpha_node_chain_both_match():
    mem = AlphaMemory()
    node2 = AlphaNode(field="value", symbol="red", output_memory=mem)
    node1 = AlphaNode(field="attribute", symbol="color", children=[node2])
    w = WME("b1", "color", "red")
    node1.activate(w)
    assert w in mem.items


def test_alpha_node_chain_first_fails():
    mem = AlphaMemory()
    node2 = AlphaNode(field="value", symbol="red", output_memory=mem)
    node1 = AlphaNode(field="attribute", symbol="color", children=[node2])
    w = WME("b1", "size", "red")
    node1.activate(w)
    assert w not in mem.items


# ---------------------------------------------------------------------------
# RootNode
# ---------------------------------------------------------------------------


def test_root_node_fans_out_to_all_children():
    mem1 = AlphaMemory()
    mem2 = AlphaMemory()
    node1 = AlphaNode(field="attribute", symbol="color", output_memory=mem1)
    node2 = AlphaNode(field="attribute", symbol="size", output_memory=mem2)
    root = RootNode(children=[node1, node2])
    w_color = WME("b1", "color", "red")
    w_size = WME("b2", "size", "large")
    root.activate(w_color)
    root.activate(w_size)
    assert w_color in mem1.items
    assert w_size in mem2.items
    assert w_color not in mem2.items
    assert w_size not in mem1.items


# ---------------------------------------------------------------------------
# build_or_share_alpha_node
# ---------------------------------------------------------------------------


def test_build_or_share_creates_new_node():
    root = RootNode()
    node = build_or_share_alpha_node(root, "attribute", "color")
    assert node in root.children
    assert node.field == "attribute"
    assert node.symbol == "color"


def test_build_or_share_reuses_existing_node():
    root = RootNode()
    n1 = build_or_share_alpha_node(root, "attribute", "color")
    n2 = build_or_share_alpha_node(root, "attribute", "color")
    assert n1 is n2
    assert len(root.children) == 1


def test_build_or_share_different_symbol_creates_new():
    root = RootNode()
    n1 = build_or_share_alpha_node(root, "attribute", "color")
    n2 = build_or_share_alpha_node(root, "attribute", "size")
    assert n1 is not n2
    assert len(root.children) == 2


# ---------------------------------------------------------------------------
# build_or_share_alpha_memory
# ---------------------------------------------------------------------------


def test_build_or_share_alpha_memory_constant_condition():
    root = RootNode()
    cond = Condition("b1", "color", "red")
    mem = root.build_or_share_alpha_memory(cond)
    assert isinstance(mem, AlphaMemory)
    w = WME("b1", "color", "red")
    root.activate(w)
    assert w in mem.items


def test_build_or_share_alpha_memory_wrong_wme_excluded():
    root = RootNode()
    cond = Condition("b1", "color", "red")
    mem = root.build_or_share_alpha_memory(cond)
    w = WME("b1", "color", "blue")
    root.activate(w)
    assert w not in mem.items


def test_build_or_share_alpha_memory_wildcard_skipped():
    root = RootNode()
    cond = Condition(WILDCARD, "color", WILDCARD)
    mem = root.build_or_share_alpha_memory(cond)
    w = WME("anything", "color", "whatever")
    root.activate(w)
    assert w in mem.items


def test_build_or_share_alpha_memory_variable_skipped():
    root = RootNode()
    cond = Condition("?x", "color", "?v")
    mem = root.build_or_share_alpha_memory(cond)
    w = WME("b1", "color", "red")
    root.activate(w)
    assert w in mem.items


def test_build_or_share_alpha_memory_shared_prefix():
    root = RootNode()
    cond1 = Condition(WILDCARD, "color", "red")
    cond2 = Condition(WILDCARD, "color", "blue")
    root.build_or_share_alpha_memory(cond1)
    root.build_or_share_alpha_memory(cond2)
    # Both conditions share the attribute=="color" node at root level
    assert len(root.children) == 1
    color_node = root.children[0]
    assert color_node.field == "attribute"
    assert color_node.symbol == "color"
    assert len(color_node.children) == 2


def test_build_or_share_alpha_memory_all_wildcards():
    root = RootNode()
    cond = Condition(WILDCARD, WILDCARD, WILDCARD)
    mem = root.build_or_share_alpha_memory(cond)
    assert isinstance(mem, AlphaMemory)
    w = WME("x", "y", "z")
    root.activate(w)
    assert w in mem.items
