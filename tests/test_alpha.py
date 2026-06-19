"""Unit tests for alpha.py — AlphaMemory and RootNode with POPO dispatch.

:see: Doorenbos §2.2, UPDATE_PLAN §Step 3
"""
from dataclasses import dataclass
from unittest.mock import MagicMock

from rete.alpha import AlphaMemory, RootNode
from rete.condition import Pattern
from rete.fact import Fact


@dataclass
class Block:
    """Minimal POPO: a coloured block."""

    color: str


@dataclass
class On:
    """Minimal POPO: upper block resting on lower."""

    upper: str
    lower: str


class Animal:
    """Base class for MRO-dispatch tests."""


class Dog(Animal):
    """Subclass for MRO-dispatch tests."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _accept_all(fact: Fact) -> bool:
    return True


def _reject_all(fact: Fact) -> bool:
    return False


# ---------------------------------------------------------------------------
# AlphaMemory
# ---------------------------------------------------------------------------


def test_alpha_memory_activate_stores_fact():
    mem = AlphaMemory(type_=Block, predicate=_accept_all)
    f = Fact(Block("red"))
    mem.activate(f)
    assert f in mem.items


def test_alpha_memory_activate_sets_back_pointer():
    mem = AlphaMemory(type_=Block, predicate=_accept_all)
    f = Fact(Block("red"))
    mem.activate(f)
    assert mem in f.alpha_memories


def test_alpha_memory_predicate_filters():
    def is_blue(fact: Fact) -> bool:
        return fact.obj.color == "blue"

    mem = AlphaMemory(type_=Block, predicate=is_blue)
    f = Fact(Block("red"))
    mem.activate(f)
    assert f not in mem.items


def test_alpha_memory_predicate_filter_skips_back_pointer():
    def is_blue(fact: Fact) -> bool:
        return fact.obj.color == "blue"

    mem = AlphaMemory(type_=Block, predicate=is_blue)
    f = Fact(Block("red"))
    mem.activate(f)
    assert mem not in f.alpha_memories


def test_alpha_memory_deactivate_removes_fact():
    mem = AlphaMemory(type_=Block, predicate=_accept_all)
    f = Fact(Block("red"))
    mem.activate(f)
    mem.deactivate(f)
    assert f not in mem.items


def test_alpha_memory_deactivate_clears_back_pointer():
    mem = AlphaMemory(type_=Block, predicate=_accept_all)
    f = Fact(Block("red"))
    mem.activate(f)
    mem.deactivate(f)
    assert mem not in f.alpha_memories


def test_alpha_memory_notifies_successors_on_activate():
    successor = MagicMock()
    mem = AlphaMemory(type_=Block, predicate=_accept_all, successors=[successor])
    f = Fact(Block("red"))
    mem.activate(f)
    successor.right_activate.assert_called_once_with(f)


def test_alpha_memory_notifies_successors_before_remove():
    removed_during_retract: list[bool] = []
    mem = AlphaMemory(type_=Block, predicate=_accept_all)
    f = Fact(Block("red"))
    mem.activate(f)

    class Probe:
        def right_retract(self, fact: Fact) -> None:
            removed_during_retract.append(fact in mem.items)

    mem.successors.append(Probe())
    mem.deactivate(f)
    assert removed_during_retract == [True]


# ---------------------------------------------------------------------------
# RootNode.activate / deactivate
# ---------------------------------------------------------------------------


def test_root_activate_stores_fact():
    root = RootNode()
    f = Fact(Block("red"))
    root.activate(f)
    assert f in root._facts


def test_root_deactivate_removes_fact():
    root = RootNode()
    f = Fact(Block("red"))
    root.activate(f)
    root.deactivate(f)
    assert f not in root._facts


def test_root_dispatches_matching_type():
    root = RootNode()
    mem = root.build_or_share_alpha_memory(Pattern(type_=Block))
    f = Fact(Block("red"))
    root.activate(f)
    assert f in mem.items


def test_root_does_not_dispatch_wrong_type():
    root = RootNode()
    mem = root.build_or_share_alpha_memory(Pattern(type_=Block))
    f = Fact(On("a", "b"))
    root.activate(f)
    assert f not in mem.items


def test_root_mro_dispatch():
    """A Dog fact must activate a Pattern(type_=Animal) alpha memory."""
    root = RootNode()
    mem = root.build_or_share_alpha_memory(Pattern(type_=Animal))
    f = Fact(Dog())
    root.activate(f)
    assert f in mem.items


def test_root_predicate_filters_at_dispatch():
    # alpha_tests receive the raw object (fact.obj), not the Fact wrapper
    def is_red(obj: Block) -> bool:
        return obj.color == "red"

    root = RootNode()
    mem = root.build_or_share_alpha_memory(
        Pattern(type_=Block, alpha_tests=(is_red,))
    )
    f = Fact(Block("blue"))
    root.activate(f)
    assert f not in mem.items


# ---------------------------------------------------------------------------
# RootNode.build_or_share_alpha_memory
# ---------------------------------------------------------------------------


def test_build_or_share_returns_alpha_memory():
    root = RootNode()
    mem = root.build_or_share_alpha_memory(Pattern(type_=Block))
    assert isinstance(mem, AlphaMemory)


def test_build_or_share_sharing_same_function():
    root = RootNode()
    p1 = Pattern(type_=Block, alpha_tests=(_accept_all,))
    p2 = Pattern(type_=Block, alpha_tests=(_accept_all,))
    assert root.build_or_share_alpha_memory(p1) is root.build_or_share_alpha_memory(p2)


def test_build_or_share_no_sharing_distinct_lambdas():
    root = RootNode()
    p1 = Pattern(type_=Block, alpha_tests=(lambda f: True,))
    p2 = Pattern(type_=Block, alpha_tests=(lambda f: True,))
    m1 = root.build_or_share_alpha_memory(p1)
    m2 = root.build_or_share_alpha_memory(p2)
    assert m1 is not m2


def test_build_or_share_different_types_not_shared():
    root = RootNode()
    m1 = root.build_or_share_alpha_memory(Pattern(type_=Block))
    m2 = root.build_or_share_alpha_memory(Pattern(type_=On))
    assert m1 is not m2


def test_build_or_share_replays_existing_facts():
    root = RootNode()
    f = Fact(Block("red"))
    root.activate(f)
    mem = root.build_or_share_alpha_memory(Pattern(type_=Block))
    assert f in mem.items


def test_build_or_share_replay_filtered_by_predicate():
    # alpha_tests receive the raw object (fact.obj), not the Fact wrapper
    def is_red(obj: Block) -> bool:
        return obj.color == "red"

    root = RootNode()
    f = Fact(Block("blue"))
    root.activate(f)
    mem = root.build_or_share_alpha_memory(
        Pattern(type_=Block, alpha_tests=(is_red,))
    )
    assert f not in mem.items


def test_build_or_share_no_double_registration():
    """Calling twice with the same pattern must not activate a fact twice."""
    root = RootNode()
    p = Pattern(type_=Block)
    mem = root.build_or_share_alpha_memory(p)
    root.build_or_share_alpha_memory(p)  # second call — must be a no-op
    f = Fact(Block("red"))
    root.activate(f)
    assert mem.items.count(f) == 1


def test_build_or_share_replay_sets_back_pointer():
    """Replayed facts must get their back-pointer set so retraction works."""
    root = RootNode()
    f = Fact(Block("red"))
    root.activate(f)
    mem = root.build_or_share_alpha_memory(Pattern(type_=Block))
    assert mem in f.alpha_memories
