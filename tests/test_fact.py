"""Tests for Fact and Token."""
from dataclasses import dataclass

from rete.fact import Fact, Token


@dataclass
class Block:
    """Minimal POPO used as a test fact object."""

    color: str


def test_fact_wraps_object():
    b = Block("red")
    f = Fact(b)
    assert f.obj is b


def test_fact_back_pointers_start_empty():
    f = Fact(Block("red"))
    assert f.alpha_memories == []
    assert f.beta_tokens == []


def test_fact_identity_hash():
    b = Block("red")
    f1 = Fact(b)
    f2 = Fact(b)
    assert f1 != f2
    assert len({f1, f2}) == 2


def test_fact_same_object_distinct():
    b = Block("red")
    assert Fact(b) is not Fact(b)


def test_token_empty_defaults():
    t = Token()
    assert t.facts == ()
    assert t.bindings == {}


def test_token_single_fact():
    f = Fact(Block("red"))
    t = Token(facts=(f,))
    assert t.facts[0] is f


def test_token_bindings_stored():
    t = Token(facts=(), bindings={"$x": 42})
    assert t.bindings["$x"] == 42


def test_token_immutable_extension():
    f1 = Fact(Block("red"))
    f2 = Fact(Block("blue"))
    t1 = Token(facts=(f1,), bindings={"$a": 1})
    t2 = Token(facts=t1.facts + (f2,), bindings={**t1.bindings, "$b": 2})
    assert len(t1.facts) == 1
    assert len(t2.facts) == 2
    assert "$b" not in t1.bindings
    assert t2.facts[:-1] == t1.facts


def test_token_bindings_merge():
    f = Fact(Block("red"))
    parent = Token(facts=(), bindings={"$x": 10})
    child = Token(facts=(f,), bindings={**parent.bindings, "$y": 20})
    assert child.bindings["$x"] == 10
    assert child.bindings["$y"] == 20


def test_token_bindings_new_overwrites_parent():
    """New bindings shadow parent bindings for the same key."""
    f = Fact(Block("red"))
    parent = Token(facts=(), bindings={"$x": 10})
    child = Token(facts=(f,), bindings={**parent.bindings, "$x": 99})
    assert child.bindings["$x"] == 99
