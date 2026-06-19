from rete.wme import Token, WME


def test_wme_fields():
    w = WME("b1", "color", "red")
    assert w.id == "b1"
    assert w.attribute == "color"
    assert w.value == "red"


def test_wme_back_pointers_start_empty():
    w = WME("b1", "color", "red")
    assert w.alpha_memories == []
    assert w.beta_tokens == []


def test_wme_identity_hash():
    w1 = WME("b1", "color", "red")
    w2 = WME("b1", "color", "red")
    assert w1 != w2
    assert len({w1, w2}) == 2


def test_token_empty():
    t = Token()
    assert t.wmes == ()


def test_token_single_wme():
    w = WME("b1", "color", "red")
    t = Token(wmes=(w,))
    assert t.wmes[0] is w


def test_token_append_immutable():
    w1 = WME("b1", "color", "red")
    w2 = WME("b2", "size", "large")
    t1 = Token(wmes=(w1,))
    t2 = Token(wmes=t1.wmes + (w2,))
    assert len(t1.wmes) == 1
    assert len(t2.wmes) == 2


def test_token_wmes_ordered():
    w1 = WME("b1", "color", "red")
    w2 = WME("b2", "size", "large")
    t = Token(wmes=(w1, w2))
    assert t.wmes[0] is w1
    assert t.wmes[1] is w2
