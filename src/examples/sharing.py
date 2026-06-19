"""Doorenbos §2.3 — node-sharing between two productions.

Two productions with identical C1^C2 share one beta memory node:

  P1 (find-stack-of-two-blocks-to-the-left-of-a-red-block):
    C1: (?x ^on ?y)
    C2: (?y ^left-of ?z)
    C3: (?z ^color red)          ← tests z

  P2 (slightly-modified-version):
    C1: (?x ^on ?y)              ← same as P1
    C2: (?y ^left-of ?z)         ← same as P1  → shared beta memory
    C3': (?y ^color red)         ← tests y, not z

Working memory w1–w9 from the thesis.
Expected:
  P1 fires for w1^w5^w9  →  x=B1, y=B2, z=B3
  P2 fires for w2^w7^w9  →  x=B1, y=B3, z=B4
"""
from __future__ import annotations

from rete import Condition, Production, ReteNetwork, WME


def _make_rhs(name):
    def rhs(token):
        x = token.wmes[0].id
        y = token.wmes[0].value
        z = token.wmes[1].value
        print(f"  {name} fired: x={x}, y={y}, z={z}")
    return rhs


def main() -> None:
    net = ReteNetwork()

    net.add_production(Production(
        lhs=[
            Condition("?x", "on",      "?y"),   # C1
            Condition("?y", "left-of", "?z"),   # C2
            Condition("?z", "color",   "red"),  # C3  — tests z
        ],
        rhs=_make_rhs("P1"),
    ))

    net.add_production(Production(
        lhs=[
            Condition("?x", "on",      "?y"),   # C1  } shared beta memory
            Condition("?y", "left-of", "?z"),   # C2  }
            Condition("?y", "color",   "red"),  # C3' — tests y (not z)
        ],
        rhs=_make_rhs("P2"),
    ))

    wmes = [
        WME("B1", "on",      "B2"),    # w1
        WME("B1", "on",      "B3"),    # w2
        WME("B1", "color",   "red"),   # w3
        WME("B2", "on",      "table"), # w4
        WME("B2", "left-of", "B3"),    # w5
        WME("B2", "color",   "blue"),  # w6
        WME("B3", "left-of", "B4"),    # w7
        WME("B3", "on",      "table"), # w8
        WME("B3", "color",   "red"),   # w9
    ]

    print("Adding WMEs w1–w9:")
    for i, wme in enumerate(wmes, 1):
        print(f"  w{i}: ({wme.id} ^{wme.attribute} {wme.value})")
        net.add_wme(wme)

    print(f"\nConflict set: {len(net.conflict_set)} instantiation(s)")
    for inst in net.conflict_set:
        inst.production.rhs(inst.token)

    bindings = {
        (i.token.wmes[0].id, i.token.wmes[0].value, i.token.wmes[1].value)
        for i in net.conflict_set
    }
    assert bindings == {("B1", "B2", "B3"), ("B1", "B3", "B4")}


if __name__ == "__main__":
    main()
