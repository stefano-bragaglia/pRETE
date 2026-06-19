"""Doorenbos §2.1 — canonical blocks-world example (Figure 2.2b).

Production:  find-stack-of-two-blocks-to-the-left-of-a-red-block
  C1: (?x ^on ?y)
  C2: (?y ^left-of ?z)
  C3: (?z ^color red)

Working memory w1–w9 from the thesis.
Expected single match: w1 ^ w5 ^ w9  →  x=B1, y=B2, z=B3
"""
from __future__ import annotations

from rete import Condition, Production, ReteNetwork, WME


def main() -> None:
    net = ReteNetwork()

    def rhs(token):
        x = token.wmes[0].id
        y = token.wmes[0].value
        z = token.wmes[1].value
        print(f"  fired: x={x}, y={y}, z={z}")

    net.add_production(Production(
        lhs=[
            Condition("?x", "on",      "?y"),     # C1
            Condition("?y", "left-of", "?z"),     # C2
            Condition("?z", "color",   "red"),    # C3
        ],
        rhs=rhs,
    ))

    # Working memory w1–w9 (Doorenbos §2.1)
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
        rhs(inst.token)

    assert len(net.conflict_set) == 1
    t = net.conflict_set[0].token
    assert (t.wmes[0].id, t.wmes[0].value, t.wmes[1].value) == ("B1", "B2", "B3")


if __name__ == "__main__":
    main()
