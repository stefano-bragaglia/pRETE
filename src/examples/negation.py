"""Doorenbos §2.7 — negative conditions (NegativeJoinNode).

Production:  find-blocks-on-table-not-blue
  C1:  (?x ^on table)         ← positive: x is on the table
  -C2: (?x ^color blue)       ← negated:  x must NOT be blue

Working memory w1–w9 from the thesis.
  B2 is on table (w4) AND is blue (w6) → excluded by -C2
  B3 is on table (w8) AND is red  (w9) → passes -C2 → fires

Expected single match: w8  →  x=B3
"""
from __future__ import annotations

from rete import Condition, Production, ReteNetwork, WME


def main() -> None:
    net = ReteNetwork()

    def rhs(token):
        # Negated conditions do not contribute a WME to the token.
        x = token.wmes[0].id
        print(f"  fired: x={x} is on the table and not blue")

    net.add_production(Production(
        lhs=[
            Condition("?x", "on",    "table"),              # C1  positive
            Condition("?x", "color", "blue", negated=True), # -C2 negated
        ],
        rhs=rhs,
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
        rhs(inst.token)

    assert len(net.conflict_set) == 1
    assert net.conflict_set[0].token.wmes[0].id == "B3"


if __name__ == "__main__":
    main()
