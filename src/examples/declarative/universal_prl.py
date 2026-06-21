"""ES-6 forall (universal quantification) example loaded from PRL.

Rule fires when every pending Order has a matching Approval.
Retracting one Approval un-fires it.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "universal.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _run() -> list[str]:
    """Return the results list after two-phase run — used by tests.

    Phase 1: insert two pending orders and both approvals → rule fires.
    Phase 2: retract one approval → rule un-fires (no new append).
    """
    results: list = []
    engine, types = _setup({"results": results})
    Order = types["Order"]
    Approval = types["Approval"]

    o1 = Fact(Order(id="O1", status="pending"))
    o2 = Fact(Order(id="O2", status="pending"))
    a1 = Fact(Approval(order_id="O1"))
    a2 = Fact(Approval(order_id="O2"))
    for f in (o1, o2, a1, a2):
        engine.add_fact(f)
    engine.run()

    engine.remove_fact(a2)
    engine.run()

    return results


def main() -> None:
    """Run with printed output and embedded assertions."""
    results = _run()
    print(f"Batch status: {results}")
    assert results == ["batch_ready"]


if __name__ == "__main__":
    main()
