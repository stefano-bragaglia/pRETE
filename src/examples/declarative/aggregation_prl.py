"""ES-9 accumulate example loaded from PRL.

Demonstrates incremental accumulate semantics: the rule fires when the
total exceeds the threshold, and the downstream token is retracted when
a fact is removed and the total drops below the threshold.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "aggregation.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _run() -> tuple[list[float], list[float]]:
    """Return ``(phase1_results, phase2_results)``.

    Phase 1: two orders of 600 each (total 1200 > 1000) → rule fires.
    Phase 2: retract one order (total 600 ≤ 1000) → no new activation.
    """
    results: list = []
    engine, types = _setup({"results": results})
    Order = types["Order"]

    f1 = Fact(Order(customer_id="C1", amount=600.0))
    f2 = Fact(Order(customer_id="C1", amount=600.0))
    engine.add_fact(f1)
    engine.add_fact(f2)
    engine.run()
    phase1 = list(results)
    results.clear()

    engine.remove_fact(f2)
    engine.run()
    phase2 = list(results)

    return phase1, phase2


def main() -> None:
    """Run with printed output and embedded assertions."""
    phase1, phase2 = _run()
    print(f"Phase 1 (total > 1000): {phase1}")
    print(f"Phase 2 (total ≤ 1000): {phase2}")
    assert 1200.0 in phase1
    assert phase2 == []


if __name__ == "__main__":
    main()
