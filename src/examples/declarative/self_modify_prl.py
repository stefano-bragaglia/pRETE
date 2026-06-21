"""ES-2 @no-loop tag example loaded from PRL.

The rule inserts Score(95) which also matches ``value > 90``.
Without @no-loop this would loop; with it the rule fires exactly once.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "self_modify.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _run() -> tuple[int, list[int]]:
    """Return ``(fire_count, sorted_scores_in_wm)`` — used by tests."""
    counter = [0]
    engine, types = _setup({"counter": counter})
    Score = types["Score"]
    engine.add_fact(Fact(Score(value=150)))
    engine.run()
    remaining = sorted(
        f.obj.value
        for f in engine.network.root._facts
        if isinstance(f.obj, Score)
    )
    return counter[0], remaining


def main() -> None:
    """Run with printed output and embedded assertions."""
    fire_count, scores = _run()
    print(f"Rule fired {fire_count} time(s); WM scores: {scores}")
    assert fire_count == 1
    assert scores == [95]


if __name__ == "__main__":
    main()
