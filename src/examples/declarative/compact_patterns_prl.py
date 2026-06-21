"""ES-4 positional/named constraint shorthand loaded from PRL.

Point(0, 0) positional form and Point(y=0) named form both fire
for the origin point.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "compact_patterns.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _run() -> set[str]:
    """Return the set of rule hits — used by tests."""
    hits: list = []
    engine, types = _setup({"hits": hits})
    engine.add_fact(Fact(types["Point"](x=0, y=0)))
    engine.run()
    return set(hits)


def main() -> None:
    """Run with printed output and embedded assertions."""
    hits = _run()
    print(f"Rules fired: {hits}")
    assert hits == {"origin-pos", "x-axis-named"}


if __name__ == "__main__":
    main()
