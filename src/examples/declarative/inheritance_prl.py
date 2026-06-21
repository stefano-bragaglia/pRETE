"""ES-1 type inheritance example loaded from PRL.

A single ``Animal()`` pattern fires for both ``Dog`` and ``Cat`` facts.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "inheritance.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _run() -> list[str]:
    """Return sorted list of animal names that matched — used by tests."""
    results: list = []
    engine, types = _setup({"results": results})
    engine.add_fact(Fact(types["Dog"](name="Rex", breed="Labrador")))
    engine.add_fact(Fact(types["Cat"](name="Fluffy", indoor=True)))
    engine.run()
    return sorted(results)


def main() -> None:
    """Run with printed output and embedded assertions."""
    names = _run()
    print(f"Animals greeted: {names}")
    assert names == ["Fluffy", "Rex"]


if __name__ == "__main__":
    main()
