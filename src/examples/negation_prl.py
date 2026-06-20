"""Doorenbos §2.7 — negation example loaded from PRL.

Expected match: x=B3 (on table, not blue).
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "negation.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _make_facts(types: dict) -> list[Fact]:
    """Working-memory facts (Doorenbos §2.7 arrangement)."""
    On, Color = types["On"], types["Color"]
    return [
        Fact(On(upper="B1", lower="B2")),
        Fact(On(upper="B1", lower="B3")),
        Fact(Color(block="B1", color="red")),
        Fact(On(upper="B2", lower="table")),
        Fact(Color(block="B2", color="blue")),
        Fact(Color(block="B3", color="red")),
        Fact(On(upper="B3", lower="table")),
    ]


def _run() -> list[str]:
    """Return block names matching the rule — used by tests."""
    results: list = []
    engine, types = _setup({"results": results})
    for f in _make_facts(types):
        engine.add_fact(f)
    engine.run()
    return results


def main() -> None:
    """Run with printed output and embedded assertions."""
    results = _run()
    for x in results:
        print(f"fired: x={x} is on table and not blue")
    assert results == ["B3"]


if __name__ == "__main__":
    main()
