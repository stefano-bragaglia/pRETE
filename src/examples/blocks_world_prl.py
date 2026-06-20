"""Doorenbos §2.1 — blocks-world example loaded from PRL.

Expected single match: x=B1, y=B2, z=B3.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "blocks_world.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _make_facts(types: dict) -> list[Fact]:
    """Working-memory facts w1–w9 (Doorenbos §2.1)."""
    On, LeftOf, Color = types["On"], types["LeftOf"], types["Color"]
    return [
        Fact(On(upper="B1", lower="B2")),
        Fact(On(upper="B1", lower="B3")),
        Fact(Color(block="B1", color="red")),
        Fact(On(upper="B2", lower="table")),
        Fact(LeftOf(left="B2", right="B3")),
        Fact(Color(block="B2", color="blue")),
        Fact(LeftOf(left="B3", right="B4")),
        Fact(On(upper="B3", lower="table")),
        Fact(Color(block="B3", color="red")),
    ]


def _run() -> list[tuple[str, str, str]]:
    """Return ``[(x, y, z)]`` for each match — used by tests."""
    results: list = []
    engine, types = _setup({"results": results})
    for f in _make_facts(types):
        engine.add_fact(f)
    engine.run()
    return results


def main() -> None:
    """Run with printed output and embedded assertions."""
    results = _run()
    for x, y, z in results:
        print(f"fired: x={x}, y={y}, z={z}")
    assert results == [("B1", "B2", "B3")]


if __name__ == "__main__":
    main()
