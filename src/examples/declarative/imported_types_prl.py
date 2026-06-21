"""ES-5 import example loaded from PRL.

The .prl file declares its own ``from examples.domain import ...`` so
``load_prl`` needs no pre-built types dict.  Run with src/ on sys.path
(guaranteed by hatch / pytest via the src-layout in pyproject.toml).
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "imported_types.prl"


def _run() -> list[int]:
    """Return sorted list of flagged vehicle years — used by tests."""
    flagged: list = []
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types={"flagged": flagged}, engine=engine)
    for p in prods:
        engine.add_production(p)
    Vehicle = types["Vehicle"]
    Fleet = types["Fleet"]
    engine.add_fact(Fact(Fleet(owner="Acme", size=15)))
    engine.add_fact(Fact(Vehicle(make="Ford", model="F-150", year=2005)))
    engine.add_fact(Fact(Vehicle(make="Toyota", model="Camry", year=2020)))
    engine.run()
    return sorted(flagged)


def main() -> None:
    """Run with printed output and embedded assertions."""
    years = _run()
    print(f"Flagged vehicle years: {years}")
    assert years == [2005]


if __name__ == "__main__":
    main()
