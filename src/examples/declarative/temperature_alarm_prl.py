"""Temperature monitoring loaded from PRL.

Rule: /Temperature[value >= 80] → append Alert to injected list.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "temperature_alarm.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _run() -> list:
    """Return the list of Alert objects raised — used by tests."""
    alerts: list = []
    engine, types = _setup({"alerts": alerts})
    engine.add_fact(Fact(types["Temperature"](sensor="T1", value=60.0)))
    engine.add_fact(Fact(types["Temperature"](sensor="T2", value=95.0)))
    engine.run()
    return alerts


def main() -> None:
    """Run with printed output and embedded assertions."""
    alerts = _run()
    for a in alerts:
        print(f"[{a.severity}] {a.message}")
    assert len(alerts) == 1
    assert "T2" in alerts[0].message


if __name__ == "__main__":
    main()
