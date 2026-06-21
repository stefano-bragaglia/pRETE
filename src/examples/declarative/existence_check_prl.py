"""ES-7 exists quantification example loaded from PRL.

Rule fires once per Account with at least one overdue Invoice —
not once per Invoice, proving ExistsNode semantics.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "existence_check.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _run() -> list[str]:
    """Return the alerts list — used by tests.

    Three overdue Invoices for one Account must produce exactly one alert.
    """
    alerts: list = []
    engine, types = _setup({"alerts": alerts})
    Account = types["Account"]
    Invoice = types["Invoice"]

    engine.add_fact(Fact(Account(id="A1")))
    for _ in range(3):
        engine.add_fact(Fact(Invoice(account_id="A1", overdue=True)))
    engine.run()
    return alerts


def main() -> None:
    """Run with printed output and embedded assertions."""
    alerts = _run()
    print(f"Overdue alerts: {alerts}")
    assert alerts == ["A1"]


if __name__ == "__main__":
    main()
