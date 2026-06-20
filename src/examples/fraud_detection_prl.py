"""Fraud detection loaded from PRL — NCC round-trip.

Rule: Transaction ∧ Account ∧ NOT Authorization → append tx_id.
Three-phase test: no auth → fires; auth present → no re-fire; auth
removed → re-fires.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "fraud_detection.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _run() -> list[str]:
    """Execute the three-phase round-trip; return the fired list."""
    fired: list[str] = []
    engine, types = _setup({"fired": fired})

    tx   = Fact(types["Transaction"](id="tx1", account_id="acc1", amount=500.0))
    acct = Fact(types["Account"](id="acc1", limit=1000.0))
    engine.add_fact(tx)
    engine.add_fact(acct)
    engine.run()                    # phase 1: no auth → fired=["tx1"]

    auth = Fact(types["Authorization"](transaction_id="tx1"))
    engine.add_fact(auth)
    engine.run()                    # phase 2: auth present → no re-fire

    engine.remove_fact(auth)
    engine.run()                    # phase 3: auth removed → fired=["tx1","tx1"]
    return fired


def main() -> None:
    """Run with printed output and embedded assertions."""
    fired = _run()
    print(f"fired sequence: {fired}")
    assert fired == ["tx1", "tx1"]


if __name__ == "__main__":
    main()
