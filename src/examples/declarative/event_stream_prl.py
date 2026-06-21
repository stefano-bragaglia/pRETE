"""ES-8 CEP event expiry example loaded from PRL.

A StockTick with @expires(30s) is automatically retracted when the
logical clock advances past its timestamp + 30 s.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "event_stream.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _run() -> tuple[int, bool]:
    """Return ``(pre_expire_alert_count, tick_still_in_wm_after_expiry)``.

    Uses a deterministic timestamp of 0.0 and advances to 31.0, which is
    past the 30 s expiry window.
    """
    alerts: list = []
    engine, types = _setup({"alerts": alerts})
    StockTick = types["StockTick"]

    tick = Fact(StockTick(ts=0.0, symbol="ACME", price=150.0))
    engine.add_fact(tick)
    engine.run()
    pre_count = len(alerts)

    engine.advance_clock(31.0)
    engine.run()
    post_present = tick in engine.network.root._facts

    return pre_count, post_present


def main() -> None:
    """Run with printed output and embedded assertions."""
    pre, present = _run()
    print(f"Alerts before expiry: {pre}, tick in WM after: {present}")
    assert pre == 1
    assert present is False


if __name__ == "__main__":
    main()
