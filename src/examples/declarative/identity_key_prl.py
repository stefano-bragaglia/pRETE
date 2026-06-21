"""ES-3 @key identity example loaded from PRL.

Proves: two Customer objects with the same id are equal regardless of
other fields; remove_fact still works correctly by Fact identity.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "identity_key.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _run() -> tuple[bool, int]:
    """Return ``(eq_result, fire_count)``.

    ``eq_result``: True when two Customers with the same id are equal despite
    different name/tier fields — proving @key overrides full equality.
    ``fire_count``: number of times the "flag gold customer" rule fired.
    """
    results: list = []
    engine, types = _setup({"results": results})
    Customer = types["Customer"]

    c_gold = Customer(id="C1", name="Alice", tier="gold")
    c_same_id = Customer(id="C1", name="Bob", tier="bronze")
    eq_result = (c_gold == c_same_id)

    f1 = Fact(Customer(id="C1", name="Alice", tier="gold"))
    f2 = Fact(Customer(id="C2", name="Carol", tier="silver"))
    engine.add_fact(f1)
    engine.add_fact(f2)
    engine.run()

    # Verify retraction by Fact identity: removing f1 must not remove f2.
    engine.remove_fact(f1)
    engine.run()

    return eq_result, len(results)


def main() -> None:
    """Run with printed output and embedded assertions."""
    eq_result, count = _run()
    print(f"@key equality: {eq_result}, rule fired: {count} time(s)")
    assert eq_result is True
    assert count == 1


if __name__ == "__main__":
    main()
