"""ES-6 or-disjunction example loaded from PRL.

The rule fires when either a matching Person or a matching Employee
fact is present — demonstrated with two independent engine runs.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "disjunction.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _run() -> tuple[list[str], list[str]]:
    """Return ``(results_person_branch, results_employee_branch)``."""
    r1: list = []
    engine1, types1 = _setup({"results": r1})
    engine1.add_fact(Fact(types1["Person"](name="Alice", age=25)))
    engine1.run()

    r2: list = []
    engine2, types2 = _setup({"results": r2})
    engine2.add_fact(Fact(types2["Employee"](name="Bob", status="active")))
    engine2.run()

    return r1, r2


def main() -> None:
    """Run with printed output and embedded assertions."""
    r1, r2 = _run()
    print(f"Person branch: {r1}, Employee branch: {r2}")
    assert r1 == ["Alice"]
    assert r2 == ["Bob"]


if __name__ == "__main__":
    main()
