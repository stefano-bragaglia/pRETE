"""Loan application approval loaded from PRL.

Rules: underage (age < 21) and bankruptcy history (year > 1990).
PRL-compiled LoanApplication has no default values; all fields required.
"""
from __future__ import annotations

from pathlib import Path

from rete import Fact, InferenceEngine, load_prl

_PRL = Path(__file__).parent / "prl" / "loan_application.prl"


def _setup(ctx: dict | None = None) -> tuple[InferenceEngine, dict]:
    engine = InferenceEngine()
    types, prods = load_prl(_PRL.read_text(), types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


def _make_facts(
    types: dict,
    alice_loan: Fact,
    bob_loan: Fact,
    carol_loan: Fact,
) -> list[Fact]:
    """Return the seven WM facts for the loan scenario."""
    Applicant = types["Applicant"]
    Bankruptcy = types["Bankruptcy"]
    return [
        Fact(Applicant(name="Alice", age=17)), alice_loan,
        Fact(Applicant(name="Bob",   age=35)), bob_loan,
        Fact(Applicant(name="Carol", age=30)), carol_loan,
        Fact(Bankruptcy(name="Carol", year=2005)),
    ]


def _run() -> tuple:
    """Return ``(alice.obj, bob.obj, carol.obj)`` after rule firing."""
    engine, types = _setup()
    Loan = types["LoanApplication"]
    alice_loan = Fact(Loan(applicant="Alice", approved=True, explanation=""))
    bob_loan   = Fact(Loan(applicant="Bob",   approved=True, explanation=""))
    carol_loan = Fact(Loan(applicant="Carol", approved=True, explanation=""))
    for f in _make_facts(types, alice_loan, bob_loan, carol_loan):
        engine.add_fact(f)
    engine.run()
    return alice_loan.obj, bob_loan.obj, carol_loan.obj


def _print_outcome(obj) -> None:
    status = "DENIED" if not obj.approved else "APPROVED"
    reason = f" ({obj.explanation})" if obj.explanation else ""
    print(f"  {obj.applicant}: {status}{reason}")


def main() -> None:
    """Run with printed output and embedded assertions."""
    alice, bob, carol = _run()
    for obj in (alice, bob, carol):
        _print_outcome(obj)
    assert (alice.approved, alice.explanation) == (False, "Underage")
    assert bob.approved is True
    assert (carol.approved, carol.explanation) == (False, "Bankruptcy")


if __name__ == "__main__":
    main()
