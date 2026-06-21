"""Loan application approval — from Drools reference (Mortgage Rules).

Rules
-----
* *Underage*: Applicant(age < 21) ∧ LoanApplication(applicant == $name, approved)
  → set approved=False, explanation="Underage", update_fact.
* *Bankruptcy history*: LoanApplication(applicant == $name, approved) ∧
  Bankruptcy(name == $name, year > 1990)
  → set approved=False, explanation="Bankruptcy", update_fact.

The ``_approved`` alpha guard on ``LoanApplication`` prevents re-firing
after ``update_fact`` flips the flag to ``False``.
"""
from __future__ import annotations

from dataclasses import dataclass

from rete.condition import JoinSpec, Pattern, Production
from rete.engine import InferenceEngine
from rete.fact import Fact


@dataclass
class Applicant:
    """Loan applicant."""

    name: str
    age: int


@dataclass
class LoanApplication:
    """A loan application, initially approved."""

    applicant: str
    approved: bool = True
    explanation: str = ""


@dataclass
class Bankruptcy:
    """Bankruptcy record for an applicant."""

    name: str
    year: int


def _underage(obj: Applicant) -> bool:
    """Alpha test: applicant is under 21."""
    return obj.age < 21


def _approved(obj: LoanApplication) -> bool:
    """Alpha test: application is still approved (prevents re-fire loop)."""
    return obj.approved


def _recent_bankruptcy(obj: Bankruptcy) -> bool:
    """Alpha test: bankruptcy occurred after 1990."""
    return obj.year > 1990


def build_engine() -> InferenceEngine:
    """Return an :class:`InferenceEngine` loaded with the two loan rules."""
    engine = InferenceEngine()

    def _deny(loan_fact: Fact, reason: str) -> None:
        loan_fact.obj.approved = False
        loan_fact.obj.explanation = reason
        engine.update_fact(loan_fact)

    # Rule 1 — Underage
    engine.add_production(Production(
        lhs=[
            Pattern(Applicant, alpha_tests=(_underage,), bindings=(("$name", "name"),)),
            Pattern(LoanApplication, alpha_tests=(_approved,),
                    join_tests=(JoinSpec("applicant", "$name"),)),
        ],
        rhs=lambda t: _deny(t.facts[1], "Underage"),
    ))

    # Rule 2 — Bankruptcy history
    engine.add_production(Production(
        lhs=[
            Pattern(LoanApplication, alpha_tests=(_approved,),
                    bindings=(("$name", "applicant"),)),
            Pattern(Bankruptcy, alpha_tests=(_recent_bankruptcy,),
                    join_tests=(JoinSpec("name", "$name"),)),
        ],
        rhs=lambda t: _deny(t.facts[0], "Bankruptcy"),
    ))

    return engine


def main() -> None:
    """Run the loan example and print outcomes."""
    engine = build_engine()

    alice_loan = Fact(LoanApplication("Alice"))
    bob_loan   = Fact(LoanApplication("Bob"))
    carol_loan = Fact(LoanApplication("Carol"))

    facts = [
        Fact(Applicant("Alice", 17)),
        alice_loan,
        Fact(Applicant("Bob", 35)),
        bob_loan,
        Fact(Applicant("Carol", 30)),
        carol_loan,
        Fact(Bankruptcy("Carol", 2005)),
    ]

    print("Asserting facts:")
    for f in facts:
        print(f"  {f.obj}")
        engine.add_fact(f)

    engine.run()

    print("\nOutcomes:")
    for loan in (alice_loan, bob_loan, carol_loan):
        obj = loan.obj
        status = "DENIED" if not obj.approved else "APPROVED"
        reason = f" ({obj.explanation})" if obj.explanation else ""
        print(f"  {obj.applicant}: {status}{reason}")


if __name__ == "__main__":
    main()
