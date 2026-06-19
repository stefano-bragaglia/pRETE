"""Fraud detection — unauthorized transactions (NCC group).

Rule
----
* *Unauthorized*: Transaction(account_id=$account_id) ∧
  Account(id==$account_id) ∧ NOT Authorization(transaction_id==$tx_id)
  → record the transaction id as flagged.

The NOT clause is an :class:`NccGroup`, exercising the full
``NccNode`` / ``NccPartnerNode`` path.  The round-trip test
(add Authorization → no re-fire; remove it → re-fire) verifies that
``NccPartnerNode.left_retract`` and ``left_activate`` work correctly.
"""
from __future__ import annotations

from dataclasses import dataclass

from rete.condition import JoinSpec, NccGroup, Pattern, Production
from rete.engine import InferenceEngine
from rete.fact import Fact


@dataclass
class Transaction:
    """A financial transaction."""

    id: str
    account_id: str
    amount: float


@dataclass
class Account:
    """A bank account with a credit limit."""

    id: str
    limit: float


@dataclass
class Authorization:
    """Explicit authorization for a transaction."""

    transaction_id: str


@dataclass
class Flag:
    """A fraud flag raised by the engine."""

    transaction_id: str
    reason: str


def build_engine(fired: list[str]) -> InferenceEngine:
    """Return an engine with the unauthorized-transaction rule.

    :param fired: mutable list; RHS appends the transaction id on each firing.
    """
    engine = InferenceEngine()

    engine.add_production(Production(
        lhs=[
            Pattern(Transaction, bindings=(
                ("$tx_id", "id"), ("$account_id", "account_id"),
            )),
            Pattern(Account, join_tests=(JoinSpec("id", "$account_id"),)),
            NccGroup(conditions=(
                Pattern(Authorization,
                        join_tests=(JoinSpec("transaction_id", "$tx_id"),)),
            )),
        ],
        rhs=lambda t: fired.append(t.bindings["$tx_id"]),
    ))

    return engine


def main() -> None:
    """Run the fraud-detection round-trip example."""
    fired: list[str] = []
    engine = build_engine(fired)

    tx   = Fact(Transaction("tx1", "acc1", 500.0))
    acct = Fact(Account("acc1", 1000.0))

    print("Step 1 — add transaction and account (no authorization):")
    engine.add_fact(tx)
    engine.add_fact(acct)
    engine.run()
    print(f"  fired: {fired}")
    assert fired == ["tx1"]

    print("\nStep 2 — add authorization:")
    auth = Fact(Authorization("tx1"))
    engine.add_fact(auth)
    engine.run()
    print(f"  fired: {fired}")
    assert fired == ["tx1"]  # no re-fire; NCC count=1

    print("\nStep 3 — remove authorization:")
    engine.remove_fact(auth)
    engine.run()
    print(f"  fired: {fired}")
    assert fired == ["tx1", "tx1"]  # re-fired; NCC count=0


if __name__ == "__main__":
    main()
