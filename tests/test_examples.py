"""Deterministic assertions for all examples in src/examples/.

Part A tests call each example's main() (smoke + embedded asserts) then
check the captured result list.  Part B tests build an engine inline to
capture side effects without relying on print output.
"""
from rete.fact import Fact


# ---------------------------------------------------------------------------
# Part A — Doorenbos examples (rewritten)
# ---------------------------------------------------------------------------


def test_blocks_world():
    from examples.programmatic.blocks_world import _run, main
    results = _run()
    assert results == [("B1", "B2", "B3")]
    main()  # smoke-test: runs asserts embedded in main()


def test_negation():
    from examples.programmatic.negation import _run, main
    results = _run()
    assert results == ["B3"]
    main()


def test_sharing():
    from examples.programmatic.sharing import _run, main
    results = _run()
    assert set(results) == {("B1", "B2", "B3"), ("B1", "B3", "B4")}
    main()


# ---------------------------------------------------------------------------
# Part B — Loan application
# ---------------------------------------------------------------------------


def test_loan_application():
    from examples.programmatic.loan_application import (
        Applicant,
        Bankruptcy,
        LoanApplication,
        build_engine,
    )
    engine = build_engine()

    alice_loan = Fact(LoanApplication("Alice"))
    bob_loan   = Fact(LoanApplication("Bob"))
    carol_loan = Fact(LoanApplication("Carol"))

    for f in [
        Fact(Applicant("Alice", 17)),
        alice_loan,
        Fact(Applicant("Bob", 35)),
        bob_loan,
        Fact(Applicant("Carol", 30)),
        carol_loan,
        Fact(Bankruptcy("Carol", 2005)),
    ]:
        engine.add_fact(f)

    engine.run()

    assert (
        alice_loan.obj.approved, alice_loan.obj.explanation,
        bob_loan.obj.approved,
        carol_loan.obj.approved, carol_loan.obj.explanation,
    ) == (False, "Underage", True, False, "Bankruptcy")


# ---------------------------------------------------------------------------
# Part B — Temperature alarm
# ---------------------------------------------------------------------------


def test_temperature_alarm():
    from examples.programmatic.temperature_alarm import Temperature, build_engine

    alerts = []
    engine = build_engine(alerts)

    engine.add_fact(Fact(Temperature("T1", 60.0)))
    engine.add_fact(Fact(Temperature("T2", 95.0)))
    engine.run()

    assert len(alerts) == 1
    assert alerts[0].severity == "HIGH"
    assert "T2" in alerts[0].message


# ---------------------------------------------------------------------------
# Part B — Family tree
# ---------------------------------------------------------------------------


def test_family_tree():
    from examples.programmatic.family_tree import Parent, build_engine

    known: set[tuple[str, str]] = set()
    engine = build_engine(known)

    for f in [
        Fact(Parent("A", "B")),
        Fact(Parent("B", "C")),
        Fact(Parent("C", "D")),
    ]:
        engine.add_fact(f)

    engine.run()

    assert known == {
        ("A", "B"), ("A", "C"), ("A", "D"),
        ("B", "C"), ("B", "D"),
        ("C", "D"),
    }


# ---------------------------------------------------------------------------
# Part B — Fraud detection (NCC round-trip)
# ---------------------------------------------------------------------------


def test_fraud_detection():
    from examples.programmatic.fraud_detection import (
        Account,
        Authorization,
        Transaction,
        build_engine,
    )

    fired: list[str] = []
    engine = build_engine(fired)

    tx   = Fact(Transaction("tx1", "acc1", 500.0))
    acct = Fact(Account("acc1", 1000.0))
    engine.add_fact(tx)
    engine.add_fact(acct)
    engine.run()
    assert fired == ["tx1"]          # no auth → NCC count=0 → fired

    auth = Fact(Authorization("tx1"))
    engine.add_fact(auth)
    engine.run()
    assert fired == ["tx1"]          # auth present → NCC count=1 → inst retracted

    engine.remove_fact(auth)
    engine.run()
    assert fired == ["tx1", "tx1"]   # auth gone → NCC count=0 → re-fired


# ---------------------------------------------------------------------------
# Part C — PRL-loaded equivalents
# ---------------------------------------------------------------------------


class TestPrlExamples:
    """PRL drivers produce the same results as the Python originals."""

    def test_blocks_world_prl(self) -> None:
        from examples.declarative.blocks_world_prl import _run
        assert _run() == [("B1", "B2", "B3")]

    def test_negation_prl(self) -> None:
        from examples.declarative.negation_prl import _run
        assert _run() == ["B3"]

    def test_sharing_prl(self) -> None:
        from examples.declarative.sharing_prl import _run
        assert set(_run()) == {("B1", "B2", "B3"), ("B1", "B3", "B4")}

    def test_loan_application_prl(self) -> None:
        from examples.declarative.loan_application_prl import _run
        alice, bob, carol = _run()
        assert (alice.approved, alice.explanation) == (False, "Underage")
        assert bob.approved is True
        assert (carol.approved, carol.explanation) == (False, "Bankruptcy")

    def test_temperature_alarm_prl(self) -> None:
        from examples.declarative.temperature_alarm_prl import _run
        alerts = _run()
        assert len(alerts) == 1
        assert "T2" in alerts[0].message

    def test_family_tree_prl(self) -> None:
        from examples.declarative.family_tree_prl import _run
        assert _run() == {
            ("A", "B"), ("A", "C"), ("A", "D"),
            ("B", "C"), ("B", "D"), ("C", "D"),
        }

    def test_fraud_detection_prl(self) -> None:
        from examples.declarative.fraud_detection_prl import _run
        assert _run() == ["tx1", "tx1"]


# ===========================================================================
# ES-1 through ES-9 examples
# ===========================================================================


class TestExtrasExamples:
    """One test per ES-1 through ES-9 example driver."""

    def test_inheritance_prl(self) -> None:
        from examples.declarative.inheritance_prl import _run
        assert _run() == ["Fluffy", "Rex"]

    def test_identity_key_prl(self) -> None:
        from examples.declarative.identity_key_prl import _run
        eq_result, count = _run()
        assert eq_result is True
        assert count == 1

    def test_compact_patterns_prl(self) -> None:
        from examples.declarative.compact_patterns_prl import _run
        assert _run() == {"origin-pos", "x-axis-named"}

    def test_self_modify_prl(self) -> None:
        from examples.declarative.self_modify_prl import _run
        fire_count, scores = _run()
        assert fire_count == 1
        assert scores == [95]

    def test_imported_types_prl(self) -> None:
        from examples.declarative.imported_types_prl import _run
        assert _run() == [2005]

    def test_disjunction_prl(self) -> None:
        from examples.declarative.disjunction_prl import _run
        r1, r2 = _run()
        assert r1 == ["Alice"]
        assert r2 == ["Bob"]

    def test_universal_prl(self) -> None:
        from examples.declarative.universal_prl import _run
        assert _run() == ["batch_ready"]

    def test_existence_check_prl(self) -> None:
        from examples.declarative.existence_check_prl import _run
        assert _run() == ["A1"]

    def test_event_stream_prl(self) -> None:
        from examples.declarative.event_stream_prl import _run
        pre, present = _run()
        assert pre == 1
        assert present is False

    def test_aggregation_prl(self) -> None:
        from examples.declarative.aggregation_prl import _run
        phase1, phase2 = _run()
        assert 1200.0 in phase1
        assert phase2 == []
