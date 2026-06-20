"""End-to-end integration tests for the PRL pipeline (``load_prl`` → engine).

Each test exercises the full stack: PRL source text → tokenize → parse →
compile → RETE network → engine.run().

Fact-binding convention: ``$t`` in the then-block becomes ``t`` — a
``Fact`` wrapper.  Access the underlying object via ``t.obj.field``.
"""
from __future__ import annotations

from dataclasses import fields as dc_fields

from rete.engine import InferenceEngine
from rete.fact import Fact
from rete.prl import load_prl


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _setup(
    prl: str,
    ctx: dict | None = None,
) -> tuple[InferenceEngine, dict]:
    """Compile *prl*, wire productions into a fresh engine, return both.

    :param prl: PRL source text.
    :param ctx: optional namespace seed (types and/or side-effect channels).
    :returns: ``(engine, resolved_types)``.
    """
    engine = InferenceEngine()
    types, prods = load_prl(prl, types=ctx, engine=engine)
    for p in prods:
        engine.add_production(p)
    return engine, types


# ===========================================================================
# Declare
# ===========================================================================

class TestDeclare:
    """``declare`` blocks are compiled into usable Python dataclasses."""

    def test_class_generated(self) -> None:
        _, types = _setup("declare Temp\n  value: float\nend")
        assert "Temp" in types
        assert types["Temp"].__name__ == "Temp"

    def test_java_type_mapping(self) -> None:
        src = "declare Bean\n  name: str\n  count: int\nend"
        _, types = _setup(src)
        flds = dc_fields(types["Bean"])
        assert flds[0].type is str
        assert flds[1].type is int

    def test_instance_creation(self) -> None:
        _, types = _setup("declare Temp\n  value: float\nend")
        obj = types["Temp"](value=42.0)
        assert obj.value == 42.0

    def test_cross_declare_reference(self) -> None:
        src = (
            "declare Inner\n  x: int\nend\n"
            "declare Outer\n  inner: Inner\nend"
        )
        _, types = _setup(src)
        obj = types["Outer"](inner=types["Inner"](x=5))
        assert obj.inner.x == 5


# ===========================================================================
# OOPath patterns
# ===========================================================================

class TestOoPath:
    """``/TypeName[constraints]`` patterns fire and bind correctly."""

    def test_fires_for_matching_value(self) -> None:
        fired: list[bool] = []
        src = (
            "declare Temp\n  value: float\nend\n"
            'rule "r" when\n  /Temp[value >= 80]\nthen\n  fired.append(True)\nend'
        )
        engine, types = _setup(src, {"fired": fired})
        engine.add_fact(Fact(types["Temp"](value=95.0)))
        assert engine.run() == 1
        assert fired == [True]

    def test_does_not_fire_for_non_matching_value(self) -> None:
        src = (
            "declare Temp\n  value: float\nend\n"
            'rule "r" when\n  /Temp[value >= 80]\nthen\n  pass\nend'
        )
        engine, types = _setup(src)
        engine.add_fact(Fact(types["Temp"](value=60.0)))
        assert engine.run() == 0

    def test_no_constraint_matches_any_instance(self) -> None:
        fired: list[bool] = []
        src = (
            "declare Temp\n  value: float\nend\n"
            'rule "r" when\n  /Temp\nthen\n  fired.append(True)\nend'
        )
        engine, types = _setup(src, {"fired": fired})
        engine.add_fact(Fact(types["Temp"](value=10.0)))
        engine.add_fact(Fact(types["Temp"](value=20.0)))
        engine.run()
        assert len(fired) == 2

    def test_fact_binding_exposes_fact_wrapper(self) -> None:
        captured: list = []
        src = (
            "declare Temp\n  value: float\nend\n"
            'rule "r" when\n  $t: /Temp[value >= 80]\n'
            "then\n  captured.append(t)\nend"
        )
        engine, types = _setup(src, {"captured": captured})
        engine.add_fact(Fact(types["Temp"](value=95.0)))
        engine.run()
        assert len(captured) == 1
        assert captured[0].obj.value == 95.0


# ===========================================================================
# Traditional patterns
# ===========================================================================

class TestTraditional:
    """``TypeName(constraints)`` patterns fire correctly."""

    def test_fires_for_matching_value(self) -> None:
        fired: list[bool] = []
        src = (
            "declare Temp\n  value: float\nend\n"
            'rule "r" when\n  Temp(value > 50)\nthen\n  fired.append(True)\nend'
        )
        engine, types = _setup(src, {"fired": fired})
        engine.add_fact(Fact(types["Temp"](value=100.0)))
        assert engine.run() == 1

    def test_does_not_fire_for_non_matching_value(self) -> None:
        src = (
            "declare Temp\n  value: float\nend\n"
            'rule "r" when\n  Temp(value > 50)\nthen\n  pass\nend'
        )
        engine, types = _setup(src)
        engine.add_fact(Fact(types["Temp"](value=10.0)))
        assert engine.run() == 0


# ===========================================================================
# Bindings and joins
# ===========================================================================

class TestBindingsAndJoins:
    """Field bindings and cross-fact join tests."""

    def test_field_binding_value_in_rhs(self) -> None:
        captured: list = []
        src = (
            "declare Temp\n  sensor: str\n  value: float\nend\n"
            'rule "r" when\n  /Temp[value >= 80, $s: sensor]\n'
            "then\n  captured.append(s)\nend"
        )
        engine, types = _setup(src, {"captured": captured})
        engine.add_fact(Fact(types["Temp"](sensor="S1", value=95.0)))
        engine.run()
        assert captured == ["S1"]

    def test_cross_fact_join_fires(self) -> None:
        fired: list = []
        src = (
            "declare Applicant\n  name: str\n  age: int\nend\n"
            "declare Loan\n  applicant: str\nend\n"
            'rule "underage" when\n'
            "  Applicant(age < 21, $name: name)\n"
            "  Loan(applicant == $name)\n"
            "then\n  fired.append(name)\nend"
        )
        engine, types = _setup(src, {"fired": fired})
        engine.add_fact(Fact(types["Applicant"](name="Alice", age=17)))
        engine.add_fact(Fact(types["Loan"](applicant="Alice")))
        engine.add_fact(Fact(types["Applicant"](name="Bob", age=30)))
        engine.add_fact(Fact(types["Loan"](applicant="Bob")))
        engine.run()
        assert fired == ["Alice"]

    def test_cross_fact_join_suppressed_for_mismatch(self) -> None:
        fired: list = []
        src = (
            "declare A\n  key: str\nend\n"
            "declare B\n  key: str\nend\n"
            'rule "r" when\n  A($k: key)\n  B(key == $k)\n'
            "then\n  fired.append(True)\nend"
        )
        engine, types = _setup(src, {"fired": fired})
        engine.add_fact(Fact(types["A"](key="x")))
        engine.add_fact(Fact(types["B"](key="y")))
        engine.run()
        assert fired == []


# ===========================================================================
# Fact binding
# ===========================================================================

class TestFactBinding:
    """``$fact: /Type[…]`` exposes the Fact wrapper in the then-block."""

    def test_fact_wrapper_is_fact_instance(self) -> None:
        captured: list = []
        src = (
            "declare Temp\n  value: float\nend\n"
            'rule "r" when\n  $t: /Temp\n'
            "then\n  captured.append(t)\nend"
        )
        engine, types = _setup(src, {"captured": captured})
        engine.add_fact(Fact(types["Temp"](value=42.0)))
        engine.run()
        assert len(captured) == 1
        assert isinstance(captured[0], Fact)

    def test_mutation_via_obj_attribute(self) -> None:
        src = (
            "declare App\n  approved: bool\nend\n"
            'rule "r" when\n  $app: App(approved == True)\n'
            "then\n  app.obj.approved = False\nend"
        )
        engine, types = _setup(src)
        fact = Fact(types["App"](approved=True))
        engine.add_fact(fact)
        engine.run()
        assert fact.obj.approved is False


# ===========================================================================
# Negation
# ===========================================================================

_NOT_BLUE_PRL = (
    "declare Block\n  name: str\nend\n"
    "declare Color\n  color: str\nend\n"
    'rule "r" when\n  Block()\n  not Color(color == "blue")\n'
    "then\n  fired.append(True)\nend"
)


class TestNegation:
    """``not Pattern`` suppresses the rule when the blocking fact is present."""

    def test_fires_when_negated_fact_absent(self) -> None:
        fired: list = []
        engine, types = _setup(_NOT_BLUE_PRL, {"fired": fired})
        engine.add_fact(Fact(types["Block"](name="B1")))
        engine.add_fact(Fact(types["Color"](color="red")))
        engine.run()
        assert fired == [True]

    def test_suppressed_when_negated_fact_present(self) -> None:
        fired: list = []
        engine, types = _setup(_NOT_BLUE_PRL, {"fired": fired})
        engine.add_fact(Fact(types["Block"](name="B1")))
        engine.add_fact(Fact(types["Color"](color="blue")))
        engine.run()
        assert fired == []


# ===========================================================================
# NCC (negated conjunctive conditions)
# ===========================================================================

_NCC_PRL = (
    "declare Trig\nend\n"
    "declare A\n  v: int\nend\n"
    "declare B\n  v: int\nend\n"
    'rule "r" when\n  Trig()\n  not ( A(v == 1) B(v == 2) )\n'
    "then\n  fired.append(True)\nend"
)


class TestNcc:
    """``not ( p1 p2 )`` fires only when the joint sub-match is absent."""

    def test_fires_when_ncc_unsatisfied(self) -> None:
        fired: list = []
        engine, types = _setup(_NCC_PRL, {"fired": fired})
        engine.add_fact(Fact(types["Trig"]()))
        engine.add_fact(Fact(types["A"](v=1)))   # A present, B absent
        engine.run()
        assert fired == [True]

    def test_suppressed_when_ncc_satisfied(self) -> None:
        fired: list = []
        engine, types = _setup(_NCC_PRL, {"fired": fired})
        engine.add_fact(Fact(types["Trig"]()))
        engine.add_fact(Fact(types["A"](v=1)))
        engine.add_fact(Fact(types["B"](v=2)))
        engine.run()
        assert fired == []


# ===========================================================================
# RHS helpers
# ===========================================================================

class TestRhsHelpers:
    """``insert``, ``retract``, and ``update`` helpers work inside then-blocks."""

    def test_insert_triggers_second_rule(self) -> None:
        captured: list = []
        src = (
            "declare Temp\n  value: float\nend\n"
            "declare Alert\n  msg: str\nend\n"
            'rule "hot" when\n  /Temp[value >= 80]\n'
            'then\n  insert(Alert("HIGH"))\nend\n'
            'rule "alert" when\n  /Alert[msg == "HIGH"]\n'
            "then\n  captured.append(True)\nend"
        )
        engine, types = _setup(src, {"captured": captured})
        engine.add_fact(Fact(types["Temp"](value=95.0)))
        engine.run()
        assert captured == [True]

    def test_retract_clears_conflict_set(self) -> None:
        src = (
            "declare Temp\n  value: float\nend\n"
            'rule "remove" when\n  $t: /Temp[value < 0]\n'
            "then\n  retract(t)\nend"
        )
        engine, types = _setup(src)
        engine.add_fact(Fact(types["Temp"](value=-5.0)))
        assert engine.run() == 1
        assert engine.network.conflict_set == []

    def test_update_triggers_re_evaluation(self) -> None:
        src = (
            "declare App\n  approved: bool\nend\n"
            'rule "deny" when\n  $app: App(approved == True)\n'
            "then\n  app.obj.approved = False\n  update(app)\nend"
        )
        engine, types = _setup(src)
        fact = Fact(types["App"](approved=True))
        engine.add_fact(fact)
        engine.run()
        assert fact.obj.approved is False


# ===========================================================================
# Comments
# ===========================================================================

class TestComments:
    """PRL comments are stripped before parsing."""

    def test_line_comment_stripped(self) -> None:
        src = (
            "// leading comment\n"
            "declare Temp  // inline\n  value: float\nend"
        )
        _, types = _setup(src)
        assert "Temp" in types

    def test_block_comment_stripped(self) -> None:
        src = (
            "/* block\n   comment */\n"
            "declare Temp\n  value: float\nend"
        )
        _, types = _setup(src)
        assert "Temp" in types


# ===========================================================================
# End-to-end: temperature alarm
# ===========================================================================

class TestEndToEnd:
    """Full two-declare, one-rule program exercising the whole pipeline."""

    def test_temperature_alarm(self) -> None:
        alerts: list = []
        src = (
            "declare Temperature\n  sensor: str\n  value: float\nend\n"
            "declare Alert\n  severity: str\n  message: str\nend\n"
            'rule "too-hot"\n'
            "  when\n"
            "    $t: /Temperature[value >= 80]\n"
            "  then\n"
            '    alerts.append(Alert("HIGH", "Sensor " + t.obj.sensor + " too hot"))\n'
            "end"
        )
        engine, types = _setup(src, {"alerts": alerts})
        T = types["Temperature"]
        engine.add_fact(Fact(T(sensor="S1", value=60.0)))
        engine.add_fact(Fact(T(sensor="S2", value=95.0)))
        engine.run()
        assert len(alerts) == 1
        assert "S2" in alerts[0].message


# ===========================================================================
# Inheritance (ES-1)
# ===========================================================================

class TestInheritance:
    """``extends`` in declare — parent patterns fire for child-type facts."""

    def test_parent_pattern_fires_for_child_fact(self) -> None:
        fired: list[str] = []
        src = (
            "declare Animal\n  name: str\nend\n"
            "declare Dog extends Animal\n  breed: str\nend\n"
            'rule "animal" when\n'
            "  $a: Animal()\n"
            "then\n"
            "  results.append(a.obj.name)\n"
            "end\n"
        )
        engine, types = _setup(src, {"results": fired})
        engine.add_fact(Fact(types["Dog"](name="Rex", breed="Lab")))
        engine.run()
        assert "Rex" in fired

    def test_child_pattern_does_not_fire_for_sibling(self) -> None:
        fired: list[str] = []
        src = (
            "declare Animal\n  name: str\nend\n"
            "declare Dog extends Animal\n  breed: str\nend\n"
            "declare Cat extends Animal\n  indoor: bool\nend\n"
            'rule "dog" when\n'
            "  $d: Dog()\n"
            "then\n"
            "  results.append(d.obj.name)\n"
            "end\n"
        )
        engine, types = _setup(src, {"results": fired})
        engine.add_fact(Fact(types["Cat"](name="Whiskers", indoor=True)))
        engine.run()
        assert fired == []

    def test_two_levels_of_inheritance(self) -> None:
        fired: list[str] = []
        src = (
            "declare Animal\n  name: str\nend\n"
            "declare Dog extends Animal\n  breed: str\nend\n"
            "declare Labrador extends Dog\n  colour: str\nend\n"
            'rule "any animal" when\n'
            "  $a: Animal()\n"
            "then\n"
            "  results.append(a.obj.name)\n"
            "end\n"
        )
        engine, types = _setup(src, {"results": fired})
        lab = types["Labrador"](name="Buddy", breed="Lab", colour="gold")
        engine.add_fact(Fact(lab))
        engine.run()
        assert "Buddy" in fired


# ===========================================================================
# No-loop (ES-2)
# ===========================================================================

class TestNoLoop:
    """``@no-loop`` prevents a rule re-activating itself via ``update``."""

    def test_no_loop_tag_fires_once_despite_update(self) -> None:
        results: list = []
        src = (
            "declare Counter\n  value: int\nend\n"
            "@no-loop\n"
            'rule "inc" when\n'
            "  $c: Counter(value < 5)\n"
            "then\n"
            "  c.obj.value += 1\n"
            "  results.append(c.obj.value)\n"
            "  update(c)\n"
            "end"
        )
        engine, types = _setup(src, {"results": results})
        engine.add_fact(Fact(types["Counter"](value=0)))
        fired = engine.run(max_steps=10)
        assert fired == 1
        assert len(results) == 1

    def test_without_no_loop_fires_until_condition_fails(self) -> None:
        """Without ``@no-loop`` the rule fires on every re-inserted match."""
        results: list = []
        src = (
            "declare Counter\n  value: int\nend\n"
            'rule "inc" when\n'
            "  $c: Counter(value < 5)\n"
            "then\n"
            "  c.obj.value += 1\n"
            "  results.append(c.obj.value)\n"
            "  update(c)\n"
            "end"
        )
        engine, types = _setup(src, {"results": results})
        engine.add_fact(Fact(types["Counter"](value=0)))
        engine.run()
        assert len(results) == 5  # fires for 0→1, 1→2, 2→3, 3→4, 4→5

    def test_no_loop_attribute_also_fires_once(self) -> None:
        results: list = []
        src = (
            "declare Counter\n  value: int\nend\n"
            'rule "inc"\n'
            "  no-loop\n"
            "when\n"
            "  $c: Counter(value < 5)\n"
            "then\n"
            "  c.obj.value += 1\n"
            "  results.append(c.obj.value)\n"
            "  update(c)\n"
            "end"
        )
        engine, types = _setup(src, {"results": results})
        engine.add_fact(Fact(types["Counter"](value=0)))
        fired = engine.run(max_steps=10)
        assert fired == 1
        assert len(results) == 1


# ===========================================================================
# @key field equality (ES-3)
# ===========================================================================

class TestKeyField:
    """``@key`` produces key-only equality; WM still tracks facts by identity."""

    def test_objects_with_same_key_are_equal(self) -> None:
        _, types = _setup("declare C\n  @key\n  id: int\n  name: str\nend")
        C = types["C"]
        assert C(id=1, name="Alice") == C(id=1, name="Bob")

    def test_objects_with_different_key_are_unequal(self) -> None:
        _, types = _setup("declare C\n  @key\n  id: int\n  name: str\nend")
        C = types["C"]
        assert C(id=1, name="Alice") != C(id=2, name="Alice")

    def test_separate_facts_retracted_independently(self) -> None:
        """Two @key-equal objects in separate Facts are independently tracked."""
        engine, types = _setup("declare C\n  @key\n  id: int\n  name: str\nend")
        C = types["C"]
        f1 = Fact(C(id=1, name="Alice"))
        f2 = Fact(C(id=1, name="Bob"))
        engine.add_fact(f1)
        engine.add_fact(f2)
        engine.remove_fact(f1)
        assert f1 not in engine.network.conflict_set
        assert f2.obj.name == "Bob"   # f2 still intact

    def test_key_field_in_prl_source(self) -> None:
        """End-to-end: @key parsed from PRL, equality works in caller code."""
        _, types = _setup(
            "declare Customer\n  @key\n  customerId: int\n  tier: str\nend"
        )
        C = types["Customer"]
        assert C(customerId=42, tier="gold") == C(customerId=42, tier="silver")
        assert C(customerId=1, tier="gold") != C(customerId=2, tier="gold")


# ===========================================================================
# Shorthand constraint patterns (ES-4)
# ===========================================================================

class TestShorthandPatterns:
    """Integration: positional and named constraints match facts correctly."""

    _POINT_SRC = "declare Point\n  x: int\n  y: int\nend\n"

    def test_positional_both_match(self) -> None:
        src = self._POINT_SRC + 'rule "r"\nwhen\n  Point(0, 0)\nthen\n  pass\nend'
        _, _ = _setup(src)  # must compile without error

    def test_positional_equivalent_to_compare(self) -> None:
        pos = self._POINT_SRC + 'rule "r"\nwhen\n  Point(0, 0)\nthen\n  pass\nend'
        cmp = self._POINT_SRC + \
              'rule "r"\nwhen\n  Point(x == 0, y == 0)\nthen\n  pass\nend'
        _, _ = _setup(pos)
        _, _ = _setup(cmp)

    def test_named_constraint_compiles(self) -> None:
        src = self._POINT_SRC + 'rule "r"\nwhen\n  Point(y=0)\nthen\n  pass\nend'
        _, _ = _setup(src)

    def test_positional_fact_fires_rule(self) -> None:
        results: list[str] = []
        src = (
            self._POINT_SRC
            + 'rule "origin"\nwhen\n  Point(0, 0)\nthen\n  results.append("hit")\nend'
        )
        engine, types = _setup(src, ctx={"results": results})
        engine.add_fact(Fact(types["Point"](x=0, y=0)))
        engine.run()
        assert results == ["hit"]

    def test_positional_non_matching_fact_does_not_fire(self) -> None:
        results: list[str] = []
        src = (
            self._POINT_SRC
            + 'rule "origin"\nwhen\n  Point(0, 0)\nthen\n  results.append("hit")\nend'
        )
        engine, types = _setup(src, ctx={"results": results})
        engine.add_fact(Fact(types["Point"](x=1, y=0)))
        engine.run()
        assert results == []

    def test_named_fact_fires_rule(self) -> None:
        results: list[str] = []
        src = (
            self._POINT_SRC
            + 'rule "y-axis"\nwhen\n  Point(y=0)\nthen\n  results.append("hit")\nend'
        )
        engine, types = _setup(src, ctx={"results": results})
        engine.add_fact(Fact(types["Point"](x=99, y=0)))
        engine.run()
        assert results == ["hit"]


# ===========================================================================
# Public API
# ===========================================================================

class TestPublicApi:
    """``load_prl`` is re-exported from the top-level ``rete`` package."""

    def test_load_prl_importable_from_rete(self) -> None:
        from rete import load_prl as _lp
        assert callable(_lp)
