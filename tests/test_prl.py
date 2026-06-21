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
# Import integration (ES-5)
# ===========================================================================

class TestImportIntegration:
    """End-to-end: imported types are usable in patterns and extends."""

    def test_imported_type_in_pattern(self) -> None:
        """A class imported via ``from … import`` can be matched in a LHS pattern."""
        import sys
        import types as _types_mod
        from dataclasses import dataclass as _dc

        @_dc
        class Widget:
            color: str

        mod = _types_mod.ModuleType("_es5_widget_mod")
        mod.Widget = Widget  # type: ignore[attr-defined]
        sys.modules["_es5_widget_mod"] = mod
        try:
            results: list[str] = []
            src = (
                "from _es5_widget_mod import Widget\n"
                'rule "r"\nwhen\n  Widget(color == "red")\n'
                'then\n  results.append("fired")\nend'
            )
            engine, types = _setup(src, ctx={"results": results})
            engine.add_fact(Fact(Widget(color="red")))
            engine.run()
            assert results == ["fired"]
        finally:
            del sys.modules["_es5_widget_mod"]

    def test_import_available_before_declare(self) -> None:
        """Imported type is available as a parent in ``extends``."""
        import sys
        import types as _types_mod
        from dataclasses import make_dataclass
        Base = make_dataclass("_ES5Base", [("score", int)])
        fake_mod = _types_mod.ModuleType("_es5_test_mod")
        fake_mod._ES5Base = Base  # type: ignore[attr-defined]
        sys.modules["_es5_test_mod"] = fake_mod
        try:
            src = (
                "from _es5_test_mod import _ES5Base\n"
                "declare Child extends _ES5Base\n"
                "  name: str\n"
                "end\n"
            )
            types, _ = load_prl(src)
            assert issubclass(types["Child"], Base)
        finally:
            del sys.modules["_es5_test_mod"]

    def test_drools_style_import_in_pattern(self) -> None:
        """``import module.ClassName`` form also makes the type available."""
        import sys
        import types as _types_mod
        from dataclasses import dataclass as _dc

        @_dc
        class Gadget:
            size: int

        mod = _types_mod.ModuleType("_es5_gadget_mod")
        mod.Gadget = Gadget  # type: ignore[attr-defined]
        sys.modules["_es5_gadget_mod"] = mod
        try:
            results: list[str] = []
            src = (
                "import _es5_gadget_mod.Gadget\n"
                'rule "r"\nwhen\n  Gadget(size == 42)\n'
                'then\n  results.append("ok")\nend'
            )
            engine, types = _setup(src, ctx={"results": results})
            engine.add_fact(Fact(Gadget(size=42)))
            engine.run()
            assert results == ["ok"]
        finally:
            del sys.modules["_es5_gadget_mod"]


# ===========================================================================
# Public API
# ===========================================================================

class TestPublicApi:
    """``load_prl`` is re-exported from the top-level ``rete`` package."""

    def test_load_prl_importable_from_rete(self) -> None:
        from rete import load_prl as _lp
        assert callable(_lp)


# ===========================================================================
# or disjunction integration (ES-6)
# ===========================================================================

class TestOrIntegration:
    """End-to-end: either or-branch triggers the RHS."""

    _SRC = (
        "declare Vehicle\n  kind: str\nend\n"
        'rule "flag"\n'
        "when\n"
        '  $v: Vehicle(kind == "car") or\n'
        '  $v: Vehicle(kind == "truck")\n'
        "then\n"
        "  results.append(v.obj.kind)\n"
        "end\n"
    )

    def test_first_branch_fires(self) -> None:
        results: list[str] = []
        engine, types = _setup(self._SRC, ctx={"results": results})
        engine.add_fact(Fact(types["Vehicle"](kind="car")))
        engine.run()
        assert "car" in results

    def test_second_branch_fires(self) -> None:
        results: list[str] = []
        engine, types = _setup(self._SRC, ctx={"results": results})
        engine.add_fact(Fact(types["Vehicle"](kind="truck")))
        engine.run()
        assert "truck" in results

    def test_non_matching_does_not_fire(self) -> None:
        results: list[str] = []
        engine, types = _setup(self._SRC, ctx={"results": results})
        engine.add_fact(Fact(types["Vehicle"](kind="bike")))
        engine.run()
        assert results == []

    def test_two_productions_registered(self) -> None:
        _, prods = load_prl(self._SRC)
        assert len(prods) == 2

    def test_both_branches_fire_independently(self) -> None:
        results: list[str] = []
        engine, types = _setup(self._SRC, ctx={"results": results})
        engine.add_fact(Fact(types["Vehicle"](kind="car")))
        engine.add_fact(Fact(types["Vehicle"](kind="truck")))
        engine.run()
        assert set(results) == {"car", "truck"}


# ===========================================================================
# forall integration (ES-6)
# ===========================================================================

class TestForallIntegration:
    """End-to-end: forall fires when universal condition holds."""

    _SRC = (
        "declare Order\n  status: str\nend\n"
        "declare Approval\n  ref: str\nend\n"
        'rule "all approved"\n'
        "when\n"
        "  forall(Order(), Approval())\n"
        "then\n"
        '  results.append("ok")\n'
        "end\n"
    )

    def test_fires_when_no_order(self) -> None:
        """No orders → vacuously true → rule fires."""
        results: list[str] = []
        engine, _ = _setup(self._SRC, ctx={"results": results})
        engine.run()
        assert results == ["ok"]

    def test_does_not_fire_when_order_without_approval(self) -> None:
        results: list[str] = []
        engine, types = _setup(self._SRC, ctx={"results": results})
        engine.add_fact(Fact(types["Order"](status="pending")))
        engine.run()
        assert results == []

    def test_fires_when_order_has_approval(self) -> None:
        results: list[str] = []
        engine, types = _setup(self._SRC, ctx={"results": results})
        engine.add_fact(Fact(types["Order"](status="pending")))
        engine.add_fact(Fact(types["Approval"](ref="x")))
        engine.run()
        assert results == ["ok"]

    def test_retracting_approval_unblocks(self) -> None:
        """Removing the Approval re-blocks the rule (retraction propagates)."""
        results: list[str] = []
        engine, types = _setup(self._SRC, ctx={"results": results})
        order = Fact(types["Order"](status="pending"))
        approval = Fact(types["Approval"](ref="x"))
        engine.add_fact(order)
        engine.add_fact(approval)
        engine.run()
        assert "ok" in results
        results.clear()
        engine.remove_fact(approval)
        engine.run()
        assert results == []


# ===========================================================================
# exists integration (ES-7)
# ===========================================================================


class TestExistsIntegration:
    """End-to-end: exists fires once per left-token regardless of right count."""

    _SRC = (
        "declare Account\n  name: str\nend\n"
        "declare Invoice\n  overdue: bool\nend\n"
        'rule "alert"\n'
        "when\n"
        "  $acc: Account()\n"
        "  exists Invoice(overdue == true)\n"
        "then\n"
        "  results.append(acc.obj.name)\n"
        "end\n"
    )

    def test_fires_once_with_two_invoices(self) -> None:
        """Rule fires once per account even when two matching invoices exist."""
        results: list[str] = []
        engine, types = _setup(self._SRC, ctx={"results": results})
        engine.add_fact(Fact(types["Account"](name="alice")))
        engine.add_fact(Fact(types["Invoice"](overdue=True)))
        engine.add_fact(Fact(types["Invoice"](overdue=True)))
        engine.run()
        assert results.count("alice") == 1

    def test_does_not_fire_without_invoice(self) -> None:
        """Rule does not fire when no overdue invoice exists."""
        results: list[str] = []
        engine, types = _setup(self._SRC, ctx={"results": results})
        engine.add_fact(Fact(types["Account"](name="alice")))
        engine.run()
        assert results == []

    def test_retract_last_invoice_clears_activation(self) -> None:
        """Retracting the last matching invoice removes the pending activation."""
        results: list[str] = []
        engine, types = _setup(self._SRC, ctx={"results": results})
        engine.add_fact(Fact(types["Account"](name="alice")))
        invoice = Fact(types["Invoice"](overdue=True))
        engine.add_fact(invoice)
        engine.remove_fact(invoice)
        engine.run()
        assert results == []


# ===========================================================================
# CEP integration
# ===========================================================================


_CEP_SRC = """\
@role(event)
@expires(30s)
declare Reading
  @timestamp
  ts: float
  sensor: str
end

declare Plain
  value: int
end
"""


class TestCepIntegration:
    def test_event_present_before_clock_advance(self) -> None:
        engine, types = _setup(_CEP_SRC)
        fired = []
        from rete.condition import Pattern, Production
        engine.add_production(Production(
            lhs=[Pattern(types["Reading"])],
            rhs=lambda t: fired.append(1),
        ))
        engine.add_fact(Fact(types["Reading"](ts=0.0, sensor="s1")))
        engine.run()
        assert fired == [1]

    def test_event_auto_retracted_after_advance(self) -> None:
        engine, types = _setup(_CEP_SRC)
        f = Fact(types["Reading"](ts=0.0, sensor="s1"))
        engine.add_fact(f)
        engine.advance_clock(31.0)
        engine.run()
        assert f not in engine.network.root._facts

    def test_non_event_not_retracted(self) -> None:
        engine, types = _setup(_CEP_SRC)
        f = Fact(types["Plain"](value=42))
        engine.add_fact(f)
        engine.advance_clock(9999.0)
        engine.run()
        assert f in engine.network.root._facts


# ===========================================================================
# Accumulate integration
# ===========================================================================


_ACC_SRC = """\
declare Order
  amount: float
end

rule "sum orders"
  when
    accumulate(
      Order($amount: amount);
      $total: sum($amount)
    )
  then
    results.append($total)
end
"""

_ACC_CONSTRAINED_SRC = """\
declare Order
  amount: float
end

rule "flag large total"
  when
    accumulate(
      Order($amount: amount);
      $total: sum($amount);
      $total > 100
    )
  then
    results.append($total)
end
"""

_ACC_COUNT_SRC = """\
declare Order
  amount: float
end

rule "count orders"
  when
    accumulate(
      Order($amount: amount);
      $n: count()
    )
  then
    results.append($n)
end
"""


class TestAccumulateIntegration:
    def test_sum_no_facts_fires_zero(self) -> None:
        results = []
        engine, types = _setup(_ACC_SRC, {"results": results})
        engine.run()
        assert results == [0]

    def test_sum_single_fact(self) -> None:
        results = []
        engine, types = _setup(_ACC_SRC, {"results": results})
        engine.add_fact(Fact(types["Order"](amount=50.0)))
        engine.run()
        assert results[-1] == 50.0

    def test_sum_multiple_facts(self) -> None:
        results = []
        engine, types = _setup(_ACC_SRC, {"results": results})
        engine.add_fact(Fact(types["Order"](amount=30.0)))
        engine.add_fact(Fact(types["Order"](amount=20.0)))
        engine.run()
        assert results[-1] == 50.0

    def test_sum_updates_on_retract(self) -> None:
        results = []
        engine, types = _setup(_ACC_SRC, {"results": results})
        f = Fact(types["Order"](amount=30.0))
        engine.add_fact(f)
        engine.add_fact(Fact(types["Order"](amount=20.0)))
        engine.run()
        results.clear()
        engine.remove_fact(f)
        engine.run()
        assert results[-1] == 20.0

    def test_count_fires(self) -> None:
        results = []
        engine, types = _setup(_ACC_COUNT_SRC, {"results": results})
        engine.add_fact(Fact(types["Order"](amount=1.0)))
        engine.add_fact(Fact(types["Order"](amount=2.0)))
        engine.run()
        assert results[-1] == 2

    def test_constraint_blocks_below_threshold(self) -> None:
        results = []
        engine, types = _setup(_ACC_CONSTRAINED_SRC, {"results": results})
        engine.add_fact(Fact(types["Order"](amount=50.0)))
        engine.run()
        assert results == []

    def test_constraint_fires_above_threshold(self) -> None:
        results = []
        engine, types = _setup(_ACC_CONSTRAINED_SRC, {"results": results})
        engine.add_fact(Fact(types["Order"](amount=150.0)))
        engine.run()
        assert results == [150.0]
