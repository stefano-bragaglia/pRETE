"""Temperature monitoring — from Drools reference (MonitoringService).

Rule
----
* *Too hot*: Temperature(value >= 80) → add Alert("HIGH", message).

The RHS inserts a new Fact into the engine from within the firing cycle,
demonstrating that ``engine.add_fact`` called during ``run()`` produces
a new conflict-set entry visible in the next iteration.
"""
from __future__ import annotations

from dataclasses import dataclass

from rete.condition import Pattern, Production
from rete.engine import InferenceEngine
from rete.fact import Fact


@dataclass
class Temperature:
    """A temperature reading from a named sensor."""

    sensor: str
    value: float


@dataclass
class Alert:
    """An alarm triggered by a rule."""

    severity: str
    message: str


def _too_hot(obj: Temperature) -> bool:
    """Alpha test: temperature is at or above the alarm threshold."""
    return obj.value >= 80.0


def build_engine(alerts: list[Alert] | None = None) -> InferenceEngine:
    """Return an engine with the too-hot alarm rule.

    :param alerts: optional list to collect fired :class:`Alert` objects
                   (used by tests for side-effect capture without add_fact).
    """
    engine = InferenceEngine()

    def _rhs(token) -> None:
        sensor = token.bindings["$sensor"]
        value  = token.bindings["$value"]
        alert  = Alert("HIGH", f"Sensor {sensor}: {value}°C")
        if alerts is not None:
            alerts.append(alert)
        else:
            engine.add_fact(Fact(alert))

    engine.add_production(Production(
        lhs=[Pattern(
            Temperature,
            alpha_tests=(_too_hot,),
            bindings=(("$sensor", "sensor"), ("$value", "value")),
        )],
        rhs=_rhs,
    ))

    return engine


def main() -> None:
    """Run the temperature alarm example."""
    captured: list[Alert] = []
    engine = build_engine(captured)

    readings = [
        Fact(Temperature("T1", 60.0)),
        Fact(Temperature("T2", 95.0)),
    ]

    print("Adding temperature readings:")
    for f in readings:
        print(f"  {f.obj}")
        engine.add_fact(f)

    engine.run()

    print("\nAlerts raised:")
    for a in captured:
        print(f"  [{a.severity}] {a.message}")


if __name__ == "__main__":
    main()
