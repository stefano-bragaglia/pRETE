"""Shared domain classes imported by the imported_types example."""
from dataclasses import dataclass


@dataclass
class Vehicle:
    make: str
    model: str
    year: int


@dataclass
class Fleet:
    owner: str
    size: int
