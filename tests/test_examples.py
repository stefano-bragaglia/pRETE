"""Smoke-tests for the paper examples (Doorenbos 1995).

Each test runs the corresponding example in src/examples/ end-to-end and
relies on the assertions already embedded in main().  If you want to
understand what each scenario does, start there — the examples are the
canonical reference, not this file.
"""
from examples.blocks_world import main as blocks_world
from examples.negation import main as negation
from examples.sharing import main as sharing


def test_blocks_world():
    blocks_world()


def test_sharing():
    sharing()


def test_negation():
    negation()
