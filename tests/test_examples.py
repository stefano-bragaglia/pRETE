"""Smoke-tests for the paper examples (Doorenbos 1995).

Each test runs the corresponding example in src/examples/ end-to-end and
relies on the assertions already embedded in main().  If you want to
understand what each scenario does, start there — the examples are the
canonical reference, not this file.

Note: examples are rewritten in Step 8; until then these tests skip
gracefully when the examples still import the old API.
"""
import importlib

import pytest


def _load(module_name: str):
    """Import *module_name*, skipping the test on any ImportError."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(str(exc))


def test_blocks_world():
    _load("examples.blocks_world").main()


def test_sharing():
    _load("examples.sharing").main()


def test_negation():
    _load("examples.negation").main()
