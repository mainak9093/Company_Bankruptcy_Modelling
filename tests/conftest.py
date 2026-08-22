"""Pytest configuration for the test suite.

Puts ``src/`` on the import path once, here, instead of in every test module.

It also disables bytecode caching for the test session. The modules in ``src/``
are imported as top-level modules from several entry points (``train.py``,
``cross_validate.py``, ``make_report_figures.py``, the tests). When a training
or figure run is in progress while the tests are collected, both processes race
to write ``src/__pycache__``, and an import can fail against a half-written
``.pyc`` -- which surfaces as a collection error rather than a test failure.
Tests gain nothing from cached bytecode, so the race is simply removed.
"""

import os
import sys

SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

sys.dont_write_bytecode = True
