"""Pytest configuration for the FLAC test-suite.

Ensures ``src.*`` imports resolve to *this* repository checkout rather than a
stale ``pip install .`` copy that may live in site-packages (a known pitfall on
the development machine). We prepend the repo root (three directory levels up
from this file: ``src/tests/`` -> ``src/`` -> repo root) to the front of
``sys.path`` so it wins over any installed copy.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
sys.path.insert(0, _REPO_ROOT)
