"""Repo-root ``tools`` package (exp_05+).

Small operator/diagnostic scripts that live outside ``src`` but import from it.
Each script prepends the repo root to ``sys.path`` before any ``src.*`` import so
it wins over the editable ``rir2rir`` install, whose finder maps ``src`` to a
divergent sibling checkout (see ``src/tests/conftest.py`` for the same guard).
"""
