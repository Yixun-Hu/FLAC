"""exp_37 INCIDENT regression (2026-11-06 ~20:21-20:52 EST): the synthesis fixture wrote at the REAL pinned roots.

``test_exp37_b2b3_spec.py``'s module fixture called ``registered_world`` without the monkeypatch, so
``World.synthesis()`` wrote its synthetic cells at ``exp37_synthesis.cell_paths`` with the real ``LEAF_ROOT`` /
``EXP35_ROOT`` / ``DUMP_ROOT``: it overwrote the 4 reused exp_35 phase-(i) cells' metrics JSONs and receipts and their
4 NAS dump manifests, and created 16 files in the leaf worktree and 8 dump directories on the NAS. These tests pin the
fix: the fixture refuses (before its first write) any target outside its own root, and ``registered_world`` cannot be
called without the patch. CPU; nothing is written outside ``tmp_path``.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 2.
"""
import inspect

import pytest

from exp37_world import World, patch_synthesis, registered_world
from src.localization import exp37_synthesis as syn


def test_an_unpatched_synthesis_fixture_refuses_before_writing(tmp_path):
    world = World(tmp_path)                                  # NOT patched: the pinned roots are real worktrees / NAS
    with pytest.raises(RuntimeError, match="outside"):
        world.synthesis()
    assert not (tmp_path / "synthesis.json").exists()


def test_registered_world_requires_the_patch():
    assert inspect.signature(registered_world).parameters["monkeypatch"].default is inspect.Parameter.empty


def test_a_patched_fixture_writes_only_under_its_root(tmp_path, monkeypatch):
    world = World(tmp_path)
    patch_synthesis(monkeypatch, world)
    world.synthesis()
    for orbit in ("0", "0_180", "c4"):
        for theta in (0, 90):
            for k in (1, 8):
                assert all(str(p).startswith(str(tmp_path)) for p in syn.cell_paths(orbit, theta, k).values())
