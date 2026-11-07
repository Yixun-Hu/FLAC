"""exp_37 fix round 2 (Codex code review 2, BLOCKING 5): ``tools/exp37_build_synthesis_cells.py`` builds the complete,
validated synthesis section from the artifacts at the pinned layout (the 8 new cells in the leaf's run worktree, the 4
reused C4 cells in the exp_35 run worktree, their dump manifests under the exp25 dump root), refuses missing or
misattributed artifacts (exit 2, nothing written) and writes once. CPU; the world's stand-in roots.
Written by the exp_37 Coder (Claude Opus 5.5, max effort), fix round 2.
"""
import json
from pathlib import Path

import pytest

from exp37_world import World, patch_synthesis
from src.localization import exp37_synthesis as syn
from tools import exp37_build_synthesis_cells as cells_tool


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path)
    patch_synthesis(monkeypatch, w)
    w.synthesis()
    monkeypatch.setattr(syn, "REUSED", w.reused)
    return w


def test_the_section_is_built_from_the_artifacts_and_validated(world, tmp_path):
    out = tmp_path / "out" / "synthesis_cells.json"
    assert cells_tool.main(["--output", str(out)]) == 0
    section = json.loads(out.read_text())
    assert section["status"] == "complete" and section["leaf_commit"] == syn.LEAF_COMMIT
    assert section["metrics_keys"] == list(cells_tool.METRICS) and len(section["cells"]) == 12
    assert "EXPECT_HEAD=" + syn.LEAF_COMMIT in section["runbook"] and syn.LEAF_ROOT in section["runbook"]
    assert section["cells"] == json.loads((world.root / "synthesis.json").read_text())["cells"]
    assert cells_tool.main(["--output", str(out)]) == 2                    # write-once


def test_a_missing_or_misattributed_artifact_writes_nothing(world, tmp_path):
    victim = Path(syn.cell_paths("0_180", 90, 8)["receipt"])
    body = json.loads(victim.read_text())
    victim.write_text(json.dumps({**body, "rc": 1}, indent=2))
    out = tmp_path / "a" / "synthesis_cells.json"
    assert cells_tool.main(["--output", str(out)]) == 2 and not out.exists()
    victim.unlink()
    assert cells_tool.main(["--output", str(out)]) == 2 and not out.exists()
