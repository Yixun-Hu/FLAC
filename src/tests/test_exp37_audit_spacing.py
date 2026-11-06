"""exp_37 round B-c: the parameterized geometry audit and the pilot rebuild (plan §4, §10 "B2 mathematics": the
0.75 m audit's single GLOBAL height branch pinned to z_band, all 5,337 grids non-empty, the exact same 64 query IDs
required). Pure helpers on CPU; the full audits are validation runs (logged). Written by the exp_37 Coder (Claude
Opus 5.5, max effort), round B-c.
"""
import json
from pathlib import Path

import pytest

from src.localization.ar_queries import load_context_manifest
from src.localization.pilot import build_pilot_manifest, load_pilot_manifest
from tools import audit_localization_geometry as audit_tool
from tools import build_localization_pilot as pilot_tool

REPO = Path(__file__).resolve().parents[2]
E9 = REPO / "worklog/worklog_yixun/exp_09_localization_grid_preflight_claude"


def test_the_z_branch_is_computed_unless_pinned():
    assert audit_tool.decide_z_branch("full_height", [3, 4], None) == "full_height"
    assert audit_tool.decide_z_branch("full_height", [3, 4], "z_band") == "z_band"
    assert audit_tool.decide_z_branch("z_band", [3, 4], "z_band") == "z_band"


def test_a_pinned_z_band_with_an_empty_grid_is_refused():
    with pytest.raises(RuntimeError, match="empty"):
        audit_tool.decide_z_branch("full_height", [3, 0, 4], "z_band")
    with pytest.raises(ValueError, match="z branch"):
        audit_tool.decide_z_branch("z_band", [3], "sideways")


def test_the_cli_takes_the_spacing_and_the_pin(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(audit_tool, "audit", lambda *a, **kw: seen.update(kw) or {"sha256": "x", "geometry_gate": "PASS",
                                                                                   "z_branch": "z_band", "totals": {}})
    monkeypatch.setattr(audit_tool, "_markdown", lambda result: "")
    monkeypatch.setattr("sys.argv", ["audit", "--context-manifest", "c.json", "--mesh-root", str(tmp_path),
                                     "--output-json", str(tmp_path / "a.json"), "--output-md", str(tmp_path / "a.md"),
                                     "--fast-cost-audit", "--grid-spacing", "0.75", "--pin-z-branch", "z_band"])
    audit_tool.main()
    assert seen["grid_spacing"] == 0.75 and seen["pin_z_branch"] == "z_band" and seen["compute_topology"] is False


def test_the_committed_pilot_is_rebuilt_exactly_and_another_seed_is_refused():
    context = load_context_manifest(E9 / "context_manifest_exp01_seed42.json")
    audit = json.loads((E9 / "geometry_audit.json").read_text())
    reference = load_pilot_manifest(E9 / "pilot_manifest_seed42_4_per_room.json")
    rebuilt = build_pilot_manifest(context, audit, queries_per_room=4, seed=42)
    assert rebuilt == reference                                     # same payload, same sha256
    assert pilot_tool.same_queries_reasons(rebuilt, reference) == []
    other = build_pilot_manifest(context, audit, queries_per_room=4, seed=43)
    reasons = pilot_tool.same_queries_reasons(other, reference)
    assert reasons and "query" in reasons[0]


def test_the_pilot_cli_refuses_to_write_other_queries(monkeypatch, tmp_path):
    out = tmp_path / "pilot.json"
    monkeypatch.setattr("sys.argv", ["build", "--context-manifest", str(E9 / "context_manifest_exp01_seed42.json"),
                                     "--geometry-audit", str(E9 / "geometry_audit.json"), "--output", str(out),
                                     "--seed", "43", "--require-same-queries-as",
                                     str(E9 / "pilot_manifest_seed42_4_per_room.json")])
    with pytest.raises(SystemExit) as stop:
        pilot_tool.main()
    assert stop.value.code == 2 and not out.exists()
