"""exp_37 fix round 2 (Codex code review 2, BLOCKING 5): the B3 synthesis cells are bound by their OWN artifacts.

``exp37_synthesis.validate_section`` accepts the 12 cells (B-F@40k x {0} / {0,180} / C4 x theta {0, 90} x K {1, 8},
seed 42) only if every metrics JSON, schema-2 receipt and dump manifest is unique, hashes to the spec, and itself
carries the cell: checkpoint, SDPA backend + pinned overlay / launcher, orbit, theta, K (dataset config + sha256),
seed, evaluator identity (pin = the leaf commit for the 8 new cells, H35 for the 4 reused C4 cells; tree digests; model
config; the whole argv), source type, stem, and a dump manifest that is the receipt's and binds the metrics file.
CPU, synthetic artifacts as the exp_25 runner writes them. Written by the exp_37 Coder (Claude Opus 5.5, max effort).
"""
import hashlib
import json
from pathlib import Path

import pytest

from exp37_world import World, patch_synthesis
from src.localization import exp37_synthesis as syn


@pytest.fixture
def section(tmp_path, monkeypatch):
    world = World(tmp_path)
    patch_synthesis(monkeypatch, world)
    body = json.loads(world.synthesis().read_text())
    monkeypatch.setattr(syn, "REUSED", world.reused)
    return world, body


def test_a_complete_section_validates_from_its_own_artifacts(section):
    _world, body = section
    out = syn.validate_section(body)
    assert len(out["cells"]) == 12 and {c["source"] for c in out["cells"]} == {"new", "reused_exp35"}
    c4 = [c for c in out["cells"] if c["orbit"] == "c4"]
    assert len(c4) == 4 and all(c["pin"] == syn.H35_PIN for c in c4)
    assert all(c["pin"] == "e" * 40 for c in out["cells"] if c["orbit"] != "c4")
    assert all(set(c["per_scene_mean"]) == {"T60", "C50"} for c in out["cells"])


def _rehash(cell, key):
    cell[syn.SHA_KEY[key]] = hashlib.sha256(Path(cell[key]).read_bytes()).hexdigest()


def _edit_json(cell, key, change):
    path = Path(cell[key])
    payload = json.loads(path.read_text())
    change(payload)
    path.write_text(json.dumps(payload, indent=2) if key == "receipt" else json.dumps(payload))
    _rehash(cell, key)


@pytest.mark.parametrize("change, needle", [
    (lambda c: _edit_json(c, "metrics_json", lambda m: m.update(frame_avg_angles=[0.0, 90.0, 180.0, 270.0])),
     "frame_avg_angles"),
    (lambda c: _edit_json(c, "metrics_json", lambda m: m.update(rotate_deg=45.0)), "rotate_deg"),
    (lambda c: _edit_json(c, "metrics_json", lambda m: m.update(seed=43)), "seed"),
    (lambda c: _edit_json(c, "metrics_json", lambda m: m["by_scene"]["Cafe"].pop("C50")), "per scene"),
    (lambda c: _edit_json(c, "receipt", lambda r: r.update(dataset_config=syn.DATASET[8][0])), "dataset_config"),
    (lambda c: _edit_json(c, "receipt", lambda r: r.update(expect_head="f" * 40)), "expect_head"),
    (lambda c: _edit_json(c, "receipt", lambda r: r.update(attention_backend="flash_attn")), "attention_backend"),
    (lambda c: _edit_json(c, "receipt", lambda r: r.update(ckpt_sha256_loaded="0" * 64)), "ckpt_sha256_loaded"),
    (lambda c: _edit_json(c, "receipt", lambda r: r["argv"].__setitem__(r["argv"].index("--seed") + 1, "43")), "--seed"),
    (lambda c: _edit_json(c, "receipt", lambda r: r.pop("lane")), "key set"),
    (lambda c: _edit_json(c, "receipt", lambda r: r.update(post_digest="4" * 64)), "digest"),
    (lambda c: c.update(source="reused_exp35"), "source"),
    (lambda c: (Path(c["dump_manifest"]).write_text(json.dumps({"n": 1})), _rehash(c, "dump_manifest")), "dump manifest"),
    (lambda c: Path(c["metrics_json"]).write_text("{}"), "hash"),
], ids=["orbit", "theta", "seed", "scene-metric", "K", "pin", "backend", "ckpt", "argv-seed", "receipt-keys", "digest",
        "source", "dump-manifest", "unhashed"])
def test_every_misattribution_is_refused(section, change, needle):
    _world, body = section
    change(body["cells"][2])                                               # {0} theta 90 K1: a new cell
    with pytest.raises(RuntimeError, match=needle):
        syn.validate_section(body)


def test_a_reused_cell_must_carry_the_h35_pin(section):
    _world, body = section
    cell = next(c for c in body["cells"] if c["orbit"] == "c4")
    _edit_json(cell, "receipt", lambda r: r.update(expect_head="e" * 40))
    with pytest.raises(RuntimeError, match="expect_head"):
        syn.validate_section(body)


def test_artifacts_are_never_shared_between_cells(section):
    _world, body = section
    for key in ("metrics_json", "metrics_sha256"):
        body["cells"][1][key] = body["cells"][0][key]                      # one metrics file for two cells
    with pytest.raises(RuntimeError, match="reused"):
        syn.validate_section(body)


def test_a_swapped_receipt_names_another_cell(section):
    _world, body = section
    a, b = body["cells"][0], body["cells"][1]
    a["receipt"], b["receipt"] = b["receipt"], a["receipt"]
    a["receipt_sha256"], b["receipt_sha256"] = b["receipt_sha256"], a["receipt_sha256"]
    with pytest.raises(RuntimeError, match="cell|stem"):
        syn.validate_section(body)


def test_the_registered_synthesis_identity_is_pinned():
    """The Planner's leaf (exp-37-b3-synthesis launch commit d3013734, its run worktree) and the reused exp_35
    phase-(i) C4 cells with their full digests (the leaf's 19-26-27 record)."""
    assert syn.LEAF_COMMIT == "d30137349027914af7f04462b05347f0f9e03c02"
    assert syn.LEAF_ROOT == "/home/yixunhu/codespace/exp-37-b3-synthesis"
    assert syn.EXP35_ROOT == "/home/yixunhu/codespace/exp-35-orbitrir-yaw"
    assert syn.REUSED["exp35_BF_T_S40000_K1_s42"]["metrics_sha256"].startswith("1e1768e3")
    assert syn.REUSED["exp35_BF_C4_S40000_K8_s42_th90"]["dump_manifest_sha256"].startswith("975cb721")
    assert sorted(syn.REUSED) == sorted(syn.cell_name(o, t, k)[0] for o in ("c4",) for t in (0, 90) for k in (1, 8))


def test_the_artifacts_must_sit_at_the_pinned_layout(section, tmp_path):
    _world, body = section
    cell = body["cells"][0]
    moved = tmp_path / "elsewhere.json"
    moved.write_bytes(Path(cell["metrics_json"]).read_bytes())
    cell["metrics_json"] = str(moved)
    with pytest.raises(RuntimeError, match="layout"):
        syn.validate_section(body)


def test_a_reused_cell_must_be_the_pinned_exp35_evidence(section, monkeypatch):
    _world, body = section
    name = next(c["cell"] for c in body["cells"] if c["source"] == "reused_exp35")
    monkeypatch.setitem(syn.REUSED, name, {**syn.REUSED[name], "receipt_sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="pinned"):
        syn.validate_section(body)


def test_the_section_must_name_the_leaf_launch_commit(section):
    _world, body = section
    body["leaf_commit"] = "f" * 40
    with pytest.raises(RuntimeError, match="leaf commit"):
        syn.validate_section(body)
