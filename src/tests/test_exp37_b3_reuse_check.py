"""exp_37 final fix round (Codex code review 3, H-04): the read-only B3 reuse check ``tools/exp37_b3_reuse_check.py``.

The 4 reused exp_35 phase-(i) C4 cells are prerequisites, checked by their exact pinned digests (metrics JSON,
receipt, dump manifest; and the stream sidecar the receipt binds), never by existence:
- ``--stage run`` (before ``B3S.run``: the leaf's RPL-37 replay needs the reference) requires the C4 / K8 theta-90 cell
  (metrics ``ef2840c7...``);
- ``--stage cells`` (before ``B3S.cells``) requires all four (T K1 ``1e1768e3...``, T K8 ``f167470e...``, C4 K1
  ``9674cfff...``, C4 K8 ``ef2840c7...``).
On success it writes one hashed, write-once record (``--output``; the runbook binds its sha256 into the step receipt)
and exits 0; a missing or wrong-hash artifact exits 2 and writes nothing. CPU; the world's stand-in roots (patched),
never the live exp_35 / NAS roots. Written by the exp_37 Coder (Claude Opus 5.5, max effort), final fix round: fixed
after review 3, not re-reviewed (cap).
"""
import hashlib
import json
from pathlib import Path

import pytest

from exp37_world import World, patch_synthesis
from src.localization import exp37_synthesis as syn
from src.localization.pilot import canonical_sha256
from tools import exp37_b3_reuse_check as reuse

K8 = ("c4", 90, 8)
K1 = ("c4", 90, 1)


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path)
    patch_synthesis(monkeypatch, w)
    w.synthesis()
    monkeypatch.setattr(syn, "REUSED", w.reused)
    return w


def test_the_registered_digests_are_the_pinned_ones():
    assert syn.REUSED["exp35_BF_C4_S40000_K8_s42_th90"]["metrics_sha256"].startswith("ef2840c7")
    assert syn.REUSED["exp35_BF_C4_S40000_K1_s42_th90"]["metrics_sha256"].startswith("9674cfff")
    assert syn.REUSED["exp35_BF_T_S40000_K1_s42"]["metrics_sha256"].startswith("1e1768e3")
    assert syn.REUSED["exp35_BF_T_S40000_K8_s42"]["metrics_sha256"].startswith("f167470e")
    assert reuse.STAGES == {"run": [K8], "cells": [("c4", 0, 1), ("c4", 0, 8), K1, K8]}


@pytest.mark.parametrize("stage, cells", [("run", 1), ("cells", 4)])
def test_a_complete_reference_passes_and_is_recorded_once(world, tmp_path, stage, cells):
    out = tmp_path / "gates" / f"b3s_reuse_{stage}.json"
    assert reuse.main(["--stage", stage, "--output", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["record_type"] == "exp37_b3_reuse_check" and record["stage"] == stage and record["verdict"] == "ok"
    assert record["sha256"] == canonical_sha256({k: v for k, v in record.items() if k != "sha256"})
    assert len(record["cells"]) == cells and all(len(a) == 4 for a in record["cells"].values())
    assert reuse.main(["--stage", stage, "--output", str(out)]) == 2                      # write-once


def _victim(cell, key):
    paths = syn.cell_paths(*cell)
    return Path(paths[key]) if key != "stream" else Path(paths["metrics_json"]).with_name(
        f"{syn.cell_stem(*cell)}.stream.json")


@pytest.mark.parametrize("key", ["metrics_json", "receipt", "dump_manifest", "stream"])
@pytest.mark.parametrize("damage", ["missing", "wrong-hash"])
def test_a_missing_or_wrong_hash_k8_reference_refuses_before_the_leaf(world, tmp_path, capsys, key, damage):
    victim = _victim(K8, key)
    if damage == "missing":
        victim.unlink()
    else:
        victim.write_bytes(victim.read_bytes() + b" ")
    out = tmp_path / "gates" / "b3s_reuse_run.json"
    assert reuse.main(["--stage", "run", "--output", str(out)]) == 2 and not out.exists()
    assert "exp35_BF_C4_S40000_K8_s42_th90" in capsys.readouterr().err


@pytest.mark.parametrize("cell", [("c4", 0, 1), ("c4", 0, 8), K1])
@pytest.mark.parametrize("damage", ["missing", "wrong-hash"])
def test_every_reused_cell_is_required_before_the_cells(world, tmp_path, cell, damage):
    victim = _victim(cell, "metrics_json")
    if damage == "missing":
        victim.unlink()
    else:
        victim.write_text("{}")
    assert reuse.main(["--stage", "run", "--output", str(tmp_path / "run.json")]) == 0   # the K8 reference is intact
    out = tmp_path / "cells.json"
    assert reuse.main(["--stage", "cells", "--output", str(out)]) == 2 and not out.exists()


def test_the_digests_come_from_the_pins_not_from_the_files(world, tmp_path, monkeypatch):
    victim = _victim(K8, "metrics_json")
    victim.write_text("{}")                                               # consistent with nothing pinned
    name = syn.cell_name(*K8)[0]
    pinned = dict(syn.REUSED[name])
    assert pinned["metrics_sha256"] != hashlib.sha256(victim.read_bytes()).hexdigest()
    assert reuse.main(["--stage", "run", "--output", str(tmp_path / "x.json")]) == 2
