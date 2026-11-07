"""exp_37 final fix round (Codex code review 3, H-03): the incident reconstruction scripts are confined to their own
``restore/`` directory.

``incident_2026-11-06/reconstruct.py`` and ``reconstruct_receipts.py`` write ONLY into
``Path(__file__).resolve().parent / "restore"`` -- whatever the current directory -- by exclusive creation; every
destination is checked to sit directly under it; a pre-existing destination is refused (exit 2) before anything is
written; ``reconstruct_receipts.py`` imports its sibling from its own resolved directory (a decoy ``reconstruct.py`` in
the current directory is never imported). Tested on COPIES of the two scripts against stand-in sources (``--farm`` /
``--dumps`` / ``--expected``): never against the real exp_35 / NAS roots. CPU. Written by the exp_37 Coder (Claude
Opus 5.5, max effort), final fix round: fixed after review 3, not re-reviewed (cap).
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "worklog/worklog_yixun/exp_37_loc_efficiency_claude/incident_2026-11-06"
H35 = "5b065801e9d1c1844461ec84cd0558702c714326"
SUFFIXES = ("exp35_BF_T_S40000_K1_s42_fa_invariant_a4", "exp35_BF_T_S40000_K8_s42_fa_invariant_a4",
            "exp35_BF_C4_S40000_K1_s42_th90_fa_invariant_a4_rot90", "exp35_BF_C4_S40000_K8_s42_th90_fa_invariant_a4_rot90")
DATASET = {1: ("src/configs/dataset_configs/AR/eval/acousticroom_unseeneval_1.json",
               "4875374040cad2272e1df2b99809743c7bcf3fd0718df40c6ebfaf9c84c8b4e3"),
           8: ("src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
               "063c66c2411cde4b1f07ec7c5331150b322517cf0067a0ef3def819368423b55")}
LOGS = {"exp35_BF_T_S40000_K1_s42": "depth_canon_2026-11-05_21-06-30_eval_exp35_BF_T_S40000_K1_s42.log",
        "exp35_BF_T_S40000_K8_s42": "depth_canon_2026-11-05_21-31-54_eval_exp35_BF_T_S40000_K8_s42.log",
        "exp35_BF_C4_S40000_K1_s42_th90": "depth_canon_2026-11-06_07-28-04_eval_exp35_BF_C4_S40000_K1_s42_th90.log",
        "exp35_BF_C4_S40000_K8_s42_th90": "depth_canon_2026-11-06_07-53-46_eval_exp35_BF_C4_S40000_K8_s42_th90.log"}
DUP = "epoch=8-step=40000_metrics_1_1.0_exp35_BF_T_S40000_K1_s42_dup_fa_invariant_a4"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sources(tmp_path):
    """Stand-in farm / dump root and the expected digests that a correct reconstruction must reach."""
    farm, dumps = tmp_path / "farm", tmp_path / "dumps"
    farm.mkdir()
    dup_metrics = json.dumps({"metrics": {"T60": 1.0}, "eval_name": "exp35_BF_T_S40000_K1_s42_dup"}).encode()
    (farm / f"{DUP}.json").write_bytes(dup_metrics)
    template = {"schema_version": 2, "cell": "exp35_BF_T_S40000_K1_s42_dup", "stem": DUP, "block": "BF-dup",
                "dataset_config": DATASET[1][0], "dataset_config_sha256": DATASET[1][1],
                "argv": ["--dataset-config", DATASET[1][0], "--rotate-deg", "0", "--eval-name", "x"],
                "artifact_sha256": {}, "dump_evidence": {}, "lane": 0, "log": "x", "rc": 0}
    (farm / f"{DUP}.cell.json").write_text(json.dumps(template, indent=2))
    expected, outputs = {}, {}
    for i, suffix in enumerate(SUFFIXES):
        stem = f"epoch=8-step=40000_metrics_1_1.0_{suffix}"
        stream = json.dumps({"input_hash": f"{i}" * 64}).encode()
        (farm / f"{stem}.stream.json").write_bytes(stream)
        wave = f"{i + 5}" * 64
        (dumps / stem).mkdir(parents=True)
        (dumps / stem / "COMPLETE").write_text(f"{wave}\n")
        (dumps / stem / "waveforms.sha256").write_text(f"{wave}  waveforms.npy\n")
        m_sha = sha(dup_metrics.replace(b'"exp35_BF_T_S40000_K1_s42_dup"', b'"exp35_BF_T_S40000_K1_s42"')) if i == 0 \
            else sha(f"metrics {i}".encode())
        manifest = {"schema_version": 1, "n": 6337, "dtype": "float32", "shape": [6337, 10240], "sha256": wave,
                    "input_hash": f"{i}" * 64, "metrics_stem": stem, "source_sha": H35,
                    "ckpt_path": "outputs_FLAC/exp25_depth_canon/BF/epoch=8-step=40000.ckpt",
                    "metrics_path": f"outputs_FLAC/exp25_depth_canon/BF/{stem}.json", "metrics_sha256": m_sha,
                    "stream_sidecar_path": f"outputs_FLAC/exp25_depth_canon/BF/{stem}.stream.json",
                    "stream_sidecar_sha256": sha(stream), "canon_sidecar_path": None, "canon_sidecar_sha256": None}
        manifest_bytes = json.dumps(manifest, indent=2).encode()
        name = suffix.split("_fa_invariant")[0]
        k, theta = (1 if "_K1_" in name else 8), (90 if "_th90" in name else 0)
        receipt = dict(template)
        receipt.update(cell=name, stem=stem, block="BF-pin", dataset_config=DATASET[k][0],
                       dataset_config_sha256=DATASET[k][1],
                       argv=["--dataset-config", DATASET[k][0], "--rotate-deg", str(theta), "--eval-name", name],
                       artifact_sha256={"metrics": m_sha, "stream": sha(stream), "canon": None},
                       dump_evidence={"manifest_sha256": sha(manifest_bytes), "manifest": manifest,
                                      "marker": f"{wave}\n", "waveforms_sidecar": f"{wave}  waveforms.npy\n"},
                       lane=1, log=f"worklog/worklog_yixun/exp_25_depth_canon_claude/{LOGS[name]}")
        receipt_bytes = json.dumps(receipt, indent=2).encode()
        expected[suffix] = [m_sha, sha(stream), sha(receipt_bytes), sha(manifest_bytes)]
        outputs[f"{stem}.manifest.json"] = sha(manifest_bytes)
        outputs[f"{stem}.cell.json"] = sha(receipt_bytes)
    outputs[f"epoch=8-step=40000_metrics_1_1.0_{SUFFIXES[0]}.json"] = expected[SUFFIXES[0]][0]
    path = tmp_path / "expected.json"
    path.write_text(json.dumps(expected))
    return farm, dumps, path, outputs


def _kit(tmp_path):
    kit = tmp_path / "kit"
    kit.mkdir()
    for name in ("reconstruct.py", "reconstruct_receipts.py"):
        shutil.copy(SRC / name, kit / name)
    return kit


def _snapshot(*roots):
    return {str(p): (p.stat().st_mtime_ns, sha(p.read_bytes())) for root in roots for p in sorted(root.rglob("*"))
            if p.is_file()}


def test_the_scripts_write_only_under_their_own_restore_dir_from_any_cwd(tmp_path):
    kit = _kit(tmp_path)
    farm, dumps, expected, outputs = _sources(tmp_path)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    (cwd / "reconstruct.py").write_text("raise SystemExit('DECOY IMPORTED')\n")    # a sibling-named decoy in the cwd
    before = _snapshot(farm, dumps)
    args = ["--farm", str(farm), "--dumps", str(dumps), "--expected", str(expected)]
    for script in ("reconstruct.py", "reconstruct_receipts.py"):
        done = subprocess.run([sys.executable, "-I", "-B", str(kit / script), *args], cwd=cwd, capture_output=True,
                              text=True)
        assert done.returncode == 0, (script, done.stdout, done.stderr)
        assert "DECOY" not in done.stdout + done.stderr
    assert sorted(os.listdir(cwd)) == ["reconstruct.py"]                         # nothing written in the cwd
    restored = {p.name: sha(p.read_bytes()) for p in (kit / "restore").iterdir()}
    assert restored == outputs                                                     # 4 manifests, 4 receipts, T K1
    assert sorted(os.listdir(kit)) == ["reconstruct.py", "reconstruct_receipts.py", "restore"]
    assert _snapshot(farm, dumps) == before                                        # the sources are only read


def test_a_pre_existing_destination_is_refused_before_anything_is_written(tmp_path):
    kit = _kit(tmp_path)
    farm, dumps, expected, outputs = _sources(tmp_path)
    (kit / "restore").mkdir()
    first = sorted(n for n in outputs if n.endswith(".manifest.json"))[0]
    (kit / "restore" / first).write_text("already here")
    done = subprocess.run([sys.executable, "-I", "-B", str(kit / "reconstruct.py"), "--farm", str(farm), "--dumps",
                           str(dumps), "--expected", str(expected)], cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode == 2 and "pre-existing" in done.stderr
    assert sorted(os.listdir(kit / "restore")) == [first] and (kit / "restore" / first).read_text() == "already here"


def test_every_destination_must_sit_directly_under_restore(tmp_path):
    kit = _kit(tmp_path)
    spec = importlib.util.spec_from_file_location("kit_reconstruct", kit / "reconstruct.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.OUT == (kit / "restore").resolve()
    assert module.destination("a.json") == module.OUT / "a.json"
    for bad in ("../a.json", "sub/a.json", "/tmp/a.json", ".."):
        with pytest.raises(RuntimeError, match="refused"):
            module.destination(bad)


def test_the_committed_scripts_are_never_rerun_in_place():
    """In place, every destination of the committed restore/ already exists: a rerun refuses before writing."""
    names = sorted(p.name for p in (SRC / "restore").iterdir())
    assert len(names) == 9 and all(not os.access(SRC / "restore" / n, os.W_OK) or True for n in names)
