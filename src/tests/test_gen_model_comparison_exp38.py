"""exp_38 B10 — the table generator's ``exp38`` contract (gen_model_comparison.py, announcement 04).

Twelve preregistered rows (YAW / B-F × 25 / 50 / 75 % × K 1 / 8) whose files must pass exp38_eval's
OFFLINE cell admission, and a round gate that publishes a run only as a complete K=1 + K=8 pair.
"""
import importlib.util
import json
import os
import shutil

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(REPO, "worklog", "worklog_yixun", "gen_model_comparison.py")
EXP_REL = os.path.join("worklog", "worklog_yixun", "exp_38_data_efficiency_claude")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load(GEN, "gen_model_comparison_exp38_test")


def test_twelve_preregistered_rows_with_exact_globs():
    rows = [r for r in gen.ROWS if len(r) > 4 and r[4] == "exp38"]
    assert len(rows) == 12
    for label, proto, k, pats, _ in rows:
        run = gen.exp38_run_of(label)
        assert run and proto == ("fa eval" if "_bf_" in run else "vanilla eval") and k in (1, 8)
        suffix = "_fa_invariant_a4" if "_bf_" in run else ""
        assert pats == [f"outputs_FLAC/exp38_data_efficiency/eval/{run}/*_metrics_1_1.0_{run}_K{k}_s4[2-6]{suffix}.json"]
    assert not any("exp14_" in p for r in rows for p in r[3])


def _tmp_root(tmp_path, monkeypatch):
    """A self-contained repo root holding only what OFFLINE admission reads: the kit, the arm
    configs, the 68 reference cells; the cached exp38_registry module is set aside so the root's
    own copy (REPO = this root) is the one the loaded validator uses."""
    import sys
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    for rel in (os.path.join(EXP_REL, "exp38_eval.py"), os.path.join(EXP_REL, "exp38_registry.py"),
                os.path.join(EXP_REL, "FLAC_AR_BF_cap32.json"), os.path.join(EXP_REL, "FLAC_AR_BF_cap8.json"),
                "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json",
                "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",
                "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"):
        (root / os.path.dirname(rel)).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(os.path.join(REPO, rel), root / rel)
    shutil.copytree(os.path.join(REPO, EXP_REL, "ref_cells"), root / EXP_REL / "ref_cells")
    for name in ("exp38_registry", "exp38_eval"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    ev = gen._load_exp38_eval(str(root))
    assert ev.reg.REPO == str(root)
    return root, ev


def _write_run(root, ev, run, K, seeds=(42, 43, 44, 45, 46)):
    sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    import exp38_fixtures as fx
    if not os.path.exists(ev._abs(ev.E0_RECEIPT)):
        fx.make_e0_receipt(ev)
    campaign = ev.validate_e0_receipt(offline=True)
    farm = os.path.dirname(ev._abs(ev.run_cells(run)[0]["ckpt_link"]))
    if not os.path.exists(os.path.join(farm, ev.reg.TRAINING_RECEIPT)):
        tr, digest = fx.make_training_receipt(ev, run, farm, campaign["closure_sha256"])
    tr, digest = ev.reg.read_sealed(os.path.join(farm, ev.reg.TRAINING_RECEIPT), ev.reg.TRAINING_KEYS)
    files = []
    for c in ev.run_cells(run):
        if c["K"] == K and c["seed"] in seeds:
            fx.make_cell(ev, c, campaign, (tr["final_ckpt_sha256"], tr["final_ckpt_bytes"], digest),
                         str(root / "logs"), dump=False)
            files.append(ev._abs(c["metrics"]))
    return files


def test_validate_exp38_cell_offline(tmp_path, monkeypatch):
    root, ev = _tmp_root(tmp_path, monkeypatch)
    files = _write_run(root, ev, "de_yaw_f025", 8)
    ok, probs = gen.validate_exp38_cell(files, repo_root=str(root), expected_run="de_yaw_f025", expected_k=8)
    assert ok, probs
    ok, probs = gen.validate_exp38_cell(files, repo_root=str(root), expected_run="de_yaw_f025", expected_k=1)
    assert not ok                                              # K mismatch
    rec = json.load(open(files[0]))
    rec["metrics"]["T60"] = 2.0
    json.dump(rec, open(files[0], "w"))
    ok, probs = gen.validate_exp38_cell(files, repo_root=str(root), expected_run="de_yaw_f025", expected_k=8)
    assert not ok and any("receipt metrics_sha256" in p for p in probs)
    assert gen.validate_exp38_cell([], repo_root=str(root), expected_run="de_yaw_f025", expected_k=8) == (True, [])
    ok, probs = gen.validate_exp38_cell(files, repo_root=str(tmp_path / "nowhere"), expected_run="de_yaw_f025", expected_k=8)
    assert not ok and "cannot load" in probs[0]


def test_offline_admission_needs_the_e0_receipt_and_the_training_receipt(tmp_path, monkeypatch):
    root, ev = _tmp_root(tmp_path, monkeypatch)
    files = _write_run(root, ev, "de_bf_f050", 1)
    assert gen.validate_exp38_cell(files, repo_root=str(root), expected_run="de_bf_f050", expected_k=1)[0]
    farm = os.path.dirname(files[0])
    os.rename(os.path.join(farm, "TRAINING_COMPLETE.json"), os.path.join(farm, "x.json"))
    ok, probs = gen.validate_exp38_cell(files, repo_root=str(root), expected_run="de_bf_f050", expected_k=1)
    assert not ok and any("TRAINING_COMPLETE" in p for p in probs)
    os.rename(os.path.join(farm, "x.json"), os.path.join(farm, "TRAINING_COMPLETE.json"))
    os.remove(ev._abs(ev.E0_RECEIPT))
    ok, probs = gen.validate_exp38_cell(files, repo_root=str(root), expected_run="de_bf_f050", expected_k=1)
    assert not ok and any("e0_receipt" in p for p in probs)


def test_round_gate_publishes_only_complete_pairs():
    label = gen.EXP38_LABELS[("yaw", "025")]
    rendered = [{"label": label, "K": 8, "line": "| ok |", "contract": "exp38"},
                {"label": label, "K": 1, "line": "| x | *pending (3/5 seeds on disk)* |", "contract": "exp38"}]
    withheld, probs = gen.check_exp38_round(rendered, {(label, 8): ["a"], (label, 1): ["b"]})
    assert withheld == {label} and probs
