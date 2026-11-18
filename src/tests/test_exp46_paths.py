"""exp_46 -- names and paths (plan_c4flip_data_curve.md r3 §3.2 "Names / paths"; Codex plan r2 finding 1).

* Every exp_46 WRITE path or name -- save roots, campaign / snapshot roots, farm, dump root, records, ledger, owner
  records, submit locks, run-dir internals, probes, job names -- lies in exp_46's own namespace: none names exp_38's
  or exp_45's. The shared A6000 card locks ``.exp38_card<i>.lock`` are the ONLY intentional sharing.
* The path-literal sweep is a permanent test: every string constant of the exp_46 kit (docstrings excluded) that
  names exp_38 / exp_45 (``exp38`` / ``exp-38`` / ``exp_38`` / ``exp45`` / ``exp-45`` / ``exp_45`` followed by a path or
  name character) is registered here with its disposition -- a read-only exp_38 dependency, a protected root of the
  isolation guard, the shared card lock, or a refusal text. An unregistered one fails.
* The inherited A6000 training-SMOKE destination ``outputs_FLAC/exp38_smoke`` is refused before any gate or write.
"""
import ast
import os
import re
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPREL46 = "worklog/worklog_yixun/exp_46_c4flip_data_curve_claude"
EXP = os.path.join(REPO, EXPREL46)
sys.path.insert(0, EXP)
import exp46_registry as reg  # noqa: E402
import exp46_eval as ev  # noqa: E402
import exp46_submit_neuronic as sub  # noqa: E402
import exp46_supervise as sv  # noqa: E402

KIT = ("exp46_registry.py", "exp46_supervise.py", "exp46_submit_neuronic.py", "exp46_init_digest.py", "exp46_eval.py",
       "exp46_collect.py", "exp46_table.py", "exp46_regression_compare.py")
# another experiment's name followed by a path / name character (prose like "exp_38's" is not a literal path)
FOREIGN = re.compile(r"exp[-_]?(38|45)(?=[-_./a-zA-Z0-9]|$)")
RUN = "exp46_RFf025_l40_2x32"

# the registered dispositions of every foreign literal in the kit (the commit message of the sweep lists them too)
READ_ONLY_EXP38 = "read-only exp_38 dependency (EXP38_DIR tools / artifacts, never written)"
PROTECTED = "protected root of the isolation guard (refused as a destination)"
SHARED_LOCK = "the shared A6000 card lock (the only intentional sharing)"
REFUSAL = "refusal text (the A6000 SMOKE destination is never written)"
EXP38_IMPORT = "exp_38 results import (plan r3 §3.2, the approved r2-6 materialisation into the generator root)"
REGISTERED = {}            # filled below: {(file, literal): disposition}


def _register(files, literals, disposition):
    for f in files:
        for lit in literals:
            REGISTERED[(f, lit)] = disposition


def _string_constants(path):
    """Every str constant of a module except docstrings (module / class / function first-statement strings)."""
    tree = ast.parse(open(path).read(), path)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docs.add(id(first.value))
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def foreign_literals():
    out = {}
    for name in KIT:
        for s in _string_constants(os.path.join(EXP, name)):
            if FOREIGN.search(s):
                key = ("CODE: " + s.strip().split("\n")[0]) if "\n" in s.strip() else s    # an embedded program
                out[(name, key)] = out.get((name, key), 0) + 1
    return out


def test_every_foreign_literal_of_the_kit_is_registered_with_its_disposition():
    got = set(foreign_literals())
    unregistered = sorted(got - set(REGISTERED))
    stale = sorted(set(REGISTERED) - got)
    assert not unregistered and not stale, "\n" + "\n".join(f"{kind} {f} {s!r}" for kind, items in
                                                       (("UNREGISTERED", unregistered), ("STALE", stale)) for f, s in items)


def test_the_sbatch_names_only_exp46():
    text = open(os.path.join(EXP, "exp46_train.sbatch")).read()
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert not FOREIGN.search(code), [m.group(0) for m in FOREIGN.finditer(code)]
    assert "exp46_supervise.py" in code and "^exp46_" in code


def _write_paths():
    """Every write path / name the exp_46 kit uses, as registered constants or the functions that build them."""
    paths = [p["save_root"] for p in reg.PLATFORMS.values()]
    paths += list(reg.CAMPAIGN_ROOT.values()) + list(sv.SNAPSHOT_ROOT.values()) + [ev.SNAPSHOT_ROOT, ev.FARM,
                                                                                   ev.DUMP_ROOT, ev.EVAL_LOG_DIR]
    paths += [reg.EXP_DIR, reg.E0_RECEIPT, reg.ENV_RECEIPT_LOCK, reg.env_receipt_rel("neuronic"),
              reg.env_receipt_rel("a6000"), reg.STORAGE_LEDGER, reg.STORAGE_LOCK, *reg.RUN_DIR_OWN, reg.RECEIPTS_DIR,
              reg.TRAINING_RECEIPT, reg.COMMIT_MARKER, reg.LOSS_RECORD, reg.LOSS_RECORD_HPARAMS]
    paths += [sub.SUBLOG, sub.OWNERS, sub.owner_dir(RUN, "FULL", "/x"), sub.submit_lock_path(RUN, "FULL", "/x"),
              sub.job_name(RUN, "FULL", "a" * 32), sub.BRANCH]
    paths += [reg.save_dir("neuronic", RUN), reg.smoke_receipt_rel(RUN, "neuronic", "ts")]
    return paths


def test_no_exp46_write_path_names_another_experiments_namespace():
    paths = _write_paths()
    assert len(paths) > 25 and [p for p in paths if FOREIGN.search(p)] == []
    # the namespaced ones carry exp_46's own name (the rest are fixed file names inside exp_46's run dirs)
    for p in paths[:9] + [reg.EXP_DIR, reg.E0_RECEIPT, reg.STORAGE_LEDGER, sub.SUBLOG, sub.OWNERS, sub.BRANCH]:
        assert re.search(r"exp[-_]?46|c4flip", p), p


def test_the_run_dir_internals_and_probes_are_exp46_names():
    assert ".exp46_lock" in reg.RUN_DIR_OWN and ".exp46_tools" in reg.RUN_DIR_OWN
    assert not any(n.startswith(".exp38") for n in reg.RUN_DIR_OWN)
    assert reg._own_entry(".exp46_lock.released_2026") and reg._own_entry(".exp46_probe_1_2")
    assert not reg._own_entry(".exp38_lock") and not reg._own_entry(".exp38_lock.released_2026")
    import inspect
    assert inspect.signature(reg.storage_gate).parameters["tag"].default == "exp46"


def test_the_shared_card_locks_are_the_only_intentional_sharing():
    assert sv.LOCKDIR == ev.LOCKDIR == "/home/yixunhu/codespace"
    for name in ("exp46_supervise.py", "exp46_eval.py"):
        text = open(os.path.join(EXP, name)).read()
        assert '".exp38_card' in text or 'f".exp38_card' in text, name


def test_the_snapshot_roots_are_exp46s():
    assert sv.SNAPSHOT_ROOT == {"a6000": "/home/yixunhu/codespace/exp-46-snapshots"}
    assert ev.SNAPSHOT_ROOT == "/home/yixunhu/codespace/exp-46-snapshots"
    assert ev.eval_snapshot_dir("c" * 64).startswith(ev.SNAPSHOT_ROOT + "/")


def test_the_a6000_smoke_destination_is_refused_before_any_gate_or_write(tmp_path, monkeypatch):
    touched = []
    monkeypatch.setattr(reg, "REPO", str(tmp_path))
    monkeypatch.setattr(sv, "common_gates", lambda a, p: touched.append("common_gates"))
    monkeypatch.setattr(sv, "campaign_gates", lambda a, p: touched.append("campaign_gates"))
    a = sv.parse(["--platform", "a6000", "--mode", "SMOKE_DDP", "--run", RUN, "--expect-sha", "f" * 40,
                  "--token", "a" * 32])
    with pytest.raises(sv.Refused, match="no A6000 training path"):
        sv.mode_campaign(a, "a6000")
    assert touched == [] and os.listdir(str(tmp_path)) == []
    assert sv._dispatch(a) != 0 and touched == [] and not (tmp_path / "outputs_FLAC").exists()


def test_the_dry_run_names_no_a6000_smoke_destination(capsys, monkeypatch):
    monkeypatch.setattr(sv, "common_gates", lambda a, p: ("rf", "025", "2x32", sys.executable,
                                                          {"argv": ["--save-dir", "/x"], "run_dir_state": "fresh"}, {}))
    monkeypatch.setattr(sv, "backend_gate", lambda *x: {})
    monkeypatch.setattr(reg, "closure_digest", lambda repo=None: {"closure_sha256": "c" * 64})
    sv.mode_dry_run(sv.parse(["--platform", "a6000", "--mode", "DRY_RUN", "--run", RUN]), "a6000")
    out = capsys.readouterr().out
    assert "exp38_smoke" not in out and "SMOKE_DDP: refused" in out


# the sweep's registered dispositions (plan r3 §3.2 "Names / paths": renamed, refused, or a registered read-only exp_38
# dependency; every exp_45 / exp_38 WRITE literal was renamed, see test_no_exp46_write_path_names_another_experiments_)
L1_PARENT = "the L1 comparison's parent label / exp_45's registered node ids (read-only junit inputs, plan r3 §3.2)"
_register(["exp46_registry.py"], ["worklog/worklog_yixun/exp_38_data_efficiency_claude", "/exp38_registry.py",
                                  "/exp38_subset_audit.py", "/exp38_content_manifest.py"], READ_ONLY_EXP38)
_register(["exp46_registry.py"], ["/media/diskstation/yixunhu/FLAC/checkpoints/exp38_data_efficiency",
                                  "/n/fs/gatrdp/codespace/FLAC/outputs_FLAC/exp38_data_efficiency",
                                  "/media/diskstation/yixunhu/FLAC/exp38_pred_waveforms",
                                  "outputs_FLAC/exp38_data_efficiency/eval"], PROTECTED)
_register(["exp46_eval.py", "exp46_supervise.py"], [".exp38_card"], SHARED_LOCK)
_register(["exp46_supervise.py"], [" is refused (the inherited SMOKE destination outputs_FLAC/exp38_smoke is never "
                                   "written); SMOKE runs on neuronic only"], REFUSAL)
_register(["exp46_regression_compare.py"], [
    "--exp38", "exp38", "new_exp45_skips",
    "src/tests/test_exp45_receipts.py::test_a6000_links_need_the_registered_targets",
    "src/tests/test_exp45_receipts.py::test_mount_fstype_sees_the_nas_behind_autofs",
    "src/tests/test_exp45_receipts.py::test_the_environment_snapshot_is_complete_on_this_box",
    "src/tests/test_exp45_receipts.py::test_the_real_gpu0_smoke_evidence_passes_the_exact_gate[vanilla-de_yaw_f025]",
    "src/tests/test_exp45_registry.py::test_the_real_rf_100_percent_endpoint_satisfies_the_exact_schema",
    r"^src/tests/(test_exp45_[^/]+|test_gen_model_comparison_exp45)\.py::"], L1_PARENT)
_register(["exp46_table.py"], [
    "/c4flip_data_curve_exp38_import.json", "--exp38-commit", "exp38_commit", "--exp38-commit must be a full 40-hex "
    "commit id", "exp38_import_sha256", "exp46_exp38_import_", "import-exp38", "CODE: import json, os, sys",
    "OrbitRIR (B-F), exp_38", "YAW-Aug, exp_38", "outputs_FLAC/exp38_data_efficiency/eval"], EXP38_IMPORT)
