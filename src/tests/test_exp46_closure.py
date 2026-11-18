"""exp_46 -- the campaign closure (plan_c4flip_data_curve.md r3 §3.2 "Closure"; Codex plan r1 finding 1, r2 finding 3).

``_in_closure`` checks folder membership before ``EXTERNAL_CONFIGS``, so a file of the exp_46 folder enters the closure
only through the ``exp46_`` prefix rule or ``CLOSURE_EXPLICIT``: exp_46's two runtime inputs there -- the C4F config and
the f100 delta content manifest -- are admitted by EXACT path (independent of the prefix rule). exp_43's RF config is
no runtime input (plan r3 §2). Tests: exact-path admission; a throw-away git tree lists and hashes both, and one
changed byte of either changes the closure digest; the real snapshot carries the C4F config at its pinned sha256, has
the tree's closure digest, and the kit run FROM the isolated snapshot root resolves and admits the config there.
"""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude")
sys.path.insert(0, EXP)
import exp46_registry as reg  # noqa: E402

C4F = "worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/exp46_FLAC_AR_YAWAUG_C4FLIP.json"
MANIFEST = "worklog/worklog_yixun/exp_46_c4flip_data_curve_claude/exp46_f100_delta_manifest.txt.gz"
RF_CONFIG = "worklog/worklog_yixun/exp_43_std_aug_claude/FLAC_AR_YAWAUG_FLIP.json"


def test_the_c4f_config_and_the_f100_manifest_are_admitted_by_exact_path(monkeypatch):
    assert reg.ARMS["c4f"]["config"] == C4F and reg.F100_DELTA_MANIFEST == MANIFEST
    assert reg.CLOSURE_EXPLICIT == (C4F, MANIFEST)
    assert all(reg._in_closure(rel) for rel in reg.CLOSURE_EXPLICIT)
    monkeypatch.setattr(reg, "CLOSURE_EXP_PREFIXES", ("zz_never_",))      # without the prefix rule: still admitted
    assert all(reg._in_closure(rel) for rel in reg.CLOSURE_EXPLICIT)
    assert not reg._in_closure(f"{reg.EXP_DIR}/exp46_registry.py")
    monkeypatch.undo()
    for rel in (f"{reg.EXP_DIR}/c4flip_data_curve_worklog.md", f"{reg.EXP_DIR}/sub/exp46_x.py", RF_CONFIG,
                "src/tests/test_exp46_closure.py", f"{reg.EXP_DIR}/c4flip_data_curve_f100_delta_manifest.txt.gz"):
        assert not reg._in_closure(rel), rel
    assert RF_CONFIG not in reg.EXTERNAL_CONFIGS


def _git_tree(tmp_path, files):
    repo = tmp_path / "tree"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for rel, data in files.items():
        os.makedirs(repo / os.path.dirname(rel), exist_ok=True)
        (repo / rel).write_bytes(data)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)            # ls-files reads the index
    return repo


def test_a_tree_lists_and_hashes_both_and_one_changed_byte_changes_the_digest(tmp_path):
    files = {C4F: open(os.path.join(REPO, C4F), "rb").read(), MANIFEST: b"\x1f\x8b synthetic manifest bytes",
             f"{reg.EXP_DIR}/c4flip_data_curve_note.md": b"prose", f"{reg.EXP_DIR}/exp46_registry.py": b"kit code",
             "src/x.py": b"src", RF_CONFIG: b"the RF config"}
    repo = _git_tree(tmp_path, files)
    listed = reg.closure_files(str(repo))
    assert {C4F, MANIFEST, f"{reg.EXP_DIR}/exp46_registry.py", "src/x.py"} <= set(listed)
    assert f"{reg.EXP_DIR}/c4flip_data_curve_note.md" not in listed and RF_CONFIG not in listed
    d0 = reg.closure_digest(str(repo))
    for rel in (C4F, MANIFEST):
        orig = (repo / rel).read_bytes()
        (repo / rel).write_bytes(orig[:-1] + bytes([orig[-1] ^ 1]))
        assert reg.closure_digest(str(repo)) != d0, rel
        (repo / rel).write_bytes(orig)
        assert reg.closure_digest(str(repo)) == d0
    (repo / RF_CONFIG).write_bytes(b"another RF config")                          # not a runtime input: no effect
    assert reg.closure_digest(str(repo)) == d0


def test_the_real_snapshot_carries_the_c4f_config_and_the_kit_admits_it_from_the_isolated_root(tmp_path):
    run = reg.required_runs()[0]
    files = reg.snapshot_files(run)
    if C4F not in files:
        pytest.skip("the C4F config is not committed yet (the closure lists tracked files only)")
    assert RF_CONFIG not in files
    snap = tmp_path / "snap"
    out = reg.snapshot_tools(run, str(snap))
    manifest = dict(reversed(ln.split("  ", 1)) for ln in (snap / "SNAPSHOT.sha256").read_text().strip().split("\n"))
    assert manifest[C4F] == reg.ARMS["c4f"]["config_sha256"] == reg.sha256_file(str(snap / C4F))
    assert reg.closure_digest(str(snap)) == reg.closure_digest() and reg.verify_snapshot(str(snap), out["snapshot_sha256"])
    # the kit run FROM the isolated snapshot root (its own REPO by the SNAPSHOT.sha256 marker) admits the config there
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); import exp46_registry as r; "
            "cfg = r.check_config('c4f', '2x32'); print(json.dumps([r.REPO, cfg['training']['yaw_aug']]))")
    res = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(snap / reg.EXP_DIR)], capture_output=True,
                         text=True, cwd=str(tmp_path))
    assert res.returncode == 0, res.stderr[-2000:]
    root, block = json.loads(res.stdout.strip().splitlines()[-1])
    assert os.path.realpath(root) == os.path.realpath(str(snap)) and block == reg.C4F_YAW_AUG
    cfg = snap / C4F
    cfg.write_bytes(cfg.read_bytes() + b" ")
    with pytest.raises(reg.GateError):
        reg.verify_snapshot(str(snap))
    with pytest.raises(reg.GateError):
        reg.check_config("c4f", "2x32", repo=str(snap))
