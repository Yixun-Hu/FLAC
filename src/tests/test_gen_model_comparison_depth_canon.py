"""exp_25 depth_canon rows of the comparison-table generator (announcement 04; plan v9 §5):
CANON (T block), P1-pin and YAW-pin, each K, gated by the ``exp25`` contract -- every file
must pass ``depth_canon_collect.verify_cell`` (importlib-loaded from the evidence root, like
the exp_14 validator), so single-seed / partial / invalid cells are structurally excluded.

Written by the exp_25 Coder seat (Claude Opus 5.5, max effort) per plan_depth_canon.md v9.
"""
import glob
import hashlib
import importlib.util
import json
import os
import shutil

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPREL = os.path.join("worklog", "worklog_yixun", "exp_25_depth_canon_claude")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load(os.path.join(REPO, "worklog", "worklog_yixun", "gen_model_comparison.py"), "gen_model_comparison")
T = _load(os.path.join(REPO, "src", "tests", "test_depth_canon_collect.py"), "tdc_for_gen")
EXP25 = [r for r in G.ROWS if len(r) > 4 and r[4] == "exp25"]
EXP15_ROWS = [
    ("yaw-aug vanilla YAWAUG @40k (exp_15)", "vanilla eval, theta=0 (scene-routed)", 1,
     ["outputs_FLAC/exp15_YAWAUG/**/*exp15_YAWAUG_tbl_S40000_s4[2-6]_K1.json"], "exp15"),
    ("yaw-aug vanilla YAWAUG @40k (exp_15)", "vanilla eval, theta=0 (scene-routed)", 8,
     ["outputs_FLAC/exp15_YAWAUG/**/*exp15_YAWAUG_tbl_S40000_s4[2-6]_K8.json"], "exp15")]


def test_rows_are_registered_for_three_arms_both_k():
    assert sorted((G.exp25_arm_of(r[0]), r[2]) for r in EXP25) == [
        (a, k) for a in ("CANON", "P1", "YAW") for k in (1, 8)]
    labels = {G.exp25_arm_of(r[0]): r[0] for r in EXP25}
    assert "re-measured at the exp_25 pin" in labels["YAW"] and "re-measured at the exp_25 pin" in labels["P1"]
    for r in EXP25:
        assert "fa" not in r[1] and not G.is_exp11_row(r[3]) and not G.is_exp14_row(r[3])
        assert G.protocol_label(r[1], G.is_batched_orbit_row(r[3]), True) == r[1]
    assert [r for r in G.ROWS if r[0].startswith("yaw-aug vanilla YAWAUG")] == EXP15_ROWS  # untouched


def test_globs_match_the_metrics_file_only(tmp_path):
    for r in EXP25:
        (pat,) = r[3]
        name = pat.replace("*", "epoch=8-step=40000", 1).replace("[2-6]", "3")
        metric = tmp_path / name
        metric.parent.mkdir(parents=True, exist_ok=True)
        stem = str(metric)[:-len(".json")]
        for p in [metric] + [stem + x for x in (".stream.json", ".canon.json", ".cell.json",
                                                ".json.incomplete-2026")]:
            open(p, "w").write("{}")
        assert glob.glob(os.path.join(str(tmp_path), pat)) == [str(metric)], r[0]


def _tree(tmp_path):
    """A fake evidence root: the collector + manifest (N = 8, dumps under tmp), the configs
    the cells pin, a launcher checkpoint manifest; outputs land under its own farm."""
    root = tmp_path / "maintree"
    exp = root / EXPREL
    exp.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "src").symlink_to(os.path.join(REPO, "src"))
    (root / "eval_FLAC.py").symlink_to(os.path.join(REPO, "eval_FLAC.py"))
    for f in ("depth_canon_collect.py", "FLAC_AR_CANON.json"):
        shutil.copy(os.path.join(REPO, EXPREL, f), exp / f)
    for d, f in (("exp_07_fa_scratch_claude", "FLAC_AR_BVp1.json"), ("exp_15_yaw_aug_claude", "FLAC_AR_YAWAUG.json")):
        (root / "worklog" / "worklog_yixun" / d).mkdir(parents=True)
        shutil.copy(os.path.join(REPO, "worklog", "worklog_yixun", d, f), root / "worklog" / "worklog_yixun" / d / f)
    m = json.loads(json.dumps(T.MANIFEST))
    m["dump_root"] = str(tmp_path / "dumps")
    (exp / "depth_canon_cells.json").write_text(json.dumps(m))
    T._canon_manifest_file(exp)
    (root / "worklog" / "worklog_yixun" / "model_comparison.md").write_text("PREVIOUS\n")
    return root, m


def _write_row(root, m, arm, k, seeds=(42, 43, 44, 45, 46), **kw):
    sub = {"CANON": dict(block="T"), "P1": dict(block="P1-pin", arm="P1", sub="T"),
           "YAW": dict(block="YAW-pin", arm="YAW", sub="T")}[arm]
    out = []
    for s in seeds:
        r = T.row(K=k, seed=s, **sub)
        rel = T.write_cell(root, r, dump_root=m["dump_root"], **kw)
        out.append(os.path.join(str(root), rel))
    return out


def _files(root, k, arm):
    (pat,) = next(r for r in EXP25 if G.exp25_arm_of(r[0]) == arm and r[2] == k)[3]
    return sorted(glob.glob(os.path.join(str(root), pat)))


@pytest.mark.parametrize("arm", ["CANON", "P1", "YAW"])
def test_a_complete_valid_cell_validates_and_renders_numbers(tmp_path, arm):
    root, m = _tree(tmp_path)
    written = _write_row(root, m, arm, 8)
    files = _files(root, 8, arm)
    assert files == sorted(written)
    ok, problems = G.validate_exp25_cell(files, repo_root=str(root), expected_arm=arm, expected_k=8)
    assert ok, problems
    label = next(r[0] for r in EXP25 if G.exp25_arm_of(r[0]) == arm)
    line, blocked = G.render_row(label, "vanilla eval (exp_25 pin)", 8, files, repo_root=str(root),
                                 contract="exp25")
    assert not blocked and "BLOCKED" not in line and "pending" not in line and "| 5 |" in line


def test_invalid_partial_or_misfiled_cells_are_excluded(tmp_path):
    root, m = _tree(tmp_path)
    files = _write_row(root, m, "P1", 1)
    T.edit_json(files[2], lambda d: d.update(cond_autocast="default"))      # an off-protocol seed
    ok, problems = G.validate_exp25_cell(files, repo_root=str(root), expected_arm="P1", expected_k=1)
    assert not ok and any("cond_autocast" in p for p in problems)
    os.unlink(files[2])                                                     # four valid seeds
    four = [f for f in files if os.path.exists(f)]
    assert G.validate_exp25_cell(four, repo_root=str(root), expected_arm="P1", expected_k=1)[0]
    line, blocked = G.render_row("x", "vanilla eval (exp_25 pin)", 1, four, repo_root=str(root), contract="exp25")
    assert "pending (4/5" in line and not blocked                           # never a number
    ok, problems = G.validate_exp25_cell(four, repo_root=str(root), expected_arm="YAW", expected_k=1)
    assert not ok and any("P1" in p for p in problems)                       # bound to the declared row
    ok, problems = G.validate_exp25_cell(four, repo_root=str(root), expected_arm="P1", expected_k=8)
    assert not ok


def _main(root, monkeypatch):
    monkeypatch.setattr(G, "ROWS", EXP25)
    assert G.main(["--repo-root", str(root)]) == 0
    text = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    rows = {(G.exp25_arm_of(ln.split("|")[1].strip()), int(ln.split("|")[3])): ln
            for ln in text.splitlines() if ln.startswith("| ") and "exp_25" in ln.split("|")[1]}
    return text, rows


def test_a_round_at_two_pins_publishes_nothing(tmp_path, monkeypatch):
    root, m = _tree(tmp_path)
    for arm in ("CANON", "P1", "YAW"):
        for k in (1, 8):
            _write_row(root, m, arm, k, pin="b" * 40 if arm == "YAW" else T.PIN)
    text, rows = _main(root, monkeypatch)
    assert len(rows) == 6 and all("WITHHELD" in ln for ln in rows.values()), rows
    assert "exp_25 rows WITHHELD" in text and "pin" in text


def test_an_arm_publishes_only_as_a_two_k_pair(tmp_path, monkeypatch):
    root, m = _tree(tmp_path)
    for k in (1, 8):
        _write_row(root, m, "CANON", k)
    _write_row(root, m, "P1", 1)
    text, rows = _main(root, monkeypatch)
    assert "WITHHELD" not in rows[("CANON", 1)] and "WITHHELD" not in rows[("CANON", 8)]
    assert "± " in rows[("CANON", 8)] and "WITHHELD" in rows[("P1", 1)]
    assert "pending (0/5" in rows[("YAW", 1)]



# ---- rB-F1 B6 (Codex rB HIGH; decision A7): table regeneration needs NO NAS ---- #
def _receipt(metrics_path):
    return metrics_path[:-len(".json")] + ".cell.json"


def test_validation_is_offline_from_the_committed_receipt(tmp_path):
    root, m = _tree(tmp_path)
    files = _write_row(root, m, "CANON", 1)
    shutil.rmtree(m["dump_root"])                                   # the NAS unmounted / moved / collected
    ok, problems = G.validate_exp25_cell(sorted(files), repo_root=str(root), expected_arm="CANON", expected_k=1)
    assert ok, problems
    line, blocked = G.render_row(G.EXP25_LABELS["CANON"], "vanilla eval + canonicalize (exp_25 pin)", 1,
                                 sorted(files), repo_root=str(root), contract="exp25")
    assert not blocked and "| 5 |" in line and "± " in line


@pytest.mark.parametrize("edit, needle", [
    (lambda d: d["dump_evidence"].update(marker="0" * 64 + "\n"), "marker"),
    (lambda d: d["artifact_sha256"].update(metrics="0" * 64), "artifact_sha256"),
    (lambda d: d.update(campaign_digest="e" * 64, spawn_digest="e" * 64, post_digest="e" * 64), "digest")],
    ids=["evidence", "artifact", "second_campaign_digest"])
def test_a_tampered_or_foreign_receipt_blocks_the_row(tmp_path, edit, needle):
    root, m = _tree(tmp_path)
    files = sorted(_write_row(root, m, "YAW", 8))
    T.edit_json(_receipt(files[3]), edit)
    ok, problems = G.validate_exp25_cell(files, repo_root=str(root), expected_arm="YAW", expected_k=8)
    assert not ok and any(needle in p for p in problems), problems
    line, blocked = G.render_row("x", "vanilla eval (exp_25 pin)", 8, files, repo_root=str(root), contract="exp25")
    assert blocked and "BLOCKED" in line



# ---- rB-F1 NIT c: every pre-exp25 row renders byte-identically (differential golden) ---- #
BASE_COMMIT = "e5902b9"   # gen_model_comparison.py before exp_25's additive rows (e93c42a)
GEN_PATH = os.path.join(REPO, "worklog", "worklog_yixun", "gen_model_comparison.py")


def _generator_at(commit):
    """The generator as committed at ``commit``, executed as if it lived at its real path."""
    import subprocess
    import types
    src = subprocess.run(["git", "-C", REPO, "show", f"{commit}:worklog/worklog_yixun/gen_model_comparison.py"],
                         check=True, capture_output=True, text=True).stdout
    mod = types.ModuleType(f"gen_model_comparison_at_{commit}")
    mod.__file__ = GEN_PATH
    exec(compile(src, f"{GEN_PATH}@{commit}", "exec"), mod.__dict__)
    return mod


def _render_lines(mod, out, monkeypatch):
    """The whole table over THIS checkout's committed evidence, written to ``out`` (never the
    tracked file), minus the one intentional global metadata line (the generation timestamp)."""
    real = mod.repo_paths
    monkeypatch.setattr(mod, "repo_paths", lambda r=None: dict(real(r), out=str(out)))
    assert mod.main(["--repo-root", REPO, "--allow-partial-exp11"]) == 0
    return [ln for ln in out.read_text().splitlines() if not ln.startswith("Auto-generated by")]


def test_every_pre_exp25_row_renders_byte_identically(tmp_path, monkeypatch):
    base_mod = _generator_at(BASE_COMMIT)
    base = _render_lines(base_mod, tmp_path / "base.md", monkeypatch)
    cur = _render_lines(G, tmp_path / "cur.md", monkeypatch)
    exp25 = [ln for ln in cur if any(ln.startswith(f"| {label} |") for label in G.EXP25_LABELS.values())]
    assert len(exp25) == 6                                               # the additive rows, nothing else
    rest = [ln for ln in cur if ln not in exp25]
    digest = lambda lines: hashlib.sha256("\n".join(lines).encode()).hexdigest()  # noqa: E731
    assert digest(rest) == digest(base) and rest == base
    rows = [ln for ln in base if ln.startswith("| ") and not ln.startswith("| Model")]
    assert len(rows) == len(base_mod.ROWS)                               # every pre-exp25 spec is covered
