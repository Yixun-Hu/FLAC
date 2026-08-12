"""Tests for the model-comparison generator's exp_11 validation gate (review B5/B7).

Round-4 review: the validator was advisory — the table generator globbed raw
JSONs and aggregated whatever it found, so an unvalidated (or unprovable) exp_11
row could reach `model_comparison.md`. It is now a gate: an exp_11 row whose cell
fails `exp11_validate_rows` is refused and rendered as blocked, never as numbers.

The second half of the review's concern is disclosure (B7): rows evaluated under
the legacy per-angle loop and rows evaluated under the batched orbit are not
interchangeable, so the table must say which each row is instead of labelling
everything "fa eval".
"""
import importlib.util
import json
import os

import pytest


_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # src/tests/ -> src/ -> repo root
_GEN_PY = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "gen_model_comparison.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_model_comparison", _GEN_PY)
    assert spec is not None and spec.loader is not None, f"cannot load {_GEN_PY}"
    mod = importlib.util.module_from_spec(spec)
    # the module writes the table at import time only under __main__
    spec.loader.exec_module(mod)
    return mod


G = _load_module()


def _row_spec(label, proto, k, pats, **extra):
    return (label, proto, k, pats, extra) if extra else (label, proto, k, pats)


# --------------------------------------------------------------------------- #
# 1. exp_11 rows are recognised and carry their execution label
# --------------------------------------------------------------------------- #
def test_importing_the_generator_does_not_rewrite_the_table():
    """Re-review item 6: at module scope the write ran on IMPORT, so merely
    running pytest could regenerate the frozen table."""
    table = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "model_comparison.md")
    before = open(table, "rb").read()
    _load_module()                      # a second import, deliberately
    assert open(table, "rb").read() == before, "importing the generator rewrote the table"
    assert hasattr(G, "main"), "table generation must live behind main()"


def test_exp11_rows_are_detected_from_their_patterns():
    assert G.is_exp11_row(["outputs_FLAC/exp11_C8/**/*exp11_C8_conf_S40000*.json"])
    assert not G.is_exp11_row(["outputs_FLAC/exp07_BF/**/*exp10_BF40_K8*.json"])
    assert not G.is_exp11_row(["weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K1_seed42.json"])


def test_every_registered_fa_row_will_declare_its_orbit_execution():
    """B7 via the deferred migration: the LABEL is applied at render time, so the
    contract is that protocol_label() discloses loop-vs-batched for every fa row
    once the migration runs (see the deferral test below for why it waits)."""
    for row in G.ROWS:
        label, proto, _K, pats = row[0], row[1], row[2], row[3]
        if "fa" not in proto:
            continue
        migrated = G.protocol_label(proto, G.is_batched_orbit_row(pats),
                                    evidence_ready=True)
        assert ("legacy-loop" in migrated) or ("batched" in migrated), (
            f"row {label!r} would render as {migrated!r} — it must disclose loop vs batched")


def test_historical_fa_rows_migrate_to_legacy_loop():
    for row in G.ROWS:
        proto, pats = row[1], row[3]
        # is_batched_orbit_row is what main() actually passes; using is_exp11_row
        # here would let an exp_14 row "pass" this test under a label the
        # generator never renders for it.
        if "fa" in proto and not G.is_batched_orbit_row(pats):
            assert "legacy-loop" in G.protocol_label(proto, False, evidence_ready=True)


def row_pats(label):
    for row in G.ROWS:
        if row[0] == label:
            return row[3]
    return []


# --------------------------------------------------------------------------- #
# 2. the gate itself
# --------------------------------------------------------------------------- #
def _write_valid_cell(tmp_path, arm="C8", step=40000, k=8, seeds=(42, 43, 44, 45, 46),
                      cell="conf", source_sha="d" * 40):
    """Five rows that pass exp11_validate_rows (sidecars included).

    ``cell`` selects the registered namespace (conf, or q9 for the Q9 round) and
    ``source_sha`` lets a caller build a round that spans two evaluator pins.
    A vanilla arm gets the vanilla schema: no orbit in the filename, and both
    orbit-provenance fields explicitly null."""
    spec = importlib.util.spec_from_file_location(
        "v", os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude",
                          "exp11_validate_rows.py"))
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    vanilla = V.is_vanilla_arm(arm)
    n_ang = V.ARM_ORBITS[arm]
    angles = None if vanilla else V.orbit_for(arm)
    # a real (tiny) file so --verify-hashes can recompute, inside the arm's
    # canonical run directory so the containment check is exercised too
    ck_dir = tmp_path / "outputs_FLAC" / f"exp11_{arm}" / f"FLAC_exp11_{arm}" / f"exp11_{arm}" / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ck_file = ck_dir / f"epoch=8-step={step}.ckpt"
    ck_file.write_bytes(b"synthetic checkpoint")
    ck = str(ck_file)
    cfg_file = tmp_path / (f"FLAC_AR_VANCKPT.json" if vanilla else f"FLAC_AR_BF_{arm}.json")
    cfg_file.write_text('{"training": {}}' if vanilla
                        else '{"training": {"cond_method": "fa_invariant"}}')
    import hashlib
    ck_sha = hashlib.sha256(ck_file.read_bytes()).hexdigest()
    cfg_sha = hashlib.sha256(cfg_file.read_bytes()).hexdigest()
    # the FULL emission set eval_FLAC writes (pinned in the validator)
    metrics = {k: 12.0 for k in V.EMITTED_METRIC_KEYS}
    paths = []
    for seed in seeds:
        ev = f"exp11_{arm}_{cell}_S{step}_s{seed}_K{k}"
        suffix = "" if vanilla else f"_fa_invariant_a{n_ang}"
        name = f"epoch=8-step={step}_metrics_1_1.0_{ev}{suffix}.json"
        rec = {"metrics": metrics, "ckpt_path": ck, "rotate_deg": 0.0,
               "cond_method": "vanilla" if vanilla else "fa_invariant",
               "frame_avg_angles": angles,
               "cond_autocast": "bf16",
               "orbit_execution": "n/a" if vanilla else "batched",
               "frame_avg_fwd_cap": None if vanilla else 64,
               "source_sha": source_sha, "batch_size": 64,
               "n_samples": 6337,
               "dataset_config": V.EVAL_CONFIG_FOR_K[k], "seed": seed, "cfg_scale": 1.0,
               "steps": 1, "eval_name": ev, "weights_source": "ema", "device": "cuda"}
        side = {"arm": arm, "step": step, "seed": seed, "K": k, "eval_name": ev,
                "cfg_scale": 1.0, "steps": 1,
                "model_config": str(cfg_file), "model_config_sha256": cfg_sha,
                "dataset_config": V.EVAL_CONFIG_FOR_K[k],
                "ckpt_path": ck, "ckpt_sha256": ck_sha, "use_ema": True,
                "frame_avg_angles": angles,
                "cond_method": "vanilla" if vanilla else "fa_invariant",
                "cond_autocast": "bf16", "commit": source_sha}
        p = tmp_path / name
        p.write_text(json.dumps(rec))
        (tmp_path / (name + ".screenmeta.json")).write_text(json.dumps(side))
        paths.append(str(p))
    return paths


def test_gate_passes_a_validated_exp11_cell(tmp_path):
    paths = _write_valid_cell(tmp_path)
    ok, problems = G.validate_exp11_cell(paths, repo_root=str(tmp_path))
    assert ok, problems


def test_gate_refuses_a_cell_with_a_missing_sidecar(tmp_path):
    paths = _write_valid_cell(tmp_path)
    os.remove(paths[0] + ".screenmeta.json")
    ok, problems = G.validate_exp11_cell(paths, repo_root=str(tmp_path))
    assert not ok and any("sidecar" in p for p in problems)


def test_gate_refuses_a_legacy_loop_row_in_an_exp11_cell(tmp_path):
    paths = _write_valid_cell(tmp_path)
    rec = json.load(open(paths[0]))
    rec["orbit_execution"] = "loop"
    open(paths[0], "w").write(json.dumps(rec))
    ok, problems = G.validate_exp11_cell(paths, repo_root=str(tmp_path))
    assert not ok and any("orbit_execution" in p for p in problems)


def test_gate_refuses_a_four_seed_table_cell(tmp_path):
    paths = _write_valid_cell(tmp_path, seeds=(42, 43, 44, 45))
    ok, problems = G.validate_exp11_cell(paths, repo_root=str(tmp_path))
    assert not ok and any("46" in p for p in problems)


def test_gate_refuses_a_screen_cell_as_a_table_row(tmp_path):
    """A single-seed futility screen must never become a table row."""
    paths = _write_valid_cell(tmp_path, seeds=(42,))
    ok, problems = G.validate_exp11_cell(paths, repo_root=str(tmp_path))
    assert not ok


def test_gate_reports_empty_cells_as_not_blocking():
    """No files yet is 'pending', which the generator already renders — the gate
    must not turn an empty cell into a failure."""
    ok, problems = G.validate_exp11_cell([])
    assert ok and problems == []


def test_render_blocks_an_invalid_exp11_row(tmp_path):
    paths = _write_valid_cell(tmp_path)
    rec = json.load(open(paths[0]))
    rec["n_samples"] = 64                        # a partial-split row
    open(paths[0], "w").write(json.dumps(rec))
    line, blocked = G.render_row("C8 @40k", "fa eval (batched)", 8, paths, repo_root=str(tmp_path))
    assert blocked
    assert "BLOCKED" in line and "12.000" not in line, line


def test_render_emits_numbers_for_a_valid_row(tmp_path):
    paths = _write_valid_cell(tmp_path)
    line, blocked = G.render_row("C8 @40k", "fa eval (batched)", 8, paths, repo_root=str(tmp_path))
    assert not blocked and "12.000" in line


# --------------------------------------------------------------------------- #
# re-review items 3, 4b, 7
# --------------------------------------------------------------------------- #
def test_table_validation_recomputes_hashes(tmp_path):
    """Item 3: the generator's gate must not trust the sidecar's own hashes."""
    paths = _write_valid_cell(tmp_path)
    ok, problems = G.validate_exp11_cell(paths, repo_root=str(tmp_path))
    assert ok, problems
    side = json.load(open(paths[0] + ".screenmeta.json"))
    side["model_config_sha256"] = "e" * 64             # tampered
    json.dump(side, open(paths[0] + ".screenmeta.json", "w"))
    ok, problems = G.validate_exp11_cell(paths, repo_root=str(tmp_path))
    assert not ok and any("model_config_sha256" in p for p in problems), problems


def test_two_k_outer_gate(tmp_path):
    """Item 4b: a table update must carry BOTH K cells of an exp_11 row."""
    assert G.check_two_k_coverage({("C8 @40k", 1): True, ("C8 @40k", 8): True}) == []
    missing = G.check_two_k_coverage({("C8 @40k", 8): True})
    assert missing and "C8 @40k" in missing[0] and "K=1" in missing[0]
    # a blocked half also breaks the transaction
    half_blocked = G.check_two_k_coverage({("C8 @40k", 1): False, ("C8 @40k", 8): True})
    assert half_blocked and "C8 @40k" in half_blocked[0]


def test_label_migration_is_deferred_until_the_exp10_evidence_returns():
    """Item 7: relabelling rewrites the table, and regenerating right now would
    replace published exp_10 numbers with 'pending' because their JSONs are off
    this machine. So the labels migrate ONLY when that evidence is present."""
    assert G.exp10_evidence_present() in (True, False)
    labels_now = G.protocol_label("fa eval", exp11=False, evidence_ready=False)
    assert labels_now == "fa eval"                     # unchanged while deferred
    labels_after = G.protocol_label("fa eval", exp11=False, evidence_ready=True)
    assert labels_after == "fa eval (legacy-loop)"
    assert G.protocol_label("fa eval", exp11=True, evidence_ready=False) == "fa eval (batched)"


def test_deferred_migration_emits_a_loud_header_note():
    header = G.build_header(evidence_ready=False)
    joined = "\n".join(header)
    assert "DEFERRED" in joined.upper()
    assert "legacy-loop" in joined and "exp_10" in joined
    ready = "\n".join(G.build_header(evidence_ready=True))
    assert "DEFERRED" not in ready.upper()



# --------------------------------------------------------------------------- #
# 5. the three GO-check table residuals
# --------------------------------------------------------------------------- #
def _fake_main_tree(tmp_path, rows):
    """A minimal MAIN tree: the generator's own file layout plus a row spec."""
    root = tmp_path / "maintree"
    (root / ".git").mkdir(parents=True)      # the validator locates its root by .git
    (root / "worklog" / "worklog_yixun" / "exp_11_fa_orbit_claude").mkdir(parents=True)
    (root / "worklog" / "worklog_yixun" / "model_comparison.md").write_text("PREVIOUS\n")
    # the generator resolves its validator against the root as well
    import shutil
    shutil.copy(G.EXP11_VALIDATOR,
                root / "worklog" / "worklog_yixun" / "exp_11_fa_orbit_claude"
                     / "exp11_validate_rows.py")
    return root


def test_generator_resolves_everything_against_an_explicit_repo_root(tmp_path, monkeypatch):
    """Residual 3: measurement runs from PINNED WORKTREES, so the cwd is routinely
    not the main tree. Globbing evidence relative to an ambient cwd would render
    published rows as *pending* (or read a worktree's stale copy)."""
    root = _fake_main_tree(tmp_path, G.ROWS)
    elsewhere = tmp_path / "a_pinned_worktree"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)                  # <- the whole point
    paths = G.repo_paths(str(root))
    assert paths["root"] == os.path.realpath(str(root))
    assert paths["out"] == os.path.join(os.path.realpath(str(root)), "worklog",
                                        "worklog_yixun", "model_comparison.md")
    assert paths["validator"].startswith(os.path.realpath(str(root)))
    rc = G.main(["--repo-root", str(root)])
    assert rc == 0
    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    assert "Model comparison" in written, "the table was not written into the given root"
    assert "PREVIOUS" not in written
    # and nothing was written relative to the cwd
    assert not (elsewhere / "worklog").exists(), "the generator wrote relative to the cwd"


def test_generated_table_actually_carries_the_deferral_note(tmp_path, monkeypatch):
    """Residual 2: build_header() existed but main() re-inlined its own header, so
    the LABEL MIGRATION DEFERRED note — the only thing telling a reader which fa
    rows are legacy-loop — never reached the file."""
    root = _fake_main_tree(tmp_path, G.ROWS)
    monkeypatch.setattr(G, "exp10_evidence_present", lambda *a, **k: False)
    assert G.main(["--repo-root", str(root)]) == 0
    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    assert "LABEL MIGRATION DEFERRED" in written
    for line in G.build_header(evidence_ready=False):
        assert line.strip() == "" or line in written, f"header line missing: {line[:60]}"


def test_two_k_gate_aborts_the_write_and_never_publishes_a_lone_k(tmp_path, monkeypatch):
    """Residual 1: the two-K check was advisory — it appended a footnote while the
    lone K row published its numbers anyway."""
    root = _fake_main_tree(tmp_path, G.ROWS)
    out = root / "worklog" / "worklog_yixun" / "model_comparison.md"
    before = out.read_text()
    # one exp_11 label, K=8 only
    monkeypatch.setattr(G, "ROWS", [
        ("C8 @40k", "fa eval", 8, ["outputs_FLAC/exp11_C8/**/*exp11_C8_conf_S40000*.json"]),
    ])
    monkeypatch.setattr(G, "render_row",
                        lambda label, proto, K, files, **kw: (
                            f"| {label} | {proto} | {K} | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |",
                            False))
    monkeypatch.setattr(G, "check_two_k_coverage",
                        lambda status: ["C8 @40k: K=1 cell is absent — a table update must carry both K"])
    rc = G.main(["--repo-root", str(root)])
    assert rc != 0, "an incomplete two-K update must fail, not warn"
    assert out.read_text() == before, "the table was rewritten during a failed transaction"
    # with the explicit override the row still may not carry numbers
    assert G.main(["--repo-root", str(root), "--allow-partial-exp11"]) == 0
    written = out.read_text()
    assert "WITHHELD" in written
    assert "| C8 @40k | fa eval | 8 | 5 | 1.0 |" not in written


def test_two_k_gate_lets_a_complete_pair_through(tmp_path, monkeypatch):
    root = _fake_main_tree(tmp_path, G.ROWS)
    monkeypatch.setattr(G, "ROWS", [
        ("C8 @40k", "fa eval", 1, ["outputs_FLAC/exp11_C8/**/*K1*.json"]),
        ("C8 @40k", "fa eval", 8, ["outputs_FLAC/exp11_C8/**/*K8*.json"]),
    ])
    monkeypatch.setattr(G, "render_row",
                        lambda label, proto, K, files, **kw: (
                            f"| {label} | {proto} | {K} | 5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |",
                            False))
    assert G.main(["--repo-root", str(root)]) == 0
    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    assert "WITHHELD" not in written
    assert written.count("| C8 @40k | fa eval") == 2


# --------------------------------------------------------------------------- #
# 6. GO-recheck: the CLI is real, and paths resolve to the ROOT not the cwd
# --------------------------------------------------------------------------- #
def _tree_with_exp11_evidence(tmp_path, k=8, arm="C8", step=40000, seeds=(42, 43, 44, 45, 46)):
    """A main tree carrying a full, VALID exp_11 cell — so the generator has to
    do real work (glob, validate, hash) rather than render 'pending' twice."""
    import shutil
    root = tmp_path / "maintree"
    ev_dir = root / "outputs_FLAC" / f"exp11_{arm}"
    ev_dir.mkdir(parents=True)
    (root / ".git").mkdir()                  # ditto: a main tree is a git repo
    # ...and carries the importable repo surface the validator reaches for:
    # src.* (orbit constants) and eval_FLAC.build_output_paths (filename proof)
    (root / "src").symlink_to(os.path.join(_REPO_ROOT, "src"))
    (root / "eval_FLAC.py").symlink_to(os.path.join(_REPO_ROOT, "eval_FLAC.py"))
    (root / "worklog" / "worklog_yixun" / "exp_11_fa_orbit_claude").mkdir(parents=True)
    (root / "worklog" / "worklog_yixun" / "model_comparison.md").write_text("PREVIOUS\n")
    shutil.copy(G.EXP11_VALIDATOR,
                root / "worklog" / "worklog_yixun" / "exp_11_fa_orbit_claude"
                     / "exp11_validate_rows.py")
    paths = _write_valid_cell(ev_dir, arm=arm, step=step, k=k, seeds=seeds)
    return root, paths


def test_repo_root_is_reachable_from_the_command_line(tmp_path):
    """The reviewer's find: main() parsed [] instead of sys.argv, so --repo-root
    parsed and was then thrown away — every CLI run silently used the default
    root. Invoke the generator as a PROCESS, from a foreign cwd."""
    import subprocess, sys
    root, _ = _tree_with_exp11_evidence(tmp_path)
    foreign = tmp_path / "pinned_worktree"
    foreign.mkdir()
    out_file = root / "worklog" / "worklog_yixun" / "model_comparison.md"
    before = out_file.read_text()
    proc = subprocess.run([sys.executable, _GEN_PY, "--repo-root", str(root)],
                          cwd=str(foreign), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    written = out_file.read_text()
    assert written != before, "--repo-root was ignored: the target table was not rewritten"
    assert "Model comparison" in written
    assert str(root) in proc.stdout, proc.stdout
    # the default root (this real repo) must NOT have been touched
    assert not (foreign / "worklog").exists()


def test_repo_root_actually_finds_evidence_from_a_foreign_cwd(tmp_path, monkeypatch):
    """The evidence globs, the validator and the hash recomputation must all
    resolve against --repo-root. Registers a row pointing at the synthetic cell
    and asserts real NUMBERS come out — 'pending' would mean nothing resolved."""
    import subprocess, sys, json as _json
    root, paths = _tree_with_exp11_evidence(tmp_path)
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    # a two-K registration so the transactional gate is satisfied
    rows_py = tmp_path / "rows.py"
    driver = f'''
import importlib.util, sys, os
spec = importlib.util.spec_from_file_location("gen", {_GEN_PY!r})
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
G.ROWS = [("C8 @40k (synthetic)", "fa eval", 8,
           ["outputs_FLAC/exp11_C8/*_K8_*fa_invariant_a8.json"]),
          ("C8 @40k (synthetic)", "fa eval", 1,
           ["outputs_FLAC/exp11_C8/*_K1_*fa_invariant_a8.json"])]
sys.exit(G.main(["--repo-root", {str(root)!r}, "--allow-partial-exp11"]))
'''
    rows_py.write_text(driver)
    # K=1 half of the pair, so both rows have evidence
    _write_valid_cell(root / "outputs_FLAC" / "exp11_C8", arm="C8", step=40000, k=1)
    proc = subprocess.run([sys.executable, str(rows_py)], cwd=str(foreign),
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    rows = [ln for ln in written.splitlines() if ln.startswith("| C8 @40k (synthetic)")]
    assert len(rows) == 2, f"expected both K rows, got {rows}"
    for row in rows:   # ("BLOCKED" also appears in the header prose, so check ROWS)
        assert "12.000" in row, f"evidence did not resolve against --repo-root: {row}"
        assert "BLOCKED" not in row and "WITHHELD" not in row and "pending" not in row, row


# --------------------------------------------------------------------------- #
# 7. GO-recheck 3: regeneration must never destroy published evidence
# --------------------------------------------------------------------------- #
_HDR = ("| Model | eval | K | n | T60 ↓ | C50 ↓ | EDT ↓ | R@1 ↑ | R@5 ↑ | R@10 ↑ |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n")


def _published(root, *rows):
    (root / "worklog" / "worklog_yixun" / "model_comparison.md").write_text(
        "# Model comparison — cross-experiment results table\n\n" + _HDR + "".join(rows))


def _numeric(label, k, n=5, v="9.969 ± 0.039"):
    return f"| {label} | vanilla eval | {k} | {n} | {v} | {v} | {v} | {v} | {v} | {v} |\n"


def _pending(label, k, n=0):
    return f"| {label} | vanilla eval | {k} | {n} | *pending ({n}/5 seeds on disk)* | | | | | |\n"


def _render_as(mapping):
    """A render_row stub driven by {(label, K): markdown_line}."""
    def _stub(label, proto, K, files, **kw):
        return mapping[(label, K)], False
    return _stub


def test_parse_table_rows_reads_the_published_table():
    rows = G.parse_table_rows(_HDR + _numeric("A row", 1) + _pending("A row", 8, n=4))
    assert rows[("A row", 1)] == {"n": 5, "numeric": True}
    assert rows[("A row", 8)] == {"n": 4, "numeric": False}


def test_numbers_turning_into_pending_aborts_without_writing(tmp_path, monkeypatch):
    """The exact accident this guard exists for: a run with evidence off-disk
    rewrites published numbers as 'pending (0/5 seeds on disk)'."""
    root = _fake_main_tree(tmp_path, G.ROWS)
    out = root / "worklog" / "worklog_yixun" / "model_comparison.md"
    _published(root, _numeric("Anchor row", 1), _numeric("Anchor row", 8))
    before = out.read_text()
    monkeypatch.setattr(G, "ROWS", [("Anchor row", "vanilla eval", 1, ["nowhere/*.json"]),
                                    ("Anchor row", "vanilla eval", 8, ["nowhere/*.json"])])
    monkeypatch.setattr(G, "render_row", _render_as({("Anchor row", 1): _pending("Anchor row", 1).rstrip(),
                                                     ("Anchor row", 8): _pending("Anchor row", 8).rstrip()}))
    assert G.main(["--repo-root", str(root)]) != 0
    assert out.read_text() == before, "a regressing regeneration still wrote the table"


def test_seed_count_regression_aborts(tmp_path, monkeypatch):
    """4/5 -> 0/5 is a regression even though neither cell carries numbers."""
    root = _fake_main_tree(tmp_path, G.ROWS)
    out = root / "worklog" / "worklog_yixun" / "model_comparison.md"
    _published(root, _pending("Half row", 8, n=4))
    before = out.read_text()
    monkeypatch.setattr(G, "ROWS", [("Half row", "vanilla eval", 8, ["nowhere/*.json"])])
    monkeypatch.setattr(G, "render_row", _render_as({("Half row", 8): _pending("Half row", 8, n=0).rstrip()}))
    assert G.main(["--repo-root", str(root)]) != 0
    assert out.read_text() == before


def test_a_dropped_row_is_a_regression(tmp_path, monkeypatch):
    root = _fake_main_tree(tmp_path, G.ROWS)
    out = root / "worklog" / "worklog_yixun" / "model_comparison.md"
    _published(root, _numeric("Anchor row", 1), _numeric("Gone row", 8))
    before = out.read_text()
    monkeypatch.setattr(G, "ROWS", [("Anchor row", "vanilla eval", 1, ["nowhere/*.json"])])
    monkeypatch.setattr(G, "render_row", _render_as({("Anchor row", 1): _numeric("Anchor row", 1).rstrip()}))
    assert G.main(["--repo-root", str(root)]) != 0
    assert out.read_text() == before


def test_the_flag_allows_an_audited_retraction(tmp_path, monkeypatch, capsys):
    root = _fake_main_tree(tmp_path, G.ROWS)
    out = root / "worklog" / "worklog_yixun" / "model_comparison.md"
    _published(root, _numeric("Anchor row", 1))
    monkeypatch.setattr(G, "ROWS", [("Anchor row", "vanilla eval", 1, ["nowhere/*.json"])])
    monkeypatch.setattr(G, "render_row", _render_as({("Anchor row", 1): _pending("Anchor row", 1).rstrip()}))
    assert G.main(["--repo-root", str(root), "--allow-row-regression"]) == 0
    written = out.read_text()
    assert "pending" in written
    # audited: the affected row is named, in the file and on stderr
    assert "REGRESSED" in written and "Anchor row (K=1)" in written
    assert "Anchor row (K=1)" in capsys.readouterr().err


def test_growing_and_fresh_rows_are_not_regressions(tmp_path, monkeypatch):
    """Adding rows, and adding seeds to existing ones, must stay unremarkable."""
    root = _fake_main_tree(tmp_path, G.ROWS)
    out = root / "worklog" / "worklog_yixun" / "model_comparison.md"
    _published(root, _pending("Growing row", 1, n=4))
    monkeypatch.setattr(G, "ROWS", [("Growing row", "vanilla eval", 1, ["nowhere/*.json"]),
                                    ("Brand new row", "vanilla eval", 8, ["nowhere/*.json"])])
    monkeypatch.setattr(G, "render_row", _render_as({
        ("Growing row", 1): _numeric("Growing row", 1, n=5).rstrip(),
        ("Brand new row", 8): _pending("Brand new row", 8).rstrip()}))
    assert G.main(["--repo-root", str(root)]) == 0
    written = out.read_text()
    assert "REGRESSED" not in written
    assert "Brand new row" in written


# --------------------------------------------------------------------------- #
# 8. VANL (Q9) — the vanilla arm of this recipe, labelled apart from legacy rows
# --------------------------------------------------------------------------- #
def _exp11_vanl_rows():
    """The exp_11 (Q9) VANL rows. Scoped by contract, not by the substring "VANL":
    exp_14's round-3 rows carry the same arm name and would otherwise be swept in."""
    return [r for r in G.ROWS if "VANL" in r[0] and len(r) > 4 and r[4] == "q9"]


def test_vanl_rows_are_registered_as_a_two_k_pair():
    vanl = _exp11_vanl_rows()
    assert len(vanl) == 2, f"expected a K=1/K=8 pair, got {vanl}"
    assert {r[2] for r in vanl} == {1, 8}
    for label, proto, _k, pats in [r[:4] for r in vanl]:
        assert proto == "vanilla eval (batched-era)", proto
        assert all("exp11_VANL" in p for p in pats), pats


def test_vanl_is_labelled_apart_from_legacy_vanilla_rows():
    """VANL vs C4L is frame averaging alone; VANL vs a legacy vanilla row still
    carries the recipe/environment shift the C4L bridge measured. The table must
    not let the two comparisons look alike."""
    protos = {r[1] for r in G.ROWS}
    assert "vanilla eval (batched-era)" in protos
    assert "vanilla eval" in protos                       # the legacy rows keep theirs
    legacy = [r for r in G.ROWS if r[1] == "vanilla eval"]
    assert legacy and all("VANL" not in r[0] for r in legacy)


def test_vanl_rows_are_exp11_rows_and_therefore_gated():
    """They must go through the exp_11 validator like every other exp_11 row."""
    vanl = _exp11_vanl_rows()
    for _label, _proto, _k, pats in [r[:4] for r in vanl]:
        assert G.is_exp11_row(pats), pats


def test_vanl_renders_pending_until_the_conf_block_lands(tmp_path):
    root = _fake_main_tree(tmp_path, G.ROWS)
    vanl = _exp11_vanl_rows()[0]
    line, blocked = G.render_row(vanl[0], vanl[1], vanl[2], [], repo_root=str(root))
    assert not blocked
    assert "pending (0/5 seeds on disk)" in line


def _probe_name_for(pat):
    """A concrete filename that the recursive glob ``pat`` should match."""
    body = pat.replace("**/", "sub/").replace("[2-6]", "2")
    return body.replace("*", "epoch=8-step=40000_metrics_1_1.0_", 1).replace("*", "")


def test_no_exp11_row_glob_also_matches_its_sidecar(tmp_path):
    """A trailing '*.json' matches '<name>.json.screenmeta.json' too, which hands
    the validator ten files for a five-seed cell and BLOCKS the row on a glob bug.

    The earlier version of this test built its probe by stripping '**/', so the
    probe did not match the metric glob either and the sidecar's non-match proved
    nothing. It now writes REAL files and uses glob.glob(recursive=True): first
    prove the metric side matches, then prove the sidecar beside it does not."""
    import glob as _glob
    checked = 0
    for label, _proto, _k, pats in [r[:4] for r in G.ROWS]:
        if not G.is_exp11_row(pats):
            continue
        for pat in pats:
            name = _probe_name_for(pat)
            metric = tmp_path / name
            metric.parent.mkdir(parents=True, exist_ok=True)
            metric.write_text("{}")
            sidecar = tmp_path / (name + ".screenmeta.json")
            sidecar.write_text("{}")
            hits = _glob.glob(os.path.join(str(tmp_path), pat), recursive=True)
            assert str(metric) in hits, (
                f"{label}: the probe {name!r} does not match its own glob {pat!r} — "
                "the test would be vacuous")
            assert str(sidecar) not in hits, (
                f"{label}: glob {pat!r} also matches its sidecar")
            checked += 1
    assert checked >= 4, f"only {checked} exp_11 globs exercised"


def test_vanl_and_c4l_q9_rows_share_the_q9_namespace():
    q9 = [r for r in G.ROWS if "_q9_" in r[3][0]]
    assert {r[2] for r in q9} == {1, 8}
    arms = {("VANL" if "VANL" in r[3][0] else "C4L") for r in q9}
    assert arms == {"VANL", "C4L"}, arms



# --------------------------------------------------------------------------- #
# 9. Q-pin review: POPULATED q9 cells end-to-end, and the one-pin round gate
# --------------------------------------------------------------------------- #
def _q9_cell(ev_root, arm, k, seeds=(42, 43, 44, 45, 46), sha="d" * 40):
    """Five q9 rows for one arm x K, written where the registered glob finds them."""
    return _write_valid_cell(ev_root, arm=arm, step=40000, k=k, seeds=seeds,
                             cell="q9", source_sha=sha)


def _q9_tree(tmp_path, **kw):
    import shutil
    root = tmp_path / "maintree"
    (root / ".git").mkdir(parents=True)
    (root / "src").symlink_to(os.path.join(_REPO_ROOT, "src"))
    (root / "eval_FLAC.py").symlink_to(os.path.join(_REPO_ROOT, "eval_FLAC.py"))
    (root / "worklog" / "worklog_yixun" / "exp_11_fa_orbit_claude").mkdir(parents=True)
    (root / "worklog" / "worklog_yixun" / "model_comparison.md").write_text("PREVIOUS\n")
    shutil.copy(G.EXP11_VALIDATOR,
                root / "worklog" / "worklog_yixun" / "exp_11_fa_orbit_claude"
                     / "exp11_validate_rows.py")
    for arm in ("VANL", "C4L"):
        (root / "outputs_FLAC" / f"exp11_{arm}").mkdir(parents=True)
    return root


def test_populated_q9_rows_pass_under_the_q9_contract_and_fail_under_table(tmp_path):
    """The reviewer's direct reproduction: the same five rows validate under q9
    and are inadmissible under the generator's old hard-coded 'table'."""
    root = _q9_tree(tmp_path)
    paths = _q9_cell(root / "outputs_FLAC" / "exp11_VANL", "VANL", 8)
    validator = str(root / "worklog" / "worklog_yixun" / "exp_11_fa_orbit_claude"
                    / "exp11_validate_rows.py")
    ok, problems = G.validate_exp11_cell(paths, repo_root=str(root),
                                         validator_path=validator, contract="q9")
    assert ok, problems
    ok_t, problems_t = G.validate_exp11_cell(paths, repo_root=str(root),
                                             validator_path=validator, contract="table")
    assert not ok_t and any("not admissible" in p for p in problems_t), problems_t


def test_a_complete_single_pin_q9_round_renders_numbers(tmp_path, monkeypatch):
    """End-to-end through main(): four populated cells, one pin, real numbers."""
    root = _q9_tree(tmp_path)
    for arm, sub in (("VANL", "exp11_VANL"), ("C4L", "exp11_C4L")):
        for k in (1, 8):
            _q9_cell(root / "outputs_FLAC" / sub, arm, k)
    assert G.main(["--repo-root", str(root)]) == 0
    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    q9_rows = [ln for ln in written.splitlines()
               if ln.startswith("| fa-recipe vanilla VANL") or ln.startswith("| C4L @40k re-measured")]
    assert len(q9_rows) == 4, q9_rows
    for row in q9_rows:
        assert "12.000" in row, f"populated q9 row did not render numbers: {row}"
        assert "BLOCKED" not in row and "WITHHELD" not in row, row


def test_q9_round_is_withheld_when_a_comparator_is_missing(tmp_path):
    """One arm must not publish without the other: the delta is the estimand."""
    root = _q9_tree(tmp_path)
    for k in (1, 8):
        _q9_cell(root / "outputs_FLAC" / "exp11_VANL", "VANL", k)
    rc = G.main(["--repo-root", str(root)])
    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    assert "Q9 round WITHHELD" in written or rc != 0
    vanl = [ln for ln in written.splitlines() if ln.startswith("| fa-recipe vanilla VANL")]
    assert all("WITHHELD" in ln for ln in vanl), vanl


def test_q9_round_is_withheld_when_the_cells_span_two_pins(tmp_path):
    """Individually valid, jointly meaningless: a delta across two evaluator pins
    confounds frame averaging with whatever else moved between them."""
    root = _q9_tree(tmp_path)
    for k in (1, 8):
        _q9_cell(root / "outputs_FLAC" / "exp11_VANL", "VANL", k, sha="d" * 40)
        _q9_cell(root / "outputs_FLAC" / "exp11_C4L", "C4L", k, sha="e" * 40)
    G.main(["--repo-root", str(root)])
    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    assert "spans MORE THAN ONE evaluator pin" in written, written[-800:]


def test_q9_rows_declare_their_contract():
    q9 = [r for r in G.ROWS if len(r) > 4 and r[4] == "q9"]
    assert len(q9) == 4, q9
    assert {r[2] for r in q9} == {1, 8}


# --------------------------------------------------------------------------- #
# 10. exp_14 Z rows — five arms re-measured at ONE pin (plan §5.7, amended)
# --------------------------------------------------------------------------- #
# Round-3 amendment (worklog 2026-08-11 12:39): exp_11's Q9 already populated the
# VANL row, so exp_14 no longer REPLACES any exp_11 spec — it ADDS its own
# clearly-labelled Z rows and leaves exp_11's specs, rows and validator alone.
import hashlib                                                       # noqa: E402

import test_yaw_gen_collect as T                                     # noqa: E402

EXP14_LABEL = "exp_14 Z (one-pin re-measurement)"
V14 = T.V


def _exp14_rows():
    return [r for r in G.ROWS if len(r) > 4 and r[4] == "exp14z"]


def test_exp14_rows_registered_for_all_five_arms_both_k():
    rows = _exp14_rows()
    assert len(rows) == 10, rows
    arms = {r[0].split()[0] for r in rows}
    assert arms == set(V14.ARMS), arms
    for arm in V14.ARMS:
        ks = {r[2] for r in rows if r[0].startswith(arm + " ")}
        assert ks == {1, 8}, (arm, ks)


def test_exp14_rows_carry_the_exact_label():
    for label, proto, _k, pats in [r[:4] for r in _exp14_rows()]:
        assert EXP14_LABEL in label, label
        assert "exp14_" in pats[0] and "_zref_" in pats[0], pats
        assert proto in ("vanilla eval (batched-era)", "fa eval (batched)"), proto


def test_exp14_rows_are_not_gated_by_the_exp11_validator():
    """Their evidence lives UNDER outputs_FLAC/exp11_<ARM>/ (the arm's own run
    directory), so a naive 'exp11_ in the pattern' test would hand exp_14 rows to
    a validator that only understands exp_11 eval names and block every one."""
    for _label, _proto, _k, pats in [r[:4] for r in _exp14_rows()]:
        assert G.is_exp14_row(pats), pats
        assert not G.is_exp11_row(pats), pats
        assert G.is_batched_orbit_row(pats), pats


def test_exp14_fa_rows_are_labelled_batched_not_legacy_loop():
    """exp_14 evaluates through the same batched orbit as exp_11; the deferred
    legacy-loop migration must not relabel them."""
    for _label, proto, _k, pats in [r[:4] for r in _exp14_rows()]:
        migrated = G.protocol_label(proto, G.is_batched_orbit_row(pats),
                                    evidence_ready=True)
        assert "legacy-loop" not in migrated, migrated


def test_no_exp14_row_glob_also_matches_its_sidecar(tmp_path):
    import glob as _glob
    for label, _proto, _k, pats in [r[:4] for r in _exp14_rows()]:
        for pat in pats:
            name = _probe_name_for(pat)
            metric = tmp_path / name
            metric.parent.mkdir(parents=True, exist_ok=True)
            metric.write_text("{}")
            for extra in (".screenmeta.json",):
                (tmp_path / (name + extra)).write_text("{}")
            (tmp_path / (name[:-len(".json")] + ".stream.json")).write_text("{}")
            hits = _glob.glob(os.path.join(str(tmp_path), pat), recursive=True)
            assert str(metric) in hits, f"{label}: probe does not match its own glob"
            assert len(hits) == 1, f"{label}: glob {pat!r} also matched {hits}"


# --- exp_11's registered specs must come through this round untouched --------
EXP11_SPEC_DIGEST = "57820d20d7a47befb3a8a60a13893dfe91eeb3123449dd458b93c77152ea07d2"


def _exp11_spec_fingerprint():
    rows = [repr(tuple(r)) for r in G.ROWS if G.is_exp11_row(r[3])]
    return len(rows), hashlib.sha256("\n".join(rows).encode()).hexdigest()


def test_exp11_row_specs_are_byte_untouched():
    """Round-3 contract: exp_14 ADDS rows. If this digest moves, an exp_11 spec
    was edited — which this round is not allowed to do."""
    count, digest = _exp11_spec_fingerprint()
    assert (count, digest) == (EXP11_ROW_COUNT, EXP11_SPEC_DIGEST), (
        f"exp_11 row specs changed: {count} rows, digest {digest}")


EXP11_ROW_COUNT = 12


# --- the exp_14 gate itself --------------------------------------------------
def _exp14_tree(tmp_path, count=T.COUNT):
    import shutil
    root = tmp_path / "maintree"
    (root / ".git").mkdir(parents=True)
    (root / "src").symlink_to(os.path.join(_REPO_ROOT, "src"))
    (root / "eval_FLAC.py").symlink_to(os.path.join(_REPO_ROOT, "eval_FLAC.py"))
    for sub in ("exp_11_fa_orbit_claude", "exp_14_yaw_gen_claude"):
        (root / "worklog" / "worklog_yixun" / sub).mkdir(parents=True)
    (root / "worklog" / "worklog_yixun" / "model_comparison.md").write_text("PREVIOUS\n")
    shutil.copy(G.EXP11_VALIDATOR, root / "worklog" / "worklog_yixun"
                / "exp_11_fa_orbit_claude" / "exp11_validate_rows.py")
    exp14 = root / "worklog" / "worklog_yixun" / "exp_14_yaw_gen_claude"
    shutil.copy(os.path.join(_REPO_ROOT, "worklog", "worklog_yixun",
                             "exp_14_yaw_gen_claude", "exp14_validate_cell.py"),
                exp14 / "exp14_validate_cell.py")
    (exp14 / "exp14_ckpt_expect.json").write_text(json.dumps(
        {"step": V14.STEP, "arms": {a: {"sha256": s} for a, s in T.CKPT_SHA.items()}}))
    return root


def _exp14_cell_files(root, arm, k, seeds=(42, 43, 44, 45, 46), pin=T.PIN, **kw):
    out = []
    for seed in seeds:
        cell = V14.Cell(arm, "zref", V14.STEP, seed, k, None)
        out.append(T.write_cell(str(root / "outputs_FLAC"), cell, pin=pin, **kw))
    return out


@pytest.fixture()
def small_count(monkeypatch):
    """The campaign's real stream is 6,337 positions; a unit test uses 8."""
    monkeypatch.setattr(G, "EXP14_EXPECTED_COUNT", T.COUNT)
    return T.COUNT


def test_exp14_expected_count_is_the_full_published_split():
    assert G.EXP14_EXPECTED_COUNT == 6337 == V14.EXPECTED_COUNT


def test_exp14_gate_passes_a_complete_validated_cell(tmp_path, small_count):
    root = _exp14_tree(tmp_path)
    files = _exp14_cell_files(root, "C8", 8)
    ok, problems = G.validate_exp14_cell(files, repo_root=str(root))
    assert ok, problems


@pytest.mark.parametrize("patch,needle", [
    ({"weights_source": "online"}, "weights_source"),
    ({"cond_autocast": "default"}, "cond_autocast"),
    ({"batch_size": 32}, "batch_size"),
])
def test_exp14_gate_refuses_a_cell_off_protocol(tmp_path, small_count, patch, needle):
    root = _exp14_tree(tmp_path)
    files = _exp14_cell_files(root, "C8", 8)
    rec = json.load(open(files[0]))
    rec.update(patch)
    json.dump(rec, open(files[0], "w"))
    ok, problems = G.validate_exp14_cell(files, repo_root=str(root))
    assert not ok and any(needle in p for p in problems), problems


def test_exp14_gate_refuses_a_wrong_checkpoint_digest(tmp_path, small_count):
    """A table row must name WHICH checkpoint produced it (round-2 review B4)."""
    root = _exp14_tree(tmp_path)
    files = _exp14_cell_files(root, "C8", 8, ckpt_sha="0" * 64)
    ok, problems = G.validate_exp14_cell(files, repo_root=str(root))
    assert not ok and any("ckpt_sha256" in p for p in problems), problems


def test_exp14_gate_refuses_a_four_seed_cell(tmp_path, small_count):
    root = _exp14_tree(tmp_path)
    files = _exp14_cell_files(root, "C8", 8, seeds=(42, 43, 44, 45))
    ok, problems = G.validate_exp14_cell(files, repo_root=str(root))
    assert not ok and any("46" in p for p in problems), problems


def test_exp14_gate_refuses_a_rotated_cell_as_a_table_row(tmp_path, small_count):
    """Only the unrotated Z block is a comparable table row; an R cell measured a
    different estimand and would sit in the table looking like a θ=0 number."""
    root = _exp14_tree(tmp_path)
    files = [T.write_cell(str(root / "outputs_FLAC"),
                          V14.Cell("C8", "rgen", V14.STEP, s, 8, None))
             for s in (42, 43, 44, 45, 46)]
    ok, problems = G.validate_exp14_cell(files, repo_root=str(root))
    assert not ok and any("zref" in p or "unrotated" in p for p in problems), problems


def test_exp14_gate_validates_the_stream_sidecar_when_it_is_present(tmp_path,
                                                                    small_count):
    root = _exp14_tree(tmp_path)
    files = _exp14_cell_files(root, "C8", 8)
    stream_path = V14.stream_path(files[0])
    stream = json.load(open(stream_path))
    stream["input_hash"] = "e" * 64                       # no longer recomputable
    json.dump(stream, open(stream_path, "w"))
    ok, problems = G.validate_exp14_cell(files, repo_root=str(root))
    assert not ok and any("input_hash" in p for p in problems), problems


def test_exp14_gate_refuses_a_row_whose_stream_sidecar_is_absent(tmp_path,
                                                                small_count):
    """SUPERSEDED BY REVIEW B3. This test used to assert the opposite — that a
    missing assignment audit was a declared scope reduction and a numeric row
    could still publish. It cannot: without the sidecar the split order,
    substitutions, tuple count and hashes are unproved, and a later collector run
    is not coupled to publication."""
    root = _exp14_tree(tmp_path)
    files = _exp14_cell_files(root, "C8", 8)
    for path in files:
        os.remove(V14.stream_path(path))
    ok, problems = G.validate_exp14_cell(files, repo_root=str(root),
                                         expected_arm="C8", expected_k=8)
    assert not ok and any("stream" in p for p in problems), problems
    header = "\n".join(G.build_header(evidence_ready=True))
    assert "too large to commit" not in header, (
        "the header still declares the scope reduction the review rejected")


def test_exp14_round_needs_one_pin(tmp_path, small_count):
    """Two READY arms at different pins: neither publishes."""
    root = _exp14_tree(tmp_path)
    cells = {}
    for arm, pin in (("VANL", "a" * 40), ("C8", "b" * 40)):
        for k in (1, 8):
            cells[(f"{arm} @40k", k)] = _exp14_cell_files(root, arm, k, pin=pin)
    # every row validates; only the PIN differs — status is not the issue here
    status = {key: True for key in cells}
    withheld, problems = G.check_exp14_round(cells, status)
    assert problems and any("pin" in p for p in problems), problems
    assert withheld == {"VANL @40k", "C8 @40k"}, withheld


def test_exp14_round_gates_each_arm_independently(tmp_path, small_count):
    """R3F4: a complete arm publishes even while another arm is half-landed."""
    root = _exp14_tree(tmp_path)
    cells = {("VANL @40k", 1): _exp14_cell_files(root, "VANL", 1),
             ("VANL @40k", 8): _exp14_cell_files(root, "VANL", 8),
             ("C8 @40k", 8): _exp14_cell_files(root, "C8", 8),
             ("C8 @40k", 1): []}
    status = {key: bool(files) for key, files in cells.items()}
    withheld, problems = G.check_exp14_round(cells, status)
    assert "VANL @40k" not in withheld, withheld
    assert "C8 @40k" in withheld
    assert problems and any("C8" in p and "K=1" in p for p in problems), problems


def test_exp14_round_needs_both_k(tmp_path, small_count):
    root = _exp14_tree(tmp_path)
    cells = {("C8 @40k", 8): _exp14_cell_files(root, "C8", 8), ("C8 @40k", 1): []}
    withheld, problems = G.check_exp14_round(cells, {("C8 @40k", 8): True,
                                                     ("C8 @40k", 1): False})
    assert problems and any("K=1" in p for p in problems), problems
    assert withheld == {"C8 @40k"}


def test_exp14_round_is_not_a_problem_before_anything_lands():
    assert G.check_exp14_round({("C8 @40k", 8): [], ("C8 @40k", 1): []},
                               {("C8 @40k", 8): False, ("C8 @40k", 1): False}) \
        == (set(), [])


def test_exp14_rows_render_pending_until_the_block_lands(tmp_path):
    root = _exp14_tree(tmp_path)
    row = _exp14_rows()[0]
    line, blocked = G.render_row(row[0], row[1], row[2], [], repo_root=str(root),
                                 contract="exp14z")
    assert not blocked and "pending (0/5 seeds on disk)" in line


def test_populated_exp14_rows_render_numbers_end_to_end(tmp_path, small_count,
                                                        monkeypatch):
    root = _exp14_tree(tmp_path)
    for arm in V14.ARMS:
        for k in (1, 8):
            _exp14_cell_files(root, arm, k)
    assert G.main(["--repo-root", str(root)]) == 0
    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    rows = [ln for ln in written.splitlines() if EXP14_LABEL in ln and ln.startswith("|")]
    assert len(rows) == 10, rows
    for line in rows:
        assert "BLOCKED" not in line and "WITHHELD" not in line and "pending" not in line
        assert " ± " in line, line


def test_an_incomplete_exp14_round_is_withheld_never_numeric(tmp_path, small_count):
    """One K of one arm missing: the arm's rows publish as WITHHELD, and no
    number from the incomplete round reaches the table."""
    root = _exp14_tree(tmp_path)
    for arm in V14.ARMS:
        for k in (1, 8):
            if arm == "C16" and k == 1:
                continue
            _exp14_cell_files(root, arm, k)
    G.main(["--repo-root", str(root)])
    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    c16 = [ln for ln in written.splitlines()
           if ln.startswith("| C16 ") and EXP14_LABEL in ln]
    assert c16 and all("WITHHELD" in ln or "pending" in ln for ln in c16), c16


# --------------------------------------------------------------------------- #
# 11. round-3 fixes to the exp_14 table contract
# --------------------------------------------------------------------------- #
def test_exp14_gate_binds_evidence_to_the_declared_row(tmp_path, small_count):
    """R3F2 (B2): a homogeneous five-seed C16 payload renamed to match a C8 glob
    validated under the C8 label, because the expected arm/K never reached the
    gate. The row contract must bind them."""
    root = _exp14_tree(tmp_path)
    files = _exp14_cell_files(root, "C16", 8)
    ok, problems = G.validate_exp14_cell(files, repo_root=str(root),
                                         expected_arm="C8", expected_k=8)
    assert not ok and any("C16" in p and "C8" in p for p in problems), problems
    ok2, _ = G.validate_exp14_cell(files, repo_root=str(root),
                                   expected_arm="C16", expected_k=8)
    assert ok2


def test_exp14_gate_binds_the_declared_k(tmp_path, small_count):
    root = _exp14_tree(tmp_path)
    files = _exp14_cell_files(root, "C8", 1)
    ok, problems = G.validate_exp14_cell(files, repo_root=str(root),
                                         expected_arm="C8", expected_k=8)
    assert not ok and any("K" in p for p in problems), problems


def test_exp14_gate_rejects_a_duplicate_seed_set(tmp_path, small_count):
    """R3F2: six files with a duplicated seed passed (only MISSING seeds were
    checked) and agg_files then averaged all six."""
    root = _exp14_tree(tmp_path)
    files = _exp14_cell_files(root, "C8", 8)
    import shutil
    dupe = files[0].replace("_s42_", "_s42dup_")
    shutil.copy(files[0], dupe)
    ok, problems = G.validate_exp14_cell(files + [dupe], repo_root=str(root),
                                         expected_arm="C8", expected_k=8)
    assert not ok and any("5" in p or "duplicate" in p.lower() for p in problems), problems


def test_exp14_gate_requires_the_stream_audit_at_publication(tmp_path, small_count):
    """R3F3 (B3): the stream sidecar is REQUIRED — the kit passes --record-stream
    for every cell, so its absence is a missing artifact, not a scope decision."""
    root = _exp14_tree(tmp_path)
    files = _exp14_cell_files(root, "C8", 8)
    os.remove(V14.stream_path(files[0]))
    ok, problems = G.validate_exp14_cell(files, repo_root=str(root),
                                         expected_arm="C8", expected_k=8)
    assert not ok and any("stream" in p for p in problems), problems
    line, blocked = G.render_row("C8 @40k — " + EXP14_LABEL, "fa eval (batched)", 8,
                                 files, repo_root=str(root), contract="exp14z")
    assert blocked and "BLOCKED" in line


def test_exp14_gate_uses_the_top_level_validator(tmp_path, small_count):
    """R3F3: 'the same predicate' means the imported validate_cell, not a
    re-implementation that happens to call some of its parts."""
    import inspect
    src = inspect.getsource(G._exp14_cell_reasons)
    assert "validate_cell(" in src, src


def test_exp14_rows_publish_per_arm_not_per_campaign(tmp_path, small_count):
    """R3F4 (B4): §5.7 sequencing regenerates the table when VANL reaches 5/5 on
    both K. The round gate must therefore publish VANL alone, and not withhold it
    until the other four arms finish."""
    root = _exp14_tree(tmp_path)
    for k in (1, 8):
        _exp14_cell_files(root, "VANL", k)
    assert G.main(["--repo-root", str(root)]) == 0
    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    vanl = [ln for ln in written.splitlines()
            if ln.startswith("| VANL ") and EXP14_LABEL in ln]
    assert len(vanl) == 2, vanl
    for line in vanl:
        assert "WITHHELD" not in line and "pending" not in line and " ± " in line, line
    others = [ln for ln in written.splitlines()
              if EXP14_LABEL in ln and ln.startswith("|") and not ln.startswith("| VANL ")]
    assert others and all("pending" in ln for ln in others), others


def test_exp14_one_k_arm_is_still_withheld(tmp_path, small_count):
    """...but an arm with only one K is still not publishable."""
    root = _exp14_tree(tmp_path)
    _exp14_cell_files(root, "C8", 8)
    G.main(["--repo-root", str(root)])
    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    c8 = [ln for ln in written.splitlines()
          if ln.startswith("| C8 ") and EXP14_LABEL in ln]
    assert c8 and all("WITHHELD" in ln or "pending" in ln for ln in c8), c8


def test_exp14_arms_must_share_one_campaign_pin(tmp_path, small_count):
    """R3F4: independent per-arm transactions, ONE campaign pin across all of
    them — otherwise VANL-vs-Cn is a cross-pin comparison again."""
    root = _exp14_tree(tmp_path)
    for k in (1, 8):
        _exp14_cell_files(root, "VANL", k, pin="a" * 40)
        _exp14_cell_files(root, "C8", k, pin="b" * 40)
    G.main(["--repo-root", str(root)])
    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    rows = [ln for ln in written.splitlines() if EXP14_LABEL in ln and ln.startswith("|")]
    published = [ln for ln in rows if " ± " in ln]
    assert not published, f"a mixed-pin campaign published numbers: {published}"
    assert "MORE THAN ONE evaluator pin" in written


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_metric_never_reaches_the_table(tmp_path, small_count, bad):
    """R3F5 (B5): NaN must not propagate into a published cell."""
    root = _exp14_tree(tmp_path)
    files = _exp14_cell_files(root, "C8", 8)
    rec = json.load(open(files[0]))
    rec["metrics"]["T60"] = bad
    with open(files[0], "w") as fh:
        json.dump(rec, fh)
    ok, problems = G.validate_exp14_cell(files, repo_root=str(root),
                                         expected_arm="C8", expected_k=8)
    assert not ok and any("T60" in p for p in problems), problems


def test_agg_files_refuses_a_broken_payload_for_any_row_family(tmp_path):
    """R3F5: the aggregation itself must not average a NaN or raise KeyError deep
    inside a table regeneration — for exp_11 rows just as much as exp_14 ones."""
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"metrics": {k: 1.0 for _n, k in G.KEYS}}))
    assert G.agg_files([str(good)])[1] == 1
    for payload in ({"metrics": {"T60": float("nan")}},
                    {"metrics": {"T60": 1.0}}):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            G.agg_files([str(bad)])


# --------------------------------------------------------------------------- #
# 12. round-3 CLOSURE fix FX2 (finding A4/B2, REPRODUCED by the reviewer)
#
# check_exp14_round declared an arm ready from FILE COUNTS alone and never saw
# the validation result computed while rendering. Five valid K=1 files plus five
# K=8 files carrying one NaN therefore published a numeric K=1 row beside a
# BLOCKED K=8 row — the two-K transaction violated in the one direction it
# exists to prevent.
# --------------------------------------------------------------------------- #
def test_check_exp14_round_consumes_validation_status(tmp_path, small_count):
    root = _exp14_tree(tmp_path)
    cells = {("C8 @40k", 1): _exp14_cell_files(root, "C8", 1),
             ("C8 @40k", 8): _exp14_cell_files(root, "C8", 8)}
    ok_status = {("C8 @40k", 1): True, ("C8 @40k", 8): True}
    withheld, problems = G.check_exp14_round(cells, ok_status)
    assert withheld == set() and problems == []
    # ...and the same files with the K=8 row failing validation
    bad_status = {("C8 @40k", 1): True, ("C8 @40k", 8): False}
    withheld, problems = G.check_exp14_round(cells, bad_status)
    assert withheld == {"C8 @40k"}, withheld
    assert problems and any("valid" in p.lower() for p in problems), problems


def test_a_nan_in_one_k_withholds_its_partner_end_to_end(tmp_path, small_count):
    """The reviewer's exact reproduction: five valid K=1 files, five K=8 files
    with one T60=NaN. Neither row may publish numbers."""
    root = _exp14_tree(tmp_path)
    _exp14_cell_files(root, "C8", 1)
    k8 = _exp14_cell_files(root, "C8", 8)
    rec = json.load(open(k8[0]))
    rec["metrics"]["T60"] = float("nan")
    with open(k8[0], "w") as fh:
        json.dump(rec, fh)
    G.main(["--repo-root", str(root)])
    written = (root / "worklog" / "worklog_yixun" / "model_comparison.md").read_text()
    c8 = [ln for ln in written.splitlines()
          if ln.startswith("| C8 ") and EXP14_LABEL in ln]
    assert len(c8) == 2, c8
    for line in c8:
        assert " ± " not in line, f"a numeric partner leaked past the transaction: {line}"
        assert "BLOCKED" in line or "WITHHELD" in line or "pending" in line, line
