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
        migrated = G.protocol_label(proto, G.is_exp11_row(pats), evidence_ready=True)
        assert ("legacy-loop" in migrated) or ("batched" in migrated), (
            f"row {label!r} would render as {migrated!r} — it must disclose loop vs batched")


def test_historical_fa_rows_migrate_to_legacy_loop():
    for row in G.ROWS:
        proto, pats = row[1], row[3]
        if "fa" in proto and not G.is_exp11_row(pats):
            assert "legacy-loop" in G.protocol_label(proto, False, evidence_ready=True)


def row_pats(label):
    for row in G.ROWS:
        if row[0] == label:
            return row[3]
    return []


# --------------------------------------------------------------------------- #
# 2. the gate itself
# --------------------------------------------------------------------------- #
def _write_valid_cell(tmp_path, arm="C8", step=40000, k=8, seeds=(42, 43, 44, 45, 46)):
    """Five conf rows that pass exp11_validate_rows (sidecars included)."""
    spec = importlib.util.spec_from_file_location(
        "v", os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude",
                          "exp11_validate_rows.py"))
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    n_ang = V.ARM_ORBITS[arm]
    angles = V.orbit_for(arm)
    # a real (tiny) file so --verify-hashes can recompute, inside the arm's
    # canonical run directory so the containment check is exercised too
    ck_dir = tmp_path / "outputs_FLAC" / f"exp11_{arm}" / f"FLAC_exp11_{arm}" / f"exp11_{arm}" / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ck_file = ck_dir / f"epoch=8-step={step}.ckpt"
    ck_file.write_bytes(b"synthetic checkpoint")
    ck = str(ck_file)
    cfg_file = tmp_path / f"FLAC_AR_BF_{arm}.json"
    cfg_file.write_text('{"training": {"cond_method": "fa_invariant"}}')
    import hashlib
    ck_sha = hashlib.sha256(ck_file.read_bytes()).hexdigest()
    cfg_sha = hashlib.sha256(cfg_file.read_bytes()).hexdigest()
    # the FULL emission set eval_FLAC writes (pinned in the validator)
    metrics = {k: 12.0 for k in V.EMITTED_METRIC_KEYS}
    paths = []
    for seed in seeds:
        ev = f"exp11_{arm}_conf_S{step}_s{seed}_K{k}"
        name = f"epoch=8-step={step}_metrics_1_1.0_{ev}_fa_invariant_a{n_ang}.json"
        rec = {"metrics": metrics, "ckpt_path": ck, "rotate_deg": 0.0,
               "cond_method": "fa_invariant", "frame_avg_angles": angles,
               "cond_autocast": "bf16", "orbit_execution": "batched",
               "frame_avg_fwd_cap": 64, "source_sha": "d" * 40, "batch_size": 64,
               "n_samples": 6337,
               "dataset_config": V.EVAL_CONFIG_FOR_K[k], "seed": seed, "cfg_scale": 1.0,
               "steps": 1, "eval_name": ev, "weights_source": "ema", "device": "cuda"}
        side = {"arm": arm, "step": step, "seed": seed, "K": k, "eval_name": ev,
                "cfg_scale": 1.0, "steps": 1,
                "model_config": str(cfg_file), "model_config_sha256": cfg_sha,
                "dataset_config": V.EVAL_CONFIG_FOR_K[k],
                "ckpt_path": ck, "ckpt_sha256": ck_sha, "use_ema": True,
                "frame_avg_angles": angles, "cond_method": "fa_invariant",
                "cond_autocast": "bf16", "commit": "d" * 40}
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
