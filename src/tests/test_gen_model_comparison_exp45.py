"""exp_45 — the table generator's ``exp45`` contract (gen_model_comparison.py, announcement 04; plan_rf_data_curve.md
r3 §3.2), mirroring test_gen_model_comparison_exp38.py: six preregistered rows (RF x 25 / 50 / 75 % x K 1 / 8) whose
files must pass exp45_eval's OFFLINE cell admission, and a round gate that publishes a run only as a complete
K=1 + K=8 pair at one evaluator pin.

Written by the exp_45 Coder (Claude Opus 5.5).
"""
import importlib.util
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(REPO, "worklog", "worklog_yixun", "gen_model_comparison.py")
EXP_REL = os.path.join("worklog", "worklog_yixun", "exp_45_rf_data_curve_claude")
EXP38_REL = os.path.join("worklog", "worklog_yixun", "exp_38_data_efficiency_claude")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load(GEN, "gen_model_comparison_exp45_test")
RUNS = [f"exp45_RFf{t}_l40_2x32" for t in ("025", "050", "075")]


def test_six_preregistered_rows_with_exact_globs():
    rows = [r for r in gen.ROWS if len(r) > 4 and r[4] == "exp45"]
    assert len(rows) == 6 and sorted({gen.exp45_run_of(r[0]) for r in rows}) == RUNS
    for label, proto, k, pats, _ in rows:
        run = gen.exp45_run_of(label)
        assert proto == "vanilla eval" and k in (1, 8)
        assert pats == [f"outputs_FLAC/exp45_rf_data_curve/eval/{run}/*_metrics_1_1.0_{run}_K{k}_s4[2-6].json"]
    assert not any("exp14_" in p or "exp11_" in p for r in rows for p in r[3])
    assert not any(gen.is_exp11_row(r[3]) or gen.is_batched_orbit_row(r[3]) for r in rows)
    labels = [r[0] for r in gen.ROWS]
    assert len(set(gen.EXP45_LABELS.values()) & set(labels)) == 3 and not any(
        lab.startswith(("data-curve ", "data-eff ")) for lab in gen.EXP45_LABELS.values())


def _tmp_root(tmp_path, monkeypatch):
    """A self-contained root holding only what OFFLINE admission reads: the exp_45 kit, the RF + E0 configs, exp_38's
    read-only reference cells and E0a decision; the cached exp45 modules are set aside so the root's own copies
    (REPO = this root) are the ones the loaded validator uses."""
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    for rel in (os.path.join(EXP_REL, "exp45_eval.py"), os.path.join(EXP_REL, "exp45_registry.py"),
                "worklog/worklog_yixun/exp_43_std_aug_claude/FLAC_AR_YAWAUG_FLIP.json",
                "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json",
                "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",
                "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json",
                os.path.join(EXP38_REL, "data_efficiency_e0a_decision.json"),
                os.path.join(EXP38_REL, "data_efficiency_e0a_decision.json.sha256")):
        (root / os.path.dirname(rel)).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(os.path.join(REPO, rel), root / rel)
    shutil.copytree(os.path.join(REPO, EXP38_REL, "ref_cells"), root / EXP38_REL / "ref_cells")
    for name in ("exp45_registry", "exp45_eval"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    ev = gen._load_exp45_eval(str(root))
    assert ev.reg.REPO == str(root)
    return root, ev


def _write_run(root, ev, run, K, seeds=(42, 43, 44, 45, 46)):
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import exp45_fixtures as fx
    if not os.path.exists(ev._abs(ev.E0_RECEIPT)):
        fx.make_e0_receipt(ev, log_dir=str(root / "logs"))
    campaign = ev.validate_e0_receipt(offline=True)
    farm = os.path.dirname(ev._abs(ev.run_cells(run)[0]["ckpt_link"]))
    if not os.path.exists(os.path.join(farm, ev.reg.TRAINING_RECEIPT)):
        fx.make_training_receipt(ev, run, farm, campaign["closure_sha256"], e0_sha=campaign["sha256"],
                                 e0_path=campaign["e0_receipt"])
    tr, digest = ev.reg.read_sealed(os.path.join(farm, ev.reg.TRAINING_RECEIPT), ev.reg.TRAINING_KEYS)
    files = []
    for c in ev.run_cells(run):
        if c["K"] == K and c["seed"] in seeds:
            fx.make_cell(ev, c, campaign, (tr["final_ckpt_sha256"], tr["final_ckpt_bytes"], digest),
                         str(root / "logs"), dump=False)
            files.append(ev._abs(c["metrics"]))
    return files


def test_validate_exp45_cell_offline(tmp_path, monkeypatch):
    root, ev = _tmp_root(tmp_path, monkeypatch)
    files = _write_run(root, ev, RUNS[0], 8)
    ok, probs = gen.validate_exp45_cell(files, repo_root=str(root), expected_run=RUNS[0], expected_k=8)
    assert ok, probs
    ok, probs = gen.validate_exp45_cell(files, repo_root=str(root), expected_run=RUNS[0], expected_k=1)
    assert not ok                                              # K mismatch
    ok, probs = gen.validate_exp45_cell(files, repo_root=str(root), expected_run=RUNS[1], expected_k=8)
    assert not ok                                              # another run's row
    stream = files[1][:-len(".json")] + ".stream.json"
    rec = json.load(open(stream))
    rec["assignment_hash"] = "0" * 64
    json.dump(rec, open(stream, "w"))
    ok, probs = gen.validate_exp45_cell(files, repo_root=str(root), expected_run=RUNS[0], expected_k=8)
    assert not ok and any("assignment_hash" in p for p in probs)
    rec = json.load(open(files[0]))
    rec["metrics"]["T60"] = 2.0
    json.dump(rec, open(files[0], "w"))
    ok, probs = gen.validate_exp45_cell(files, repo_root=str(root), expected_run=RUNS[0], expected_k=8)
    assert not ok and any("receipt metrics_sha256" in p for p in probs)
    assert gen.validate_exp45_cell([], repo_root=str(root), expected_run=RUNS[0], expected_k=8) == (True, [])
    ok, probs = gen.validate_exp45_cell(files, repo_root=str(tmp_path / "nowhere"), expected_run=RUNS[0], expected_k=8)
    assert not ok and "cannot load" in probs[0]


def test_offline_admission_needs_the_e0_receipt_the_training_receipt_and_its_loss_record(tmp_path, monkeypatch):
    root, ev = _tmp_root(tmp_path, monkeypatch)
    files = _write_run(root, ev, RUNS[1], 1)
    assert gen.validate_exp45_cell(files, repo_root=str(root), expected_run=RUNS[1], expected_k=1)[0]
    farm = os.path.dirname(files[0])
    csv = os.path.join(farm, "receipts", "loss_record.csv")
    good = open(csv, "rb").read()
    open(csv, "wb").write(good.replace(b"4.9e-05", b"4.8e-05", 1))
    ok, probs = gen.validate_exp45_cell(files, repo_root=str(root), expected_run=RUNS[1], expected_k=1)
    assert not ok and any("loss record" in p for p in probs)
    open(csv, "wb").write(good)
    os.rename(os.path.join(farm, "TRAINING_COMPLETE.json"), os.path.join(farm, "x.json"))
    ok, probs = gen.validate_exp45_cell(files, repo_root=str(root), expected_run=RUNS[1], expected_k=1)
    assert not ok and any("TRAINING_COMPLETE" in p for p in probs)
    os.rename(os.path.join(farm, "x.json"), os.path.join(farm, "TRAINING_COMPLETE.json"))
    os.remove(ev._abs(ev.E0_RECEIPT))
    ok, probs = gen.validate_exp45_cell(files, repo_root=str(root), expected_run=RUNS[1], expected_k=1)
    assert not ok and any("e0_receipt" in p for p in probs)


def test_published_exp45_rows_carry_the_registered_disclosures(tmp_path, monkeypatch):
    """Codex code r1 finding 15: under the shared table, published exp_45 rows carry the plan r3 §8 disclosures,
    loaded from the exp_45 kit at the generator root (exp45_registry.disclosures("generator") -- the one source the
    exp_45 tables print); none while every exp_45 row is pending or withheld; fail closed when they cannot be loaded."""
    root, ev = _tmp_root(tmp_path, monkeypatch)
    label = gen.EXP45_LABELS["025"]
    rendered = [{"label": label, "K": k, "line": "| ok |", "contract": "exp45"} for k in (8, 1)]
    lines, probs = gen.exp45_disclosure_lines(rendered, set(), str(root))
    assert probs == [] and len(ev.reg.disclosures("generator")) == 11
    assert all(d in lines for d in ev.reg.disclosures("generator"))
    pending = [dict(r, line=f"| {label} | x | {r['K']} | 0 | *pending (0/5 seeds on disk)* | | | | | |") for r in rendered]
    assert gen.exp45_disclosure_lines(pending, set(), str(root)) == ([], [])
    assert gen.exp45_disclosure_lines(rendered, {label}, str(root)) == ([], [])
    other = [dict(r, contract="exp38") for r in rendered]
    assert gen.exp45_disclosure_lines(other, set(), str(root)) == ([], [])
    lines, probs = gen.exp45_disclosure_lines(rendered, set(), str(tmp_path / "nowhere"))
    assert lines == [] and probs and "disclosures" in probs[0]
    import inspect
    src = inspect.getsource(gen.main)                  # main() appends them after the exp_45 round gate
    assert src.index("check_exp45_round(rendered") < src.index("exp45_disclosure_lines(rendered")


def test_a_generator_root_materialised_from_the_checkout_resolves_the_exp45_rows(tmp_path, monkeypatch):
    """Codex code r1 finding 12: what ``exp45_collect.py materialise`` copies besides the evidence
    (generator_static()) is exactly enough for the shared generator's exp_45 contract in a root that does not track
    the exp_45 folder (the neuronic main checkout): with those files + a run's published cells, the rows validate."""
    sys.path.insert(0, os.path.join(REPO, EXP_REL))
    import exp45_collect as col
    root = tmp_path / "generator_root"
    (root / ".git").mkdir(parents=True)
    for rel in col.generator_static():
        (root / os.path.dirname(rel)).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(os.path.join(REPO, rel), root / rel)
    for name in ("exp45_registry", "exp45_eval"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    ev = gen._load_exp45_eval(str(root))
    assert ev.reg.REPO == str(root)
    files = _write_run(root, ev, RUNS[2], 1)
    ok, probs = gen.validate_exp45_cell(files, repo_root=str(root), expected_run=RUNS[2], expected_k=1)
    assert ok, probs


def test_the_generator_writes_its_table_to_out_never_into_the_evidence_root(tmp_path, monkeypatch):
    """Codex code r2 finding 1: --repo-root selects the EVIDENCE only; --out selects the table, so the L8
    regenerations read the neuronic main checkout and write the table into the build checkout B."""
    root = tmp_path / "main_checkout"
    (root / "worklog" / "worklog_yixun").mkdir(parents=True)
    table = root / "worklog" / "worklog_yixun" / "model_comparison.md"
    table.write_text("main's tracked table\n")
    out = tmp_path / "B" / "worklog" / "worklog_yixun" / "model_comparison.md"
    out.parent.mkdir(parents=True)
    monkeypatch.setattr(gen, "ROWS", [])
    assert gen.main(["--repo-root", str(root), "--out", str(out)]) == 0
    assert table.read_text() == "main's tracked table\n" and out.read_text().startswith("# Model comparison")


def test_round_gate_publishes_only_complete_pairs_at_one_pin(tmp_path):
    label = gen.EXP45_LABELS["025"]
    rendered = [{"label": label, "K": 8, "line": "| ok |", "contract": "exp45"},
                {"label": label, "K": 1, "line": "| x | *pending (3/5 seeds on disk)* |", "contract": "exp45"}]
    withheld, probs = gen.check_exp45_round(rendered, {(label, 8): ["a"], (label, 1): ["b"]})
    assert withheld == {label} and probs
    pins = {}
    for k, pin in ((8, "a" * 40), (1, "b" * 40)):
        pins[k] = tmp_path / f"m{k}.json"
        pins[k].write_text(json.dumps({"source_sha": pin}))
    rendered = [{"label": label, "K": k, "line": "| ok |", "contract": "exp45"} for k in (8, 1)]
    withheld, probs = gen.check_exp45_round(rendered, {(label, 8): [str(pins[8])], (label, 1): [str(pins[1])]})
    assert withheld == {label} and "pins" in probs[0]
    pins[1].write_text(json.dumps({"source_sha": "a" * 40}))
    assert gen.check_exp45_round(rendered, {(label, 8): [str(pins[8])], (label, 1): [str(pins[1])]}) == (set(), [])
