"""exp_46 — the table generator's ``exp46`` contract (gen_model_comparison.py, announcement 04;
plan_c4flip_data_curve.md r3 §3.2 "Generator"), mirroring test_gen_model_comparison_exp45.py minus the bridge: eight
preregistered rows (C4F x 25 / 50 / 75 / 100 % x K 1 / 8) whose files must pass exp46_eval's OFFLINE cell admission
(f100's sealed delta verification included), a round gate that publishes a run only as a complete K=1 + K=8 pair at
one evaluator pin, and the plan r3 §8 disclosures under published exp_46 rows. exp_45's contract stays DORMANT: its
six rows keep rendering as pending placeholders, and a regeneration with exp_46 evidence and no exp_45 evidence (no
exp_45 file at all in the root) succeeds -- exp_46 publication never requires an exp_45 row or a bridge artifact.

Written by the exp_46 Coder (Claude Opus 5.5).
"""
import importlib.util
import json
import os
import shutil
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = os.path.join(REPO, "worklog", "worklog_yixun", "gen_model_comparison.py")
EXP_REL = os.path.join("worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude")
EXP38_REL = os.path.join("worklog", "worklog_yixun", "exp_38_data_efficiency_claude")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load(GEN, "gen_model_comparison_exp46_test")
RUNS = [f"exp46_C4Ff{t}_l40_2x32" for t in ("025", "050", "075", "100")]


def test_eight_preregistered_rows_with_exact_globs_and_exp45s_six_unchanged():
    rows = [r for r in gen.ROWS if len(r) > 4 and r[4] == "exp46"]
    assert len(rows) == 8 and sorted((gen.exp46_run_of(r[0]), r[2]) for r in rows) == sorted(
        (run, k) for run in RUNS for k in (1, 8))
    for label, proto, k, pats, _ in rows:
        run = gen.exp46_run_of(label)
        assert proto == "vanilla eval" and k in (1, 8)
        assert pats == [f"outputs_FLAC/exp46_c4flip_data_curve/eval/{run}/*_metrics_1_1.0_{run}_K{k}_s4[2-6].json"]
    assert not any("exp14_" in p or "exp11_" in p for r in rows for p in r[3])
    assert not any(gen.is_exp11_row(r[3]) or gen.is_batched_orbit_row(r[3]) for r in rows)
    labels = [r[0] for r in gen.ROWS]
    assert len(set(gen.EXP46_LABELS.values()) & set(labels)) == 4 and not any(
        lab.startswith(("data-curve ", "data-eff ", "rf-data ")) for lab in gen.EXP46_LABELS.values())
    assert gen.exp46_run_of(gen.EXP45_LABELS["025"]) is None and gen.exp45_run_of(gen.EXP46_LABELS["025"]) is None
    exp45 = [r for r in gen.ROWS if len(r) > 4 and r[4] == "exp45"]      # dormant: exp_45's contract as it was
    assert len(exp45) == 6 and sorted({gen.exp45_run_of(r[0]) for r in exp45}) == [
        f"exp45_RFf{t}_l40_2x32" for t in ("025", "050", "075")]


def _tmp_root(tmp_path, monkeypatch, delta_pin="e" * 64):
    """A self-contained root holding only what OFFLINE admission reads (exp46_collect.generator_static(): the exp_46
    kit's evaluator + registry, the C4F and E0 configs, exp_38's read-only reference cells and E0a decision) -- and no
    exp_45 file at all; the cached exp46 modules are set aside so the root's own copies (REPO = this root) are the
    ones the loaded validator uses. ``delta_pin`` stands in for the f100 delta manifest's L2 pin."""
    sys.path.insert(0, os.path.join(REPO, EXP_REL))
    import exp46_collect as col
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    for rel in col.generator_static():
        (root / os.path.dirname(rel)).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(os.path.join(REPO, rel), root / rel)
    assert not [rel for rel in col.generator_static() if "exp_45" in rel or "exp45" in rel]
    for name in ("exp46_registry", "exp46_eval"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    ev = gen._load_exp46_eval(str(root))
    assert ev.reg.REPO == str(root)
    monkeypatch.setattr(ev.reg, "F100_DELTA_SHA256", delta_pin)
    return root, ev


def _write_run(root, ev, run, K, seeds=(42, 43, 44, 45, 46), metrics=None):
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import exp46_fixtures as fx
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
                         str(root / "logs"), dump=False, metrics=metrics)
            files.append(ev._abs(c["metrics"]))
    return files


def test_validate_exp46_cell_offline(tmp_path, monkeypatch):
    root, ev = _tmp_root(tmp_path, monkeypatch)
    files = _write_run(root, ev, RUNS[0], 8)
    ok, probs = gen.validate_exp46_cell(files, repo_root=str(root), expected_run=RUNS[0], expected_k=8)
    assert ok, probs
    ok, probs = gen.validate_exp46_cell(files, repo_root=str(root), expected_run=RUNS[0], expected_k=1)
    assert not ok                                              # K mismatch
    ok, probs = gen.validate_exp46_cell(files, repo_root=str(root), expected_run=RUNS[1], expected_k=8)
    assert not ok                                              # another run's row
    stream = files[1][:-len(".json")] + ".stream.json"
    rec = json.load(open(stream))
    rec["assignment_hash"] = "0" * 64
    json.dump(rec, open(stream, "w"))
    ok, probs = gen.validate_exp46_cell(files, repo_root=str(root), expected_run=RUNS[0], expected_k=8)
    assert not ok and any("assignment_hash" in p for p in probs)
    rec = json.load(open(files[0]))
    rec["metrics"]["T60"] = 2.0
    json.dump(rec, open(files[0], "w"))
    ok, probs = gen.validate_exp46_cell(files, repo_root=str(root), expected_run=RUNS[0], expected_k=8)
    assert not ok and any("receipt metrics_sha256" in p for p in probs)
    assert gen.validate_exp46_cell([], repo_root=str(root), expected_run=RUNS[0], expected_k=8) == (True, [])
    ok, probs = gen.validate_exp46_cell(files, repo_root=str(tmp_path / "nowhere"), expected_run=RUNS[0], expected_k=8)
    assert not ok and "cannot load" in probs[0]


def test_offline_admission_needs_the_e0_receipt_the_training_receipt_and_its_loss_record(tmp_path, monkeypatch):
    root, ev = _tmp_root(tmp_path, monkeypatch)
    files = _write_run(root, ev, RUNS[1], 1)
    assert gen.validate_exp46_cell(files, repo_root=str(root), expected_run=RUNS[1], expected_k=1)[0]
    farm = os.path.dirname(files[0])
    csv = os.path.join(farm, "receipts", "loss_record.csv")
    good = open(csv, "rb").read()
    open(csv, "wb").write(good.replace(b"4.9e-05", b"4.8e-05", 1))
    ok, probs = gen.validate_exp46_cell(files, repo_root=str(root), expected_run=RUNS[1], expected_k=1)
    assert not ok and any("loss record" in p for p in probs)
    open(csv, "wb").write(good)
    os.rename(os.path.join(farm, "TRAINING_COMPLETE.json"), os.path.join(farm, "x.json"))
    ok, probs = gen.validate_exp46_cell(files, repo_root=str(root), expected_run=RUNS[1], expected_k=1)
    assert not ok and any("TRAINING_COMPLETE" in p for p in probs)
    os.rename(os.path.join(farm, "x.json"), os.path.join(farm, "TRAINING_COMPLETE.json"))
    os.remove(ev._abs(ev.E0_RECEIPT))
    ok, probs = gen.validate_exp46_cell(files, repo_root=str(root), expected_run=RUNS[1], expected_k=1)
    assert not ok and any("e0_receipt" in p for p in probs)


def test_the_f100_rows_need_the_sealed_delta_verification_at_the_pinned_manifest(tmp_path, monkeypatch):
    """plan r3 §3.2 "Data binding for f100": the generator's offline admission of an f100 cell re-opens the run's
    retained delta verification record (through the training receipt's offline check); missing, changed, or under no
    / another pin -> the row does not validate. Every pin is INJECTED (fix round 1, Codex code r1 finding 4): the
    unpinned case is the injected None, so the test holds before and after the L2 pin commit alike."""
    root, ev = _tmp_root(tmp_path, monkeypatch)
    files = _write_run(root, ev, RUNS[3], 8)
    assert gen.validate_exp46_cell(files, repo_root=str(root), expected_run=RUNS[3], expected_k=8) == (True, [])
    farm = os.path.dirname(files[0])
    rec = os.path.join(farm, "receipts", ev.reg.F100_VERIFICATION)
    good = open(rec, "rb").read()
    for pin in (None, "f" * 64):
        monkeypatch.setattr(ev.reg, "F100_DELTA_SHA256", pin)
        ok, probs = gen.validate_exp46_cell(files, repo_root=str(root), expected_run=RUNS[3], expected_k=8)
        assert not ok and any("f100 delta" in p for p in probs), probs
    monkeypatch.setattr(ev.reg, "F100_DELTA_SHA256", "e" * 64)
    os.remove(rec)
    ok, probs = gen.validate_exp46_cell(files, repo_root=str(root), expected_run=RUNS[3], expected_k=8)
    assert not ok and any("f100" in p for p in probs), probs
    open(rec, "wb").write(good.replace(b'"PASS"', b'"FAIL"'))
    ok, probs = gen.validate_exp46_cell(files, repo_root=str(root), expected_run=RUNS[3], expected_k=8)
    assert not ok, probs


def test_published_exp46_rows_carry_the_registered_disclosures(tmp_path, monkeypatch):
    """Under the shared table, published exp_46 rows carry the plan r3 §8 disclosures, loaded from the exp_46 kit at
    the generator root (exp46_registry.disclosures("generator") -- the one source the exp_46 tables print); none while
    every exp_46 row is pending or withheld; fail closed when they cannot be loaded."""
    root, ev = _tmp_root(tmp_path, monkeypatch)
    label = gen.EXP46_LABELS["025"]
    rendered = [{"label": label, "K": k, "line": "| ok |", "contract": "exp46"} for k in (8, 1)]
    lines, probs = gen.exp46_disclosure_lines(rendered, set(), str(root))
    assert probs == [] and len(ev.reg.disclosures("generator")) == 11
    assert lines == ["", gen.EXP46_DISCLOSURE_HEADER, ""] + list(ev.reg.disclosures("generator"))
    pending = [dict(r, line=f"| {label} | x | {r['K']} | 0 | *pending (0/5 seeds on disk)* | | | | | |") for r in rendered]
    assert gen.exp46_disclosure_lines(pending, set(), str(root)) == ([], [])
    assert gen.exp46_disclosure_lines(rendered, {label}, str(root)) == ([], [])
    for other in ("exp45", "exp38"):
        assert gen.exp46_disclosure_lines([dict(r, contract=other) for r in rendered], set(), str(root)) == ([], [])
    lines, probs = gen.exp46_disclosure_lines(rendered, set(), str(tmp_path / "nowhere"))
    assert lines == [] and probs and "disclosures" in probs[0]
    import inspect
    src = inspect.getsource(gen.main)                  # main() appends them after the exp_46 round gate
    assert src.index("check_exp46_round(rendered") < src.index("exp46_disclosure_lines(rendered")
    assert src.index("exp45_disclosure_lines(rendered") < src.index("check_exp46_round(rendered")


def test_round_gate_publishes_only_complete_pairs_at_one_pin(tmp_path):
    label = gen.EXP46_LABELS["100"]
    rendered = [{"label": label, "K": 8, "line": "| ok |", "contract": "exp46"},
                {"label": label, "K": 1, "line": "| x | *pending (3/5 seeds on disk)* |", "contract": "exp46"}]
    withheld, probs = gen.check_exp46_round(rendered, {(label, 8): ["a"], (label, 1): ["b"]})
    assert withheld == {label} and probs
    pins = {}
    for k, pin in ((8, "a" * 40), (1, "b" * 40)):
        pins[k] = tmp_path / f"m{k}.json"
        pins[k].write_text(json.dumps({"source_sha": pin}))
    rendered = [{"label": label, "K": k, "line": "| ok |", "contract": "exp46"} for k in (8, 1)]
    withheld, probs = gen.check_exp46_round(rendered, {(label, 8): [str(pins[8])], (label, 1): [str(pins[1])]})
    assert withheld == {label} and "pins" in probs[0]
    pins[1].write_text(json.dumps({"source_sha": "a" * 40}))
    assert gen.check_exp46_round(rendered, {(label, 8): [str(pins[8])], (label, 1): [str(pins[1])]}) == (set(), [])
    other = [dict(r, contract="exp45") for r in rendered]      # exp_45's rows are not exp_46's to gate
    assert gen.check_exp46_round(other, {}) == (set(), [])


def _generate(root, out, label):
    """main() over the root, the table written to ``out``: (Markdown, the rows of ``label``)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    assert gen.main(["--repo-root", str(root), "--out", str(out)]) == 0
    md = out.read_text()
    return md, [ln for ln in md.split("\n") if ln.startswith(f"| {label} |")]


def test_exp45_is_dormant_a_regen_with_exp46_evidence_and_no_exp45_evidence_publishes(tmp_path, monkeypatch):
    """plan r3 §3.2 "Generator": through main(), over a root holding a complete exp_46 C4F f=25 % run (K=8 + K=1,
    five seeds each) and NO exp_45 file at all, the two C4F rows publish with the registered disclosures, the other
    six exp_46 rows stay pending, and exp_45's six rows render exactly the pending placeholders they render with no
    evidence at all -- no exp_45 row, bridge artifact or exp_45 kit is required (none exists to read)."""
    root, ev = _tmp_root(tmp_path, monkeypatch)
    monkeypatch.setattr(gen, "ROWS", [r for r in gen.ROWS if len(r) > 4 and r[4] in ("exp45", "exp46")])
    empty = tmp_path / "empty_root"
    (empty / ".git").mkdir(parents=True)
    md0, _ = _generate(empty, tmp_path / "B0" / "model_comparison.md", gen.EXP46_LABELS["025"])
    exp45_lines = lambda md: [ln for ln in md.split("\n") if any(ln.startswith(f"| {lab} |")  # noqa: E731
                                                                for lab in gen.EXP45_LABELS.values())]
    assert len(exp45_lines(md0)) == 6 and all("*pending (0/5 seeds on disk)*" in ln for ln in exp45_lines(md0))
    assert gen.EXP46_DISCLOSURE_HEADER not in md0
    _write_run(root, ev, RUNS[0], 8)
    _write_run(root, ev, RUNS[0], 1)
    assert not [d for d, _, fs in os.walk(root) for f in fs if "exp45" in f or "exp_45" in d]
    md, rows = _generate(root, tmp_path / "B" / "model_comparison.md", gen.EXP46_LABELS["025"])
    assert len(rows) == 2 and all(r.count(" ± ") == 6 for r in rows), rows
    assert exp45_lines(md) == exp45_lines(md0)                          # dormant: unchanged placeholders
    assert md.count(gen.EXP46_DISCLOSURE_HEADER) == 1 and all(d in md for d in ev.reg.disclosures("generator"))
    assert "RF-100 bridge" not in md and "exp_45 rows WITHHELD" not in md and "exp_46 rows WITHHELD" not in md
    others = [ln for ln in md.split("\n") if any(ln.startswith(f"| {gen.EXP46_LABELS[t]} |") for t in ("050", "075", "100"))]
    assert len(others) == 6 and all("*pending (0/5 seeds on disk)*" in ln for ln in others)


def test_an_incomplete_exp46_pair_is_withheld_through_main(tmp_path, monkeypatch):
    """A run with only its K=8 cells on disk publishes nothing: both rows WITHHELD, no disclosures (fail closed)."""
    root, ev = _tmp_root(tmp_path, monkeypatch)
    monkeypatch.setattr(gen, "ROWS", [r for r in gen.ROWS if len(r) > 4 and r[4] == "exp46"])
    _write_run(root, ev, RUNS[2], 8)
    md, rows = _generate(root, tmp_path / "B" / "model_comparison.md", gen.EXP46_LABELS["075"])
    assert len(rows) == 2 and all(" ± " not in r for r in rows) and any("WITHHELD" in r for r in rows), rows
    assert "**exp_46 rows WITHHELD:**" in md and gen.EXP46_DISCLOSURE_HEADER not in md
