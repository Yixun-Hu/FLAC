"""exp_45 — the table builder (exp45_table.py; plan_rf_data_curve.md r3 §3.2, Codex plan r2 finding 5, r3 findings 1, 4).

A synthetic exp_38 results bundle is committed in a throw-away git repository (never in this one): exp_38's own kit
and fixtures (exp38_fixtures.py) generate a sealed E0 receipt with its eight cells, six sealed training records
copied to the farm and the 60 offline-admissible fraction cells, in a SEPARATE process. ``import-exp38`` admits it
through the isolated adapter; with the fake NAS gone (no /media path anywhere), a missing or changed file of
EXP38_IMPORT_FILES, a commit not descending from the base, an adapter opening anything outside the list -- each
refuses. ``build`` refuses a missing cell, a value differing from its source, a wrong commit or a resealed
record, and never reads exp_38's preliminary files.

Written by the exp_45 Coder (Claude Opus 5.5).
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_45_rf_data_curve_claude")
sys.path.insert(0, EXP)
import exp45_table as tbl  # noqa: E402

E38 = "worklog/worklog_yixun/exp_38_data_efficiency_claude"
COPY = [f"{E38}/exp38_eval.py", f"{E38}/exp38_registry.py", f"{E38}/ref_cells", f"{E38}/FLAC_AR_BF_cap32.json",
        f"{E38}/data_efficiency_e0a_decision.json", f"{E38}/data_efficiency_e0a_decision.json.sha256",
        "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json",
        "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BF.json",
        "worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json"]
GEN = textwrap.dedent("""
    import json, os, sys
    repo, tests, nas = sys.argv[1:4]
    sys.path.insert(0, os.path.join(repo, {e38!r}))
    sys.path.insert(0, tests)
    import exp38_eval as ev
    import exp38_fixtures as fx
    assert ev.reg.REPO == repo, ev.reg.REPO
    campaign, rec = fx.make_e0_receipt(ev, log_dir=os.path.join(nas, "e0_logs"))
    for run in {runs!r}:
        rd = os.path.join(nas, run)
        fx.make_training_receipt(ev, run, rd, campaign["closure_sha256"], e0_sha=campaign["sha256"],
                                 e0_path=campaign["e0_receipt"])
        fx.copy_to_farm(ev, rd, os.path.dirname(ev._abs(ev.run_cells(run)[0]["ckpt_link"])))
        tr, digest = ev.reg.read_sealed(os.path.join(rd, ev.reg.TRAINING_RECEIPT), ev.reg.TRAINING_KEYS)
        for i, c in enumerate(ev.run_cells(run)):
            m = {{k: 1.0 + 0.01 * i + (0.5 if "_bf_" in run else 0.0) for k in ev.METRIC_KEYS}}
            fx.make_cell(ev, c, campaign, (tr["final_ckpt_sha256"], tr["final_ckpt_bytes"], digest),
                         os.path.join(nas, "logs"), dump=False, metrics=m)
    print("GENERATED", campaign["sha256"])
""").format(e38=E38, runs=list(tbl.EXP38_RUNS))


def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo, msg):
    _git(repo, "add", "-f", "-A")
    _git(repo, "-c", "user.name=exp45-test", "-c", "user.email=exp45-test", "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """(repo, base commit, bundle commit, fake NAS dir) of a synthetic exp_38 results bundle."""
    import shutil
    base = tmp_path_factory.mktemp("exp38_bundle")
    repo, nas = str(base / "repo"), str(base / "nas")
    os.makedirs(repo)
    _git(repo, "init", "-q")
    archive = subprocess.run(["git", "-C", REPO, "archive", "--format=tar", "HEAD", "--", *COPY], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", repo], input=archive.stdout, check=True)
    root = _commit(repo, "exp_38 kit (the base)")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONNOUSERSITE"] = "1"
    out = subprocess.run([sys.executable, "-I", "-B", "-c", GEN, repo, TESTS, nas], cwd=repo, env=env,
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0 and "GENERATED" in out.stdout, out.stderr[-2000:]
    sha = _commit(repo, "exp_38 results bundle (synthetic)")
    shutil.rmtree(nas)                                           # the NAS is absent from here on
    os.makedirs(os.path.join(repo, os.path.dirname(tbl.IMPORT_RECORD)), exist_ok=True)
    return repo, root, sha


@pytest.fixture
def imp(bundle, monkeypatch):
    repo, root, sha = bundle
    monkeypatch.setattr(tbl, "EXP38_BASE", root)
    rec = os.path.join(repo, tbl.IMPORT_RECORD)
    for p in (rec, rec + ".sha256"):
        if os.path.exists(p):
            os.remove(p)
    return repo, root, sha


def test_the_import_admits_the_bundle_offline_with_no_nas(imp):
    repo, root, sha = imp
    out = tbl.import_exp38(sha, repo=repo)
    rec, digest = tbl.reg.read_sealed(os.path.join(repo, tbl.IMPORT_RECORD), tbl.IMPORT_KEYS)
    assert out["sha256"] == digest and rec["commit"] == sha and len(rec["cells"]) == 60
    assert sorted(rec["rows"]) == sorted(f"{a}|{t}|{K}" for a in ("yaw", "bf") for t in tbl.FRACS for K in (8, 1))
    assert abs(rec["rows"]["yaw|025|8"]["T60"][0] - (1.0 + 0.01 * 2)) < 1e-12           # seeds 42-46 = cells 0-4
    assert abs(rec["rows"]["bf|075|1"]["EDT"][0] - (1.5 + 0.01 * 7)) < 1e-12 and rec["rows"]["bf|075|1"]["EDT"][2] == 5
    files = tbl.exp38_import_files(repo)
    assert sorted(rec["files"]) == files and set(rec["opened"]) <= set(files)
    assert not [f for f in rec["opened"] if f.startswith("NAS:")]
    # every family of the derived list was really opened by exp_38's own offline admission
    for needle in ("data_efficiency_e0_receipt.json", "REF_MANIFEST.sha256", "TRAINING_COMPLETE.json", "TRAINING_COMMITTED.json",
                   "CHECKPOINTS.sha256", "receipts/train.log", "receipts/smoke_log.log", "receipts/SNAPSHOT.links",
                   "evidence/run_evidence.json.rank1.json", ".dump_manifest.json", ".stream.json", ".receipt.json",
                   "FLAC_AR_BF_cap32.json", "FLAC_AR_YAWAUG.json", "at_launch_"):
        assert any(needle in f for f in rec["opened"]), needle
    assert len(rec["attestations"]["checkpoints"]) == 60
    with pytest.raises(tbl.TableError, match="sealed once"):
        tbl.import_exp38(sha, repo=repo)


def _variant(imp, fn, msg):
    """A new commit on top of the bundle with ``fn(repo)`` applied."""
    repo, root, sha = imp
    fn(repo)
    new = _commit(repo, msg)
    _git(repo, "reset", "-q", "--hard", sha)
    return new


def test_a_missing_or_changed_import_file_refuses(imp):
    repo, root, sha = imp
    stems = tbl.exp38_cell_stems()
    miss = _variant(imp, lambda r: os.remove(os.path.join(r, stems["de_bf_f050_K1_s44"][3] + ".dump_manifest.json")), "drop")
    with pytest.raises(tbl.TableError, match="misses"):
        tbl.import_exp38(miss, repo=repo)
    farm = os.path.join(repo, tbl.EXP38_FARM, "de_yaw_f075", "receipts", "train.log")
    gone = _variant(imp, lambda r: os.remove(os.path.join(r, tbl.EXP38_FARM, "de_yaw_f075", "receipts", "train.log")), "drop log")
    with pytest.raises(tbl.TableError, match="misses"):
        tbl.import_exp38(gone, repo=repo)

    def bump(r):
        p = os.path.join(r, stems["de_yaw_f025_K8_s43"][3] + ".json")
        d = json.load(open(p))
        d["metrics"]["T60"] += 1e-6
        json.dump(d, open(p, "w"))
    changed = _variant(imp, bump, "change")
    with pytest.raises(tbl.TableError, match="not admitted"):
        tbl.import_exp38(changed, repo=repo)
    assert not os.path.exists(os.path.join(repo, tbl.IMPORT_RECORD)) and os.path.exists(farm)


def test_only_a_full_commit_descending_from_the_base_is_imported(imp, monkeypatch):
    repo, root, sha = imp
    with pytest.raises(tbl.TableError, match="full 40-hex"):
        tbl.import_exp38(sha[:12], repo=repo)
    monkeypatch.setattr(tbl, "EXP38_BASE", "f" * 40)
    with pytest.raises(tbl.TableError):
        tbl.import_exp38(sha, repo=repo)


def test_an_adapter_opening_a_file_outside_the_list_refuses(imp, monkeypatch):
    repo, root, sha = imp
    real = tbl.exp38_import_files
    monkeypatch.setattr(tbl, "exp38_import_files", lambda r: [f for f in real(r) if not f.endswith("receipts/SNAPSHOT.links")])
    with pytest.raises(tbl.TableError, match="outside EXP38_IMPORT_FILES"):
        tbl.import_exp38(sha, repo=repo)


def test_materialisation_copies_the_verified_bytes_to_their_exact_paths(imp, tmp_path):
    repo, root, sha = imp
    gen = tmp_path / "generator_root"
    gen.mkdir()
    tbl.import_exp38(sha, generator_root=str(gen), repo=repo)
    rec, _ = tbl.reg.read_sealed(os.path.join(repo, tbl.IMPORT_RECORD), tbl.IMPORT_KEYS)
    # Codex code r1 finding 12: EVERY admitted file and exp_38's kit code reach the generator root (it need not carry
    # exp_38's folder): the configs and reference cells too, and exp38_eval.py / exp38_registry.py as imported
    copied = sorted(rec["files"])
    assert copied and all(tbl.reg.sha256_file(str(gen / r)) == rec["files"][r] for r in copied)
    assert {f"{tbl.reg.EXP38_DIR}/exp38_eval.py", f"{tbl.reg.EXP38_DIR}/exp38_registry.py"} <= set(rec["code"])
    assert all(tbl.reg.sha256_file(str(gen / r)) == d == tbl.reg.sha256_file(os.path.join(repo, r))
               for r, d in rec["code"].items())
    assert (gen / tbl.EXP38_FARM / "de_bf_f025").is_dir()
    os.remove(os.path.join(repo, tbl.IMPORT_RECORD))
    os.remove(os.path.join(repo, tbl.IMPORT_RECORD) + ".sha256")
    victim = gen / copied[0]
    victim.write_bytes(victim.read_bytes() + b" ")
    with pytest.raises(tbl.TableError, match="other bytes"):
        tbl.import_exp38(sha, generator_root=str(gen), repo=repo)


# ------------------------------------------------------------------ build
@pytest.fixture
def built(imp, tmp_path, monkeypatch):
    """A sealed import record + a fake exp_45 side (offline admission patched, a published collector JSON)."""
    repo, root, sha = imp
    tbl.import_exp38(sha, repo=repo)
    rows = {f"rf|{t}|{K}": {k: [2.0, 0.1, 5] for k in tbl.ev.METRIC_KEYS} for t in tbl.FRACS for K in (8, 1)}
    monkeypatch.setattr(tbl.ev, "validate_e0_receipt", lambda offline=False: {"head": "e" * 40, "closure_sha256": "c" * 64,
                                                                            "sha256": "d" * 64})
    monkeypatch.setattr(tbl, "rf_rows", lambda campaign: json.loads(json.dumps(rows)))
    res = {"rows": [{"model": "rf", "fraction": k.split("|")[1], "K": int(k.split("|")[2]),
                     "stats": {m: {"mean": v[0], "sd": v[1], "n": v[2]} for m, v in st.items()}} for k, st in rows.items()]}
    path = os.path.join(repo, tbl.reg.EXP_DIR, "rf_data_curve_results.json")
    json.dump(res, open(path, "w"))
    for p in (tbl.TABLE_MD, tbl.TABLE_EVIDENCE, tbl.TABLE_EVIDENCE + ".sha256"):
        if os.path.exists(os.path.join(repo, p)):
            os.remove(os.path.join(repo, p))
    return repo, sha, path


def test_build_renders_all_eighteen_cells_and_the_disclosures(built):
    repo, sha, _ = built
    md = tbl.build(sha, repo=repo)
    for label in tbl.METHODS.values():
        assert md.count(f"| {label} | ") == 6
    assert sum(ln.count(" ± ") for ln in md.split("\n") if ln.startswith("| ")) == 18 * 6
    assert all(d in md for d in tbl.DISCLOSURES) and sha[:12] in md
    ev_rec, _ = tbl.reg.read_sealed(os.path.join(repo, tbl.TABLE_EVIDENCE), {"schema", "exp38_commit", "exp38_import_sha256",
                                                                             "exp45_campaign", "rf_results_sha256", "rows",
                                                                             "written_at"})
    assert ev_rec["exp38_commit"] == sha and len(ev_rec["rows"]) == 18


def test_build_refuses_a_wrong_commit_a_resealed_record_or_a_changed_value(built):
    repo, sha, results = built
    with pytest.raises(tbl.TableError, match="not"):
        tbl.build("0" * 40, repo=repo)
    res = json.load(open(results))
    res["rows"][0]["stats"]["T60"]["mean"] = 2.0001
    json.dump(res, open(results, "w"))
    with pytest.raises(tbl.TableError, match="differs from its source"):
        tbl.build(sha, repo=repo)
    res["rows"][0]["stats"]["T60"]["mean"] = 2.0
    json.dump(res, open(results, "w"))
    p = os.path.join(repo, tbl.IMPORT_RECORD)
    rec = json.load(open(p))
    rec["rows"]["yaw|050|8"]["C50"][0] += 1e-9                                       # rows != the record's own cells
    os.remove(p)
    os.remove(p + ".sha256")
    tbl.reg.write_sealed(p, rec)
    with pytest.raises(tbl.TableError, match="per-cell"):
        tbl.build(sha, repo=repo)
    open(p, "ab").write(b" ")                                                       # the sidecar binds the record
    with pytest.raises(tbl.reg.GateError):
        tbl.build(sha, repo=repo)


def _reseal_import(repo, rec):
    p = os.path.join(repo, tbl.IMPORT_RECORD)
    for f in (p, p + ".sha256"):
        os.remove(f)
    tbl.reg.write_sealed(p, rec)


def _tracked_variant(repo, sha, fn, msg):
    """A commit on top of ``sha`` changing tracked files only (the untracked import record / results stay put)."""
    fn(repo)
    _git(repo, "-c", "user.name=exp45-test", "-c", "user.email=exp45-test", "commit", "-q", "-a", "-m", msg)
    new = _git(repo, "rev-parse", "HEAD")
    _git(repo, "reset", "-q", "--hard", sha)
    return new


def test_build_reopens_the_bundle_blobs_at_the_requested_commit(built):
    """Codex code r1 finding 6: build re-extracts the bundle at the requested commit (git archive, read-only) and
    re-verifies the record against those blobs -- every EXP38_IMPORT_FILES hash, every cell's metrics, the
    attestations its receipts carry and the E0 campaign -- so a record whose cells AND rows were changed together,
    whose file hash or attestation was edited, or which was relabelled to another commit, refuses after resealing."""
    repo, sha, _ = built
    good = json.load(open(os.path.join(repo, tbl.IMPORT_RECORD)))
    name = "de_yaw_f050_K8_s43"

    def mutated(fn):
        rec = json.loads(json.dumps(good))
        fn(rec)
        return rec

    def cells_and_rows(r):
        r["cells"][name]["metrics"]["T60"] += 0.5
        r["rows"] = tbl.exp38_rows(r["cells"])                      # internally consistent again
    some_file = sorted(good["files"])[0]
    cases = [(cells_and_rows, name),
             (lambda r: r["files"].update({some_file: "0" * 64}), "sha256"),
             (lambda r: r["attestations"]["checkpoints"].update({name: "0" * 64}), name),
             (lambda r: r["campaign"].update(head="0" * 40), "campaign")]
    for fn, needle in cases:
        _reseal_import(repo, mutated(fn))
        with pytest.raises(tbl.TableError, match=needle):
            tbl.build(sha, repo=repo)
    stems = tbl.exp38_cell_stems()

    def bump(r):
        p = os.path.join(r, stems[name][3] + ".json")
        d = json.load(open(p))
        d["metrics"]["T60"] += 0.5
        json.dump(d, open(p, "w"))
    other = _tracked_variant(repo, sha, bump, "a later bundle commit with another value")
    _reseal_import(repo, mutated(lambda r: r.update(commit=other)))  # the record relabelled to that commit
    with pytest.raises(tbl.TableError, match="sha256"):
        tbl.build(other, repo=repo)
    _reseal_import(repo, good)
    assert sha[:12] in tbl.build(sha, repo=repo)                     # the true record still builds


def test_the_disclosures_have_one_source_and_appear_under_every_exp45_table(built):
    """Codex code r1 finding 15: the plan r3 §8 disclosures live in ONE place (exp45_registry.DISCLOSURES) and print
    under every exp_45 table -- this comparison table, the collector's results Markdown (all but item 7: it has no
    OrbitRIR row) and the shared generator's exp_45 block (test_gen_model_comparison_exp45.py) -- so they cannot drift."""
    reg = tbl.reg
    assert len(reg.DISCLOSURES) == 11 and tbl.DISCLOSURES == reg.disclosures("table") == reg.DISCLOSURES
    assert reg.disclosures("collect") == tuple(d for i, d in enumerate(reg.DISCLOSURES, 1) if i != 7)
    gen = os.path.join(REPO, "worklog", "worklog_yixun", "gen_model_comparison.py")
    for d in reg.DISCLOSURES:                            # the text itself is written once, in the registry
        needle = d[3:60]
        holders = sorted(f for f in os.listdir(EXP) if f.endswith(".py") and needle in open(os.path.join(EXP, f)).read())
        assert holders == ["exp45_registry.py"] and needle not in open(gen).read(), (needle, holders)
    repo, sha, _ = built
    md = tbl.build(sha, repo=repo)
    assert all(d in md for d in reg.disclosures("table"))
    import exp45_collect as col
    rows = {(m, f, K): {"stats": {k: (1.0, 0.1, 5) for k in tbl.ev.METRIC_KEYS}, "source": "s"}
            for m, f in [("rf", t) for t in tbl.FRACS] + [("rf_bridge", "100")] for K in (8, 1)}
    cmd = col.render(rows, [])
    assert all(d in cmd for d in reg.disclosures("collect")) and reg.DISCLOSURES[6] not in cmd


def test_build_refuses_a_missing_cell(built, monkeypatch):
    repo, sha, _ = built
    real = tbl.rf_rows
    monkeypatch.setattr(tbl, "rf_rows", lambda c: {k: v for k, v in real(c).items() if k != "rf|075|1"})
    with pytest.raises(tbl.TableError, match="missing"):
        tbl.build(sha, repo=repo)


def test_build_never_reads_the_preliminary_files(built, monkeypatch):
    import builtins
    repo, sha, _ = built
    opened, real_open = [], builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: opened.append(str(f)) or real_open(f, *a, **k))
    tbl.build(sha, repo=repo)
    assert opened and not [p for p in opened if "prelim_eval" in p or "_sessions" in p]
    import inspect
    assert "prelim" not in inspect.getsource(tbl.build) + inspect.getsource(tbl.rf_rows)
