"""exp_45 — isolation from exp_38's roots (Codex code r2 finding 6).

exp_38's protected roots are its two save roots (A6000 NAS, neuronic), its dump root, and -- in every tree involved --
its farm and its worklog folder, all compared as realpaths. Before ANY mutation (probe, lock, ledger, owner record,
directory), an exp_45 output namespace -- a save root, the dump root, the farm -- that equals, contains or lies
inside a protected root refuses, so an alias (a symlink, a bind of the same directory) can never route exp_45's
writes into exp_38's. A materialisation destination may not equal or lie inside a protected root, writes no file
through an aliased directory, and exp45_collect.py writes nothing into exp_38's namespaces (exp_38's reference files
must already be in the destination, byte-identical).

Written by the exp_45 Coder (Claude Opus 5.5).
"""
import hashlib
import importlib.util
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_45_rf_data_curve_claude")
sys.path.insert(0, EXP)
import exp45_registry as reg  # noqa: E402

RUN = "exp45_RFf025_l40_2x32"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_protected_roots_are_exp38s_own_values():
    """The protected roots are read from exp_38's own registry and evaluator (never retyped)."""
    src38 = open(os.path.join(REPO, reg.EXP38_DIR, "exp38_registry.py")).read()
    ev38 = open(os.path.join(REPO, reg.EXP38_DIR, "exp38_eval.py")).read()
    for root in reg.EXP38_SAVE_ROOTS:
        assert f'"save_root": "{root}"' in src38, root
    assert f'DUMP_ROOT = "{reg.EXP38_DUMP_ROOT}"' in ev38 and f'FARM = "{reg.EXP38_FARM}"' in ev38
    assert len(reg.EXP38_SAVE_ROOTS) == 2 and reg.EXP38_DIR.endswith("exp_38_data_efficiency_claude")
    roots = reg.exp38_protected_roots([REPO])
    assert os.path.realpath(os.path.join(REPO, reg.EXP38_DIR)) in roots and len(roots) == 5


@pytest.fixture
def protected(tmp_path, monkeypatch):
    """A fake exp_38 save root, dump root and tree (farm + worklog folder) the registry treats as protected."""
    save38, dump38 = tmp_path / "exp38_save_root", tmp_path / "exp38_dumps"
    for d in (save38, dump38):
        d.mkdir()
    monkeypatch.setattr(reg, "EXP38_SAVE_ROOTS", (str(save38),))
    monkeypatch.setattr(reg, "EXP38_DUMP_ROOT", str(dump38))
    return save38, dump38


def _listing(d):
    return sorted(os.path.relpath(os.path.join(a, f), str(d)) for a, ds, fs in os.walk(str(d)) for f in ds + fs)


def test_equal_contained_and_containing_aliases_are_refused(tmp_path, protected):
    save38, _ = protected
    alias = tmp_path / "exp45_alias"
    alias.symlink_to(save38)
    for path in (alias, alias / "exp45_RFf025_l40_2x32", tmp_path, save38):   # equal (via the link), inside, contains
        assert reg.isolation_problems(str(path), "x", trees=[str(tmp_path / "tree")]), path
    assert reg.isolation_problems(str(tmp_path / "exp45_sibling"), "x", trees=[str(tmp_path / "tree")]) == []
    assert reg.isolation_problems(str(tmp_path), "x", trees=[str(tmp_path / "tree")], contains=False) == []
    with pytest.raises(reg.GateError, match="exp_38"):
        reg.check_isolation(str(alias), "the exp_45 save root", trees=[str(tmp_path / "tree")])


def test_the_storage_probe_and_the_ledger_refuse_an_aliased_root_before_writing(tmp_path, protected):
    save38, _ = protected
    alias = tmp_path / "alias"
    alias.symlink_to(save38)
    with pytest.raises(reg.GateError, match="exp_38"):
        reg.storage_gate("neuronic", str(alias), 0)
    with pytest.raises(reg.GateError, match="exp_38"):
        reg.reserve_storage(RUN, "a" * 32, str(alias / RUN), str(alias), need=1, floor=0)
    assert _listing(save38) == []                                    # no probe, lock or ledger file


def test_the_submitter_refuses_an_aliased_save_root_before_any_lock_or_owner(tmp_path, protected, monkeypatch):
    import types
    sub = _load(os.path.join(EXP, "exp45_submit_neuronic.py"), "exp45_submit_isolation")
    save38, _ = protected
    alias = tmp_path / "alias"
    alias.symlink_to(save38)
    monkeypatch.setattr(sub, "preflight", lambda *a: "a" * 40)
    calls = []

    def runner(argv, capture_output=True, text=True):
        calls.append(argv)
        out = "PartitionName=all MaxTime=7-00:00:00" if argv[0] == "scontrol" else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    for fn in (lambda: sub.submit(RUN, "FULL", "3-00:00:00", "flash", reg.FLASH_VERSION, False, False, root=str(alias),
                                  runner=runner),
               lambda: sub.abandon(RUN, "FULL", root=str(alias), runner=runner),
               lambda: sub.reconcile(RUN, "FULL", root=str(alias), runner=runner),
               lambda: sub.release(RUN, root=str(alias), runner=runner)):
        with pytest.raises(reg.GateError, match="exp_38"):
            fn()
    assert _listing(save38) == [] and not [c for c in calls if c[0] == "sbatch"]


def test_the_supervisor_refuses_an_aliased_save_root_before_any_claim_or_directory(tmp_path, protected, monkeypatch):
    sv = _load(os.path.join(EXP, "exp45_supervise.py"), "exp45_supervise_isolation")
    save38, _ = protected
    alias = tmp_path / "alias"
    alias.symlink_to(save38)
    monkeypatch.setitem(reg.PLATFORMS["a6000"], "save_root", str(alias))
    touched = []
    monkeypatch.setattr(sv, "common_gates", lambda a, p: ("rf", "025", "2x32", sys.executable, {"argv": []}, {}))
    monkeypatch.setattr(sv, "campaign_gates", lambda a, p: touched.append("campaign_gates") or ("f" * 40, "d" * 64, "c" * 64))
    monkeypatch.setattr(reg, "TRAINING_PLATFORM", "a6000")
    a = sv.parse(["--platform", "a6000", "--mode", "FULL", "--run", RUN, "--expect-sha", "f" * 40, "--token", "a" * 32])
    with pytest.raises(reg.GateError, match="exp_38"):
        sv.mode_campaign(a, "a6000")
    assert touched == [] and _listing(save38) == []


def test_the_evaluator_refuses_an_aliased_dump_root_or_farm(tmp_path, protected, monkeypatch):
    ev = _load(os.path.join(EXP, "exp45_eval.py"), "exp45_eval_isolation")
    _, dump38 = protected
    alias = tmp_path / "dump_alias"
    alias.symlink_to(dump38)
    monkeypatch.setattr(ev, "DUMP_ROOT", str(alias))
    with pytest.raises(reg.GateError, match="exp_38"):
        ev.campaign_gates()
    monkeypatch.setattr(ev, "DUMP_ROOT", str(tmp_path / "exp45_dumps"))
    monkeypatch.setattr(ev, "FARM", os.path.join(REPO, reg.EXP38_FARM, "nested"))   # inside exp_38's farm in this tree
    with pytest.raises(reg.GateError, match="exp_38"):
        ev.campaign_gates()


def _publication(root, col):
    """A published exp_45 results bundle in ``root`` (JSON + Markdown + one evidence file + generator_static())."""
    (root / reg.EXP_DIR).mkdir(parents=True)
    ev_rel = "outputs_FLAC/exp45_rf_data_curve/eval/E0a/x_metrics.json"
    for rel, data in [(ev_rel, b"e0 cell")] + [(rel, b"static " + rel.encode()) for rel in col.generator_static()]:
        os.makedirs(root / os.path.dirname(rel), exist_ok=True)
        (root / rel).write_bytes(data)
    manifest = {"artifacts": {ev_rel: hashlib.sha256(b"e0 cell").hexdigest()}, "runs": {}}
    rows = {(m, f, K): {"stats": {k: (1.0, 0.1, 5) for k in col.ev.METRIC_KEYS}, "source": "s"}
            for m, f in [("rf", t) for t in ("025", "050", "075")] + [("rf_bridge", "100")] for K in (8, 1)}
    col.publish(rows, [], manifest, repo=str(root))


def test_materialise_writes_nothing_into_exp38s_namespaces_or_through_an_alias(tmp_path, protected):
    import exp45_collect as col
    src = tmp_path / "published"
    _publication(src, col)
    exp38_owned = [r for r in col.generator_static() if r.startswith(reg.EXP38_DIR + "/")]
    assert exp38_owned
    dest = tmp_path / "generator_root"
    dest.mkdir()
    with pytest.raises(col.ev.EvalError, match="exp_38"):            # exp_38's references are not in dest
        col.materialise(str(dest), root=str(src))
    assert not os.path.exists(dest / reg.EXP38_DIR)                     # and nothing was written into its folder
    for rel in exp38_owned:                                             # present, identical: verified, never written
        os.makedirs(dest / os.path.dirname(rel), exist_ok=True)
        (dest / rel).write_bytes((src / rel).read_bytes())
    before = {rel: os.lstat(dest / rel).st_mtime_ns for rel in exp38_owned}
    out = col.materialise(str(dest), root=str(src))
    assert out["already_identical"] == len(exp38_owned) and before == {rel: os.lstat(dest / rel).st_mtime_ns
                                                                         for rel in exp38_owned}
    inside = dest / reg.EXP38_DIR / "nested_dest"                      # a destination inside exp_38's worklog folder
    inside.mkdir()
    with pytest.raises(col.ev.EvalError, match="exp_38"):
        col.materialise(str(inside), root=str(src))
    dest2 = tmp_path / "generator_root_2"                              # an aliased exp_45 directory inside dest
    (dest2 / "outputs_FLAC").mkdir(parents=True)
    (dest2 / "outputs_FLAC" / "exp45_rf_data_curve").symlink_to(protected[0])
    for rel in exp38_owned:
        os.makedirs(dest2 / os.path.dirname(rel), exist_ok=True)
        (dest2 / rel).write_bytes((src / rel).read_bytes())
    with pytest.raises(col.ev.EvalError, match="alias"):
        col.materialise(str(dest2), root=str(src))
    assert _listing(protected[0]) == []


def test_the_exp38_import_materialiser_refuses_a_protected_or_aliased_destination(tmp_path, protected):
    """exp45_table.materialise copies exp_38's OWN verified files to their registered relative paths in the
    generator root (plan r3 §3.2): the root itself may not equal or lie inside a protected root (an exp_38 save /
    dump root, or any tree's exp_38 farm / worklog folder), and no file is written through an aliased directory."""
    import exp45_table as tbl
    src = tmp_path / "bundle"
    rel = "outputs_FLAC/exp38_data_efficiency/eval/de_yaw_f025/x.json"
    os.makedirs(src / os.path.dirname(rel))
    (src / rel).write_bytes(b"cell")
    hashes = {rel: hashlib.sha256(b"cell").hexdigest()}
    for bad in (protected[0], tmp_path / "tree" / reg.EXP38_DIR / "x", tmp_path / "tree" / reg.EXP38_FARM):
        os.makedirs(bad, exist_ok=True)
        with pytest.raises(tbl.TableError, match="exp_38"):
            tbl.materialise(str(src), hashes, str(bad))
    gen = tmp_path / "generator_root"
    (gen / "outputs_FLAC").mkdir(parents=True)
    (gen / "outputs_FLAC" / "exp38_data_efficiency").symlink_to(protected[0])     # an aliased exp_38 farm parent
    with pytest.raises(tbl.TableError, match="alias"):
        tbl.materialise(str(src), hashes, str(gen))
    assert _listing(protected[0]) == []
    ok = tmp_path / "generator_root_ok"
    ok.mkdir()
    tbl.materialise(str(src), hashes, str(ok))                                      # the registered path: copied
    assert (ok / rel).read_bytes() == b"cell"
