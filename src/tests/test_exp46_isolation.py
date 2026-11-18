"""exp_46 — isolation from exp_38's AND exp_45's roots (plan_c4flip_data_curve.md r3 §3.2 "Isolation"; Codex plan
r1 finding 3, r2 finding 1; inherited from exp_45's Codex code r2 finding 6).

The protected roots are read from their owners' own files: exp_38's two save roots (A6000 NAS, neuronic), its dump
root, its snapshot namespace and -- in every tree involved -- its farm, its worklog folder and its A6000 SMOKE folder;
exp_45's two save roots (with its storage ledger, owners, locks and run dirs under them), its dump root, its A6000
campaign / publication trees and -- in every tree -- its farm and its worklog folder; all compared as realpaths
(symlinks followed). Before ANY mutation (probe, lock, ledger, owner record, snapshot, link, directory), an exp_46
output namespace -- a save root, the dump root, the farm, a snapshot root -- that equals, contains or lies inside a
protected root refuses, so an alias (a symlink, a bind of the same directory) can never route exp_46's writes into
exp_38's or exp_45's: a matrix of every protected root x {direct, nested, symlinked} x every mutating entry point,
with zero writes. A materialisation destination may not equal or lie inside a protected root and writes no file
through an aliased directory; exp46_collect.py writes nothing into exp_38's namespaces (exp_38's reference files must
already be in the destination, byte-identical). The one exception is exp_45's, kept exactly: exp46_table.materialise
copies exp_38's OWN verified import files to their registered relative paths in a generator root -- and no path
either materialiser handles lies in an exp_45 namespace.

Written by the exp_46 Coder (Claude Opus 5.5).
"""
import hashlib
import importlib.util
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude")
sys.path.insert(0, EXP)
import exp46_registry as reg  # noqa: E402

RUN = "exp46_C4Ff025_l40_2x32"


@pytest.fixture(autouse=True)
def _exp45_ledger_stand_in(tmp_path, monkeypatch):
    """C3 (plan r3 §3.2 "Storage") reads exp_45's REAL ledger path at run time; here an absent stand-in, so the real
    exp_45 state never decides a test."""
    monkeypatch.setattr(reg, "EXP45_LEDGER", str(tmp_path / "exp45_ledger_stand_in" / ".exp45_storage.json"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


X45F = "1e37ead2a352bd78d50cdda9ed02e95738a4ad04"


def _at_x45f(rel):
    import subprocess
    return subprocess.run(["git", "-C", REPO, "show", f"{X45F}:{rel}"], capture_output=True, text=True, check=True).stdout


def test_the_protected_roots_are_exp38s_and_exp45s_own_values():
    """The protected roots are read from exp_38's own registry / evaluator and from exp_45's own kit at X45f (never
    retyped); exp_38's snapshot namespace is exp_45's inherited SNAPSHOT_ROOT, its SMOKE folder exp_45's supervisor's."""
    src38 = open(os.path.join(REPO, reg.EXP38_DIR, "exp38_registry.py")).read()
    ev38 = open(os.path.join(REPO, reg.EXP38_DIR, "exp38_eval.py")).read()
    for root in reg.EXP38_SAVE_ROOTS:
        assert f'"save_root": "{root}"' in src38, root
    assert f'DUMP_ROOT = "{reg.EXP38_DUMP_ROOT}"' in ev38 and f'FARM = "{reg.EXP38_FARM}"' in ev38
    assert len(reg.EXP38_SAVE_ROOTS) == 2 and reg.EXP38_DIR.endswith("exp_38_data_efficiency_claude")
    e45 = "worklog/worklog_yixun/exp_45_rf_data_curve_claude"
    assert reg.EXP45_DIR == e45
    reg45, ev45, sv45 = (_at_x45f(f"{e45}/exp45_{n}.py") for n in ("registry", "eval", "supervise"))
    for root in reg.EXP45_SAVE_ROOTS:
        assert f'"save_root": "{root}"' in reg45, root
    assert f'DUMP_ROOT = "{reg.EXP45_DUMP_ROOT}"' in ev45 and f'FARM = "{reg.EXP45_FARM}"' in ev45
    assert f'SNAPSHOT_ROOT = "{reg.EXP38_SNAPSHOT_ROOT}"' in ev45 and f'"{reg.EXP38_SNAPSHOT_ROOT}"' in sv45
    assert f'"{reg.EXP38_SMOKE_DIR.split("/")[1]}"' in sv45 and reg.EXP38_SMOKE_DIR == "outputs_FLAC/exp38_smoke"
    assert f'CAMPAIGN_ROOT = {{"a6000": "{reg.EXP45_A6000_ROOTS[0]}"}}' in reg45
    assert "~/codespace/exp-45-publish" in _at_x45f(f"{e45}/plan_rf_data_curve.md")
    assert reg.EXP45_A6000_ROOTS[1] == "/home/yixunhu/codespace/exp-45-publish"
    roots = reg.protected_roots([REPO])
    assert len(roots) == len(reg.PROTECTED_ABSOLUTE) + len(reg.PROTECTED_IN_TREE) == 9 + 5
    assert os.path.realpath(os.path.join(REPO, reg.EXP45_DIR)) in roots
    assert os.path.realpath(os.path.join(REPO, reg.EXP38_DIR)) in roots


ABS_KINDS = ("exp38 NAS save root", "exp38 neuronic save root", "exp38 dump root", "exp38 snapshots",
             "exp45 NAS save root", "exp45 neuronic save root", "exp45 dump root", "exp45 campaign", "exp45 publish")


@pytest.fixture
def protected_all(tmp_path, monkeypatch):
    """Stand-ins for every absolute protected root (the per-tree ones are checked in any tree by name)."""
    stand = {k: tmp_path / ("protected_" + k.replace(" ", "_")) for k in ABS_KINDS}
    for d in stand.values():
        d.mkdir()
    monkeypatch.setattr(reg, "PROTECTED_ABSOLUTE", tuple(str(v) for v in stand.values()))
    return stand


@pytest.fixture
def protected(protected_all):
    """exp_45's two-root fixture shape: an exp_38 save root and dump root the registry treats as protected."""
    return protected_all["exp38 NAS save root"], protected_all["exp38 dump root"]


def _listing(d):
    return sorted(os.path.relpath(os.path.join(a, f), str(d)) for a, ds, fs in os.walk(str(d)) for f in ds + fs)


def test_equal_contained_and_containing_aliases_are_refused(tmp_path, protected):
    save38, _ = protected
    alias = tmp_path / "exp46_alias"
    alias.symlink_to(save38)
    for path in (alias, alias / "exp46_C4Ff025_l40_2x32", tmp_path, save38):   # equal (via the link), inside, contains
        assert reg.isolation_problems(str(path), "x", trees=[str(tmp_path / "tree")]), path
    assert reg.isolation_problems(str(tmp_path / "exp46_sibling"), "x", trees=[str(tmp_path / "tree")]) == []
    assert reg.isolation_problems(str(tmp_path), "x", trees=[str(tmp_path / "tree")], contains=False) == []
    with pytest.raises(reg.GateError, match="exp_38"):
        reg.check_isolation(str(alias), "the exp_46 save root", trees=[str(tmp_path / "tree")])


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
    sub = _load(os.path.join(EXP, "exp46_submit_neuronic.py"), "exp46_submit_isolation")
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
    sv = _load(os.path.join(EXP, "exp46_supervise.py"), "exp46_supervise_isolation")
    save38, _ = protected
    alias = tmp_path / "alias"
    alias.symlink_to(save38)
    monkeypatch.setitem(reg.PLATFORMS["a6000"], "save_root", str(alias))
    touched = []
    monkeypatch.setattr(sv, "common_gates", lambda a, p: ("c4f", "025", "2x32", sys.executable, {"argv": []}, {}))
    monkeypatch.setattr(sv, "campaign_gates", lambda a, p: touched.append("campaign_gates") or ("f" * 40, "d" * 64, "c" * 64))
    monkeypatch.setattr(reg, "TRAINING_PLATFORM", "a6000")
    a = sv.parse(["--platform", "a6000", "--mode", "FULL", "--run", RUN, "--expect-sha", "f" * 40, "--token", "a" * 32])
    with pytest.raises(reg.GateError, match="exp_38"):
        sv.mode_campaign(a, "a6000")
    assert touched == [] and _listing(save38) == []


def test_the_evaluator_refuses_an_aliased_dump_root_or_farm(tmp_path, protected, monkeypatch):
    ev = _load(os.path.join(EXP, "exp46_eval.py"), "exp46_eval_isolation")
    _, dump38 = protected
    alias = tmp_path / "dump_alias"
    alias.symlink_to(dump38)
    monkeypatch.setattr(ev, "DUMP_ROOT", str(alias))
    with pytest.raises(reg.GateError, match="exp_38"):
        ev.campaign_gates()
    monkeypatch.setattr(ev, "DUMP_ROOT", str(tmp_path / "exp46_dumps"))
    monkeypatch.setattr(ev, "FARM", os.path.join(REPO, reg.EXP38_FARM, "nested"))   # inside exp_38's farm in this tree
    with pytest.raises(reg.GateError, match="exp_38"):
        ev.campaign_gates()


def _publication(root, col):
    """A published exp_46 results bundle in ``root`` (JSON + Markdown + one evidence file + generator_static())."""
    (root / reg.EXP_DIR).mkdir(parents=True)
    ev_rel = "outputs_FLAC/exp46_c4flip_data_curve/eval/E0a/x_metrics.json"
    for rel, data in [(ev_rel, b"e0 cell")] + [(rel, b"static " + rel.encode()) for rel in col.generator_static()]:
        os.makedirs(root / os.path.dirname(rel), exist_ok=True)
        (root / rel).write_bytes(data)
    manifest = {"artifacts": {ev_rel: hashlib.sha256(b"e0 cell").hexdigest()}, "runs": {}}
    rows = {(m, f, K): {"stats": {k: (1.0, 0.1, 5) for k in col.ev.METRIC_KEYS}, "source": "s"}
            for m, f in [("c4f", reg.parse_run_id(r)[1]) for r in reg.campaign_runs()] for K in (8, 1)}
    col.publish(rows, [], manifest, repo=str(root))


def test_materialise_writes_nothing_into_exp38s_namespaces_or_through_an_alias(tmp_path, protected):
    import exp46_collect as col
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
    dest2 = tmp_path / "generator_root_2"                              # an aliased exp_46 directory inside dest
    (dest2 / "outputs_FLAC").mkdir(parents=True)
    (dest2 / "outputs_FLAC" / "exp46_c4flip_data_curve").symlink_to(protected[0])
    for rel in exp38_owned:
        os.makedirs(dest2 / os.path.dirname(rel), exist_ok=True)
        (dest2 / rel).write_bytes((src / rel).read_bytes())
    with pytest.raises(col.ev.EvalError, match="alias"):
        col.materialise(str(dest2), root=str(src))
    assert _listing(protected[0]) == []


def test_the_exp38_import_materialiser_refuses_a_protected_or_aliased_destination(tmp_path, protected):
    """exp46_table.materialise copies exp_38's OWN verified files to their registered relative paths in the
    generator root (plan r3 §3.2): the root itself may not equal or lie inside a protected root (an exp_38 save /
    dump root, or any tree's exp_38 farm / worklog folder), and no file is written through an aliased directory."""
    import exp46_table as tbl
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


# ------------------------------------------------------------------ exp_46: every protected root, every mutating path
IN_TREE = {"exp38 farm": reg.EXP38_FARM, "exp38 worklog": reg.EXP38_DIR, "exp38 SMOKE folder": reg.EXP38_SMOKE_DIR,
           "exp45 farm": reg.EXP45_FARM, "exp45 worklog": reg.EXP45_DIR}


def _runner(calls):
    import types

    def runner(argv, capture_output=True, text=True):
        calls.append(argv)
        out = "PartitionName=all MaxTime=7-00:00:00" if argv[0] == "scontrol" else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    return runner


@pytest.mark.parametrize("how", ["direct", "nested", "symlinked"])
@pytest.mark.parametrize("kind", ABS_KINDS + tuple(IN_TREE))
def test_every_mutating_path_refuses_every_protected_root_before_any_write(tmp_path, protected_all, monkeypatch, kind,
                                                                         how):
    """exp_46 plan r3 §3.2 "Isolation" (Codex plan r1 finding 3, r2 finding 1): a destination equal to, inside, or
    symlinked onto any exp_38 / exp_45 protected root refuses -- realpath, symlinks followed -- at every mutating entry
    point (the storage probe and ledger, submit / abandon / reconcile / release, the supervisor, the evaluator's gates,
    prepare, the evaluation snapshot writer, prepare-e0) before any write: the protected root is left untouched."""
    root = protected_all[kind] if kind in protected_all else tmp_path / "tree" / IN_TREE[kind]
    root.mkdir(parents=True, exist_ok=True)
    dest = {"direct": root, "nested": root / "deeper" / "dest", "symlinked": tmp_path / "alias_dest"}[how]
    if how == "symlinked":
        dest.symlink_to(root)
    elif how == "nested":
        dest.mkdir(parents=True)
    before = _listing(root)
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        reg.check_isolation(str(dest), "x")
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        reg.storage_gate("neuronic", str(dest), 0)
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        reg.reserve_storage(RUN, "a" * 32, str(dest / RUN), str(dest), need=1, floor=0)
    sub = _load(os.path.join(EXP, "exp46_submit_neuronic.py"), "exp46_submit_isolation_matrix")
    monkeypatch.setattr(sub, "preflight", lambda *a: "a" * 40)
    calls = []
    for fn in (lambda: sub.submit(RUN, "FULL", "3-00:00:00", "flash", reg.FLASH_VERSION, False, False, root=str(dest),
                                  runner=_runner(calls)),
               lambda: sub.abandon(RUN, "FULL", root=str(dest), runner=_runner(calls)),
               lambda: sub.reconcile(RUN, "FULL", root=str(dest), runner=_runner(calls)),
               lambda: sub.release(RUN, root=str(dest), runner=_runner(calls))):
        with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
            fn()
    assert not [c for c in calls if c[0] == "sbatch"]
    sv = _load(os.path.join(EXP, "exp46_supervise.py"), "exp46_supervise_isolation_matrix")
    monkeypatch.setitem(reg.PLATFORMS["a6000"], "save_root", str(dest))
    monkeypatch.setattr(sv, "common_gates", lambda a, p: ("c4f", "025", "2x32", sys.executable, {"argv": []}, {}))
    monkeypatch.setattr(sv, "campaign_gates", lambda a, p: (_ for _ in ()).throw(AssertionError("gates reached")))
    monkeypatch.setattr(reg, "TRAINING_PLATFORM", "a6000")
    a = sv.parse(["--platform", "a6000", "--mode", "FULL", "--run", RUN, "--expect-sha", "f" * 40, "--token", "a" * 32])
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        sv.mode_campaign(a, "a6000")
    monkeypatch.setitem(reg.PLATFORMS["a6000"], "save_root", str(tmp_path / "exp46_save_root"))
    monkeypatch.setitem(sv.SNAPSHOT_ROOT, "a6000", str(dest))                  # the A6000 snapshot root aliased
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        sv.mode_campaign(a, "a6000")
    ev = _load(os.path.join(EXP, "exp46_eval.py"), "exp46_eval_isolation_matrix")
    for attr in ("DUMP_ROOT", "FARM", "SNAPSHOT_ROOT"):
        with monkeypatch.context() as m:
            m.setattr(ev, attr, str(dest))
            with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
                ev.campaign_gates()
            with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
                ev.prepare_e0()
    with monkeypatch.context() as m:
        m.setattr(ev, "FARM", str(dest))
        with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
            ev.prepare(RUN)
    with monkeypatch.context() as m:
        m.setattr(ev, "SNAPSHOT_ROOT", str(dest))
        with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
            ev.eval_snapshot("c" * 64, create=True)
    import exp46_collect as col
    import exp46_table as tbl
    with pytest.raises(col.ev.EvalError, match="exp_38 / exp_45"):                 # the publication materialiser
        col.materialise(str(dest), root=str(tmp_path / "published"))
    with pytest.raises(tbl.TableError, match="exp_38 / exp_45"):                   # the exp_38 import materialiser
        tbl.materialise(str(tmp_path / "bundle"), {}, str(dest))
    assert _listing(root) == before                                         # zero writes into the protected root
    assert not (tmp_path / "exp46_save_root").exists()


def test_the_exp38_materialisation_exception_is_kept_exactly_and_never_reaches_exp45():
    """exp_45's exception (its Codex code r2 finding 6), unchanged: exp46_table.materialise copies exp_38's OWN
    verified import files -- the exp_38 configs, its E0 / reference / decision records, its farm cells and receipts --
    to their registered relative paths, and exp46_collect.materialise only verifies exp_38's files. No path either
    handles lies in an exp_45 namespace (exp_46 has no bridge)."""
    import exp46_collect as col
    import exp46_table as tbl
    exp45 = (reg.EXP45_DIR + "/", reg.EXP45_FARM + "/", "outputs_FLAC/exp45_")
    static = col.generator_static()
    assert [r for r in static if r.startswith((reg.EXP38_DIR + "/", reg.EXP38_FARM + "/"))]
    assert not [r for r in static if r.startswith(exp45) or "exp_45" in r or "exp45" in r]
    assert tbl.EXP38_FARM == reg.EXP38_FARM and not [c for c in tbl.EXP38_CONFIGS if c.startswith(exp45)]
    assert all(v[3].startswith(reg.EXP38_FARM + "/") for v in tbl.exp38_cell_stems().values())


# ------------------------------------------------------------------ fix round 1, Codex code r1 finding 1: no bytecode
def _copied_tree(tmp_path):
    """A throw-away checkout (a .git marker) holding the exp_46 kit and exp_38's tools, copied without bytecode."""
    import shutil
    root = tmp_path / "tree"
    (root / ".git").mkdir(parents=True)
    for rel in (reg.EXP_DIR, reg.EXP38_DIR):
        (root / rel).mkdir(parents=True)
        for name in sorted(os.listdir(os.path.join(REPO, rel))):
            if name.endswith(".py"):
                shutil.copyfile(os.path.join(REPO, rel, name), root / rel / name)
    return root


def _bytecode(root):
    """Every __pycache__ directory and .pyc file under ``root``, relative."""
    return sorted(os.path.relpath(os.path.join(a, n), str(root)) for a, ds, fs in os.walk(str(root))
                  for n in ds + fs if n == "__pycache__" or n.endswith(".pyc"))


@pytest.mark.parametrize("argv", [["--verify", "x.txt.gz"], ["--build", "out/x.txt.gz"]], ids=["verify", "build"])
def test_the_standalone_manifest_tool_writes_no_bytecode_into_exp38s_folder(tmp_path, argv):
    """Codex code r1 finding 1 (HIGH): the documented plain invocation `python exp46_content_manifest.py --build /
    --verify` (no -B, PYTHONDONTWRITEBYTECODE ignored by -I) imports exp_38's tools from exp_38's folder; it disables
    bytecode writes before those imports, so no __pycache__ appears there -- nor anywhere else in the tree."""
    import subprocess
    root = _copied_tree(tmp_path)
    assert _bytecode(root) == []
    res = subprocess.run([sys.executable, "-I", str(root / reg.EXP_DIR / "exp46_content_manifest.py"), *argv],
                         cwd=str(root), capture_output=True, text=True, timeout=300)
    assert res.returncode in (1, 2), res.stderr[-2000:]                 # refused or FAIL: the imports did happen
    assert "Traceback" not in res.stderr, res.stderr[-2000:]
    assert _bytecode(root) == [], _bytecode(root)


def test_a_supervisor_process_writes_no_bytecode_into_exp38s_folder(tmp_path):
    """The supervisor imports exp_38's subset audit and content manifest from exp_38's folder (the sbatch runs it with
    -B; a standalone DRY_RUN does not): importing it disables bytecode writes for the process first."""
    import subprocess
    root = _copied_tree(tmp_path)
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import exp46_supervise; import exp38_subset_audit; "
            "import exp38_content_manifest; import exp46_content_manifest; print(sys.dont_write_bytecode)")
    res = subprocess.run([sys.executable, "-I", "-c", code, str(root / reg.EXP_DIR)], cwd=str(root),
                         capture_output=True, text=True, timeout=300)
    assert res.returncode == 0 and res.stdout.strip().endswith("True"), res.stderr[-2000:]
    # the supervisor module's own bytecode (written while it is imported, inside exp_46's folder) is exp_46's; nothing
    # may appear anywhere else
    assert [p for p in _bytecode(root) if not p.startswith(reg.EXP_DIR + "/")] == [], _bytecode(root)
