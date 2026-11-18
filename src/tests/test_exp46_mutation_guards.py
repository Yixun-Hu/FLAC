"""exp_46 fix round 2 -- Codex code r2 finding 1 (HIGH): the ACTUAL destination of every mutation below a checked root.

An isolated root is not enough: a symlinked CHILD below it (the storage lock, the owner directory, a submit lock, a
ledger temp file, a run-dir child; a dangling link included, whose absent target open() / mkdir() would create) routes
the write into another namespace. exp46_registry.check_destination validates the destination itself -- lexically
below its root, no symlinked component from the root down (alias_problem), the resolved path isolated -- and every
mutation of the kit goes through a guarded primitive that calls it first (the audit at the end of this file).

Written by the exp_46 Coder (Claude Opus 5.5).
"""
import ast
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude")
sys.path.insert(0, EXP)
import exp46_registry as reg  # noqa: E402

RUN = "exp46_C4Ff025_l40_2x32"
ABS_KINDS = ("exp38 NAS save root", "exp38 neuronic save root", "exp38 dump root", "exp38 snapshots",
             "exp45 NAS save root", "exp45 neuronic save root", "exp45 dump root", "exp45 campaign", "exp45 publish")
IN_TREE = {"exp38 farm": reg.EXP38_FARM, "exp38 worklog": reg.EXP38_DIR, "exp38 SMOKE folder": reg.EXP38_SMOKE_DIR,
           "exp45 farm": reg.EXP45_FARM, "exp45 worklog": reg.EXP45_DIR}
KINDS = ABS_KINDS + tuple(IN_TREE)


@pytest.fixture(autouse=True)
def _exp45_ledger_stand_in(tmp_path, monkeypatch):
    """C3 reads exp_45's REAL ledger path at run time; here an absent stand-in."""
    monkeypatch.setattr(reg, "EXP45_LEDGER", str(tmp_path / "exp45_ledger_stand_in" / ".exp45_storage.json"))


@pytest.fixture
def protected_all(tmp_path, monkeypatch):
    """Stand-ins for every absolute protected root (the per-tree ones are protected in any tree by name)."""
    stand = {k: tmp_path / ("protected_" + k.replace(" ", "_")) for k in ABS_KINDS}
    for d in stand.values():
        d.mkdir()
    monkeypatch.setattr(reg, "PROTECTED_ABSOLUTE", tuple(str(v) for v in stand.values()))
    return stand


def protected_root(kind, protected_all, tmp_path):
    root = protected_all[kind] if kind in protected_all else tmp_path / "other_tree" / IN_TREE[kind]
    root.mkdir(parents=True, exist_ok=True)
    return root


def listing(d):
    return sorted(os.path.relpath(os.path.join(a, f), str(d)) for a, ds, fs in os.walk(str(d)) for f in ds + fs)


# ------------------------------------------------------------------ the primitives, at every depth, every root
def _alias(root, prot, depth, target):
    """A symlinked component below ``root`` pointing into the protected ``prot`` -- at the leaf, the parent or deeper;
    to an ABSENT target or an existing directory -- and the destination path written through it."""
    tgt = prot / ("absent_" + depth) if target == "absent" else prot
    if depth == "leaf":
        (root / "x.lock").symlink_to(tgt)
        return root / "x.lock"
    if depth == "parent":
        (root / "sub").symlink_to(tgt)
        return root / "sub" / "x.json"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "c").symlink_to(tgt)
    return root / "a" / "b" / "c" / "d" / "x.json"


def _primitives(path, root, tmp_path):
    src = tmp_path / "source.bin"
    src.write_bytes(b"source")
    mover = root / "mover"
    mover.mkdir(exist_ok=True)
    return {"open a": lambda: reg.open_below(str(path), "a", str(root), "a lock"),
            "open ab": lambda: reg.open_below(str(path), "ab", str(root), "a log"),
            "open w": lambda: reg.open_below(str(path), "w", str(root), "a file"),
            "open x": lambda: reg.open_below(str(path), "x", str(root), "a file"),
            "makedirs": lambda: reg.makedirs_below(str(path), str(root), "a directory"),
            "mkdir": lambda: reg.mkdir_below(str(path), str(root), "a claim"),
            "atomic write": lambda: reg.atomic_write_bytes(str(path), b"x", root=str(root)),
            "sealed write": lambda: reg.write_sealed(str(path), {"a": 1}, root=str(root)),
            "no-clobber publication": lambda: reg.publish_no_clobber(str(path), b"x", root=str(root)),
            "copy": lambda: reg.copyfile_below(str(src), str(path), str(root), "a copy"),
            "symlink": lambda: reg.symlink_below(str(src), str(path), str(root), "a link"),
            "rename": lambda: reg.rename_below(str(mover), str(path), str(root), "a move")}


PRIMITIVES = ("open a", "open ab", "open w", "open x", "makedirs", "mkdir", "atomic write", "sealed write",
              "no-clobber publication", "copy", "symlink", "rename")


@pytest.mark.parametrize("target", ["absent", "existing"])
@pytest.mark.parametrize("depth", ["leaf", "parent", "deep"])
@pytest.mark.parametrize("kind", KINDS)
def test_every_primitive_refuses_a_symlinked_child_into_every_protected_root(tmp_path, protected_all, kind, depth,
                                                                              target):
    prot = protected_root(kind, protected_all, tmp_path)
    before = listing(prot)
    for name in PRIMITIVES:
        root = tmp_path / f"exp46_root_{name.replace(' ', '_')}"
        root.mkdir()
        path = _alias(root, prot, depth, target)
        with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
            _primitives(path, root, tmp_path)[name]()
        assert listing(prot) == before, name                        # zero writes into the protected root


def test_the_primitives_write_at_the_lexical_place_and_refuse_any_other_alias_or_place(tmp_path, protected_all):
    """No alias: each primitive writes exactly where its path names. A symlinked component to an UNPROTECTED place is
    refused too (the write would still leave exp_46's namespace), and so is a path outside its root."""
    for name in PRIMITIVES:
        root = tmp_path / f"ok_{name.replace(' ', '_')}"
        root.mkdir()
        path = root / "sub" / "x.out"
        if name not in ("makedirs",):
            path.parent.mkdir()
        out = _primitives(path, root, tmp_path)[name]()
        if hasattr(out, "close"):
            out.close()
        assert os.path.lexists(path) and not os.path.islink(path) or name == "symlink", name
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for name in PRIMITIVES:
        root = tmp_path / f"alias_{name.replace(' ', '_')}"
        root.mkdir()
        (root / "sub").symlink_to(elsewhere)
        with pytest.raises(reg.GateError, match="alias"):
            _primitives(root / "sub" / "x.out", root, tmp_path)[name]()
        with pytest.raises(reg.GateError, match="outside its checked root"):
            _primitives(tmp_path / "outside.out", root, tmp_path)[name]()
    assert listing(elsewhere) == [] and not os.path.lexists(tmp_path / "outside.out")


def test_the_atomic_writers_never_write_through_a_planted_temp_name(tmp_path, protected_all):
    """The ledger / receipt temp file (.<name>.tmp.<pid>) is created exclusively and never through a link: a link
    planted at that exact name into a protected root refuses, nothing written; the target is untouched."""
    prot = protected_all["exp45 neuronic save root"]
    root = tmp_path / "save"
    root.mkdir()
    for name in (reg.STORAGE_LEDGER, "receipt.json", "receipt.json.sha256"):
        (root / f".{name}.tmp.{os.getpid()}").symlink_to(prot / f"absent_{name}")
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        reg.atomic_write_bytes(str(root / reg.STORAGE_LEDGER), b"{}", root=str(root))
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        reg.write_sealed(str(root / "receipt.json"), {"a": 1}, root=str(root))
    assert listing(prot) == [] and not os.path.lexists(root / reg.STORAGE_LEDGER)
    assert not os.path.lexists(root / "receipt.json")


def test_the_registered_exp38_import_exception_admits_exactly_exp38s_namespaces_below_its_root(tmp_path, protected_all):
    """exp46_table.materialise's registered exception (exp_45's r2-6): below a generator root, exp_38's worklog and
    farm namespaces may be written with ``allow`` -- nothing else: exp_45's namespaces, exp_38's SMOKE folder and every
    absolute protected root still refuse; without ``allow`` exp_38's namespaces refuse too."""
    gen = tmp_path / "generator_root"
    gen.mkdir()
    allow = (reg.EXP38_DIR, reg.EXP38_FARM)
    for rel in allow:
        dest = gen / rel / "x.json"
        with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
            reg.check_destination(str(dest), str(gen), "an exp_38 import file")
        assert reg.check_destination(str(dest), str(gen), "an exp_38 import file", allow=allow) == str(dest)
    for rel in (reg.EXP45_DIR, reg.EXP45_FARM, reg.EXP38_SMOKE_DIR):
        with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
            reg.check_destination(str(gen / rel / "x.json"), str(gen), "a file", allow=allow)
    (gen / "outputs_FLAC").mkdir()
    (gen / "outputs_FLAC" / "exp38_data_efficiency").symlink_to(protected_all["exp38 NAS save root"])
    with pytest.raises(reg.GateError, match="alias"):
        reg.check_destination(str(gen / reg.EXP38_FARM / "x.json"), str(gen), "a farm cell", allow=allow)
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        reg.check_destination(str(protected_all["exp38 dump root"] / "x"), str(protected_all["exp38 dump root"]),
                              "a file", allow=allow)


# ------------------------------------------------------------------ the registry's entry points: aliased children
def _targets(prot, target):
    """An alias target in the protected root: an ABSENT file (open() / mkdir() would create it) or an existing one."""
    if target == "absent":
        return prot / "absent_target"
    theirs = prot / "theirs"
    theirs.write_text("theirs\n")
    return theirs


LEDGER_CHILDREN = {"the storage-ledger lock": reg.STORAGE_LOCK, "the storage ledger": reg.STORAGE_LEDGER,
                   "the ledger temp file": f".{reg.STORAGE_LEDGER}.tmp.{os.getpid()}"}


@pytest.mark.parametrize("target", ["absent", "existing"])
@pytest.mark.parametrize("child", sorted(LEDGER_CHILDREN))
@pytest.mark.parametrize("kind", KINDS)
def test_the_storage_ledger_refuses_an_aliased_child_before_any_write(tmp_path, protected_all, kind, child, target):
    """Codex code r2 finding 1's probe: a pre-existing .exp46_storage.lock symlink to an ABSENT file inside a protected
    root, reached by an otherwise valid reservation, created that file; the same for the ledger and its temp name.
    Every ledger entry point now refuses (GateError) before any write; the protected root is untouched."""
    prot = protected_root(kind, protected_all, tmp_path)
    link_to = _targets(prot, target)
    before = listing(prot)
    root = tmp_path / "exp46_save_root"
    root.mkdir()
    (root / LEDGER_CHILDREN[child]).symlink_to(link_to)
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        reg.reserve_storage(RUN, "a" * 32, str(root / RUN), str(root), need=1, floor=0)
    for fn in (lambda: reg.recheck_storage(RUN, "a" * 32, str(root), floor=0),
               lambda: reg.release_storage(RUN, "a" * 32, str(root))):
        if child != "the ledger temp file":                     # (they never write the ledger: no temp name)
            with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
                fn()
    assert listing(prot) == before and (target == "absent" or link_to.read_text() == "theirs\n")


@pytest.mark.parametrize("target", ["absent", "existing"])
@pytest.mark.parametrize("kind", KINDS)
def test_the_environment_receipt_lock_refuses_an_alias_before_any_write(tmp_path, protected_all, monkeypatch, kind,
                                                                       target):
    """The receipt lock beside the environment receipt (and the hygiene gate's probe of it): an aliased lock refuses
    before the environment is probed and before any write."""
    prot = protected_root(kind, protected_all, tmp_path)
    link_to = _targets(prot, target)
    before = listing(prot)
    tree = tmp_path / "tree"
    (tree / reg.EXP_DIR).mkdir(parents=True)
    (tree / reg.ENV_RECEIPT_LOCK).symlink_to(link_to)
    monkeypatch.setattr(reg, "env_receipt", lambda *a, **k: (_ for _ in ()).throw(AssertionError("environment probed")))
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        reg.ensure_env_receipt("neuronic", repo=str(tree))
    tmp_name = f"{reg.EXP_DIR}/.c4flip_data_curve_env_receipt_neuronic.json.tmp.1"
    (tree / tmp_name).write_text("x")
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        reg._stale_env_receipt_temps([tmp_name], str(tree))
    assert listing(prot) == before


@pytest.mark.parametrize("kind", KINDS)
def test_a_run_dir_aliased_into_a_protected_root_refuses_the_registrys_run_dir_writers(tmp_path, protected_all, kind):
    """The run dir is a child of the checked save root: an aliased run dir refuses the receipt retention and the commit
    marker before any write (they would otherwise create receipts/ and the marker inside the protected root)."""
    import json
    prot = protected_root(kind, protected_all, tmp_path)
    before = listing(prot)
    root = tmp_path / "exp46_save_root"
    root.mkdir()
    (root / RUN).symlink_to(prot)
    rd = str(root / RUN)
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    (tmp_path / "receipt.json.sha256").write_text("x  receipt.json\n")
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        reg.retain_receipts(rd, {"e0": str(receipt)})
    record = tmp_path / "at_launch.json"
    record.write_text(json.dumps({"status": "COMPLETE", "run": RUN, "training_receipt_sha256": "7" * 64}))
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        reg.write_commit_marker(rd, RUN, "7" * 64, str(record))
    assert listing(prot) == before


# ------------------------------------------------------------------ the submitter's entry points: aliased children
def _load(name, alias):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, os.path.join(EXP, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _runner(calls):
    import types

    def runner(argv, capture_output=True, text=True):
        calls.append(argv)
        out = "PartitionName=all MaxTime=7-00:00:00" if argv[0] == "scontrol" else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    return runner


SUBMIT_CHILDREN = {"the owners folder": ".exp46_owners", "the submit lock": f".exp46_submit_{RUN}__FULL.lock"}


@pytest.mark.parametrize("child", sorted(SUBMIT_CHILDREN))
@pytest.mark.parametrize("kind", KINDS)
def test_the_submitter_refuses_an_aliased_child_before_any_write_reservation_or_sbatch(tmp_path, protected_all,
                                                                                   monkeypatch, kind, child):
    """Codex code r2 finding 1: a symlinked .exp46_owners routed the owner record into another namespace (and a
    symlinked submit lock created its absent target there). Every destination of the submission is validated before
    the first write: no reservation is taken, no sbatch issued, the protected root untouched."""
    prot = protected_root(kind, protected_all, tmp_path)
    before = listing(prot)
    root = tmp_path / "exp46_save_root"
    root.mkdir()
    (root / SUBMIT_CHILDREN[child]).symlink_to(prot if child == "the owners folder" else prot / "absent.lock")
    monkeypatch.setattr(reg, "free_bytes", lambda path: 10 ** 15)     # room enough: only the alias may refuse
    sub = _load("exp46_submit_neuronic.py", "exp46_submit_guards")
    monkeypatch.setattr(sub, "preflight", lambda *a: "a" * 40)
    calls = []
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        sub.submit(RUN, "FULL", "3-00:00:00", "flash", reg.FLASH_VERSION, False, False, root=str(root),
                   runner=_runner(calls))
    assert listing(prot) == before and not [c for c in calls if c[0] == "sbatch"]
    assert reg.storage_reservation(RUN, str(root)) is None                # no stranded reservation


@pytest.mark.parametrize("kind", KINDS)
def test_abandon_refuses_an_aliased_archive_folder_before_moving_anything(tmp_path, protected_all, kind):
    """An aliased .exp46_owners/abandoned would receive the moved owner record: every destination of the abandon is
    validated before the first rename (nothing moves, nothing is released)."""
    import json
    prot = protected_root(kind, protected_all, tmp_path)
    before = listing(prot)
    sub = _load("exp46_submit_neuronic.py", "exp46_submit_guards_abandon")
    root = tmp_path / "exp46_save_root"
    od = root / sub.OWNERS / f"{RUN}__SMOKE"
    od.mkdir(parents=True)
    token = "b" * 32
    intent = {"schema": 1, "token": token, "run": RUN, "mode": "SMOKE", "head": "a" * 40, "time": "01:00:00",
              "backend": "flash", "flash_version": reg.FLASH_VERSION, "allow_nonmatched": False,
              "save_dir": sub.run_dir(RUN, "SMOKE", str(root)), "job_name": sub.job_name(RUN, "SMOKE", token),
              "state": "submitting", "ts": "2020-01-01T00:00:00", "job_id": None, "reconciled": False,
              "partition_max_time": "7-00:00:00"}
    (od / "intent.json").write_text(json.dumps(intent))
    (root / sub.OWNERS / "abandoned").symlink_to(prot)
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        sub.abandon(RUN, "SMOKE", root=str(root), runner=_runner([]))
    assert listing(prot) == before and (od / "intent.json").is_file()


# ------------------------------------------------------------------ the supervisor's and the evaluator's entry points
@pytest.mark.parametrize("kind", KINDS)
def test_the_card_locks_and_the_child_log_refuse_an_alias_before_any_write(tmp_path, protected_all, kind):
    prot = protected_root(kind, protected_all, tmp_path)
    before = listing(prot)
    sv = _load("exp46_supervise.py", "exp46_supervise_guards")
    locks = tmp_path / "locks"
    locks.mkdir()
    (locks / ".exp38_card0.lock").symlink_to(prot / "absent.lock")
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        sv.lock_cards([0], str(locks))
    (locks / "child.log").symlink_to(prot / "absent.log")
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        sv.OwnedChild([sys.executable, "-c", "pass"], str(tmp_path), dict(os.environ), str(locks / "child.log"))
    aliased = tmp_path / "aliased_dir"
    aliased.symlink_to(prot)
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        sv.Record([str(aliased / "at_launch.json")], x=1)
    assert listing(prot) == before


@pytest.mark.parametrize("kind", KINDS)
def test_the_loss_record_move_refuses_an_aliased_receipts_folder(tmp_path, protected_all, kind):
    import exp46_fixtures as fx
    prot = protected_root(kind, protected_all, tmp_path)
    before = listing(prot)
    sv = _load("exp46_supervise.py", "exp46_supervise_guards_loss")
    rd = tmp_path / "exp46_save_root" / RUN
    v0 = rd / reg.LOGGER_DIR / "version_0"
    v0.mkdir(parents=True)
    (v0 / "hparams.yaml").write_text("{}\n")
    (v0 / "metrics.csv").write_bytes(fx.loss_csv(reg))
    (rd / reg.RECEIPTS_DIR).symlink_to(prot)
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        sv.move_loss_record(str(rd))
    assert listing(prot) == before and (v0 / "metrics.csv").is_file()


@pytest.mark.parametrize("kind", KINDS)
def test_the_evaluator_refuses_an_aliased_farm_directory_before_any_claim_link_or_copy(tmp_path, protected_all,
                                                                                    monkeypatch, kind):
    prot = protected_root(kind, protected_all, tmp_path)
    before = listing(prot)
    ev = _load("exp46_eval.py", "exp46_eval_guards")
    tree = tmp_path / "tree"
    (tree / ev.FARM).mkdir(parents=True)
    monkeypatch.setattr(reg, "REPO", str(tree))
    cell = ev.run_cells(RUN)[0]
    (tree / ev.FARM / RUN).symlink_to(prot)
    src = tmp_path / "x.json"
    src.write_text("{}")
    for fn in (lambda: ev.claim_cell(cell), lambda: ev._link(f"{ev.FARM}/{RUN}/x.ckpt", str(src)),
               lambda: ev._copy_once(str(src), ev._abs(f"{ev.FARM}/{RUN}/x.json"))):
        with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
            fn()
    assert listing(prot) == before


@pytest.mark.parametrize("kind", KINDS)
def test_the_publication_lock_and_the_aux_writers_refuse_an_alias_before_any_write(tmp_path, protected_all, monkeypatch,
                                                                                  kind):
    """The collector's publication lock; the delta manifest's (exp_38 writer's) temp name; the L1 report destination."""
    import exp46_collect as col
    prot = protected_root(kind, protected_all, tmp_path)
    before = listing(prot)
    tree = tmp_path / "tree"
    (tree / reg.EXP_DIR).mkdir(parents=True)
    (tree / col.LOCK).symlink_to(prot / "absent.lock")
    rows = {(m, f, K): {"stats": {k: (1.0, 0.1, 5) for k in col.ev.METRIC_KEYS}, "source": "s"}
            for m, f in [("c4f", reg.parse_run_id(r)[1]) for r in reg.campaign_runs()] for K in (8, 1)}
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        col.publish(rows, [], {"artifacts": {}, "runs": {}}, repo=str(tree))
    assert not os.path.lexists(tree / col.OUT_JSON)
    cm46 = _load("exp46_content_manifest.py", "exp46_cm_guards")
    monkeypatch.setattr(cm46, "delta_files", lambda repo=None: (_ for _ in ()).throw(AssertionError("split read")))
    out = tmp_path / "manifests" / "m.txt.gz"
    out.parent.mkdir()
    (out.parent / f".m.txt.gz.tmp.{os.getpid()}").symlink_to(prot / "absent.gz")
    with pytest.raises(reg.GateError, match="exp_38 / exp_45"):
        cm46.build(str(out))
    rc = _load("exp46_regression_compare.py", "exp46_rc_guards")
    (tmp_path / "report.json").symlink_to(prot / "absent.json")
    assert rc.main(["--merged", "x", "--exp38", "x", "--exp43", "x", "--merged-root", "/m", "--exp38-root", "/p",
                    "--exp43-root", "/q", "--out", str(tmp_path / "report.json")]) == 2
    assert listing(prot) == before


# ------------------------------------------------------------------ the audit: every mutating call of the kit
AUDIT_KIT = ("exp46_registry.py", "exp46_submit_neuronic.py", "exp46_supervise.py", "exp46_eval.py", "exp46_collect.py",
             "exp46_table.py", "exp46_content_manifest.py", "exp46_regression_compare.py", "exp46_init_digest.py",
             "exp46_nas_publish.py")
RAW = {"os.mkdir", "os.makedirs", "os.rename", "os.replace", "os.symlink", "os.link", "os.unlink", "os.remove",
       "os.rmdir", "os.open", "os.truncate", "os.chmod", "shutil.copyfile", "shutil.copy", "shutil.copy2",
       "shutil.copytree", "shutil.move", "shutil.rmtree", "tempfile.mkstemp", "tempfile.mkdtemp", "tar.extractall"}
NEEDS_ROOT = ("atomic_write_bytes", "write_sealed", "publish_no_clobber")     # the root: 3rd positional or root=
PRIMITIVE = "a guarded primitive itself: check_destination first"
# Codex code r2 finding 1: every raw mutation left in the kit, (file, function, call) -> why it cannot leave a checked
# place. Everything else goes through a guarded primitive naming its checked root (the worklog lists every site).
AUDITED = {
    ("exp46_registry.py", "_fsync_dir", "os.open"): "O_RDONLY | O_DIRECTORY: opens a directory to fsync it (a read)",
    ("exp46_registry.py", "atomic_write_bytes", "os.open"): "the temp name after check_destination: O_EXCL | O_NOFOLLOW",
    ("exp46_registry.py", "atomic_write_bytes", "os.replace"): "the validated temp entry onto the validated destination",
    ("exp46_registry.py", "publish_no_clobber", "tempfile.mkstemp"): "a random O_EXCL | O_NOFOLLOW name, validated dir",
    ("exp46_registry.py", "publish_no_clobber", "os.link"): "the temp file hard-linked to the validated target",
    ("exp46_registry.py", "publish_no_clobber", "os.unlink"): "its own temp name",
    ("exp46_registry.py", "storage_gate", "os.unlink"): "its own exclusive probe (unlink never follows a link)",
    ("exp46_registry.py", "open_below", "os.open"): PRIMITIVE + "; O_NOFOLLOW",
    ("exp46_registry.py", "makedirs_below", "os.makedirs"): PRIMITIVE,
    ("exp46_registry.py", "mkdir_below", "os.mkdir"): PRIMITIVE,
    ("exp46_registry.py", "rename_below", "os.rename"): PRIMITIVE + " (source and destination)",
    ("exp46_registry.py", "symlink_below", "os.symlink"): PRIMITIVE,
    ("exp46_supervise.py", "move_loss_record", "os.rmdir"): "the emptied logger folders below the validated run dir",
    ("exp46_eval.py", "shutil_rmtree", "shutil.rmtree"): "a cell claim after check_destination (rmtree refuses links)",
    ("exp46_table.py", "import_exp38", "tempfile.mkdtemp"): "a private 0700 dir in the isolation-checked temp dir",
    ("exp46_table.py", "import_exp38", "shutil.rmtree"): "that private directory",
    ("exp46_table.py", "verify_record_at_commit", "tempfile.mkdtemp"): "a private 0700 dir in the checked temp dir",
    ("exp46_table.py", "verify_record_at_commit", "shutil.rmtree"): "that private directory",
    ("exp46_table.py", "extract_bundle", "tar.extractall"): "into the private dir; every member a plain relative file",
    ("exp46_nas_publish.py", "stage", "os.mkdir"): "the staging dir after check_destination: exclusive (EEXIST)",
    ("exp46_nas_publish.py", "_rename_noreplace", "os.rename"): "the no-RENAME_NOREPLACE fallback after a fresh lexists "
                                                                 "check: a directory rename never nests or replaces",
}


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return f"{inner}.{node.attr}" if inner else None
    return None


class _Calls(ast.NodeVisitor):
    """Every call of a module with its innermost enclosing function."""

    def __init__(self):
        self.stack, self.out = ["<module>"], []

    def visit_FunctionDef(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        self.out.append((self.stack[-1], node))
        self.generic_visit(node)


def mutation_calls():
    """(file, enclosing function, call) of every raw mutation, every write-mode open() and every root-less writer
    call in the kit."""
    raw, opens, rootless = set(), [], []
    for name in AUDIT_KIT:
        visitor = _Calls()
        visitor.visit(ast.parse(open(os.path.join(EXP, name)).read(), name))
        for where, node in visitor.out:
            call = _dotted(node.func) or ""
            if call in RAW or call.endswith(".extractall"):
                raw.add((name, where, "tar.extractall" if call.endswith(".extractall") else call))
            if call == "open":
                mode = node.args[1] if len(node.args) > 1 else next(
                    (k.value for k in node.keywords if k.arg == "mode"), None)
                if mode is not None and not (isinstance(mode, ast.Constant) and set(str(mode.value)) <= set("rbt")):
                    opens.append((name, where, ast.unparse(node)[:80]))
            if call.split(".")[-1] in NEEDS_ROOT and not (len(node.args) > 2 or any(k.arg == "root" for k in node.keywords)):
                rootless.append((name, where, ast.unparse(node)[:80]))
    return raw, opens, rootless


def test_every_mutating_call_of_the_kit_is_guarded_or_audited():
    """Codex code r2 finding 1: the kit's mutating paths have no unguarded open() / mkdir / rename / symlink / copy.
    Every write-mode open is open_below; every atomic / sealed / no-clobber write names its checked root; the only raw
    mutation calls left are the AUDITED ones (the primitives' own implementations and mutations that cannot leave a
    checked place) -- a new raw call, or a stale audit entry, fails here."""
    raw, opens, rootless = mutation_calls()
    assert opens == [], opens
    assert rootless == [], rootless
    assert sorted(raw - set(AUDITED)) == [] and sorted(set(AUDITED) - raw) == [], (sorted(raw - set(AUDITED)),
                                                                                  sorted(set(AUDITED) - raw))


# ------------------------------------------------------------------ fix round 3, Codex code r3 finding 1: '..' after a link
@pytest.fixture
def dotdot(tmp_path, monkeypatch):
    """The A6000 campaign tree's required AcousticRooms link into the NAS. '<tree>/AcousticRooms/../checkpoints/
    exp38_data_efficiency/report.json' is '<tree>/checkpoints/...' to abspath (which collapses '..' lexically) but exp_38's
    NAS checkpoint namespace to the kernel (which resolves the link first, then '..'). A decoy folder sits where abspath
    points, so a writer that checks one place and writes the other is visible either way."""
    nas = tmp_path / "diskstation" / "FLAC"
    (nas / "AcousticRooms").mkdir(parents=True)
    prot = nas / "checkpoints" / "exp38_data_efficiency"
    prot.mkdir(parents=True)
    monkeypatch.setattr(reg, "PROTECTED_ABSOLUTE", (str(prot),))
    tree = tmp_path / "exp-46-campaign"
    tree.mkdir()
    (tree / "AcousticRooms").symlink_to(nas / "AcousticRooms")
    decoy = tree / "checkpoints" / "exp38_data_efficiency"
    decoy.mkdir(parents=True)
    return tree, prot, decoy, str(tree / "AcousticRooms" / ".." / "checkpoints" / "exp38_data_efficiency" / "report.json")


def test_a_dotdot_after_a_symlink_never_reaches_a_protected_root_through_any_writer(dotdot, tmp_path):
    """Every guarded primitive writes exactly the path it validated (check_destination's return value), resolved as the
    kernel will (resolve_dotdot): a '..' that would cancel a symlink refuses. Through each actual writer, Codex's path
    reaches neither exp_38's namespace nor the decoy; a '..' over a REAL directory still resolves (kernel = abspath)."""
    tree, prot, decoy, bad = dotdot
    for name in PRIMITIVES:
        with pytest.raises(reg.GateError, match=r"'\.\.'"):
            _primitives(bad, tree, tmp_path)[name]()
        assert listing(prot) == [] and listing(decoy) == [], name
    with pytest.raises(reg.GateError, match=r"'\.\.'"):                  # a '..' after a link in the ROOT refuses too
        reg.atomic_write_bytes(str(decoy / "x.json"), b"x", root=str(tree / "AcousticRooms" / ".." / "checkpoints"))
    assert listing(prot) == [] and listing(decoy) == []
    (tree / "real").mkdir()                                               # '..' over a real directory: the kernel's
    ok = str(tree / "real" / ".." / "ok.json")                            # place, which is abspath's
    assert reg.resolve_dotdot(ok) == str(tree / "ok.json")
    reg.atomic_write_bytes(ok, b"ok", root=str(tree))
    assert (tree / "ok.json").read_bytes() == b"ok" and listing(prot) == []


def test_the_entry_points_refuse_a_dotdot_destination_before_any_write(dotdot, tmp_path, monkeypatch):
    """Codex's example through the comparator's --out, the init digest's --json, the delta manifest's --build and the
    NAS staging root: each refuses before reading its inputs, building a model or writing anything."""
    tree, prot, decoy, bad = dotdot
    rc = _load("exp46_regression_compare.py", "exp46_rc_dotdot")
    junit = tmp_path / "junit.xml"
    junit.write_text('<?xml version="1.0"?><testsuites><testsuite name="pytest"><testcase classname="src.tests.test_a" '
                     'name="test_x" time="0.1"></testcase></testsuite></testsuites>')
    assert rc.main(["--merged", str(junit), "--exp38", str(junit), "--exp43", str(junit), "--merged-root", "/m",
                    "--exp38-root", "/p", "--exp43-root", "/q", "--out", bad]) == 2
    idg = _load("exp46_init_digest.py", "exp46_idg_dotdot")
    monkeypatch.setattr(idg, "init_digest", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model built")))
    assert idg.main(["--run", RUN, "--json", bad]) == 2
    cm46 = _load("exp46_content_manifest.py", "exp46_cm_dotdot")
    monkeypatch.setattr(cm46, "delta_files", lambda repo=None: (_ for _ in ()).throw(AssertionError("split read")))
    with pytest.raises(reg.GateError, match=r"'\.\.'"):
        cm46.build(bad)
    nas = _load("exp46_nas_publish.py", "exp46_nas_dotdot")
    with pytest.raises(reg.GateError, match=r"'\.\.'"):
        nas.stage(os.path.dirname(bad), RUN)
    assert listing(prot) == [] and listing(decoy) == []
