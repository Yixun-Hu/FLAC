"""exp_46 -- f100's data binding (plan_c4flip_data_curve.md r3 §3.2 "Data binding"; Codex plan r1 finding 2, r2 findings 2
and 3): the delta content manifest tool and the in-job re-verification before sealing.

* exp46_content_manifest.delta_files: exp_38's content-file rule over the full split minus exp_38's f075 set -- on THIS
  repo's real splits exactly the registered 145,494 files (72,747 waveforms + 72,747 metadata files, no depth file);
  a different count or a depth file refuses.
* build / verify on a tiny synthetic dataset: PASS; a changed file, a missing file, a missing / unreadable / unpinned
  / other manifest -> FAIL; reads are sequential (one thread) with the caller's check before every file.
* the record (verify_record) is exactly registry.F100_VERIFICATION_KEYS and admitted by f100_verification_problems.
* every f100 launch needs the pinned manifest (check_f100_delta_pinned).
* the training receipt binds the sealed record (f100 only); online and offline admission re-open it: missing, changed,
  FAIL, unpinned -> refused; a fraction run carrying one -> refused.
* the supervisor's f100 FULL branch: the re-verification runs after the child and BEFORE the training receipt and
  the commit marker; PASS -> sealed with the record bound; FAIL, an exception and a cancellation (signal) -> unsealed,
  no admissible receipt, the run lock stays.
"""
import gzip
import json
import os
import sys
import threading

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_46_c4flip_data_curve_claude")
sys.path.insert(0, EXP)
sys.path.insert(0, TESTS)
import exp46_registry as reg  # noqa: E402
import exp46_content_manifest as cm46  # noqa: E402
import exp46_fixtures as fx  # noqa: E402

F100 = "exp46_C4Ff100_l40_2x32"
F025 = "exp46_C4Ff025_l40_2x32"


@pytest.fixture(autouse=True)
def _exp45_ledger_stand_in(tmp_path, monkeypatch):
    """C3 (plan r3 §3.2 "Storage") reads exp_45's REAL ledger path at run time; here an absent stand-in, so the real
    exp_45 state never decides a test."""
    monkeypatch.setattr(reg, "EXP45_LEDGER", str(tmp_path / "exp45_ledger_stand_in" / ".exp45_storage.json"))


# ------------------------------------------------------------------ the delta file set
def test_the_real_delta_is_the_registered_145494_files_waveforms_and_metadata_only():
    files = cm46.delta_files(REPO)
    assert len(files) == reg.F100_DELTA_FILES == 145_494
    wav = [f for f in files if f.startswith("single_channel_ir_1/")]
    meta = [f for f in files if f.startswith("metadata/")]
    assert len(wav) == len(meta) == 72_747 and len(wav) + len(meta) == len(files)
    full = set(cm46.full_files(REPO))
    assert set(files) <= full and not any(f.startswith("depth_map/") for f in files)
    assert len({f for f in full if f.startswith("depth_map/")} - set(cm46.cm38.content_files(REPO))) == 0


def _tiny_repo(tmp_path, monkeypatch, extra_room=False):
    """A repo with a 3-target f075 split and a 5-target full split (2 new targets, both of receivers f075 already
    has, as in the real data) plus the dataset files; the registered count is pinned to the tiny delta's."""
    repo = tmp_path / "repo"
    f075 = {"Scene": {"room_1": ["S001_R001_hybrid_IR.wav", "S002_R001_hybrid_IR.wav"], "room_2": ["S002_R002_hybrid_IR.wav"]}}
    full = {"Scene": {"room_1": ["S001_R001_hybrid_IR.wav", "S002_R001_hybrid_IR.wav", "S003_R001_hybrid_IR.wav"],
                      "room_2": ["S002_R002_hybrid_IR.wav", "S001_R002_hybrid_IR.wav"]}}
    for rel, data in (("data/AR/train_frac075_s2026.json", f075), ("data/AR/train.json", full)):
        os.makedirs(repo / os.path.dirname(rel), exist_ok=True)
        (repo / rel).write_text(json.dumps(data))
    ds = repo / "AcousticRooms"
    for rel in cm46.cm38.content_files(str(repo), split_rel="data/AR/train.json"):
        os.makedirs(ds / os.path.dirname(rel), exist_ok=True)
        (ds / rel).write_bytes(f"bytes of {rel}".encode())
    monkeypatch.setattr(reg, "F100_DELTA_FILES", 4)            # S003 wav + meta (room_1), S001 wav + meta (room_2)
    return repo


def test_the_delta_is_the_full_splits_new_waveforms_and_metadata(tmp_path, monkeypatch):
    repo = _tiny_repo(tmp_path, monkeypatch)
    assert cm46.delta_files(str(repo)) == ["metadata/Scene/room_1/S003_R001.json", "metadata/Scene/room_2/S001_R002.json",
                                           "single_channel_ir_1/Scene/room_1/S003_R001_hybrid_IR.wav",
                                           "single_channel_ir_1/Scene/room_2/S001_R002_hybrid_IR.wav"]


def test_a_delta_with_a_depth_file_or_another_count_refuses(tmp_path, monkeypatch):
    repo = _tiny_repo(tmp_path, monkeypatch)
    monkeypatch.setattr(reg, "F100_DELTA_FILES", 5)
    with pytest.raises(reg.GateError, match="registered 5"):
        cm46.delta_files(str(repo))
    monkeypatch.setattr(reg, "F100_DELTA_FILES", 4)
    real = cm46.cm38.content_files
    monkeypatch.setattr(cm46.cm38, "content_files", lambda repo=None, split_rel=None: real(repo, split_rel) + (
        ["depth_map/Scene/room_9/9.npy"] if split_rel else []))
    monkeypatch.setattr(reg, "F100_DELTA_FILES", 5)
    with pytest.raises(reg.GateError, match="depth"):
        cm46.delta_files(str(repo))


# ------------------------------------------------------------------ build / verify (sequential)
def _built(tmp_path, monkeypatch):
    repo = _tiny_repo(tmp_path, monkeypatch)
    man = str(repo / reg.F100_DELTA_MANIFEST)
    os.makedirs(os.path.dirname(man), exist_ok=True)
    out = cm46.build(man, "AcousticRooms", str(repo))
    assert out["files"] == 4
    monkeypatch.setattr(reg, "F100_DELTA_SHA256", out["sha256_text"])
    return repo, man, out


def test_build_then_verify_passes_and_is_sequential_with_the_check_before_every_file(tmp_path, monkeypatch):
    repo, man, out = _built(tmp_path, monkeypatch)
    threads, checks = set(), []
    real = cm46.cm38._hash_one
    monkeypatch.setattr(cm46.cm38, "_hash_one", lambda root, rel: threads.add(threading.get_ident()) or real(root, rel))
    assert cm46.verify(man, "AcousticRooms", str(repo), out["sha256_text"], check=lambda: checks.append(1)) == []
    assert threads == {threading.get_ident()} and len(checks) == 4
    with pytest.raises(reg.GateError):
        cm46.build(man, "AcousticRooms", str(repo))                                 # never over an existing manifest


@pytest.mark.parametrize("break_it", ["changed file", "missing file", "missing manifest", "unreadable manifest",
                                      "other pin", "unpinned"])
def test_verify_fails_on_every_break(tmp_path, monkeypatch, break_it):
    repo, man, out = _built(tmp_path, monkeypatch)
    want = out["sha256_text"]
    target = repo / "AcousticRooms" / "metadata/Scene/room_2/S001_R002.json"
    if break_it == "changed file":
        target.write_bytes(b"changed")
    elif break_it == "missing file":
        target.unlink()
    elif break_it == "missing manifest":
        os.remove(man)
    elif break_it == "unreadable manifest":
        open(man, "ab").write(b"junk")
    elif break_it == "other pin":
        want = "0" * 64
    else:
        want = None
    probs = cm46.verify(man, "AcousticRooms", str(repo), want)
    assert probs and isinstance(probs[0], str)
    rec = cm46.verify_record(man, "AcousticRooms", str(repo), want)
    assert set(rec) == reg.F100_VERIFICATION_KEYS and rec["verdict"] == "FAIL" and reg.f100_verification_problems(rec)


def test_a_stop_during_the_verification_propagates(tmp_path, monkeypatch):
    repo, man, out = _built(tmp_path, monkeypatch)

    class Stop(Exception):
        pass
    seen = []

    def check():
        seen.append(1)
        if len(seen) == 3:
            raise Stop()
    with pytest.raises(Stop):
        cm46.verify_record(man, "AcousticRooms", str(repo), out["sha256_text"], check=check)


def test_the_pass_record_is_the_registered_schema_and_admitted(tmp_path, monkeypatch):
    repo, man, out = _built(tmp_path, monkeypatch)
    monkeypatch.setattr(reg, "F100_DELTA_MANIFEST", os.path.relpath(man, str(repo)))
    rec = cm46.verify_record(man, "AcousticRooms", str(repo), out["sha256_text"])
    assert set(rec) == reg.F100_VERIFICATION_KEYS and rec["verdict"] == "PASS" and rec["problems"] == []
    assert rec["files"] == 4 and rec["manifest"] == reg.F100_DELTA_MANIFEST and rec["manifest_sha256"] == out["sha256_text"]
    assert reg.f100_verification_problems(rec) == []
    for k, v in (("verdict", "FAIL"), ("problems", ["x"]), ("files", 3), ("manifest_sha256", "0" * 64), ("kind", "x"),
                 ("manifest", "elsewhere.txt.gz"), ("schema", 2)):
        assert reg.f100_verification_problems(dict(rec, **{k: v})), k
    assert reg.f100_verification_problems({k: rec[k] for k in sorted(rec)[1:]})
    monkeypatch.setattr(reg, "F100_DELTA_SHA256", None)
    assert reg.f100_verification_problems(rec)


def test_every_f100_launch_needs_the_pinned_manifest(tmp_path, monkeypatch):
    # the unpinned state is INJECTED (fix round 1, Codex code r1 finding 4): before the L2 pin commit and after it alike
    monkeypatch.setattr(reg, "F100_DELTA_SHA256", None)             # unpinned: every f100 launch refuses
    with pytest.raises(reg.GateError, match="not pinned"):
        reg.check_f100_delta_pinned()
    repo, man, out = _built(tmp_path, monkeypatch)
    assert reg.check_f100_delta_pinned(str(repo))["sha256"] == out["sha256_text"]
    monkeypatch.setattr(reg, "F100_DELTA_SHA256", "0" * 64)
    with pytest.raises(reg.GateError, match="not the pinned"):
        reg.check_f100_delta_pinned(str(repo))
    os.remove(man)
    with pytest.raises(reg.GateError, match="unreadable"):
        reg.check_f100_delta_pinned(str(repo))


def test_the_manifest_is_in_the_closure_by_exact_path():
    assert reg.F100_DELTA_MANIFEST == f"{reg.EXP_DIR}/exp46_f100_delta_manifest.txt.gz"
    assert reg.F100_DELTA_MANIFEST in reg.CLOSURE_EXPLICIT and reg._in_closure(reg.F100_DELTA_MANIFEST)


# ------------------------------------------------------------------ the training receipt binds the record
import test_exp46_receipts as TR  # noqa: E402
from test_exp46_receipts import tree  # noqa: E402,F401  (its campaign-tree fixture)


def _write_f100(tree, rd, log, retained, paths, digests, delta):
    return reg.write_training_receipt(F100, "a6000", "flash", TR.HEAD, str(rd), str(log), 0, digests["snapshot"],
                                      digests["e0"], {"smoke_receipt": paths["smoke"], "sha256": digests["smoke"]},
                                      TR.A6000_GPUS, "2026-11-07T04:00:00-0500", digests["env"], retained,
                                      reg.AR_CONTENT_SHA256, repo=str(tree), tree=str(tree), delta_verification=delta)


def test_an_f100_training_receipt_binds_its_sealed_delta_verification_online_and_offline(tree, monkeypatch):
    monkeypatch.setattr(reg, "F100_DELTA_SHA256", "a" * 64)
    rd, log, retained, paths, digests = TR._full_run(tree, monkeypatch, run=F100)
    vpath = str(rd / reg.RECEIPTS_DIR / reg.F100_VERIFICATION)
    with pytest.raises(reg.GateError, match="no in-job delta verification"):
        _write_f100(tree, rd, log, retained, paths, digests, None)
    vsha = reg.write_sealed(vpath, fx.f100_verification_record(reg))
    rec = _write_f100(tree, rd, log, retained, paths, digests, vpath)
    assert rec["f100_delta_sha256"] == "a" * 64 and rec["f100_delta_verification_sha256"] == vsha
    assert rec["receipts"]["f100_delta_verification"] == reg.F100_VERIFICATION
    assert rec["data_content_checks"] == ["pre-launch", "post-run", "f100 delta post-run"]
    fx.commit(reg, str(rd), F100, reg.sha256_file(str(rd / reg.TRAINING_RECEIPT)))
    got, _ = reg.check_training_receipt(F100, str(rd), closure=TR.CLOSURE, e0_sha=digests["e0"])
    assert got == rec
    # the record changed after sealing (re-sealed: a consistent sidecar) -> refused
    for p_ in (vpath, vpath + ".sha256"):
        os.remove(p_)
    reg.write_sealed(vpath, fx.f100_verification_record(reg, files=1))
    with pytest.raises(reg.GateError, match="f100 delta verification"):
        reg.check_training_receipt(F100, str(rd), closure=TR.CLOSURE, e0_sha=digests["e0"])
    for p_ in (vpath, vpath + ".sha256"):
        os.remove(p_)
    with pytest.raises(reg.GateError):
        reg.check_training_receipt(F100, str(rd), closure=TR.CLOSURE, e0_sha=digests["e0"])
    vsha2 = reg.write_sealed(vpath, fx.f100_verification_record(reg))
    assert vsha2 == vsha
    reg.check_training_receipt(F100, str(rd), closure=TR.CLOSURE, e0_sha=digests["e0"])      # restored: admitted
    monkeypatch.setattr(reg, "F100_DELTA_SHA256", "b" * 64)
    with pytest.raises(reg.GateError):                          # another pin: neither the receipt nor the record match
        reg.check_training_receipt(F100, str(rd), closure=TR.CLOSURE, e0_sha=digests["e0"])


def test_a_fail_record_never_seals(tree, monkeypatch):
    monkeypatch.setattr(reg, "F100_DELTA_SHA256", "a" * 64)
    rd, log, retained, paths, digests = TR._full_run(tree, monkeypatch, run=F100)
    vpath = str(rd / reg.RECEIPTS_DIR / reg.F100_VERIFICATION)
    reg.write_sealed(vpath, fx.f100_verification_record(reg, verdict="FAIL", problems=["1 group(s) differ"]))
    with pytest.raises(reg.GateError, match="f100"):
        _write_f100(tree, rd, log, retained, paths, digests, vpath)
    assert not (rd / reg.TRAINING_RECEIPT).exists()


def test_a_fraction_run_never_carries_a_delta_verification(tree, monkeypatch):
    monkeypatch.setattr(reg, "F100_DELTA_SHA256", "a" * 64)
    rd25, log25, ret25, paths25, dig25 = TR._full_run(tree, monkeypatch, run=F025)
    vpath = str(rd25 / reg.RECEIPTS_DIR / reg.F100_VERIFICATION)
    reg.write_sealed(vpath, fx.f100_verification_record(reg))
    with pytest.raises(reg.GateError, match="only f100"):
        reg.write_training_receipt(F025, "a6000", "flash", TR.HEAD, str(rd25), str(log25), 0, dig25["snapshot"],
                                   dig25["e0"], {"smoke_receipt": paths25["smoke"], "sha256": dig25["smoke"]},
                                   TR.A6000_GPUS, "2026-11-07T04:00:00-0500", dig25["env"], ret25,
                                   reg.AR_CONTENT_SHA256, repo=str(tree), tree=str(tree), delta_verification=vpath)


# ------------------------------------------------------------------ the supervisor seals only after a PASS
import test_exp46_supervise as TS  # noqa: E402


def _fake_record(verdict="PASS"):
    return fx.f100_verification_record(reg, verdict=verdict, problems=[] if verdict == "PASS" else ["1 group(s) differ"])


@pytest.mark.parametrize("outcome", ["pass", "fail", "exception", "cancelled"])
def test_the_f100_full_branch_reverifies_the_delta_before_sealing(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(reg, "F100_DELTA_SHA256", "a" * 64)
    order = []

    def fake_verify(manifest, dataset_root="AcousticRooms", repo=None, want_sha=None, check=None):
        order.append(("verify", os.path.exists(os.path.join(str(tmp_path / "nas" / F100), reg.TRAINING_RECEIPT))))
        assert manifest == reg._abs(reg.F100_DELTA_MANIFEST) and want_sha == "a" * 64 and check is TS.sv.check_signals
        if outcome == "exception":
            raise RuntimeError("synthetic verification crash")
        if outcome == "cancelled":
            TS.sv._PENDING.append(15)                              # a SIGTERM arrives mid-verification
            try:
                check()
            finally:
                TS.sv._PENDING.clear()
        return _fake_record("PASS" if outcome == "pass" else "FAIL")
    monkeypatch.setattr(cm46, "verify_record", fake_verify)
    if outcome == "pass":
        code, rec, rd, calls = TS._campaign(tmp_path, monkeypatch, run=F100)
        assert code == 0 and rec["status"] == "COMPLETE" and order == [("verify", False)]
        vpath = os.path.join(str(rd), reg.RECEIPTS_DIR, reg.F100_VERIFICATION)
        assert TS.DELTA_KW == [vpath] and rec["f100_delta_verification"]["verdict"] == "PASS"
        assert reg.read_sealed(vpath, reg.F100_VERIFICATION_KEYS)[1] == rec["f100_delta_verification"]["sha256"]
        assert calls[-1] == ("commits", [("COMPLETE", True)])
        return
    if outcome == "fail":
        code, rec, rd, calls = TS._campaign(tmp_path, monkeypatch, run=F100)
        assert code == TS.sv.EXIT_END and rec["status"].startswith("end audit failed") and "f100 delta" in rec["status"]
        assert rec["f100_delta_verification"]["verdict"] == "FAIL"
    else:
        exc = RuntimeError if outcome == "exception" else TS.sv.Stopped
        with pytest.raises(exc):
            TS._campaign(tmp_path, monkeypatch, run=F100)
        rd = tmp_path / "nas" / F100
    assert TS.DELTA_KW == [] and order == [("verify", False)]                 # no training receipt was ever written
    assert not os.path.exists(os.path.join(str(rd), reg.TRAINING_RECEIPT))
    assert not os.path.exists(os.path.join(str(rd), reg.COMMIT_MARKER))
    assert os.path.isdir(os.path.join(str(rd), ".exp46_lock"))                # the crash marker stays


def test_the_fraction_full_branch_does_not_verify_a_delta(tmp_path, monkeypatch):
    monkeypatch.setattr(cm46, "verify_record", lambda *a, **k: (_ for _ in ()).throw(AssertionError("verified")))
    code, rec, rd, calls = TS._campaign(tmp_path, monkeypatch, run=F025)
    assert code == 0 and TS.DELTA_KW == [None] and "f100_delta_verification" not in rec
