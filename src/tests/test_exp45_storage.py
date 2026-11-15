"""exp_45 — the shared storage reservation (exp45_registry.reserve_storage / recheck_storage / release_storage;
plan_rf_data_curve.md r3 §3.2, Codex plan r2 finding 7, r3 finding 2).

One flock'd ledger per save root. The submitter reserves a FULL candidate's full need (16 checkpoints + 2 GiB) before
sbatch and before any owner intent, refusing unless free - sum(outstanding of the existing reservations) - need >= the
30 GiB floor (the candidate subtracted once, before insertion); the supervisor re-checks at FULL start with
free - sum(outstanding of all reservations, its own included) >= floor (no extra candidate term; idempotent);
outstanding = reserved - bytes already written. Release only on seal (supervisor) or abandon (submitter); a crash keeps
the hold. A fake filesystem (free bytes from a file, sparse run-dir files) -- no real space is used.

Written by the exp_45 Coder (Claude Opus 5.5).
"""
import json
import multiprocessing as mp
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_45_rf_data_curve_claude")
sys.path.insert(0, EXP)
import exp45_registry as reg  # noqa: E402

GIB = 2 ** 30
RUNS = reg.required_runs()
TOKENS = {r: f"{i + 1:x}" * 32 for i, r in enumerate(RUNS)}
NEED = reg.RUN_RESERVE_BYTES


@pytest.fixture
def fs(tmp_path, monkeypatch):
    """A fake filesystem: free bytes are read from a file (so forked children see changes too)."""
    root = tmp_path / "save_root"
    root.mkdir()
    free = tmp_path / "free_bytes"
    monkeypatch.setattr(reg, "free_bytes", lambda path: int(free.read_text()))

    def set_free(n):
        free.write_text(str(int(n)))
    set_free(70 * GIB)
    return root, set_free


def _rd(root, run):
    return str(root / run)


def test_the_reservation_is_the_registered_full_need():
    assert NEED == 16 * reg.CKPT_BYTES + 2 * GIB and reg.CKPT_BYTES >= 723_927_531          # the largest E0 endpoint
    assert reg.STORAGE_FLOOR_BYTES == 30 * GIB


def _reserve(args):
    root, run = args
    try:
        reg.reserve_storage(run, TOKENS[run], os.path.join(root, run), root)
        return run
    except reg.GateError:
        return None


def test_three_concurrent_candidates_never_share_the_floor(fs):
    """60 GiB free: 60 - 0 - 12.8 and 60 - 12.8 - 12.8 keep 30 GiB, a third does not -- whatever the interleaving."""
    root, set_free = fs
    set_free(60 * GIB)
    with mp.get_context("fork").Pool(3) as pool:
        got = pool.map(_reserve, [(str(root), r) for r in RUNS])
    won = [g for g in got if g]
    assert len(won) == 2
    led = json.load(open(root / reg.STORAGE_LEDGER))
    assert sorted(led["reservations"]) == sorted(won) and all(v["reserved_bytes"] == NEED for v in led["reservations"].values())


def test_one_run_from_submission_through_full_start_is_counted_once(fs):
    """Codex plan r3 finding 2: 70 GiB free, three 12.8-GiB reservations pass at submission (70 - 25.6 - 12.8 >= 30);
    each FULL-start re-check subtracts the ledger's outstanding only (70 - 38.4 >= 30), never the run twice."""
    root, set_free = fs
    for r in RUNS:
        reg.reserve_storage(r, TOKENS[r], _rd(root, r), str(root))
    for r in RUNS:
        out = reg.recheck_storage(r, TOKENS[r], str(root))
        assert out["outstanding_bytes"] == 3 * NEED and out["free_bytes"] == 70 * GIB
    with pytest.raises(reg.GateError, match="already holds"):
        reg.reserve_storage(RUNS[0], "f" * 32, _rd(root, RUNS[0]), str(root))       # one reservation per run
    set_free(70 * GIB - 2 * GIB)                     # the margin was 70 - 3 x 12.79 - 30 = 1.64 GiB
    with pytest.raises(reg.GateError, match="floor"):
        reg.recheck_storage(RUNS[1], TOKENS[RUNS[1]], str(root))                     # someone else used 2 GiB


KIB = 1024


def test_partially_written_runs_count_only_their_outstanding_bytes(fs):
    """outstanding = reserved - bytes already written, counted as ALLOCATED bytes (Codex code r1 finding 5); KiB
    scale (real bytes, so the allocation is real): three 64-KiB reservations, 20 KiB written by the first run."""
    root, set_free = fs
    need = 64 * KIB
    for r in RUNS:
        reg.reserve_storage(r, TOKENS[r], _rd(root, r), str(root), need=need, floor=0)
    d = root / RUNS[0]
    d.mkdir()
    (d / "epoch=2-step=2500.ckpt").write_bytes(os.urandom(20 * KIB))
    os.symlink("/nonexistent/target", d / "link")                                   # links are not counted
    set_free(70 * GIB - 20 * KIB)                                                    # the run wrote those bytes
    out = reg.recheck_storage(RUNS[2], TOKENS[RUNS[2]], str(root), floor=0)
    assert out["outstanding_bytes"] == 3 * need - 20 * KIB and out["free_bytes"] == 70 * GIB - 20 * KIB
    assert reg.written_bytes(str(d)) == 20 * KIB and reg.written_bytes(str(root / "absent")) == 0
    (d / "epoch=4-step=5000.ckpt").write_bytes(os.urandom(need))                     # written beyond the reservation
    assert reg.recheck_storage(RUNS[2], TOKENS[RUNS[2]], str(root), floor=0)["outstanding_bytes"] == 2 * need


def test_written_bytes_credit_only_allocated_bytes(tmp_path):
    """Conservative credit (Codex code r1 finding 5): a file's credit is min(apparent size, allocated blocks) -- a
    sparse or not-yet-allocated file credits nothing it has not taken from the filesystem."""
    d = tmp_path / "rd"
    d.mkdir()
    with open(d / "sparse.ckpt", "wb") as fh:                                       # apparent 5 GiB, nothing allocated
        fh.truncate(5 * GIB)
    (d / "real.ckpt").write_bytes(os.urandom(64 * KIB))
    os.symlink("/nonexistent/target", d / "link")
    assert reg.written_bytes(str(d)) == 64 * KIB


@pytest.mark.parametrize("where", ["after the free-space sample", "after the written-bytes scan"])
@pytest.mark.parametrize("check", ["reserve", "recheck"])
def test_a_write_between_the_two_observations_is_never_credited_against_older_free_space(tmp_path, monkeypatch,
                                                                                          where, check):
    """Codex code r1 finding 5: the credited written bytes are counted BEFORE free space is sampled, so an 8-KiB
    checkpoint landing between the two observations can only shrink the computed headroom. KiB scale: run A holds a
    20-KiB reservation with nothing written, 25 KiB are free, the floor is 10 KiB; after the write the true headroom is
    (25 - 8) - (20 - 8) = 5 KiB (recheck) or 5 - 1 = 4 KiB (a 1-KiB candidate): both refuse, whichever observation
    the write lands after (free sampled before the scan would credit the 8 KiB against the older 25 KiB: 13 / 12)."""
    root = tmp_path / "root"
    root.mkdir()
    run_a, run_b = RUNS[0], RUNS[1]
    d = root / run_a
    real_written = reg.written_bytes
    state = {"free": 1024 * KIB, "armed": False, "landed": False}

    def land():
        if state["armed"] and not state["landed"]:
            d.mkdir(exist_ok=True)
            (d / "epoch=0-step=2500.ckpt").write_bytes(os.urandom(8 * KIB))     # the trainer's write, mid-check
            state["landed"] = True

    def free(path):
        f = state["free"] - real_written(str(d))         # every written byte has left the free space
        if where == "after the free-space sample":
            land()
        return f

    real_credits = reg._file_credits

    def credits(path):
        n = real_credits(path)
        if where == "after the written-bytes scan" and path == str(d):
            land()
        return n
    monkeypatch.setattr(reg, "free_bytes", free)
    monkeypatch.setattr(reg, "_file_credits", credits)
    reg.reserve_storage(run_a, TOKENS[run_a], str(d), str(root), need=20 * KIB, floor=10 * KIB)   # 1 MiB free: fine
    state.update(free=25 * KIB, armed=True)
    with pytest.raises(reg.GateError, match="floor"):
        if check == "recheck":
            reg.recheck_storage(run_a, TOKENS[run_a], str(root), floor=10 * KIB)
        else:
            reg.reserve_storage(run_b, TOKENS[run_b], str(root / run_b), str(root), need=1 * KIB, floor=10 * KIB)
    assert state["landed"]
    assert sorted(json.load(open(root / reg.STORAGE_LEDGER))["reservations"]) == [run_a]


def test_repeated_rechecks_give_the_same_answer_and_change_nothing(fs):
    root, set_free = fs
    reg.reserve_storage(RUNS[0], TOKENS[RUNS[0]], _rd(root, RUNS[0]), str(root))
    before = (root / reg.STORAGE_LEDGER).read_bytes()
    answers = [reg.recheck_storage(RUNS[0], TOKENS[RUNS[0]], str(root)) for _ in range(5)]
    assert all(a == answers[0] for a in answers) and (root / reg.STORAGE_LEDGER).read_bytes() == before
    with pytest.raises(reg.GateError, match="no storage reservation"):
        reg.recheck_storage(RUNS[0], "0" * 32, str(root))                            # another token's run
    with pytest.raises(reg.GateError, match="no storage reservation"):
        reg.recheck_storage(RUNS[1], TOKENS[RUNS[1]], str(root))                     # never reserved


def test_release_on_seal_or_abandon_and_a_crash_keeps_the_hold(fs):
    root, set_free = fs
    for r in RUNS[:2]:
        reg.reserve_storage(r, TOKENS[r], _rd(root, r), str(root))
    # a crashed run releases nothing: its hold stays until it is abandoned
    assert reg.recheck_storage(RUNS[0], TOKENS[RUNS[0]], str(root))["outstanding_bytes"] == 2 * NEED
    with pytest.raises(reg.GateError, match="belongs to token"):
        reg.release_storage(RUNS[0], "0" * 32, str(root))
    assert reg.release_storage(RUNS[0], TOKENS[RUNS[0]], str(root))["run"] == RUNS[0]   # sealed / abandoned
    assert reg.release_storage(RUNS[0], TOKENS[RUNS[0]], str(root)) is None             # idempotent
    assert sorted(json.load(open(root / reg.STORAGE_LEDGER))["reservations"]) == [RUNS[1]]
    assert reg.recheck_storage(RUNS[1], TOKENS[RUNS[1]], str(root))["outstanding_bytes"] == NEED


def test_a_refused_candidate_is_not_inserted_and_the_ledger_is_strict(fs):
    root, set_free = fs
    set_free(30 * GIB + NEED - 1)
    with pytest.raises(reg.GateError, match="floor"):
        reg.reserve_storage(RUNS[0], TOKENS[RUNS[0]], _rd(root, RUNS[0]), str(root))
    assert not (root / reg.STORAGE_LEDGER).exists()
    set_free(30 * GIB + NEED)
    reg.reserve_storage(RUNS[0], TOKENS[RUNS[0]], _rd(root, RUNS[0]), str(root))      # exactly at the floor
    for bad in ('{"schema": 2, "reservations": {}}', "[]", "{not json",
                '{"schema": 1, "reservations": {"x": {"run": "x"}}}'):
        (root / reg.STORAGE_LEDGER).write_text(bad)
        with pytest.raises(reg.GateError, match="ledger"):
            reg.recheck_storage(RUNS[0], TOKENS[RUNS[0]], str(root))
    for bad_run, bad_token in (("de_yaw_f025", TOKENS[RUNS[0]]), (RUNS[0], "short")):
        with pytest.raises(reg.GateError):
            reg.reserve_storage(bad_run, bad_token, "/x", str(root))
    with pytest.raises(reg.GateError, match="does not exist"):
        reg.reserve_storage(RUNS[0], TOKENS[RUNS[0]], "/x", str(root / "missing"))


@pytest.mark.parametrize("change", ["truncation", "replacement", "deletion"])
@pytest.mark.parametrize("check", ["reserve", "recheck"])
def test_a_file_changing_across_the_free_space_sample_earns_only_its_smaller_credit(tmp_path, monkeypatch, change, check):
    """Codex code r2 finding 2: written-byte credit is the MINIMUM of two scans that bracket the free-space sample,
    per file identity (st_dev, st_ino). KiB scale: run A holds 20 KiB with an 8-KiB checkpoint written; 17 KiB are
    free; just before free space is sampled the checkpoint is truncated / replaced by a new file / deleted -- the
    sample then reads 25 KiB, while the first scan saw 8 KiB. Credited once against the later sample, the headroom
    would read 25 - (20 - 8) = 13 KiB (12 with a 1-KiB candidate) over a 10-KiB floor; the true one is 25 - 20 = 5."""
    root = tmp_path / "root"
    root.mkdir()
    run_a, run_b = RUNS[0], RUNS[1]
    d = root / run_a
    d.mkdir()
    ck = d / "epoch=0-step=2500.ckpt"
    ck.write_bytes(os.urandom(8 * KIB))
    state = {"free": 1024 * KIB, "armed": False, "done": False}

    def real_credits_total():                                       # the filesystem's own view: allocated bytes
        return sum(min(st.st_size, st.st_blocks * 512) for st in (os.lstat(os.path.join(d, f)) for f in os.listdir(d)))

    def free(path):
        if state["armed"] and not state["done"]:
            state["done"] = True
            if change == "truncation":
                open(ck, "r+b").truncate(0)
            elif change == "replacement":
                os.remove(ck)
                sample = state["free"] - real_credits_total()
                ck.write_bytes(os.urandom(8 * KIB))                   # a NEW file, after the sample
                return sample
            else:
                os.remove(ck)
        return state["free"] - real_credits_total()
    monkeypatch.setattr(reg, "free_bytes", free)
    reg.reserve_storage(run_a, TOKENS[run_a], str(d), str(root), need=20 * KIB, floor=10 * KIB)
    state.update(free=25 * KIB, armed=True)                          # 25 KiB capacity, 8 of them written: 17 free
    with pytest.raises(reg.GateError, match="floor"):
        if check == "recheck":
            reg.recheck_storage(run_a, TOKENS[run_a], str(root), floor=10 * KIB)
        else:
            reg.reserve_storage(run_b, TOKENS[run_b], str(root / run_b), str(root), need=1 * KIB, floor=10 * KIB)
    assert state["done"]


@pytest.mark.parametrize("where", ["inside one run dir", "across two reservations"])
def test_a_hard_linked_file_under_a_reserved_run_dir_refuses(tmp_path, monkeypatch, where):
    """Codex code r2 finding 2: a regular file with more than one hard link would earn its allocation once per name
    (or once per reservation): any such file under a reserved run dir refuses the storage check (GateError)."""
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(reg, "free_bytes", lambda path: 10 ** 12)
    for r in RUNS[:2]:
        (root / r).mkdir()
        reg.reserve_storage(r, TOKENS[r], str(root / r), str(root), need=64 * KIB, floor=0)
    ck = root / RUNS[0] / "epoch=0-step=2500.ckpt"
    ck.write_bytes(os.urandom(8 * KIB))
    os.link(ck, (root / RUNS[0] / "copy.ckpt") if where == "inside one run dir" else (root / RUNS[1] / "copy.ckpt"))
    for fn in (lambda: reg.recheck_storage(RUNS[0], TOKENS[RUNS[0]], str(root), floor=0),
               lambda: reg.reserve_storage(RUNS[2], TOKENS[RUNS[2]], str(root / RUNS[2]), str(root), need=KIB, floor=0)):
        with pytest.raises(reg.GateError, match="hard link"):
            fn()
    assert sorted(json.load(open(root / reg.STORAGE_LEDGER))["reservations"]) == sorted(RUNS[:2])


# ------------------------------------------------------------------ crash recovery of the submitter (Codex code r1 finding 7)
import exp45_submit_neuronic as sub  # noqa: E402

RUN = RUNS[0]


class Sched:
    """A fake scheduler: ``jobs`` = {job name: {job id: state}}; squeue shows the live ones, sacct every one."""

    def __init__(self, jobs=None):
        self.jobs, self.calls = {k: dict(v) for k, v in (jobs or {}).items()}, []

    def __call__(self, argv, capture_output=True, text=True):
        import types
        self.calls.append(argv)
        tool, out = os.path.basename(argv[0]), ""
        if tool == "scontrol":
            out = "PartitionName=all MaxTime=7-00:00:00 State=UP"
        elif tool == "sbatch":
            out = "4242;cluster"
        elif tool == "squeue":
            name = argv[argv.index("-n") + 1]
            out = "".join(f"{j}|{s}\n" for j, s in self.jobs.get(name, {}).items() if s in sub.LIVE)
        elif tool == "sacct":
            name = argv[argv.index("--name") + 1]
            out = "".join(f"{j}|{name}|{s}\n" for j, s in self.jobs.get(name, {}).items())
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")


class Crash(BaseException):
    """A process death at this point of the code (nothing after it runs)."""


@pytest.fixture
def subm(tmp_path, monkeypatch):
    monkeypatch.setattr(sub, "preflight", lambda run, mode, allow, dry, backend: "a" * 40)
    monkeypatch.setattr(reg, "free_bytes", lambda path: 10 ** 15)
    monkeypatch.setattr(sub, "SUBLOG", str(tmp_path / "submissions.log"))
    root = tmp_path / "save_root"
    root.mkdir()
    return root


def _submit(root, mode="FULL", runner=None):
    return sub.submit(RUN, mode, "3-00:00:00", "flash", reg.FLASH_VERSION, False, False, root=str(root),
                      runner=runner or Sched())


def _held(root):
    p = root / reg.STORAGE_LEDGER
    return json.load(open(p))["reservations"] if p.exists() else {}


def _crash_submit(root, monkeypatch, window, mode="FULL"):
    """Run submit() and kill it at ``window``; returns the reservation token it persisted (or None)."""
    if window == "after the reservation":
        real = reg.reserve_storage

        def reserve_then_die(*a, **k):
            real(*a, **k)
            raise Crash()
        monkeypatch.setattr(reg, "reserve_storage", reserve_then_die)
    elif window == "after the owner dir":
        monkeypatch.setattr(sub, "_write_intent", lambda path, rec: (_ for _ in ()).throw(Crash()))
    elif window == "after the intent's temp file":
        def tmp_then_die(path, rec):
            with open(path + ".tmp", "w") as fh:
                json.dump(rec, fh)
            raise Crash()
        monkeypatch.setattr(sub, "_write_intent", tmp_then_die)
    with pytest.raises(Crash):
        _submit(root, mode)
    monkeypatch.undo()                         # the next process runs the real code again
    monkeypatch.setattr(sub, "preflight", lambda run, mode, allow, dry, backend: "a" * 40)
    monkeypatch.setattr(reg, "free_bytes", lambda path: 10 ** 15)
    held = _held(root)
    return held[RUN]["token"] if RUN in held else None


@pytest.mark.parametrize("window", ["after the reservation", "after the owner dir", "after the intent's temp file"])
def test_a_submit_killed_before_its_intent_is_recovered_by_abandon(subm, monkeypatch, tmp_path, window):
    """Window 1 (finding 7): the reservation (and maybe an empty owner dir) persist, no intent.json. sbatch is issued
    only after intent.json exists, so abandon may release the hold only once the scheduler shows NO job of the
    reservation's token; until then the run stays blocked. reconcile names the state and points to abandon."""
    root = subm
    monkeypatch.setattr(sub, "SUBLOG", str(tmp_path / "submissions.log"))
    token = _crash_submit(root, monkeypatch, window)
    monkeypatch.setattr(sub, "SUBLOG", str(tmp_path / "submissions.log"))
    assert token and list(_held(root)) == [RUN]
    od = sub.owner_dir(RUN, "FULL", str(root))
    assert os.path.isdir(od) == (window != "after the reservation") and not os.path.exists(os.path.join(od, "intent.json"))
    with pytest.raises(reg.GateError):
        _submit(root)                                                          # blocked until recovered
    with pytest.raises(sub.SubmitError, match="abandon"):
        sub.reconcile(RUN, "FULL", root=str(root), runner=Sched())
    name = sub.job_name(RUN, "FULL", token)
    for state in ("PENDING", "COMPLETED"):                    # a job of the token: no intent ever existed -> refuse
        with pytest.raises(sub.SubmitError, match="refuse"):
            sub.abandon(RUN, "FULL", root=str(root), runner=Sched({name: {"777": state}}))
        assert list(_held(root)) == [RUN]
    out = sub.abandon(RUN, "FULL", root=str(root), runner=Sched())
    assert out["storage_released"] and _held(root) == {} and not os.path.exists(od)
    rel = json.load(open(root / reg.STORAGE_LEDGER))["releases"]
    assert [(r["run"], r["token"]) for r in rel] == [(RUN, token)] and "recovered" in rel[0]["reason"]
    assert (out["owner_moved_to"] is None) == (window == "after the reservation")
    with pytest.raises(sub.SubmitError):
        sub.abandon(RUN, "FULL", root=str(root), runner=Sched())                # idempotent: nothing left to abandon
    assert _submit(root)["job_id"] == "4242"                                    # the run may be submitted afresh


def test_a_smoke_submit_killed_before_its_intent_is_recovered_by_abandon(subm, monkeypatch, tmp_path):
    """Window 1 without a reservation (a SMOKE): an owner dir with no intent; abandon archives it."""
    root = subm
    _crash_submit(root, monkeypatch, "after the intent's temp file", mode="SMOKE")
    monkeypatch.setattr(sub, "SUBLOG", str(tmp_path / "submissions.log"))
    od = sub.owner_dir(RUN, "SMOKE", str(root))
    assert os.path.isdir(od) and _held(root) == {}
    tok = json.load(open(os.path.join(od, "intent.json.tmp")))["token"]
    with pytest.raises(sub.SubmitError, match="refuse"):
        sub.abandon(RUN, "SMOKE", root=str(root), runner=Sched({sub.job_name(RUN, "SMOKE", tok): {"9": "RUNNING"}}))
    out = sub.abandon(RUN, "SMOKE", root=str(root), runner=Sched())
    assert not os.path.exists(od) and os.path.isdir(out["owner_moved_to"])
    assert _submit(root, mode="SMOKE")["job_id"] == "4242"


def test_an_abandon_killed_after_archiving_the_owner_is_recovered(subm, monkeypatch, tmp_path):
    """Window 2 (finding 7): abandon archived the owner record, then died before releasing the reservation. The next
    abandon finds the archived record of the reservation's token and releases the hold -- only once every job of the
    token is terminal (the hold is never freed before that)."""
    root = subm
    out = _submit(root)
    token, name = out["token"], sub.job_name(RUN, "FULL", out["token"])
    real = reg.release_storage
    monkeypatch.setattr(reg, "release_storage", lambda *a, **k: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        sub.abandon(RUN, "FULL", root=str(root), runner=Sched({name: {"4242": "FAILED"}}))
    monkeypatch.setattr(reg, "release_storage", real)
    assert not os.path.exists(sub.owner_dir(RUN, "FULL", str(root))) and list(_held(root)) == [RUN]
    with pytest.raises(reg.GateError):
        _submit(root)                                                          # blocked until recovered
    with pytest.raises(sub.SubmitError, match="not terminal"):
        sub.abandon(RUN, "FULL", root=str(root), runner=Sched({name: {"4242": "RUNNING"}}))
    assert list(_held(root)) == [RUN]
    got = sub.abandon(RUN, "FULL", root=str(root), runner=Sched({name: {"4242": "FAILED"}}))
    assert got["storage_released"] and _held(root) == {} and "archived" in got["recovered"]
    assert _submit(root)["job_id"] == "4242"


def test_recovery_never_runs_while_a_submission_holds_its_lock(subm, monkeypatch, tmp_path):
    """The (run, mode) submission lock: submit() holds it from the reservation to the 'submitted' intent, so a
    concurrent reconcile / abandon refuses instead of mistaking a submission in flight for a crash."""
    import fcntl
    root = subm
    path = sub.submit_lock_path(RUN, "FULL", str(root))
    fh = open(path, "a")
    fcntl.flock(fh, fcntl.LOCK_EX)
    try:
        for fn in (sub.abandon, sub.reconcile):
            with pytest.raises(sub.SubmitError, match="in progress"):
                fn(RUN, "FULL", root=str(root), runner=Sched())
        with pytest.raises(sub.SubmitError, match="in progress"):
            _submit(root)
        assert _held(root) == {}
    finally:
        fh.close()
    assert _submit(root)["job_id"] == "4242"

