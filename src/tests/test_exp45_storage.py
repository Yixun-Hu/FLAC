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


def test_partially_written_runs_count_only_their_outstanding_bytes(fs):
    root, set_free = fs
    for r in RUNS:
        reg.reserve_storage(r, TOKENS[r], _rd(root, r), str(root))
    d = root / RUNS[0]
    d.mkdir()
    with open(d / "epoch=2-step=2500.ckpt", "wb") as fh:                             # a sparse 5 GiB "checkpoint"
        fh.truncate(5 * GIB)
    os.symlink("/nonexistent/target", d / "link")                                   # links are not counted
    set_free(70 * GIB - 5 * GIB)                                                     # the run wrote those bytes
    out = reg.recheck_storage(RUNS[2], TOKENS[RUNS[2]], str(root))
    assert out["outstanding_bytes"] == 3 * NEED - 5 * GIB and out["free_bytes"] == 65 * GIB
    assert reg.written_bytes(str(d)) == 5 * GIB and reg.written_bytes(str(root / "absent")) == 0
    with open(d / "epoch=4-step=5000.ckpt", "wb") as fh:                             # written beyond the reservation
        fh.truncate(NEED)
    assert reg.recheck_storage(RUNS[2], TOKENS[RUNS[2]], str(root))["outstanding_bytes"] == 2 * NEED


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
