"""exp_35 round 6a (ii): the supervisor's pure parts (plan_orbitrir_yaw_evidence.md r2 §7, §11 B2; r3 §12 B1-B2;
r4 §13; §14 A1.3) -- the frozen 33-chunk phase-(i) order, the end-of-training predictor, the fail-closed state file
(schema v1, atomic writes), the O_EXCL reservations and their restart consistency, the nvidia-smi / /proc tables.

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3/r4.
"""
import importlib.util
import json
import os

import pytest

import test_depth_canon_collect as T

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SUP = os.path.join(REPO, "worklog", "worklog_yixun", "exp_35_orbitrir_yaw_evidence_claude", "exp35_supervisor.py")


def _load():
    spec = importlib.util.spec_from_file_location("exp35_supervisor", SUP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load()
H35 = "a" * 40
SEEDS = (42, 43, 44, 45, 46)


# --------------------------------------------------------------------------- #
# the frozen 33-chunk order (§7) -- the E2-FAIL block never in it (§14 A1.3)
# --------------------------------------------------------------------------- #
def test_the_33_chunks_are_the_registered_phase_i_order():
    ch = S.chunk_plan()
    assert [c["n"] for c in ch] == list(range(1, 34))
    assert [(c["kind"], c["cell"], c["gate"]) for c in ch[:5]] == [
        ("cell", "exp35_P1_T_S40000_K1_s42", "E1"), ("cell", "exp35_P1_T_S40000_K8_s42", "E1"),
        ("cell", "exp35_BF_T_S40000_K1_s42", None), ("cell", "exp35_BF_T_S40000_K8_s42", "E2"), ("probe", None, "PROBE")]
    rest = [c["cell"] for c in ch[5:]]
    assert rest == ([f"exp35_BF_T_S40000_K{k}_s{s}" for s in SEEDS[1:] for k in (1, 8)]            # T s43-46
                    + [f"exp35_BF_R_S40000_K{k}_s{s}" for s in SEEDS for k in (1, 8)]              # R
                    + [f"exp35_BF_T_S40000_K{k}_s42_dup" for k in (1, 8)]                          # dup
                    + [f"exp35_BF_C4_S40000_K{k}_s42_th{d}" for d in (90, 180, 270, 45) for k in (1, 8)])   # C4+45
    assert all(c["kind"] == "cell" and c["gate"] is None for c in ch[5:])


def test_the_32_chunk_cells_are_the_phase_i_rows_of_the_manifest():
    m = json.load(open(os.path.join(REPO, "worklog", "worklog_yixun", "exp_25_depth_canon_claude", "depth_canon_cells.json")))
    want = {c["cell"] for c in m["cells"] if c["block"] in ("CTL", "BF-pin", "BF-dup")}
    got = [c["cell"] for c in S.chunk_plan() if c["kind"] == "cell"]
    assert len(got) == len(set(got)) == 32 and set(got) == want
    assert not any(c["block"] == "E2-FAIL" for c in m["cells"] if c["cell"] in got)


def test_the_registered_planning_durations():
    ch = {c["cell"] or "probe": c for c in S.chunk_plan()}
    assert S.planning_minutes(ch["exp35_P1_T_S40000_K1_s42"]) == 17.6 and S.planning_minutes(ch["exp35_P1_T_S40000_K8_s42"]) == 19.2
    assert S.planning_minutes(ch["exp35_BF_R_S40000_K1_s44"]) == 24.0 and S.planning_minutes(ch["exp35_BF_T_S40000_K8_s42_dup"]) == 35.0
    assert S.planning_minutes(ch["probe"]) == 10.0
    assert (S.PREDICT_S_PER_STEP, S.PREDICT_MARGIN_S, S.MAX_STEPS, S.GUARD_POLL_S, S.HANDOVER_GAP_S) == (
        3.5, 1800, 40000, 300, 60)


# --------------------------------------------------------------------------- #
# the end-of-training predictor (§13 B1): spawn shared only if s43 cannot end before the cell does
# --------------------------------------------------------------------------- #
def _report(step, mtime):
    return {"step": {"global": step}, "log": {"mtime": mtime}, "finished": step >= 40000}


def test_the_predictor_uses_3_5_s_per_step_from_the_last_log_write():
    chunk = next(c for c in S.chunk_plan() if c["cell"] == "exp35_BF_T_S40000_K8_s42")        # 35 min planning
    need = 2 * 35 * 60 + 1800
    step = 40000 - 2000                                                                       # earliest end: 7000 s
    ok, d = S.predictor_ok(_report(step, 1000.0), now=1000.0 + 7000 - need - 1, chunk=chunk)
    assert ok and d["earliest_end"] == 1000.0 + 2000 * 3.5 and d["needed_after"] < d["earliest_end"]
    ok, _ = S.predictor_ok(_report(step, 1000.0), now=1000.0 + 7000 - need, chunk=chunk)       # equal: no
    assert not ok
    ok, _ = S.predictor_ok(_report(step, 0.0), now=1000.0 + 7000 - need - 1, chunk=chunk)     # a stale log write
    assert not ok


def test_the_predictor_refuses_without_a_step_or_after_the_training():
    chunk = S.chunk_plan()[0]
    assert not S.predictor_ok({"step": None, "log": {"mtime": 0.0}}, now=0.0, chunk=chunk)[0]
    assert not S.predictor_ok(_report(40000, 1e12), now=0.0, chunk=chunk)[0]


# --------------------------------------------------------------------------- #
# the state file: schema v1, fail-closed load, atomic writes (temp + fsync + rename + directory fsync)
# --------------------------------------------------------------------------- #
def _state():
    return S.new_state(h35=H35, cells_sha256="c" * 64, supervisor_sha256="d" * 64, approved_sha256="e" * 64)


def test_a_new_state_plans_every_chunk():
    st = _state()
    assert st["schema_version"] == 1 and st["stop_latch"] is None and st["baseline"] is None
    assert [c["state"] for c in st["chunks"]] == ["planned"] * 33 and st["reservation"] is None
    assert [(c["n"], c["kind"], c["cell"], c["gate"]) for c in st["chunks"]] == [
        (c["n"], c["kind"], c["cell"], c["gate"]) for c in S.chunk_plan()]


def test_the_state_is_written_atomically(tmp_path, monkeypatch):
    path, calls = str(tmp_path / "state.json"), []
    real = os.fsync
    monkeypatch.setattr(S.os, "fsync", lambda fd: (calls.append(os.fstat(fd).st_mode), real(fd)))
    S.write_state(path, _state())
    assert json.load(open(path)) == _state() and len(calls) == 2           # the temp file, then the directory
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
    import stat
    assert stat.S_ISREG(calls[0]) and stat.S_ISDIR(calls[1])


def test_a_crash_before_the_rename_leaves_the_old_state(tmp_path, monkeypatch):
    path = str(tmp_path / "state.json")
    S.write_state(path, _state())
    monkeypatch.setattr(S.os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("crash")))
    st = _state()
    st["stop_latch"] = {"reason": "x", "time": "t"}
    with pytest.raises(OSError):
        S.write_state(path, st)
    assert json.load(open(path)) == _state()


@pytest.mark.parametrize("edit, needle", [
    (lambda d: d.update(schema_version=2), "schema"),
    (lambda d: d.update(h35="b" * 40), "h35"),
    (lambda d: d.update(cells_sha256="0" * 64), "cells_sha256"),
    (lambda d: d.update(supervisor_sha256="0" * 64), "supervisor_sha256"),
    (lambda d: d["chunks"].pop(), "chunks"),
    (lambda d: d["chunks"][3].update(cell="exp35_BF_OFF_S40000_K1_s43_th45"), "chunks"),
    (lambda d: d["chunks"][7].update(state="done"), "state"),
    (lambda d: [d["chunks"][i].update(state="gates_verified") for i in (0, 2)], "order"),
    (lambda d: [d["chunks"][i].update(state="in_flight") for i in (0, 1)], "order"),
])
def test_an_inconsistent_state_refuses(tmp_path, edit, needle):
    path = str(tmp_path / "state.json")
    doc = _state()
    edit(doc)
    S.write_state(path, doc)
    with pytest.raises(ValueError, match=needle):
        S.load_state(path, h35=H35, cells_sha256="c" * 64, supervisor_sha256="d" * 64)


def test_a_corrupt_or_absent_state(tmp_path):
    assert S.load_state(str(tmp_path / "absent.json"), h35=H35, cells_sha256="c" * 64, supervisor_sha256="d" * 64) is None
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ValueError, match="corrupt"):
        S.load_state(str(tmp_path / "bad.json"), h35=H35, cells_sha256="c" * 64, supervisor_sha256="d" * 64)


# --------------------------------------------------------------------------- #
# reservations (§12 B1.4): O_EXCL, never overwritten; restart consistency
# --------------------------------------------------------------------------- #
G0, G1 = "GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e", "GPU-6ac2604a-e84e-0615-1221-ccbed7c268da"


def test_a_reservation_is_created_exclusively(tmp_path):
    p = S.create_reservation(str(tmp_path), G1, epoch=1, h35=H35, now=100.0)
    doc = json.load(open(p))
    assert p == os.path.join(str(tmp_path), f"{G1}.json")
    assert doc == {"holder": "exp35", "h35": H35, "gpu_uuid": G1, "epoch": 1, "created": 100.0,
                   "branch": "exp-35-orbitrir-yaw"}
    with pytest.raises(FileExistsError):
        S.create_reservation(str(tmp_path), G1, epoch=2, h35=H35, now=200.0)
    assert json.load(open(p))["epoch"] == 1                                   # never overwritten
    S.remove_reservation(p)
    assert not os.path.exists(p)


@pytest.mark.parametrize("files, held, ok", [
    ({}, None, True),                                                         # before the first spawn
    ({G1: 1}, (G1, 1), True),
    ({G0: 2}, (G0, 2), True),
    ({}, (G1, 1), False),                                                     # neither
    ({G0: 2, G1: 1}, (G0, 2), False),                                         # both
    ({G1: 2}, (G1, 1), False),                                                # another epoch
    ({G1: 1}, None, False),                                                   # a file of ours the state does not hold
])
def test_restart_reservation_consistency(tmp_path, files, held, ok):
    for uuid, epoch in files.items():
        S.create_reservation(str(tmp_path), uuid, epoch=epoch, h35=H35, now=1.0)
    reasons = S.reservation_reasons(str(tmp_path), None if held is None else {"uuid": held[0], "epoch": held[1]},
                                    (G0, G1))
    assert (reasons == []) is ok, reasons


def test_another_holders_reservation_is_not_ours(tmp_path):
    (tmp_path / f"{G0}.json").write_text(json.dumps({"holder": "exp22", "gpu_uuid": G0}))
    assert S.reservation_reasons(str(tmp_path), {"uuid": G1, "epoch": 1}, (G0, G1)) != []      # ours is missing
    S.create_reservation(str(tmp_path), G1, epoch=1, h35=H35, now=1.0)
    assert S.reservation_reasons(str(tmp_path), {"uuid": G1, "epoch": 1}, (G0, G1)) == []
    assert S.held_by_other(str(tmp_path), G0) and not S.held_by_other(str(tmp_path), G1)


# --------------------------------------------------------------------------- #
# the GPU tables (fail-closed parsing) and our own process tree
# --------------------------------------------------------------------------- #
def test_the_nvidia_smi_tables_parse_fail_closed():
    assert S.gpu_map(f"0, {G0}\n1, {G1}\n") == {0: G0, 1: G1}
    assert S.compute_apps(f"3027719, {G0}\n3028196, {G1}\n") == [(3027719, G0), (3028196, G1)]
    assert S.compute_apps("") == []
    for bad in ("0 GPU-x\n", f"0, {G0}\n0, {G1}\n"):
        with pytest.raises(ValueError):
            S.gpu_map(bad)
    with pytest.raises(ValueError):
        S.compute_apps("abc, def\n")


def _proc(tmp, table):
    """A fake /proc: ``{pid: ppid}`` stat files."""
    for pid, ppid in table.items():
        os.makedirs(tmp / str(pid), exist_ok=True)
        (tmp / str(pid) / "stat").write_text(f"{pid} (python x) S {ppid} 1 1 0 -1 4194304 0 0 0 0 0 0 0 0 20 0 1 0 999 0 0\n")
    return str(tmp)


def test_descendants_follow_the_parent_links(tmp_path):
    proc = _proc(tmp_path, {10: 1, 11: 10, 12: 11, 13: 1, 14: 13})
    assert S.descendants(10, proc) == {10, 11, 12} and S.descendants(13, proc) == {13, 14}
    assert S.descendants(99, proc) == set()
