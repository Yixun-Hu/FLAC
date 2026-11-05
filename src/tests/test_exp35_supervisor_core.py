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
PINS = dict(h35=H35, cells_sha256="c" * 64, supervisor_sha256="d" * 64, approved_sha256="e" * 64)
WHERE = dict(his_log="/x.log", reservations="/r")                # the configured log and reservation directory
ANCHOR = {"inode": 5, "device": 2, "size": 9, "offset": 9, "tail_sha256": "5" * 64}


def _state():
    return S.new_state(**PINS)


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
        S.load_state(path, **PINS, **WHERE)


def test_a_corrupt_or_absent_state(tmp_path):
    assert S.load_state(str(tmp_path / "absent.json"), **PINS, **WHERE) is None
    (tmp_path / "bad.json").write_text("{")
    with pytest.raises(ValueError, match="corrupt"):
        S.load_state(str(tmp_path / "bad.json"), **PINS, **WHERE)


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
                                    (G0, G1), H35)
    assert (reasons == []) is ok, reasons


def test_another_holders_reservation_is_not_ours(tmp_path):
    (tmp_path / f"{G0}.json").write_text(json.dumps({"holder": "exp22", "gpu_uuid": G0}))
    assert S.reservation_reasons(str(tmp_path), {"uuid": G1, "epoch": 1}, (G0, G1), H35) != []      # ours is missing
    S.create_reservation(str(tmp_path), G1, epoch=1, h35=H35, now=1.0)
    assert S.reservation_reasons(str(tmp_path), {"uuid": G1, "epoch": 1}, (G0, G1), H35) == []
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


# --------------------------------------------------------------------------- #
# Codex C3b BLOCKING 1: a state carries the evidence of every claim it makes -- or it refuses
# --------------------------------------------------------------------------- #
def _done(st, upto):
    """Chunks 1..upto exactly as the loop leaves them: gates_verified with all their evidence."""
    for c in st["chunks"][:upto]:
        c.update(state="gates_verified", lane=G1, shared=True, started="t0", log="x.log", ended="t1")
        if c["kind"] == "cell":
            c.update(runner_rc=0, receipt_sha256="1" * 64, campaign_digest="d" * 64)
        else:
            c.update(probe_rc=0, probe_contract={"rc": 0, "out": "BF PROBE VALID"}, probe_json_sha256="2" * 64,
                     sidecar_sha256="3" * 64, log_sha256="4" * 64)
        c.update(**{"E1": {"e1": {"rc": 0, "verdict": "equal", "out": "ok"}},
                    "E2": {"e2": {"rc": 1, "verdict": "FAIL", "out": "x"}}}.get(c["gate"], {}))
        c["final_guard"] = {"status": "OK", "exit": 0, "t": 2.0, "time": "t2", "step": 101, "anchor": dict(ANCHOR)}
    st.update(baseline={"time": 1.0, "step": 100, "log": {"path": "/x.log", "inode": 5, "size": 9}},
              lane={"uuid": G1, "shared": True}, reservation={"uuid": G1, "epoch": 1, "path": f"/r/{G1}.json"},
              his_log_anchor=dict(ANCHOR),
              events=[{"t": 1.0, "time": "t", "what": "supervisor started (a new state)"}])
    return st


def _load(tmp_path, doc, **pins):
    path = str(tmp_path / "state.json")
    S.write_state(path, doc)
    return S.load_state(path, **dict(PINS, **WHERE, **pins))


def test_states_carrying_their_evidence_load(tmp_path):
    assert _load(tmp_path, _done(_state(), 7)) is not None
    crash = _done(_state(), 7)                                                  # died while chunk 8 ran
    crash["chunks"][7].update(state="in_flight", lane=G1, shared=True, started="t", log="y.log")
    assert _load(tmp_path, crash) is not None
    failed = _done(_state(), 7)                                                 # chunk 8 failed: recorded + latched
    failed["chunks"][7].update(state="in_flight", lane=G1, shared=True, started="t", log="y.log", ended="t2",
                               runner_rc=1)
    failed["stop_latch"] = {"reason": "chunk 8 exited rc 1", "time": "t2"}
    assert _load(tmp_path, failed) is not None
    done = _done(_state(), 33)
    done.update(lane=None, reservation=None)                                    # released at the end
    assert _load(tmp_path, done) is not None
    early = _state()                                                            # a hand-over before our first spawn
    early.update(lane={"uuid": G0, "shared": False}, reservation={"uuid": G0, "epoch": 1, "path": f"/r/{G0}.json"})
    assert _load(tmp_path, early) is not None


@pytest.mark.parametrize("pin", ["cells_sha256", "supervisor_sha256", "approved_sha256"])
@pytest.mark.parametrize("bad", [None, "", "x" * 64])
def test_a_state_is_neither_created_nor_loaded_against_a_missing_or_malformed_pin(tmp_path, pin, bad):
    """Codex C3c BLOCKING 1: the expected pins themselves must be sha256 values (a missing cells file hashes to None)."""
    with pytest.raises(ValueError, match="pin"):
        S.new_state(**dict(PINS, **{pin: bad}))
    S.write_state(str(tmp_path / "state.json"), _done(_state(), 3))
    with pytest.raises(ValueError, match="pin"):
        S.load_state(str(tmp_path / "state.json"), **dict(PINS, **WHERE, **{pin: bad}))


STATE_EDITS = {
    "a_fake_gates_verified_prefix": lambda d: [c.update(state="gates_verified") for c in d["chunks"][7:9]],
    "a_missing_receipt": lambda d: d["chunks"][2].pop("receipt_sha256"),
    "a_missing_rc": lambda d: d["chunks"][3].pop("runner_rc"),
    "a_failed_rc": lambda d: d["chunks"][3].update(runner_rc=1),
    "a_missing_e1": lambda d: d["chunks"][0].pop("e1"),
    "a_failed_e1": lambda d: d["chunks"][1]["e1"].update(rc=1, verdict="mismatch"),
    "a_missing_e2": lambda d: d["chunks"][3].pop("e2"),
    "an_e2_verdict_off_its_rc": lambda d: d["chunks"][3]["e2"].update(verdict="PASS"),
    "a_missing_probe_hash": lambda d: d["chunks"][4].pop("sidecar_sha256"),
    "a_failed_probe_contract": lambda d: d["chunks"][4]["probe_contract"].update(rc=1),
    "a_bad_digest": lambda d: d["chunks"][5].update(campaign_digest="not-hex"),
    "two_campaign_digests": lambda d: d["chunks"][5].update(campaign_digest="e" * 64),
    "another_approved_tuple": lambda d: d.update(approved_sha256="0" * 64),
    "another_campaign": lambda d: d.update(campaign="exp25"),
    "a_malformed_baseline": lambda d: d.update(baseline={"time": "t", "step": 1, "log": {}}),
    "no_baseline_after_a_spawn": lambda d: d.update(baseline=None),
    "a_malformed_anchor": lambda d: d.update(his_log_anchor={"inode": 1}),
    "lane_and_reservation_disagree": lambda d: d["lane"].update(uuid=G0),
    "a_lane_without_its_reservation": lambda d: d.update(reservation=None),
    "a_shared_lane_on_gpu0": lambda d: d.update(lane={"uuid": G0, "shared": True},
                                                reservation={"uuid": G0, "epoch": 2, "path": f"/r/{G0}.json"}),
    "a_reservation_elsewhere": lambda d: d["reservation"].update(path="/r/other.json"),
    "a_reservation_epoch_0": lambda d: d["reservation"].update(epoch=0),
    "an_empty_latch": lambda d: d.update(stop_latch={}),
    "a_latch_without_a_reason": lambda d: d.update(stop_latch={"reason": "", "time": "t"}),
    "a_latch_with_more_keys": lambda d: d.update(stop_latch={"reason": "r", "time": "t", "x": 1}),
    "a_malformed_event": lambda d: d["events"].append({"what": "x"}),
    "an_extra_chunk_key": lambda d: d["chunks"][0].update(note="x"),
    "evidence_on_a_planned_chunk": lambda d: d["chunks"][20].update(runner_rc=0),
    "a_failure_without_the_latch": lambda d: d["chunks"][7].update(state="in_flight", lane=G1, shared=True,
                                                                   started="t", log="x", ended="t", runner_rc=1),
    "a_shared_flag_off_its_lane": lambda d: d["chunks"][2].update(shared="yes"),
    # Codex C3c BLOCKING 1
    "a_runner_verified_flipped_to_gates_verified": lambda d: d["chunks"][7].update(
        state="gates_verified", lane=G1, shared=True, started="t", log="x", ended="t", runner_rc=0,
        receipt_sha256="1" * 64, campaign_digest="d" * 64),                       # every field but its final guard
    "a_final_guard_that_stopped": lambda d: d["chunks"][2]["final_guard"].update(status="STOP", exit=1),
    "a_final_guard_without_its_anchor": lambda d: d["chunks"][2]["final_guard"].pop("anchor"),
    "a_final_guard_with_a_bad_anchor": lambda d: d["chunks"][2]["final_guard"]["anchor"].update(offset=99),
    "another_baseline_log": lambda d: d["baseline"]["log"].update(path="/other.log"),
    "a_negative_baseline_inode": lambda d: d["baseline"]["log"].update(inode=-1),
    "a_negative_baseline_size": lambda d: d["baseline"]["log"].update(size=-5),
    "a_reservation_outside_its_directory": lambda d: d["reservation"].update(path=f"/elsewhere/{G1}.json"),
    "an_in_flight_failure_with_receipt_fields": lambda d: (d["chunks"][7].update(
        state="in_flight", lane=G1, shared=True, started="t", log="x", ended="t", runner_rc=1, receipt_sha256="1" * 64,
        campaign_digest="d" * 64), d.update(stop_latch={"reason": "r", "time": "t"})),
    "an_in_flight_failure_with_a_string_rc": lambda d: (d["chunks"][7].update(
        state="in_flight", lane=G1, shared=True, started="t", log="x", ended="t", runner_rc="1"),
        d.update(stop_latch={"reason": "r", "time": "t"})),
    "an_in_flight_failure_without_an_end": lambda d: (d["chunks"][7].update(
        state="in_flight", lane=G1, shared=True, started="t", log="x", ended=None, runner_rc=1),
        d.update(stop_latch={"reason": "r", "time": "t"})),
}


@pytest.mark.parametrize("name", sorted(STATE_EDITS))
def test_a_state_without_its_evidence_refuses(tmp_path, name):
    doc = _done(_state(), 7)
    STATE_EDITS[name](doc)
    with pytest.raises(ValueError, match="inconsistent"):
        _load(tmp_path, doc)


# --------------------------------------------------------------------------- #
# Codex C3b HIGH 2: a reservation of ours is authenticated field by field; a malformed file is never "free"
# --------------------------------------------------------------------------- #
RES_EDITS = {
    "holder": lambda d: d.update(holder="exp22"), "h35": lambda d: d.update(h35="b" * 40),
    "branch": lambda d: d.update(branch="main"), "gpu_uuid": lambda d: d.update(gpu_uuid=G0),
    "epoch_type": lambda d: d.update(epoch="1"), "created_type": lambda d: d.update(created="t"),
    "an_extra_key": lambda d: d.update(x=1), "a_missing_key": lambda d: d.pop("branch"),
}


@pytest.mark.parametrize("name", sorted(RES_EDITS))
def test_a_reservation_of_ours_is_authenticated_field_by_field(tmp_path, name):
    p = S.create_reservation(str(tmp_path), G1, epoch=1, h35=H35, now=1.0)
    doc = json.load(open(p))
    RES_EDITS[name](doc)
    open(p, "w").write(json.dumps(doc))
    assert S.reservation_reasons(str(tmp_path), {"uuid": G1, "epoch": 1}, (G0, G1), H35) != []


def test_a_malformed_reservation_file_is_refused_and_never_free(tmp_path):
    (tmp_path / f"{G1}.json").write_text("{not json")
    assert S.reservation_reasons(str(tmp_path), {"uuid": G1, "epoch": 1}, (G0, G1), H35) != []
    assert S.reservation_reasons(str(tmp_path), None, (G0, G1), H35) != []
    assert S.held_by_other(str(tmp_path), G1) is True
    (tmp_path / f"{G0}.json").write_text("[]")
    assert S.held_by_other(str(tmp_path), G0) is True
