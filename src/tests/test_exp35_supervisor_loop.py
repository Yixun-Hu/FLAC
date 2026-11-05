"""exp_35 round 6a (iii-b): the supervisor's chunk loop (plan_orbitrir_yaw_evidence.md r2 §7, §11 B2; r3 §12 B1-B3; r4 §13)
on a fake world -- fake time, a fake nvidia-smi table, a fake /proc read by the REAL identity helper, spawned
processes that finish after N polls (their evaluator a child on the lane GPU while they run), Yixun's log read by
the REAL guard, and injected gates. It never signals: FakeProc has no kill / terminate.

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3/r4.
"""
import hashlib
import json
import os
import shutil

import pytest

import test_exp35_supervisor_core as SC
import test_exp35_training_guard as TG
from test_exp35_supervisor_lanes import (G0, G1, N_CHUNKS, S, TR0, TR1, FakeGates, _lane, make, proc_entry,  # noqa
                                         state, sup)


# --------------------------------------------------------------------------- #
# a clean campaign
# --------------------------------------------------------------------------- #
def test_a_clean_campaign_runs_the_33_chunks_in_order_on_the_shared_gpu1(tmp_path):
    w, g, cfg = make(tmp_path)
    assert sup(w, g, cfg).run() == 0
    plan = S.chunk_plan()
    assert len(w.spawned) == N_CHUNKS
    for rec, c in zip(w.spawned, plan):
        if c["kind"] == "cell":
            assert rec["argv"][-4:] == ["--lanes", "1", "--only", c["cell"]] and rec["env"]["EXPECT_HEAD"] == "a" * 40
            assert rec["env"]["GPU_MAX_POLLS"] == "1"
        else:
            assert rec["argv"][2:] == [S.PROBE, "--arm", "BF", "--ckpt-path", S.BF_FARM_LINK]
        assert _lane(rec)[0] == G1 and rec["files"] == [f"{G1}.json"]                     # reserved while it runs
    assert all(_lane(r)[1:] == ("1", cfg["approved"]) for r in w.spawned if "--lanes" in r["argv"])
    st = state(cfg)
    assert [c["state"] for c in st["chunks"]] == ["gates_verified"] * N_CHUNKS and st["stop_latch"] is None
    assert st["baseline"]["step"] == 3 * TG.T and list(w.res.iterdir()) == []              # released at the end
    assert st["chunks"][0]["runner_rc"] == 0 and st["chunks"][0]["receipt_sha256"]
    assert st["chunks"][4]["probe_rc"] == 0 and st["chunks"][4]["probe_contract"]["rc"] == 0
    assert {"probe_json_sha256", "sidecar_sha256", "log_sha256"} <= set(st["chunks"][4])
    assert ("E1", 1) in g.calls and ("E1", 8) in g.calls and ("E2",) in g.calls
    assert max(s for s in w.sleeps) <= S.GUARD_POLL_S


# --------------------------------------------------------------------------- #
# stops: a guard violation, a new process on GPU 1, failures -- the running cell always finishes
# --------------------------------------------------------------------------- #
def test_a_guard_violation_while_a_cell_runs_stops_after_it(tmp_path):
    w, g, cfg = make(tmp_path)
    w.behaviour = lambda argv: (3, 0)

    def slow(world):
        with open(world.log, "a") as f:
            f.write("\r".join(TG.epoch_lines(3, lambda n: 12.0)[0][:2 * 150]) + "\r")
    w.at_spawn(3, slow)
    assert sup(w, g, cfg).run() == 3
    st = state(cfg)
    assert len(w.spawned) == 3 and (st["chunks"][2]["state"], st["chunks"][2]["runner_rc"]) == ("runner_verified", 0)
    assert "guard" in st["stop_latch"]["reason"] and all(p.returncode is not None for p in w.procs)   # it finished


def test_a_new_process_on_gpu1_while_a_cell_runs_stops_after_it(tmp_path):
    w, g, cfg = make(tmp_path)
    w.behaviour = lambda argv: (3, 0)
    w.at_spawn(2, lambda world: world.apps.append((77777, G1)))
    assert sup(w, g, cfg).run() == 3
    st = state(cfg)
    assert len(w.spawned) == 2 and "77777" in st["stop_latch"]["reason"] and st["chunks"][1]["runner_rc"] == 0


@pytest.mark.parametrize("fail", ["runner_rc", "verify", "e1_mismatch", "e1_refused", "e2_refused", "finalize"])
def test_a_failure_latches_the_campaign(tmp_path, fail):
    w, g, cfg = make(tmp_path)
    if fail == "runner_rc":
        w.behaviour = lambda argv: (2, 1 if "exp35_BF_T_S40000_K1_s42" in argv else 0)
    elif fail == "verify":
        g.bad_verify = {"exp35_P1_T_S40000_K8_s42"}
    elif fail == "e1_mismatch":
        g.e1_rc = {1: 0, 8: 1}
    elif fail == "e1_refused":
        g.e1_rc = {1: 2, 8: 0}
    elif fail == "e2_refused":
        g.e2_rc = 2
    else:
        g.finalize_rc = 1
    assert sup(w, g, cfg).run() == 3
    want = {"runner_rc": 3, "verify": 2, "e1_mismatch": 2, "e1_refused": 1, "e2_refused": 4, "finalize": 5}[fail]
    assert len(w.spawned) == want and state(cfg)["stop_latch"]


def test_an_e2_fail_decides_the_45_degree_source_only(tmp_path):
    w, g, cfg = make(tmp_path)
    g.e2_rc = 1
    assert sup(w, g, cfg).run() == 0 and len(w.spawned) == N_CHUNKS
    assert state(cfg)["chunks"][3]["e2"] == {"rc": 1, "verdict": "FAIL", "out": "E2 out"}


# --------------------------------------------------------------------------- #
# co-tenancy: identity, predictor, the hand-over to GPU 0
# --------------------------------------------------------------------------- #
def test_the_trainer_exit_hands_over_to_gpu0_atomically(tmp_path):
    w, g, cfg = make(tmp_path)
    w.at_spawn(6, lambda world: world.remove_trainers())
    assert sup(w, g, cfg).run() == 0 and len(w.spawned) == N_CHUNKS
    assert [_lane(r) for r in w.spawned[:6] if "--lanes" in r["argv"]] == [(G1, "1", cfg["approved"])] * 5
    assert w.spawned[4]["argv"][2] == S.PROBE and w.spawned[4]["uuid"] == G1               # the probe: no runner
    assert [_lane(r) for r in w.spawned[6:]] == [(G0, None, None)] * (N_CHUNKS - 6)          # ALLOW_SHARED_GPU unset
    assert all(r["files"] == [f"{G1}.json"] for r in w.spawned[:6]) and all(
        r["files"] == [f"{G0}.json"] for r in w.spawned[6:])
    st = state(cfg)
    assert any(e["what"] == "hand-over to GPU 0" for e in st["events"]) and st["chunks"][-1]["lane"] == G0
    gap = [e for e in st["events"] if e["what"].startswith("GPU 0 free")]
    assert len(gap) == 2 and gap[1]["t"] - gap[0]["t"] >= S.HANDOVER_GAP_S


def _table_job(world):
    world.remove_trainers()
    world.apps.append((88888, G1))
    proc_entry(world.procdir, 88888, cmdline=b"python\x00exp30_attn_eval.py\x00")


def test_his_table_job_starting_while_our_cell_runs_stops_after_it(tmp_path):
    """§13 B1: a NEW process on GPU 1 while our cell runs latches a stop after the running cell (the residual risk
    the Planner accepted), recorded and reported -- no further spawn."""
    w, g, cfg = make(tmp_path)
    w.at_spawn(4, _table_job)
    assert sup(w, g, cfg).run() == 3 and len(w.spawned) == 4
    assert "88888" in state(cfg)["stop_latch"]["reason"] and state(cfg)["chunks"][3]["runner_rc"] == 0


def test_his_table_job_after_the_training_takes_no_shared_cell(tmp_path, monkeypatch):
    """The predictor holds the shared spawns near s43's end; the trainer then exits and his table job takes GPU 1:
    the hand-over moves every remaining chunk to GPU 0, none ever shares a card with the table job."""
    w, g, cfg = make(tmp_path)
    monkeypatch.setattr(S, "MAX_STEPS", 3 * TG.T + 500)
    w.hooks.append((lambda world: len(world.sleeps) >= 3, _table_job))
    assert sup(w, g, cfg).run() == 0
    assert len(w.spawned) == N_CHUNKS and all(r["uuid"] == G0 for r in w.spawned)


@pytest.mark.parametrize("case", ["extra_process", "pid_reuse"])
def test_another_process_on_gpu1_means_no_shared_spawn(tmp_path, case):
    w, g, cfg = make(tmp_path)
    if case == "extra_process":
        w.apps.append((99999, G1))
        release = lambda world: world.apps.remove((99999, G1))  # noqa: E731
    else:
        proc_entry(w.procdir, TR1, start=111)                                   # same PID, another start time
        release = lambda world: world.remove_trainers()  # noqa: E731
    w.hooks.append((lambda world: len(world.sleeps) >= 5, release))
    assert sup(w, g, cfg).run() == 0
    assert w.spawned[0]["sleeps"] >= 5 and len(w.spawned) == N_CHUNKS                      # nothing before the release
    if case == "pid_reuse":
        assert all(r["uuid"] == G0 for r in w.spawned)                           # never shared with a stranger
    st = state(cfg)
    assert any("no shared spawn" in e["what"] for e in st["events"])


def test_the_predictor_refuses_a_shared_cell_that_could_meet_his_table_job(tmp_path, monkeypatch):
    w, g, cfg = make(tmp_path)
    monkeypatch.setattr(S, "MAX_STEPS", 3 * TG.T + 500)                         # s43 ends in <= 1750 s
    w.hooks.append((lambda world: len(world.sleeps) >= 3, lambda world: world.remove_trainers()))
    assert sup(w, g, cfg).run() == 0
    assert all(r["uuid"] == G0 for r in w.spawned)
    assert any("predictor" in e["what"] for e in state(cfg)["events"])


def test_a_busy_gpu0_or_another_holder_delays_the_handover(tmp_path):
    w, g, cfg = make(tmp_path)
    (w.res / f"{G0}.json").write_text(json.dumps({"holder": "exp22", "gpu_uuid": G0}))
    w.at_spawn(2, lambda world: world.remove_trainers(gpu0=False))              # rank 0 still on GPU 0

    def free_gpu0(world):
        world.apps.remove((TR0, G0))
        shutil.rmtree(world.procdir / str(TR0))
    w.hooks.append((lambda world: len(world.sleeps) >= 12, free_gpu0))
    w.hooks.append((lambda world: len(world.sleeps) >= 20, lambda world: (world.res / f"{G0}.json").unlink()))
    assert sup(w, g, cfg).run() == 0
    st = state(cfg)
    assert all(r["uuid"] == G0 for r in w.spawned[2:]) and len(w.spawned) == N_CHUNKS
    assert any("GPU 0 busy" in e["what"] for e in st["events"]) and any("another holder" in e["what"] for e in st["events"])


def test_a_foreign_process_on_gpu0_before_a_spawn_latches(tmp_path):
    w, g, cfg = make(tmp_path)
    w.at_spawn(1, lambda world: world.remove_trainers())

    def stranger(world):
        world.apps.append((66666, G0))
    w.at_spawn(3, stranger)
    assert sup(w, g, cfg).run() == 3
    assert len(w.spawned) == 3 and "GPU 0" in state(cfg)["stop_latch"]["reason"]


# --------------------------------------------------------------------------- #
# restarts and the log's integrity
# --------------------------------------------------------------------------- #
def test_a_set_latch_refuses_to_run(tmp_path):
    w, g, cfg = make(tmp_path)
    st = S.new_state(h35=cfg["h35"], cells_sha256=cfg["cells_sha256"], supervisor_sha256=cfg["supervisor_sha256"],
                     approved_sha256=cfg["approved_sha256"])
    st["stop_latch"] = {"reason": "earlier", "time": "t"}
    S.write_state(cfg["state"], st)
    assert sup(w, g, cfg).run() == 3 and w.spawned == []


@pytest.mark.parametrize("receipt", [False, True], ids=["no_verified_receipt", "a_verified_receipt"])
def test_an_in_flight_chunk_at_restart(tmp_path, receipt):
    """§12 B2: the supervisor died while chunk 3 ran -- a verified receipt is reconciled (its gates re-run), no
    verified receipt latches (never an adoption of a result that cannot be verified)."""
    w, g, cfg = make(tmp_path)
    w.at_spawn(3, lambda world: (_ for _ in ()).throw(KeyboardInterrupt()))    # the supervisor dies mid-chunk
    with pytest.raises(KeyboardInterrupt):
        sup(w, g, cfg).run()
    assert state(cfg)["chunks"][2]["state"] == "in_flight" and len(w.spawned) == 3
    w.spawned.clear()
    g2 = FakeGates()
    g2.bad_verify, g2.probe_json = set() if receipt else {"exp35_BF_T_S40000_K1_s42"}, cfg["probe_json"]
    rc = sup(w, g2, cfg).run()
    st = state(cfg)
    if receipt:
        assert rc == 0 and len(w.spawned) == N_CHUNKS - 3 and st["chunks"][2]["adopted_at_restart"] is True
    else:
        assert rc == 3 and w.spawned == [] and "in flight" in st["stop_latch"]["reason"]


def test_a_runner_verified_chunk_re_runs_its_gates_first(tmp_path):
    w, g, cfg = make(tmp_path)
    st = S.new_state(h35=cfg["h35"], cells_sha256=cfg["cells_sha256"], supervisor_sha256=cfg["supervisor_sha256"],
                     approved_sha256=cfg["approved_sha256"])
    st["chunks"][0].update(state="runner_verified", lane=G1, shared=True, started="t", log="x.log", ended="t",
                           runner_rc=0, receipt_sha256="1" * 64, campaign_digest="d" * 64)
    S.create_reservation(cfg["reservations"], G1, epoch=1, h35=cfg["h35"], now=w.t)
    st.update(reservation={"uuid": G1, "epoch": 1, "path": S.reservation_path(cfg["reservations"], G1)},
              lane={"uuid": G1, "shared": True},
              baseline={"time": w.t, "step": 3 * TG.T, "log": {"path": cfg["his_log"], "inode": 1, "size": 1}})
    S.write_state(cfg["state"], st)
    assert sup(w, g, cfg).run() == 0
    assert g.calls[0] == ("E1", 1) and len(w.spawned) == N_CHUNKS - 1


def test_inconsistent_reservations_at_restart_latch(tmp_path):
    w, g, cfg = make(tmp_path)
    st = S.new_state(h35=cfg["h35"], cells_sha256=cfg["cells_sha256"], supervisor_sha256=cfg["supervisor_sha256"],
                     approved_sha256=cfg["approved_sha256"])
    st.update(reservation={"uuid": G1, "epoch": 1, "path": S.reservation_path(cfg["reservations"], G1)},
              lane={"uuid": G1, "shared": True})                                # the state holds it; the file is gone
    S.write_state(cfg["state"], st)
    assert sup(w, g, cfg).run() == 3 and w.spawned == []
    assert "reservation" in state(cfg)["stop_latch"]["reason"]


def test_a_rotated_log_is_a_hard_stop(tmp_path):
    w, g, cfg = make(tmp_path)

    def rotate(world):
        os.rename(world.log, str(world.log) + ".1")
        TG.write_log(world.log, [4.23, 4.23, 4.23])
    w.at_spawn(2, rotate)
    assert sup(w, g, cfg).run() == 3 and len(w.spawned) == 2
    assert "inode" in state(cfg)["stop_latch"]["reason"]


def test_the_supervisor_never_signals_any_process():
    src = open(SC.SUP).read()
    for banned in ("os.kill", "killpg", ".kill(", ".terminate(", "send_signal", "signal."):
        assert banned not in src, banned


# --------------------------------------------------------------------------- #
# Codex C3c BLOCKING 1 / HIGH 1: per-chunk final-guard evidence; the reservation re-authenticated before every
# spawn and at every poll
# --------------------------------------------------------------------------- #
def test_every_gates_verified_chunk_carries_its_final_guard(tmp_path):
    w, g, cfg = make(tmp_path)
    assert sup(w, g, cfg).run() == 0
    for c in state(cfg)["chunks"]:
        fg = c["final_guard"]
        assert (fg["status"], fg["exit"]) == ("OK", 0) and fg["step"] == 3 * TG.T and fg["anchor"]["size"] > 0


def _ours_path(w):
    return w.res / f"{G1}.json"


SPOIL = {
    "deleted": lambda w: _ours_path(w).unlink(),
    "replaced_by_another_holder": lambda w: _ours_path(w).write_text(json.dumps({"holder": "exp22", "gpu_uuid": G1})),
    "another_epoch": lambda w: _ours_path(w).write_text(json.dumps(dict(json.loads(_ours_path(w).read_text()), epoch=2))),
    "corrupt": lambda w: _ours_path(w).write_text("{not json"),
}


@pytest.mark.parametrize("how", sorted(SPOIL))
def test_a_reservation_lost_while_a_cell_runs_stops_after_it(tmp_path, how):
    w, g, cfg = make(tmp_path)
    w.behaviour = lambda argv: (3, 0)
    w.at_spawn(3, SPOIL[how])
    assert sup(w, g, cfg).run() == 3 and len(w.spawned) == 3
    st = state(cfg)
    assert "reservation" in st["stop_latch"]["reason"] and st["chunks"][2]["runner_rc"] == 0     # never signalled


class SpoilAfterVerify(FakeGates):
    """Spoils the reservation right after a cell's verification: between its watch and the next spawn."""

    def __init__(self, world, cell, how):
        super().__init__()
        self.w, self.cell, self.how = world, cell, how

    def verify(self, chunk):
        out = super().verify(chunk)
        if chunk["cell"] == self.cell:
            SPOIL[self.how](self.w)
        return out


@pytest.mark.parametrize("how", sorted(SPOIL))
def test_a_reservation_lost_between_chunks_spawns_nothing_more(tmp_path, how):
    w, _, cfg = make(tmp_path)
    g = SpoilAfterVerify(w, "exp35_BF_T_S40000_K1_s42", how)                    # chunk 3
    assert sup(w, g, cfg).run() == 3 and len(w.spawned) == 3
    assert "reservation" in state(cfg)["stop_latch"]["reason"]


# --------------------------------------------------------------------------- #
# Codex C3c HIGH 3: the running-cell watch compares the COMPLETE approved identity; HIGH 4: the probe ledger
# --------------------------------------------------------------------------- #
def test_the_trainer_pid_reused_while_a_cell_runs_stops_after_it(tmp_path):
    w, g, cfg = make(tmp_path)
    w.behaviour = lambda argv: (3, 0)
    w.at_spawn(3, lambda world: proc_entry(world.procdir, TR1, start=111))     # the same PID, another process
    assert sup(w, g, cfg).run() == 3 and len(w.spawned) == 3
    st = state(cfg)
    assert str(TR1) in st["stop_latch"]["reason"] and "identity" in st["stop_latch"]["reason"]
    assert st["chunks"][2]["runner_rc"] == 0                                    # watched to its end, never signalled


class SpoilTheProbe(FakeGates):
    def __init__(self, how):
        super().__init__()
        self.how = how

    def finalize(self, log, digest):
        rc, out = super().finalize(log, digest)
        target = {"json": self.probe_json, "sidecar": self.probe_json + ".sha256",
                  "log": os.path.join(S.REPO, log)}.get(self.how)
        if target:
            os.unlink(target)
        elif self.how == "unsealed_json":
            with open(self.probe_json, "a") as f:
                f.write(" ")
        elif self.how == "another_log":
            doc = json.load(open(self.probe_json))
            doc["provenance"]["launch"]["log_path"] = "elsewhere.log"
            text = json.dumps(doc, indent=1) + "\n"
            open(self.probe_json, "w").write(text)
            open(self.probe_json + ".sha256", "w").write(
                f"{hashlib.sha256(text.encode()).hexdigest()}  {os.path.basename(self.probe_json)}\n")
        return rc, out


@pytest.mark.parametrize("how", ["json", "sidecar", "log", "unsealed_json", "another_log"])
def test_a_finalized_probe_without_its_consistent_ledger_latches(tmp_path, how):
    """Codex C3c HIGH 4: finalizer rc 0 is not enough -- the probe JSON, its sidecar and the final log must exist, the
    sidecar must seal the JSON, and the JSON's launch record must name THIS log and its sha256."""
    w, _, cfg = make(tmp_path)
    g = SpoilTheProbe(how)
    g.probe_json = cfg["probe_json"]
    assert sup(w, g, cfg).run() == 3 and len(w.spawned) == 5
    st = state(cfg)
    assert st["chunks"][4]["state"] == "runner_verified" and "probe" in st["stop_latch"]["reason"]


def test_the_probe_ledger_hashes_are_the_files(tmp_path):
    w, g, cfg = make(tmp_path)
    assert sup(w, g, cfg).run() == 0
    c5 = state(cfg)["chunks"][4]
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    assert (c5["probe_json_sha256"], c5["sidecar_sha256"], c5["log_sha256"]) == (
        sha(cfg["probe_json"]), sha(cfg["probe_json"] + ".sha256"), sha(c5["log"]))
