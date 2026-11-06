"""exp_35 phase (ii) kit, fix 3 (Codex P2K-1 BLOCKING 2): the phase-(i) gate first applies the H35 supervisor's
COMPLETE pure state schema -- exact per-state chunk keys; lane / shared / started / log / ended; receipt and
campaign-digest hashes; the exact E1 / E2 / probe result schemas and probe hashes; every passing final_guard and its
log anchor; the baseline and the training-log anchor; the events; the registered approved-tuple pin -- ported from
exp35_supervisor.py / exp35_training_guard.py at H35 and checked here against THOSE files (loaded from the git objects)
reason for reason, on the archived REAL states and on every deletion / malformed-field mutation; only then the
terminal conditions (33 / 33 gates_verified, no latch / lane / reservation, E2 PASS, the exact digest, the release
event last).

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import copy
import functools
import glob
import hashlib
import importlib.util
import json
import os
import sys

import pytest

import exp35_phase_ii_sandbox as SB

K = SB.load_kit()
PROD_RES = "/home/yixunhu/codespace/.gpu_reservations"                    # a path STRING (never accessed here)
SNAPSHOTS = sorted(glob.glob(os.path.join(SB.REPO, SB.X35, "orbitrir_yaw_evidence_2026-11-0*_phase_i_state_snapshot.json")))
SUP_SHA = hashlib.sha256(SB.real_supervisor()).hexdigest()


@functools.lru_cache(maxsize=None)
def _h35_supervisor(root):
    """The H35 supervisor and its training guard, from the git objects, as an importable pair under ``root``."""
    for name in ("exp35_supervisor.py", "exp35_training_guard.py"):
        with open(os.path.join(root, name), "wb") as f:
            f.write(SB.git_show(SB.H35_REAL, f"{SB.X35}/{name}"))
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("h35_supervisor_for_the_schema", os.path.join(root, "exp35_supervisor.py"))
    sup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sup)
    return sup


@pytest.fixture(scope="module")
def sup(tmp_path_factory):
    return _h35_supervisor(str(tmp_path_factory.mktemp("h35_supervisor")))


def _both(sup, doc, **kw):
    pins = dict(h35=SB.H35_REAL, cells_sha256=K.CELLS_SHA256, supervisor_sha256=SUP_SHA, approved_sha256=K.APPROVED_SHA256,
                his_log=K.HIS_LOG, reservations=PROD_RES)
    pins.update(kw)
    return K.supervisor_state_reasons(copy.deepcopy(doc), **pins), sup.state_reasons(copy.deepcopy(doc), **pins)


def test_the_registered_pins_of_the_port():
    assert K.APPROVED_SHA256 == "238d890d44253b602e7b510435483e7be3362ad1d5a9511cc1fc5c722a9a605e"
    assert K.APPROVED_SHA256 == hashlib.sha256(open(os.path.join(
        SB.REPO, SB.X35, "exp35_approved_cotenants_2026-11-05_20-19-20.json"), "rb").read()).hexdigest()
    assert K.HIS_LOG == ("/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn/logs/"
                         "train_exp30_ACEattn_s43_20261104T061255_741947.log")
    assert all(json.load(open(p))["approved_sha256"] == K.APPROVED_SHA256 for p in SNAPSHOTS) and len(SNAPSHOTS) == 3


@pytest.mark.parametrize("path", SNAPSHOTS, ids=os.path.basename)
def test_every_archived_real_state_passes_the_schema_exactly_as_the_supervisor_judges_it(sup, path):
    doc = json.load(open(path))
    kit, h35 = _both(sup, doc)
    assert kit == h35 == []
    why = K.phase_i_reasons(doc, h35=SB.H35_REAL, digest=SB.DIGEST_REAL, supervisor_sha256=SUP_SHA, reservations=PROD_RES)
    assert why and not any("schema" in x for x in why)                     # incomplete, never malformed
    assert any("not gates_verified" in x for x in why)


def test_the_completed_state_passes_both(sup):
    assert _both(sup, SB.completed_state()) == ([], [])


# --------------------------------------------------------------------------- #
# every required field, deleted or malformed: refused, with the supervisor's very reasons
# --------------------------------------------------------------------------- #
def _del(path):
    def f(d):
        node = d
        for k in path[:-1]:
            node = node[k]
        del node[path[-1]]
    return f


def _set(path, value):
    def f(d):
        node = d
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
    return f


C = "chunks"
CELL, E1, E2, PROBE = 9, 0, 3, 4          # chunk indices: a plain cell, the first E1, the E2, the probe
TOP = [_del([k]) for k in K.PHASE_I_KEYS] + [
    _set(["extra"], 1), _set(["schema_version"], 2), _set(["campaign"], "exp22"), _set(["h35"], "b" * 40),
    _set(["cells_sha256"], "0" * 64), _set(["supervisor_sha256"], "0" * 64), _set(["approved_sha256"], "1" * 64),
    _set(["approved_sha256"], "short")]
BASELINE = [_del(["baseline", k]) for k in ("time", "step", "log")] + [_del(["baseline", "log", k]) for k in (
    "path", "inode", "size")] + [
    _set(["baseline", "time"], -1), _set(["baseline", "time"], "t"), _set(["baseline", "step"], -1),
    _set(["baseline", "step"], True), _set(["baseline", "step"], 1.5), _set(["baseline", "log", "path"], "/elsewhere"),
    _set(["baseline", "log", "inode"], "1"), _set(["baseline", "log", "size"], -1), _set(["baseline", "extra"], 1),
    _set(["baseline"], None)]
ANCHOR = [_del(["his_log_anchor", k]) for k in K.ANCHOR_KEYS] + [
    _set(["his_log_anchor", "inode"], "1"), _set(["his_log_anchor", "device"], -1),
    _set(["his_log_anchor", "offset"], 10 ** 12), _set(["his_log_anchor", "tail_sha256"], "abc"),
    _set(["his_log_anchor", "size"], True), _set(["his_log_anchor", "extra"], 1), _set(["his_log_anchor"], [])]
EVENTS = [_del(["events", 0, k]) for k in ("t", "time", "what")] + [
    _set(["events", 0, "what"], ""), _set(["events", 0, "t"], -1.0), _set(["events", 0, "time"], 5),
    _set(["events"], {}), _set(["events", 0], "an event")]
LATCH = [_set(["stop_latch"], {"reason": ""}), _set(["stop_latch"], {"reason": "x", "time": ""}),
         _set(["stop_latch"], "latched"), _set(["stop_latch"], {"time": "t", "reason": "x"})]
LANE = [_set(["lane"], {"uuid": SB.GPU1, "shared": True}),
        _set(["reservation"], {"uuid": SB.GPU1, "epoch": 1, "path": f"{PROD_RES}/{SB.GPU1}.json", "sha256": "0" * 64})]
CHUNK = [_del([C, CELL, k]) for k in ("lane", "shared", "started", "log", "ended", "runner_rc", "receipt_sha256",
                                       "campaign_digest", "final_guard")] + [
    _set([C, CELL, "lane"], "GPU-x y"), _set([C, CELL, "lane"], SB.GPU0), _set([C, CELL, "shared"], 1),
    _set([C, CELL, "started"], ""), _set([C, CELL, "log"], ""), _set([C, CELL, "ended"], ""),
    _set([C, CELL, "runner_rc"], "0"), _set([C, CELL, "runner_rc"], True), _set([C, CELL, "runner_rc"], 1),
    _set([C, CELL, "receipt_sha256"], "abc"), _set([C, CELL, "campaign_digest"], 5),
    _set([C, CELL, "adopted_at_restart"], False), _set([C, CELL, "extra"], 1), _set([C, CELL, "state"], "done"),
    _set([C, CELL, "state"], "planned")]
GUARD = [_del([C, CELL, "final_guard", k]) for k in ("status", "exit", "t", "time", "step", "anchor")] + [
    _set([C, CELL, "final_guard", "status"], "STOP"), _set([C, CELL, "final_guard", "exit"], 1),
    _set([C, CELL, "final_guard", "exit"], False), _set([C, CELL, "final_guard", "t"], -1),
    _set([C, CELL, "final_guard", "time"], ""), _set([C, CELL, "final_guard", "step"], -1),
    _set([C, CELL, "final_guard", "anchor"], None), _set([C, CELL, "final_guard", "anchor", "tail_sha256"], "x"),
    _set([C, CELL, "final_guard", "extra"], 1)]
GATES = [_del([C, E1, "e1"]), _set([C, E1, "e1", "verdict"], "mismatch"), _set([C, E1, "e1", "rc"], 1),
         _set([C, E1, "e1", "rc"], False), _set([C, E1, "e1", "extra"], 1), _del([C, E1, "e1", "out"]),
         _del([C, E2, "e2"]), _set([C, E2, "e2", "verdict"], "FAIL"), _set([C, E2, "e2", "rc"], 2),
         _del([C, E2, "e2", "out"]), _set([C, E2, "e2", "out"], 7)] + [_del([C, PROBE, k]) for k in (
    "probe_contract", "probe_json_sha256", "sidecar_sha256", "log_sha256", "probe_rc")] + [
    _set([C, PROBE, "probe_contract", "rc"], "0"), _set([C, PROBE, "probe_contract", "extra"], 1),
    _set([C, PROBE, "probe_contract", "rc"], 1), _set([C, PROBE, "probe_json_sha256"], "x"),
    _set([C, PROBE, "sidecar_sha256"], None), _set([C, PROBE, "log_sha256"], "0" * 63), _set([C, PROBE, "probe_rc"], 1)]
STRUCT = [_set([C, 31, "state"], "planned"), _set([C, 32, "n"], 34), _set([C, 7, "campaign_digest"], "0" * 64)]
MUTATIONS = TOP + BASELINE + ANCHOR + EVENTS + LATCH + LANE + CHUNK + GUARD + GATES + STRUCT


@pytest.mark.parametrize("k", range(len(MUTATIONS)))
def test_every_required_field_is_refused_with_the_supervisors_own_reasons(sup, k):
    doc = SB.completed_state()
    MUTATIONS[k](doc)
    kit, h35 = _both(sup, doc)
    assert kit == h35 and kit, (k, kit, h35)
    why = K.phase_i_reasons(doc, h35=SB.H35_REAL, digest=SB.DIGEST_REAL, supervisor_sha256=SUP_SHA, reservations=PROD_RES)
    assert why and set(f"the H35 supervisor's schema: {x}" for x in kit) <= set(why)


@pytest.mark.parametrize("k", range(len(MUTATIONS)))
def test_the_port_agrees_with_the_supervisor_on_the_real_mid_campaign_state_too(sup, k):
    doc = json.load(open(SNAPSHOTS[-1]))
    try:
        MUTATIONS[k](doc)
    except (KeyError, IndexError, TypeError):
        pytest.skip("this mutation addresses a field the mid-campaign snapshot does not carry yet")
    kit, h35 = _both(sup, doc)
    assert kit == h35, (k, kit, h35)
