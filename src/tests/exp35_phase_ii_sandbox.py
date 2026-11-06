"""exp_35 phase (ii) kit tests: the shared sandbox (plan_orbitrir_yaw_evidence.md §15; Codex plan §15 r1 S6 "a
completely sandboxed harness"). Everything here lives under pytest's tmp_path: never the real runner, a GPU
command, the NAS, a real reservation file, a production process identity or the run worktree. The real H35
bytes the kit pins (the cells manifest, the CANON manifest, the supervisor) are read from git OBJECTS only.

Written by the exp_35 phase-(ii) kit Coder (Claude Opus 5.5, max effort).
"""
import copy
import functools
import importlib.util
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
X35 = "worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude"
X25 = "worklog/worklog_yixun/exp_25_depth_canon_claude"
KIT_PY = os.path.join(REPO, X35, "exp35_phase_ii.py")
KIT_SH = os.path.join(REPO, X35, "exp35_phase_ii.sh")
H35_REAL = "5b065801e9d1c1844461ec84cd0558702c714326"
DIGEST_REAL = "da82d0121413b792e2f42b94026b884d371d995c17e2dcc111c7edd58f8650c1"
SNAPSHOT = os.path.join(REPO, X35, "orbitrir_yaw_evidence_2026-11-06_01-54-17_phase_i_state_snapshot.json")
GPU0, GPU1 = "GPU-0a1aaa90-3043-0321-2b0c-c35b9036624e", "GPU-6ac2604a-e84e-0615-1221-ccbed7c268da"
FINAL_EVENT = "phase (i) complete; the reservation released"


@functools.lru_cache(maxsize=None)
def load_kit():
    """The kit helper under test, imported by path (one copy per process; no bytecode written)."""
    spec = importlib.util.spec_from_file_location("exp35_phase_ii", KIT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.lru_cache(maxsize=None)
def git_show(rev, path):
    """The bytes of ``path`` at ``rev`` from the shared object store (read-only; no worktree is touched)."""
    return subprocess.run(["git", "-C", REPO, "show", f"{rev}:{path}"], capture_output=True, check=True).stdout


def real_cells():
    return git_show(H35_REAL, f"{X25}/depth_canon_cells.json")


def real_canon():
    return git_show(H35_REAL, f"{X25}/depth_canon_2026-10-26_07-44-41_exp25_CANON_checkpoints.sha256")


def real_supervisor():
    return git_show(H35_REAL, f"{X35}/exp35_supervisor.py")


def snapshot_state():
    """The archived REAL phase-(i) state of 01:54 EST Nov 6 (13 / 33 gates_verified, chunk 14 in flight)."""
    with open(SNAPSHOT) as f:
        return json.load(f)


def completed_state(**over):
    """The real snapshot carried to a completed phase (i) exactly as the supervisor ends it: every chunk
    gates_verified (the plain cell chunks shaped like the real chunk 13), the lane and the reservation None, the
    final release event last. ``over`` replaces top-level keys (h35, campaign_digest of every chunk, ...)."""
    doc = snapshot_state()
    proto = copy.deepcopy(doc["chunks"][12])                                # a real plain gates_verified cell chunk
    for c in doc["chunks"]:
        if c["state"] != "gates_verified":
            n, cell = c["n"], c["cell"]
            c.clear()
            c.update(copy.deepcopy(proto), n=n, cell=cell)
    doc["lane"] = doc["reservation"] = None
    t = doc["events"][-1]["t"] + 50000.0
    doc["events"].append({"t": t, "time": "2026-11-06T11:20:00-05:00", "what": FINAL_EVENT})
    digest = over.pop("campaign_digest", None)
    if digest is not None:
        for c in doc["chunks"]:
            if "campaign_digest" in c:
                c["campaign_digest"] = digest
    doc.update(over)
    return doc
