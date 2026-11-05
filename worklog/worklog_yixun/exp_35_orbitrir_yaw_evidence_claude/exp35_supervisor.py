#!/usr/bin/env python3
"""exp_35 supervisor (plan_orbitrir_yaw_evidence.md r2 §7, §11 B2; r3 §12 B1-B3; r4 §13; §14 A1.3): the frozen
33-chunk phase-(i) loop on Yixun's shared GPU 1, its identity-bound co-tenancy, the atomic hand-over to GPU 0, the
reservation files, the persistent fail-closed state and the post-chunk E1 / E2 / probe gates. It NEVER signals any
process: a violation only stops further spawns (the running cell finishes).

Part 1 (round 6a ii): the pure parts -- the chunk order, the end-of-training predictor, the state file (schema v1,
atomic temp + fsync + rename + directory fsync), the O_EXCL reservations, the nvidia-smi and /proc tables.

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3/r4.
"""
import json
import os
import re
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
EXP35 = os.path.relpath(HERE, REPO)
RESERVATIONS = "/home/yixunhu/codespace/.gpu_reservations"                  # §7
GPU1_UUID = "GPU-6ac2604a-e84e-0615-1221-ccbed7c268da"                       # §11 B2: Yixun's s43 rank 1
BRANCH = "exp-35-orbitrir-yaw"
SEEDS = (42, 43, 44, 45, 46)
MAX_STEPS = 40000                                                            # the s43 training's max_steps
PREDICT_S_PER_STEP, PREDICT_MARGIN_S = 3.5, 1800                             # §13 B1
GUARD_POLL_S, HANDOVER_GAP_S = 300, 60                                       # §11 B2 (<= 5 min), §12 B1.4 (a)
PLAN_MIN = {"P1": {1: 17.6, 8: 19.2}, "BF": {1: 24.0, 8: 35.0}, "probe": 10.0}   # §12 B1.3; the probe: §7 ETA
STATE_SCHEMA = 1
CHUNK_STATES = ("planned", "in_flight", "runner_verified", "gates_verified")
STATE_KEYS = ("schema_version", "campaign", "h35", "cells_sha256", "supervisor_sha256", "approved_sha256", "baseline",
              "his_log_anchor", "lane", "reservation", "chunks", "stop_latch", "events")


def chunk_plan():
    """§7's frozen order: CTL K1 (E1), CTL K8 (E1), BF T s42 K1, BF T s42 K8 (E2), the probe, then T s43-46, R,
    dup, C4+45 -- one cell per chunk, K1 before K8 at each seed / angle. The E2-FAIL block is never in it (§14 A1)."""
    T = lambda k, s, x="": ("cell", f"exp35_BF_T_S40000_K{k}_s{s}{x}", None)  # noqa: E731
    out = [("cell", "exp35_P1_T_S40000_K1_s42", "E1"), ("cell", "exp35_P1_T_S40000_K8_s42", "E1"), T(1, 42),
           ("cell", "exp35_BF_T_S40000_K8_s42", "E2"), ("probe", None, "PROBE")]
    out += [T(k, s) for s in SEEDS[1:] for k in (1, 8)]
    out += [("cell", f"exp35_BF_R_S40000_K{k}_s{s}", None) for s in SEEDS for k in (1, 8)]
    out += [T(k, 42, "_dup") for k in (1, 8)]
    out += [("cell", f"exp35_BF_C4_S40000_K{k}_s42_th{d}", None) for d in (90, 180, 270, 45) for k in (1, 8)]
    return [{"n": i + 1, "kind": kind, "cell": cell, "gate": gate} for i, (kind, cell, gate) in enumerate(out)]


def planning_minutes(chunk):
    if chunk["kind"] == "probe":
        return PLAN_MIN["probe"]
    m = re.match(r"exp35_(P1|BF)_\w+_K(\d)_s", chunk["cell"])
    return PLAN_MIN[m[1]][int(m[2])]


def predictor_ok(report, *, now, chunk):
    """§13 B1: a SHARED spawn only if s43's earliest end -- its last log write + (40,000 - step) x 3.5 s -- lies
    after now + 2 x the chunk's planning duration + 30 min."""
    step = (report.get("step") or {}).get("global")
    if step is None or report.get("finished"):
        return False, {"why": "no training step" if step is None else "the training has finished"}
    earliest = report["log"]["mtime"] + (MAX_STEPS - step) * PREDICT_S_PER_STEP
    need = now + 2 * planning_minutes(chunk) * 60 + PREDICT_MARGIN_S
    return earliest > need, {"step": step, "earliest_end": earliest, "needed_after": need}


def new_state(*, h35, cells_sha256, supervisor_sha256, approved_sha256):
    return {"schema_version": STATE_SCHEMA, "campaign": "exp35", "h35": h35, "cells_sha256": cells_sha256,
            "supervisor_sha256": supervisor_sha256, "approved_sha256": approved_sha256, "baseline": None,
            "his_log_anchor": None, "lane": None, "reservation": None,
            "chunks": [dict(c, state="planned") for c in chunk_plan()], "stop_latch": None, "events": []}


def write_state(path, doc):
    """Atomic: a temp file in the same directory, fsync, rename over the state, fsync the directory (§13 NIT 1)."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".exp35_state.", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    dfd = os.open(d, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def state_reasons(doc, *, h35, cells_sha256, supervisor_sha256):
    if not isinstance(doc, dict) or list(doc) != list(STATE_KEYS):
        return [f"the state's keys are not {list(STATE_KEYS)}"]
    out = [f"{k}: {doc[k]!r} != {want!r}" for k, want in (("schema_version", STATE_SCHEMA), ("h35", h35),
           ("cells_sha256", cells_sha256), ("supervisor_sha256", supervisor_sha256)) if doc[k] != want]
    plan = [(c["n"], c["kind"], c["cell"], c["gate"]) for c in chunk_plan()]
    if [(c.get("n"), c.get("kind"), c.get("cell"), c.get("gate")) for c in doc["chunks"]] != plan:
        return out + ["the state's chunks are not the frozen 33-chunk order"]
    states = [c.get("state") for c in doc["chunks"]]
    out += [f"chunk {i + 1}: unknown state {s!r}" for i, s in enumerate(states) if s not in CHUNK_STATES]
    done = 0
    while done < len(states) and states[done] == "gates_verified":
        done += 1
    if not out and any(s != "planned" for s in states[done + 1:]):
        out.append(f"chunk order violated: {states[:done + 3]}... (only the first unfinished chunk may be started)")
    return out


def load_state(path, *, h35, cells_sha256, supervisor_sha256):
    """The state, None when absent; ``ValueError`` when corrupt or inconsistent (a refusal to run)."""
    try:
        with open(path) as f:
            doc = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as err:
        raise ValueError(f"the supervisor state {path} is corrupt ({err})") from err
    why = state_reasons(doc, h35=h35, cells_sha256=cells_sha256, supervisor_sha256=supervisor_sha256)
    if why:
        raise ValueError(f"the supervisor state {path} is inconsistent: " + "; ".join(why))
    return doc


def _fsync_dir(d):
    fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def reservation_path(dirpath, uuid):
    return os.path.join(dirpath, f"{uuid}.json")


def create_reservation(dirpath, uuid, *, epoch, h35, now):
    """§12 B1.4: O_CREAT | O_EXCL, never overwritten (``FileExistsError`` when the file exists)."""
    path = reservation_path(dirpath, uuid)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump({"holder": "exp35", "h35": h35, "gpu_uuid": uuid, "epoch": epoch, "created": now, "branch": BRANCH}, f)
        f.flush()
        os.fsync(f.fileno())
    _fsync_dir(dirpath)
    return path


def remove_reservation(path):
    os.unlink(path)
    _fsync_dir(os.path.dirname(path))


def _ours(dirpath, uuid):
    try:
        with open(reservation_path(dirpath, uuid)) as f:
            doc = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {"holder": "?"}
    return doc if isinstance(doc, dict) else {"holder": "?"}


def held_by_other(dirpath, uuid):
    doc = _ours(dirpath, uuid)
    return doc is not None and doc.get("holder") != "exp35"


def reservation_reasons(dirpath, held, uuids):
    """[] iff exactly the reservation the state holds (``{"uuid", "epoch"}`` or None) is ours on disk."""
    mine = {u: d for u in uuids for d in [_ours(dirpath, u)] if d is not None and d.get("holder") == "exp35"}
    if held is None:
        return [f"exp_35 reservation file(s) {sorted(mine)} exist but the state holds none"] if mine else []
    if set(mine) != {held["uuid"]}:
        return [f"the state holds {held['uuid']} (epoch {held['epoch']}), the exp_35 files on disk are {sorted(mine)}"]
    got = mine[held["uuid"]].get("epoch")
    return [] if got == held["epoch"] else [f"the reservation of {held['uuid']} has epoch {got}, the state {held['epoch']}"]


def gpu_map(text):
    """``{index: uuid}`` from ``nvidia-smi --query-gpu=index,uuid --format=csv,noheader``; ValueError when malformed."""
    out = {}
    for line in [ln for ln in text.splitlines() if ln.strip()]:
        m = re.fullmatch(r"\s*(\d+),\s*(GPU-[0-9A-Za-z-]+)\s*", line)
        if not m or int(m[1]) in out or m[2] in out.values():
            raise ValueError(f"malformed or duplicate nvidia-smi index,uuid row {line!r}")
        out[int(m[1])] = m[2]
    if not out:
        raise ValueError("nvidia-smi lists no GPU")
    return out


def compute_apps(text):
    """``[(pid, uuid)]`` from ``nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader``."""
    out = []
    for line in [ln for ln in text.splitlines() if ln.strip()]:
        m = re.fullmatch(r"\s*(\d+),\s*(GPU-[0-9A-Za-z-]+)\s*", line)
        if not m:
            raise ValueError(f"malformed nvidia-smi compute-app row {line!r}")
        out.append((int(m[1]), m[2]))
    return out


def descendants(pid, proc="/proc"):
    """``pid`` and every process below it (by the ppid of ``/proc/<p>/stat``): our runner's own subtree."""
    parent = {}
    for name in os.listdir(proc):
        if name.isdigit():
            try:
                with open(os.path.join(proc, name, "stat")) as f:
                    text = f.read()
                parent[int(name)] = int(text[text.rindex(")") + 2:].split()[1])
            except (OSError, ValueError, IndexError):
                continue
    out, todo = set(), [pid] if pid in parent else []
    while todo:
        p = todo.pop()
        out.add(p)
        todo += [c for c, pp in parent.items() if pp == p and c not in out]
    return out
