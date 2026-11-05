#!/usr/bin/env python3
"""exp_35 supervisor (plan_orbitrir_yaw_evidence.md r2 §7, §11 B2; r3 §12 B1-B3; r4 §13; §14 A1.3): the frozen
33-chunk phase-(i) loop on Yixun's shared GPU 1, its identity-bound co-tenancy, the atomic hand-over to GPU 0, the
reservation files, the persistent fail-closed state and the post-chunk E1 / E2 / probe gates. It NEVER signals any
process: a violation only stops further spawns (the running cell finishes).

Part 1 (round 6a ii): the pure parts -- the chunk order, the end-of-training predictor, the state file (schema v1,
atomic temp + fsync + rename + directory fsync), the O_EXCL reservations, the nvidia-smi and /proc tables.
Part 2: ``Supervisor`` -- (round 6a iii-a) lane ownership: per chunk the guard, the GPU / identity tables, the
predictor, waits that never spawn, the atomic hand-over to GPU 0; (round 6a iii-b) the chunk transaction: a
reservation, the spawn (the runner with ONE cell, or the launcher on the probe), polls every <= 5 min (a guard violation
or a new process on the lane latches a stop AFTER the running cell), verification, the gates, restart reconciliation.
Part 3 (round 6a iv): the real world, the collector's CLIs as the gates, the read-only dry run and the CLI:

  --h35 SHA --approved TUPLE --approved-sha256 HEX [--state F] [--reservations D] [--his-log F] [--logdir D]
  --dry-run        plan the 33 chunks, check identity / reservations / the guard READ-ONLY (spawns and writes nothing)
  --external-shas  print the sha256 of Yixun's queue script, evaluator and admission helper (§13 NIT 3)
  exit 0: complete (or the dry run holds); 2: bad arguments; 3: refused or stopped (the stop latch names why)

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3/r4.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True     # Codex C3c HIGH 2: no dynamic import below writes bytecode (the dry run writes NOTHING)
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


def _pin_reasons(h35, *shas):
    """Codex C3c BLOCKING 1: no state is created or loaded against a missing or malformed pin."""
    ok = isinstance(h35, str) and re.fullmatch(r"[0-9a-f]{40}", h35) and all(_hex(v) for v in shas)
    return [] if ok else [f"a missing or malformed pin among H35 {h35!r} and the sha256 values {shas!r}"]


def new_state(*, h35, cells_sha256, supervisor_sha256, approved_sha256):
    if _pin_reasons(h35, cells_sha256, supervisor_sha256, approved_sha256):
        raise ValueError(_pin_reasons(h35, cells_sha256, supervisor_sha256, approved_sha256)[0])
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


HEX64, GPU_RE = re.compile(r"[0-9a-f]{64}"), re.compile(r"GPU-[0-9A-Za-z-]+")
CHUNK_BASE, CHUNK_FLIGHT = ("n", "kind", "cell", "gate", "state"), ("lane", "shared", "started", "log")
GATE_KEYS = {"E1": ("e1",), "E2": ("e2",), "PROBE": ("probe_contract", "probe_json_sha256", "sidecar_sha256", "log_sha256"),
             None: ()}
VERDICTS = {"E1": ({0: "equal", 1: "mismatch"}, "refused", (0,)), "E2": ({0: "PASS", 1: "FAIL"}, "REFUSED", (0, 1))}


def _hex(v):
    return isinstance(v, str) and HEX64.fullmatch(v) is not None


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0


def _text(v):
    return isinstance(v, str) and v != ""


def _gate_reasons(c, latched):
    """A chunk's gate result: well formed; passed when gates_verified; a failure only under the stop latch."""
    g, n = c["gate"], c["n"]
    if g is None or (c["state"] == "runner_verified" and not any(k in c for k in GATE_KEYS[g])):
        return []
    if not all(k in c for k in GATE_KEYS[g]):
        return [f"chunk {n}: an incomplete {g} result"]
    if g == "PROBE":
        pc = c["probe_contract"]
        if not (isinstance(pc, dict) and list(pc) == ["rc", "out"] and type(pc["rc"]) is int and isinstance(pc["out"], str)):
            return [f"chunk {n}: a malformed probe contract result"]
        passed = pc["rc"] == 0 and all(_hex(c[k]) for k in GATE_KEYS[g][1:])
    else:
        r, (names, other, good) = c[GATE_KEYS[g][0]], VERDICTS[g]
        if not (isinstance(r, dict) and list(r) == ["rc", "verdict", "out"] and type(r["rc"]) is int
                and isinstance(r["out"], str) and r["verdict"] == names.get(r["rc"], other)):
            return [f"chunk {n}: a malformed {g} result {r!r:.120}"]
        passed = r["rc"] in good
    if not passed and (c["state"] == "gates_verified" or not latched):
        return [f"chunk {n}: a failed {g} result in a {c['state']} chunk" + ("" if latched else " without the stop latch")]
    return []


def _chunk_reasons(c, latched):
    """Codex C3b BLOCKING 1: each state carries exactly the evidence it claims."""
    n, st, cell = c["n"], c["state"], c["kind"] == "cell"
    keys, base, flight = set(c), set(CHUNK_BASE), set(CHUNK_BASE + CHUNK_FLIGHT)
    if st == "planned":
        return [] if keys == base else [f"chunk {n}: a planned chunk carries {sorted(keys - base)}"]
    if not flight <= keys:
        return [f"chunk {n} ({st}): missing {sorted(flight - keys)}"]
    out = [] if (isinstance(c["lane"], str) and GPU_RE.fullmatch(c["lane"]) and type(c["shared"]) is bool
                 and c["shared"] == (c["lane"] == GPU1_UUID) and _text(c["started"]) and _text(c["log"])) else [
        f"chunk {n}: a malformed lane / shared / started / log"]
    rc_key = "runner_rc" if cell else "probe_rc"
    run, receipt = {"ended", rc_key}, {"receipt_sha256", "campaign_digest"} if cell else set()
    extra = keys - flight
    if st == "in_flight":                         # a crash mid-chunk (bare) or a recorded failure (under the latch)
        if extra and (extra != run or not latched or type(c[rc_key]) is not int or not _text(c["ended"])):
            return out + [f"chunk {n}: in flight with {sorted(extra)} (a recorded failure: exactly {sorted(run)}, an int "
                          "rc and its end, under the stop latch)"]
        return out
    gate, adopt = set(GATE_KEYS[c["gate"]]), {"adopted_at_restart"} if cell else set()
    final = {"final_guard"} if st == "gates_verified" else set()
    need = run | receipt | (gate | final if st == "gates_verified" else set())
    if not need <= extra or extra - run - receipt - gate - adopt - final:
        return out + [f"chunk {n} ({st}): carries {sorted(extra)}, needs {sorted(need)}"]
    adopted = c.get("adopted_at_restart") is True
    if "adopted_at_restart" in c and not adopted:
        out.append(f"chunk {n}: adopted_at_restart {c['adopted_at_restart']!r}")
    if not ((type(c[rc_key]) is int and c[rc_key] == 0) or (adopted and c[rc_key] is None)):
        out.append(f"chunk {n}: rc {c[rc_key]!r} in a {st} chunk")
    if not (_text(c["ended"]) or (adopted and c["ended"] is None)):
        out.append(f"chunk {n}: ended {c['ended']!r}")
    if cell and not (_hex(c["receipt_sha256"]) and _hex(c["campaign_digest"])):
        out.append(f"chunk {n}: the receipt / campaign digest are not sha256 values")
    fg = c.get("final_guard")
    if st == "gates_verified" and not (
            isinstance(fg, dict) and list(fg) == ["status", "exit", "t", "time", "step", "anchor"] and fg["status"] == "OK"
            and type(fg["exit"]) is int and fg["exit"] == 0 and _num(fg["t"]) and _text(fg["time"])
            and type(fg["step"]) is int and fg["step"] >= 0 and fg["anchor"] is not None
            and not guard_module().anchor_schema_reasons(fg["anchor"])):
        out.append(f"chunk {n}: gates_verified without a passing final-guard record ({fg!r:.120})")
    return out + _gate_reasons(c, latched)


def _top_reasons(doc, his_log, reservations):
    """baseline (on THE configured log), the guard anchor, lane + reservation (in THE configured directory), the latch,
    the events -- exact (Codex C3b BLOCKING 1, C3c BLOCKING 1)."""
    b, lane, res, latch = doc["baseline"], doc["lane"], doc["reservation"], doc["stop_latch"]
    out = [] if b is None or (isinstance(b, dict) and list(b) == ["time", "step", "log"] and _num(b["time"])
                              and type(b["step"]) is int and b["step"] >= 0 and isinstance(b["log"], dict)
                              and list(b["log"]) == ["path", "inode", "size"] and b["log"]["path"] == his_log
                              and type(b["log"]["inode"]) is int and b["log"]["inode"] >= 0
                              and type(b["log"]["size"]) is int and b["log"]["size"] >= 0) else [
        f"a malformed baseline, or one on another log than {his_log}: {b!r:.160}"]
    out += guard_module().anchor_schema_reasons(doc["his_log_anchor"])
    if (lane is None) != (res is None):
        out.append("the lane and the reservation must be set together")
    elif lane is not None and not (
            isinstance(lane, dict) and list(lane) == ["uuid", "shared"] and isinstance(lane["uuid"], str)
            and GPU_RE.fullmatch(lane["uuid"]) and type(lane["shared"]) is bool and lane["shared"] == (lane["uuid"] == GPU1_UUID)
            and isinstance(res, dict) and list(res) == ["uuid", "epoch", "path"] and res["uuid"] == lane["uuid"]
            and type(res["epoch"]) is int and res["epoch"] >= 1 and res["path"] == reservation_path(reservations, res["uuid"])):
        out.append(f"the lane {lane!r} and the reservation {res!r} do not agree")
    if latch is not None and not (isinstance(latch, dict) and list(latch) == ["reason", "time"]
                                  and _text(latch["reason"]) and _text(latch["time"])):
        out.append(f"a malformed stop latch {latch!r:.120}")
    if not (isinstance(doc["events"], list) and all(isinstance(e, dict) and _num(e.get("t")) and _text(e.get("time"))
                                                    and _text(e.get("what")) for e in doc["events"])):
        out.append("malformed events")
    return out


def state_reasons(doc, *, h35, cells_sha256, supervisor_sha256, approved_sha256, his_log, reservations):
    if not isinstance(doc, dict) or list(doc) != list(STATE_KEYS):
        return [f"the state's keys are not {list(STATE_KEYS)}"]
    out = [f"{k}: {doc[k]!r} != {want!r}" for k, want in (
        ("schema_version", STATE_SCHEMA), ("campaign", "exp35"), ("h35", h35), ("cells_sha256", cells_sha256),
        ("supervisor_sha256", supervisor_sha256), ("approved_sha256", approved_sha256)) if doc[k] != want]
    plan = [(c["n"], c["kind"], c["cell"], c["gate"]) for c in chunk_plan()]
    if not isinstance(doc["chunks"], list) or not all(isinstance(c, dict) for c in doc["chunks"]) or [
            (c.get("n"), c.get("kind"), c.get("cell"), c.get("gate")) for c in doc["chunks"]] != plan:
        return out + ["the state's chunks are not the frozen 33-chunk order"]
    states = [c.get("state") for c in doc["chunks"]]
    out += [f"chunk {i + 1}: unknown state {s!r}" for i, s in enumerate(states) if s not in CHUNK_STATES]
    if out:
        return out
    done = 0
    while done < len(states) and states[done] == "gates_verified":
        done += 1
    if any(s != "planned" for s in states[done + 1:]):
        out.append(f"chunk order violated: {states[:done + 3]}... (only the first unfinished chunk may be started)")
    latched = doc["stop_latch"] is not None
    out += [x for c in doc["chunks"] for x in _chunk_reasons(c, latched)] + _top_reasons(doc, his_log, reservations)
    digests = {c["campaign_digest"] for c in doc["chunks"] if "campaign_digest" in c}
    out += [f"the chunks carry {len(digests)} campaign digests"] if len(digests) > 1 else []
    started = any(s != "planned" for s in states)
    if started != (doc["baseline"] is not None):
        out.append("a baseline exactly when a chunk has started")
    if started and done < len(states) and doc["reservation"] is None:
        out.append("a started, unfinished campaign holds no reservation")
    return out


def load_state(path, *, h35, cells_sha256, supervisor_sha256, approved_sha256, his_log, reservations):
    """The state, None when absent; ``ValueError`` when a pin is malformed or the state corrupt / inconsistent."""
    if _pin_reasons(h35, cells_sha256, supervisor_sha256, approved_sha256):
        raise ValueError(_pin_reasons(h35, cells_sha256, supervisor_sha256, approved_sha256)[0])
    try:
        with open(path) as f:
            doc = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as err:
        raise ValueError(f"the supervisor state {path} is corrupt ({err})") from err
    why = state_reasons(doc, h35=h35, cells_sha256=cells_sha256, supervisor_sha256=supervisor_sha256,
                        approved_sha256=approved_sha256, his_log=his_log, reservations=reservations)
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


RES_KEYS = ("holder", "h35", "gpu_uuid", "epoch", "created", "branch")


def read_reservation(dirpath, uuid):
    """The reservation file of ``uuid`` parsed: None when absent; ``ValueError`` when unreadable or malformed."""
    try:
        with open(reservation_path(dirpath, uuid)) as f:
            doc = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as err:
        raise ValueError(f"the reservation file of {uuid} is unreadable or malformed ({err})") from err
    if not isinstance(doc, dict) or not isinstance(doc.get("holder"), str):
        raise ValueError(f"the reservation file of {uuid} is not a reservation ({doc!r:.120})")
    return doc


def ours_reasons(doc, uuid, h35):
    """[] iff ``doc`` is exactly THIS campaign's exp_35 reservation of ``uuid`` (Codex C3b HIGH 2)."""
    ok = list(doc) == list(RES_KEYS) and doc["holder"] == "exp35" and doc["h35"] == h35 and doc["gpu_uuid"] == uuid \
        and doc["branch"] == BRANCH and type(doc["epoch"]) is int and doc["epoch"] >= 1 and _num(doc["created"])
    return [] if ok else [f"the exp_35 reservation of {uuid} is not this campaign's: {doc!r:.200}"]


def held_by_other(dirpath, uuid):
    """True when another holder -- or an unreadable / malformed file -- holds ``uuid``: never taken as free."""
    try:
        doc = read_reservation(dirpath, uuid)
    except ValueError:
        return True
    return doc is not None and doc["holder"] != "exp35"


def reservation_reasons(dirpath, held, uuids, h35):
    """[] iff exactly the reservation the state holds (``{"uuid", "epoch"}`` or None) is ours on disk, every exp_35
    file authenticated field by field; an unreadable or malformed file refuses."""
    out, mine = [], {}
    for u in uuids:
        try:
            doc = read_reservation(dirpath, u)
        except ValueError as err:
            out.append(str(err))
            continue
        if doc is not None and doc["holder"] == "exp35":
            why = ours_reasons(doc, u, h35)
            out += why
            mine.update({} if why else {u: doc})
    if out:
        return out
    if held is None:
        return [f"exp_35 reservation file(s) {sorted(mine)} exist but the state holds none"] if mine else []
    if set(mine) != {held["uuid"]}:
        return [f"the state holds {held['uuid']} (epoch {held['epoch']}), the exp_35 files on disk are {sorted(mine)}"]
    got = mine[held["uuid"]]["epoch"]
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


# --------------------------------------------------------------------------- #
# part 2 (round 6a iii): the loop
# --------------------------------------------------------------------------- #
RUNNER = "worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_eval_run.sh"
PROBE = "worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py"
LAUNCHER, NO_FLASH_DIR = "overlays/no_flash/exp35_no_flash_launch.py", "overlays/no_flash"
PY = "/home/yixunhu/miniconda3/envs/flac/bin/python"
BF_FARM_LINK = "outputs_FLAC/exp25_depth_canon/BF/epoch=8-step=40000.ckpt"   # the runner stages it (chunks 3-4)
WAIT_POLL_S = 60
CLEAN_ENV = ("ALLOW_SHARED_GPU", "EXP35_APPROVED_COTENANTS", "EXP35_APPROVED_COTENANTS_SHA256", "DRY_RUN", "PYTHONPATH",
             "CUDA_VISIBLE_DEVICES", "EXPECT_HEAD", "GPU_MAX_POLLS", "GPU_POLL_SECONDS")
_MODULES = {}


def _module(name):
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
        _MODULES[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MODULES[name])
    return _MODULES[name]


def identity_module():
    return _module("exp35_gpu_identity")


def guard_module():
    return _module("exp35_training_guard")


def _iso(t):
    return datetime.datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def _sha_file(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _env(**kw):
    return dict({k: v for k, v in os.environ.items() if k not in CLEAN_ENV}, **kw)


def runner_command(cfg, chunk, index, shared):
    """The runner on ONE cell; GPU_MAX_POLLS=1 so it never waits silently; the tuple only on the shared GPU 1."""
    env = _env(EXPECT_HEAD=cfg["h35"], GPU_MAX_POLLS="1", GPU_POLL_SECONDS="0")
    if shared:
        env.update(ALLOW_SHARED_GPU="1", EXP35_APPROVED_COTENANTS=cfg["approved"],
                   EXP35_APPROVED_COTENANTS_SHA256=cfg["approved_sha256"])
    return ["bash", RUNNER, "--lanes", str(index), "--only", chunk["cell"]], env


def probe_command(index):
    """§12 B3: chunk 5 -- the launcher directly on the probe with the frozen argv, the overlay ONLY on PYTHONPATH."""
    env = _env(CUDA_VISIBLE_DEVICES=str(index), HF_HUB_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.path.join(REPO, NO_FLASH_DIR))
    return [PY, LAUNCHER, PROBE, "--arm", "BF", "--ckpt-path", BF_FARM_LINK], env


class Supervisor:
    """``world``: now / sleep / nvsmi / spawn; ``gates``: verify / e1 / e2 / finalize (the collector's checks)."""

    def __init__(self, cfg, world, gates):
        self.cfg, self.w, self.g, self.G, self.I = cfg, world, gates, guard_module(), identity_module()
        self.st = self.last = self.note = None
        self.uuids = ()

    def save(self):
        write_state(self.cfg["state"], self.st)

    def event(self, what, **kw):
        t = self.w.now()
        self.st["events"].append(dict({"t": t, "time": _iso(t), "what": what}, **kw))
        self.save()
        print(f"[{_iso(t)}] {what} {json.dumps(kw) if kw else ''}", flush=True)

    def wait_note(self, what, **kw):
        if what != self.note:                       # one event per new reason of waiting
            self.note = what
            self.event(what, **kw)

    def latch(self, reason):
        if self.st["stop_latch"] is None:
            self.st["stop_latch"] = {"reason": reason, "time": _iso(self.w.now())}
        self.event("STOP LATCH", reason=reason)

    def observe(self):
        """The guard over his log; its anchor persisted at every observation (§13 NIT 1); None once latched."""
        r = self.G.observe(self.cfg["his_log"], baseline=(self.st["baseline"] or {}).get("step"),
                           anchor=self.st["his_log_anchor"])
        if r["exit"] in (0, 1):
            self.st["his_log_anchor"] = r["anchor"]
            self.save()
        if r["exit"] != 0:
            self.latch(f"the training guard: {r['status']}: {'; '.join(r['reasons'][:3])}")
            return None
        self.last = r
        return r

    def tables(self):
        m = gpu_map(self.w.nvsmi("map"))
        if len(m) != 2 or GPU1_UUID not in m.values():
            raise ValueError(f"the GPU map {m} is not this box's two cards with {GPU1_UUID}")
        return m, compute_apps(self.w.nvsmi("apps"))

    def trainer_alive(self, approved):
        """The approved processes all still run with their frozen identity (any change: they are gone)."""
        for p in approved["processes"]:
            try:
                live = self.I.proc_identity(p["pid"], self.cfg["proc"])
            except ValueError:
                return False
            if any(live[k] != p[k] for k in self.I.FIELDS):
                return False
        return True

    def ready(self, chunk):
        """Wait until a lane may take ``chunk``: ``(uuid, index, shared)``, or None after a latch."""
        while True:
            if self.observe() is None:
                return None
            try:
                m, apps = self.tables()
                approved = self.I.load_tuple(self.cfg["approved"], self.cfg["approved_sha256"])
            except ValueError as err:
                self.latch(f"the GPU / identity tables: {err}")
                return None
            idx, lane = {u: i for i, u in m.items()}, self.st["lane"]
            if lane is not None and not lane["shared"]:                 # after the hand-over: GPU 0, never shared
                foreign = sorted(p for p, u in apps if u == lane["uuid"])
                if foreign:
                    self.latch(f"a foreign process on GPU 0 ({lane['uuid']}) before chunk {chunk['n']}: pids {foreign}")
                    return None
                return lane["uuid"], idx[lane["uuid"]], False
            why = self.I.check_reasons([p for p, u in apps if u == GPU1_UUID], GPU1_UUID, approved, self.cfg["proc"])
            alive = self.trainer_alive(approved)
            if not why:
                ok, d = predictor_ok(self.last, now=self.w.now(), chunk=chunk)
                if ok:
                    self.note = None
                    return GPU1_UUID, idx[GPU1_UUID], True
                self.wait_note("predictor: no shared spawn (s43 could end before this chunk does)", chunk=chunk["n"], **d)
            elif alive:
                self.wait_note("no shared spawn: GPU 1 holds more than the approved trainer", why=why[:2])
            else:
                self.wait_note("no shared spawn: the approved trainer is gone; hand-over to GPU 0", why=why[:2])
            if not alive and self.handover(m, apps):
                continue
            if self.st["stop_latch"] is not None:
                return None
            self.w.sleep(WAIT_POLL_S)

    def handover(self, m, apps):
        """§12 B1.4: GPU 0 free on two checks 60 s apart -> its O_EXCL reservation -> the state (epoch + 1) -> the
        GPU 1 file removed. False: not now (waiting)."""
        uuid0 = next(u for u in m.values() if u != GPU1_UUID)
        for check in ("1st", "2nd"):
            busy = sorted(p for p, u in apps if u == uuid0)
            if busy:
                self.wait_note("hand-over waits: GPU 0 busy", pids=busy)
                return False
            self.event(f"GPU 0 free ({check} check)")
            if check == "1st":
                self.w.sleep(HANDOVER_GAP_S)
                try:
                    apps = self.tables()[1]
                except ValueError as err:
                    self.latch(f"the GPU tables during the hand-over: {err}")
                    return False
        old = self.st["reservation"]
        epoch = (old["epoch"] if old else 0) + 1
        try:
            path = create_reservation(self.cfg["reservations"], uuid0, epoch=epoch, h35=self.cfg["h35"], now=self.w.now())
        except FileExistsError:
            self.wait_note("hand-over waits: GPU 0 is reserved by another holder")
            return False
        self.st.update(lane={"uuid": uuid0, "shared": False}, reservation={"uuid": uuid0, "epoch": epoch, "path": path})
        self.save()
        if old:
            remove_reservation(old["path"])
        self.event("hand-over to GPU 0", epoch=epoch)
        return True

    def execute(self, c, lane):
        uuid, index, shared = lane
        if self.st["baseline"] is None:                                 # §11 B2: our first spawn
            r = self.last
            self.st["baseline"] = {"time": self.w.now(), "step": r["step"]["global"], "log": {
                "path": self.cfg["his_log"], "inode": r["log"]["inode"], "size": r["log"]["size"]}}
        if self.st["reservation"] is None:
            try:
                path = create_reservation(self.cfg["reservations"], uuid, epoch=1, h35=self.cfg["h35"], now=self.w.now())
            except FileExistsError:
                self.latch(f"{uuid} is reserved by another holder at our first spawn")
                return
            self.st.update(reservation={"uuid": uuid, "epoch": 1, "path": path}, lane={"uuid": uuid, "shared": shared})
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(self.w.now()))
        if c["kind"] == "probe":
            (argv, env), log = probe_command(index), os.path.join(self.cfg["logdir"], f"orbitrir_bf_cond_probe_{ts}.log")
        else:
            (argv, env), log = runner_command(self.cfg, c, index, shared), os.path.join(
                self.cfg["logdir"], f"exp35_supervisor_{ts}_chunk{c['n']:02d}_{c['cell']}.log")
        why = reservation_reasons(self.cfg["reservations"], self.st["reservation"], self.uuids, self.cfg["h35"])
        if why:                                        # re-authenticated right before every spawn (Codex C3c HIGH 1)
            self.latch(f"the reservation before chunk {c['n']}: " + "; ".join(why))
            return
        c.update(state="in_flight", lane=uuid, shared=shared, started=_iso(self.w.now()), log=log)
        self.save()
        rc = self.watch(self.w.spawn(argv, env, log), c, uuid, shared)
        c.update({"ended": _iso(self.w.now()), "probe_rc" if c["kind"] == "probe" else "runner_rc": rc})
        self.save()
        if rc != 0:
            self.latch(f"chunk {c['n']} ({c['cell'] or 'the probe'}) exited rc {rc}")
            return
        self.verified(c)

    def watch(self, proc, c, uuid, shared):
        """Poll the running chunk, NEVER signalling it: the guard and the lane's process set every <= 5 min."""
        try:
            approved = {p["pid"]: p for p in self.I.load_tuple(self.cfg["approved"], self.cfg["approved_sha256"])[
                "processes"]} if shared else {}
        except ValueError as err:                      # still never signalled: watch it to its end, then stop
            self.latch(f"the approved tuple while chunk {c['n']} ran: {err}")
            approved = {}
        while proc.poll() is None:
            self.w.sleep(GUARD_POLL_S)
            if self.st["stop_latch"] is not None:
                continue
            self.observe()
            try:
                apps = self.tables()[1]
            except ValueError as err:
                self.latch(f"the GPU tables while chunk {c['n']} ran: {err}")
                continue
            why = reservation_reasons(self.cfg["reservations"], self.st["reservation"], self.uuids, self.cfg["h35"])
            if why:                                    # ... and at every poll (Codex C3c HIGH 1)
                self.latch(f"the reservation changed while chunk {c['n']} ran: " + "; ".join(why))
                continue
            ours = descendants(proc.pid, self.cfg["proc"])
            bad = sorted(p for p, u in apps if u == uuid and p not in ours and (p not in approved or self.replaced(approved[p])))
            if bad and self.st["stop_latch"] is None:     # every foreign process vs the COMPLETE identity (C3c HIGH 3)
                self.latch(f"a new process, or a process with another identity, on {'GPU 1' if shared else 'GPU 0'} "
                           f"({uuid}) while chunk {c['n']} ran: pids {bad}")
        return proc.returncode

    def replaced(self, entry):
        """The approved PID now runs another process (start time, cwd, cmdline or logs differ). Unreadable = gone: a
        trainer that exited is not a replacement; a reuse is caught at the next poll or the pre-spawn check."""
        try:
            live = self.I.proc_identity(entry["pid"], self.cfg["proc"])
        except ValueError:
            return False
        return any(live[k] != entry[k] for k in self.I.FIELDS)

    def probe_ledger_reasons(self, c, j, rel, digest):
        """Codex C3c HIGH 4: the three files hashed, the sidecar sealing the JSON, the JSON's launch record naming THIS
        log, its sha256 and the campaign digest."""
        out = [f"{k} absent" for k in ("probe_json_sha256", "sidecar_sha256", "log_sha256") if not _hex(c[k])]
        if out:
            return out
        try:
            with open(j + ".sha256") as f:
                side = f.read()
            with open(j) as f:
                launch = json.load(f)["provenance"]["launch"]
        except (OSError, ValueError, KeyError, TypeError) as err:
            return [f"the finalized probe JSON is unreadable ({err})"]
        if side != f"{c['probe_json_sha256']}  {os.path.basename(j)}\n":
            out.append("the sidecar does not seal the probe JSON")
        if not (isinstance(launch, dict) and launch.get("log_path") == rel and launch.get("log_sha256") == c["log_sha256"]
                and launch.get("campaign_digest") == digest):
            out.append("the probe's launch record does not name this log, its sha256 and the campaign digest")
        return out

    def campaign_digest(self):
        return next((c["campaign_digest"] for c in self.st["chunks"] if c.get("campaign_digest")), None)

    def verified(self, c, adopted=False):
        """-> runner_verified (a cell: its receipt verified under the campaign's ONE digest), then its gates."""
        if c["kind"] == "cell":
            ok, info = self.g.verify(c)
            digest = self.campaign_digest()
            if not ok or (digest is not None and info["campaign_digest"] != digest):
                self.latch(f"chunk {c['n']} {c['cell']}: verification refused: {info}")
                return
            c.update(receipt_sha256=info["receipt_sha256"], campaign_digest=info["campaign_digest"])
        c.update(state="runner_verified", **({"adopted_at_restart": True} if adopted else {}))
        self.save()
        if self.st["stop_latch"] is None:
            self.gates(c)

    def gates(self, c):
        """E1 after chunks 1-2 (a mismatch: a hard stop), E2 after chunk 4 (the 45-degree source only), the probe's
        finalization + PROBE_CONTRACT_BF after chunk 5 (§13 NIT 2 ledger), and the final guard on every chunk."""
        if c["gate"] == "E1":
            k = 1 if "_K1_" in c["cell"] else 8
            rc, out = self.g.e1(k)
            c["e1"] = {"rc": rc, "verdict": {0: "equal", 1: "mismatch"}.get(rc, "refused"), "out": out[-2000:]}
            bad = rc != 0 and f"E1 K{k} {c['e1']['verdict']}: the control cell is not the exp_25 cell (a hard stop)"
        elif c["gate"] == "E2":
            rc, out = self.g.e2()
            c["e2"] = {"rc": rc, "verdict": {0: "PASS", 1: "FAIL"}.get(rc, "REFUSED"), "out": out[-2000:]}
            bad = rc not in (0, 1) and "E2 refused"
        elif c["gate"] == "PROBE":
            rel, digest = os.path.relpath(c["log"], REPO), self.campaign_digest()
            rc, out = self.g.finalize(rel, digest)
            j = self.cfg.get("probe_json") or os.path.join(HERE, "orbitrir_bf_cond_probe.json")
            c.update(probe_contract={"rc": rc, "out": out[-2000:]}, probe_json_sha256=_sha_file(j),
                     sidecar_sha256=_sha_file(j + ".sha256"), log_sha256=_sha_file(c["log"]))
            why = self.probe_ledger_reasons(c, j, rel, digest) if rc == 0 else []
            bad = (rc != 0 and "the B-F probe: finalization / PROBE_CONTRACT_BF refused") or (
                why and "the B-F probe's ledger: " + "; ".join(why))
        else:
            bad = False
        self.save()
        if bad:
            self.latch(bad)
            return
        g = self.observe()                            # the final guard, kept as this chunk's evidence (C3c BLOCKING 1)
        if g is not None:
            c.update(final_guard={"status": g["status"], "exit": g["exit"], "t": self.w.now(), "time": _iso(self.w.now()),
                                  "step": g["step"]["global"], "anchor": g["anchor"]}, state="gates_verified")
            self.save()

    def reconcile(self):
        """§12 B2: in flight -> its receipt must verify (then its gates), else the latch; runner_verified -> its gates."""
        for c in self.st["chunks"]:
            if c["state"] == "in_flight":
                if c["kind"] == "cell" and self.g.verify(c)[0]:
                    self.event("reconcile: the in-flight chunk's receipt verifies; its gates re-run", chunk=c["n"])
                    c.update(ended=None, runner_rc=None)               # unknown: the supervisor was not there
                    return self.verified(c, adopted=True)
                return self.latch(f"chunk {c['n']} was in flight when the supervisor stopped and has no verified receipt")
            if c["state"] == "runner_verified":
                self.event("reconcile: re-running the gates", chunk=c["n"])
                return self.gates(c)

    def run(self):
        """0: phase (i) complete; 3: refused or stopped (the latch names why)."""
        c = self.cfg
        try:
            st = load_state(c["state"], h35=c["h35"], cells_sha256=c["cells_sha256"], supervisor_sha256=c["supervisor_sha256"],
                            approved_sha256=c["approved_sha256"], his_log=c["his_log"], reservations=c["reservations"])
            self.uuids = uuids = tuple(sorted(self.tables()[0].values()))
            self.st = st or new_state(h35=c["h35"], cells_sha256=c["cells_sha256"], supervisor_sha256=c["supervisor_sha256"],
                                      approved_sha256=c["approved_sha256"])
        except ValueError as err:
            print(f"SUPERVISOR REFUSED: {err}")
            return 3
        if self.st["stop_latch"]:
            print(f"SUPERVISOR REFUSED: the stop latch is set ({self.st['stop_latch']}); only the Planner clears it")
            return 3
        self.event("supervisor started" + (" (a new state)" if st is None else " (resumed)"))
        why = reservation_reasons(c["reservations"], self.st["reservation"], uuids, c["h35"])
        if why:
            self.latch("the reservations are inconsistent at the start: " + "; ".join(why))
            return 3
        self.reconcile()
        for ch in self.st["chunks"]:
            if self.st["stop_latch"] is not None:
                break
            if ch["state"] == "planned":
                lane = self.ready(ch)
                if lane is None:
                    break
                self.execute(ch, lane)
        if self.st["stop_latch"] is not None or any(ch["state"] != "gates_verified" for ch in self.st["chunks"]):
            return 3
        if self.st["reservation"]:
            remove_reservation(self.st["reservation"]["path"])
            self.st.update(reservation=None, lane=None)
        self.event("phase (i) complete; the reservation released")
        return 0


# --------------------------------------------------------------------------- #
# part 3 (round 6a iv): the real world, the collector gates, the dry run, the CLI
# --------------------------------------------------------------------------- #
COLLECT = "worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py"
CELLS = "worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cells.json"
PROBE_JSON = f"{EXP35}/orbitrir_bf_cond_probe.json"
STATE, PROC = os.path.join(HERE, "exp35_supervisor_state.json"), "/proc"
_KIT = "/home/yixunhu/codespace/flac-fork-attn/worklog/worklog_yixun/exp_30_attn"
HIS_LOG = f"{_KIT}/logs/train_exp30_ACEattn_s43_20261104T061255_741947.log"                   # §7
EXTERNAL = {"queue": f"{_KIT}/exp30_attn_queue.sh", "evaluator": f"{_KIT}/exp30_attn_eval.sh",   # §13 NIT 3
            "admission": f"{_KIT}/exp30_attn_queue_state.py"}


class World:
    """The real world: wall time, sleep, nvidia-smi (any failure is a ValueError), a chunk spawned with its log."""

    def now(self):
        return time.time()

    def sleep(self, s):
        time.sleep(s)

    def nvsmi(self, query):
        argv = ["nvidia-smi", {"map": "--query-gpu=index,uuid", "apps": "--query-compute-apps=pid,gpu_uuid"}[query],
                "--format=csv,noheader"]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as err:
            raise ValueError(f"nvidia-smi {argv[1]}: {err}") from err
        if r.returncode != 0:
            raise ValueError(f"nvidia-smi {argv[1]} failed (rc {r.returncode}): {r.stderr.strip()[:200]}")
        return r.stdout

    def spawn(self, argv, env, log):
        with open(log, "w") as f:
            return subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)


def collector_module():
    if "collector" not in _MODULES:
        spec = importlib.util.spec_from_file_location("depth_canon_collect_supervisor", os.path.join(REPO, COLLECT))
        _MODULES["collector"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MODULES["collector"])
    return _MODULES["collector"]


class CollectorGates:
    """The gates through the collector's own CLIs at the pin (CPU subprocesses of the flac interpreter)."""

    def __init__(self, cfg, run=subprocess.run):
        self.cfg, self.run = cfg, run
        self.repo, self.cells = cfg.get("repo", REPO), cfg.get("cells", os.path.join(REPO, CELLS))

    def _cli(self, *args):
        r = self.run([PY, COLLECT, *args], cwd=self.repo, capture_output=True, text=True,
                     env=_env(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"))
        return r.returncode, (r.stdout or "") + (r.stderr or "")

    def verify(self, chunk):
        """The cell's receipt (located by the collector's own cell_paths) and ``--verify-cell`` under its digest."""
        C = collector_module()
        m = C.load_cells(self.cells)
        row = next(r for r in m["cells"] if r["cell"] == chunk["cell"])
        target, _ = C.expected_checkpoint(row, m, None)
        p = C.cell_paths(row, os.path.join(m["farm_root"], row["arm"], os.path.basename(target)), m["dump_root"])
        try:
            with open(os.path.join(self.repo, p.receipt), "rb") as f:
                raw = f.read()
            digest = json.loads(raw)["campaign_digest"]
        except (OSError, ValueError, KeyError, TypeError) as err:
            return False, {"reasons": [f"the receipt {p.receipt}: {err}"]}
        if not _hex(digest):
            return False, {"reasons": [f"the receipt {p.receipt} carries no campaign digest"]}
        rc, out = self._cli("--verify-cell", p.stem, "--pin", self.cfg["h35"], "--campaign-digest", digest)
        if rc != 0:
            return False, {"reasons": [out[-1500:]]}
        return True, {"receipt_sha256": hashlib.sha256(raw).hexdigest(), "campaign_digest": digest}

    def e1(self, k):
        return self._cli("--exp35-ctl-check", str(k), "--pin", self.cfg["h35"])

    def e2(self):
        return self._cli("--exp35-e2", "--pin", self.cfg["h35"])

    def finalize(self, log, digest):
        return self._cli("--finalize-bf-probe", PROBE_JSON, "--probe-log", log, "--campaign-digest", digest or "",
                         "--pin", self.cfg["h35"])


def dry_run(cfg, world):
    """§11 R8: plan the 33 chunks and check the live state READ-ONLY -- nvidia-smi, /proc, the reservations, the guard
    over his log, the state if one exists. Nothing is spawned or written. 0 iff identity, reservations and guard hold."""
    I, G = identity_module(), guard_module()
    try:
        m, apps = gpu_map(world.nvsmi("map")), compute_apps(world.nvsmi("apps"))
        if len(m) != 2 or GPU1_UUID not in m.values():                 # exactly as Supervisor.tables() (C3c HIGH 2)
            raise ValueError(f"the GPU map {m} is not this box's two cards with {GPU1_UUID}")
        approved = I.load_tuple(cfg["approved"], cfg["approved_sha256"])
        st = load_state(cfg["state"], h35=cfg["h35"], cells_sha256=cfg["cells_sha256"],
                        supervisor_sha256=cfg["supervisor_sha256"], approved_sha256=cfg["approved_sha256"],
                        his_log=cfg["his_log"], reservations=cfg["reservations"])
    except ValueError as err:
        print(f"DRY RUN REFUSED: {err}")
        return 1
    idx, lane = {u: i for i, u in m.items()}, (st or {}).get("lane") or {"uuid": GPU1_UUID, "shared": True}
    print(f"exp_35 supervisor dry run {_iso(world.now())}: H35 {cfg['h35']}; state {'resumed' if st else 'none (new)'}; "
          f"lane {lane['uuid']} (index {idx.get(lane['uuid'])}, shared {lane['shared']}); GPU map {m}")
    for c in chunk_plan():
        state = st["chunks"][c["n"] - 1]["state"] if st else "planned"
        argv = probe_command(idx.get(lane["uuid"]))[0] if c["kind"] == "probe" else runner_command(
            cfg, c, idx.get(lane["uuid"]), lane["shared"])[0]
        print(f"  chunk {c['n']:2d} {state:15s} gate {c['gate'] or '-':5s} {planning_minutes(c):4.1f} min  {' '.join(argv)}")
    on_lane = sorted(p for p, u in apps if u == lane["uuid"])         # the state's CURRENT lane (C3c HIGH 2)
    if lane["shared"]:
        why = I.check_reasons(on_lane, GPU1_UUID, approved, cfg["proc"])
        print(("identity OK" if not why else "IDENTITY MISMATCH") + f": GPU 1 foreign pids {on_lane}; the approved tuple "
              f"{[p['pid'] for p in approved['processes']]}" + "".join(f"\n    - {x}" for x in why))
    else:
        why = [f"pids {on_lane}"] if on_lane else []
        print("GPU 0 OK: no foreign compute process" if not why else f"FOREIGN PROCESS ON GPU 0 ({lane['uuid']}): pids {on_lane}")
    res = reservation_reasons(cfg["reservations"], (st or {}).get("reservation"), tuple(sorted(m.values())), cfg["h35"])
    print("reservations OK" if not res else "RESERVATIONS INCONSISTENT: " + "; ".join(res))
    g = G.observe(cfg["his_log"], baseline=((st or {}).get("baseline") or {}).get("step"),
                  anchor=(st or {}).get("his_log_anchor"))
    print(f"guard {'OK' if g['exit'] == 0 else g['status']}: step {g.get('step')}; {'; '.join(g['reasons'][:3]) or 'no violation'}")
    nxt = next((c for c in chunk_plan() if not st or st["chunks"][c["n"] - 1]["state"] != "gates_verified"), None)
    if nxt is not None and g["exit"] == 0 and lane["shared"]:            # the predictor guards SHARED spawns only
        ok, d = predictor_ok(g, now=world.now(), chunk=nxt)
        print(f"predictor for chunk {nxt['n']}: {'a shared spawn allowed' if ok else 'NO shared spawn'} {json.dumps(d)}")
    latch = (st or {}).get("stop_latch")
    print(f"STOP LATCH set: {latch}" if latch else "no stop latch")
    good = not why and not res and g["exit"] == 0 and not latch
    print("DRY RUN OK: nothing was spawned or written" if good else "DRY RUN REFUSED (nothing was spawned or written)")
    return 0 if good else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h35")
    ap.add_argument("--approved", help="the approved identity tuple (exp35_gpu_identity.py --snapshot)")
    ap.add_argument("--approved-sha256", help="the launch entry's sha256 of the tuple's bytes")
    ap.add_argument("--state", default=STATE)
    ap.add_argument("--reservations", default=RESERVATIONS)
    ap.add_argument("--his-log", default=HIS_LOG)
    ap.add_argument("--logdir", default=HERE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--external-shas", action="store_true")
    ap.add_argument("--proc", default=PROC, help=argparse.SUPPRESS)       # tests: a fake /proc tree
    args = ap.parse_args(argv)
    if args.external_shas:
        out = {k: {"path": p, "sha256": _sha_file(p)} for k, p in EXTERNAL.items()}
        print(json.dumps(out, indent=1))
        return 0 if all(v["sha256"] for v in out.values()) else 1
    if not (args.h35 and re.fullmatch(r"[0-9a-f]{40}", args.h35) and args.approved and _hex(args.approved_sha256)):
        print("SUPERVISOR REFUSED: --h35 (40-hex), --approved and --approved-sha256 (64-hex) are required")
        return 2
    if not (_hex(_sha_file(os.path.join(REPO, CELLS))) and _hex(_sha_file(os.path.abspath(__file__)))):
        print(f"SUPERVISOR REFUSED: the cells manifest {CELLS} or the supervisor cannot be hashed (a pin, C3c BLOCKING 1)")
        return 2
    cfg = {"h35": args.h35, "state": args.state, "approved": args.approved, "approved_sha256": args.approved_sha256,
           "reservations": args.reservations, "his_log": args.his_log, "logdir": args.logdir, "proc": args.proc,
           "cells_sha256": _sha_file(os.path.join(REPO, CELLS)), "supervisor_sha256": _sha_file(os.path.abspath(__file__)),
           "probe_json": os.path.join(REPO, PROBE_JSON)}
    world = World()
    if args.dry_run:
        return dry_run(cfg, world)
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != args.h35:
        print(f"SUPERVISOR REFUSED: HEAD {head} is not H35 {args.h35}")
        return 3
    os.makedirs(args.reservations, exist_ok=True)
    return Supervisor(cfg, world, CollectorGates(cfg)).run()


if __name__ == "__main__":
    sys.exit(main())
