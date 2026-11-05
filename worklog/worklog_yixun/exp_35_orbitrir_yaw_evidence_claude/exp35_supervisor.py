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

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3/r4.
"""
import datetime
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import time

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
