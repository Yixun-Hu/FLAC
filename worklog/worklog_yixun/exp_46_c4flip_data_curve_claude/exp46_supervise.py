#!/usr/bin/env python3
"""exp_46 training supervisor (forked from exp_45's exp45_supervise.py at X45f, from exp_38's; plan_c4flip_data_curve.md r3 §3.2) —
the one process that owns a training launch on either platform
(plan_data_efficiency.md r2 §10 B6 / B7; Codex code r1 HIGH 4, 5, 7).

Called by exp46_train.sbatch (neuronic; exp_38's A6000 launcher is not forked) after it
sanitised the environment; it runs every gate, holds the locks, launches train.py FROM an immutable
closure snapshot, owns the child's process group, gates the first step, and publishes the run's
sealed receipt only after every end audit passes:

    MODE=DRY_RUN    every CPU-side gate + the exact commands; spawns nothing
    MODE=SMOKE      neuronic only: the production-topology 30-step smoke -> sealed SMOKE_DDP receipt
    MODE=FULL       neuronic only: the monolithic 0 -> 40,000 run -> sealed TRAINING_COMPLETE.json in the run dir

exp_46 has NO A6000 training path (plan r3 §3.2; Codex code r1 finding 12): SMOKE_DDP is refused on every platform
and SMOKE on the A6000 box, both before any gate, claim, lock, probe or directory (the inherited A6000 SMOKE
destination outputs_FLAC/exp38_smoke is never written); FULL refuses on any platform but neuronic (D1 / D2).

Process ownership: the child is started in its own session (pgid == its pid) with the card locks
passed down (``pass_fds``), so the locks live as long as ANY process of the run lives -- a killed
supervisor cannot release a card under a running trainer. TERM / INT / HUP are forwarded to that
group only while the leader is unreaped (its pid -- and so the group id -- cannot be reused) and
its /proc start time is the one recorded at spawn; after GRACE_S the group gets SIGKILL. Every exit
path rewrites the launch record atomically; a failed run keeps its ``.exp46_lock`` crash marker.

Exit: 0 ok · 2 a gate refused · 3 environment / launch error · 4 the start gate failed (our group
terminated) · 5 an end gate / audit failed · 128+N terminated by signal N · else the child's status.

Written by the exp_38 Coder (Claude Opus 5.5).
"""
import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time

# Codex code r1 finding 1: exp_38's tools are imported from exp_38's own folder; the sbatch runs this process with
# -B, a standalone invocation (DRY_RUN from a command sheet) does not -- no import of this process caches bytecode
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import exp46_registry as reg  # noqa: E402
# exp_38's read-only tools (exp38_subset_audit, exp38_content_manifest) are imported from exp_38's folder
sys.path.insert(1, os.path.join(reg.REPO, reg.EXP38_DIR))

EXIT_GATE, EXIT_ENV, EXIT_START, EXIT_END = 2, 3, 4, 5
GRACE_S = 120
START_TIMEOUT_S = 5400
EVIDENCE_TIMEOUT_S = 1800
SNAPSHOT_ROOT = {"a6000": "/home/yixunhu/codespace/exp-46-snapshots"}
LOCKDIR = "/home/yixunhu/codespace"              # .exp38_card<i>.lock, shared with exp38_eval.py + exp46_eval.py
SLURM_RANK_VARS = ("SLURM_NTASKS", "SLURM_JOB_NAME", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID",
                   "SLURM_NTASKS_PER_NODE", "SLURM_NPROCS")
A6000_PY = re.compile(r".*/envs/flac/bin/python[0-9.]*$")


class Refused(RuntimeError):
    def __init__(self, msg, code=EXIT_GATE):
        super().__init__(msg)
        self.code = code


# ---------------------------------------------------------------- signals
_PENDING = []
_ATTEMPT = {"path": None, "done": False}       # the neuronic attempt record this process claimed


def _on_signal(signum, _frame):
    _PENDING.append(signum)


def install_signal_handlers():
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _on_signal)


class Stopped(Exception):
    def __init__(self, signum):
        super().__init__(f"signal {signum}")
        self.signum = signum


def check_signals():
    if _PENDING:
        raise Stopped(_PENDING[0])


def sleep_checked(seconds):
    end = time.monotonic() + seconds
    while True:
        check_signals()
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, 0.5))


# ---------------------------------------------------------------- the owned child
def proc_start_time(pid):
    """Field 22 (starttime) of /proc/<pid>/stat, or None if the pid is gone."""
    try:
        text = open(f"/proc/{pid}/stat").read()
    except OSError:
        return None
    return int(text[text.rindex(")") + 2:].split()[19])


class OwnedChild:
    """A child in its own session whose group we may signal only while we own it."""

    def __init__(self, cmd, cwd, env, log_path, pass_fds=()):
        self.log = open(log_path, "ab")
        self.p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=self.log,
                                  stderr=subprocess.STDOUT, start_new_session=True, pass_fds=tuple(pass_fds))
        self.pid = self.p.pid
        self.start = proc_start_time(self.pid)
        self.pgid = os.getpgid(self.pid)
        if self.pgid != self.pid or self.start is None:
            self.p.kill()
            self.p.wait()
            raise Refused(f"child {self.pid} did not start its own process group", EXIT_ENV)
        self.rc = None

    def exited(self):
        """True once the leader has exited (it stays a zombie: NOT reaped, the pid stays ours)."""
        if self.rc is not None:
            return True
        return os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None

    def owned(self):
        return self.rc is None and proc_start_time(self.pid) == self.start

    def signal_group(self, sig):
        if self.owned():
            try:
                os.killpg(self.pgid, sig)
            except ProcessLookupError:
                pass

    def reap(self):
        if self.rc is None:
            self.signal_group(signal.SIGKILL)      # stragglers of the group, while the pgid is still ours
            self.rc = self.p.wait()
            self.log.close()
        return self.rc

    def terminate(self, grace=GRACE_S):
        """TERM the group, wait up to ``grace`` s, then KILL; bounded; returns the exit status."""
        self.signal_group(signal.SIGTERM)
        end = time.monotonic() + grace
        while not self.exited() and time.monotonic() < end:
            time.sleep(0.5)
        if not self.exited():
            self.signal_group(signal.SIGKILL)
            end = time.monotonic() + 60
            while not self.exited() and time.monotonic() < end:
                time.sleep(0.5)
        return self.reap()


def supervise(child, record, start_check=None, start_timeout=START_TIMEOUT_S, evidence_check=None,
              evidence_timeout=EVIDENCE_TIMEOUT_S, grace=GRACE_S, poll=2.0):
    """Own ``child`` until it exits. ``start_check()`` -> None (not yet) / [] (PASS) / [problems];
    ``evidence_check()`` likewise. A failed gate or a signal terminates OUR group (bounded) and
    finalizes ``record``. Returns the process exit code this supervisor should use."""
    try:
        for name, check, timeout in (("start", start_check, start_timeout), ("evidence", evidence_check, evidence_timeout)):
            if check is None:
                continue
            end = time.monotonic() + timeout
            while True:
                res = check()
                if res is not None:
                    break
                if child.exited():
                    res = [f"the training process exited before the {name} gate"]
                    break
                if time.monotonic() > end:
                    res = [f"no {name} gate result within {timeout} s"]
                    break
                sleep_checked(poll)
            record.update(**{f"{name}_gate": "PASS" if not res else res})
            if res:
                rc = child.terminate(grace)
                record.update(strict=True, status=f"{name} gate failed; our process group terminated", child_rc=rc,
                              finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
                return EXIT_START
        while not child.exited():
            sleep_checked(poll)
        rc = child.reap()
        record.update(strict=True, child_rc=rc, child_exited_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        return 128 - rc if rc < 0 else rc
    except Stopped as st:
        rc = child.terminate(grace)
        record.update(strict=True, status=f"terminated by signal {st.signum}; our process group stopped", child_rc=rc,
                      finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        return 128 + st.signum
    except BaseException:                        # a gate callback / record write raised: never leave the group
        if child.rc is None:
            child.terminate(grace)
        raise


class Record:
    """The launch record, rewritten atomically (JSON) at every change. ``paths[0]`` is AUTHORITATIVE:
    its write must succeed at creation (before any launch) and at every terminal transition
    (``strict=True``), else OSError propagates; the other copies are best-effort, and a failed
    non-terminal update of the authoritative copy is reported, not fatal, while a trainer runs."""

    def __init__(self, paths, **fields):
        self.paths, self.data, self.sealed = list(paths), dict(fields), False
        self.flush(strict=True)

    def seal(self):
        """The commit marker now binds this record's bytes (Codex code r1 finding 1): nothing rewrites it after this
        -- a later update (a release report, an interrupt's error status) is reported, never written."""
        self.sealed = True

    def update(self, strict=False, **fields):
        if self.sealed:
            print(f"launch record sealed by the commit marker: not rewritten ({sorted(fields)})", file=sys.stderr,
                  flush=True)
            return
        self.data.update(fields)
        self.flush(strict=strict)

    def flush(self, strict=False):
        text = (json.dumps(self.data, indent=1, sort_keys=True, default=str) + "\n").encode()
        for i, p in enumerate(self.paths):
            try:
                reg.atomic_write_bytes(p, text)
            except OSError as err:
                if i == 0 and strict:
                    raise
                print(f"record write {p} failed: {err}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- gates
def git(*args, repo=None):
    return subprocess.run(["git", "-C", repo or reg.REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def child_env(platform, backend, launch_root, cuda_devices=None):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")
           and k not in reg.LAUNCHER_VARS and not k.startswith(reg.LAUNCHER_PREFIXES)}
    for k in list(env):
        # neuronic: PL 2.1 must elect TorchElastic, not SLURM (the SLURM rank variables are recorded first);
        # this box: no SLURM variable at all (SLURM_PROCID would shift train.py's seed)
        if (platform == "neuronic" and k in SLURM_RANK_VARS) or (platform == "a6000" and k.startswith("SLURM_")):
            env.pop(k)
    env.update({"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1"})
    if backend == "sdpa":
        env.update({"PYTHONPATH": os.path.join(launch_root, "overlays", "no_flash"), "PYTHONDONTWRITEBYTECODE": "1"})
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    return env


def backend_gate(backend, launch_root, env):
    """Resolve ``flash_attn`` AND import the production transformer exactly as the child will (same
    interpreter, cwd and env): flash needs a real flash_attn bound as ``src.models.transformer.
    flash_attn_func``; sdpa needs the overlay (or none) and that name unbound."""
    code = ("import importlib.util, json, os, sys; sys.path.insert(0, os.getcwd()); "
            "s = importlib.util.find_spec('flash_attn'); "
            "import src.models.transformer as T; "
            "print(json.dumps([os.path.realpath(s.origin) if s and s.origin else '', T.flash_attn_func is not None]))")
    out = subprocess.run([sys.executable, "-c", code], cwd=launch_root, env=env, capture_output=True, text=True, timeout=600)
    if out.returncode:
        raise Refused(f"backend probe failed: {out.stderr.strip()[-400:]}", EXIT_ENV)
    origin, bound = json.loads(out.stdout.strip().split("\n")[-1])
    overlay = os.path.realpath(os.path.join(launch_root, "overlays", "no_flash", "flash_attn.py"))
    if backend == "flash" and (not origin or origin == overlay or bound is not True):
        raise Refused(f"ATTN_BACKEND=flash but the child's transformer binds flash_attn_func={bound} ({origin or 'absent'})")
    if backend == "sdpa" and (origin not in ("", overlay) or bound is not False):
        raise Refused(f"ATTN_BACKEND=sdpa but the child would import {origin} (flash bound: {bound})")
    return {"attention_backend": backend, "flash_attn": origin or "absent", "flash_attn_func_bound": bound}


def interpreter_gate(platform, expect_python=None):
    exe = os.path.realpath(sys.executable)
    if platform == "a6000" and not A6000_PY.match(exe):
        raise Refused(f"python resolves to {exe}, not the flac env", EXIT_ENV)
    if platform == "neuronic" and (not expect_python or exe != os.path.realpath(expect_python)):
        raise Refused(f"python is {exe}, not {expect_python}", EXIT_ENV)
    probs = reg.env_problems() + reg.launcher_env_problems(platform)
    if probs:
        raise Refused("; ".join(probs), EXIT_ENV)
    return exe


def nvidia_rows():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free,name", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=60)
    if out.returncode or apps.returncode:
        raise Refused("nvidia-smi query failed", EXIT_ENV)
    rows = [[x.strip() for x in ln.split(",")] for ln in out.stdout.strip().split("\n") if ln.strip()]
    app_rows = [[x.strip() for x in ln.split(",")] for ln in apps.stdout.strip().split("\n") if ln.strip()]
    return rows, app_rows


def a6000_gpu_gate(cards, policy, min_free, rows=None, apps=None):
    if rows is None:
        rows, apps = nvidia_rows()
    if len(rows) != 2:
        raise Refused(f"expected exactly two visible GPUs, got {len(rows)}")
    out = []
    for idx in cards:
        row = [r for r in rows if r[0] == str(idx)]
        if len(row) != 1 or not row[0][1].startswith("GPU-") or not row[0][2].isdigit():
            raise Refused(f"GPU {idx}: unparseable nvidia-smi row", EXIT_ENV)
        _, uuid, free, name = row[0]
        if name != "NVIDIA RTX A6000" or uuid not in reg.A6000_UUIDS:
            raise Refused(f"GPU {idx} is {name} {uuid}, not a registered A6000")
        foreign = [a[1] for a in apps if a[0] == uuid]
        print(f"GPU {idx} {uuid} ({name}): {free} MiB free, foreign compute pids {foreign}", flush=True)
        if policy == "idle" and foreign:
            raise Refused(f"GPU {idx} is not idle (GPU_POLICY=idle)")
        if int(free) < min_free:
            raise Refused(f"GPU {idx}: {free} MiB free < {min_free} MiB")
        out.append([uuid, name])
    return sorted(out)


def neuronic_allocation_gate(ngpu, env):
    """Slurm allocation == the rung: one node, one task, ``ngpu`` distinct visible L40s as seen by
    CUDA_VISIBLE_DEVICES, SLURM_GPUS_ON_NODE, nvidia-smi (cgroup) and torch in the child's env."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise Refused("not inside a Slurm allocation")
    if os.environ.get("SLURM_JOB_NUM_NODES", "1") != "1" or os.environ.get("SLURM_NTASKS", "1") != "1":
        raise Refused("the allocation must be one node / one task")
    cvd = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]
    if len(cvd) != ngpu or len(set(cvd)) != ngpu:
        raise Refused(f"CUDA_VISIBLE_DEVICES {cvd} is not {ngpu} distinct devices")
    if os.environ.get("SLURM_GPUS_ON_NODE") not in (None, str(ngpu)):
        raise Refused(f"SLURM_GPUS_ON_NODE {os.environ.get('SLURM_GPUS_ON_NODE')} != {ngpu}")
    code = ("import json, torch; n = torch.cuda.device_count(); "
            "print(json.dumps([torch.cuda.get_device_name(i) for i in range(n)]))")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    names = json.loads(out.stdout.strip().split("\n")[-1]) if out.returncode == 0 else None
    if not names or len(names) != ngpu or any("L40" not in n for n in names):
        raise Refused(f"torch sees {names}, not {ngpu} x L40")
    rows = reg._gpu_rows()
    return reg.gpu_identity("neuronic", ngpu, [[r[1], r[2]] for r in rows])


LOCK_HOLDER = r"""
import os, sys, time
pg = int(sys.argv[1])

def alive():
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                st = open("/proc/%s/stat" % d).read()
            except OSError:
                continue
            if int(st[st.rindex(")") + 2:].split()[2]) == pg:
                return True
    return False

while alive():
    time.sleep(2)
"""


def start_lock_holder(pgid, fds):
    """A detached process holding the card-lock fds until NO process of group ``pgid`` remains (the
    trainer, every rank Lightning's launcher spawns -- a plain Popen that does not inherit fds -- and
    every data worker share that group). Survives a killed supervisor."""
    if not fds:
        return None
    return subprocess.Popen([sys.executable, "-I", "-c", LOCK_HOLDER, str(pgid)], pass_fds=tuple(fds),
                            start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def lock_cards(cards, lockdir):
    fds = []
    for c in cards:
        path = os.path.join(lockdir, f".exp38_card{c}.lock")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            for f in fds:
                os.close(f)
            raise Refused(f"another exp_38 training / evaluation holds card {c} ({path})")
        fds.append(fd)
    return fds


def campaign_gates(a, platform):
    """HEAD / clean tree / untracked / environment receipt / closure."""
    if platform == "a6000" and os.path.realpath(reg.REPO) != reg.CAMPAIGN_ROOT["a6000"]:
        raise Refused(f"{a.mode} runs only from the campaign tree {reg.CAMPAIGN_ROOT['a6000']}")
    head = git("rev-parse", "HEAD")
    if not a.expect_sha or a.expect_sha != head:
        raise Refused(f"EXPECT_SHA ({a.expect_sha or 'unset'}) != HEAD {head}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise Refused("tracked tree is not clean")
    probs = reg.check_untracked(platform)
    if probs:
        raise Refused("campaign tree: " + "; ".join(probs))
    # the first SMOKE(s) create the neuronic environment receipt -- serialised under its lock, since the sheet submits
    # the three SMOKEs together (Codex code r1 finding 14); every other launch only checks it
    env = reg.ensure_env_receipt(platform) if platform == "neuronic" and a.mode == "SMOKE" else reg.env_receipt(platform)
    return head, env["sha256"], reg.closure_digest()["closure_sha256"]


def common_gates(a, platform):
    arm, tag, rung = reg.parse_run_id(a.run)
    reg.check_rung(platform, rung, a.allow_nonmatched_rung)
    exe = interpreter_gate(platform, a.expect_python)
    d = reg.describe_run(a.run, platform, save_root=a.dry_save_root, allow_nonmatched=a.allow_nonmatched_rung)
    import exp38_subset_audit as sa
    probs = sa.verify(reg._abs(reg.SUBSET_AUDIT), "AcousticRooms", reg.REPO)
    if probs:
        raise Refused("subset audit --verify: " + "; ".join(probs))
    pins = {**reg.vae_check(), **reg.vit_pin_check()}
    return arm, tag, rung, exe, d, pins


# ---------------------------------------------------------------- the FULL loss record (plan r3 §3.2)
FULL_LOGGER_ENTRIES = ["version_0", "version_0/hparams.yaml", "version_0/metrics.csv"]


def move_loss_record(rd):
    """FULL only, after the child exits and BEFORE the end audit (Codex plan r2 finding 2): <rd>/lightning_logs must
    hold exactly version_0/{hparams.yaml, metrics.csv}; the CSV is validated, both files move into receipts/ as
    loss_record.csv / loss_record_hparams.yaml and the emptied folders are removed. Anything else (version_1, an
    extra file, a missing CSV) raises -- an end-audit failure; nothing moves then."""
    ll = os.path.join(rd, reg.LOGGER_DIR)
    got = reg.logger_entries(rd) if os.path.isdir(ll) and not os.path.islink(ll) else None
    if got != FULL_LOGGER_ENTRIES:
        raise reg.GateError(f"{ll}: entries {got} != exactly {FULL_LOGGER_ENTRIES}")
    v0 = os.path.join(ll, "version_0")
    moves = {reg.LOSS_RECORD: os.path.join(v0, "metrics.csv"), reg.LOSS_RECORD_HPARAMS: os.path.join(v0, "hparams.yaml")}
    if any(os.path.islink(s) or not os.path.isfile(s) for s in moves.values()):
        raise reg.GateError(f"{v0}: the logger files must be regular files")
    out = reg.check_loss_record(moves[reg.LOSS_RECORD])
    rdir = os.path.join(rd, reg.RECEIPTS_DIR)
    os.makedirs(rdir, exist_ok=True)
    for name in moves:
        if os.path.lexists(os.path.join(rdir, name)):
            raise reg.GateError(f"{rdir}/{name} exists")
    for name, src in moves.items():
        os.rename(src, os.path.join(rdir, name))
    os.rmdir(v0)
    os.rmdir(ll)
    reg._fsync_dir(rdir)
    return out


def f100_delta_reverification(rd, rec):
    """f100 FULL only, after the child and the f075 post-run verification, BEFORE the end audit, the training receipt
    and the commit marker (plan r3 §3.2 "Data binding", Codex plan r2 finding 2): the 145,494 delta files are re-read
    SEQUENTIALLY in the job (the GPUs idle meanwhile) against the pinned delta manifest. The sealed record goes to
    receipts/ whatever its verdict and the launch record names it; a FAIL raises (an end-audit failure: unsealed); an
    exception or a stop (the signal check runs between files) propagates: no record, no receipt, the run lock stays."""
    import exp46_content_manifest as cm46
    vrec = cm46.verify_record(reg._abs(reg.F100_DELTA_MANIFEST), "AcousticRooms", reg.REPO,
                              want_sha=reg.F100_DELTA_SHA256, check=check_signals)
    path = os.path.join(rd, reg.RECEIPTS_DIR, reg.F100_VERIFICATION)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    reg.write_sealed(path, vrec)
    rec.update(f100_delta_verification={"verdict": vrec["verdict"], "record": path, "sha256": reg.sha256_file(path)})
    if vrec["verdict"] != "PASS":
        raise reg.GateError("f100 delta content verification FAILED: " + "; ".join(vrec["problems"][:3]))
    return path


# ---------------------------------------------------------------- modes
def ts_now():
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def launch_cmd(platform, ngpu, argv):
    if platform == "neuronic":
        return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                f"--nproc_per_node={ngpu}", *argv]
    return [sys.executable, *argv]


def mode_dry_run(a, platform):
    arm, tag, rung, exe, d, pins = common_gates(a, platform)
    ngpu = reg.RUNGS[rung]["ngpu"]
    env = child_env(platform, a.backend, reg.REPO, "0,1" if platform == "a6000" else None)
    print(f"pins: {json.dumps({**pins, **backend_gate(a.backend, reg.REPO, env)})}")
    print(f"closure: {json.dumps(reg.closure_digest())}")
    print(f"run dir {d['argv'][d['argv'].index('--save-dir') + 1]}: {d['run_dir_state']}")
    full = launch_cmd(platform, ngpu, reg.build_train_argv(a.run, platform, a.dry_save_root, a.allow_nonmatched_rung))
    # the isolated --dry-save-root, when given, for the SMOKE dir too (Codex code r1 finding 10): the dry run then
    # names and checks nothing of the live save root
    print("FULL (cwd = the closure snapshot): " + " ".join(full))
    if platform == "neuronic":
        smoke_dir = os.path.join(a.dry_save_root or reg.PLATFORMS[platform]["save_root"], f"{a.run}_smoke")
        smoke = launch_cmd(platform, ngpu, reg.build_smoke_argv(a.run, reg.SMOKE_STEPS_DDP, smoke_dir, True, platform))
        print("SMOKE (cwd = the closure snapshot): " + " ".join(smoke))
    else:                            # exp_46 plan r3 §3.2: no A6000 training path (never outputs_FLAC/exp38_smoke)
        print("SMOKE_DDP: refused (exp_46 has no A6000 training path; SMOKE runs on neuronic only)")
    head = git("rev-parse", "HEAD")
    checks = [("E0 receipt (FULL needs it)", lambda: reg.check_e0_receipt(platform)),
              ("SMOKE_DDP receipt (FULL needs it)", lambda: reg.check_smoke_receipt(a.run, platform, a.backend, head)),
              ("environment receipt", lambda: reg.env_receipt(platform))]
    if tag == reg.FULL_TAG:
        checks.append(("f100 delta content manifest (SMOKE and FULL need it)", reg.check_f100_delta_pinned))
    for label, fn in checks:
        try:
            print(f"{label}: {json.dumps(fn())}")
        except (reg.GateError, OSError, ValueError, KeyError) as err:
            print(f"{label}: would refuse now ({err})")
    if platform == "a6000" and os.path.realpath(reg.REPO) != reg.CAMPAIGN_ROOT["a6000"]:
        print(f"note: not the campaign tree {reg.CAMPAIGN_ROOT['a6000']} (SMOKE_DDP / FULL refuse here)")
    print(f"DRY_RUN OK {a.run} on {platform} (nothing spawned)")
    return 0


def mode_smoke_single(a):
    """exp_38's a6000 one-rank preliminary smoke is NOT an exp_46 launch path (Codex code r1 finding 11): plan r3 §5 L5
    runs every SMOKE on 2 x L40 at 2 x 32, so this mode refuses before any gate, card lock or directory."""
    raise Refused("the single-rank SMOKE: " + reg.WORLD_SIZE_REFUSAL.format(n=1, what="SMOKE") +
                  "; use the production-topology SMOKE (neuronic MODE=SMOKE through exp46_submit_neuronic.py)")


def mode_campaign(a, platform):
    """SMOKE / FULL on neuronic (exp_46 has no A6000 training path: SMOKE_DDP, and SMOKE on the A6000 box, refuse
    first)."""
    if a.mode == "SMOKE_DDP" or (a.mode == "SMOKE" and platform == "a6000"):
        # exp_46 plan r3 §3.2 (Names / paths; Codex plan r2 finding 1): the inherited A6000 SMOKE destination
        # outputs_FLAC/exp38_smoke is refused -- before any gate, claim, lock, probe or directory (fix round 1, with
        # Codex code r1 finding 12: an A6000 SMOKE used to run every gate, nvidia-smi included, before refusing)
        raise Refused(f"exp_46 has no A6000 training path: {a.mode} on {platform} is refused (the inherited SMOKE "
                      "destination outputs_FLAC/exp38_smoke is never written); SMOKE runs on neuronic only")
    arm, tag, rung, exe, d, pins = common_gates(a, platform)
    if tag == reg.FULL_TAG:          # f100 (plan r3 §3.2): its delta manifest pinned + present before ANY work, SMOKE too
        reg.check_f100_delta_pinned()
    # the save root (owner claim, run dir, ledger, probe all live under it) is isolated from exp_38's roots before any
    # mutation (Codex code r2 finding 6)
    reg.check_isolation(reg.PLATFORMS[platform]["save_root"], f"the {platform} save root")
    if platform == "a6000":          # the snapshot namespace too, before any directory (plan r3 §3.2 "Isolation")
        reg.check_isolation(SNAPSHOT_ROOT["a6000"], "the A6000 snapshot root")
    ngpu = reg.RUNGS[rung]["ngpu"]
    full = a.mode == "FULL"
    if full and platform != reg.TRAINING_PLATFORM:
        raise Refused(f"D1/D2: every exp_46 training runs on {reg.TRAINING_PLATFORM}; FULL refuses on {platform}")
    head, env_sha, closure = campaign_gates(a, platform)
    e0 = smoke = owner = None
    if platform == "neuronic":
        import importlib.metadata as md
        try:
            fv = md.version("flash-attn")
        except md.PackageNotFoundError:
            fv = ""
        if a.backend == "flash" and (not a.expect_flash_version or fv != a.expect_flash_version):
            raise Refused(f"flash-attn {fv or 'absent'} != EXPECT_FLASH_VERSION {a.expect_flash_version or 'unset'}")
        import exp46_submit_neuronic as sub
        owner = sub.claim_attempt(a.run, a.mode, a.token, head, a.backend, a.save_dir, a.expect_flash_version,
                                  a.allow_nonmatched_rung)
        _ATTEMPT["path"] = owner                 # main() finishes it on any later exit
    ts = ts_now()
    fds = []
    if platform == "a6000":
        fds = lock_cards([0, 1], a.lockdir)
        gpus = a6000_gpu_gate([0, 1], a.gpu_policy, reg.ARMS[arm]["min_free_mib"])
        cuda = "0,1"
    else:
        gpus = neuronic_allocation_gate(ngpu, child_env(platform, a.backend, reg.REPO))
        cuda = None
    if full:
        e0 = reg.check_e0_receipt(platform)
        if platform == "a6000" and e0["head"] != head:      # one frozen campaign tree on this box
            raise Refused(f"HEAD {head} != the E0 campaign's {e0['head']}")
        smoke = reg.check_smoke_receipt(a.run, platform, a.backend, head, gpus if platform == "a6000" else None)
        # the BYTES of every AR file the run can read (Codex code r2 BLOCKING 3); minutes over CIFS
        if reg.AR_CONTENT_SHA256 is None:
            raise Refused("the AR content manifest is not pinned in the registry")
        import exp38_content_manifest as cm
        cprobs = cm.verify(reg._abs(reg.AR_CONTENT_MANIFEST), "AcousticRooms", reg.REPO, want_sha=reg.AR_CONTENT_SHA256)
        if cprobs:
            raise Refused("AR content: " + "; ".join(cprobs))
    # ---- run dir + storage
    if full:
        rd = reg.save_dir(platform, a.run)
        if platform == "neuronic" and a.save_dir != rd:
            raise Refused(f"SAVE_DIR {a.save_dir} != the registry's run dir {rd}")
        root = os.path.dirname(rd)
        # the filesystem type + write probe (exp_38), then the shared reservation re-check under the ledger lock
        # (plan r3 §3.2): the submitter reserved this run before sbatch; no extra candidate term here
        storage = dict(reg.storage_gate(platform, root, 0), reservation=reg.recheck_storage(a.run, a.token, root))
    elif platform == "neuronic":
        rd = os.path.join(reg.PLATFORMS[platform]["save_root"], f"{a.run}_smoke")
        if a.save_dir != rd:
            raise Refused(f"SAVE_DIR {a.save_dir} != the registry's smoke dir {rd}")
        storage = reg.storage_gate(platform, os.path.dirname(rd), 2**30)
    else:                            # unreachable (refused first above): no A6000 SMOKE destination exists in exp_46
        raise Refused(f"exp_46 has no A6000 training path: no {a.mode} run dir on {platform}")
    reg.check_run_dir(rd, a.run, 0)
    os.makedirs(rd, exist_ok=True)
    try:
        os.mkdir(os.path.join(rd, ".exp46_lock"))
    except FileExistsError:
        raise Refused(f"run lock {rd}/.exp46_lock exists (a live or crashed run: archive it first)")
    reg.atomic_write_bytes(os.path.join(rd, ".exp46_lock", "holder"), json.dumps(
        {"pid": os.getpid(), "host": os.uname().nodename, "ts": ts, "head": head,
         "job": os.environ.get("SLURM_JOB_ID")}).encode())
    retained = None
    if full:                         # the exact receipt bytes this run is admitted under (Codex code r2 HIGH 6)
        retained = reg.retain_receipts(rd, {"e0": reg._abs(reg.E0_RECEIPT), "smoke": smoke["smoke_receipt"],
                                            "env": reg._abs(reg.env_receipt_rel(platform))},
                                       plain={"smoke_log": json.load(open(smoke["smoke_receipt"]))["log"]})
    # ---- snapshot (launch root)
    snap = (os.path.join(SNAPSHOT_ROOT["a6000"], f"{a.run}_{a.mode}_{ts}") if platform == "a6000"
            else os.path.join(rd, ".exp46_tools"))
    reg.check_isolation(snap, "the training snapshot")          # before its directory (plan r3 §3.2 "Isolation")
    os.makedirs(os.path.dirname(snap), exist_ok=True)
    snapshot = reg.snapshot_tools(a.run, snap, reg.REPO)
    if reg.closure_digest(snap)["closure_sha256"] != closure:
        raise Refused("the snapshot's closure differs from the campaign tree's", EXIT_ENV)
    env = child_env(platform, a.backend, snap, cuda)
    pins.update(backend_gate(a.backend, snap, env))
    argv = (reg.build_train_argv(a.run, platform, allow_nonmatched=a.allow_nonmatched_rung) if full
            else reg.build_smoke_argv(a.run, reg.SMOKE_STEPS_DDP, rd, True, platform))
    cmd = launch_cmd(platform, ngpu, argv)
    label = a.mode if platform == "a6000" or full else "SMOKE"
    log = reg._abs(f"{reg.EXP_DIR}/c4flip_data_curve_{ts}_{a.run}_{platform}_{label}_train.log")
    rec_paths = [reg._abs(f"{reg.EXP_DIR}/at_launch_{a.run}_{platform}_{label}_{ts}.json"),
                 os.path.join(rd, f"at_launch_{a.run}_{platform}_{label}_{ts}.json")]
    rec = Record(rec_paths, run=a.run, mode=a.mode, platform=platform, backend=a.backend, ts=ts,
                 launched_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host=os.uname().nodename, head=head,
                 closure_sha256=closure, env_receipt_sha256=env_sha, python=exe, pins=pins, describe=
                 {k: v for k, v in d.items() if k != "argv"}, e0=e0, smoke=smoke, owner=owner, gpus=gpus,
                 driver=sorted({r[3] for r in reg._gpu_rows()}),
                 gpu_policy=a.gpu_policy, storage=storage, run_dir=rd, snapshot=snapshot, cwd=snap, cmd=cmd,
                 cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"), slurm={k: v for k, v in os.environ.items()
                                                                              if k.startswith("SLURM_")},
                 supervisor_sha256=reg.sha256_file(os.path.realpath(__file__)), supervisor_pid=os.getpid(),
                 log=log, status="launching")
    print(json.dumps(rec.data, indent=1, default=str), flush=True)
    owned = {}

    def launch_and_audit():
        child = owned["child"] = OwnedChild(cmd, snap, env, log, pass_fds=fds)
        holder = start_lock_holder(child.pgid, fds)
        rec.update(pid=child.pid, pgid=child.pgid, start_time=child.start, status="running",
                   lock_holder_pid=holder.pid if holder else None)

        def start_check():
            text = open(log, errors="replace").read()
            if reg.first_step_index(reg.log_lines(text)) is None:
                return None
            return reg.log_gate(text, a.run, "start", a.backend, ngpu)

        def evidence_check():
            names = os.listdir(os.path.join(rd, "evidence")) if os.path.isdir(os.path.join(rd, "evidence")) else []
            if len([n for n in names if n.endswith(".json") and ".rank" in n]) < ngpu:
                return None
            return reg.evidence_gate(os.path.join(rd, "evidence"), a.run, ngpu, a.backend)

        rc = supervise(child, rec, start_check if full else None, evidence_check=evidence_check if full else None,
                       grace=a.grace)
        if rc != 0:
            rec.update(strict=True, status=rec.data.get("status", "running") if rc in (EXIT_START,) or rc >= 128
                       else f"train exited {rc}", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            print(f"EXP46 {a.run} {a.mode} FAILED rc={rc}; the run lock stays as the crash marker", flush=True)
            return rc
        # ---- end audits, all from the snapshot
        if snap not in sys.path:
            sys.path.insert(0, snap)
        try:
            reg.verify_snapshot(snap, snapshot["snapshot_sha256"])
            if full:
                rec.update(loss_record=move_loss_record(rd))      # FULL only; the SMOKE keeps its logger dir
                # the AR bytes again AFTER the run (Codex code r3 BLOCKING 1, defense in depth): a file changed
                # during training refuses the receipt (a change-and-restore inside the run is not detectable
                # without an immutable copy -- decision D9)
                cprobs = cm.verify(reg._abs(reg.AR_CONTENT_MANIFEST), "AcousticRooms", reg.REPO,
                                   want_sha=reg.AR_CONTENT_SHA256)
                if cprobs:
                    raise reg.GateError("AR content changed during the run: " + "; ".join(cprobs))
                delta = f100_delta_reverification(rd, rec) if tag == reg.FULL_TAG else None
                # Codex code r1 finding 3: the verifier checks signals BEFORE each file; a TERM / INT / HUP that arrived
                # during the last read, the grouping or the audits above is only pending here -- no receipt under it
                check_signals()
                out = reg.write_training_receipt(a.run, platform, a.backend, head, rd, log, rc, snapshot["snapshot_sha256"],
                                                 e0["sha256"], smoke, gpus, rec.data["launched_at"], env_sha,
                                                 retained, reg.AR_CONTENT_SHA256, repo=snap, tree=reg.REPO,
                                                 delta_verification=delta)
                verdict = "COMPLETE"
                rec.update(training_receipt=os.path.join(rd, reg.TRAINING_RECEIPT), final_ckpt_sha256=out["final_ckpt_sha256"])
            else:
                out = reg.write_smoke_receipt(a.run, platform, a.backend, head, rd, log, rc, gpus, snapshot["snapshot_sha256"],
                                              rec.data["launched_at"], ts, env_sha, repo=snap, out_repo=reg.REPO,
                                              snapshot_dir=snap)
                verdict = out["verdict"]
                rec.update(smoke_receipt=out)
        except (reg.GateError, OSError, ValueError, KeyError) as err:
            rec.update(strict=True, status=f"end audit failed: {err}", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            print(f"EXP46 {a.run} {a.mode} END AUDIT FAILED: {err}", flush=True)
            return EXIT_END
        if full:
            # THE COMMIT BOUNDARY of a FULL (Codex code r1 finding 3): the last signal check. A signal pending up to here
            # (one that arrived while the training receipt was written included) leaves the run UNSEALED -- the run
            # lock stays, no record seal, no commit marker; past this line the run commits and a later signal only
            # waits in _PENDING (the job then ends with the run sealed)
            check_signals()
        os.rename(os.path.join(rd, ".exp46_lock"), os.path.join(rd, f".exp46_lock.released_{ts}"))
        if full:
            tr_sha = reg.sha256_file(os.path.join(rd, reg.TRAINING_RECEIPT))
            rec.update(strict=True, status=verdict, training_receipt_sha256=tr_sha,
                       finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            if open(rec.paths[1], "rb").read() != open(rec.paths[0], "rb").read():
                raise OSError("the run-dir launch record copy was not written")
            # frozen BEFORE the marker is published (Codex code r2 finding 5): from the moment a publication may
            # have happened -- even if write_commit_marker raises after its rename -- nothing rewrites the record
            rec.seal()
            reg.write_commit_marker(rd, a.run, tr_sha, rec.paths[1])     # the single final commit
            try:              # sealed: its storage reservation ends -- reported as the LEDGER's own release entry
                released = reg.release_storage(a.run, a.token, os.path.dirname(rd), reason="sealed")
                print(f"storage reservation of {a.run}: " + (f"released {json.dumps(released, sort_keys=True)}"
                                                             if released else "none held"), flush=True)
            except (reg.GateError, OSError) as err:                      # the run is sealed; the hold stays
                print(f"storage reservation of {a.run} not released: {err}", flush=True)
        else:
            rec.update(strict=True, status=verdict, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        print(f"EXP46 {a.run} {a.mode} {verdict}", flush=True)
        return 0 if verdict in ("COMPLETE", "PASS") else EXIT_END


    code = EXIT_ENV
    try:
        code = launch_and_audit()
        return code
    except BaseException as err:                 # stop OUR group first, record (strict), then let main() map it
        child = owned.get("child")
        if child is not None and child.rc is None:
            child.terminate(a.grace)
        rec.update(strict=True, status=f"supervisor error: {err!r}", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        raise
    finally:
        if owner is not None:
            sub.finish_attempt(owner, code)
            _ATTEMPT["done"] = True

def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--platform", required=True, choices=sorted(reg.PLATFORMS))
    p.add_argument("--mode", required=True, choices=("DRY_RUN", "SMOKE", "SMOKE_DDP", "FULL"))
    p.add_argument("--run", required=True)
    p.add_argument("--backend", default="flash", choices=("flash", "sdpa"))
    p.add_argument("--expect-sha")
    p.add_argument("--gpu-policy", default="idle", choices=("idle", "shared"))
    p.add_argument("--smoke-gpu", type=int, default=0, choices=(0, 1))
    p.add_argument("--smoke-steps", type=int, default=20)
    p.add_argument("--allow-nonmatched-rung", action="store_true")
    p.add_argument("--token")
    p.add_argument("--save-dir")
    p.add_argument("--expect-python")
    p.add_argument("--expect-flash-version", default="")
    p.add_argument("--grace", type=float, default=GRACE_S)
    p.add_argument("--dry-save-root", help="DRY_RUN-only test hook")
    p.add_argument("--lockdir", default=LOCKDIR)
    a = p.parse_args(argv)
    if a.mode != "DRY_RUN" and (a.dry_save_root or a.lockdir != LOCKDIR):
        p.error("--dry-save-root / --lockdir are DRY_RUN-only test hooks")
    if not 1 <= a.smoke_steps <= 99:
        p.error("--smoke-steps must be 1..99")
    if a.mode != "DRY_RUN" and a.backend != reg.TRAINING_BACKEND:      # Codex code r2 finding 12
        p.error(f"exp_46 trains with backend {reg.TRAINING_BACKEND} only (plan r3 F2, L5): --backend {a.backend} "
                "is refused for SMOKE / SMOKE_DDP / FULL")
    if a.platform == "neuronic" and a.mode == "SMOKE_DDP":
        p.error("neuronic's production-topology smoke is MODE=SMOKE")
    if a.platform == "neuronic" and a.mode in ("SMOKE", "FULL") and not (a.token and re.fullmatch(r"[0-9a-f]{32}", a.token) and a.save_dir):
        p.error("a neuronic job needs its 32-hex --token and --save-dir (exp46_submit_neuronic.py sets them)")
    return a


def main(argv=None):
    a = parse(argv)
    install_signal_handlers()
    code = EXIT_ENV
    try:
        code = _dispatch(a)
        return code
    finally:
        if _ATTEMPT["path"] and not _ATTEMPT["done"]:
            import exp46_submit_neuronic as sub
            sub.finish_attempt(_ATTEMPT["path"], code)


def _dispatch(a):
    try:
        if a.mode == "DRY_RUN":
            return mode_dry_run(a, a.platform)
        if a.mode == "SMOKE" and a.platform == "a6000":
            return mode_smoke_single(a)
        return mode_campaign(a, a.platform)
    except Refused as err:
        print(f"EXP46 LAUNCH REFUSED: {err}", file=sys.stderr, flush=True)
        return err.code
    except Stopped as st:
        print(f"EXP46 stopped by signal {st.signum} before any child started", file=sys.stderr, flush=True)
        return 128 + st.signum
    except (reg.GateError, KeyError, ValueError) as err:
        print(f"EXP46 LAUNCH REFUSED: {err}", file=sys.stderr, flush=True)
        return EXIT_GATE
    except (OSError, subprocess.SubprocessError) as err:
        print(f"EXP46 LAUNCH ERROR: {err}", file=sys.stderr, flush=True)
        return EXIT_ENV


if __name__ == "__main__":
    sys.exit(main())
