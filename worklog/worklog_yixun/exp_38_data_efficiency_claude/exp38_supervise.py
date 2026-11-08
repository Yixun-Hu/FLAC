#!/usr/bin/env python3
"""exp_38 training supervisor — the one process that owns a training launch on either platform
(plan_data_efficiency.md r2 §10 B6 / B7; Codex code r1 HIGH 4, 5, 7).

Called by exp38_train_local.sh (this 2×A6000 box) and exp38_train.sbatch (neuronic) after they
sanitised the environment; it runs every gate, holds the locks, launches train.py FROM an immutable
closure snapshot, owns the child's process group, gates the first step, and publishes the run's
sealed receipt only after every end audit passes:

    MODE=DRY_RUN    every CPU-side gate + the exact commands; spawns nothing
    MODE=SMOKE      a6000: one rank on --smoke-gpu, preliminary (no DDP / SyncBN, no receipt)
                    neuronic: the production-topology 30-step smoke -> sealed SMOKE_DDP receipt
    MODE=SMOKE_DDP  a6000: the production topology (2 × 32 + SyncBN), 30 steps -> sealed receipt
    MODE=FULL       the monolithic 0 -> 40,000 run -> sealed TRAINING_COMPLETE.json in the run dir

Process ownership: the child is started in its own session (pgid == its pid) with the card locks
passed down (``pass_fds``), so the locks live as long as ANY process of the run lives -- a killed
supervisor cannot release a card under a running trainer. TERM / INT / HUP are forwarded to that
group only while the leader is unreaped (its pid -- and so the group id -- cannot be reused) and
its /proc start time is the one recorded at spawn; after GRACE_S the group gets SIGKILL. Every exit
path rewrites the launch record atomically; a failed run keeps its ``.exp38_lock`` crash marker.

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

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import exp38_registry as reg  # noqa: E402

EXIT_GATE, EXIT_ENV, EXIT_START, EXIT_END = 2, 3, 4, 5
GRACE_S = 120
START_TIMEOUT_S = 5400
EVIDENCE_TIMEOUT_S = 1800
SNAPSHOT_ROOT = {"a6000": "/home/yixunhu/codespace/exp-38-snapshots"}
LOCKDIR = "/home/yixunhu/codespace"              # .exp38_card<i>.lock, shared with exp38_eval.py
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
                record.update(status=f"{name} gate failed; our process group terminated", child_rc=rc)
                return EXIT_START
        while not child.exited():
            sleep_checked(poll)
        rc = child.reap()
        record.update(child_rc=rc, child_exited_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        return 128 - rc if rc < 0 else rc
    except Stopped as st:
        rc = child.terminate(grace)
        record.update(status=f"terminated by signal {st.signum}; our process group stopped", child_rc=rc,
                      finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        return 128 + st.signum


class Record:
    """The launch record, rewritten atomically (JSON) at every change, at every listed path."""

    def __init__(self, paths, **fields):
        self.paths, self.data = list(paths), dict(fields)
        self.flush()

    def add_path(self, path):
        self.paths.append(path)
        self.flush()

    def update(self, **fields):
        self.data.update(fields)
        self.flush()

    def flush(self):
        text = (json.dumps(self.data, indent=1, sort_keys=True, default=str) + "\n").encode()
        for p in self.paths:
            try:
                reg.atomic_write_bytes(p, text)
            except OSError as err:              # a record copy must never kill a running supervisor
                print(f"record write {p} failed: {err}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- gates
def git(*args, repo=None):
    return subprocess.run(["git", "-C", repo or reg.REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def child_env(platform, backend, launch_root, cuda_devices=None):
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE")}
    if platform == "neuronic":           # PL 2.1 must elect TorchElastic, not SLURM (recorded first)
        for k in SLURM_RANK_VARS:
            env.pop(k, None)
    env.update({"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1"})
    if backend == "sdpa":
        env.update({"PYTHONPATH": os.path.join(launch_root, "overlays", "no_flash"), "PYTHONDONTWRITEBYTECODE": "1"})
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    return env


def backend_gate(backend, launch_root, env):
    """Resolve ``flash_attn`` exactly as the child will (same interpreter, cwd and env)."""
    code = ("import importlib.util, os; s = importlib.util.find_spec('flash_attn'); "
            "print(os.path.realpath(s.origin) if s and s.origin else '')")
    out = subprocess.run([sys.executable, "-c", code], cwd=launch_root, env=env, capture_output=True, text=True, timeout=300)
    if out.returncode:
        raise Refused(f"backend probe failed: {out.stderr.strip()[-400:]}", EXIT_ENV)
    origin = out.stdout.strip()
    overlay = os.path.realpath(os.path.join(launch_root, "overlays", "no_flash", "flash_attn.py"))
    if backend == "flash" and (not origin or origin == overlay):
        raise Refused("ATTN_BACKEND=flash but the child cannot import a real flash_attn")
    if backend == "sdpa" and origin not in ("", overlay):
        raise Refused(f"ATTN_BACKEND=sdpa but the child would import {origin}")
    return {"attention_backend": backend, "flash_attn": origin or "absent"}


def interpreter_gate(platform, expect_python=None):
    exe = os.path.realpath(sys.executable)
    if platform == "a6000" and not A6000_PY.match(exe):
        raise Refused(f"python resolves to {exe}, not the flac env", EXIT_ENV)
    if platform == "neuronic" and (not expect_python or exe != os.path.realpath(expect_python)):
        raise Refused(f"python is {exe}, not {expect_python}", EXIT_ENV)
    probs = reg.env_problems()
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
    if platform == "neuronic" and a.mode == "SMOKE" and not os.path.exists(reg._abs(reg.env_receipt_rel(platform))):
        reg.env_receipt(platform, write=True)
    env = reg.env_receipt(platform)
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
    smoke_dir = (os.path.join(reg.PLATFORMS[platform]["save_root"], f"{a.run}_smoke") if platform == "neuronic"
                 else f"outputs_FLAC/exp38_smoke/{a.run}_ddp_<ts>")
    smoke = launch_cmd(platform, ngpu, reg.build_smoke_argv(a.run, reg.SMOKE_STEPS_DDP, smoke_dir, True, platform))
    print("FULL (cwd = the closure snapshot): " + " ".join(full))
    print(f"{'SMOKE' if platform == 'neuronic' else 'SMOKE_DDP'} (cwd = the closure snapshot): " + " ".join(smoke))
    if platform == "a6000" and rung == "2x32":
        print("SMOKE (one rank, preliminary): " + " ".join(launch_cmd(platform, 1, reg.build_smoke_argv(
            a.run, a.smoke_steps, f"outputs_FLAC/exp38_smoke/{a.run}_<ts>"))))
    head = git("rev-parse", "HEAD")
    for label, fn in (("E0 receipt (FULL needs it)", lambda: reg.check_e0_receipt(platform)),
                      ("SMOKE_DDP receipt (FULL needs it)", lambda: reg.check_smoke_receipt(a.run, platform, a.backend, head)),
                      ("environment receipt", lambda: reg.env_receipt(platform))):
        try:
            print(f"{label}: {json.dumps(fn())}")
        except (reg.GateError, OSError, ValueError, KeyError) as err:
            print(f"{label}: would refuse now ({err})")
    if platform == "a6000" and os.path.realpath(reg.REPO) != reg.CAMPAIGN_ROOT["a6000"]:
        print(f"note: not the campaign tree {reg.CAMPAIGN_ROOT['a6000']} (SMOKE_DDP / FULL refuse here)")
    print(f"DRY_RUN OK {a.run} on {platform} (nothing spawned)")
    return 0


def mode_smoke_single(a):
    """a6000 one-rank preliminary smoke (no DDP / SyncBN; never a gate for FULL)."""
    arm, tag, rung, exe, d, pins = common_gates(a, "a6000")
    if rung != "2x32":
        raise Refused("single-rank smoke: 2x32 runs only")
    fds = lock_cards([a.smoke_gpu], a.lockdir)
    a6000_gpu_gate([a.smoke_gpu], a.gpu_policy, reg.ARMS[arm]["min_free_mib"])
    ts = ts_now()
    sdir = os.path.join(reg.REPO, "outputs_FLAC", "exp38_smoke", f"{a.run}_{ts}")
    os.makedirs(sdir)
    log = reg._abs(f"{reg.EXP_DIR}/data_efficiency_{ts}_{a.run}_SMOKE.log")
    argv = reg.build_smoke_argv(a.run, a.smoke_steps, sdir)
    env = child_env("a6000", a.backend, reg.REPO, str(a.smoke_gpu))
    pins.update(backend_gate(a.backend, reg.REPO, env))
    rec = Record([reg._abs(f"{reg.EXP_DIR}/data_efficiency_{ts}_{a.run}_SMOKE.json")], mode="SMOKE", run=a.run, ts=ts,
                 gpu=a.smoke_gpu, steps=a.smoke_steps, backend=a.backend, head=git("rev-parse", "HEAD"), pins=pins,
                 argv=argv, note="single rank: no DDP / SyncBN; preliminary only", supervisor_pid=os.getpid(),
                 status="running")
    samples = open(os.path.join(sdir, "vram_samples.txt"), "w")
    sampler = subprocess.Popen(["nvidia-smi", "-i", str(a.smoke_gpu), "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits", "-lms", "1000"], stdout=samples,
                               stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    try:
        child = OwnedChild(launch_cmd("a6000", 1, argv), reg.REPO, env, log, pass_fds=fds)
        rec.update(pid=child.pid, pgid=child.pgid, start_time=child.start)
        rc = supervise(child, rec)
    finally:
        sampler.terminate()
        try:
            sampler.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sampler.kill()
            sampler.wait()
        samples.close()
    vals = [int(x) for x in open(os.path.join(sdir, "vram_samples.txt")).read().split() if x.isdigit()]
    n_ckpt = len([f for f in os.listdir(sdir) if f.endswith(".ckpt")])
    lg = reg.log_gate(open(log, errors="replace").read(), a.run, "end", a.backend, 1, max_steps=a.smoke_steps, min_losses=1)
    eg = reg.evidence_gate(os.path.join(sdir, "evidence"), a.run, 1, a.backend, a.smoke_steps)
    verdict = "PASS" if rc == 0 and n_ckpt == 0 and not lg and not eg else "FAIL"
    rec.update(rc=rc, checkpoints_written=n_ckpt, log_gate="PASS" if not lg else lg,
               evidence_gate="PASS" if not eg else eg, peak_vram_mib_card=max(vals) if vals else None,
               verdict=verdict, status="finished", log=log)
    print(f"SMOKE VERDICT: {verdict} (log {log})", flush=True)
    return rc if rc >= 128 else (0 if verdict == "PASS" else EXIT_END)


def mode_campaign(a, platform):
    """SMOKE_DDP / FULL on a6000, SMOKE / FULL on neuronic."""
    arm, tag, rung, exe, d, pins = common_gates(a, platform)
    ngpu = reg.RUNGS[rung]["ngpu"]
    full = a.mode == "FULL"
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
        import exp38_submit_neuronic as sub
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
    # ---- run dir + storage
    if full:
        rd = reg.save_dir(platform, a.run)
        if platform == "neuronic" and a.save_dir != rd:
            raise Refused(f"SAVE_DIR {a.save_dir} != the registry's run dir {rd}")
        storage = reg.storage_gate(platform, os.path.dirname(rd) if os.path.isdir(os.path.dirname(rd))
                                   else os.path.dirname(os.path.dirname(rd)), reg.TRAIN_FREE_GIB * 2**30)
    elif platform == "neuronic":
        rd = os.path.join(reg.PLATFORMS[platform]["save_root"], f"{a.run}_smoke")
        if a.save_dir != rd:
            raise Refused(f"SAVE_DIR {a.save_dir} != the registry's smoke dir {rd}")
        storage = reg.storage_gate(platform, os.path.dirname(rd), 2**30)
    else:
        rd = os.path.join(reg.REPO, "outputs_FLAC", "exp38_smoke", f"{a.run}_ddp_{ts}")
        storage = None
    reg.check_run_dir(rd, a.run, 0)
    os.makedirs(rd, exist_ok=True)
    try:
        os.mkdir(os.path.join(rd, ".exp38_lock"))
    except FileExistsError:
        raise Refused(f"run lock {rd}/.exp38_lock exists (a live or crashed run: archive it first)")
    reg.atomic_write_bytes(os.path.join(rd, ".exp38_lock", "holder"), json.dumps(
        {"pid": os.getpid(), "host": os.uname().nodename, "ts": ts, "head": head,
         "job": os.environ.get("SLURM_JOB_ID")}).encode())
    # ---- snapshot (launch root)
    snap = (os.path.join(SNAPSHOT_ROOT["a6000"], f"{a.run}_{a.mode}_{ts}") if platform == "a6000"
            else os.path.join(rd, ".exp38_tools"))
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
    log = reg._abs(f"{reg.EXP_DIR}/data_efficiency_{ts}_{a.run}_{platform}_{label}_train.log")
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
    def launch_and_audit():
        child = OwnedChild(cmd, snap, env, log, pass_fds=fds)
        rec.update(pid=child.pid, pgid=child.pgid, start_time=child.start, status="running")

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
            rec.update(status=rec.data.get("status", "running") if rc in (EXIT_START,) or rc >= 128
                       else f"train exited {rc}", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            print(f"EXP38 {a.run} {a.mode} FAILED rc={rc}; the run lock stays as the crash marker", flush=True)
            return rc
        # ---- end audits, all from the snapshot
        if snap not in sys.path:
            sys.path.insert(0, snap)
        try:
            reg.verify_snapshot(snap, snapshot["snapshot_sha256"])
            if full:
                out = reg.write_training_receipt(a.run, platform, a.backend, head, rd, log, rc, snapshot["snapshot_sha256"],
                                                 e0["sha256"], smoke, gpus, rec.data["launched_at"], env_sha,
                                                 repo=snap, tree=reg.REPO)
                verdict = "COMPLETE"
                rec.update(training_receipt=os.path.join(rd, reg.TRAINING_RECEIPT), final_ckpt_sha256=out["final_ckpt_sha256"])
            else:
                out = reg.write_smoke_receipt(a.run, platform, a.backend, head, rd, log, rc, gpus, snapshot["snapshot_sha256"],
                                              rec.data["launched_at"], ts, env_sha, repo=snap, out_repo=reg.REPO)
                verdict = out["verdict"]
                rec.update(smoke_receipt=out)
        except (reg.GateError, OSError, ValueError, KeyError) as err:
            rec.update(status=f"end audit failed: {err}", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            print(f"EXP38 {a.run} {a.mode} END AUDIT FAILED: {err}", flush=True)
            return EXIT_END
        os.rename(os.path.join(rd, ".exp38_lock"), os.path.join(rd, f".exp38_lock.released_{ts}"))
        rec.update(status=verdict, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        print(f"EXP38 {a.run} {a.mode} {verdict}", flush=True)
        return 0 if verdict in ("COMPLETE", "PASS") else EXIT_END


    code = EXIT_ENV
    try:
        code = launch_and_audit()
        return code
    except BaseException as err:                 # record every exit path, then let main() map it
        rec.update(status=f"supervisor error: {err!r}", finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
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
    if a.platform == "neuronic" and a.mode == "SMOKE_DDP":
        p.error("neuronic's production-topology smoke is MODE=SMOKE")
    if a.platform == "neuronic" and a.mode in ("SMOKE", "FULL") and not (a.token and re.fullmatch(r"[0-9a-f]{32}", a.token) and a.save_dir):
        p.error("a neuronic job needs its 32-hex --token and --save-dir (exp38_submit_neuronic.py sets them)")
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
            import exp38_submit_neuronic as sub
            sub.finish_attempt(_ATTEMPT["path"], code)


def _dispatch(a):
    try:
        if a.mode == "DRY_RUN":
            return mode_dry_run(a, a.platform)
        if a.mode == "SMOKE" and a.platform == "a6000":
            return mode_smoke_single(a)
        return mode_campaign(a, a.platform)
    except Refused as err:
        print(f"EXP38 LAUNCH REFUSED: {err}", file=sys.stderr, flush=True)
        return err.code
    except Stopped as st:
        print(f"EXP38 stopped by signal {st.signum} before any child started", file=sys.stderr, flush=True)
        return 128 + st.signum
    except (reg.GateError, KeyError, ValueError) as err:
        print(f"EXP38 LAUNCH REFUSED: {err}", file=sys.stderr, flush=True)
        return EXIT_GATE
    except (OSError, subprocess.SubprocessError) as err:
        print(f"EXP38 LAUNCH ERROR: {err}", file=sys.stderr, flush=True)
        return EXIT_ENV


if __name__ == "__main__":
    sys.exit(main())
