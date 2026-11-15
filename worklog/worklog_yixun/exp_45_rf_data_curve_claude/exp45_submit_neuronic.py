#!/usr/bin/env python3
"""exp_45 neuronic submitter (forked from exp_38's exp38_submit_neuronic.py; plan_rf_data_curve.md r3 §3.2) —
one monolithic job per (run, mode), exactly once (exp_38 plan r2 §10 B7;
Codex code r1 HIGH 6).

Run on the neuronic login node from the exp-38 checkout (HEAD pushed, tracked tree clean):

    exp38_submit_neuronic.py submit    --run exp45_RFf025_l40_2x32 --mode SMOKE --time 01:00:00 [--dry-run]
    exp38_submit_neuronic.py submit    --run exp45_RFf025_l40_2x32 --mode FULL  --time 3-00:00:00 [--dry-run]
    exp38_submit_neuronic.py reconcile --run exp45_RFf025_l40_2x32 --mode FULL      # after an interrupted submit
    exp38_submit_neuronic.py abandon   --run exp45_RFf025_l40_2x32 --mode FULL      # only once the job is proven gone
    exp45_submit_neuronic.py release   --run exp45_RFf025_l40_2x32                  # a SEALED run's stranded hold

Ownership: ONE stable owner record per (run, mode), ``<save root>/.exp45_owners/<run>__<MODE>/``,
created atomically (``mkdir``) with ``intent.json`` (state "submitting") BEFORE ``sbatch``; the job id
is added after ``sbatch`` (state "submitted"). The owner record is separate from the attempt payload:
the job itself writes ``attempt_<jobid>.json`` there (exactly one attempt per owner; ``--no-requeue``),
after exact-comparing its token / run / mode / HEAD / backend / flash version / rung flag / save dir
/ job name / job id with the intent. SMOKE and FULL have fixed run dirs (``<run>_smoke`` / ``<run>``),
so a second SMOKE is refused exactly like a second FULL. ``reconcile`` resolves an interrupted
submission only when exactly one job carries the token (zero or several: refuse, exit 2);
``abandon`` (all token jobs terminal) moves the owner record to ``.exp45_owners/abandoned/`` and the
run dir to ``<dir>.abandoned_<ts>`` -- an abandoned record is never found again.

Crash recovery (Codex code r1 finding 7): submit() holds the (run, mode) submission lock
``<save root>/.exp45_submit_<run>__<MODE>.lock`` from its reservation to the 'submitted' intent, and reconcile /
abandon take the same lock, so a submission in flight is never mistaken for a crash. abandon then recovers from the
PERSISTED state alone: a reservation (or an owner dir) with no intent.json -- the submit died before its intent,
so no sbatch was ever issued: released only when no job carries the reservation's token -- or a reservation whose
owner record is already archived -- abandon died before the release: released only once every job of that token is
terminal. A hold is never freed before its terminal status is established.

Written by the exp_38 Coder (Claude Opus 5.5).
"""
import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import exp45_registry as reg  # noqa: E402

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
            "BOOT_FAIL", "DEADLINE", "REVOKED"}
LIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING", "STOPPED",
        "SIGNALING", "STAGE_OUT", "REQUEUE_HOLD", "REQUEUE_FED", "SPECIAL_EXIT"}
SUBLOG = f"{reg.EXP_DIR}/exp45_neuronic_submissions.log"
OWNERS = ".exp45_owners"
BRANCH = "exp-45-rf-data-curve"            # the launch branch on origin (plan F15, N6)
PARTITION = "all"                           # exp45_train.sbatch's #SBATCH --partition
ATTEMPT_CLAIM = "attempt.claim"
# Codex code r2 finding 7: a killed submitter does not stop its sbatch child, so a 'submitting' intent with no job id
# may still turn into a job; abandon keeps the hold for this long after the intent was written
SUBMIT_GRACE_S = 3600
FLASH_VERSION_DEFAULT = reg.FLASH_VERSION


class SubmitError(reg.GateError):
    pass


def resources(run):
    _, _, rung = reg.parse_run_id(run)
    n = reg.RUNGS[rung]["ngpu"]
    return {"ngpu": n, "cpus": 8 + 7 * n, "mem": "64G" if n == 2 else "108G"}


def save_root(save_root=None):
    return save_root or reg.PLATFORMS["neuronic"]["save_root"]


def run_dir(run, mode, root=None):
    reg.parse_run_id(run)
    if mode not in ("SMOKE", "FULL"):
        raise SubmitError("mode must be SMOKE or FULL")
    return os.path.join(save_root(root), run if mode == "FULL" else f"{run}_smoke")


def owner_dir(run, mode, root=None):
    reg.parse_run_id(run)
    return os.path.join(save_root(root), OWNERS, f"{run}__{mode}")


INTENT_KEYS = frozenset({"schema", "token", "run", "mode", "head", "time", "backend", "flash_version",
                         "allow_nonmatched", "save_dir", "job_name", "state", "ts", "job_id", "reconciled",
                         "partition_max_time"})


def submit_lock_path(run, mode, root=None):
    reg.parse_run_id(run)
    return os.path.join(save_root(root), f".exp45_submit_{run}__{mode}.lock")


@contextlib.contextmanager
def submit_lock(run, mode, root=None):
    """The (run, mode) submission lock (finding 7), never waited for: a holder means a submission is in flight (a
    dead submitter's lock is released by the kernel)."""
    reg.check_isolation(save_root(root), "the exp_45 save root")   # before the lock file (Codex code r2 finding 6)
    fh = open(submit_lock_path(run, mode, root), "a")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SubmitError(f"a submission of {run} {mode} is in progress (its lock is held): retry once it ends")
        yield
    finally:
        fh.close()


def job_name(run, mode, token):
    return f"exp45-{run}-{mode.lower()}-{token[:8]}"


def sbatch_argv(run, mode, token, head, time_limit, backend, flash_version, sd, allow_nonmatched):
    r = resources(run)
    exports = {"RUN": run, "MODE": mode, "EXPECT_SHA": head, "EXP38_TOKEN": token, "ATTN_BACKEND": backend,
               "EXPECT_FLASH_VERSION": flash_version or "", "ALLOW_NONMATCHED_RUNG": "1" if allow_nonmatched else "0",
               "SAVE_DIR": sd}
    return ["sbatch", "--parsable", f"--job-name={job_name(run, mode, token)}", f"--comment=exp45:{token}",
            f"--gres=gpu:l40:{r['ngpu']}", f"--cpus-per-task={r['cpus']}", f"--mem={r['mem']}",
            f"--time={time_limit}", "--no-requeue", f"--chdir={reg.REPO}",
            f"--output={reg.REPO}/{reg.EXP_DIR}/slurm_train_%x_%j.out",
            "--export=ALL," + ",".join(f"{k}={v}" for k, v in exports.items()),
            f"{reg.REPO}/{reg.EXP_DIR}/exp45_train.sbatch"]


def _write_intent(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _git(*args):
    return subprocess.run(["git", "-C", reg.REPO, *args], capture_output=True, text=True, check=True).stdout.strip()


def preflight(run, mode, allow_nonmatched, dry, backend):
    head = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SubmitError("tracked tree is not clean")
    if not dry:
        subprocess.run(["git", "-C", reg.REPO, "fetch", "-q", "origin", BRANCH], check=True)
        if subprocess.run(["git", "-C", reg.REPO, "merge-base", "--is-ancestor", head, "FETCH_HEAD"]).returncode:
            raise SubmitError(f"HEAD {head} is not pushed to origin/{BRANCH}")
    reg.describe_run(run, "neuronic", allow_nonmatched=allow_nonmatched)
    reg.check_e0_receipt("neuronic")
    if mode == "FULL":
        reg.check_smoke_receipt(run, "neuronic", backend, head)
    return head


def _dry_preflight(run, allow_nonmatched):
    """DRY_RUN: the recipe checks only (receipts are produced at campaign time)."""
    reg.describe_run(run, "neuronic", allow_nonmatched=allow_nonmatched)
    return _git("rev-parse", "HEAD")


def slurm_seconds(text):
    """Seconds of a Slurm [D-]H:MM:SS time; None for an unlimited one; SubmitError otherwise."""
    import re
    if text in ("UNLIMITED", "INFINITE"):
        return None
    m = re.fullmatch(r"(?:(\d+)-)?(\d{1,3}):(\d{2}):(\d{2})", text or "")
    if not m or int(m.group(3)) > 59 or int(m.group(4)) > 59:
        raise SubmitError(f"{text!r} is not a Slurm time (e.g. 24:00:00 or 3-00:00:00)")
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s


def partition_max_time(runner=subprocess.run):
    """The partition's MaxTime string (``scontrol show partition all``); any query failure is fatal."""
    out = runner(["scontrol", "show", "partition", PARTITION, "-o"], capture_output=True, text=True)
    import re
    m = re.search(r"(?:^|\s)MaxTime=(\S+)", out.stdout or "") if out.returncode == 0 else None
    if not m:
        raise SubmitError(f"scontrol show partition {PARTITION} gave no MaxTime (rc {out.returncode})")
    slurm_seconds(m.group(1))
    return m.group(1)


def submit(run, mode, time_limit, backend, flash_version, allow_nonmatched, dry, root=None, runner=subprocess.run):
    if mode not in ("SMOKE", "FULL"):
        raise SubmitError("mode must be SMOKE or FULL")
    reg.check_isolation(save_root(root), "the exp_45 save root")     # before ANY mutation (Codex code r2 finding 6)
    want_s = slurm_seconds(time_limit)
    if want_s is None:
        raise SubmitError(f"--time {time_limit!r} is not a Slurm time (e.g. 24:00:00 or 3-00:00:00)")
    # the walltime against the partition's MaxTime, BEFORE any reservation / owner intent (plan r3 §3.2, Codex
    # plan r1 finding 14): refused, never clamped, never chained
    max_time = partition_max_time(runner)
    if slurm_seconds(max_time) is not None and want_s > slurm_seconds(max_time):
        raise SubmitError(f"--time {time_limit} exceeds partition {PARTITION}'s MaxTime {max_time}")
    head = preflight(run, mode, allow_nonmatched, dry, backend) if not dry else _dry_preflight(run, allow_nonmatched)
    token = uuid.uuid4().hex
    sd = run_dir(run, mode, root)
    argv = sbatch_argv(run, mode, token, head, time_limit, backend, flash_version, sd, allow_nonmatched)
    od = owner_dir(run, mode, root)
    if dry:
        return {"dry_run": True, "run_dir": sd, "owner": od, "argv": argv, "partition_max_time": max_time}
    reg.check_run_dir(sd, run, 0)
    os.makedirs(save_root(root), exist_ok=True)
    with submit_lock(run, mode, root):      # held from the reservation to the 'submitted' intent (finding 7)
        return _submit_locked(run, mode, time_limit, backend, flash_version, allow_nonmatched, root, runner, head,
                              token, sd, argv, od, max_time)


def _submit_locked(run, mode, time_limit, backend, flash_version, allow_nonmatched, root, runner, head, token, sd,
                   argv, od, max_time):
    reservation = None
    if mode == "FULL":           # the shared storage reservation, before sbatch and before any owner intent
        reservation = reg.reserve_storage(run, token, sd, save_root(root))
    os.makedirs(os.path.dirname(od), exist_ok=True)
    try:
        os.mkdir(od)
    except FileExistsError:
        if reservation is not None:
            reg.release_storage(run, token, save_root(root), reason="submission refused: the owner record exists")
        raise SubmitError(f"{od} exists: {run} {mode} is already owned (reconcile / abandon)")
    intent = {"schema": 1, "token": token, "run": run, "mode": mode, "head": head, "time": time_limit,
              "backend": backend, "flash_version": flash_version or "", "allow_nonmatched": bool(allow_nonmatched),
              "save_dir": sd, "job_name": job_name(run, mode, token), "state": "submitting",
              "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "job_id": None, "reconciled": False,
              "partition_max_time": max_time}
    _write_intent(os.path.join(od, "intent.json"), intent)
    out = runner(argv, capture_output=True, text=True)
    if out.returncode != 0:
        raise SubmitError(f"sbatch failed ({out.stderr.strip()}); the intent stays 'submitting' -> run reconcile")
    jid = out.stdout.strip().split(";")[0]
    if not jid.isdigit():
        raise SubmitError(f"sbatch returned {out.stdout!r}; run reconcile")
    intent.update(state="submitted", job_id=jid)
    _write_intent(os.path.join(od, "intent.json"), intent)
    with open(os.path.join(reg.REPO, SUBLOG), "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {run} {mode} job {jid} token {token} head {head}: {' '.join(argv)}\n")
    return {"run_dir": sd, "owner": od, "job_id": jid, "token": token, "storage": reservation}


def find_intent(run, mode, root=None):
    """The stable owner record of (run, mode) -- abandoned records live elsewhere and are never found."""
    od = owner_dir(run, mode, root)
    path = os.path.join(od, "intent.json")
    if not os.path.isfile(path):
        raise SubmitError(f"no owner record {path}")
    intent = json.load(open(path))
    if not isinstance(intent, dict) or set(intent) != INTENT_KEYS or intent["run"] != run or intent["mode"] != mode:
        raise SubmitError(f"{path} is not a {run} {mode} intent of the registered schema")
    return od, intent


def claim_attempt(run, mode, token, head, backend, sd, flash_version, allow_nonmatched, root=None):
    """Inside the job: exact-compare the allocation with the owner intent, then record the single
    attempt (refused if any attempt exists). Returns the attempt path."""
    od, intent = find_intent(run, mode, root)
    jid, jname = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURM_JOB_NAME", "")
    want = {"token": token, "run": run, "mode": mode, "head": head, "backend": backend,
            "flash_version": flash_version if backend == "flash" else "", "allow_nonmatched": bool(allow_nonmatched),
            "save_dir": sd, "job_name": job_name(run, mode, token)}
    bad = {k: (intent.get(k), v) for k, v in want.items() if intent.get(k) != v}
    if jname != intent["job_name"]:
        bad["SLURM_JOB_NAME"] = (jname, intent["job_name"])
    if intent["state"] not in ("submitting", "submitted") or intent["job_id"] not in (None, jid) or not jid.isdigit():
        bad["state/job_id"] = ((intent["state"], intent["job_id"]), jid)
    if sd != run_dir(run, mode, root):
        bad["save_dir_registry"] = (sd, run_dir(run, mode, root))
    if bad:
        raise reg.GateError(f"this job does not own {run} {mode}: {bad}")
    try:                              # ONE fixed owner-scoped claim, atomic (Codex code r2 HIGH 2)
        os.mkdir(os.path.join(od, ATTEMPT_CLAIM))
    except FileExistsError:
        raise reg.GateError(f"{od} already has an attempt (exactly once: abandon to retry)")
    with open(os.path.join(od, ATTEMPT_CLAIM, "job"), "w") as fh:
        fh.write(f"{jid} {os.uname().nodename}\n")
    path = os.path.join(od, f"attempt_{jid}.json")
    rec = {"job_id": jid, "host": os.uname().nodename, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "state": "running"}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    with os.fdopen(fd, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    return path


def finish_attempt(path, rc):
    rec = json.load(open(path))
    rec.update(state="finished", rc=rc, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    _write_intent(path, rec)


def scheduler_state(name, runner=subprocess.run):
    """(job ids, states) of every job with this exact name; any query failure is fatal."""
    q = runner(["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", name, "-o", "%i|%T"], capture_output=True, text=True)
    a = runner(["sacct", "-n", "-X", "-P", "-u", os.environ.get("USER", ""), "--name", name, "-S", "now-30days",
                "-o", "JobID,JobName%-80,State%-40"], capture_output=True, text=True)
    if q.returncode or a.returncode:
        raise SubmitError(f"scheduler query failed (squeue rc {q.returncode}, sacct rc {a.returncode})")
    jobs = {}
    for line in q.stdout.strip().split("\n"):
        if line.strip():
            jid, st = line.strip().split("|")
            jobs[jid] = st.strip()
    for line in a.stdout.strip().split("\n"):
        if line.strip():
            jid, jname, st = line.strip().split("|")
            if jname.strip() == name:
                jobs.setdefault(jid, st.strip().split()[0])
    for jid, st in jobs.items():
        if st not in TERMINAL and st not in LIVE:
            raise SubmitError(f"job {jid}: unrecognised state {st!r}")
    return jobs


def _no_intent_state(run, mode, root=None):
    """What a (run, mode) without a live intent.json still holds: its reservation, an owner dir, archived owners."""
    od = owner_dir(run, mode, root)
    res = reg.storage_reservation(run, save_root(root)) if mode == "FULL" else None
    return {"reservation_token": res["token"][:8] if res else None, "owner_dir": os.path.isdir(od),
            "archived_owners": len(_archived_owners(run, mode, root))}


def reconcile(run, mode, root=None, runner=subprocess.run):
    """Resolve an interrupted submission; refuses (SubmitError) unless the scheduler state is unambiguous."""
    with submit_lock(run, mode, root):
        if not os.path.isfile(os.path.join(owner_dir(run, mode, root), "intent.json")):
            raise SubmitError(f"{run} {mode} has no owner intent ({_no_intent_state(run, mode, root)}): the submission "
                              "never reached sbatch, or its owner is archived -- `abandon` recovers it from the "
                              "persisted state")
        return _reconcile_locked(run, mode, root, runner)


def _reconcile_locked(run, mode, root, runner):
    od, intent = find_intent(run, mode, root)
    jobs = scheduler_state(intent["job_name"], runner)
    if intent["state"] == "submitting":
        if len(jobs) != 1:
            raise SubmitError(f"{len(jobs)} job(s) carry token {intent['token'][:8]} ({jobs}): ambiguous -- "
                              "with 0, wait for the scheduler or abandon; with several, scancel the extras first")
        intent.update(state="submitted", job_id=next(iter(jobs)), reconciled=True)
        _write_intent(os.path.join(od, "intent.json"), intent)
    elif intent["state"] == "submitted":
        if set(jobs) != {intent["job_id"]}:
            raise SubmitError(f"recorded job {intent['job_id']} but the scheduler shows {jobs}: ambiguous")
    else:
        raise SubmitError(f"intent state {intent['state']!r}")
    return {"owner": od, "intent": intent, "jobs": jobs}


def _archived_owners(run, mode, root=None):
    """{dir: intent} of the archived owner records of (run, mode) whose intent.json is readable."""
    d = os.path.join(save_root(root), OWNERS, "abandoned")
    out = {}
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not name.startswith(f"{run}__{mode}."):
            continue
        try:
            intent = json.load(open(os.path.join(d, name, "intent.json")))
        except (OSError, ValueError):
            continue
        if isinstance(intent, dict) and intent.get("run") == run and intent.get("mode") == mode:
            out[os.path.join(d, name)] = intent
    return out


def _partial_token(od):
    """The token of a half-written intent (intent.json.tmp), if readable."""
    try:
        tok = json.load(open(os.path.join(od, "intent.json.tmp"))).get("token")
    except (OSError, ValueError, AttributeError):
        return None
    return tok if isinstance(tok, str) else None


def _archive_owner(od, tag):
    dest = os.path.join(os.path.dirname(od), "abandoned", f"{os.path.basename(od)}.{time.strftime('%Y%m%dT%H%M%S')}.{tag}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    os.rename(od, dest)
    return dest


def _recover(run, mode, root, runner):
    """abandon with no live intent.json (finding 7): recover from the persisted reservation and owner state only."""
    od = owner_dir(run, mode, root)
    res = reg.storage_reservation(run, save_root(root)) if mode == "FULL" else None
    if res is None:
        if not os.path.isdir(od):
            raise SubmitError(f"{run} {mode}: no owner intent, no owner dir, no storage reservation -- nothing to abandon")
        # killed after the owner dir, before intent.json (a SMOKE holds no reservation): no sbatch was issued
        tok = _partial_token(od)
        jobs = scheduler_state(job_name(run, mode, tok), runner) if tok else {}
        if jobs:
            raise SubmitError(f"job(s) {jobs} carry token {tok[:8]} although no intent was written: refuse to guess")
        return {"recovered": "an owner dir without an intent", "owner_moved_to": _archive_owner(od, "no_intent"),
                "run_dir_moved_to": None, "jobs": jobs, "storage_released": False}
    token = res["token"]
    jobs = scheduler_state(job_name(run, mode, token), runner)
    archived = {d: i for d, i in _archived_owners(run, mode, root).items() if i.get("token") == token}
    if archived:
        # window 2: abandon archived the owner record and died before the release -- terminal status re-established
        live = {j: s for j, s in jobs.items() if s not in TERMINAL}
        if live:
            raise SubmitError(f"job(s) {live} of token {token[:8]} are not terminal: scancel them first")
        recorded = {i["job_id"] for i in archived.values() if i.get("state") == "submitted"}
        if not recorded <= set(jobs):
            raise SubmitError(f"recorded job(s) {sorted(recorded - set(jobs))} are not visible to the scheduler: "
                              "refuse to guess")
        released = reg.release_storage(run, token, save_root(root), reason="abandoned (recovered: the owner record "
                                       "was archived before the release)")
        return {"recovered": "the owner record was archived, its reservation still held", "owner": sorted(archived),
                "run_dir_moved_to": None, "jobs": jobs, "storage_released": released is not None}
    # window 1: killed after the reservation, before intent.json -- sbatch is issued only after intent.json exists,
    # so no job may carry this token
    if jobs:
        raise SubmitError(f"job(s) {jobs} carry token {token[:8]} although no intent of it exists: refuse to guess")
    if os.path.lexists(res["run_dir"]) or os.path.lexists(run_dir(run, mode, root)):
        raise SubmitError(f"{res['run_dir']} exists although no job ever carried token {token[:8]}: refuse to guess")
    tok = _partial_token(od) if os.path.isdir(od) else None
    if tok not in (None, token):
        raise SubmitError(f"{od}'s partial intent names token {tok[:8]}, not the reservation's {token[:8]}: refuse to guess")
    moved = _archive_owner(od, "no_intent") if os.path.isdir(od) else None
    released = reg.release_storage(run, token, save_root(root), reason="abandoned (recovered: the submit died before "
                                   "its intent; no job carried the token)")
    return {"recovered": "a reservation without an intent", "owner_moved_to": moved, "run_dir_moved_to": None,
            "jobs": jobs, "storage_released": released is not None}


def abandon(run, mode, root=None, runner=subprocess.run):
    with submit_lock(run, mode, root):
        if not os.path.isfile(os.path.join(owner_dir(run, mode, root), "intent.json")):
            return _recover(run, mode, root, runner)
        return _abandon_locked(run, mode, root, runner)


def _abandon_locked(run, mode, root, runner):
    od, intent = find_intent(run, mode, root)
    if intent["state"] == "submitting" and intent["job_id"] is None:
        age = time.time() - time.mktime(time.strptime(intent["ts"], "%Y-%m-%dT%H:%M:%S"))
        if age < SUBMIT_GRACE_S:                      # an sbatch of this submission may still be in flight (finding 7)
            raise SubmitError(
                f"{run} {mode}: the intent is 'submitting' with no job id ({age:.0f} s old): an sbatch of it may still be "
                f"in flight (a killed submitter does not stop its child) -- the hold stays. Check `squeue -n "
                f"{intent['job_name']}` and `sacct --name {intent['job_name']}`; once a job is visible, `reconcile` "
                f"records it; abandon releases only after {SUBMIT_GRACE_S} s with no job of that name")
    jobs = scheduler_state(intent["job_name"], runner)
    live = {j: s for j, s in jobs.items() if s not in TERMINAL}
    if live:
        raise SubmitError(f"job(s) {live} of this intent are not terminal: scancel them first")
    if intent["state"] == "submitted" and intent["job_id"] not in jobs:
        raise SubmitError(f"recorded job {intent['job_id']} is not visible to the scheduler: refuse to guess")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    dest_owner = os.path.join(os.path.dirname(od), "abandoned", f"{os.path.basename(od)}.{stamp}")
    os.makedirs(os.path.dirname(dest_owner), exist_ok=True)
    sd = intent["save_dir"]
    moved = None
    if os.path.lexists(sd):
        moved = f"{sd}.abandoned_{stamp}"
        os.rename(sd, moved)
    os.rename(od, dest_owner)
    released = reg.release_storage(run, intent["token"], save_root(root), reason="abandoned") \
        if mode == "FULL" else None                                     # terminal
    return {"abandoned": od, "owner_moved_to": dest_owner, "run_dir_moved_to": moved, "jobs": jobs,
            "storage_released": released is not None}


def release(run, root=None, runner=subprocess.run):
    """Free the storage hold of a SEALED FULL run (Codex code r2 finding 8) -- idempotent, and the only thing it
    changes is that ledger entry: the run dir, its records and the owner record are never moved or touched. Refuses
    unless the run is sealed (TRAINING_COMMITTED.json sealed; it binds the run dir's current training receipt and the
    COMPLETE terminal launch record of THIS owner's attempt), the hold is this owner's (token, run dir), and every job
    of the owner's token is terminal with the recorded job visible."""
    with submit_lock(run, "FULL", root):
        res = reg.storage_reservation(run, save_root(root))
        if res is None:
            return {"run": run, "released": False, "note": "no storage hold"}
        od, intent = find_intent(run, "FULL", root)
        rd = run_dir(run, "FULL", root)
        if intent["token"] != res["token"] or res["run_dir"] != rd or intent["save_dir"] != rd:
            raise SubmitError(f"{run}'s hold (token {res['token'][:8]}, {res['run_dir']}) is not this owner's "
                              f"(token {intent['token'][:8]}, {rd})")
        cm, _ = reg.read_sealed(os.path.join(rd, reg.COMMIT_MARKER), reg.COMMIT_KEYS)
        lr = os.path.join(rd, str(cm["launch_record"]))
        lrec = json.load(open(lr))
        if (cm["schema"], cm["run"]) != (1, run) or reg.sha256_file(lr) != cm["launch_record_sha256"] \
                or reg.sha256_file(os.path.join(rd, reg.TRAINING_RECEIPT)) != cm["training_receipt_sha256"] \
                or lrec.get("status") != "COMPLETE" or lrec.get("training_receipt_sha256") != cm["training_receipt_sha256"] \
                or lrec.get("owner") != os.path.join(od, f"attempt_{intent['job_id']}.json"):
            raise SubmitError(f"{run} is not sealed by this owner's attempt (marker / launch record / training receipt)")
        jobs = scheduler_state(intent["job_name"], runner)
        live = {j: s for j, s in jobs.items() if s not in TERMINAL}
        if live:
            raise SubmitError(f"job(s) {live} of {run} are not terminal")
        if intent["state"] != "submitted" or intent["job_id"] not in jobs:
            raise SubmitError(f"recorded job {intent['job_id']} is not visible to the scheduler: refuse to guess")
        out = reg.release_storage(run, intent["token"], save_root(root),
                                  reason="released after the run was sealed (exp45_submit_neuronic.py release; run dir kept)")
        return {"run": run, "released": out is not None, "jobs": jobs}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("release")
    s.add_argument("--run", required=True)
    for name in ("submit", "reconcile", "abandon"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--mode", required=True, choices=("SMOKE", "FULL"))
        if name == "submit":
            s.add_argument("--time", required=True)
            s.add_argument("--backend", default="flash", choices=("flash", "sdpa"))
            s.add_argument("--flash-version", default=FLASH_VERSION_DEFAULT)
            s.add_argument("--allow-nonmatched-rung", action="store_true")
            s.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.cmd == "submit":
            out = submit(a.run, a.mode, a.time, a.backend, a.flash_version if a.backend == "flash" else "",
                         a.allow_nonmatched_rung, a.dry_run)
        elif a.cmd == "reconcile":
            out = reconcile(a.run, a.mode)
        elif a.cmd == "release":
            out = release(a.run)
        else:
            out = abandon(a.run, a.mode)
        print(json.dumps(out, indent=1))
    except (SubmitError, reg.GateError, subprocess.CalledProcessError, OSError, ValueError) as err:
        print(f"EXP45 SUBMIT REFUSED: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
