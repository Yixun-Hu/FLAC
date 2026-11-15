#!/usr/bin/env python3
"""exp_38 neuronic submitter — one monolithic job per (run, mode), exactly once (plan r2 §10 B7;
Codex code r1 HIGH 6).

Run on the neuronic login node from the exp-38 checkout (HEAD pushed, tracked tree clean):

    exp38_submit_neuronic.py submit    --run de_yaw_f025 --mode SMOKE --time 01:00:00 [--dry-run]
    exp38_submit_neuronic.py submit    --run de_yaw_f025 --mode FULL  --time 3-00:00:00 [--dry-run]
    exp38_submit_neuronic.py reconcile --run de_yaw_f025 --mode FULL      # after an interrupted submit
    exp38_submit_neuronic.py abandon   --run de_yaw_f025 --mode FULL      # only once the job is proven gone

Ownership: ONE stable owner record per (run, mode), ``<save root>/.exp38_owners/<run>__<MODE>/``,
created atomically (``mkdir``) with ``intent.json`` (state "submitting") BEFORE ``sbatch``; the job id
is added after ``sbatch`` (state "submitted"). The owner record is separate from the attempt payload:
the job itself writes ``attempt_<jobid>.json`` there (exactly one attempt per owner; ``--no-requeue``),
after exact-comparing its token / run / mode / HEAD / backend / flash version / rung flag / save dir
/ job name / job id with the intent. SMOKE and FULL have fixed run dirs (``<run>_smoke`` / ``<run>``),
so a second SMOKE is refused exactly like a second FULL. ``reconcile`` resolves an interrupted
submission only when exactly one job carries the token (zero or several: refuse, exit 2);
``abandon`` (all token jobs terminal) moves the owner record to ``.exp38_owners/abandoned/`` and the
run dir to ``<dir>.abandoned_<ts>`` -- an abandoned record is never found again.

Written by the exp_38 Coder (Claude Opus 5.5).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import exp38_registry as reg  # noqa: E402

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
            "BOOT_FAIL", "DEADLINE", "REVOKED"}
LIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED", "RESIZING", "STOPPED",
        "SIGNALING", "STAGE_OUT", "REQUEUE_HOLD", "REQUEUE_FED", "SPECIAL_EXIT"}
SUBLOG = f"{reg.EXP_DIR}/exp38_neuronic_submissions.log"
OWNERS = ".exp38_owners"
ATTEMPT_CLAIM = "attempt.claim"
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
                         "allow_nonmatched", "save_dir", "job_name", "state", "ts", "job_id", "reconciled"})


def job_name(run, mode, token):
    return f"exp38-{run}-{mode.lower()}-{token[:8]}"


def sbatch_argv(run, mode, token, head, time_limit, backend, flash_version, sd, allow_nonmatched):
    r = resources(run)
    exports = {"RUN": run, "MODE": mode, "EXPECT_SHA": head, "EXP38_TOKEN": token, "ATTN_BACKEND": backend,
               "EXPECT_FLASH_VERSION": flash_version or "", "ALLOW_NONMATCHED_RUNG": "1" if allow_nonmatched else "0",
               "SAVE_DIR": sd}
    return ["sbatch", "--parsable", f"--job-name={job_name(run, mode, token)}", f"--comment=exp38:{token}",
            f"--gres=gpu:l40:{r['ngpu']}", f"--cpus-per-task={r['cpus']}", f"--mem={r['mem']}",
            f"--time={time_limit}", "--no-requeue", f"--chdir={reg.REPO}",
            f"--output={reg.REPO}/{reg.EXP_DIR}/slurm_train_%x_%j.out",
            "--export=ALL," + ",".join(f"{k}={v}" for k, v in exports.items()),
            f"{reg.REPO}/{reg.EXP_DIR}/exp38_train.sbatch"]


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
        subprocess.run(["git", "-C", reg.REPO, "fetch", "-q", "origin", "exp-38-data-efficiency"], check=True)
        if subprocess.run(["git", "-C", reg.REPO, "merge-base", "--is-ancestor", head, "FETCH_HEAD"]).returncode:
            raise SubmitError(f"HEAD {head} is not pushed to origin/exp-38-data-efficiency")
    reg.describe_run(run, "neuronic", allow_nonmatched=allow_nonmatched)
    reg.check_e0_receipt("neuronic")
    if mode == "FULL":
        reg.check_smoke_receipt(run, "neuronic", backend, head)
    return head


def _dry_preflight(run, allow_nonmatched):
    """DRY_RUN: the recipe checks only (receipts are produced at campaign time)."""
    reg.describe_run(run, "neuronic", allow_nonmatched=allow_nonmatched)
    return _git("rev-parse", "HEAD")


def submit(run, mode, time_limit, backend, flash_version, allow_nonmatched, dry, root=None, runner=subprocess.run):
    if mode not in ("SMOKE", "FULL"):
        raise SubmitError("mode must be SMOKE or FULL")
    import re
    if not re.fullmatch(r"(\d+-)?\d{1,3}:\d{2}:\d{2}", time_limit or ""):
        raise SubmitError(f"--time {time_limit!r} is not a Slurm time (e.g. 24:00:00 or 3-00:00:00)")
    head = preflight(run, mode, allow_nonmatched, dry, backend) if not dry else _dry_preflight(run, allow_nonmatched)
    token = uuid.uuid4().hex
    sd = run_dir(run, mode, root)
    argv = sbatch_argv(run, mode, token, head, time_limit, backend, flash_version, sd, allow_nonmatched)
    od = owner_dir(run, mode, root)
    if dry:
        return {"dry_run": True, "run_dir": sd, "owner": od, "argv": argv}
    reg.check_run_dir(sd, run, 0)
    os.makedirs(os.path.dirname(od), exist_ok=True)
    try:
        os.mkdir(od)
    except FileExistsError:
        raise SubmitError(f"{od} exists: {run} {mode} is already owned (reconcile / abandon)")
    intent = {"schema": 1, "token": token, "run": run, "mode": mode, "head": head, "time": time_limit,
              "backend": backend, "flash_version": flash_version or "", "allow_nonmatched": bool(allow_nonmatched),
              "save_dir": sd, "job_name": job_name(run, mode, token), "state": "submitting",
              "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "job_id": None, "reconciled": False}
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
    return {"run_dir": sd, "owner": od, "job_id": jid, "token": token}


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


def reconcile(run, mode, root=None, runner=subprocess.run):
    """Resolve an interrupted submission; refuses (SubmitError) unless the scheduler state is unambiguous."""
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


def abandon(run, mode, root=None, runner=subprocess.run):
    od, intent = find_intent(run, mode, root)
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
    return {"abandoned": od, "owner_moved_to": dest_owner, "run_dir_moved_to": moved, "jobs": jobs}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
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
        else:
            out = abandon(a.run, a.mode)
        print(json.dumps(out, indent=1))
    except (SubmitError, reg.GateError, subprocess.CalledProcessError, OSError, ValueError) as err:
        print(f"EXP38 SUBMIT REFUSED: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
