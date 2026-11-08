#!/usr/bin/env python3
"""exp_38 neuronic submitter — one monolithic job per run, exactly-once (plan r2 §10 B7).

Run on the neuronic login node from the exp-38 checkout (HEAD pushed, tracked tree clean):

    exp38_submit_neuronic.py submit    --run de_yaw_f025 --mode SMOKE --time 01:00:00 [--dry-run]
    exp38_submit_neuronic.py submit    --run de_yaw_f025 --mode FULL  --time 3-00:00:00 [--dry-run]
    exp38_submit_neuronic.py reconcile --run de_yaw_f025 --mode FULL      # after an interrupted submit
    exp38_submit_neuronic.py abandon   --run de_yaw_f025 --mode FULL      # only once the job is proven gone

Ownership protocol: the run dir's ``.exp38_submitted/`` marker is created atomically (``mkdir``) with
``intent.json`` = {token, state "submitting", ...} BEFORE ``sbatch``; the job name carries the token
(``exp38-<run>-<mode>-<token8>``, also ``--comment exp38:<token>``); the job id is recorded after
``sbatch`` (state "submitted"). A second submission for the same run dir is refused by the marker.
``reconcile`` finds an interrupted submission's job by its token-bearing name (squeue + sacct; any
query failure is fatal). ``abandon`` renames the run dir to ``<run>.abandoned_<ts>`` only when the
token's job is terminal or provably never existed. SMOKE and FULL use distinct run dirs.

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
FLASH_VERSION_DEFAULT = reg.FLASH_VERSION


class SubmitError(RuntimeError):
    pass


def resources(run):
    _, _, rung = reg.parse_run_id(run)
    n = reg.RUNGS[rung]["ngpu"]
    return {"ngpu": n, "cpus": 8 + 7 * n, "mem": "64G" if n == 2 else "108G"}


def run_dir(run, mode, token=None, save_root=None):
    root = save_root or reg.PLATFORMS["neuronic"]["save_root"]
    if mode == "FULL":
        return os.path.join(root, run)
    if token is None:
        raise SubmitError("a SMOKE run dir needs its token")
    return os.path.join(root, f"{run}_smoke_{token[:8]}")


def job_name(run, mode, token):
    return f"exp38-{run}-{mode.lower()}-{token[:8]}"


def sbatch_argv(run, mode, token, head, time_limit, backend, flash_version, sd, allow_nonmatched):
    r = resources(run)
    exports = {"RUN": run, "MODE": mode, "EXPECT_SHA": head, "EXP38_TOKEN": token, "ATTN_BACKEND": backend,
               "EXPECT_FLASH_VERSION": flash_version or "", "ALLOW_NONMATCHED_RUNG": "1" if allow_nonmatched else "0",
               "SAVE_DIR": sd}
    return ["sbatch", "--parsable", f"--job-name={job_name(run, mode, token)}", f"--comment=exp38:{token}",
            f"--gres=gpu:l40:{r['ngpu']}", f"--cpus-per-task={r['cpus']}", f"--mem={r['mem']}",
            f"--time={time_limit}", f"--chdir={reg.REPO}",
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


def preflight(run, mode, allow_nonmatched, dry):
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
        reg.check_smoke_receipt(run, "neuronic")
    return head


def _dry_preflight(run, allow_nonmatched):
    """DRY_RUN: the recipe checks only (receipts are produced at campaign time)."""
    reg.describe_run(run, "neuronic", allow_nonmatched=allow_nonmatched)
    return _git("rev-parse", "HEAD")


def submit(run, mode, time_limit, backend, flash_version, allow_nonmatched, dry, save_root=None, runner=subprocess.run):
    if mode not in ("SMOKE", "FULL"):
        raise SubmitError("mode must be SMOKE or FULL")
    import re
    if not re.fullmatch(r"(\d+-)?\d{1,3}:\d{2}:\d{2}", time_limit or ""):
        raise SubmitError(f"--time {time_limit!r} is not a Slurm time (e.g. 24:00:00 or 3-00:00:00)")
    head = preflight(run, mode, allow_nonmatched, dry) if not dry else _dry_preflight(run, allow_nonmatched)
    token = uuid.uuid4().hex
    sd = run_dir(run, mode, token, save_root)
    argv = sbatch_argv(run, mode, token, head, time_limit, backend, flash_version, sd, allow_nonmatched)
    if dry:
        return {"dry_run": True, "run_dir": sd, "argv": argv}
    reg.check_run_dir(sd, run, 0)
    os.makedirs(sd, exist_ok=True)
    marker = os.path.join(sd, ".exp38_submitted")
    try:
        os.mkdir(marker)
    except FileExistsError:
        raise SubmitError(f"{marker} exists: {run} {mode} was already submitted (reconcile / abandon)")
    intent = {"token": token, "run": run, "mode": mode, "head": head, "time": time_limit, "backend": backend,
              "job_name": job_name(run, mode, token), "state": "submitting", "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _write_intent(os.path.join(marker, "intent.json"), intent)
    out = runner(argv, capture_output=True, text=True)
    if out.returncode != 0:
        raise SubmitError(f"sbatch failed ({out.stderr.strip()}); the intent stays 'submitting' -> run reconcile")
    jid = out.stdout.strip().split(";")[0]
    if not jid.isdigit():
        raise SubmitError(f"sbatch returned {out.stdout!r}; run reconcile")
    intent.update(state="submitted", job_id=jid)
    _write_intent(os.path.join(marker, "intent.json"), intent)
    with open(os.path.join(reg.REPO, SUBLOG), "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {run} {mode} job {jid} token {token} head {head}: {' '.join(argv)}\n")
    return {"run_dir": sd, "job_id": jid, "token": token}


def find_intent(run, mode, save_root=None):
    root = save_root or reg.PLATFORMS["neuronic"]["save_root"]
    cands = [os.path.join(root, run)] if mode == "FULL" else \
        [os.path.join(root, d) for d in sorted(os.listdir(root)) if d.startswith(f"{run}_smoke_")]
    found = [c for c in cands if os.path.isfile(os.path.join(c, ".exp38_submitted", "intent.json"))]
    if len(found) != 1:
        raise SubmitError(f"expected one {run} {mode} intent, found {found}")
    return found[0], json.load(open(os.path.join(found[0], ".exp38_submitted", "intent.json")))


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


def reconcile(run, mode, save_root=None, runner=subprocess.run):
    sd, intent = find_intent(run, mode, save_root)
    jobs = scheduler_state(intent["job_name"], runner)
    if intent.get("state") == "submitting" and len(jobs) == 1:
        intent.update(state="submitted", job_id=next(iter(jobs)), reconciled=True)
        _write_intent(os.path.join(sd, ".exp38_submitted", "intent.json"), intent)
    return {"run_dir": sd, "intent": intent, "jobs": jobs}


def abandon(run, mode, save_root=None, runner=subprocess.run):
    sd, intent = find_intent(run, mode, save_root)
    jobs = scheduler_state(intent["job_name"], runner)
    live = {j: s for j, s in jobs.items() if s not in TERMINAL}
    if live:
        raise SubmitError(f"job(s) {live} of this intent are not terminal: scancel them first")
    if intent.get("state") == "submitted" and intent.get("job_id") not in jobs:
        raise SubmitError(f"recorded job {intent.get('job_id')} is not visible to the scheduler: refuse to guess")
    dest = f"{sd}.abandoned_{time.strftime('%Y%m%dT%H%M%S')}"
    os.rename(sd, dest)
    return {"abandoned": sd, "moved_to": dest, "jobs": jobs}


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
