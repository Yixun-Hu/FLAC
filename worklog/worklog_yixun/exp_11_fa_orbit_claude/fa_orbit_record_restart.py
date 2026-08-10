#!/usr/bin/env python3
"""Record a RESTART leg in arm_launch_registry.json from its PUBLISHED manifest.

    python3 fa_orbit_record_restart.py C4L outputs_FLAC/exp11_C4L/<manifest>.txt
    python3 fa_orbit_record_restart.py C4L <manifest> --extend   # later, as the leg saves more

A restart is only admissible if it provably continues the audited INITIAL run, so
this refuses unless the resume checkpoint ON DISK -- always re-hashed, never
trusted from the manifest -- equals that arm's recorded final_ckpt_sha256.

Re-pin review, required fix 3. The previous version was fail-OPEN: it re-hashed
only `if os.path.isfile(resume_path)`, so a manifest naming a file that could not
be resolved was recorded on the strength of its own claimed hash, and nothing
else in the manifest was checked at all. Now:

  * the canonical resume file MUST exist, sit in the audited launch's own
    checkpoint directory, and is ALWAYS re-hashed;
  * every identity field is validated against the INITIAL registry row (arm, job,
    uuid, commit, rung, config sha, VAE and P0 manifest shas, save-dir, seed) and
    against the Q10 pins read out of the launcher itself (budget 100000, resume
    step = the audited final step, and the arm's RESTART wall pin), so recorder
    and launcher cannot disagree;
  * publication is atomic (tmp + rename) under the store lock;
  * duplicates are refused -- one leg, one row.

It also publishes the leg's PRODUCER MANIFEST (fix 2): every checkpoint this leg
produced, re-hashed from disk, into an append-only per-leg file the screen
verifies each >40k checkpoint against. Re-run with --extend as the leg saves more.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa_orbit_producer_manifest as pm            # noqa: E402
from fa_orbit_ckpt_preflight import canonical_ckpt_dir    # noqa: E402

PIN_RE = re.compile(r'^(PINNED_[A-Z0-9_]+)=(?:"([^"]*)"|(\S+))')


def read_pins(launcher):
    """The launcher's own PINNED_* values, so the recorder cannot drift from them."""
    pins = {}
    with open(launcher) as fh:
        for line in fh:
            m = PIN_RE.match(line)
            if m:
                pins[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
    return pins


def parse_manifest(path):
    raw = open(path, "rb").read()
    man = {}
    for line in raw.decode().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            k, _, rest = line.partition(" ")
            man[k] = rest.strip()
    return raw, man


def kvs(man, key):
    f = (f"{key} " + man.get(key, "")).split()
    return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}


def check_identity(arm, man, initial, pins, repo_root):
    """Every field of the RESTART manifest, against the audited INITIAL row + Q10 pins."""
    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
    tk = kvs(man, "time_limit")
    problems = []
    anchor, final_step = initial.get("final_ckpt_sha256"), initial.get("final_step")
    if not anchor:
        problems.append(f"{arm} has no audited final_ckpt_sha256 to chain from — audit the "
                        "INITIAL run's final checkpoint before recording a leg")
    if jk.get("mode") != "RESTART":
        problems.append(f"manifest mode is {jk.get('mode')!r}, not RESTART")
    for field, got in (("job", jk.get("job")), ("launch_uuid", jk.get("launch_uuid")),
                       ("commit", man.get("commit"))):
        if not got:
            problems.append(f"manifest records no {field} — a leg with no identity is not a record")
    if jk.get("job") and initial.get("job") == jk.get("job"):
        problems.append(f"manifest job {jk.get('job')} IS the INITIAL job — that is the launch "
                        "already registered, not a restart leg")
    for label, got, want in (("arm", ak.get("arm"), arm),
                             ("rung", ak.get("rung"), initial.get("rung")),
                             ("micro", ak.get("micro"), pins.get("PINNED_MB")),
                             ("ngpu", ak.get("ngpu"), pins.get("PINNED_NGPU")),
                             ("config_sha256", man.get("config_sha256"), initial.get("config_sha256")),
                             ("vae_sha256", man.get("vae_sha256"), initial.get("vae_sha256")),
                             ("p0_manifest_sha256", man.get("p0_manifest_sha256"),
                              initial.get("p0_manifest_sha256")),
                             ("save_dir", man.get("save_dir"), initial.get("save_dir"))):
        if got != want:
            problems.append(f"manifest {label} {got!r} != the audited INITIAL run's {want!r}")
    if ak.get("rung") != pins.get("PINNED_RUNG"):
        problems.append(f"manifest rung {ak.get('rung')!r} != the pinned {pins.get('PINNED_RUNG')!r}")
    if ak.get("max_steps") != pins.get("PINNED_MAXSTEPS"):
        problems.append(f"manifest max_steps {ak.get('max_steps')!r} != the Q10 budget pin "
                        f"{pins.get('PINNED_MAXSTEPS')!r}")
    if final_step is not None and str(rk.get("expected_step")) != str(final_step):
        problems.append(f"manifest expected_step {rk.get('expected_step')!r} != the audited final "
                        f"step {final_step!r} — a leg resumes where the INITIAL run ended")
    want_time = pins.get(f"PINNED_TIME_LIMIT_RESTART_{arm}")
    if tk.get("time_limit") != want_time:
        problems.append(f"manifest time_limit {tk.get('time_limit')!r} != the arm's RESTART wall "
                        f"pin {want_time!r}")
    if int(initial.get("training_seed", -1)) != 42:
        problems.append(f"registered training seed {initial.get('training_seed')!r} != 42")
    # the config the leg names must still hash to the audited value
    cfg_path = man.get("model_config", "")
    if not cfg_path or not os.path.isfile(cfg_path):
        problems.append(f"manifest model_config {cfg_path!r} does not exist")
    elif hashlib.sha256(open(cfg_path, "rb").read()).hexdigest() != initial.get("config_sha256"):
        problems.append(f"{cfg_path} no longer hashes to the audited config_sha256")
    # --- the resume file itself: MUST exist, MUST be canonical, ALWAYS re-hashed --
    resume_path = (man.get("resume_ckpt", "").split() or [""])[0]
    resume_real = ""
    if not resume_path or resume_path == "<none>":
        problems.append("manifest records no resume_ckpt — a RESTART that resumed nothing is not "
                        "a continuation of the audited run")
    elif not os.path.isfile(resolve(repo_root, resume_path)):
        problems.append(f"the resume checkpoint {resume_path} does not exist — the recorder does "
                        "NOT accept the manifest's claimed hash in its place")
    else:
        resume_real = resolve(repo_root, resume_path)
        canon = canonical_ckpt_dir(initial.get("save_dir", ""), arm, repo_root)
        if os.path.realpath(os.path.dirname(resume_real)) != canon:
            problems.append(f"the resume checkpoint is not in the audited launch's canonical "
                            f"directory {canon}")
        got = pm.sha256_file(resume_real)
        if anchor and got != anchor:
            problems.append(f"the resume file on disk hashes {got[:12]}, not the audited "
                            f"{anchor[:12]} — this leg does not continue that run")
        if rk.get("resume_ckpt_sha256") != got:
            problems.append(f"manifest resume_ckpt_sha256 {str(rk.get('resume_ckpt_sha256'))[:12]} "
                            f"!= the file's actual {got[:12]}")
    return problems, resume_real


def resolve(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="record an exp_11 RESTART leg")
    ap.add_argument("arm")
    ap.add_argument("manifest")
    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
    ap.add_argument("--launcher", default=os.path.join(HERE, "fa_orbit_train.sbatch"),
                    help="where the Q10 pins are read from")
    ap.add_argument("--producer-dir", default=HERE,
                    help="where the per-leg producer manifests are published")
    # HERE = <repo>/worklog/worklog_<user>/exp_11_fa_orbit_claude
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                    help="root the manifest's relative paths resolve against")
    ap.add_argument("--extend", action="store_true",
                    help="this leg is already recorded: extend its producer manifest only")
    ap.add_argument("--rehash-all", action="store_true",
                    help="re-hash published checkpoints too (full audit, expensive)")
    ap.add_argument("--dry-run", action="store_true", help="validate and report, publish nothing")
    args = ap.parse_args(argv)

    arm = args.arm
    pins = read_pins(args.launcher)
    if not pins.get("PINNED_MAXSTEPS"):
        raise SystemExit(f"no PINNED_* values found in {args.launcher}")

    # One writer at a time, and the lock is the registry's own DIRECTORY: no lock
    # file to leave behind in a tracked tree, and it still covers the tmp+rename.
    store = os.path.dirname(os.path.abspath(args.registry)) or "."
    lock_fd = os.open(store, os.O_RDONLY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return record(args, arm, pins)
    finally:
        os.close(lock_fd)


def record(args, arm, pins):
    reg = json.load(open(args.registry))
    initial = reg.get("arms", {}).get(arm)
    if initial is None:
        raise SystemExit(f"{arm} has no INITIAL registry entry")

    raw, man = parse_manifest(args.manifest)
    man_sha = hashlib.sha256(raw).hexdigest()
    problems, resume_real = check_identity(arm, man, initial, pins, args.repo_root)
    jk, ak, rk = kvs(man, "job"), kvs(man, "arm"), kvs(man, "resume_ckpt")
    job = jk.get("job")

    legs = reg.setdefault("restarts", {}).setdefault(arm, [])
    same = [l for l in legs if l.get("job") == job or l.get("launch_uuid") == jk.get("launch_uuid")
            or l.get("manifest_sha256") == man_sha]
    if same and not args.extend:
        raise SystemExit(f"{arm} job {job} is ALREADY recorded ({len(same)} matching leg(s)) — "
                         "one leg, one row; use --extend to extend its producer manifest")
    if len(same) > 1:
        problems.append(f"{len(same)} registry rows already claim this leg — the registry is "
                        "inconsistent; fix it before recording")
    if args.extend and not same:
        problems.append(f"--extend given but {arm} job {job} is not recorded yet")
    if problems:
        print("RECORD REFUSED:")
        for p in problems:
            print(f"  !! {p}")
        return 2

    anchor = initial["final_ckpt_sha256"]
    producer = pm.manifest_name(arm, job)
    row = {
        "manifest_path": args.manifest, "manifest_sha256": man_sha,
        "job": job, "mode": "RESTART", "launch_uuid": jk.get("launch_uuid"),
        "arm": arm, "commit": man.get("commit"), "rung": ak.get("rung"),
        "config_sha256": man.get("config_sha256"), "save_dir": man.get("save_dir"),
        "resume_ckpt": resume_real, "resume_ckpt_sha256": anchor,
        "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
        "time_limit": kvs(man, "time_limit").get("time_limit"),
        "producer_manifest": producer, "chains_to": anchor,
        "recorded_at": pm._now(),
    }
    header = {"arm": arm, "job": job, "launch_uuid": jk.get("launch_uuid"), "mode": "RESTART",
              "commit": man.get("commit"), "resume_ckpt_sha256": anchor,
              "expected_step": rk.get("expected_step"), "max_steps": ak.get("max_steps"),
              "save_dir": man.get("save_dir"), "config_sha256": man.get("config_sha256"),
              "chains_to": anchor, "leg_manifest_sha256": man_sha}

    ckpt_dir = canonical_ckpt_dir(initial["save_dir"], arm, args.repo_root)
    prod_path = os.path.join(args.producer_dir, producer)
    known = (pm.load(prod_path) or {}).get("checkpoints", {})
    found, scan_problems = pm.scan_checkpoints(
        ckpt_dir, int(rk["expected_step"]), int(ak["max_steps"]), known=known,
        rehash_all=args.rehash_all, repo_root=args.repo_root)
    added, kept, pub_problems = pm.publish(prod_path, header, found, dry_run=args.dry_run)
    if scan_problems or pub_problems:
        print("RECORD REFUSED:")
        for p in scan_problems + pub_problems:
            print(f"  !! {p}")
        return 2

    if args.extend:
        for i, leg in enumerate(legs):
            if leg.get("job") == job:
                legs[i] = {**leg, "producer_manifest": producer}
    else:
        legs.append(row)
    if not args.dry_run:
        pm.write_atomic(args.registry, reg)
    verb = "extended" if args.extend else "recorded"
    print(f"{verb} {arm} RESTART job {job} chaining to {anchor[:12]} "
          f"({'dry run, nothing written' if args.dry_run else 'published'})")
    print(f"  producer manifest {producer}: {len(added)} checkpoint(s) added, "
          f"{len(kept)} already published"
          + (f" (steps {added[0]}..{added[-1]})" if added else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
