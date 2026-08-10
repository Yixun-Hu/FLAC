#!/usr/bin/env python3
"""Per-leg PRODUCER manifests: which checkpoints a RESTART leg actually produced.

Re-pin review, required fix 2. The >40k lineage gate was EXISTENTIAL: once any
registry leg for an arm carried `mode=RESTART` and the right 40k resume hash,
every later checkpoint in that arm's canonical directory passed, because nothing
bound the evaluated checkpoint's own bytes to the leg that produced it. A
same-config checkpoint from a wrong restart, copied into the canonical directory,
was admissible.

DESIGN CHOICE (the review allowed two; this is the second, and why).
  Rejected: the restart leg's JOB hashes each checkpoint as it saves. That is the
  tighter binding, but it means editing fa_orbit_train.sbatch's training path --
  a hashing sidecar running beside torchrun -- while jobs 3662828-30 sit queued
  against that launcher, which this round is forbidden to do (and which would put
  sustained multi-GB reads next to a live training job on a shared filesystem).
  Chosen: the RECORDER captures the leg's checkpoint inventory (step -> sha256,
  re-hashed from disk) and publishes it into an APPEND-ONLY, COMMITTED per-leg
  file next to the audited registry. The screen re-hashes the checkpoint it is
  about to evaluate and requires an exact step -> sha256 -> leg match.

Why that is still immutable evidence: the file lives in the tracked experiment
directory, exactly like arm_launch_registry.json, and screens read it from the
PINNED worktree -- so a row can only enter it through a commit, and only a
commit that is in the campaign pin can be used as evidence. Within the file the
recorder is append-only: a step already published may never change its sha256 or
its path, and the header (arm, job, launch uuid, resume anchor, save-dir, config
sha, budget) may never change at all.
"""
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime

CKPT_RE = re.compile(r"^epoch=(\d+)-step=(\d+)\.ckpt$")
# Header fields that identify the LEG. Once published they are frozen: a
# republish that disagrees on any of them is a different leg wearing this file's
# name, not an extension of it.
HEADER_FIELDS = ("arm", "job", "launch_uuid", "mode", "commit", "resume_ckpt_sha256",
                 "expected_step", "max_steps", "save_dir", "config_sha256",
                 "chains_to", "leg_manifest_sha256")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_name(arm, job):
    """The per-leg file name. Flat in the experiment directory, like the registry
    and the backfill manifest, so the launcher/submitter drift gates (`$EXPDIR/*.json`)
    cover it."""
    return f"fa_orbit_producer_{arm}_job{job}.json"


def load(path):
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def rel_to(root, path):
    """Repo-relative when possible (portable across the pinned worktrees), else absolute."""
    real, root_real = os.path.realpath(path), os.path.realpath(root)
    return os.path.relpath(real, root_real) if real.startswith(root_real + os.sep) else real


def resolve(root, path):
    return path if os.path.isabs(path) else os.path.join(root, path)


def scan_checkpoints(ckpt_dir, after_step, max_step, known=None, rehash_all=False, repo_root="."):
    """Re-hash the leg's checkpoints from DISK: {str(step): {path, sha256, bytes}}.

    Only steps strictly after the resume point and no further than the budget are
    the leg's own output -- the resume checkpoint itself belongs to the INITIAL
    run and is already anchored in the registry. Steps already published are not
    re-read by default (they are immutable evidence, and each is ~724 MB on a
    shared filesystem); ``rehash_all`` forces a full audit."""
    known = known or {}
    out, problems = {}, []
    if not os.path.isdir(ckpt_dir):
        return {}, [f"checkpoint directory not found: {ckpt_dir}"]
    for name in sorted(os.listdir(ckpt_dir)):
        m = CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(2))
        if step <= after_step or step > max_step:
            continue
        path = os.path.join(ckpt_dir, name)
        key = str(step)
        if key in out:
            problems.append(f"two checkpoint files claim step {step} in {ckpt_dir}")
            continue
        if key in known and not rehash_all:
            out[key] = dict(known[key])
            continue
        out[key] = {"path": rel_to(repo_root, path), "sha256": sha256_file(path),
                    "bytes": os.path.getsize(path)}
    return out, problems


def publish(path, header, checkpoints, dry_run=False):
    """Append-only publication of a leg's inventory. Returns (added, kept, problems).

    The header is frozen and a published step may not change; the write is
    tmp+rename in the destination directory, so a reader never sees a partial
    file and a failed write leaves the previous one intact."""
    problems, added, kept = [], [], []
    old = load(path)
    if old is not None:
        for field in HEADER_FIELDS:
            if str(old.get(field)) != str(header.get(field)):
                problems.append(f"{os.path.basename(path)} is already published with "
                                f"{field}={old.get(field)!r}, not {header.get(field)!r} — a "
                                "producer manifest is immutable")
        for step, entry in sorted(old.get("checkpoints", {}).items()):
            new = checkpoints.get(step)
            if new is None:
                continue        # a published step whose file is gone stays published
            if new["sha256"] != entry["sha256"]:
                problems.append(f"step {step} is published with sha256 {entry['sha256'][:12]} but "
                                f"now hashes {new['sha256'][:12]} — a published checkpoint may "
                                "never change")
            elif os.path.basename(str(new["path"])) != os.path.basename(str(entry["path"])):
                problems.append(f"step {step} is published at {entry['path']} but now at "
                                f"{new['path']}")
    if problems:
        return [], [], problems
    merged = dict((old or {}).get("checkpoints", {}))
    for step, entry in checkpoints.items():
        (kept if step in merged else added).append(step)
        merged.setdefault(step, entry)
    doc = {k: header.get(k) for k in HEADER_FIELDS}
    doc["_comment"] = [
        "APPEND-ONLY producer manifest for one exp_11 RESTART leg (re-pin fix 2).",
        "Every checkpoint this leg produced, re-hashed from disk by",
        "fa_orbit_record_restart.py. The screen admits a >40k checkpoint only when",
        "its own sha256 matches this file's entry for exactly that step, so a",
        "valid restart row no longer vouches for any later same-config file.",
        "A published step never changes; the header never changes.",
    ]
    doc["first_published"] = (old or {}).get("first_published") or _now()
    doc["last_extended"] = _now()
    doc["checkpoints"] = dict(sorted(merged.items(), key=lambda kv: int(kv[0])))
    if not dry_run:
        write_atomic(path, doc)
    return sorted(added, key=int), sorted(kept, key=int), []


def write_atomic(path, doc):
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".{}.".format(os.path.basename(path)), dir=d)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def validate_leg(leg, row, arm, step=None):
    """Every field of a registry RESTART row, against the audited INITIAL row.

    The old gate checked two of them (mode, resume sha) and only existentially."""
    problems = []
    anchor = row.get("final_ckpt_sha256")
    for label, got, want in (("arm", leg.get("arm"), arm),
                             ("mode", leg.get("mode"), "RESTART"),
                             ("resume_ckpt_sha256", leg.get("resume_ckpt_sha256"), anchor),
                             ("chains_to", leg.get("chains_to"), anchor),
                             ("save_dir", leg.get("save_dir"), row.get("save_dir")),
                             ("config_sha256", leg.get("config_sha256"), row.get("config_sha256"))):
        if got != want:
            problems.append(f"restart leg {label} {got!r} != the audited INITIAL row's {want!r}")
    for field in ("job", "launch_uuid", "commit", "manifest_sha256", "producer_manifest"):
        if not leg.get(field):
            problems.append(f"restart leg records no {field}")
    try:
        expected, budget = int(leg["expected_step"]), int(leg["max_steps"])
    except (KeyError, TypeError, ValueError):
        problems.append(f"restart leg has no integer expected_step/max_steps "
                        f"({leg.get('expected_step')!r}/{leg.get('max_steps')!r})")
        return problems
    if str(row.get("final_step")) != str(expected):
        problems.append(f"restart leg resumed at step {expected}, but the audited INITIAL run "
                        f"ended at {row.get('final_step')}")
    if budget <= expected:
        problems.append(f"restart leg budget {budget} does not exceed its resume step {expected}")
    if step is not None and not expected < int(step) <= budget:
        problems.append(f"step {step} is outside this leg's output range "
                        f"({expected} < step <= {budget})")
    return problems


def verify_chain(reg, arm, step, ckpt_path, ckpt_sha, base_dir, repo_root="."):
    """Bind ONE checkpoint to the leg that produced it. Returns (problems, note).

    ``base_dir`` is the directory the registry was read from, so the per-leg files
    travel with it (pinned worktree, or a synthetic registry under test)."""
    row = (reg.get("arms") or {}).get(arm)
    if row is None:
        return [f"{arm} is not in the audited launch registry"], ""
    if not row.get("final_ckpt_sha256"):
        return [f"{arm} has no audited final_ckpt_sha256, so a >40k checkpoint cannot be "
                "chained to its INITIAL run"], ""
    legs = (reg.get("restarts") or {}).get(arm) or []
    if not legs:
        return [f"checkpoint at step {step} is above the INITIAL budget but {arm} has no RESTART "
                "entry in the audited registry — record the leg with fa_orbit_record_restart.py "
                "first"], ""
    why = []
    for i, leg in enumerate(legs):
        bad = validate_leg(leg, row, arm, step=step)
        if bad:
            why.append(f"leg {leg.get('job', i)}: " + "; ".join(bad))
            continue
        # The leg's OWN restart manifest is mutable evidence under gitignored
        # outputs_FLAC, exactly like the INITIAL one the screen already re-hashes:
        # it must still be there, and still be the bytes that were recorded.
        leg_man = resolve(repo_root, str(leg.get("manifest_path")))
        if not os.path.isfile(leg_man):
            why.append(f"leg {leg.get('job')}: the registered RESTART manifest {leg_man} is gone")
            continue
        got = sha256_file(leg_man)
        if got != leg.get("manifest_sha256"):
            why.append(f"leg {leg.get('job')}: RESTART manifest {leg_man} now hashes {got[:12]}, "
                       f"not the registered {str(leg.get('manifest_sha256'))[:12]} — it changed "
                       "after it was recorded")
            continue
        man_path = resolve(base_dir, str(leg.get("producer_manifest")))
        man = load(man_path)
        if man is None:
            why.append(f"leg {leg.get('job')}: producer manifest {man_path} is missing")
            continue
        head_bad = [f"producer manifest {f}={man.get(f)!r} != the registry leg's {leg.get(f)!r}"
                    for f in ("arm", "job", "launch_uuid", "resume_ckpt_sha256", "chains_to")
                    if str(man.get(f)) != str(leg.get(f))]
        if str(man.get("leg_manifest_sha256")) != str(leg.get("manifest_sha256")):
            head_bad.append("producer manifest is not the one this registry row published")
        if head_bad:
            why.append(f"leg {leg.get('job')}: " + "; ".join(head_bad))
            continue
        entry = (man.get("checkpoints") or {}).get(str(step))
        if entry is None:
            why.append(f"leg {leg.get('job')}: produced no checkpoint at step {step} "
                       f"(published: {sorted((man.get('checkpoints') or {}), key=int)})")
            continue
        if entry.get("sha256") != ckpt_sha:
            why.append(f"leg {leg.get('job')}: step {step} was published as "
                       f"{str(entry.get('sha256'))[:12]}, this file hashes {ckpt_sha[:12]} — this "
                       "is NOT the checkpoint that leg produced")
            continue
        if os.path.realpath(resolve(repo_root, str(entry.get("path")))) != os.path.realpath(ckpt_path):
            why.append(f"leg {leg.get('job')}: step {step} was published at {entry.get('path')}, "
                       f"not {ckpt_path}")
            continue
        return [], (f"producer binding OK: step {step} ({ckpt_sha[:12]}) was produced by RESTART "
                    f"job {leg.get('job')}, which resumed the audited "
                    f"{str(row['final_ckpt_sha256'])[:12]} and published it in "
                    f"{os.path.basename(man_path)}")
    return [f"no validated RESTART leg for {arm} published step {step} with sha256 "
            f"{ckpt_sha[:12]} — " + " | ".join(why)], ""
