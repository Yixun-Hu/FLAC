#!/usr/bin/env python3
"""exp_14 (fa_drawshare) admission-evidence contract: stamp it, and verify it.

Seats this round: main session claude-fable-5 (xhigh) per the model-change flag;
code by the Opus 5 Coder seat (SOP §Roles).

WHY THIS EXISTS (r1 code review, finding 3). The plan gates the campaign twice:

  * plan §3.2 - the cap-96 arm may not be committed to until a REAL 15-step DDP
    fit probe has passed ("if cap 96 does not fit, stop and report - do not
    silently fall back to cap 64");
  * plan §5 (re-review 6) - DS-CS3 does not start until DS-PA has passed its
    40,000-step admission/parity audit ("if parity fails, STOP and report rather
    than spend the second six days").

Both were prose. Prose does not stop a launch at 03:00, so each gate now has to
be discharged by a FILE this module writes and `dsarm_launch.sh` verifies before
it will start a FULL (or RESTART) run. An unstamped gate is a hard abort.

THE BIND. An evidence file is only meaningful for the code it was produced
under, so every record carries three hashes and all three are checked:

  * ``treatment_sha256``  - over the WORKING-TREE bytes of the files that can
    change the chunk plan (yaw_rotation / diffusion / factory / train.py). This
    is the substantive bind: if any of them changed, a fit probe or a parity
    audit taken beforehand describes a different method.
  * ``model_config_sha256`` - the arm JSON the evidence is about.
  * ``source_sha`` - ``git rev-parse HEAD`` at stamping time, checked against
    HEAD at launch. Plan §5 already lists "wrong SHA" as a hard abort, so this is
    the plan's own rule made executable rather than a new one.

The working tree - not the committed blob - is hashed on purpose: the launcher
runs what is on disk, and hashing the commit would bless an edited file.
``dsarm_launch.sh`` separately refuses a FULL/RESTART launch whose treatment
paths are dirty, so a stamped record still refers to committed content.

Usage (stamp, after the gate has actually passed - this file records a verdict,
it does not compute one):

  python worklog/worklog_yixun/exp_14_fa_drawshare_claude/stamp_evidence.py \
      --kind cap_fit --arm DSCS3 --verdict PASS \
      --log worklog/worklog_yixun/exp_14_fa_drawshare_claude/fa_drawshare_<ts>_DSCS3_probe_train.log \
      --notes "15-step DDP fit at micro-32x2, peak 19.4 GiB/rank"

Usage (verify by hand):

  python .../stamp_evidence.py --verify --kind cap_fit --arm DSCS3
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

SCHEMA = 1
KINDS = ("cap_fit", "dspa_40k_audit")
ARMS = {"DSPA": 32, "DSCS3": 96}

# Files whose contents can change what the chunk plan IS. Order is part of the
# hash: a stable order makes the digest reproducible across machines.
TREATMENT_PATHS = (
    "src/data/yaw_rotation.py",
    "src/training/diffusion.py",
    "src/training/factory.py",
    "train.py",
)

EXPDIR = "worklog/worklog_yixun/exp_14_fa_drawshare_claude"
EVIDENCE_DIRNAME = "evidence"
ARM_CONFIG = {"DSPA": f"{EXPDIR}/FLAC_AR_BF_DSPA.json",
              "DSCS3": f"{EXPDIR}/FLAC_AR_BF_DSCS3.json"}

# Which evidence a given (arm, mode) admission requires. dspa_40k_audit is about
# DS-PA even when DS-CS3 is the arm being launched, hence the explicit subject.
REQUIRED = {
    "DSPA": (("cap_fit", "DSPA"),),
    "DSCS3": (("cap_fit", "DSCS3"), ("dspa_40k_audit", "DSPA")),
}


def repo_root(start=None):
    """Marker-walk to the checkout root (survives worklog relocations)."""
    p = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    while not os.path.isdir(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def treatment_fingerprint(root=None):
    """sha256 over the working-tree bytes of every treatment-path file.

    The path name is hashed with the content so that moving code between two of
    these files cannot leave the digest unchanged.
    """
    root = root or repo_root()
    h = hashlib.sha256()
    for rel in TREATMENT_PATHS:
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            raise RuntimeError(f"treatment path missing: {rel}")
        h.update(rel.encode())
        h.update(b"\0")
        with open(p, "rb") as f:
            h.update(f.read())
        h.update(b"\0")
    return h.hexdigest()


def source_sha(root=None):
    root = root or repo_root()
    out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                  stderr=subprocess.DEVNULL, timeout=30)
    return out.decode().strip()


def dirty_treatment_paths(root=None):
    """Treatment-path files with uncommitted changes (staged or not)."""
    root = root or repo_root()
    out = subprocess.check_output(["git", "status", "--porcelain", "--"] + list(TREATMENT_PATHS),
                                  cwd=root, stderr=subprocess.DEVNULL, timeout=30)
    dirty = []
    for line in out.decode().splitlines():
        if line.strip():
            dirty.append(line[3:].strip())
    return dirty


def evidence_path(kind, arm, root=None):
    root = root or repo_root()
    stem = f"{kind}_{arm}.json"
    return os.path.join(root, EXPDIR, EVIDENCE_DIRNAME, stem)


def build_record(kind, arm, verdict, log, notes, root=None):
    root = root or repo_root()
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {list(KINDS)}, got {kind!r}")
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}, got {arm!r}")
    if kind == "dspa_40k_audit" and arm != "DSPA":
        raise ValueError("kind 'dspa_40k_audit' is about DS-PA; arm must be DSPA")
    cfg = os.path.join(root, ARM_CONFIG[arm])
    return {
        "schema": SCHEMA,
        "kind": kind,
        "arm": arm,
        "cap": ARMS[arm],
        "verdict": verdict,
        "source_sha": source_sha(root),
        "treatment_sha256": treatment_fingerprint(root),
        "model_config_sha256": sha256_file(cfg),
        "model_config": ARM_CONFIG[arm],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "log": log,
        "notes": notes,
    }


def verify_evidence(kind, arm, root=None, path=None):
    """``(ok, [problems], record_or_None)`` for one evidence file.

    Every check is an EQUALITY against something computed here and now; nothing
    is trusted from the file except as a claim to be refuted. A missing file is a
    failure, not an absence of evidence to be shrugged at.
    """
    root = root or repo_root()
    path = path or evidence_path(kind, arm, root)
    rel = os.path.relpath(path, root)
    if not os.path.isfile(path):
        return False, [f"missing evidence file {rel}"], None
    try:
        with open(path) as f:
            rec = json.load(f)
    except Exception as exc:
        return False, [f"{rel}: not valid JSON ({exc})"], None
    if not isinstance(rec, dict):
        return False, [f"{rel}: evidence must be a JSON object"], None

    problems = []
    if rec.get("schema") != SCHEMA or isinstance(rec.get("schema"), bool):
        problems.append(f"schema {rec.get('schema')!r} != {SCHEMA}")
    if rec.get("kind") != kind:
        problems.append(f"kind {rec.get('kind')!r} != {kind!r}")
    if rec.get("arm") != arm:
        problems.append(f"arm {rec.get('arm')!r} != {arm!r}")
    if rec.get("cap") != ARMS[arm] or isinstance(rec.get("cap"), bool):
        problems.append(f"cap {rec.get('cap')!r} != {ARMS[arm]!r}")
    if rec.get("verdict") != "PASS":
        problems.append(f"verdict {rec.get('verdict')!r} != 'PASS' -> the gate did NOT pass")

    want_treatment = treatment_fingerprint(root)
    if rec.get("treatment_sha256") != want_treatment:
        problems.append(
            f"treatment_sha256 {str(rec.get('treatment_sha256'))[:16]}... != current "
            f"{want_treatment[:16]}... -> the cap-threading code changed since this "
            "evidence was produced, so it describes a different method")
    want_cfg = sha256_file(os.path.join(root, ARM_CONFIG[arm]))
    if rec.get("model_config_sha256") != want_cfg:
        problems.append(
            f"model_config_sha256 {str(rec.get('model_config_sha256'))[:16]}... != current "
            f"{want_cfg[:16]}... -> {ARM_CONFIG[arm]} changed since this evidence was produced")
    try:
        want_sha = source_sha(root)
    except Exception as exc:
        problems.append(f"could not read git HEAD to check source_sha ({exc})")
    else:
        if rec.get("source_sha") != want_sha:
            problems.append(
                f"source_sha {str(rec.get('source_sha'))[:12]}... != HEAD {want_sha[:12]}... "
                "-> plan §5 makes a source-SHA change a hard abort; re-run the gate and re-stamp")
    log = rec.get("log")
    if not isinstance(log, str) or not log:
        problems.append("no 'log' path recorded -> the verdict is unsourced")
    elif not os.path.isfile(os.path.join(root, log)):
        problems.append(f"recorded log {log!r} does not exist -> the verdict is unsourced")

    return (not problems), problems, rec


def require_evidence(arm, root=None):
    """Verify everything ``arm`` needs before a FULL/RESTART launch.

    Returns ``(ok, lines)``; ``lines`` is what the launcher prints, pass or fail.
    """
    root = root or repo_root()
    ok_all, lines = True, []
    for kind, subject in REQUIRED[arm]:
        ok, problems, rec = verify_evidence(kind, subject, root)
        rel = os.path.relpath(evidence_path(kind, subject, root), root)
        if ok:
            lines.append(f"  OK   {kind} ({subject}): {rel}")
            lines.append(f"         verdict={rec['verdict']} cap={rec['cap']} "
                         f"created={rec['created']} log={rec['log']}")
        else:
            ok_all = False
            lines.append(f"  FAIL {kind} ({subject}): {rel}")
            for p in problems:
                lines.append(f"         - {p}")
    if not ok_all:
        lines.append("  -> stamp it with stamp_evidence.py ONLY after the gate has actually "
                     "passed; plan §3.2 (cap fit) and §5 (DS-PA 40k audit) are hard gates.")
    return ok_all, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--verify", action="store_true",
                    help="verify the existing record instead of writing one")
    ap.add_argument("--verdict", default="PASS", choices=["PASS", "FAIL"],
                    help="record a FAIL verdict too: a failed gate is evidence, and the "
                         "launcher rejects it, which is the point")
    ap.add_argument("--log", default="", help="path to the log that produced the verdict")
    ap.add_argument("--notes", default="")
    ap.add_argument("--force", action="store_true", help="overwrite an existing record")
    args = ap.parse_args(argv)

    root = repo_root()
    path = evidence_path(args.kind, args.arm, root)

    if args.verify:
        ok, problems, _ = verify_evidence(args.kind, args.arm, root)
        print(f"verify {args.kind}/{args.arm}: {'OK' if ok else 'FAILED'}")
        for p in problems:
            print(f"  - {p}")
        return 0 if ok else 2

    if not args.log:
        print("--log is REQUIRED when stamping: an unsourced verdict is not evidence")
        return 2
    if not os.path.isfile(os.path.join(root, args.log)):
        print(f"--log {args.log!r} does not exist (paths are relative to {root})")
        return 2
    if os.path.exists(path) and not args.force:
        print(f"{path} already exists; pass --force to overwrite it deliberately")
        return 2

    rec = build_record(args.kind, args.arm, args.verdict, args.log, args.notes, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")
    print(f"stamped {os.path.relpath(path, root)}")
    print(json.dumps(rec, indent=2))
    dirty = dirty_treatment_paths(root)
    if dirty:
        print("\nWARNING: treatment paths are DIRTY at stamping time: " + ", ".join(dirty))
        print("dsarm_launch.sh refuses a FULL/RESTART launch with a dirty treatment path, so "
              "commit them and re-stamp before the campaign starts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
