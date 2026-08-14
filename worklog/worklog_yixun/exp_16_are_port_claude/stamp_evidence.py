#!/usr/bin/env python3
"""exp_16 (are_port) admission-evidence contract: stamp it, and verify it.

Seat: Opus 5 Coder (SOP §Roles). Same shape as exp_14's
``fa_drawshare/stamp_evidence.py``, retargeted at exp_16's single arm.

WHY THIS EXISTS. Plan §4 gates the ARE-V launch on a real fit/plumbing probe:
the anchor path adds a per-sample B=1 VAE encode inside ``training_step``, and
the campaign's standing co-tenancy VRAM floor was measured on a DIFFERENT arm.
"Run a probe first" is prose, and prose does not stop a launch at 03:00 — so the
gate is discharged by a FILE this module writes and ``are_launch.sh`` verifies
before it will start a FULL (or RESTART) run. An unstamped gate is a hard abort.

WHAT IS **NOT** HERE. exp_14 additionally required a cross-arm sequencing record
(``dspa_40k_audit`` before DS-CS3). exp_16 Phase 1 is a SINGLE arm, so there is
no cross-arm requirement to encode, and inventing one would be a gate with no
preregistered content. ``REQUIRED`` therefore has exactly one entry.

THE BIND. An evidence file is only meaningful for the code it was produced under:

  * ``treatment_sha256``  - the WORKING-TREE bytes of every file that can change
    what the ARE objective IS (the anchor module, the two dispatch sites, the
    factory that parses lambda, and train.py). If any of them changed, an earlier
    fit probe describes a different method.
  * ``model_config_sha256`` - the arm JSON the evidence is about. This one carries
    the CALIBRATED CONSTANTS, so a re-calibration invalidates the probe by
    construction, which is correct: a different delta_hat/A_g is a different anchor.
  * ``calibration_sha256`` - the calibration RECORD those constants came from, so
    the arm's provenance (cohort, seed, estimator, escalation flags) is bound too
    and not merely the two numbers it produced.
  * ``vae_sha256`` - the frozen VAE the anchor is encoded through. ``A`` is
    ``Enc(skel) - Enc(0)``; a different codec is a different anchor, and nothing
    else in the record would notice.
  * ``source_sha`` - ``git rev-parse HEAD`` at stamping time, re-checked at launch.

Usage (stamp, AFTER the probe has actually passed - this file records a verdict,
it does not compute one):

  python worklog/worklog_yixun/exp_16_are_port_claude/stamp_evidence.py \
      --kind are_fit --arm AREV --verdict PASS \
      --log worklog/worklog_yixun/exp_16_are_port_claude/are_port_<ts>_exp16_AREV_probe_train.log \
      --notes "15-step DDP fit at micro-32x2, peak <X> GiB/rank, <Y> steps/s"

Usage (verify by hand):

  python .../stamp_evidence.py --verify --kind are_fit --arm AREV
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

SCHEMA = 1
KINDS = ("are_fit",)
# arm -> its declared training.are_lambda
ARMS = {"AREV": 1.0}

# Files whose contents can change what the ARE objective IS, or what the launch
# actually does. Order is part of the hash: a stable order makes the digest
# reproducible across machines.
#
# ``src/data/{dataset,utils}.py`` are here because the loader's RandomTimeShift
# draw is what the anchor's ``t*`` is corrected by: change how (or whether) that
# draw is published and the anchor silently mis-places itself on half the
# training set, with nothing else in the record to show for it.
#
# ``AR_md.py`` defines what ``source`` and ``depth`` MEAN (the listener frame and
# the panorama convention), which is the entire input to ``r`` and to the LOS
# gate.
#
# The launcher, its schedule/readback module, the dataset config and the train
# split are here per r1 review finding 3: a dirty edit to any of them changes the
# run -- its endpoint, its cadence, its data -- without touching a line of
# ``src/``, and round 1's fingerprint would not have noticed.
TREATMENT_PATHS = (
    "data/AR/train.json",
    "src/configs/dataset_configs/AR/train/acousticroom_train.json",
    "src/configs/dataset_configs/custom_metadata/AR_md.py",
    "src/data/are_anchor.py",
    "src/data/dataset.py",
    "src/data/utils.py",
    "src/training/diffusion.py",
    "src/training/factory.py",
    "train.py",
    "worklog/worklog_yixun/exp_16_are_port_claude/are_launch.sh",
    "worklog/worklog_yixun/exp_16_are_port_claude/readback.py",
)

EXPDIR = "worklog/worklog_yixun/exp_16_are_port_claude"
EVIDENCE_DIRNAME = "evidence"
ARM_CONFIG = {"AREV": f"{EXPDIR}/FLAC_AR_ARE.json"}
# Non-source assets the anchor is a function of. Both are hashed into every
# evidence record and re-checked at launch (54 MB + a few KB: ~0.3 s).
CALIBRATION_PATH = f"{EXPDIR}/are_calibration.json"
VAE_PATH = "weights/FLAC/VAE.safetensors"

# Which evidence a given arm's admission requires. One arm, one record.
REQUIRED = {
    "AREV": (("are_fit", "AREV"),),
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
    these files cannot leave the digest unchanged. The WORKING TREE - not the
    committed blob - is hashed on purpose: the launcher runs what is on disk.
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
    return [line[3:].strip() for line in out.decode().splitlines() if line.strip()]


def evidence_path(kind, arm, root=None):
    root = root or repo_root()
    return os.path.join(root, EXPDIR, EVIDENCE_DIRNAME, f"{kind}_{arm}.json")


def build_record(kind, arm, verdict, log, notes, root=None):
    root = root or repo_root()
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {list(KINDS)}, got {kind!r}")
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}, got {arm!r}")
    cfg = os.path.join(root, ARM_CONFIG[arm])
    return {
        "schema": SCHEMA,
        "kind": kind,
        "arm": arm,
        "are_lambda": ARMS[arm],
        "verdict": verdict,
        "source_sha": source_sha(root),
        "treatment_sha256": treatment_fingerprint(root),
        "model_config_sha256": sha256_file(cfg),
        "model_config": ARM_CONFIG[arm],
        "calibration_sha256": sha256_file(os.path.join(root, CALIBRATION_PATH)),
        "vae_sha256": sha256_file(os.path.join(root, VAE_PATH)),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "log": log,
        "notes": notes,
    }


def verify_evidence(kind, arm, root=None, path=None):
    """``(ok, [problems], record_or_None)`` for one evidence file.

    Every check is an EQUALITY against something computed here and now; nothing is
    trusted from the file except as a claim to be refuted. A missing file is a
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
    if rec.get("are_lambda") != ARMS[arm] or isinstance(rec.get("are_lambda"), bool):
        problems.append(f"are_lambda {rec.get('are_lambda')!r} != {ARMS[arm]!r}")
    if rec.get("verdict") != "PASS":
        problems.append(f"verdict {rec.get('verdict')!r} != 'PASS' -> the gate did NOT pass")

    want_treatment = treatment_fingerprint(root)
    if rec.get("treatment_sha256") != want_treatment:
        problems.append(
            f"treatment_sha256 {str(rec.get('treatment_sha256'))[:16]}... != current "
            f"{want_treatment[:16]}... -> the ARE code changed since this evidence was "
            "produced, so it describes a different method")
    want_cfg = sha256_file(os.path.join(root, ARM_CONFIG[arm]))
    if rec.get("model_config_sha256") != want_cfg:
        problems.append(
            f"model_config_sha256 {str(rec.get('model_config_sha256'))[:16]}... != current "
            f"{want_cfg[:16]}... -> {ARM_CONFIG[arm]} changed since this evidence was "
            "produced (a re-calibrated delta_hat/A_g is a different anchor)")
    for field, rel, why in (
        ("calibration_sha256", CALIBRATION_PATH,
         "the calibration record changed -> the constants' provenance no longer matches"),
        ("vae_sha256", VAE_PATH,
         "the frozen VAE changed -> Enc(skel) - Enc(0) is a different anchor"),
    ):
        target = os.path.join(root, rel)
        if not os.path.isfile(target):
            problems.append(f"{rel} is missing; {field} cannot be verified")
            continue
        want = sha256_file(target)
        if rec.get(field) != want:
            problems.append(
                f"{field} {str(rec.get(field))[:16]}... != current {want[:16]}... -> {why}")
    try:
        want_sha = source_sha(root)
    except Exception as exc:
        problems.append(f"could not read git HEAD to check source_sha ({exc})")
    else:
        if rec.get("source_sha") != want_sha:
            problems.append(
                f"source_sha {str(rec.get('source_sha'))[:12]}... != HEAD {want_sha[:12]}... "
                "-> a source change since the probe is a hard abort; re-run it and re-stamp")
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
            lines.append(f"         verdict={rec['verdict']} are_lambda={rec['are_lambda']} "
                         f"created={rec['created']} log={rec['log']}")
        else:
            ok_all = False
            lines.append(f"  FAIL {kind} ({subject}): {rel}")
            for p in problems:
                lines.append(f"         - {p}")
    if not ok_all:
        lines.append("  -> stamp it with stamp_evidence.py ONLY after the probe has actually "
                     "passed; plan §4 makes the fit probe a hard gate.")
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
        print("are_launch.sh refuses a FULL/RESTART launch with a dirty treatment path, so "
              "commit them and re-stamp before the campaign starts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
