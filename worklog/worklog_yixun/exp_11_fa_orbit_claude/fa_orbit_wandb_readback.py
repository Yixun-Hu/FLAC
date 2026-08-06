#!/usr/bin/env python3
"""exp_11 — verify WHERE the W&B run actually landed, after training.

The launcher's identity gate proves *who* we authenticate as; this proves *where*
the run was written and that it carries the manifest's identity.

Why the run directory must be SEARCHED FOR rather than assumed: ``train.py``
constructs ``pl.loggers.WandbLogger(project=..., name=...)`` (train.py:165)
without a ``save_dir``, so PL passes its own default ``save_dir='.'`` into
``wandb.init``, and that argument OVERRIDES the exported ``WANDB_DIR``. In the
pinned-path smoke (job 3646734) the run therefore landed in
``$REPO/wandb/run-20260806_164917-exp11-C4L-<run id>`` while the readback looked
under ``$WANDB_DIR/wandb`` and found nothing — training was green but the job
classified 7. The launcher still exports ``WANDB_DIR`` (other wandb artifacts do
respect it), but the readback locates the run by the id WE generated, which wandb
embeds in the directory name, across every candidate root.

Exactly one match is required: zero means the run is not where we think it is,
and more than one means the id is ambiguous — both are provenance failures.
"""
import argparse
import glob
import json
import os
import sys


def locate_run_dir(roots, run_id):
    """Find ``<root>/wandb/run-*-<run_id>`` across ``roots``.

    Returns ``(path, [])`` on exactly one match, else ``(None, [problems])``.
    Roots are searched in order but ALL are collected first, so an id that
    somehow exists under two roots is reported rather than silently preferred."""
    if not run_id:
        return None, ["no run id supplied"]
    matches = []
    for root in roots:
        if not root:
            continue
        matches.extend(sorted(glob.glob(os.path.join(root, "wandb", f"run-*-{run_id}"))))
    matches = sorted(set(matches))
    if not matches:
        return None, [f"no run directory for id {run_id} under any of {list(roots)}"]
    if len(matches) > 1:
        return None, [f"ambiguous run id {run_id}: {matches}"]
    return matches[0], []


def verify_identity(run_dir, run_id, entity=None, project=None, name=None):
    """Check the run directory's embedded id and its wandb-metadata identity."""
    problems = []
    if not run_dir or not os.path.isdir(run_dir):
        return [f"run directory {run_dir!r} does not exist"]
    if not os.path.basename(run_dir).endswith(f"-{run_id}"):
        problems.append(f"run directory {os.path.basename(run_dir)} does not carry id {run_id}")
    meta_path = os.path.join(run_dir, "files", "wandb-metadata.json")
    meta = {}
    if os.path.isfile(meta_path):
        try:
            meta = json.load(open(meta_path))
        except Exception as exc:
            problems.append(f"unreadable {meta_path}: {exc}")
            return problems
    for key, want in (("entity", entity), ("project", project), ("name", name)):
        got = meta.get(key)
        # wandb-metadata does not always carry every field; only a CONTRADICTION
        # is a failure, an absent field is not.
        if want and got is not None and got != want:
            problems.append(f"{key}={got!r} != manifest {want!r}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="verify the created W&B run identity")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--root", action="append", default=[],
                    help="candidate root; repeat. Searched as <root>/wandb/run-*-<id>")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--project", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    run_dir, problems = locate_run_dir(args.root, args.run_id)
    if problems:
        for p in problems:
            print(f"WANDB IDENTITY: {p}")
        return 1
    problems = verify_identity(run_dir, args.run_id, args.entity, args.project, args.name)
    if problems:
        print("WANDB IDENTITY MISMATCH: " + "; ".join(problems))
        return 1
    print(f"wandb run identity OK: id {args.run_id} at {run_dir} "
          f"(entity {args.entity}, project {args.project}, name {args.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
