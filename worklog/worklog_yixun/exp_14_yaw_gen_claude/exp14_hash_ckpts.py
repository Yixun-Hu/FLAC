#!/usr/bin/env python3
"""exp_14 — publish the audited per-arm checkpoint expectation (review B4).

Dedup used to accept a landed cell without ever checking WHICH checkpoint it
evaluated: `classify` supplied no expected digest, so that check silently did not
run and the wave skipped the cell as measured. A skip is a decision to keep a
number, so it has to rest on the same identity evidence a fresh run would give.

This helper writes `exp14_ckpt_expect.json`: arm -> {path, sha256, bytes, source}
for the ONE 40,000-step checkpoint of each arm. It is committed, and thereafter
read (never recomputed) by the submitter and the collector — hashing five 724 MB
files on a shared login node is a thing to do once, deliberately, not per wave.

Two sources, and the overlap is CROSS-CHECKED rather than trusted:

* exp_11's audited `arm_launch_registry.json` already records `final_ckpt_sha256`
  at `final_step` 40000 for C4L/C8/C16/C32 (written by its own recorder from the
  files on disk). Those values are re-hashed here and must agree exactly; a
  disagreement is a hard error, because it would mean the file on disk is not the
  one exp_11 audited.
* VANL has no registry entry (it was launched later, for the Q9 round), so its
  digest is established here for the first time and labelled as such.

Usage (once, deliberately — reads ~3.6 GB):
    python3 exp14_hash_ckpts.py --write
    python3 exp14_hash_ckpts.py --verify     # re-hash and compare, writes nothing
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
EXP11 = os.path.join(REPO, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude")
REGISTRY = os.path.join(EXP11, "arm_launch_registry.json")
EXPECT = os.path.join(HERE, "exp14_ckpt_expect.json")
ARMS = ("VANL", "C4L", "C8", "C16", "C32")
STEP = 40000


def checkpoint_path(output_root, arm, step=STEP):
    """The arm's checkpoint at ``step``; a non-unique match is a hard error."""
    import glob
    pat = os.path.join(output_root, f"exp11_{arm}", f"FLAC_exp11_{arm}",
                       f"exp11_{arm}", "checkpoints", f"epoch=*-step={step}.ckpt")
    hits = sorted(glob.glob(pat))
    if len(hits) != 1:
        raise SystemExit(f"{arm}: expected exactly 1 checkpoint at step {step}, found {len(hits)}")
    return hits[0]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def registry_digests():
    """``arm -> sha256`` for the arms exp_11 already audited at the 40k endpoint."""
    reg = json.load(open(REGISTRY))["arms"]
    out = {}
    for arm, row in reg.items():
        if row.get("final_ckpt_sha256") and int(row.get("final_step", -1)) == STEP:
            out[arm] = row["final_ckpt_sha256"]
    return out


def build(output_root, arms=ARMS):
    """Hash every arm's 40k checkpoint and cross-check the audited overlap."""
    audited = registry_digests()
    entries, disagreements = {}, []
    for arm in arms:
        path = checkpoint_path(output_root, arm)
        digest = sha256_file(path)
        want = audited.get(arm)
        if want is not None and want != digest:
            disagreements.append(
                f"{arm}: on-disk sha256 {digest[:12]} != exp_11's audited "
                f"final_ckpt_sha256 {want[:12]}")
        entries[arm] = {
            "path": os.path.relpath(path, REPO),
            "sha256": digest,
            "bytes": os.path.getsize(path),
            "step": STEP,
            # Where the CLAIM comes from, not just the number: a digest this file
            # established alone is weaker evidence than one that reproduces an
            # independently audited value, and a reader must be able to tell.
            "source": "exp_11 arm_launch_registry.final_ckpt_sha256 (re-hashed, agrees)"
                      if want is not None else
                      "hashed by exp14_hash_ckpts.py (no exp_11 registry entry)",
        }
    if disagreements:
        raise SystemExit("CHECKPOINT IDENTITY DISAGREEMENT:\n  " + "\n  ".join(disagreements))
    return {
        "_comment": [
            "AUDITED exp_14 checkpoint expectation: the ONE 40,000-step checkpoint",
            "each arm is evaluated from. Read (never recomputed) by",
            "exp14_validate_cell.py classify and by the collector, so a dedup SKIP",
            "rests on checkpoint identity and not merely on a file's existence",
            "(review B4). Regenerate only with exp14_hash_ckpts.py --write.",
        ],
        "step": STEP,
        "arms": entries,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-root", default=os.path.join(REPO, "outputs_FLAC"))
    ap.add_argument("--write", action="store_true", help="publish exp14_ckpt_expect.json")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash and compare against the published file; write nothing")
    ap.add_argument("--out", default=EXPECT)
    args = ap.parse_args(argv)
    if not (args.write or args.verify):
        ap.error("choose --write or --verify")
    built = build(args.output_root)
    if args.verify:
        have = json.load(open(args.out))
        bad = [f"{a}: {v['sha256'][:12]} != published {have['arms'].get(a, {}).get('sha256', '<none>')[:12]}"
               for a, v in built["arms"].items()
               if have["arms"].get(a, {}).get("sha256") != v["sha256"]]
        if bad:
            print("MISMATCH:\n  " + "\n  ".join(bad), file=sys.stderr)
            return 1
        print(f"verified {len(built['arms'])} arms against {args.out}")
        return 0
    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(built, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, args.out)                      # atomic publish
    for arm, row in built["arms"].items():
        print(f"{arm} {row['sha256'][:16]} {row['bytes']} {row['source']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
