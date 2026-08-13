#!/usr/bin/env python
"""exp_15 — admission preflight for a CHAIN leg (chain review, finding 1).

exp_11's `fa_orbit_ckpt_preflight.py` cannot admit a chain leg: it requires the
launch manifest's `max_steps` to EQUAL the restarting job's, which is exactly
what a chain changes (leg 2 runs `--max-steps 5000` against an INITIAL manifest
that says 2500), and it re-binds the launch commit, which exp_15's own
content-scoped gate deliberately allows to move. Leg 2 would have been rejected
before Lightning ever started.

So chain legs get their own contract, and it is stricter where it matters:

  * the resume checkpoint must be the LAST AUDITED TIP of the registry's leg
    chain — same step AND same sha256. A chain advances from the boundary it
    actually recorded, not from any checkpoint that happens to sit on disk;
  * the checkpoint is validated with the round-2 recorder's own primitives
    (fd-pinned hash, safe mmap load, embedded-config canonical equality, exact
    EMA mirror), so a leg cannot resume something it could not have produced;
  * the ORIGINAL launch identity is bound from the INITIAL manifest — arm, rung,
    micro-batch, ngpu, training seed, VAE sha, config sha — the things that must
    not drift across a 16-leg run;
  * the per-leg budget is allowed to GROW, but only on the checkpoint cadence and
    only up to the pre-registered cap.

Prints `CKPT_SHA256 <sha>` on success (the launcher greps for it, as it does for
exp_11's helper). Any failure exits non-zero with the reason named.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import sys


def load_recorder(path):
    spec = importlib.util.spec_from_file_location("yaw_aug_record_control", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_fields(text):
    """The launcher's manifest is `key value ...` lines; some carry several pairs."""
    fields = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) >= 3 and parts[0] in ("arm", "job"):
            for key, value in zip(parts[0::2], parts[1::2]):
                fields.setdefault(key, value)
        elif len(parts) >= 2:
            fields.setdefault(parts[0], parts[1])
    return fields


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--expected-step", required=True, type=int)
    ap.add_argument("--target", required=True, type=int)
    ap.add_argument("--cap", required=True, type=int)
    ap.add_argument("--cadence", default=2500, type=int)
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--seed", default="42")
    ap.add_argument("--vae-sha256", required=True)
    ap.add_argument("--launch-manifest", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--recorder", required=True)
    args = ap.parse_args(argv)

    problems = []

    # --- 1. the leg's own arithmetic ---------------------------------------
    if args.cap != 40000:
        problems.append(f"cap {args.cap} is not the pre-registered 40000")
    for name, value in (("expected-step", args.expected_step), ("target", args.target)):
        if value % args.cadence:
            problems.append(f"{name} {value} is not on the {args.cadence}-step cadence")
    if not args.expected_step < args.target <= args.cap:
        problems.append(f"leg {args.expected_step} -> {args.target} does not advance "
                        f"within the cap {args.cap}")
    if problems:
        sys.exit("CHAIN PREFLIGHT: " + "; ".join(problems))

    # --- 2. the registry tip this leg must continue -------------------------
    try:
        with open(args.registry, "rb") as fh:
            registry = json.loads(fh.read())
    except Exception as error:
        sys.exit(f"CHAIN PREFLIGHT: cannot read the launch registry {args.registry}: {error}")
    entry = registry.get("arms", {}).get(args.arm)
    if not entry:
        sys.exit(f"CHAIN PREFLIGHT: {args.arm} has no INITIAL entry in {args.registry}")
    legs = registry.get("legs", {}).get(args.arm, [])
    if not legs:
        sys.exit(f"CHAIN PREFLIGHT: {args.arm} has no audited legs in {args.registry}: "
                 "a chain RESTART must continue an audited boundary")
    tip = legs[-1]
    if int(tip.get("step", -1)) != args.expected_step:
        sys.exit(f"CHAIN PREFLIGHT: the audited tip is step {tip.get('step')}, but this "
                 f"leg resumes {args.expected_step} — the chain would fork")

    # --- 3. the checkpoint itself (round-2 primitives) ----------------------
    rc = load_recorder(args.recorder)
    checkpoint, digest, _identity = rc.snapshot_checkpoint(args.ckpt)

    if digest != tip.get("ckpt_sha256"):
        sys.exit(f"CHAIN PREFLIGHT: {args.ckpt} hashes to {digest} but the audited tip at "
                 f"step {args.expected_step} is {tip.get('ckpt_sha256')} — this is not the "
                 "checkpoint the chain recorded")

    step = checkpoint.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int):
        sys.exit(f"CHAIN PREFLIGHT: embedded global_step is {step!r}, expected a plain int")
    if step != args.expected_step:
        sys.exit(f"CHAIN PREFLIGHT: embedded global_step {step} != --expected-step "
                 f"{args.expected_step}")

    if "model_config" not in checkpoint:
        sys.exit(f"CHAIN PREFLIGHT: {args.ckpt} embeds no model_config")
    with open(args.config, "rb") as fh:
        want_config = json.loads(fh.read())
    if rc.canonical_bytes(checkpoint["model_config"]) != rc.canonical_bytes(want_config):
        sys.exit(f"CHAIN PREFLIGHT: the embedded model_config is not {args.config}")
    if checkpoint["model_config"].get("training", {}).get("yaw_aug") != \
            {"enabled": True, "img_w": 512, "seed": 42}:
        sys.exit("CHAIN PREFLIGHT: the embedded config does not carry the registered yaw_aug block")

    ema = rc.summarize_ema(checkpoint.get("state_dict") or {})

    for required in ("optimizer_states", "lr_schedulers"):
        if not checkpoint.get(required):
            sys.exit(f"CHAIN PREFLIGHT: {args.ckpt} carries no {required} — a warm resume "
                     "needs the optimizer and scheduler state")
    opt = checkpoint["optimizer_states"][0]
    if isinstance(opt, dict) and not opt.get("state"):
        sys.exit(f"CHAIN PREFLIGHT: {args.ckpt}'s optimizer state is CLEARED")

    # --- 4. the ORIGINAL launch identity ------------------------------------
    try:
        with open(args.launch_manifest, "rb") as fh:
            manifest_raw = fh.read()
    except OSError as error:
        sys.exit(f"CHAIN PREFLIGHT: cannot read the INITIAL manifest "
                 f"{args.launch_manifest}: {error.strerror}")
    if entry.get("manifest_sha256") and \
            hashlib.sha256(manifest_raw).hexdigest() != entry["manifest_sha256"]:
        sys.exit(f"CHAIN PREFLIGHT: {args.launch_manifest} no longer hashes to the value "
                 "registered at the INITIAL launch")
    fields = manifest_fields(manifest_raw.decode(errors="replace"))
    config_sha = hashlib.sha256(open(args.config, "rb").read()).hexdigest()
    for label, got, want in (
        ("arm", fields.get("arm"), args.arm),
        ("rung", fields.get("rung"), args.rung),
        ("config_sha256", fields.get("config_sha256"), config_sha),
        ("vae_sha256", fields.get("vae_sha256"), args.vae_sha256),
        ("registry rung", entry.get("rung"), args.rung),
        ("registry training_seed", str(entry.get("training_seed")), str(args.seed)),
        ("registry config_sha256", entry.get("config_sha256"), config_sha),
        ("registry vae_sha256", entry.get("vae_sha256"), args.vae_sha256),
    ):
        if got != want:
            problems.append(f"{label} is {got!r}, the chain's INITIAL identity is {want!r}")
    if not re.search(r"^mode INITIAL|mode INITIAL", manifest_raw.decode(errors="replace"), re.M) \
            and entry.get("mode") != "INITIAL":
        problems.append("the bound manifest is not an INITIAL launch manifest")
    if problems:
        sys.exit("CHAIN PREFLIGHT: " + "; ".join(problems))

    print(f"chain preflight OK: leg {args.expected_step} -> {args.target} of {args.cap}, "
          f"resuming the audited tip")
    print(f"  EMA {ema['ema_key_count']} keys mirror {ema['online_model_key_count']} online "
          f"DiT keys; inventory {ema['ema_inventory_sha256']}")
    print(f"  launch identity bound from {args.launch_manifest}")
    print(f"CKPT_SHA256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
