#!/usr/bin/env python3
"""Record an arm's AUDITED final checkpoint (the anchor) in arm_launch_registry.json.

    python3 fa_orbit_add_anchor.py C32
    python3 fa_orbit_add_anchor.py C32 --dry-run

The anchor -- ``final_ckpt_sha256`` + ``final_step`` -- is what every later leg
chains to: the restart preflight refuses an extension whose resume checkpoint is
not this hash, the recorder refuses a leg that does not resume it, and the screen
refuses a >40k checkpoint whose leg does not descend from it. It is therefore
written with the same rigor as a leg (fa_orbit_record_restart.py), never by hand:

  * the arm's INITIAL launch manifest must still hash to the value the registry
    audited, and every identity field in it must equal the registry row -- so the
    anchor is added to a launch whose provenance is still intact, not to whatever
    the mutable manifest under gitignored outputs_FLAC says now;
  * the checkpoint is located by the registry's own save-dir (canonical run
    directory), must be the EXACTLY ONE file at the registered final step, and is
    re-hashed from disk;
  * it is opened once and audited: embedded ``global_step`` == the registered
    budget, embedded ``model_config`` deep-equals the arm's config, warm optimizer
    state, ``lr_schedulers`` and EMA weights all present -- a stripped or foreign
    file is not an anchor;
  * publication is tmp+rename under the same exclusive lock the recorder takes;
  * an arm that already carries a DIFFERENT anchor is refused (re-anchoring would
    silently re-parent every leg already chained to the old one). Re-running with
    the same checkpoint is a no-op.
"""
import argparse
import fcntl
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fa_orbit_producer_manifest as pm                   # noqa: E402
from fa_orbit_ckpt_preflight import canonical_ckpt_dir    # noqa: E402
from fa_orbit_record_restart import kvs, parse_manifest, read_pins, resolve   # noqa: E402


def check_launch_chain(arm, row, pins, repo_root):
    """The INITIAL manifest, byte-for-byte as audited, field for field as recorded."""
    problems = []
    man_path = resolve(repo_root, str(row.get("manifest_path", "")))
    if not row.get("manifest_path") or not os.path.isfile(man_path):
        return [f"the registered INITIAL launch manifest {man_path} does not exist — the anchor "
                "cannot be bound to the launch that produced it"], {}
    raw, man = parse_manifest(man_path)
    got = hashlib.sha256(raw).hexdigest()
    if got != row.get("manifest_sha256"):
        problems.append(f"launch manifest sha256 {got[:12]} != audited "
                        f"{str(row.get('manifest_sha256'))[:12]} — it changed after registration")
    jk, ak = kvs(man, "job"), kvs(man, "arm")
    for label, a, b in (("arm", ak.get("arm"), arm),
                        ("job", jk.get("job"), row.get("job")),
                        ("mode", jk.get("mode"), "INITIAL"),
                        ("launch_uuid", jk.get("launch_uuid"), row.get("launch_uuid")),
                        ("commit", man.get("commit"), row.get("commit")),
                        ("rung", ak.get("rung"), row.get("rung")),
                        ("max_steps", ak.get("max_steps"), row.get("max_steps")),
                        ("config_sha256", man.get("config_sha256"), row.get("config_sha256")),
                        ("vae_sha256", man.get("vae_sha256"), row.get("vae_sha256")),
                        ("p0_manifest_sha256", man.get("p0_manifest_sha256"),
                         row.get("p0_manifest_sha256")),
                        ("save_dir", man.get("save_dir"), row.get("save_dir"))):
        if a != b:
            problems.append(f"manifest {label} {a!r} != the registered {b!r}")
    if ak.get("rung") != pins.get("PINNED_RUNG"):
        problems.append(f"manifest rung {ak.get('rung')!r} != the pinned {pins.get('PINNED_RUNG')!r}")
    if man.get("vae_sha256") != pins.get("PINNED_VAE_SHA256"):
        problems.append("manifest vae_sha256 != the launcher's PINNED_VAE_SHA256")
    if int(row.get("training_seed", -1)) != 42:
        problems.append(f"registered training seed {row.get('training_seed')!r} != 42")
    cfg = man.get("model_config", "")
    if not cfg or not os.path.isfile(cfg):
        problems.append(f"manifest model_config {cfg!r} does not exist")
    elif hashlib.sha256(open(cfg, "rb").read()).hexdigest() != row.get("config_sha256"):
        problems.append(f"{cfg} no longer hashes to the audited config_sha256")
    return problems, man


def find_final_ckpt(row, arm, step, repo_root):
    """The EXACTLY ONE checkpoint at the registered final step, in the canonical dir."""
    ckpt_dir = canonical_ckpt_dir(str(row.get("save_dir", "")), arm, repo_root)
    hits = sorted(glob.glob(os.path.join(ckpt_dir, f"*-step={step}.ckpt")))
    if len(hits) != 1:
        return None, [f"expected exactly 1 checkpoint at step {step} in {ckpt_dir}, found "
                      f"{len(hits)}{': ' + ', '.join(os.path.basename(h) for h in hits) if hits else ''}"]
    return hits[0], []


def audit_ckpt(path, step, config_path):
    """One torch.load: this file IS that run's finished checkpoint, warm and EMA-bearing."""
    import torch                      # deferred: --help and unit imports stay cheap
    problems = []
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return [f"{path} is not loadable as a checkpoint: {type(exc).__name__}: {exc}"]
    if not isinstance(ck, dict):
        return [f"not a Lightning checkpoint: {path}"]
    if ck.get("global_step") != step:
        problems.append(f"embedded global_step {ck.get('global_step')} != the registered final "
                        f"step {step}")
    mc = ck.get("model_config")
    if not isinstance(mc, dict):
        problems.append("checkpoint carries no embedded 'model_config' dict")
    elif mc != json.load(open(config_path)):
        problems.append(f"embedded model_config != {config_path} (parsed-object mismatch)")
    opts = ck.get("optimizer_states") or []
    if len(opts) != 1 or not opts[0].get("state"):
        problems.append("optimizer state is missing or CLEARED — a stripped file is not the "
                        "checkpoint a restart would resume")
    if not ck.get("lr_schedulers"):
        problems.append("no 'lr_schedulers' — PL 2.1 would KeyError on resume from this file")
    sd = ck.get("state_dict") or {}
    if not any(str(k).startswith("diffusion_ema.") for k in sd):
        problems.append("no EMA weights in state_dict")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="record an exp_11 arm's audited final checkpoint")
    ap.add_argument("arm")
    ap.add_argument("--step", type=int, default=None,
                    help="cross-check only; must equal the arm's registered max_steps")
    ap.add_argument("--registry", default=os.path.join(HERE, "arm_launch_registry.json"))
    ap.add_argument("--launcher", default=os.path.join(HERE, "fa_orbit_train.sbatch"))
    # HERE = <repo>/worklog/worklog_<user>/exp_11_fa_orbit_claude
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    ap.add_argument("--dry-run", action="store_true", help="audit and report, publish nothing")
    args = ap.parse_args(argv)

    pins = read_pins(args.launcher)
    if not pins.get("PINNED_RUNG"):
        raise SystemExit(f"no PINNED_* values found in {args.launcher}")
    store = os.path.dirname(os.path.abspath(args.registry)) or "."
    lock_fd = os.open(store, os.O_RDONLY)     # the recorder's lock: one writer, one store
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        return anchor(args, pins)
    finally:
        os.close(lock_fd)


def anchor(args, pins):
    arm = args.arm
    reg = json.load(open(args.registry))
    row = reg.get("arms", {}).get(arm)
    if row is None:
        raise SystemExit(f"{arm} has no INITIAL registry entry — an anchor belongs to a launch")

    try:
        step = int(row["max_steps"])
    except (KeyError, TypeError, ValueError):
        raise SystemExit(f"{arm}'s registry row has no integer max_steps to anchor at")
    if args.step is not None and args.step != step:
        raise SystemExit(f"--step {args.step} != {arm}'s registered budget {step}: the anchor is "
                         "the run's FINAL checkpoint, not an arbitrary one")

    problems, man = check_launch_chain(arm, row, pins, args.repo_root)
    ckpt, more = find_final_ckpt(row, arm, step, args.repo_root)
    problems += more
    # Only load the checkpoint once the chain itself is sound: a broken chain has
    # already refused the run, and a 724 MB read should not be spent proving it
    # twice. (check_launch_chain has already flagged a missing/unhashable config,
    # so reaching here means model_config exists and is the audited one.)
    if ckpt and not problems:
        problems += audit_ckpt(ckpt, step, man["model_config"])
    if problems:
        print("ANCHOR REFUSED:")
        for p in problems:
            print(f"  !! {p}")
        return 2

    digest = pm.sha256_file(ckpt)
    old = row.get("final_ckpt_sha256")
    if old and old != digest:
        print("ANCHOR REFUSED:")
        print(f"  !! {arm} is already anchored at {old[:12]} and this checkpoint hashes "
              f"{digest[:12]} — re-anchoring would silently re-parent every leg already chained "
              "to the old anchor")
        return 2
    if old == digest and str(row.get("final_step")) == str(step):
        print(f"{arm} is already anchored at {digest[:12]} (step {step}) — unchanged")
        return 0

    row["final_ckpt_sha256"] = digest
    row["final_step"] = step
    row["final_ckpt_path"] = pm.rel_to(args.repo_root, ckpt)
    row["anchored_at"] = pm._now()
    if not args.dry_run:
        pm.write_atomic(args.registry, reg)
    print(f"anchored {arm} at step {step}: {digest}")
    print(f"  {row['final_ckpt_path']} (launch job {row.get('job')}, commit "
          f"{str(row.get('commit'))[:12]})"
          + ("  [dry run, nothing written]" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
