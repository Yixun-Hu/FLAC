#!/usr/bin/env python3
"""exp_11 RESTART checkpoint preflight (round-3 review B2).

exp_10's `bf_resume_launch.sh` proved a restart with ONE CPU-side ``torch.load``
before spending an allocation; round 3 shipped only a path check, which would
have accepted a zero-byte file or a renamed C4 checkpoint under the C16 root
(orbit size does not change the module tree, so it would even have loaded). This
restores the exp_10 depth for the sweep:

  - the checkpoint's embedded ``global_step`` equals EXPECTED_STEP exactly;
  - its embedded ``model_config`` deep-equals this arm's config file, so the
    orbit, conditioning method and architecture it was trained under are the
    ones this run would continue;
  - full warm optimizer state (non-empty ``state``), ``lr_schedulers``, and EMA
    weights are present — a stripped/weights-only file is the wrong file, since
    exp_11 has no optimizer-reset lineage;
  - the target budget still lies ahead (``global_step < max_steps``), so a
    "restart" cannot terminate immediately on Lightning's ``>=`` stop rule while
    printing the completion literal;
  - optionally, the arm's ORIGINAL launch manifest is re-read and the restart is
    bound to the same rung, commit and budget.

Prints the checkpoint sha256 (for the restart manifest) and a lineage summary.
Exit 0 = admissible; nonzero = refuse to launch.
"""
import argparse
import hashlib
import json
import os
import sys


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ckpt_config(path):
    """The model_config embedded in a Lightning checkpoint, on CPU.

    Shared with the screen driver (fa_orbit_screen.sbatch), which asserts the
    checkpoint's own orbit before spending an evaluation on it: a screen that
    silently evaluated the wrong arm's checkpoint would poison a futility gate."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise RuntimeError(f"not a Lightning checkpoint: {path}")
    cfg = ck.get("model_config")
    if not isinstance(cfg, dict):
        raise RuntimeError(f"checkpoint carries no embedded model_config: {path}")
    return cfg, ck.get("global_step")


def load_ckpt_state_keys(path):
    """The checkpoint's state_dict KEYS (used to prove EMA weights exist before a
    screen spends a GPU: eval_FLAC silently evaluates online weights when the EMA
    entries are absent)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise RuntimeError(f"not a Lightning checkpoint: {path}")
    return list((ck.get("state_dict") or {}).keys())


def parse_manifest(path):
    """The launcher's own manifest format: whitespace-separated `key value...`."""
    out = {}
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, rest = line.partition(" ")
            out[key] = rest.strip()
    return out


def check_manifest_binding(manifest_path, arm, rung, commit, maxsteps):
    man = parse_manifest(manifest_path)
    problems = []
    fields = man.get("arm", "")
    # `arm <ARM> rung <RUNG> micro <MB> ngpu <N> max_steps <S> ...`
    tokens = ("arm " + fields).split()
    kv = {tokens[i]: tokens[i + 1] for i in range(0, len(tokens) - 1, 2)}
    if kv.get("arm") != arm:
        problems.append(f"manifest arm {kv.get('arm')!r} != {arm!r}")
    if kv.get("rung") != rung:
        problems.append(f"manifest rung {kv.get('rung')!r} != {rung!r} "
                        "(a restart may not change the rung: it would change rank count, "
                        "sampler partitioning and worker seeding mid-lineage)")
    if kv.get("max_steps") != str(maxsteps):
        problems.append(f"manifest max_steps {kv.get('max_steps')!r} != {maxsteps}")
    # Fail-CLOSED (round-3 B2 residual): an absent or empty manifest commit is not
    # "no opinion", it is missing provenance — the restart must not proceed on it.
    man_commit = man.get("commit", "").strip()
    if not man_commit:
        problems.append("launch manifest carries no 'commit' line — cannot bind the restart "
                        "to the lineage that produced this checkpoint")
    elif not commit:
        problems.append("no running commit supplied to compare against the manifest commit")
    elif man_commit != commit:
        problems.append(f"manifest commit {man_commit[:12]} != running commit {commit[:12]}")
    return problems, man


def main(argv=None):
    ap = argparse.ArgumentParser(description="exp_11 restart checkpoint preflight")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--expected-step", type=int, required=True)
    ap.add_argument("--config", required=True, help="this arm's model config json")
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--commit", default="")
    ap.add_argument("--launch-manifest", default="",
                    help="the arm's original launch manifest (binds rung/commit/budget)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.ckpt):
        print(f"PREFLIGHT: checkpoint not found: {args.ckpt}")
        return 2

    import torch  # deferred: keeps --help and unit imports cheap

    try:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:                      # truncated, empty or not a torch file
        print(f"PREFLIGHT: {args.ckpt} is not loadable as a checkpoint: "
              f"{type(exc).__name__}: {exc}")
        return 2
    if not isinstance(ck, dict):
        print(f"PREFLIGHT: not a Lightning checkpoint: {args.ckpt}")
        return 2

    problems = []
    gs = ck.get("global_step")
    if gs != args.expected_step:
        problems.append(f"global_step {gs} != EXPECTED_STEP {args.expected_step}")
    if isinstance(gs, int) and gs >= args.max_steps:
        problems.append(f"global_step {gs} >= max_steps {args.max_steps}: Lightning would stop "
                        "immediately and still print the completion literal")

    mc = ck.get("model_config")
    if not isinstance(mc, dict):
        problems.append("checkpoint carries no embedded 'model_config' dict")
    else:
        want = json.load(open(args.config))
        if mc != want:
            tr = mc.get("training", {}) if isinstance(mc.get("training"), dict) else {}
            problems.append(
                f"embedded model_config != {args.config} (parsed-object mismatch; embedded "
                f"cond_method={tr.get('cond_method')!r} angles={tr.get('frame_avg_angles')!r})")

    opts = ck.get("optimizer_states") or []
    if len(opts) != 1:
        problems.append(f"expected exactly 1 optimizer entry, found {len(opts)}")
    elif not opts[0].get("state"):
        problems.append("optimizer state is CLEARED (stripped checkpoint); exp_11 restarts are "
                        "WARM continuations and have no optimizer-reset lineage")
    if not ck.get("lr_schedulers"):
        problems.append("no 'lr_schedulers' -> PL 2.1 KeyErrors on resume")
    sd = ck.get("state_dict") or {}
    n_ema = sum(1 for k in sd if k.startswith("diffusion_ema."))
    if not n_ema:
        problems.append("no EMA weights in state_dict")

    man = {}
    if args.launch_manifest:
        if not os.path.isfile(args.launch_manifest):
            problems.append(f"launch manifest not found: {args.launch_manifest}")
        else:
            more, man = check_manifest_binding(args.launch_manifest, args.arm, args.rung,
                                               args.commit, args.max_steps)
            problems += more

    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print(f"  !! {p}")
        return 2

    digest = sha256_file(args.ckpt)
    tr = mc.get("training", {})
    print(f"restart lineage OK: {args.ckpt}")
    print(f"  global_step={gs} epoch={ck.get('epoch')} cond_method={tr.get('cond_method')!r} "
          f"angles={tr.get('frame_avg_angles')}")
    print(f"  optimizer_state=FULL ({len(opts[0]['state'])} entries) "
          f"lr={opts[0]['param_groups'][0].get('lr')} "
          f"sched_last_epoch={ck['lr_schedulers'][0].get('last_epoch')} ema_entries={n_ema}")
    if man:
        print(f"  bound to launch manifest: {args.launch_manifest}")
    print(f"CKPT_SHA256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
