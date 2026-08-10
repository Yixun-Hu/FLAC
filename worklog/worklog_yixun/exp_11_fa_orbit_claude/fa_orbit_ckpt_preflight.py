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


def kv_line(man, key):
    """One manifest line's `k v k v ...` pairs (the launcher's `arm ...`/`job ...`)."""
    f = (f"{key} " + man.get(key, "")).split()
    return {f[i]: f[i + 1] for i in range(0, len(f) - 1, 2)}


def canonical_ckpt_dir(save_dir, arm, repo_root):
    """<save_dir>/FLAC_exp11_<ARM>/exp11_<ARM>/checkpoints, as the launcher builds it.

    save_dir is recorded relative to the repo root, so it is resolved against it."""
    base = save_dir if os.path.isabs(save_dir) else os.path.join(repo_root, save_dir)
    return os.path.realpath(os.path.join(base, f"FLAC_exp11_{arm}", f"exp11_{arm}", "checkpoints"))


def check_extension_binding(manifest_path, registry_path, arm, rung, config_path, ckpt_path,
                            ckpt_sha, expected_step, max_steps, repo_root="."):
    """The 40k -> 100k EXTENSION contract (re-pin review, required fix 1).

    A crash restart continues the SAME launch: same budget, same reviewed commit,
    so `check_manifest_binding` demands both. An extension breaks both BY DESIGN
    — it raises the budget from 40000 to 100000 and runs later reviewed code —
    and demanding equality there is exactly what gave jobs 3662828-30 their third
    hard-abort path.

    What an extension must still prove is the ORIGINAL LAUNCH IDENTITY, and it
    proves it against the COMMITTED registry rather than the mutable manifest
    alone: the INITIAL manifest byte-for-byte as audited, the same job/uuid/
    launch commit/rung/config/save-dir/training seed, and a resumed checkpoint
    that IS that launch's audited final checkpoint, sitting in that launch's own
    canonical run directory. Budget and running commit may move; nothing that
    identifies the run may.
    """
    problems = []
    if not os.path.isfile(registry_path):
        return [f"audited launch registry not found: {registry_path}"], {}
    reg = json.load(open(registry_path)).get("arms", {}).get(arm)
    if reg is None:
        return [f"{arm} is not in the audited launch registry {registry_path}"], {}
    man = parse_manifest(manifest_path)
    kv, jkv = kv_line(man, "arm"), kv_line(man, "job")

    got_sha = sha256_file(manifest_path)
    if got_sha != reg.get("manifest_sha256"):
        problems.append(f"launch manifest sha256 {got_sha[:12]} != audited "
                        f"{str(reg.get('manifest_sha256'))[:12]} — the manifest changed after it "
                        "was registered")
    for label, got_v, want_v in (("arm", kv.get("arm"), arm),
                                 ("job", jkv.get("job"), reg.get("job")),
                                 ("launch mode", jkv.get("mode"), "INITIAL"),
                                 ("launch_uuid", jkv.get("launch_uuid"), reg.get("launch_uuid")),
                                 ("rung", kv.get("rung"), reg.get("rung")),
                                 ("rung (this run)", rung, reg.get("rung")),
                                 ("config_sha256", man.get("config_sha256"), reg.get("config_sha256")),
                                 ("save_dir", man.get("save_dir"), reg.get("save_dir"))):
        if got_v != want_v:
            problems.append(f"{label} {got_v!r} != registered {want_v!r}")
    man_commit = man.get("commit", "").strip()
    if not man_commit:
        problems.append("launch manifest carries no 'commit' line — cannot bind the extension to "
                        "the lineage that produced this checkpoint")
    elif man_commit != reg.get("commit"):
        problems.append(f"manifest commit {man_commit[:12]} != the registered launch commit "
                        f"{str(reg.get('commit'))[:12]}")
    if int(reg.get("training_seed", -1)) != 42:
        problems.append(f"registered training seed {reg.get('training_seed')!r} != 42")
    # The INITIAL budget is the manifest's and the registry's; the extension's is
    # this run's, and it must strictly cover the resume point without shrinking.
    initial_budget = reg.get("max_steps")
    if kv.get("max_steps") != initial_budget:
        problems.append(f"manifest max_steps {kv.get('max_steps')!r} != registered "
                        f"{initial_budget!r} (the INITIAL budget, which an extension preserves)")
    try:
        if max_steps < int(initial_budget):
            problems.append(f"extension budget {max_steps} does not extend the registered "
                            f"{initial_budget} — an extension may only raise the budget")
    except (TypeError, ValueError):
        problems.append(f"registered max_steps {initial_budget!r} is not an integer")
    if sha256_file(config_path) != reg.get("config_sha256"):
        problems.append(f"{config_path} sha256 != the registered config_sha256 "
                        f"{str(reg.get('config_sha256'))[:12]}")
    # the resumed checkpoint IS the audited anchor, in the audited run directory
    anchor, final_step = reg.get("final_ckpt_sha256"), reg.get("final_step")
    if not anchor:
        problems.append(f"{arm} has no audited final_ckpt_sha256 in the registry — the extension "
                        "has nothing to chain to (audit the arm's final checkpoint first)")
    elif ckpt_sha != anchor:
        problems.append(f"resume checkpoint sha256 {ckpt_sha[:12]} != the audited final checkpoint "
                        f"{anchor[:12]} — this leg does not continue that run")
    if final_step is not None and int(final_step) != int(expected_step):
        problems.append(f"EXPECTED_STEP {expected_step} != the registered final_step {final_step}")
    save_dir = man.get("save_dir", "")
    if not save_dir:
        problems.append("manifest records no save_dir")
    else:
        canon = canonical_ckpt_dir(save_dir, arm, repo_root)
        if os.path.realpath(os.path.dirname(ckpt_path)) != canon:
            problems.append(f"resume checkpoint {ckpt_path} does not live in the registered "
                            f"launch's canonical run directory {canon}")
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
    ap.add_argument("--extension", action="store_true",
                    help="this restart is the Q10 40k->100k EXTENSION: bind it to the audited "
                         "INITIAL launch identity instead of requiring an equal budget/commit")
    ap.add_argument("--launch-registry", default="",
                    help="the committed arm launch registry (required with --extension)")
    ap.add_argument("--repo-root", default=".",
                    help="root the registry's relative save_dir is resolved against")
    args = ap.parse_args(argv)
    if args.extension and not args.launch_registry:
        ap.error("--extension requires --launch-registry (the audited INITIAL launch row)")

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

    digest = sha256_file(args.ckpt)     # needed by the extension contract's anchor check
    man = {}
    if args.launch_manifest:
        if not os.path.isfile(args.launch_manifest):
            problems.append(f"launch manifest not found: {args.launch_manifest}")
        elif args.extension:
            more, man = check_extension_binding(
                args.launch_manifest, args.launch_registry, args.arm, args.rung, args.config,
                args.ckpt, digest, args.expected_step, args.max_steps, args.repo_root)
            problems += more
        else:
            more, man = check_manifest_binding(args.launch_manifest, args.arm, args.rung,
                                               args.commit, args.max_steps)
            problems += more
    elif args.extension:
        problems.append("--extension requires --launch-manifest (the audited INITIAL manifest)")

    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print(f"  !! {p}")
        return 2

    tr = mc.get("training", {})
    print(f"restart lineage OK: {args.ckpt}")
    print(f"  global_step={gs} epoch={ck.get('epoch')} cond_method={tr.get('cond_method')!r} "
          f"angles={tr.get('frame_avg_angles')}")
    print(f"  optimizer_state=FULL ({len(opts[0]['state'])} entries) "
          f"lr={opts[0]['param_groups'][0].get('lr')} "
          f"sched_last_epoch={ck['lr_schedulers'][0].get('last_epoch')} ema_entries={n_ema}")
    if man and args.extension:
        print(f"extension lineage OK: {args.arm} {args.expected_step} -> {args.max_steps} continues "
              f"the audited launch job {kv_line(man, 'job').get('job')} "
              f"(launch commit {man.get('commit', '')[:12]}, running commit {args.commit[:12] or '<none>'})")
        print(f"  bound to the audited launch manifest: {args.launch_manifest}")
    elif man:
        print(f"  bound to launch manifest: {args.launch_manifest}")
    print(f"CKPT_SHA256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
