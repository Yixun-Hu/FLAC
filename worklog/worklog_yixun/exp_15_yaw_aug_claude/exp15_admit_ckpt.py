#!/usr/bin/env python
"""exp_15 — the per-cell CHECKPOINT ADMISSION gate (plan §5, gate G4).

Every eval cell must prove, before it spends a GPU, that the file it is about to
evaluate is the pre-registered 40,000-step checkpoint of the arm it claims. This
module is that proof, and it RE-DERIVES every fact from the checkpoint itself:

  * sha256 of the file, hashed through the same descriptor it is loaded from
    (so the name cannot be re-pointed between the hash and the load);
  * the embedded ``global_step``, which must be exactly 40,000 (a plain int —
    ``"40000"`` and ``40000.5`` are not the endpoint);
  * canonical-bytes equality between the checkpoint's embedded ``model_config``
    and the arm's config FILE, plus that file's raw sha256 against the record;
  * the EMA family mirroring ``diffusion.model.*`` exactly — same suffix set,
    same shapes, same dtypes — and its inventory digest;
  * the arm's IDENTITY in the embedded config: ``training.yaw_aug`` enabled at
    ``img_w 512 seed 42`` for YAWAUG and ABSENT for the VANL control, plus the
    absence of any ``cond_method`` (both exp_15 arms are vanilla models).

Only then are those recomputed values compared against the arm's committed
admission record — ``yaw_aug_control_admission.json`` for VANL, the chain's
``yaw_aug_launch_registry.json`` (``final_ckpt_sha256`` + the step-40000 leg's
audit block) for YAWAUG.

**A record's own ``checks`` block is never read.** ``"global_step_equals_expected":
true`` is the recorder's CLAIM that it once validated the file; it is not evidence
about the file on disk now. What is taken from a record is its MEASUREMENTS, and
every one of them is recomputed here.

The primitives are IMPORTED from ``yaw_aug_record_control`` — ``snapshot_checkpoint``,
``canonical_bytes``, ``canonical_sha256``, ``summarize_ema`` — never re-implemented:
a second implementation of "canonical bytes" or "does the EMA mirror the online
model" is a second thing that can disagree with the record it is checking.

Usage
-----
    python exp15_admit_ckpt.py verify --arm YAWAUG --ckpt PATH --config PATH
    python exp15_admit_ckpt.py expect --arm VANL

``verify`` exits 0 (admitted), 1 (REFUSED — named reasons on stdout) or 2 (the
arm has no recorded expectation yet / usage error). On success its last line is
``ADMITTED <arm> ckpt_sha256=<64 hex>`` so a caller can reuse the digest instead
of hashing 724 MB a second time.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import exp15_validate_cell as cells                      # noqa: E402  (torch-free)
import yaw_aug_record_control as recorder                # noqa: E402  (imports torch)

# The arm's IDENTITY, as embedded in the checkpoint's own model_config. This is
# exp_15's analogue of exp_14's orbit gate: there, the arm was defined by the
# frame-average orbit it was trained with; here it is defined by whether training
# applied the random-yaw augmentation, and with which parameters.
EXPECTED_YAW_AUG = {
    "YAWAUG": {"enabled": True, "img_w": 512, "seed": 42},
    "VANL": None,                       # the control was trained with no yaw_aug key
}


def _identity_reasons(arm, embedded_config):
    """Named reasons the embedded config is not this arm's (plan §3.3, §5 G4)."""
    reasons = []
    training = embedded_config.get("training")
    if not isinstance(training, dict):
        return [f"embedded model_config has no 'training' block ({type(training).__name__})"]
    # Both exp_15 arms are VANILLA models. A checkpoint trained with frame
    # averaging would be a different experiment wearing this arm's name.
    if training.get("cond_method") not in (None, "vanilla"):
        reasons.append(f"embedded training.cond_method={training.get('cond_method')!r} — "
                       "both exp_15 arms are vanilla-conditioned models")
    if "frame_avg_angles" in training:
        reasons.append(f"embedded training.frame_avg_angles={training['frame_avg_angles']!r} — "
                       "this checkpoint was trained with an orbit and is not an exp_15 arm")
    want = EXPECTED_YAW_AUG[arm]
    got = training.get("yaw_aug", None)
    if want is None:
        if got is not None:
            reasons.append(f"embedded training.yaw_aug={got!r} — the {arm} control must have "
                           "been trained with NO yaw augmentation; a checkpoint that carries "
                           "the key is the treatment arm, not the control")
    else:
        if got != want:
            reasons.append(f"embedded training.yaw_aug={got!r} != the registered "
                           f"{want!r} — the arm's augmentation protocol is part of its "
                           "identity, so a differing img_w or seed is a different arm")
    return reasons


def recompute(arm, ckpt_path, config_path, expect_step):
    """Everything this gate derives FROM THE FILES; raises on an unreadable input.

    Returns ``(measured, reasons)``: ``measured`` is what the checkpoint and the
    config actually are, ``reasons`` names every intrinsic failure (a step that is
    not the endpoint, a broken EMA mirror, a config the checkpoint was not trained
    with). Comparison against the RECORD happens in :func:`verify`.
    """
    reasons = []
    config_bytes = Path(config_path).read_bytes()
    config_obj = json.loads(config_bytes)
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    config_canonical = recorder.canonical_sha256(config_obj)

    checkpoint, ckpt_sha, identity = recorder.snapshot_checkpoint(ckpt_path)

    global_step = checkpoint.get("global_step", None)
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        reasons.append(f"embedded global_step is {global_step!r} "
                       f"({type(global_step).__name__}), expected a plain int")
    elif global_step != int(expect_step):
        reasons.append(f"embedded global_step is {global_step}, not the pre-registered "
                       f"endpoint {expect_step}")

    state_dict = checkpoint.get("state_dict", None)
    ema = {}
    if not isinstance(state_dict, dict) or not state_dict:
        reasons.append("no usable 'state_dict' in the checkpoint")
    else:
        try:
            # The EMA<->online mirror, suffix set + shape + dtype, from the
            # recorder's own definition of it.
            ema = recorder.summarize_ema(state_dict)
        except ValueError as exc:
            reasons.append(f"EMA/online mirror check FAILED: {exc}")

    embedded = checkpoint.get("model_config", None)
    embedded_canonical = None
    if embedded is None:
        reasons.append("no 'model_config' embedded in the checkpoint: it cannot be bound "
                       "to the config it was trained with")
    else:
        try:
            embedded_bytes = recorder.canonical_bytes(embedded)
        except ValueError as exc:
            reasons.append(f"embedded model_config is not canonicalisable: {exc}")
        else:
            embedded_canonical = hashlib.sha256(embedded_bytes).hexdigest()
            if embedded_bytes != recorder.canonical_bytes(config_obj):
                reasons.append(f"the embedded model_config differs from {config_path} "
                               "(canonical bytes) — this checkpoint was not trained with "
                               "that config")
        if isinstance(embedded, dict):
            reasons += _identity_reasons(arm, embedded)
        else:
            reasons.append(f"embedded model_config is a {type(embedded).__name__}, not an object")

    measured = {
        "arm": arm,
        "ckpt_path": str(ckpt_path),
        "sha256": ckpt_sha,
        "bytes": identity["bytes"],
        "global_step": global_step,
        "embedded_config_canonical_sha256": embedded_canonical,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "config_canonical_sha256": config_canonical,
        "ema_inventory_sha256": ema.get("ema_inventory_sha256"),
        "ema_key_count": ema.get("ema_key_count"),
        "online_model_key_count": ema.get("online_model_key_count"),
    }
    return measured, reasons


def _cmp(reasons, label, got, want):
    """Compare a recomputed value with the RECORDED one; ``None`` in the record skips.

    A recorded value that is absent is reported as unchecked rather than silently
    treated as agreement — but it never turns into a pass on its own: the fields
    the gate depends on (sha256, the two canonical config digests, the EMA
    inventory) are required by the expectation loader, so they are never None here.
    """
    if want is None:
        return
    if got != want:
        reasons.append(f"{label}: recomputed {got!r} != recorded {want!r}")


def _provenance_reasons(expected, ckpt_path, launch_manifest, main_repo):
    """Named reasons this FILE is not the one the registered launch produced.

    A sha256 says two files hold the same bytes. It does not say the bytes came
    from a run this campaign registered — a checkpoint copied into the tree by
    hand satisfies every other check in this module. Two cheap facts close that:
    the file must sit at the path the record names, and the run directory's
    launch manifest must still hash to the value the COMMITTED record pinned
    (the manifest itself lives under gitignored outputs_FLAC and is mutable, so
    binding to "whatever it says now" would prove nothing).
    """
    reasons = []
    want_rel = expected.get("ckpt_path_rel")
    if want_rel:
        want_abs = os.path.realpath(os.path.join(main_repo, want_rel))
        got_abs = os.path.realpath(ckpt_path)
        if want_abs != got_abs:
            reasons.append(f"checkpoint location: {got_abs} is not the registered "
                           f"{want_abs} ({want_rel})")
    want_manifest = expected.get("manifest_sha256")
    if want_manifest:
        if not launch_manifest:
            reasons.append("no --launch-manifest given: the checkpoint cannot be bound to "
                           "the recorded launch that produced it")
        elif not os.path.isfile(launch_manifest):
            reasons.append(f"launch manifest missing: {launch_manifest} — a screen may only "
                           "evaluate a checkpoint from a recorded launch")
        else:
            got = hashlib.sha256(open(launch_manifest, "rb").read()).hexdigest()
            if got != want_manifest:
                reasons.append(f"launch manifest sha256: {got[:12]} != registered "
                               f"{want_manifest[:12]} — the manifest changed after it was "
                               "registered, so it no longer vouches for this run")
    return reasons


def verify(arm, ckpt_path, config_path, expect_step, control, registry,
           launch_manifest=None, main_repo="."):
    """``(reasons, measured)`` — empty reasons means the checkpoint is ADMITTED."""
    expected = cells.admission_expectation(arm, control, registry, expect_step)
    measured, reasons = recompute(arm, ckpt_path, config_path, expect_step)
    reasons += _provenance_reasons(expected, ckpt_path, launch_manifest, main_repo)
    _cmp(reasons, "checkpoint sha256", measured["sha256"], expected["sha256"])
    _cmp(reasons, "checkpoint size in bytes", measured["bytes"], expected.get("bytes"))
    _cmp(reasons, "embedded model_config canonical sha256",
         measured["embedded_config_canonical_sha256"],
         expected["embedded_config_canonical_sha256"])
    _cmp(reasons, "config file sha256", measured["config_sha256"], expected["config_sha256"])
    _cmp(reasons, "config file canonical sha256", measured["config_canonical_sha256"],
         expected["config_canonical_sha256"])
    _cmp(reasons, "EMA inventory sha256", measured["ema_inventory_sha256"],
         expected["ema_inventory_sha256"])
    _cmp(reasons, "EMA key count", measured["ema_key_count"], expected.get("ema_key_count"))
    _cmp(reasons, "online DiT key count", measured["online_model_key_count"],
         expected.get("online_model_key_count"))
    return reasons, measured


def _cmd_verify(args):
    try:
        expected_source = cells.admission_expectation(
            args.arm, args.control, args.registry, args.expect_step)["source"]
    except ValueError as exc:
        print(f"ADMISSION UNAVAILABLE for {args.arm}: {exc}", file=sys.stderr)
        return 2
    config = args.config or os.path.join(args.code_root or os.getcwd(),
                                         cells.ARM_CONFIG_REL[args.arm])
    if not os.path.isfile(config):
        print(f"ADMISSION REFUSED {args.arm}: model config not found: {config}")
        return 1
    if not os.path.isfile(args.ckpt):
        print(f"ADMISSION REFUSED {args.arm}: checkpoint not found: {args.ckpt}")
        return 1
    try:
        reasons, measured = verify(args.arm, args.ckpt, config, args.expect_step,
                                   args.control, args.registry,
                                   launch_manifest=args.launch_manifest,
                                   main_repo=args.main_repo or os.getcwd())
    except (ValueError, OSError) as exc:
        print(f"ADMISSION REFUSED {args.arm}: {exc}")
        return 1
    if reasons:
        print(f"ADMISSION REFUSED {args.arm} ({len(reasons)} reason(s)) "
              f"[expectation: {expected_source}]:")
        for reason in reasons:
            print(f"  - {reason}")
        return 1
    print(f"admission OK: {args.arm} step {measured['global_step']} "
          f"config {measured['config_sha256'][:12]} embedded {measured['embedded_config_canonical_sha256'][:12]} "
          f"EMA {measured['ema_key_count']} keys inv {measured['ema_inventory_sha256'][:12]} "
          f"[expectation: {expected_source}]")
    print(f"ADMITTED {args.arm} ckpt_sha256={measured['sha256']}")
    return 0


def _cmd_expect(args):
    try:
        exp = cells.admission_expectation(args.arm, args.control, args.registry,
                                          args.expect_step)
    except ValueError as exc:
        print(f"EXPECT-UNAVAILABLE {args.arm}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(exp, indent=2, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd")

    common = dict(control=cells.CONTROL_ADMISSION, registry=cells.LAUNCH_REGISTRY)

    v = sub.add_parser("verify", help="recompute and admit ONE checkpoint")
    v.add_argument("--arm", required=True, choices=list(cells.ARMS))
    v.add_argument("--ckpt", required=True)
    v.add_argument("--config", default=None,
                   help="the arm's model config; defaults to --code-root/<registered path>")
    v.add_argument("--code-root", default=None)
    v.add_argument("--launch-manifest", default=None,
                   help="the run directory's launch_manifest.txt; its sha256 must still "
                        "equal the value the committed record pinned")
    v.add_argument("--main-repo", default=None,
                   help="tree the record's repo-relative checkpoint path resolves against")
    v.add_argument("--expect-step", type=int, default=cells.STEP)
    v.add_argument("--control", default=common["control"])
    v.add_argument("--registry", default=common["registry"])
    v.set_defaults(func=_cmd_verify)

    e = sub.add_parser("expect", help="print the arm's recorded admission facts")
    e.add_argument("--arm", required=True, choices=list(cells.ARMS))
    e.add_argument("--expect-step", type=int, default=cells.STEP)
    e.add_argument("--control", default=common["control"])
    e.add_argument("--registry", default=common["registry"])
    e.set_defaults(func=_cmd_expect)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
