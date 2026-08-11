#!/usr/bin/env python
"""exp_15 — write the VANL control-admission record (plan §3.3-1, §6.4).

exp_15 contrasts the YAWAUG arm against exp_11's vanilla arm at step 40,000.
exp_11's ``arm_launch_registry.json`` pins that run's launch manifest, commit,
config, VAE, rung and seed — but **not** the checkpoint file: VANL's job ended
FAILED after the save, so its ``final_ckpt_sha256`` was never backfilled. The
control is therefore, as it stands, identified only by a path, and a path is not
evidence.

This script writes the missing binding, once: ``yaw_aug_control_admission.json``
records the checkpoint's sha256 and size, the facts embedded *inside* it
(``global_step``, EMA weight family, optimizer/scheduler state counts), the
config's freshly computed sha256 checked against the registry's pin, and exp_11's
cross-references copied verbatim. Every VANL eval cell re-hashes the checkpoint
against this record before running (plan §5, gate G4), so the record must be
immutable: the script refuses to overwrite an existing one and has no --force.

It is strictly read-only with respect to the checkpoint and the config — exp_11
owns both — and writes nothing at all if any validation fails. A failure here is
a launch-blocking finding to report, never something to patch around.

Usage::

    python worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py \\
      --ckpt outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt \\
      --config worklog/worklog_yixun/exp_11_fa_orbit_claude/FLAC_AR_VANCKPT.json \\
      --out worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_control_admission.json \\
      --expect-step 40000
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch

# exp_11's registry facts for VANL (job 3661520), copied VERBATIM from
# worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json. These are
# not recomputed here: the manifest and VAE they refer to are exp_11's artifacts,
# and re-deriving them would silently paper over a drift instead of surfacing it.
EXP11_CROSS_REFERENCES = {
    "source": "worklog/worklog_yixun/exp_11_fa_orbit_claude/arm_launch_registry.json (arm VANL)",
    "manifest_sha256": "113d06a284c6198cf9487e99a2efb7ccde94ae13e656a403fe2af0281d3de8b1",
    "commit": "81ddac372076ea92751ae09cbaf371df70f396e5",
    "training_seed": 42,
    "rung": "8x8",
    "vae_sha256": "8d82159eec35210198246f449bec6561fc19b514922f340a17515050daf7f0b9",
    "job": "3661520",
    "note": "exp_11's registry records NO final_ckpt_sha256 for VANL (job ended "
            "FAILED after the save); this record supplies that missing binding.",
}

# The registry's config pin. The --config file must hash to exactly this.
REGISTRY_CONFIG_SHA256 = "733ca52b66c43538e1b9e603e979678af95ac05d89fd1d481ebb472a285a49d8"

# Most specific first: the EMA weights live under diffusion_ema.ema_model.*,
# alongside EMA bookkeeping buffers (initted/step) under diffusion_ema.*.
EMA_PREFIX_CANDIDATES = ("diffusion_ema.ema_model.", "diffusion_ema.")

CHUNK_BYTES = 1 << 20


def sha256_file(path, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Stream a file through sha256. The checkpoint is 724 MB — never read whole."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_fd(fd, chunk_bytes: int = CHUNK_BYTES) -> str:
    """Hash the inode behind an ALREADY OPEN descriptor, from offset 0.

    Hashing by path and loading by path are two lookups: between them the name
    can be re-pointed at a different inode, and the record would then bind a
    checkpoint that was never inspected. Hashing through a descriptor we keep
    open pins the object being measured; :func:`snapshot_checkpoint` then proves
    the path still resolves to that same object.
    """
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(fd, chunk_bytes, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def _identity(stat_result):
    return {
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "bytes": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }


def safe_load_checkpoint(path):
    """Load with mmap (storages stay on disk) and the SAFE unpickler only.

    There is deliberately no ``weights_only=False`` fallback. This tool exists to
    admit a checkpoint as evidence; falling back would deserialize an unvalidated
    archive through arbitrary pickle payloads at exactly the moment the file's
    trustworthiness is in question.
    """
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as error:
        raise ValueError(
            f"{path}: the safe loader (mmap=True, weights_only=True) failed: "
            f"{type(error).__name__}: {error}. Refusing to retry with "
            "weights_only=False — an admission tool must not execute pickles."
        )


def snapshot_checkpoint(path):
    """Hash and load ONE stable snapshot; fail if the file moves underneath us.

    Returns ``(checkpoint, sha256, identity)``.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        before = os.fstat(fd)
        digest = _sha256_fd(fd)
        checkpoint = safe_load_checkpoint(path)
        after_fd = os.fstat(fd)
        after_path = os.stat(path)
        if _identity(after_fd) != _identity(before):
            raise ValueError(
                f"{path}: the checkpoint changed while it was being read "
                f"(size/mtime moved on the same inode) — the hash and the "
                "validated contents would describe different data"
            )
        if (after_path.st_dev, after_path.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(
                f"{path}: the path changed to a different file while it was being "
                "read (inode replaced) — the hash and the validated contents would "
                "describe different files"
            )
    finally:
        os.close(fd)
    return checkpoint, digest, _identity(before)


def _find_ema(state_dict):
    for prefix in EMA_PREFIX_CANDIDATES:
        matched = [k for k in state_dict if k.startswith(prefix)]
        if matched:
            return prefix, len(matched)
    raise ValueError(
        "no EMA weights found in the checkpoint state_dict (looked for prefixes "
        f"{list(EMA_PREFIX_CANDIDATES)}): exp_15 evaluates EMA weights, so a "
        "checkpoint without them cannot be admitted as the control"
    )


def canonical_sha256(obj) -> str:
    """Formatting-independent hash of a JSON-able object (sorted keys, no spaces).

    The config *file* is hashed as raw bytes to match exp_11's registry pin; the
    config embedded in the checkpoint went through pickle and has no byte form of
    its own, so the two are compared canonically.
    """
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def summarize_checkpoint(ckpt_path, expect_step: int, config_obj) -> dict:
    """Validate and summarise ONE stable, safely loaded snapshot."""
    checkpoint, sha256, identity = snapshot_checkpoint(ckpt_path)

    if "global_step" not in checkpoint:
        raise ValueError(f"{ckpt_path}: no 'global_step' in the checkpoint")
    global_step = int(checkpoint["global_step"])
    if global_step != int(expect_step):
        raise ValueError(
            f"{ckpt_path}: embedded global_step is {global_step}, expected "
            f"{expect_step} — this is not the pre-registered endpoint"
        )

    state_dict = checkpoint.get("state_dict", None)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"{ckpt_path}: no usable 'state_dict'")

    ema_prefix, ema_key_count = _find_ema(state_dict)

    # The config the checkpoint was TRAINED with. The file's sha proves only what
    # sits on disk today; this proves the checkpoint belongs to it.
    if "model_config" not in checkpoint:
        raise ValueError(
            f"{ckpt_path}: no 'model_config' embedded in the checkpoint, so the "
            "checkpoint cannot be bound to the config it was trained with"
        )
    embedded_config = checkpoint["model_config"]
    if embedded_config != config_obj:
        raise ValueError(
            f"{ckpt_path}: the embedded model_config differs from the --config "
            "file — this checkpoint was not trained with that config"
        )

    epoch = checkpoint.get("epoch", None)
    return {
        "path": str(ckpt_path),
        "bytes": identity["bytes"],
        "sha256": sha256,
        "global_step": global_step,
        "epoch": int(epoch) if epoch is not None else None,
        "state_dict_keys": len(state_dict),
        "ema_prefix": ema_prefix,
        "ema_key_count": ema_key_count,
        "embedded_config_canonical_sha256": canonical_sha256(embedded_config),
        "online_key_count": sum(1 for k in state_dict if k.startswith("diffusion.")),
        "optimizer_states": len(checkpoint.get("optimizer_states", []) or []),
        "lr_schedulers": len(checkpoint.get("lr_schedulers", []) or []),
        "loaded_with": {"mmap": True, "map_location": "cpu", "weights_only": True},
    }


def build_record(ckpt_path, config_path, expect_step: int) -> dict:
    """The full admission record. Raises (writing nothing) on any failed check."""
    # ONE read: the bytes that are hashed are the bytes that are parsed. Hashing
    # and re-opening would leave room for the file to change in between, and the
    # record would then pin a config that was never parsed.
    config_bytes = Path(config_path).read_bytes()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    if config_sha != REGISTRY_CONFIG_SHA256:
        raise ValueError(
            f"{config_path}: sha256 {config_sha} does not match exp_11's "
            f"registry pin {REGISTRY_CONFIG_SHA256} — this is not the config VANL "
            "was trained with"
        )

    config_obj = json.loads(config_bytes)
    checkpoint = summarize_checkpoint(ckpt_path, expect_step, config_obj)

    return {
        "_meta": {
            "experiment": "exp_15",
            "purpose": "immutable admission record for the VANL control checkpoint "
                       "(plan §3.3-1); re-validated by every VANL eval cell (gate G4)",
            "recorder": "worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_record_control.py",
            "recorded_utc": datetime.datetime.now(datetime.timezone.utc)
                                    .replace(microsecond=0).isoformat(),
            "torch": torch.__version__,
            "expect_step": int(expect_step),
        },
        "checkpoint": checkpoint,
        "config": {
            "path": str(config_path),
            "sha256": config_sha,
            "canonical_sha256": canonical_sha256(config_obj),
        },
        "exp_11_cross_references": dict(EXP11_CROSS_REFERENCES),
        "checks": {
            "global_step_equals_expected": True,
            "config_sha256_matches_registry": True,
            "ema_state_present": True,
            "embedded_config_equals_config_file": True,
        },
    }


def write_record(record: dict, out_path) -> None:
    """Write once. An existing record is evidence already relied upon: never touch it."""
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists; the admission record is immutable and this "
            "script has no --force. Inspect the existing record instead."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expect-step", required=True, type=int)
    args = parser.parse_args(argv)

    if args.out.exists():
        raise SystemExit(f"ADMISSION ABORTED: {args.out} already exists (immutable)")

    try:
        record = build_record(args.ckpt, args.config, args.expect_step)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"ADMISSION FAILED: {error}")

    write_record(record, args.out)
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
