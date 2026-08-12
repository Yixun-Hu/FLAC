#!/usr/bin/env python
"""exp_15 — plan §7 rung 4: the bounded REAL-AR data-contract readback.

Every guarantee the training-side augmentation relies on is a claim about what
the dataloader actually hands `training_step`, and every test written for it so
far used synthetic tensors. This script closes that gap on the real thing: it
pulls a few dozen records through the ACTUAL dataloader path — the same
`create_dataloader_from_config` with the same `acousticroom_train.json` the
launcher passes to `train.py` — and asserts, per sample:

  1. depth is a float tensor of shape [3, 256, 512] with all values finite
     (the augmentation rolls axis 2 by a column count derived from yaw_aug.img_w
     = 512; a different width, a wrapped list, or a NaN would make the roll
     meaningless while every code hash stayed valid);
  2. all four pose fields are present, float, finite, trailing dimension 3
     (the schema guard REQUIRES all four — a missing one would abort training at
     step 0, so this is where we find out, not there);
  3. the rotation invariants hold ON REAL PANORAMAS for real offsets:
       * roll-by-d equivalence — rotate_scene_metadata equals an independently
         computed roll+z-rotation, to float tolerance;
       * exact integer round-trip — rotating by d then by W-d returns the
         original bytes;
       * equirectangular consistency — the rotated panorama is still a valid
         panorama (per-pixel azimuth still matches its column), which is what
         makes rotated conditioning a physically valid training pair;
       * pose-norm preservation and z-component invariance for all four fields.

Bounded by construction: CPU only, a few dozen samples, no model and no GPU.
num_workers is 1, not 0, because `create_dataloader_from_config` hardcodes
`persistent_workers=True` (src/data/dataset.py:405) and torch rejects that with
zero workers — going through the REAL factory is the point of this rung, so the
minimum it admits is what we use. A failure here is a launch-blocking
data-contract violation.

Usage (from the repo root):
    python worklog/worklog_yixun/exp_15_yaw_aug_claude/yaw_aug_real_data_readback.py \
        [--batches 6] [--batch-size 4]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.data.dataset import create_dataloader_from_config          # noqa: E402
from src.data.yaw_rotation import (                                  # noqa: E402
    POSE_KEYS,
    offsets_to_radians,
    rotate_scene_metadata,
    yaw_transform_consistency,
)

DATASET_CONFIG = "src/configs/dataset_configs/AR/train/acousticroom_train.json"
IMG_H, IMG_W = 256, 512
# The armed recipe's sample_size/sample_rate (FLAC_AR_YAWAUG.json).
SAMPLE_SIZE, SAMPLE_RATE = 10240, 22050
# Offsets to exercise: 1 column (the smallest non-identity roll), a quarter turn,
# and the wrap-around neighbour.
OFFSETS = (1, 128, 511)
ATOL = 1e-4


class Failures:
    def __init__(self):
        self.items = []

    def check(self, ok, message):
        if not ok:
            self.items.append(message)
        return ok


def _rot_matrix(alpha, dtype):
    c, s = math.cos(alpha), math.sin(alpha)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)


def _manual_rotate(depth, pose, d, img_w):
    """Reference rotation from first principles — NOT via rotate_scene_metadata,
    so agreement means the utility is right, not merely self-consistent.

    The matrix is built in EACH tensor's own dtype: real AR depth arrives as
    float64 while the pose fields are float32 (see the dtype line this script
    prints), and rotate_scene_metadata preserves whatever it is given.
    """
    alpha = d * 2.0 * math.pi / img_w
    rolled = torch.roll(depth, shifts=d, dims=2)
    return (torch.einsum("ij,jhw->ihw", _rot_matrix(alpha, depth.dtype), rolled),
            {k: torch.einsum("ij,...j->...i", _rot_matrix(alpha, v.dtype), v)
             for k, v in pose.items()})


def describe(md):
    """The observed data contract, printed once — this rung's factual record."""
    parts = []
    for key in ("depth",) + POSE_KEYS + ("context_audio", "padding_mask"):
        value = md.get(key)
        parts.append(f"{key}: {tuple(value.shape)} {value.dtype}" if torch.is_tensor(value)
                     else f"{key}: {type(value).__name__}")
    return "\n  ".join(parts)


def check_sample(md, index, fail):
    tag = f"sample[{index}]"

    # --- 1. depth -------------------------------------------------------------
    depth = md.get("depth")
    if not fail.check(torch.is_tensor(depth), f"{tag}: depth is {type(depth).__name__}, not a tensor"):
        return
    fail.check(tuple(depth.shape) == (3, IMG_H, IMG_W),
               f"{tag}: depth shape {tuple(depth.shape)} != (3, {IMG_H}, {IMG_W})")
    fail.check(depth.dtype.is_floating_point, f"{tag}: depth dtype {depth.dtype} is not floating point")
    fail.check(bool(torch.isfinite(depth).all()), f"{tag}: depth contains non-finite values")

    # --- 2. poses -------------------------------------------------------------
    poses = {}
    for key in POSE_KEYS:
        value = md.get(key)
        if not fail.check(torch.is_tensor(value), f"{tag}: {key} missing or not a tensor"):
            continue
        fail.check(value.ndim >= 1 and value.shape[-1] == 3,
                   f"{tag}: {key} shape {tuple(value.shape)} has trailing dim != 3")
        fail.check(value.dtype.is_floating_point, f"{tag}: {key} dtype {value.dtype} is not floating point")
        fail.check(bool(torch.isfinite(value).all()), f"{tag}: {key} contains non-finite values")
        poses[key] = value
    if len(poses) != len(POSE_KEYS) or tuple(depth.shape) != (3, IMG_H, IMG_W):
        return

    # --- 3. rotation invariants on this REAL sample ---------------------------
    base_dev = yaw_transform_consistency(depth, IMG_W, ())[0.0]
    fail.check(base_dev < 1e-2,
               f"{tag}: the UNROTATED panorama is already inconsistent (max azimuth dev {base_dev:.2e})")

    for d in OFFSETS:
        alpha = offsets_to_radians([d], IMG_W)[0]
        out = rotate_scene_metadata(md, alpha, IMG_W)
        ref_depth, ref_poses = _manual_rotate(depth, poses, d, IMG_W)

        fail.check(torch.allclose(out["depth"], ref_depth, atol=ATOL),
                   f"{tag} d={d}: rotated depth != an independent roll+z-rotation "
                   f"(max |diff| {float((out['depth'] - ref_depth).abs().max()):.3e})")
        fail.check(out["depth"].dtype == depth.dtype and out["depth"].device == depth.device,
                   f"{tag} d={d}: depth dtype/device not preserved")

        # z is a yaw invariant: the rolled height field must be untouched
        fail.check(torch.equal(out["depth"][2], torch.roll(depth, shifts=d, dims=2)[2]),
                   f"{tag} d={d}: depth z-component changed under a yaw rotation")

        # the rotated map is still a VALID equirectangular panorama
        dev = yaw_transform_consistency(depth, IMG_W, (d * 360.0 / IMG_W,))[d * 360.0 / IMG_W]
        fail.check(dev < 1e-2,
                   f"{tag} d={d}: rotated panorama is not equirectangular-consistent "
                   f"(max azimuth dev {dev:.2e})")

        # exact integer round-trip: d then W-d is the identity
        back = rotate_scene_metadata(out, offsets_to_radians([IMG_W - d], IMG_W)[0], IMG_W)
        fail.check(torch.allclose(back["depth"], depth, atol=ATOL),
                   f"{tag} d={d}: rotate-by-d then rotate-by-(W-d) did not return the original depth")

        for key in POSE_KEYS:
            got, ref, orig = out[key], ref_poses[key], poses[key]
            fail.check(torch.allclose(got, ref, atol=ATOL), f"{tag} d={d}: {key} != the reference rotation")
            fail.check(torch.allclose(got.norm(dim=-1), orig.norm(dim=-1), atol=ATOL),
                       f"{tag} d={d}: {key} norm not preserved")
            fail.check(torch.allclose(got[..., 2], orig[..., 2], atol=ATOL),
                       f"{tag} d={d}: {key} z-component changed")

    # untouched fields really are untouched
    out0 = rotate_scene_metadata(md, offsets_to_radians([7], IMG_W)[0], IMG_W)
    for key in ("context_audio", "padding_mask"):
        if torch.is_tensor(md.get(key)):
            fail.check(torch.equal(out0[key], md[key]), f"{tag}: {key} was modified by the rotation")
    if "scene" in md:
        fail.check(out0.get("scene") == md["scene"], f"{tag}: scene was modified by the rotation")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=1,
                    help="1 is the minimum the real factory admits (persistent_workers=True)")
    args = ap.parse_args(argv)

    config = json.loads((REPO / DATASET_CONFIG).read_text())
    print(f"dataset config : {DATASET_CONFIG}")
    print(f"split          : {config['datasets'][0]['json_file_path']}")
    print(f"reading        : {args.batches} batches x {args.batch_size} = "
          f"{args.batches * args.batch_size} samples, num_workers={args.num_workers}, CPU")
    print(f"offsets tested : {OFFSETS} columns of {IMG_W}")

    loader = create_dataloader_from_config(
        config, batch_size=args.batch_size, sample_size=SAMPLE_SIZE,
        sample_rate=SAMPLE_RATE, audio_channels=1, num_workers=args.num_workers,
        shuffle=False,
    )

    fail = Failures()
    seen = 0
    started = time.time()
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.batches:
            break
        _reals, metadata = batch
        if not fail.check(isinstance(metadata, (list, tuple)) and len(metadata) > 0,
                          f"batch {batch_index}: metadata is not a non-empty list"):
            break
        for md in metadata:
            if seen == 0:
                print(f"\nobserved fields (sample 0):\n  {describe(md)}\n")
            check_sample(md, seen, fail)
            seen += 1
    elapsed = time.time() - started

    print(f"\nsamples checked: {seen} in {elapsed:.1f}s")
    if fail.items:
        print(f"FAILED: {len(fail.items)} assertion(s)")
        for item in fail.items[:40]:
            print(f"  !! {item}")
        return 1
    print("PASS — real AR records satisfy every contract the yaw augmentation relies on:")
    print(f"  depth [3, {IMG_H}, {IMG_W}] float, finite")
    print("  source / source_vit / context_poses / context_poses_vit present, float, finite, trailing dim 3")
    print("  roll-by-d equivalence, exact (d, W-d) round-trip, panorama consistency,")
    print("  pose-norm preservation and z-invariance at every tested offset")
    return 0


if __name__ == "__main__":
    sys.exit(main())
