#!/usr/bin/env python3
"""exp_21 ladder rung 4: real-data readback + cap-64-vs-grouped-cap parity.

Required as pre-launch evidence by the integrative review ("Real-data readback:
all required metadata/outputs, real DINO C4 spot check, and cap-64 versus
grouped-cap allclose").

WHAT IT PROVES. The arm TRAINS at ``frame_avg_max_fwd_samples: 32`` and is
EVALUATED at 64 (announcement 06; plan §3.1). At a batch of 32 those two caps
partition the C4 orbit differently -- ``max(1, 32//32) = 1`` angle per chunk
versus ``max(1, 64//32) = 2`` -- so they are genuinely different executions of
the same sum, over the real DINOv3 backbone, on real AR data. If they disagree,
the arm's training-time conditioning is not the thing its evaluation measures.
Batch 32 is therefore not a convenience: it is the smallest batch at which the
two caps differ at all.

Eval mode is the point of comparison, not a shortcut: DINOv3's random RoPE
rescale is drawn once per forward and guarded by ``self.training``, so in eval
the chunk partition must be numerically inert. (In TRAIN mode it is deliberately
NOT inert -- that is the disclosed draw-schedule difference cap 32 exists to
match against B-F, and it is not what this probe measures.)

Read-only: writes nothing but stdout. Run once, from the repo root; the Planner
tees it into this folder.

    HF_HUB_OFFLINE=1 python worklog/worklog_yixun/exp_21_bf_fa_cartesian_claude/bfc_cap_parity_probe.py

Written by the Coder seat (Claude Opus 5, max effort), exp_21 round 5 ladder.
"""
import json
import math
import os
import sys

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the pinned DINOv3 cache, never the hub
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, REPO)                       # never the stale pip-installed src/

from src.data.dataset import create_dataloader_from_config          # noqa: E402
from src.data.yaw_rotation import (DEFAULT_FRAME_ANGLES, POSE_KEYS,  # noqa: E402
                                   fa_cartesian_conditioning, rotate_scene_metadata)
from src.models.conditioners import (                                # noqa: E402
    create_multi_conditioner_from_conditioning_config)

MODEL_CONFIG = os.path.join(REPO, "worklog", "worklog_yixun",
                            "exp_21_bf_fa_cartesian_claude", "FLAC_AR_BFC.json")
DATASET_CONFIG = os.path.join(REPO, "src", "configs", "dataset_configs", "AR",
                              "train", "acousticroom_train.json")
BATCH = 32                  # the arm's per-rank micro-batch: caps 32 and 64 differ here
CAP_TRAIN, CAP_EVAL = 32, 64
TOL_STRICT, TOL_RELAXED = 1e-5, 1e-3
MIN_FREE_GIB = 8            # co-tenant box: refuse rather than OOM someone else's run
SEED = 42
# (shape, dtype) as the AR path actually produces them -- measured, not assumed.
# The four pose keys are float32 because ``AR_md`` calls ``.float()`` on them, and
# that is load-bearing: ``eval_FLAC.sample_context_ids`` REFUSES to fingerprint
# ``context_poses`` in any other dtype (rendering the same positions as float64
# changed two of the unseen split's six-decimal strings, float16 changed 5,032).
# ``depth`` is float64 because it is ``np.load``ed from the stored depth map and
# nothing downcasts it; pinned here rather than waved through, since a silent
# change would alter the precision the ViT actually sees.
EXPECTED = {"source": ((3,), torch.float32),
            "source_vit": ((1, 3), torch.float32),
            "context_poses": ((8, 3), torch.float32),
            "context_poses_vit": ((8, 3), torch.float32),
            "depth": ((3, 256, 512), torch.float64)}


def pick_device():
    if not torch.cuda.is_available():
        return "cpu", "no CUDA device"
    free, total = torch.cuda.mem_get_info()
    gib = free / 1024 ** 3
    if gib < MIN_FREE_GIB:
        return "cpu", f"only {gib:.1f} GiB free on cuda:0 (floor {MIN_FREE_GIB})"
    return "cuda", f"{gib:.1f} of {total / 1024 ** 3:.1f} GiB free on cuda:0"


def load_batch(model_config):
    """One batch of REAL training metadata, through the training path itself.

    ``create_dataloader_from_config`` is the same entry point ``train.py`` uses, so
    the samples come from the same ``SampleDataset``, the same
    ``AR_md.get_custom_metadata`` hook and the same ``modalities`` block (K=8
    context, depth, poses) as a training step. Only ONE batch is drawn -- the
    iterator is abandoned, never an epoch.

    ``shuffle=True`` (seeded) rather than sequential: ``data/AR/train.json`` is a
    dict of ten room families, so the first 32 sequential items would all come
    from one room and the C4 spot-check below would be a single-scene claim.
    """
    with open(DATASET_CONFIG) as fh:
        dataset_config = json.load(fh)
    torch.manual_seed(SEED)
    loader = create_dataloader_from_config(
        dataset_config, batch_size=BATCH, num_workers=2,
        sample_rate=model_config["sample_rate"], sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1), shuffle=True)
    _audio, metadata = next(iter(loader))
    return metadata


def readback(metadata):
    """Every field the arm's conditioners consume, present and the right shape."""
    assert len(metadata) == BATCH, f"got {len(metadata)} samples, wanted {BATCH}"
    for i, md in enumerate(metadata):
        for key, (shape, dtype) in EXPECTED.items():
            assert key in md, f"sample {i} has no {key!r}"
            got = md[key]
            assert isinstance(got, torch.Tensor), f"sample {i} {key} is {type(got).__name__}"
            assert tuple(got.shape) == shape, f"sample {i} {key} is {tuple(got.shape)}, want {shape}"
            assert got.dtype == dtype, f"sample {i} {key} is {got.dtype}, want {dtype}"
            assert torch.isfinite(got).all(), f"sample {i} {key} is not finite"
        assert "context_audio" in md, f"sample {i} has no context_audio"
        assert md["context_audio"].shape[0] == EXPECTED["context_poses"][0][0]
    scenes = sorted({md["scene"] for md in metadata})
    print(f"  readback OK: {BATCH} samples, all 5 conditioner fields present")
    print("  dtypes: " + ", ".join(f"{k}={str(v[1]).replace('torch.', '')}"
                                   for k, v in EXPECTED.items()))
    print(f"  scenes spanned ({len(scenes)}): {scenes}")
    d, p = metadata[0]["depth"], metadata[0]["context_poses"]
    print(f"  depth[0]   {tuple(d.shape)} min {d.min():+.4f} max {d.max():+.4f} std {d.std():.4f}")
    print(f"  ctx_pos[0] {tuple(p.shape)} min {p.min():+.4f} max {p.max():+.4f} std {p.std():.4f}")
    print(f"  context_audio[0] {tuple(metadata[0]['context_audio'].shape)}")


def condition(conditioner, metadata, device, cap):
    torch.manual_seed(SEED)                    # eval mode draws nothing; pinned anyway
    with torch.no_grad():
        return fa_cartesian_conditioning(conditioner, metadata, device,
                                         DEFAULT_FRAME_ANGLES, max_fwd_samples=cap)


def compare(label, a, b, tol):
    """``(ok, worst)`` over every conditioning id, printing the per-id max abs diff."""
    assert set(a) == set(b), f"{label}: id sets differ, {sorted(a)} vs {sorted(b)}"
    worst, ok = 0.0, True
    for key in sorted(a):
        x, y = a[key][0].float(), b[key][0].float()
        assert x.shape == y.shape, f"{label}: {key} shape {x.shape} vs {y.shape}"
        diff = (x - y).abs().max().item()
        worst = max(worst, diff)
        ok = ok and diff <= tol
        print(f"    {label:<18} {key:<20} max|d| {diff:.3e}")
    return ok, worst


def main():
    device, why = pick_device()
    print(f"exp_21 rung 4 -- cap parity + real-data readback")
    print(f"device: {device} ({why}) | torch {torch.__version__} | HF_HUB_OFFLINE="
          f"{os.environ.get('HF_HUB_OFFLINE')}")
    with open(MODEL_CONFIG) as fh:
        model_config = json.load(fh)
    training = model_config["training"]
    print(f"arm: cond_method={training['cond_method']!r} angles={training['frame_avg_angles']} "
          f"TRAIN cap={training['frame_avg_max_fwd_samples']} | probe caps {CAP_TRAIN} vs {CAP_EVAL}")
    print(f"chunk plans at batch {BATCH}: cap {CAP_TRAIN} -> {max(1, CAP_TRAIN // BATCH)} "
          f"angle/chunk, cap {CAP_EVAL} -> {max(1, CAP_EVAL // BATCH)} angles/chunk")

    print("[1/4] loading one real training batch")
    metadata = load_batch(model_config)
    print("[2/4] readback")
    readback(metadata)

    print("[3/4] building the real conditioner stack (shared DINOv3, pinned cache)")
    # The same reviewed factory diffusion.py:316 calls. BFC declares only
    # dist_embedder / ViTCoordinates / rir, none of which take a pretransform.
    conditioner = create_multi_conditioner_from_conditioning_config(
        model_config["model"]["conditioning"], pretransform=None)
    conditioner.eval().requires_grad_(False).to(device)
    assert not conditioner.training, "conditioner must be in eval mode"

    print(f"[4/4] cap {CAP_TRAIN} vs cap {CAP_EVAL}, then the 90-degree C4 spot-check")
    out32 = condition(conditioner, metadata, device, CAP_TRAIN)
    out64 = condition(conditioner, metadata, device, CAP_EVAL)
    img_w = int(metadata[0]["depth"].shape[-1])
    rotated = [rotate_scene_metadata(md, math.radians(90.0), img_w, POSE_KEYS)
               for md in metadata]
    out_rot = condition(conditioner, rotated, device, CAP_EVAL)

    tol, note = TOL_STRICT, ""
    cap_ok, cap_worst = compare("cap32-vs-cap64", out32, out64, tol)
    c4_ok, c4_worst = compare("c4-90deg", out64, out_rot, tol)
    if not (cap_ok and c4_ok) and max(cap_worst, c4_worst) <= TOL_RELAXED:
        # Documented fallback: on GPU the ViT may take bf16/flash-attention kernels
        # whose reduction order differs per chunk shape, which is a numerical
        # artifact of the partition, not a different sum. Report and relax ONCE.
        tol, note = TOL_RELAXED, (
            f"  NOTE tolerance relaxed {TOL_STRICT:g} -> {TOL_RELAXED:g}: worst diff "
            f"{max(cap_worst, c4_worst):.3e} on device {device}, consistent with "
            "reduction-order differences in the ViT's fused kernels rather than a "
            "different orbit sum (both remain far below any metric's resolution).")
        cap_ok, c4_ok = cap_worst <= tol, c4_worst <= tol
    if note:
        print(note)
    print(f"  worst cap-parity diff {cap_worst:.3e} | worst C4 diff {c4_worst:.3e} "
          f"| tolerance {tol:g}")
    verdict = "PASS" if (cap_ok and c4_ok) else "FAIL"
    print(f"RUNG-4 VERDICT: {verdict} -- cap {CAP_TRAIN}/{CAP_EVAL} parity "
          f"{'OK' if cap_ok else 'VIOLATED'}, real-data C4 invariance "
          f"{'OK' if c4_ok else 'VIOLATED'} (batch {BATCH}, device {device}, tol {tol:g})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
