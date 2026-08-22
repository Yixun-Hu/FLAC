#!/usr/bin/env python3
"""exp_21 ladder rung 4: real-data readback + cap-64-vs-grouped-cap parity.

Required as pre-launch evidence by the integrative review ("Real-data readback:
all required metadata/outputs, real DINO C4 spot check, and cap-64 versus
grouped-cap allclose").

WHAT IT PROVES. The arm TRAINS at ``frame_avg_max_fwd_samples: 32`` and is
EVALUATED at 64 (announcement 06; plan §3.1). At the arm's per-rank micro-batch
of 32 those caps partition the C4 orbit differently -- ``max(1, 32//32) = 1``
angle per chunk versus ``max(1, 64//32) = 2`` -- so they are genuinely different
executions of the same sum, over the real DINOv3 backbone, on real AR data. If
they disagree, the arm's training-time conditioning is not the thing its
evaluation measures.

Batch 32 is chosen because it is the ARM'S OWN micro-batch, not because it is the
smallest batch that separates the caps: with three nonzero C4 angles the plans
already diverge at batch 11 (cap 32 -> chunks {2,1}, cap 64 -> {3}), as the
pre-launch review notes. Testing the rung the arm actually trains at is the point.

BOTH PRECISIONS ARE MEASURED. The registered evaluation conditions under BF16
autocast (`--cond-autocast bf16`, announcement 05; `exp21_protocol` emits it on
every one of the 34 cells), so BF16 is the protocol that will produce the
published numbers and it is the binding measurement here. FP32 is reported
alongside as the exactness reference: it separates "the two chunk plans compute
the same sum" from "the autocast kernels round it the same way". The autocast
context and the matmul precision are taken from ``eval_FLAC`` itself
(:func:`eval_FLAC.resolve_cond_autocast`), never re-spelled here.

Eval mode is the point of comparison, not a shortcut: DINOv3's random RoPE
rescale is drawn once per forward and guarded by ``self.training``, so in eval
the chunk partition must be numerically inert. This CANNOT and does not claim
train-mode equivalence -- train-mode RoPE draws, gradients and activation
checkpointing are all absent by construction (pre-launch review, finding 1).

FAIL-CLOSED. The arm's training contract is asserted against the config file, and
the orbit is driven by the CONFIG's angle list, so a drifted method, angle list
or cap aborts instead of printing PASS. A tolerance is never relaxed in-script:
if the registered precision cannot meet it, the measured diffs are printed and
the probe exits nonzero for the Planner to judge.

Read-only: writes nothing but stdout. Run from the repo root; the Planner tees it.

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

import eval_FLAC                                                     # noqa: E402
from src.data.dataset import create_dataloader_from_config           # noqa: E402
from src.data.yaw_rotation import (POSE_KEYS, fa_cartesian_conditioning,  # noqa: E402
                                   rotate_scene_metadata)
from src.models.conditioners import (                                # noqa: E402
    create_multi_conditioner_from_conditioning_config)

EXPDIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CONFIG = os.path.join(EXPDIR, "FLAC_AR_BFC.json")
DATASET_CONFIG = os.path.join(REPO, "src", "configs", "dataset_configs", "AR",
                              "train", "acousticroom_train.json")
BATCH = 32                  # the arm's per-rank micro-batch
CAP_EVAL = 64               # the registered EVALUATION cap (exp21_validate_cell)
COND_AUTOCAST = "bf16"      # the registered conditioning precision (announcement 05)
TOL = 1e-5
MIN_FREE_GIB = 8            # co-tenant box: refuse rather than OOM someone else's run
MIN_SCENES = 6              # a C4 claim over one or two rooms is not a claim
SEED = 42
# The arm's training contract, asserted against the config rather than assumed.
REQUIRED_TRAINING = {"cond_method": "fa_cartesian",
                     "frame_avg_angles": [0.0, 90.0, 180.0, 270.0],
                     "frame_avg_max_fwd_samples": 32}
# (shape, dtype) as the AR path actually produces them -- measured, not assumed.
# The four pose keys are float32 because ``AR_md`` calls ``.float()`` on them, and
# that is load-bearing: ``eval_FLAC.sample_context_ids`` REFUSES to fingerprint
# ``context_poses`` in any other dtype. ``depth`` is float64 because it is
# ``np.load``ed from the stored map and nothing downcasts it (the geometry
# conditioner casts to float32 before DINO); pinned so a silent change is visible.
EXPECTED = {"source": ((3,), torch.float32),
            "source_vit": ((1, 3), torch.float32),
            "context_poses": ((8, 3), torch.float32),
            "context_poses_vit": ((8, 3), torch.float32),
            "context_audio": ((8, 1, 9600), torch.float32),
            "depth": ((3, 256, 512), torch.float64)}


def load_training_contract():
    """The arm's config, ASSERTED. A drifted arm must abort, never print PASS."""
    with open(MODEL_CONFIG) as fh:
        model_config = json.load(fh)
    training = model_config.get("training") or {}
    for key, want in REQUIRED_TRAINING.items():
        got = training.get(key, "<absent>")
        assert got == want and type(got) is type(want), (
            f"{os.path.basename(MODEL_CONFIG)} training.{key} is {got!r} "
            f"({type(got).__name__}), the registered arm is {want!r} "
            f"({type(want).__name__}) -- refusing to certify a drifted arm")
    return model_config, training


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
    from one room and the C4 spot-check would be a single-scene claim.
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
    """Every field the arm's conditioners consume: present, right shape, finite."""
    assert len(metadata) == BATCH, f"got {len(metadata)} samples, wanted {BATCH}"
    for i, md in enumerate(metadata):
        for key, (shape, dtype) in EXPECTED.items():
            assert key in md, f"sample {i} has no {key!r}"
            got = md[key]
            assert isinstance(got, torch.Tensor), f"sample {i} {key} is {type(got).__name__}"
            assert tuple(got.shape) == shape, f"sample {i} {key} is {tuple(got.shape)}, want {shape}"
            assert got.dtype == dtype, f"sample {i} {key} is {got.dtype}, want {dtype}"
            assert torch.isfinite(got).all(), f"sample {i} {key} is not finite"
    scenes = sorted({md["scene"] for md in metadata})
    assert len(scenes) >= MIN_SCENES, (
        f"the batch spans only {len(scenes)} room families {scenes}; a C4 claim over "
        f"fewer than {MIN_SCENES} is not a claim about the split")
    print(f"  readback OK: {BATCH} samples, all {len(EXPECTED)} conditioner fields "
          "present, shape+dtype+finite asserted")
    print("  dtypes: " + ", ".join(f"{k}={str(v[1]).replace('torch.', '')}"
                                   for k, v in EXPECTED.items()))
    print(f"  scenes spanned ({len(scenes)} >= {MIN_SCENES}): {scenes}")
    d, p = metadata[0]["depth"], metadata[0]["context_poses"]
    print(f"  depth[0]   {tuple(d.shape)} min {d.min():+.4f} max {d.max():+.4f} std {d.std():.4f}")
    print(f"  ctx_pos[0] {tuple(p.shape)} min {p.min():+.4f} max {p.max():+.4f} std {p.std():.4f}")


def autocast_ctx(mode, device):
    """EXACTLY eval_FLAC's conditioning-precision context for ``mode``.

    Resolved by ``eval_FLAC.resolve_cond_autocast`` itself, so the probe cannot
    drift from the evaluator it is certifying (this is the same two-line dispatch
    ``evaluate_model`` wraps its conditioner call in).
    """
    enabled, dtype = eval_FLAC.resolve_cond_autocast(mode)
    if not enabled:
        import contextlib
        return contextlib.nullcontext()
    return (torch.amp.autocast(device) if dtype is None
            else torch.amp.autocast(device, dtype=dtype))


def condition(conditioner, metadata, device, cap, angles, mode):
    torch.manual_seed(SEED)                    # eval mode draws nothing; pinned anyway
    with torch.no_grad(), autocast_ctx(mode, device):
        return fa_cartesian_conditioning(conditioner, metadata, device, angles,
                                         max_fwd_samples=cap)


def compare(label, a, b, tol):
    """``(ok, worst)`` over every conditioning id -- tensors AND masks."""
    assert set(a) == set(b), f"{label}: id sets differ, {sorted(a)} vs {sorted(b)}"
    worst, ok = 0.0, True
    for key in sorted(a):
        x, y = a[key][0].float(), b[key][0].float()
        assert x.shape == y.shape, f"{label}: {key} shape {x.shape} vs {y.shape}"
        diff = (x - y).abs().max().item()
        worst = max(worst, diff)
        ok = ok and diff <= tol
        # the mask is half the conditioning contract: a replaced or reshaped mask
        # is invisible to a tensor comparison (r2 review §a's lesson, one layer up)
        mx, my = a[key][1], b[key][1]
        if mx is None or my is None:
            assert mx is None and my is None, f"{label}: {key} mask present on one side only"
        else:
            assert mx.shape == my.shape, f"{label}: {key} mask shape {mx.shape} vs {my.shape}"
            assert torch.equal(mx.float(), my.float()), f"{label}: {key} mask values differ"
        print(f"    {label:<22} {key:<20} max|d| {diff:.3e}")
    return ok, worst


def run_precision(conditioner, metadata, rotated, device, angles, train_cap, mode):
    """Cap parity + the 90-degree C4 spot-check at ONE conditioning precision."""
    print(f"  --- conditioning precision: {mode} "
          f"({'autocast bf16' if mode == 'bf16' else 'no autocast, fp32'}) ---")
    out_train = condition(conditioner, metadata, device, train_cap, angles, mode)
    out_eval = condition(conditioner, metadata, device, CAP_EVAL, angles, mode)
    out_rot = condition(conditioner, rotated, device, CAP_EVAL, angles, mode)
    cap_ok, cap_worst = compare(f"[{mode}] cap{train_cap}-vs-cap{CAP_EVAL}",
                                out_train, out_eval, TOL)
    c4_ok, c4_worst = compare(f"[{mode}] c4-90deg", out_eval, out_rot, TOL)
    print(f"  [{mode}] cap-parity {cap_worst:.3e} -> {'OK' if cap_ok else 'VIOLATED'}"
          f" | C4 {c4_worst:.3e} -> {'OK' if c4_ok else 'VIOLATED'}"
          f" | tolerance {TOL:g}")
    return {"cap_ok": cap_ok, "c4_ok": c4_ok, "cap": cap_worst, "c4": c4_worst}


def main():
    model_config, training = load_training_contract()
    angles = tuple(float(a) for a in training["frame_avg_angles"])   # from the CONFIG
    train_cap = int(training["frame_avg_max_fwd_samples"])
    device, why = pick_device()
    print("exp_21 rung 4 -- cap parity + real-data readback")
    print(f"device: {device} ({why}) | torch {torch.__version__} | HF_HUB_OFFLINE="
          f"{os.environ.get('HF_HUB_OFFLINE')}")
    print(f"arm config ASSERTED: cond_method={training['cond_method']!r} "
          f"angles={list(angles)} TRAIN cap={train_cap} (vs eval cap {CAP_EVAL})")
    print(f"chunk plans at batch {BATCH}: cap {train_cap} -> {max(1, train_cap // BATCH)} "
          f"angle/chunk, cap {CAP_EVAL} -> {max(1, CAP_EVAL // BATCH)} angles/chunk")
    torch.set_float32_matmul_precision('medium')          # as evaluate_model does

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

    print(f"[4/4] cap {train_cap} vs {CAP_EVAL} and the 90-degree C4 check, at BOTH "
          "precisions")
    img_w = int(metadata[0]["depth"].shape[-1])
    rotated = [rotate_scene_metadata(md, math.radians(90.0), img_w, POSE_KEYS)
               for md in metadata]
    results = {}
    for mode in (COND_AUTOCAST, "off"):
        results[mode] = run_precision(conditioner, metadata, rotated, device, angles,
                                      train_cap, mode)

    # The two checks are reported SEPARATELY, at each precision, because they are
    # two different claims and can fail independently. Cap parity is rung 4's own
    # deliverable ("cap-64 versus grouped-cap allclose"); the C4 spot-check is the
    # real-data invariance claim.
    ok = all(r["cap_ok"] and r["c4_ok"] for r in results.values())
    for mode, r in results.items():
        tag = "REGISTERED" if mode == COND_AUTOCAST else "fp32 reference"
        print(f"  {tag:<14} ({mode:<4}): cap-parity {r['cap']:.3e} "
              f"{'OK' if r['cap_ok'] else 'VIOLATED'} | C4 {r['c4']:.3e} "
              f"{'OK' if r['c4_ok'] else 'VIOLATED'}")
    print(f"RUNG-4 VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(batch {BATCH}, device {device}, tolerance {TOL:g}, "
          f"angles {list(angles)}, caps {train_cap}/{CAP_EVAL})")
    if not ok:
        # Never self-approve a relaxation: the numbers are reported and the Planner
        # decides. A bf16 mantissa is 8 bits, so ~2**-7 = 7.8e-3 on tensors of this
        # magnitude is one representable step -- but whether that is harmless is a
        # judgement about the DOWNSTREAM metric limits (plan §5 pre-registers T60
        # 0.005 / C50 0.0005 / EDT 0.006 / R@1 0.15), not one this probe may make.
        print("  TOLERANCE NOT RELAXED IN-SCRIPT. For the Planner to judge, per "
              "precision: " + "; ".join(
                  f"{m}: cap {r['cap']:.3e} / C4 {r['c4']:.3e}"
                  for m, r in results.items()))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
