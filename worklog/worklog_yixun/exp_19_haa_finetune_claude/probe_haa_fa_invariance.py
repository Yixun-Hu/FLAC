#!/usr/bin/env python3
"""exp_19 R1 — is ``fa_invariant`` conditioning C4-invariant on HAA metadata?

**The gate.** Frame averaging was built and proven on AcousticRooms, whose depth
panorama is rendered at the LISTENER. HAA reverses the roles: the panorama is
rendered at the SOURCE, the raw map is flipped vertically (``HAA_md.py:46``,
``np.flipud``) and the poses are receivers expressed in the source-centred frame
(``HAA_md.py:64-71``). The C4 machinery has never been run against that
convention. If any part of it interacts with the rotation, the HAA-BF arm would
train against conditioning that is not invariant while every log looks healthy —
so plan §3 R1 makes this probe a HARD launch gate: it runs before HAA-BF starts,
and a failure STOPS the arm. It does not "fix" a sign; the convention is the
released one and changing it is Yixun's call, not the probe's.

**Structure.** The measurable core is pure and dataset-free:

    invariance_gap(cond_fn, md, angles) -> float

It walks the orbit with the repo's own :func:`rotate_scene_metadata`, calls
``cond_fn`` on each rotated copy, and returns the largest absolute difference
between ANY TWO orbit elements — an invariant function must give the same answer
whichever element you start from, so the pairwise maximum is the honest statistic
(against the angle-0 pass alone it can be under-reported by up to 2x). The CLI
half supplies the real ``cond_fn``: FLAC's conditioner stack driven through
:func:`invariant_conditioning`, on one real HAA batch. Only the CLI imports model
code, so the unit tests never pull in DINOv3.

**Reading the number.** ``THRESHOLD`` is 1e-5 (plan §3 R1). Exact invariance is
algebraic — the frame average sums the same terms in the same order for every
orbit element — so the residual is float noise from composing rotations, which on
float32 conditioning tensors sits far below 1e-5. The probe therefore runs in
FLOAT32: under ``--cond-autocast bf16`` (the evaluation default for the FA arms)
bf16's ~3e-3 resolution would swamp the gate, and a gate that cannot fail is not
a gate. Precision is reported in the record.

⚠️ **What this gate does NOT certify** (Codex exp_19 r1, finding 4 — a standing
limitation, not a bug to be fixed by more code). The subject and the oracle share
one primitive: the orbit walked here calls :func:`rotate_scene_metadata`, and the
frame average inside ``invariant_conditioning`` calls the very same function. A
sign or gauge error living IN that primitive therefore moves both sides together
and remains algebraically invisible — averaging over a group is invariant whether
or not the group action is the physically correct one. What a PASS does certify,
on real HAA data and the arm's own weights:

  * the fa/rotate pipeline runs end to end on HAA-shaped metadata (source-position
    panorama, ``np.flipud`` map, source-centred poses) without shape, dtype, mask
    or key-set breakage;
  * the frame average is exactly invariant on the C4 orbit, per conditioning id
    AND per mask, at 1e-5 on the real conditioner tensors — i.e. the machinery
    composes and the averaging arithmetic holds at HAA's shapes.

Gauge correctness itself rests elsewhere and is cited, not re-derived here: the
planner's code reading in ``plan_haa_finetune.md`` §3 R1 (HAA and AR share
``convert_equirect_to_camera_coord`` byte for byte; HAA's only differences are the
vertical ``np.flipud`` and the source/listener role swap, neither of which touches
the azimuth about which the rotation acts), together with FA's own sign-closure
argument — ``yaw_transform_consistency`` checks that a rotated panorama's stored
per-pixel azimuths still match their columns, which a roll/rotation sign
disagreement breaks. Read the PASS as "the pipeline is consistent here", never as
"the convention is proven correct".

Usage (gate):
    HF_HUB_OFFLINE=1 python worklog/worklog_yixun/exp_19_haa_finetune_claude/probe_haa_fa_invariance.py \
        --model-config <arm config> --ckpt-path <the arm's extracted init>
``--ckpt-path`` is REQUIRED: the gate must measure the conditioner weights the arm
will actually start from, loaded through train.py's own consumer path. A freshly
constructed stack has random panorama/pose coupling and is not the thing being
launched (r1 finding 5).

Exit code 0 = invariant within threshold; 1 = FAILED (stop the arm); 2 = refused.

Written by the exp_19 coder seat (Claude Opus 5, max effort).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)          # repo root before any stale site-packages copy

from src.data.yaw_rotation import DEFAULT_FRAME_ANGLES, rotate_scene_metadata  # noqa: E402

# The orbit is the repo's, not a private copy: B-F trains on DEFAULT_FRAME_ANGLES
# and a probe that gated a different orbit would gate a different arm.
C4_ANGLES = DEFAULT_FRAME_ANGLES

# Plan §3 R1. Also the parser default, so a flagless run IS the registered gate.
THRESHOLD = 1e-5

DEFAULT_MODEL_CONFIG = os.path.join(
    ROOT, "src", "configs", "model_configs", "FLAC", "HAA", "FLAC_HAA_finetune.json")
DEFAULT_DATASET_CONFIG = os.path.join(
    ROOT, "src", "configs", "dataset_configs", "HAA", "eval", "haa_test.json")


# --------------------------------------------------------------------------- #
# pure core
# --------------------------------------------------------------------------- #
def check_closed_orbit(angles, tol: float = 1e-9) -> None:
    """Raise unless ``angles`` (degrees) is a finite subgroup of the yaw rotations.

    Frame averaging is invariant *because* it averages over a GROUP: for
    ``G`` closed under composition, ``(1/|G|) sum_g f(g h x)`` re-indexes to the
    same sum for every ``h`` in ``G``. Over a set that is not closed — {0, 90, 180}
    is the tempting one — the average is invariant under nothing, so the probe
    would measure a real, large, meaningless gap and the gate would fail for a
    reason no one could act on. Identity-first is required as well: it is
    :func:`invariant_conditioning`'s own contract (``angles[0] must be 0.0``), and
    a probe accepting an orbit the trainer rejects would gate an unreachable
    configuration.
    """
    if isinstance(angles, (str, bytes)) or not hasattr(angles, "__iter__"):
        raise ValueError(f"angles must be a sequence of degrees, got {angles!r}")
    angles = list(angles)
    if not angles:
        raise ValueError("angles must be non-empty (the orbit needs at least the identity)")

    vals = []
    for a in angles:
        if isinstance(a, bool) or not isinstance(a, (int, float)):
            raise ValueError(f"angles must be numbers in degrees, got {a!r}")
        if not math.isfinite(float(a)):
            raise ValueError(f"angles must be finite, got {a!r}")
        vals.append(float(a) % 360.0)

    if abs(vals[0]) > tol:
        raise ValueError(
            f"angles[0] must be 0.0 (the identity pass), got {angles[0]!r} — this is "
            "invariant_conditioning's contract, not a convention of this probe")

    for i, a in enumerate(vals):
        for j in range(i + 1, len(vals)):
            if abs(a - vals[j]) <= tol:
                raise ValueError(
                    f"duplicate orbit element {angles[i]!r} / {angles[j]!r}: the "
                    "average would weight one group element twice")

    for a in vals:
        for b in vals:
            composed = (a + b) % 360.0
            if not any(min(abs(composed - c), 360.0 - abs(composed - c)) <= tol for c in vals):
                raise ValueError(
                    f"angles are not closed under composition: {a} + {b} = {composed} "
                    f"deg is not in {sorted(vals)} — this is not a group, so a frame "
                    "average over it is invariant under nothing")


def _sha256_file(path, chunk=1 << 20):
    """Content sha of the measured init, for the gate's provenance record."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _flatten_outputs(out, path="out"):
    """Flatten a conditioning output into ``{path: tensor}``.

    Accepts a bare tensor, the conditioner's ``{id: [tensor, mask]}``, and any
    nesting of dicts/lists of those. ``None`` entries (an absent mask) contribute
    nothing. Everything else raises: silently ignoring an unrecognised entry is
    how a probe ends up measuring fewer tensors than it reports.
    """
    if out is None:
        return {}
    if torch.is_tensor(out):
        return {path: out}
    if isinstance(out, dict):
        flat = {}
        for key, value in out.items():
            flat.update(_flatten_outputs(value, f"{path}[{key}]"))
        return flat
    if isinstance(out, (list, tuple)):
        flat = {}
        for i, value in enumerate(out):
            flat.update(_flatten_outputs(value, f"{path}[{i}]"))
        return flat
    raise TypeError(
        f"cond_fn returned {type(out).__name__} at {path}; expected tensors, or "
        "dicts/lists of tensors (the conditioner's {id: [tensor, mask]} shape)")


def invariance_gaps(cond_fn, md, angles=C4_ANGLES, img_w=None):
    """Per-output-entry orbit gaps for one metadata sample.

    Parameters
    ----------
    cond_fn : callable
        ``md -> conditioning``. In the gate this is FLAC's conditioner stack under
        :func:`invariant_conditioning`; in the tests, a synthetic stand-in.
    md : dict
        ONE sample's metadata (``depth`` [3,H,W] plus the pose fields). Never
        mutated: each orbit element is a fresh shallow copy from
        :func:`rotate_scene_metadata`.
    angles : sequence of float
        Orbit in degrees, identity first, closed under composition.
    img_w : int, optional
        Panorama width. ``None`` reads it from ``md['depth']``; it is what turns an
        angle into an exact integer column roll, so it is never guessed.

    Returns
    -------
    dict
        ``{flattened output path: max |difference| between any two orbit elements}``
        computed in float64 (a bf16/fp16 difference would be quantised by the very
        precision the gate is trying to resolve).
    """
    check_closed_orbit(angles)

    if img_w is None:
        depth = md.get("depth") if isinstance(md, dict) else None
        if not torch.is_tensor(depth):
            raise ValueError(
                "img_w could not be resolved: the metadata carries no 'depth' tensor. "
                "Pass img_w explicitly — the column roll must be exact, so the width "
                "is never assumed.")
        img_w = int(depth.shape[-1])
    img_w = int(img_w)
    if img_w <= 0:
        raise ValueError(f"img_w must be > 0, got {img_w}")

    flats = []
    for deg in angles:
        rotated = rotate_scene_metadata(md, math.radians(float(deg)), img_w)
        flats.append(_flatten_outputs(cond_fn(rotated)))

    base = flats[0]
    if not base:
        raise ValueError(
            "cond_fn produced no tensors: the probe would report a gap of 0.0 while "
            "measuring nothing, which is the one way this gate can pass falsely")

    for deg, flat in zip(angles[1:], flats[1:]):
        if set(flat) != set(base):
            missing = sorted(set(base) ^ set(flat))
            raise ValueError(
                f"cond_fn must return the same structure at every orbit angle; at "
                f"{deg} deg these entries differ: {missing[:5]}")

    gaps = {}
    for key in base:
        values = []
        for deg, flat in zip(angles, flats):
            tensor = flat[key]
            if tuple(tensor.shape) != tuple(base[key].shape):
                raise ValueError(
                    f"{key}: cond_fn returned shape {tuple(tensor.shape)} at {deg} deg "
                    f"but {tuple(base[key].shape)} at {angles[0]} deg")
            values.append(tensor.detach().to(torch.float64))
        worst = 0.0
        if values[0].numel():
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    worst = max(worst, float((values[i] - values[j]).abs().max().item()))
        gaps[key] = worst
    return gaps


def invariance_gap(cond_fn, md, angles=C4_ANGLES, img_w=None) -> float:
    """The worst orbit gap over every entry of ``cond_fn``'s output (see above)."""
    return max(invariance_gaps(cond_fn, md, angles, img_w).values())


# --------------------------------------------------------------------------- #
# CLI: the same measurement on the real HAA stack
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG,
                    help="FLAC model config whose conditioner stack is probed")
    ap.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG,
                    help="HAA dataset config supplying real metadata")
    ap.add_argument("--ckpt-path", default=None,
                    help="the arm's extracted init (REQUIRED): the gate measures "
                         "the conditioner weights the arm will actually start from")
    ap.add_argument("--angles", type=float, nargs="+", default=list(C4_ANGLES),
                    help="orbit in degrees, identity first (default: the C4 orbit B-F trains on)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="max tolerated orbit gap (plan R1: 1e-5)")
    ap.add_argument("--num-samples", type=int, default=4,
                    help="how many HAA samples to probe")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--out", default=os.path.join(HERE, "probe_haa_fa_invariance_result.json"),
                    help="JSON record of the gate")
    return ap


def _build_stack(model_config_path, dataset_config_path, ckpt_path, device):
    """The ARM's conditioner (init loaded) + one HAA batch.

    ``ckpt_path`` is loaded through train.py's own consumer path — the same
    ``load_ckpt_state_dict`` + prefix transforms + ``load_state_dict(strict=True)``
    that ``train.py:139-148`` performs — so the tensors measured are the ones the
    finetune starts from. Before this, the gate built a fresh model, whose
    panorama/pose coupling is random; since the blind spot documented in the tests
    is exactly that a POSE-LINEAR conditioner cannot reveal a rotation error, the
    learned nonlinear coupling is the part worth measuring (Codex r1 finding 5).

    Imported lazily: the pure core must stay importable, and unit-testable,
    without transformers, DINOv3 or the dataset.
    """
    from src.data.dataset import create_dataloader_from_config
    from src.models.factory import create_model_from_config
    from src.models.utils import load_ckpt_state_dict

    with open(model_config_path) as f:
        model_config = json.load(f)
    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    model = create_model_from_config(model_config)

    weights = load_ckpt_state_dict(ckpt_path)                       # train.py:141
    weights = {k.replace('diffusion.', ''): v for k, v in weights.items()}      # 142
    weights = {k.replace('autoencoder.', ''): v for k, v in weights.items()}    # 143
    weights = {k: v for k, v in weights.items() if 'discriminator' not in k}    # 146
    weights = {k: v for k, v in weights.items() if 'losses' not in k}           # 147
    model.load_state_dict(weights, strict=True)                                 # 148
    print(f"  init   : {ckpt_path} loaded strictly ({len(weights)} tensors)")

    conditioner = model.conditioner.to(device)
    conditioner.eval().requires_grad_(False)   # eval mode also disables DINOv3's
                                               # random RoPE rescale, which would
                                               # otherwise inject per-forward noise

    dl = create_dataloader_from_config(
        dataset_config,
        batch_size=1,
        sample_size=model_config["sample_size"],
        sample_rate=model_config["sample_rate"],
        audio_channels=model_config.get("audio_channels", 1),
        num_workers=0,
        shuffle=False,
    )
    return conditioner, dl


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    angles = tuple(float(a) for a in args.angles)
    try:
        check_closed_orbit(angles)
    except ValueError as e:
        print(f"probe_haa_fa_invariance REFUSED: {e}", file=sys.stderr)
        return 2

    # The gate must measure the ARM's weights, not a freshly initialised stack
    # (Codex r1 finding 5). No default, and no "measure something anyway" path.
    if not args.ckpt_path:
        print("probe_haa_fa_invariance REFUSED: --ckpt-path is required — the gate "
              "measures the conditioner weights the arm starts from, and a freshly "
              "constructed stack has random panorama/pose coupling", file=sys.stderr)
        return 2

    from src.data.yaw_rotation import invariant_conditioning

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"exp_19 R1 gate — fa_invariant conditioning on HAA metadata")
    print(f"  model  : {args.model_config}")
    print(f"  data   : {args.dataset_config}")
    print(f"  orbit  : {angles}   threshold {args.threshold:g}   device {device}")
    print(f"  dtype  : float32 (no autocast — bf16 cannot resolve a 1e-5 gate)")
    print( "  scope  : certifies pipeline/shape/mask consistency on real HAA data;"
           " subject and oracle share rotate_scene_metadata, so a shared gauge"
           " error is invisible here (see the module docstring + plan §3 R1)")

    try:
        conditioner, dl = _build_stack(args.model_config, args.dataset_config,
                                       args.ckpt_path, device)
    except Exception as e:                       # a stack we cannot build is not a pass
        print(f"probe_haa_fa_invariance REFUSED: could not build the stack: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2

    def cond_fn(sample):
        """Every measured output, MASKS INCLUDED (r1 non-blocking finding).

        A mask that moved with the rotation would change which tokens the DiT
        attends to just as surely as a moved feature would."""
        out = invariant_conditioning(conditioner, [sample], device, angles)
        measured = {}
        for cid, entry in out.items():
            tensor = entry[0]
            mask = entry[1] if len(entry) > 1 else None
            measured[cid] = [
                tensor.detach().float().cpu(),
                mask.detach().cpu() if torch.is_tensor(mask) else mask,
            ]
        return measured

    records, worst = [], 0.0
    with torch.no_grad():
        for batch in dl:
            _, metadata = batch
            for md in metadata:
                gaps = invariance_gaps(cond_fn, md, angles)
                worst = max(worst, max(gaps.values()))
                records.append({"scene": md.get("scene"),
                                "gaps": {k: v for k, v in sorted(gaps.items())}})
                for key, value in sorted(gaps.items()):
                    print(f"    {md.get('scene')}  {key}: {value:.3e}")
                if len(records) >= args.num_samples:
                    break
            if len(records) >= args.num_samples:
                break

    if not records:
        print("probe_haa_fa_invariance REFUSED: the dataloader yielded no samples",
              file=sys.stderr)
        return 2

    passed = worst <= args.threshold
    result = {
        "passed": bool(passed),
        "worst_gap": worst,
        "threshold": args.threshold,
        "angles": list(angles),
        "model_config": args.model_config,
        "dataset_config": args.dataset_config,
        # Which weights were measured — the record is worthless if the gate and
        # the launch cannot be shown to refer to the same init.
        "ckpt_path": args.ckpt_path,
        "ckpt_sha256": _sha256_file(args.ckpt_path),
        "device": str(device),
        "dtype": "float32",
        "measures_masks": True,
        "scope": ("pipeline/shape/mask consistency on real HAA data with the arm's "
                  "own weights; subject and oracle share rotate_scene_metadata, so a "
                  "shared gauge error is not detectable here (plan §3 R1)"),
        "n_samples": len(records),
        "per_sample": records,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  record : {args.out}")

    print(f"  worst orbit gap: {worst:.3e}  (threshold {args.threshold:g})")
    if passed:
        print("  R1 GATE PASS — fa_invariant conditioning is C4-invariant on HAA metadata")
        return 0
    print("  R1 GATE FAIL — do NOT launch HAA-BF; report to Yixun (plan §3 R1: the "
          "sign convention must not be silently changed)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
