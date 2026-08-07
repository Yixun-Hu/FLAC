#!/usr/bin/env python3
"""exp_04 trained-checkpoint equivariance guard (delta plan §3; base plan §4.5b / §4.6).

exp_04 keeps the ORIGINAL pooling (``pooler_output`` — the patch mean) and changes only the
head, so the condition this guard must certify is ``MLP(mean(tokens))``, computed end-to-end
through ``GeometryConditioner.forward`` at the registered yaw angles.

SCOPE (fail-closed): ``dino_pool == 'mean'`` alone does NOT identify this arm — the LEGACY
mean+bare-Linear arm reports the same pooling. So the guard additionally requires the MLP
head (``Linear -> GELU -> Linear``); a legacy or max-pool build is REFUSED rather than
silently certified as something it is not.

Statistic and bound (r2-N7): the **A2b relative-L2 residual**
``||cond(T_k v) - cond(v)||_F / (||cond(v)||_F + 1e-12)`` (reduced in float64), which must
be ``<= 1e-4`` — the same registered bound the exp_01/exp-09 equivariance criteria use.
The property is ARCHITECTURAL (azimuth roll permutes the token axis; the mean over that
axis is permutation-invariant up to floating-point reduction order; the MLP is pointwise on
the pooled vector), so it must hold at any weights — training can only expose a wiring/gauge
regression, never create one.

Only WHOLE token columns are exact yaws: at 256x512 / patch 16 the token width is 32, so
{90, 180, 270} degrees are 8, 16 and 24 columns. An angle that is not a whole column is
REFUSED rather than silently measured against an interpolation error.

CPU-only and offline by default. Usage:

    HF_HUB_OFFLINE=1 python worklog/worklog_yixun/exp_09_cyl_no_ssl/guard_exp04n_equivariance.py \\
        /n/fs/gatrdp/outputs/exp04n_meanpoolmlp/.../epoch=..-step=40000.ckpt

Exit 0 = PASS (every residual within the bound), non-zero = the guard REFUSES the
checkpoint. The checkpoint is opened read-only; nothing is written.
"""
import argparse
import copy
import json
import math
import os
import sys

# CPU-only (mandate) + dodge the deepspeed op-probe, BEFORE torch is imported anywhere.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _repo_root(p):  # marker-walk (``.git`` is a FILE in a worktree)
    p = os.path.abspath(p)
    while not os.path.exists(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from cylindrical_dinov3 import physical_yaw  # noqa: E402

DEFAULT_BOUND = 1e-4                      # the registered A2b bound (plan §2 / D2)
DEFAULT_ANGLES = (90.0, 180.0, 270.0)     # registered yaws
DEFAULT_CONFIG = os.path.join(HERE, "FLAC_AR_exp04n.json")


# ------------------------------------------------------------------------------------ #
# fixtures: a deterministic, NON-axisymmetric scene
# ------------------------------------------------------------------------------------ #
def nonaxisymmetric_depth(height, width):
    """A geometrically consistent equirectangular XYZ field [3, H, W] whose radius depends
    on azimuth — an axisymmetric field would make the whole test vacuous."""
    j = torch.arange(width, dtype=torch.float32)
    theta = (j + 0.5) * 2.0 * math.pi / width
    i = torch.arange(height, dtype=torch.float32)
    el = (i + 0.5) * math.pi / height - math.pi / 2.0
    theta_g = theta.view(1, width).expand(height, width)
    el_g = el.view(height, 1).expand(height, width)
    d = (3.0 + torch.sin(theta_g) + 0.5 * torch.sin(2.0 * theta_g)
         + 0.25 * torch.cos(3.0 * theta_g))
    return torch.stack([d * torch.cos(el_g) * torch.cos(theta_g),
                        d * torch.cos(el_g) * torch.sin(theta_g),
                        d * torch.sin(el_g)], dim=0).contiguous()


def _rz(alpha):
    c, s = math.cos(alpha), math.sin(alpha)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def base_sample(height, width, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"coord": torch.randn(1, 3, generator=g), "depth": nonaxisymmetric_depth(height, width)}


def yaw_sample(sample, k_tokens, patch_size, width_tokens):
    """Physically yaw a sample by ``k_tokens`` whole token columns: the XYZ field through the
    PACKAGE's ``physical_yaw`` (roll + Rz on x,y) and the pose by the same Rz."""
    alpha = 2.0 * math.pi * k_tokens / width_tokens
    return {"coord": torch.einsum("ij,...j->...i", _rz(alpha), sample["coord"]),
            "depth": physical_yaw(sample["depth"].unsqueeze(0), k_tokens * patch_size).squeeze(0)}


def rel_l2(a, b):
    """A2b statistic: ``||a - b||_F / (||b||_F + 1e-12)``, reduced in float64."""
    a64, b64 = a.detach().to(torch.float64), b.detach().to(torch.float64)
    return (torch.linalg.vector_norm(a64 - b64)
            / (torch.linalg.vector_norm(b64) + 1e-12)).item()


# ------------------------------------------------------------------------------------ #
# the check
# ------------------------------------------------------------------------------------ #
def assert_exp04n_arm(geom):
    """The conditioner must be THIS arm: mean pooling AND the MLP head.

    PLAN-NOTE (minimal fail-closed reading of delta §3): the delta says "identical structure"
    to the exp_03 guard, whose scope check is ``dino_pool == 'max'``. The mirror-image check
    alone would NOT be equivalent here: the LEGACY mean + bare-Linear arm also reports
    ``dino_pool == 'mean'``, so a legacy checkpoint would be silently certified as exp_04.
    The head shape is therefore required as well — the fail-closed reading."""
    pool = getattr(geom, "dino_pool", None)
    if pool != "mean":
        raise RuntimeError(
            f"guard scope: conditioner dino_pool={pool!r}, expected 'mean' — this guard "
            "certifies the exp_04 mean-pool + MLP head, and would otherwise silently certify "
            "a condition the checkpoint does not serve"
        )
    head = getattr(geom, "lin_proj", None)
    if not (isinstance(head, nn.Sequential) and len(head) == 3
            and type(head[0]) is nn.Linear and type(head[1]) is nn.GELU
            and type(head[2]) is nn.Linear):
        raise RuntimeError(
            f"guard scope: conditioning head is {head!r}, expected the MLP "
            "Linear -> GELU -> Linear — mean pooling alone does not distinguish exp_04 from "
            "the LEGACY mean + bare-Linear arm; refuse"
        )


def check_equivariance(geom, angles=DEFAULT_ANGLES, bound=DEFAULT_BOUND,
                       height=256, width=512, seed=0, device="cpu", verbose=True):
    """A2b relative-L2 residual of the SERVED condition at each angle. Returns
    ``{angle: residual}``; raises RuntimeError if the conditioner is not the exp_04 arm,
    if an angle is not a whole token column, or if any residual exceeds ``bound``."""
    assert_exp04n_arm(geom)
    patch_size = geom.vit.config.patch_size
    if width % patch_size or height % patch_size:
        raise RuntimeError(f"geometry {height}x{width} is not divisible by the patch size {patch_size}")
    width_tokens = width // patch_size

    base = base_sample(height, width, seed=seed)
    was_training = geom.training
    geom.eval()
    try:
        with torch.no_grad():
            cond_base = geom([base], device=device)[0]
            residuals = {}
            for angle in angles:
                exact = float(angle) / 360.0 * width_tokens
                k = int(round(exact))
                if abs(exact - k) > 1e-9:
                    raise RuntimeError(
                        f"angle {angle} deg is not a whole token column at width {width} "
                        f"(patch {patch_size} -> {width_tokens} columns): {exact} columns. Only "
                        "whole-column yaws are exact; refuse rather than measure interpolation."
                    )
                cond_rot = geom([yaw_sample(base, k, patch_size, width_tokens)], device=device)[0]
                residuals[float(angle)] = rel_l2(cond_rot, cond_base)
    finally:
        if was_training:
            geom.train()

    worst_angle = max(residuals, key=residuals.get)
    worst = residuals[worst_angle]
    if verbose:
        print("[guard] A2b relative-L2 residual of MLP(mean(tokens)) per angle: "
              + ", ".join(f"{a:g}deg={r:.3e}" for a, r in sorted(residuals.items())))
    if worst > bound:
        raise RuntimeError(
            f"equivariance GUARD FAILED: worst A2b relative-L2 residual {worst:.3e} at "
            f"{worst_angle:g} deg exceeds the registered bound {bound:.0e} — the served "
            "condition is not yaw-invariant"
        )
    return residuals


# ------------------------------------------------------------------------------------ #
# checkpoint loading (mirrors eval_FLAC.py's remap, including the EMA promotion)
# ------------------------------------------------------------------------------------ #
def load_geometry_conditioners(ckpt_path, model_config_path=DEFAULT_CONFIG, device="cpu"):
    """Build the exp04n model from its config, load the checkpoint into it with eval_FLAC's
    key remap (EMA promoted exactly when the config says ``use_ema`` AND EMA keys exist), and
    return its GeometryConditioners. Raises on a load that leaves parameters at random init."""
    from src.models import create_model_from_config

    with open(model_config_path) as f:
        model_config = json.load(f)
    training_config = model_config.get("training", None) or {}

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state_dict = dict(state_dict)
    for key in list(state_dict.keys()):
        if key.startswith("diffusion."):
            state_dict[key.replace("diffusion.", "", 1)] = state_dict.pop(key)
    used_ema = False
    if training_config.get("use_ema", False) and any(
        k.startswith("diffusion_ema.ema_model.") for k in state_dict
    ):
        used_ema = True
        for key in list(state_dict.keys()):
            if key.startswith("diffusion_ema.ema_model."):
                state_dict[key.replace("diffusion_ema.ema_model.", "model.")] = state_dict.pop(key)
    print(f"[guard] weight stream: {'EMA' if used_ema else 'ONLINE'} "
          f"(config use_ema={training_config.get('use_ema', False)})")

    model = create_model_from_config(copy.deepcopy(model_config))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing]
    if real_missing:
        raise RuntimeError(
            f"checkpoint load left {len(real_missing)} parameters at random init, e.g. "
            f"{real_missing[:5]} — refuse to certify"
        )
    model.eval().requires_grad_(False)
    model.to(device)
    geoms = [c for c in model.conditioner.conditioners.values()
             if getattr(c, "name", None) == "GeometryConditioner"]
    if len(geoms) != 2:
        raise RuntimeError(f"expected 2 GeometryConditioners, got {len(geoms)}")
    return geoms


def main(argv=None):
    parser = argparse.ArgumentParser(description="exp04n trained-checkpoint equivariance guard")
    parser.add_argument("ckpt", help="the trained checkpoint to certify (opened read-only)")
    parser.add_argument("--model-config", default=DEFAULT_CONFIG,
                        help="the exp04n model config the checkpoint was trained with")
    parser.add_argument("--angles", default=",".join(str(int(a)) for a in DEFAULT_ANGLES),
                        help="comma-separated registered yaw angles in degrees")
    parser.add_argument("--bound", type=float, default=DEFAULT_BOUND,
                        help="the registered A2b relative-L2 bound (default 1e-4)")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.ckpt):
        raise RuntimeError(f"checkpoint not found: {args.ckpt}")
    angles = tuple(float(a) for a in args.angles.split(","))
    geoms = load_geometry_conditioners(args.ckpt, args.model_config, device=args.device)
    for i, geom in enumerate(geoms):
        print(f"[guard] conditioner {i}: {geom.name} (dino_pool={geom.dino_pool})")
        check_equivariance(geom, angles=angles, bound=args.bound, height=args.height,
                           width=args.width, seed=args.seed, device=args.device)
    print(f"GUARD PASS: the served condition MLP(mean(tokens)) is yaw-invariant within "
          f"{args.bound:.0e} at angles {angles} for {os.path.basename(args.ckpt)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
