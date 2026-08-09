#!/usr/bin/env python3
"""exp_06 trained-checkpoint equivariance guard (plan B3).

The condition this guard certifies is the one the exp06n model ACTUALLY serves,
``Linear(amax(last_hidden_state, dim=1))``, computed end-to-end through
``GeometryConditioner.forward`` at the registered yaw angles.

SCOPE (fail-closed, B3): ``dino_pool == 'max'`` alone does NOT identify this arm — the
exp_03 max+MLP arm reports the same pooling. So the guard additionally requires the
LEGACY bare ``nn.Linear`` head; a max+MLP or legacy-mean build is REFUSED rather than
silently certified as something it is not.

NON-VACUITY, all hard asserts (B3 a-f):
  (a) the yawed input genuinely differs from the unyawed input (the fixture is
      non-axisymmetric, and that is CHECKED, not assumed);
  (b) the composed yaw op (Rz on the pose + roll/rotation of the XYZ field) equals the
      PACKAGE's ``physical_yaw`` of the base input field — two independent code paths
      cross-checked;
  (c) every angle is a WHOLE token column (else REFUSE: only whole-column yaws are exact);
  (d) the backbone token field undergoes the EXPECTED column roll (rel-L2 <= bound);
  (e) the served condition's A2b relative-L2 residual is <= 1e-4 at {90, 180, 270};
  (f) a GAUGE-OFF rebuild of the same checkpoint must BREAK invariance to >= 1e-3 (the
      negative control that gives (e) teeth), and the control itself REFUSES an invariant
      model.

Statistic and bound: the A2b relative-L2 residual
``||cond(T_k v) - cond(v)||_F / (||cond(v)||_F + 1e-12)`` (reduced in float64), bound
1e-4 — the registered exp_01/exp-09 criterion. The property is ARCHITECTURAL (azimuth
roll permutes the token axis; ``amax`` over that axis is permutation-invariant; the
Linear is pointwise on the pooled vector), so it must hold at any weights — training can
only expose a wiring/gauge regression, never create one.

At 256x512 / patch 16 the token width is 32, so {90, 180, 270} degrees are 8, 16 and 24
columns.

CPU-only and offline by default. Usage:

    HF_HUB_OFFLINE=1 python worklog/worklog_yixun/exp_09_cyl_no_ssl/guard_exp06n_equivariance.py \\
        /n/fs/gatrdp/outputs/exp06n_maxpoollinear/.../epoch=..-step=40000.ckpt

Exit 0 = PASS (positive check on BOTH conditioners + the gauge-off negative control),
non-zero = the guard REFUSES the checkpoint. The checkpoint is opened read-only; nothing
is written.
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

DEFAULT_BOUND = 1e-4                      # the registered A2b bound (plan §1)
NEGATIVE_FLOOR = 1e-3                     # 10x the bound: "unmistakably broken" (B3 f)
DEFAULT_ANGLES = (90.0, 180.0, 270.0)     # registered yaws
DEFAULT_CONFIG = os.path.join(HERE, "FLAC_AR_exp06n.json")


# ------------------------------------------------------------------------------------ #
# fixtures: a deterministic, NON-axisymmetric scene
# ------------------------------------------------------------------------------------ #
def nonaxisymmetric_depth(height, width):
    """A geometrically consistent equirectangular XYZ field [3, H, W] whose radius depends
    on azimuth — an axisymmetric field would make the whole test vacuous (checked in (a))."""
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


def input_field(sample, max_value):
    """The [1, 3, H, W] tensor GeometryConditioner.forward feeds the backbone."""
    coord = sample["coord"].float().unsqueeze(0)          # [1, 1, 3]
    depth = sample["depth"].float().unsqueeze(0)          # [1, 3, H, W]
    return (coord[:, 0, :, None, None] - depth) / max_value


def rel_l2(a, b):
    """A2b statistic: ``||a - b||_F / (||b||_F + 1e-12)``, reduced in float64."""
    a64, b64 = a.detach().to(torch.float64), b.detach().to(torch.float64)
    return (torch.linalg.vector_norm(a64 - b64)
            / (torch.linalg.vector_norm(b64) + 1e-12)).item()


# ------------------------------------------------------------------------------------ #
# arm scope (B3: refuses max+MLP and legacy-mean)
# ------------------------------------------------------------------------------------ #
def assert_exp06n_arm(geom):
    """The conditioner must be THIS arm: max pooling AND the LEGACY bare Linear head."""
    pool = getattr(geom, "dino_pool", None)
    if pool != "max":
        raise RuntimeError(
            f"guard scope: conditioner dino_pool={pool!r}, expected 'max' — this guard "
            "certifies the exp_06 max-pool + bare-Linear head, and would otherwise silently "
            "certify a condition the checkpoint does not serve"
        )
    head = getattr(geom, "lin_proj", None)
    if type(head) is not nn.Linear:
        raise RuntimeError(
            f"guard scope: conditioning head is {head!r}, expected the LEGACY bare "
            "nn.Linear — max pooling alone does not distinguish exp_06 from the exp_03 "
            "max+MLP arm; refuse"
        )


# ------------------------------------------------------------------------------------ #
# the checks
# ------------------------------------------------------------------------------------ #
def _whole_column(angle, width_tokens, width, patch_size):
    exact = float(angle) / 360.0 * width_tokens
    k = int(round(exact))
    if abs(exact - k) > 1e-9:
        raise RuntimeError(
            f"angle {angle} deg is not a whole token column at width {width} "
            f"(patch {patch_size} -> {width_tokens} columns): {exact} columns. Only "
            "whole-column yaws are exact; refuse rather than measure interpolation."
        )
    return k


def _checked_yaw(geom, base, angle, k, patch_size, base_field):
    """Build the yawed sample for ``angle`` and hard-assert fixture fidelity (B3 a+b)."""
    width_tokens = base["depth"].shape[-1] // patch_size
    rot = yaw_sample(base, k, patch_size, width_tokens)
    rot_field = input_field(rot, geom.max_value)
    if torch.allclose(rot_field, base_field, atol=1e-3):
        raise RuntimeError(
            f"fixture non-vacuity: the {angle:g}-deg yawed input is indistinguishable from "
            "the base input — the invariance measurement would be vacuous (is the depth "
            "field axisymmetric?)"
        )
    if not torch.allclose(rot_field, physical_yaw(base_field, k * patch_size), atol=1e-5):
        raise RuntimeError(
            f"fixture fidelity: the {angle:g}-deg yaw op does NOT equal the package "
            "physical_yaw of the base input field — the harness is not measuring the "
            "registered transformation; refuse"
        )
    return rot, rot_field


def _roll_tokens(tokens, k, height_tokens, width_tokens):
    b, n, c = tokens.shape
    if n != height_tokens * width_tokens:
        raise RuntimeError(f"token count {n} != {height_tokens}x{width_tokens}")
    return torch.roll(tokens.view(b, height_tokens, width_tokens, c),
                      shifts=k, dims=2).reshape(b, n, c)


def check_equivariance(geom, angles=DEFAULT_ANGLES, bound=DEFAULT_BOUND,
                       height=256, width=512, seed=0, device="cpu", verbose=True):
    """POSITIVE check: A2b relative-L2 residual of the SERVED condition at each angle,
    with the B3 non-vacuity asserts (a)-(d) hard-enforced per angle. Returns
    ``{angle: residual}``; raises RuntimeError on any refusal."""
    assert_exp06n_arm(geom)
    patch_size = geom.vit.config.patch_size
    if width % patch_size or height % patch_size:
        raise RuntimeError(f"geometry {height}x{width} is not divisible by the patch size {patch_size}")
    width_tokens = width // patch_size
    height_tokens = height // patch_size

    base = base_sample(height, width, seed=seed)
    base_field = input_field(base, geom.max_value)
    was_training = geom.training
    geom.eval()
    try:
        with torch.no_grad():
            cond_base = geom([base], device=device)[0]
            tokens_base = geom.vit(base_field.to(device)).last_hidden_state
            residuals = {}
            for angle in angles:
                k = _whole_column(angle, width_tokens, width, patch_size)
                rot, rot_field = _checked_yaw(geom, base, angle, k, patch_size, base_field)
                # (d) the token field must match the EXPECTED column roll
                tokens_rot = geom.vit(rot_field.to(device)).last_hidden_state
                roll_res = rel_l2(tokens_rot, _roll_tokens(tokens_base, k,
                                                           height_tokens, width_tokens))
                if roll_res > bound:
                    raise RuntimeError(
                        f"token-field roll assert: at {angle:g} deg the backbone tokens are "
                        f"NOT the k={k} column roll of the base tokens (rel-L2 {roll_res:.3e} "
                        f"> {bound:.0e}) — the premise of max-pool invariance is broken"
                    )
                cond_rot = geom([rot], device=device)[0]
                residuals[float(angle)] = rel_l2(cond_rot, cond_base)
    finally:
        if was_training:
            geom.train()

    worst_angle = max(residuals, key=residuals.get)
    worst = residuals[worst_angle]
    if verbose:
        print("[guard] A2b relative-L2 residual of Linear(amax(tokens)) per angle: "
              + ", ".join(f"{a:g}deg={r:.3e}" for a, r in sorted(residuals.items())))
    if worst > bound:
        raise RuntimeError(
            f"equivariance GUARD FAILED: worst A2b relative-L2 residual {worst:.3e} at "
            f"{worst_angle:g} deg exceeds the registered bound {bound:.0e} — the served "
            "condition is not yaw-invariant"
        )
    return residuals


def check_gauge_off_negative_control(geom, angles=DEFAULT_ANGLES, floor=NEGATIVE_FLOOR,
                                     height=256, width=512, seed=0, device="cpu",
                                     verbose=True):
    """NEGATIVE control (B3 f): on a GAUGE-OFF build of the same arm the same fixture must
    BREAK invariance to >= ``floor`` (10x the bound). Fed an invariant (gauge-on) model it
    REFUSES — a control that measures nothing must never report success. The token-roll
    assert (d) is deliberately absent here: gauge-off, the roll is EXPECTED to fail."""
    assert_exp06n_arm(geom)
    patch_size = geom.vit.config.patch_size
    if width % patch_size or height % patch_size:
        raise RuntimeError(f"geometry {height}x{width} is not divisible by the patch size {patch_size}")
    width_tokens = width // patch_size

    base = base_sample(height, width, seed=seed)
    base_field = input_field(base, geom.max_value)
    was_training = geom.training
    geom.eval()
    try:
        with torch.no_grad():
            cond_base = geom([base], device=device)[0]
            residuals = {}
            for angle in angles:
                k = _whole_column(angle, width_tokens, width, patch_size)
                rot, _ = _checked_yaw(geom, base, angle, k, patch_size, base_field)
                cond_rot = geom([rot], device=device)[0]
                residuals[float(angle)] = rel_l2(cond_rot, cond_base)
    finally:
        if was_training:
            geom.train()

    worst = max(residuals.values())
    if verbose:
        print("[guard] NEGATIVE CONTROL (gauge-off) residual per angle: "
              + ", ".join(f"{a:g}deg={r:.3e}" for a, r in sorted(residuals.items())))
    if worst < floor:
        raise RuntimeError(
            f"negative control has NO TEETH: gauge-off worst residual {worst:.3e} < "
            f"{floor:.0e} — the model is still invariant, so the positive check above "
            "proves nothing (was the control fed a gauge-ON build?)"
        )
    return residuals


# ------------------------------------------------------------------------------------ #
# checkpoint loading (mirrors eval_FLAC.py's remap, including the EMA promotion)
# ------------------------------------------------------------------------------------ #
def load_geometry_conditioners(ckpt_path, model_config_path=DEFAULT_CONFIG, device="cpu",
                               gauge_override=None):
    """Build the exp06n model from its config, load the checkpoint into it with eval_FLAC's
    key remap (EMA promoted exactly when the config says ``use_ema`` AND EMA keys exist), and
    return its GeometryConditioners. ``gauge_override`` (e.g. ``"none"``) rebuilds the SAME
    weights with the gauge forced off — the negative-control model (the gauge is
    parameter-free, so the state dict is unchanged). Raises on a load that leaves
    parameters at random init."""
    from src.models import create_model_from_config

    with open(model_config_path) as f:
        model_config = json.load(f)
    training_config = model_config.get("training", None) or {}
    if gauge_override is not None:
        n = 0
        for c in model_config["model"]["conditioning"]["configs"]:
            if c["type"] == "ViTCoordinates":
                c["config"]["ViT"]["gauge"] = gauge_override
                n += 1
        if n != 2:
            raise RuntimeError(f"gauge_override: patched {n} ViT blocks, expected 2")
        print(f"[guard] gauge_override={gauge_override!r} on both ViT blocks "
              "(negative-control build)")

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
    parser = argparse.ArgumentParser(description="exp06n trained-checkpoint equivariance guard")
    parser.add_argument("ckpt", help="the trained checkpoint to certify (opened read-only)")
    parser.add_argument("--model-config", default=DEFAULT_CONFIG,
                        help="the exp06n model config the checkpoint was trained with")
    parser.add_argument("--angles", default=",".join(str(int(a)) for a in DEFAULT_ANGLES),
                        help="comma-separated registered yaw angles in degrees")
    parser.add_argument("--bound", type=float, default=DEFAULT_BOUND,
                        help="the registered A2b relative-L2 bound (default 1e-4)")
    parser.add_argument("--negative-floor", type=float, default=NEGATIVE_FLOOR,
                        help="the gauge-off control must reach at least this (default 1e-3)")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.ckpt):
        raise RuntimeError(f"checkpoint not found: {args.ckpt}")
    angles = tuple(float(a) for a in args.angles.split(","))

    # POSITIVE check on BOTH conditioners of the real (gauge-on) build
    geoms = load_geometry_conditioners(args.ckpt, args.model_config, device=args.device)
    for i, geom in enumerate(geoms):
        print(f"[guard] conditioner {i}: {geom.name} (dino_pool={geom.dino_pool})")
        check_equivariance(geom, angles=angles, bound=args.bound, height=args.height,
                           width=args.width, seed=args.seed, device=args.device)

    # NEGATIVE control (B3 f): the SAME checkpoint, gauge OFF, must break invariance
    geoms_off = load_geometry_conditioners(args.ckpt, args.model_config, device=args.device,
                                           gauge_override="none")
    check_gauge_off_negative_control(geoms_off[0], angles=angles, floor=args.negative_floor,
                                     height=args.height, width=args.width, seed=args.seed,
                                     device=args.device)

    print(f"GUARD PASS: the served condition Linear(amax(tokens)) is yaw-invariant within "
          f"{args.bound:.0e} at angles {angles} for {os.path.basename(args.ckpt)}; the "
          f"gauge-off negative control broke invariance (>= {args.negative_floor:.0e}) as required")
    return 0


if __name__ == "__main__":
    sys.exit(main())
