#!/usr/bin/env python
"""exp-09 Stage A -- real-data convention audit (gauge gate).

Governing plan: ``worklog/worklog_yixun/exp_06_flac_no_ssl_claude/plan_flac_no_ssl.md``
(Revision 4.1, APPROVED), section 1 (Stage A, gates A1/A2a-A2d). This module is the
authority-implementing runner; the plan text is authoritative over any comment here.

WHAT THIS DOES
--------------
The cylindrical DINOv3 gauge (``cylindrical_dinov3.gauge``) declares a fixed convention
(azimuth increases with column, Rz CCW about +z, channels (x,y,z)=(0,1,2)). Its own
tests cannot see FLAC's real data. This audit closes that gap on ONE real, fingerprinted,
non-degenerate AcousticRooms sample built by FLAC's actual metadata code:

* **A1** (static) -- re-derive and serialise the convention facts, quoting the live source
  lines (``AR_md.convert_equirect_to_camera_coord``; ``yaw_rotation.azimuth_rotation_matrix``
  + ``rotate_scene_metadata``; the gauge channel contract).
* **A2a** (pre-model residual) -- the encoder input captured from the REAL
  ``GeometryConditioner.forward`` on the REAL-``rotate_scene_metadata``-rotated sample equals
  ``Rz(alpha_j) . Roll_{16j}`` of the base captured field, rel-err <= 1e-6. Proves the DATA
  transform is the gauge's assumed physical yaw ``T_k`` before any model runs.
* **A2b** (model equivariance) -- the gauge-ON official-weights model is pooled-invariant AND
  patch roll-equivariant (undo the expected token-column roll of ``16j/16 = j`` columns),
  each rel-err <= 1e-4.
* **A2c** (negative controls) -- gauge-OFF on the same pair, and channel-permuted (y,z,x) input
  with gauge-ON: patch AND pooled residuals each >= 1e-2 (proves the A2b test is sensitive).
* **A2d** (three-way status) -- TOTAL over every gate. ANY validity failure (A2a residual,
  either control, fingerprint mismatch, non-degeneracy, non-finite value, capture/hook failure,
  provenance mismatch, serialisation failure) => ``invalid_infrastructure`` (exit 3). Validity
  established but A2b failing => ``valid_convention_failure`` (exit 2, selects gauge-OFF). All
  pass => ``valid_pass`` (exit 0, selects gauge-ON).

Output: atomic finite-JSON ``audit_convention.json`` (tmp + ``os.replace``; ``allow_nan=False``;
non-finite values are recorded as ``null`` with the field name flagged, never written as NaN/Inf).

CPU FP32 eager only. Reads data/checkpoints read-only. This runner is executed by the Planner
under records-before-run; the Coder deliverable is this code + tests + tiny-fixture evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import typing as tp
from pathlib import Path

import numpy as np
import torch

# --- make the FLAC worktree importable (`import src.*`) regardless of CWD -------------------
# This file lives at <worktree>/worklog/worklog_yixun/exp_09_cyl_no_ssl/audit_convention.py,
# so the worktree root is parents[3].
_WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))


# ============================================================================================
# Registered parameters (plan sec 1, A2). Serialised into the audit record.
# ============================================================================================
PATCH_SIZE = 16                      # DINOv3 vits16 patch; 16 px == 1 token column.
ROLL_MULTIPLE = 16                   # alpha_j uses 16-px-multiple yaws (16j px == j token cols).
J_VALUES: tp.Tuple[int, ...] = (1, 4, 8, 16, 24)   # plan: j in {1,4,8,16,24}; j=4 -> 45deg, j=8 -> 90deg.
A2A_RTOL = 1e-6                      # pre-model residual bound (FP32).
A2B_RTOL = 1e-4                      # model pooled-invariance & patch-equivariance bound.
CONTROL_MIN = 1e-2                   # negative controls must each be at least this large.
GEOM_MAX_VALUE = 1.0                 # FLAC_AR source_vit conditioner max_value (config: max_value=1).
CHANNELS = {"x": 0, "y": 1, "z": 2}  # fixed (x,y,z)=(0,1,2) contract.

# Status -> process exit code (plan sec 1: 0 iff valid_pass; 2 valid_convention_failure; 3 invalid).
STATUS_VALID_PASS = "valid_pass"
STATUS_CONVENTION_FAILURE = "valid_convention_failure"
STATUS_INVALID = "invalid_infrastructure"
EXIT_CODES = {STATUS_VALID_PASS: 0, STATUS_CONVENTION_FAILURE: 2, STATUS_INVALID: 3}

# Default read-only locations.
DEFAULT_CHECKPOINT = (
    "/home/yixunhu/.cache/huggingface/hub/models--facebook--dinov3-vits16-pretrain-lvd1689m/"
    "snapshots/114c1379950215c8b35dfcd4e90a5c251dde0d32"
)
DEFAULT_CYL_REPO = "/home/yixunhu/codespace/cylindrical-dinov3"
# AcousticRooms data roots to try, in order. The live FLAC checkout is DELIBERATELY EXCLUDED
# (a training run executes from it and must never be touched); these are equivalent read-only
# copies of the same dataset. A root is accepted only if the chosen sample's metadata json and
# depth npy both exist and are readable.
DEFAULT_DATA_ROOTS: tp.Tuple[str, ...] = (
    "/home/yixunhu/codespace/xRIR_code/data",
)
# Canonical pinned sample (first reachable train.json entry). Deterministic + reproducible.
DEFAULT_SAMPLE = {"scene": "Cafe", "sub": "Cafe_idx_0", "wav": "S001_R0044_hybrid_IR.wav"}
AR_MD_REL = "src/configs/dataset_configs/custom_metadata/AR_md.py"
YAW_ROTATION_REL = "src/data/yaw_rotation.py"


class InvalidInfrastructure(Exception):
    """Raised for any condition that must map to ``invalid_infrastructure`` (plan A2d)."""


class _CaptureAbort(Exception):
    """Internal sentinel: raised from the ViT forward-pre-hook once the encoder input is
    captured, to abort the (unneeded) ViT forward. Keeps the capture independent of the ViT."""


# ============================================================================================
# Geometry primitives -- the gauge's declared convention, re-implemented INDEPENDENTLY of
# FLAC's yaw_rotation so that A2a is a genuine cross-check (pipeline vs. assumed convention),
# not a tautology.
# ============================================================================================
def rz_matrix(alpha_rad: float) -> torch.Tensor:
    """CCW rotation about +z, per the gauge contract (``gauge.py`` docstring):
    ``x' = cos*x - sin*y``, ``y' = sin*x + cos*y``, ``z' = z``."""
    c, s = math.cos(alpha_rad), math.sin(alpha_rad)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)


def apply_channel_rotation(field: torch.Tensor, alpha_rad: float) -> torch.Tensor:
    """Apply ``Rz(alpha)`` channelwise to a ``[B, 3, H, W]`` XYZ field."""
    rot = rz_matrix(alpha_rad).to(field.dtype)
    return torch.einsum("ij,bjhw->bihw", rot, field)


def physical_yaw_expected(field: torch.Tensor, dj_pixels: int, alpha_rad: float) -> torch.Tensor:
    """The gauge's assumed physical yaw ``T_k = Rz(alpha) o Roll_{dj}`` on the encoder input.

    ``Roll_{dj}`` moves column contents to LARGER u (``torch.roll(..., shifts=+dj, dims=-1)``),
    matching the gauge's ``Roll_k`` and FLAC ``rotate_scene_metadata`` (``torch.roll(depth,
    shifts=dj, dims=2)``)."""
    rolled = torch.roll(field, shifts=dj_pixels, dims=-1)
    return apply_channel_rotation(rolled, alpha_rad)


def roll_token_columns(patch_grid: torch.Tensor, cols: int) -> torch.Tensor:
    """Roll a ``[B, H_t, W_t, D]`` token grid by ``cols`` along the azimuth (W_t) axis."""
    return torch.roll(patch_grid, shifts=cols, dims=2)


def rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative L2 error ``||actual - expected|| / max(||expected||, tiny)``."""
    denom = expected.norm().item()
    return float((actual - expected).norm().item() / (denom if denom > 1e-30 else 1e-30))


def patch_grid_from_tokens(last_hidden_state: torch.Tensor, height_px: int, width_px: int) -> torch.Tensor:
    """Reshape patch-only ``[B, N, D]`` tokens (row-major H_t outer, W_t inner -- see
    ``modeling_cylindrical_dinov3``: ``phi.repeat(num_patches_h)``) into ``[B, H_t, W_t, D]``."""
    b, n, d = last_hidden_state.shape
    h_t, w_t = height_px // PATCH_SIZE, width_px // PATCH_SIZE
    if h_t * w_t != n:
        raise InvalidInfrastructure(
            f"token count {n} != H_t*W_t = {h_t}*{w_t} for a {height_px}x{width_px} input"
        )
    return last_hidden_state.reshape(b, h_t, w_t, d)


# ============================================================================================
# A2a / A2b / A2c metric computers (pure; unit-tested with tiny fixtures).
# ============================================================================================
def a2a_residual(c_base: torch.Tensor, c_rotated: torch.Tensor, dj_pixels: int, alpha_rad: float) -> float:
    """A2a: rel-err between the REAL rotated-pipeline field and ``Rz(alpha) . Roll_{dj}`` of
    the base field. Small (<= 1e-6) iff the data transform is the assumed physical yaw."""
    expected = physical_yaw_expected(c_base, dj_pixels, alpha_rad)
    return rel_err(c_rotated, expected)


def _pooled_patch_residuals(
    out_base, out_rot, token_roll: int, height_px: int, width_px: int
) -> tp.Tuple[float, float]:
    """Pooled-invariance rel-err and patch roll-equivariance rel-err (undo the token roll)."""
    pooled = rel_err(out_rot.pooler_output, out_base.pooler_output)
    base_grid = patch_grid_from_tokens(out_base.last_hidden_state, height_px, width_px)
    rot_grid = patch_grid_from_tokens(out_rot.last_hidden_state, height_px, width_px)
    # Undo the expected token-column roll of +token_roll on the rotated field, compare to base.
    undone = roll_token_columns(rot_grid, -token_roll)
    patch = rel_err(undone, base_grid)
    return pooled, patch


def a2b_metrics(
    model_on, c_base: torch.Tensor, c_rot: torch.Tensor, token_roll: int, height_px: int, width_px: int
) -> tp.Tuple[float, float]:
    """A2b: gauge-ON pooled invariance and patch roll-equivariance rel-errs."""
    with torch.no_grad():
        out_base = model_on(c_base)
        out_rot = model_on(c_rot)
    return _pooled_patch_residuals(out_base, out_rot, token_roll, height_px, width_px)


def a2c_controls(
    model_on,
    model_off,
    c_base: torch.Tensor,
    c_rot: torch.Tensor,
    token_roll: int,
    height_px: int,
    width_px: int,
) -> tp.Dict[str, float]:
    """A2c negative controls (each must be >= CONTROL_MIN):
    (1) gauge-OFF on the same pair; (2) channel-permuted (y,z,x) input with gauge-ON."""
    with torch.no_grad():
        off_base = model_off(c_base)
        off_rot = model_off(c_rot)
        perm = [CHANNELS["y"], CHANNELS["z"], CHANNELS["x"]]  # (y, z, x)
        perm_base = model_on(c_base[:, perm])
        perm_rot = model_on(c_rot[:, perm])
    off_pooled, off_patch = _pooled_patch_residuals(off_base, off_rot, token_roll, height_px, width_px)
    perm_pooled, perm_patch = _pooled_patch_residuals(perm_base, perm_rot, token_roll, height_px, width_px)
    return {
        "gauge_off_pooled": off_pooled,
        "gauge_off_patch": off_patch,
        "channel_perm_pooled": perm_pooled,
        "channel_perm_patch": perm_patch,
    }


def controls_pass(control_values: tp.Mapping[str, float]) -> bool:
    """A negative-control cell passes iff EVERY residual is at least CONTROL_MIN."""
    return all(v >= CONTROL_MIN for v in control_values.values())


# ============================================================================================
# A2d status classification (plan sec 1; mutation m4 target). TOTAL over all gates.
# ============================================================================================
def classify_audit_status(
    a2a_pass: bool, controls_all_pass: bool, other_validity_pass: bool, a2b_pass: bool
) -> str:
    """Three-way status. Validity gates (A2a residual, negative controls, and everything in
    ``other_validity_pass`` -- fingerprint, non-degeneracy, finiteness, capture, provenance,
    serialisation) dominate: ANY validity failure => invalid_infrastructure. Only when validity
    holds does A2b decide gauge-ON (valid_pass) vs gauge-OFF (valid_convention_failure)."""
    validity_ok = bool(a2a_pass) and bool(controls_all_pass) and bool(other_validity_pass)
    if not validity_ok:
        return STATUS_INVALID
    return STATUS_VALID_PASS if a2b_pass else STATUS_CONVENTION_FAILURE


def exit_code_for(status: str) -> int:
    return EXIT_CODES[status]


# ============================================================================================
# Finiteness + atomic serialisation (mutation m5 target).
# ============================================================================================
def collect_non_finite(obj: tp.Any, path: str = "") -> tp.List[str]:
    """Recursively list json-paths of any non-finite float (NaN/Inf) in a record."""
    bad: tp.List[str] = []
    if isinstance(obj, float):
        if not math.isfinite(obj):
            bad.append(path or "<root>")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad.extend(collect_non_finite(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad.extend(collect_non_finite(v, f"{path}[{i}]"))
    return bad


def sanitize_non_finite(obj: tp.Any) -> tp.Any:
    """Replace every non-finite float with ``None`` so the record is valid finite JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_non_finite(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_non_finite(v) for v in obj]
    return obj


def atomic_write_json(path: str, record: tp.Mapping[str, tp.Any]) -> None:
    """Serialise ``record`` to ``path`` atomically: write a temp file in the same directory,
    ``fsync``, then ``os.replace`` (atomic on POSIX). ``allow_nan=False`` -> any non-finite
    number raises BEFORE the destination is touched, and the temp file is removed. On failure
    the destination is left untouched (no partial write)."""
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".audit_convention.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(record, fh, allow_nan=False, indent=2, sort_keys=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)  # atomic rename
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# ============================================================================================
# Provenance.
# ============================================================================================
def _git_sha(repo: str) -> tp.Dict[str, tp.Any]:
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", repo, "status", "--porcelain"], stderr=subprocess.DEVNULL
            ).decode().strip()
        )
        return {"sha": sha, "dirty": dirty}
    except Exception as exc:  # noqa: BLE001
        return {"sha": None, "dirty": None, "error": repr(exc)}


def _sha256_file(path: str) -> tp.Optional[str]:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def build_provenance(checkpoint: str, cyl_repo: str, worktree: str) -> tp.Dict[str, tp.Any]:
    import cylindrical_dinov3  # local import so tests can import this module cheaply

    weights_file = os.path.join(checkpoint, "model.safetensors")
    return {
        "cylindrical_dinov3": {
            "version": getattr(cylindrical_dinov3, "__version__", None),
            "package_file": getattr(cylindrical_dinov3, "__file__", None),
            "git": _git_sha(cyl_repo),
        },
        "official_weights": {
            "checkpoint": checkpoint,
            "revision": os.path.basename(checkpoint.rstrip("/")),
            "weights_file": weights_file,
            "sha256": _sha256_file(weights_file),
        },
        "flac_worktree": {"path": worktree, "git": _git_sha(worktree)},
    }


# ============================================================================================
# Real-sample acquisition -- via FLAC's actual AR_md metadata module (read-only data).
# ============================================================================================
def _load_ar_md(worktree: str):
    """Load the REAL AR_md custom-metadata module the way ``dataset.py`` does
    (``importlib.util.spec_from_file_location``)."""
    ar_md_path = os.path.join(worktree, AR_MD_REL)
    spec = importlib.util.spec_from_file_location("exp09_ar_md", ar_md_path)
    if spec is None or spec.loader is None:
        raise InvalidInfrastructure(f"cannot load AR_md module at {ar_md_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_data_root(
    candidates: tp.Sequence[str], scene: str, sub: str, wav: str
) -> str:
    """Pick the first candidate AcousticRooms root where the sample's metadata json AND depth
    npy exist and are readable. Raises InvalidInfrastructure if none is reachable."""
    parts = wav.split("_")
    src, rec = int(parts[0][1:]), int(parts[1][1:])
    tried = []
    for root in candidates:
        meta = os.path.join(root, "metadata", scene, sub, f"S00{src}_R00{rec}.json")
        depth = os.path.join(root, "depth_map", scene, sub, f"{rec}.npy")
        tried.append({"root": root, "metadata": meta, "meta_ok": os.access(meta, os.R_OK),
                      "depth": depth, "depth_ok": os.access(depth, os.R_OK)})
        if os.access(meta, os.R_OK) and os.access(depth, os.R_OK):
            return root
    raise InvalidInfrastructure(f"AcousticRooms data unreachable; tried: {tried}")


def load_real_sample(
    worktree: str, data_root: str, scene: str, sub: str, wav: str
) -> tp.Dict[str, tp.Any]:
    """Build ONE real sample's metadata through the REAL ``AR_md.get_custom_metadata``.

    Only ``depth`` + ``poses`` are loaded (``acoustic_context`` is disabled: it is irrelevant to
    the geometry convention, reads audio wavs, and uses ``np.random.choice`` -- non-deterministic).
    Returns the metadata dict plus the raw arrays and fingerprint inputs."""
    ar_md = _load_ar_md(worktree)
    rel = os.path.join("single_channel_ir_1", scene, sub, wav)
    full = os.path.join(data_root, rel)
    info = {
        "path": full,
        "relpath": rel,
        "modalities": {
            "acoustic_context": {"load": False},
            "depth": {"load": True},
            "poses": {"load": True},
        },
    }
    try:
        md = ar_md.get_custom_metadata(info, None)  # AR_md never reads `audio`.
    except Exception as exc:  # noqa: BLE001
        raise InvalidInfrastructure(f"AR_md.get_custom_metadata failed: {exc!r}") from exc
    for key in ("source_vit", "depth"):
        if key not in md:
            raise InvalidInfrastructure(f"real sample missing '{key}' from AR_md metadata")
    return {"md": md, "rel": rel, "full": full, "data_root": data_root,
            "scene": scene, "sub": sub, "wav": wav}


def sample_fingerprint(md: tp.Mapping[str, tp.Any], ident: tp.Mapping[str, str]) -> tp.Dict[str, tp.Any]:
    """sha256 of the raw depth + source arrays plus scene identifiers. Also non-degeneracy /
    finiteness diagnostics (per-channel std > 0; depth non-constant; finite; source radius > 0)."""
    depth = md["depth"]
    source = md["source_vit"]
    depth_np = np.ascontiguousarray(depth.detach().cpu().numpy())
    source_np = np.ascontiguousarray(source.detach().cpu().numpy())
    hasher = hashlib.sha256()
    hasher.update(json.dumps(ident, sort_keys=True).encode())
    hasher.update(str(depth_np.dtype).encode()); hasher.update(str(depth_np.shape).encode())
    hasher.update(depth_np.tobytes())
    hasher.update(str(source_np.dtype).encode()); hasher.update(str(source_np.shape).encode())
    hasher.update(source_np.tobytes())

    per_channel_std = [float(depth[c].std().item()) for c in range(depth.shape[0])]
    depth_finite = bool(torch.isfinite(depth).all().item())
    source_finite = bool(torch.isfinite(source).all().item())
    sx, sy = float(source[..., 0].reshape(-1)[0]), float(source[..., 1].reshape(-1)[0])
    source_radius = math.hypot(sx, sy)
    depth_std = float(depth.std().item())
    nondegenerate = (
        all(s > 0.0 for s in per_channel_std)
        and depth_std > 0.0
        and depth_finite and source_finite
        and source_radius > 1e-6
    )
    return {
        "sha256": hasher.hexdigest(),
        "depth_shape": list(depth_np.shape),
        "depth_dtype": str(depth_np.dtype),
        "source_vit": source_np.reshape(-1).tolist(),
        "per_channel_std": per_channel_std,
        "depth_std": depth_std,
        "depth_finite": depth_finite,
        "source_finite": source_finite,
        "source_radius": source_radius,
        "nondegenerate": nondegenerate,
    }


# ============================================================================================
# Encoder-input capture -- via the REAL GeometryConditioner.forward path.
# ============================================================================================
def build_geometry_conditioner(height_px: int, width_px: int, max_value: float):
    """Build a REAL ``GeometryConditioner`` through FLAC's actual factory
    (``create_multi_conditioner_from_conditioning_config``). The ViT injected into it is a tiny
    local ``SimpleViT`` and is NEVER run -- the plan states the conditioner's ViT "is NOT the
    point here"; A2 captures the encoder input BEFORE the ViT and feeds it to the cylindrical
    model separately. Only ``max_value`` and the real forward's ``(coord - depth)/max_value``
    computation matter here, and both are the real ones."""
    from src.models.conditioners import create_multi_conditioner_from_conditioning_config

    config = {
        "cond_dim": 256,
        "configs": [
            {
                "id": "source_vit",
                "type": "ViTCoordinates",
                "config": {
                    "max_value": max_value,
                    "ViT": {
                        "ch_dim": 3,
                        "img_h": height_px,
                        "img_w": width_px,
                        "patch_h": PATCH_SIZE,
                        "patch_w": PATCH_SIZE,
                        "dim": 32,
                        "depth": 1,
                        "heads": 2,
                    },
                },
            }
        ],
    }
    return create_multi_conditioner_from_conditioning_config(config)


def capture_encoder_input(multi_conditioner, md: tp.Mapping[str, tp.Any]) -> torch.Tensor:
    """Run ``md`` through the REAL ``MultiConditioner.forward`` / ``GeometryConditioner.forward``
    and capture the exact encoder input ``c = (coord - depth)/max_value`` fed to ``self.vit``,
    via a forward-pre-hook that aborts the ViT. Returns a ``[1, 3, H, W]`` tensor; raises
    InvalidInfrastructure on any capture/hook failure (empty or wrong-shape field)."""
    geom = multi_conditioner.conditioners["source_vit"]
    captured: tp.Dict[str, torch.Tensor] = {}

    def _pre_hook(_module, args):
        captured["c"] = args[0].detach().clone()
        raise _CaptureAbort()

    handle = geom.vit.register_forward_pre_hook(_pre_hook)
    try:
        try:
            multi_conditioner([dict(md)], "cpu")
        except _CaptureAbort:
            pass
    finally:
        handle.remove()

    if "c" not in captured:
        raise InvalidInfrastructure("encoder-input capture failed: hook never fired")
    field = captured["c"]
    if field.ndim != 4 or field.shape[0] != 1 or field.shape[1] != 3:
        raise InvalidInfrastructure(f"captured field has wrong shape {tuple(field.shape)}, expected [1,3,H,W]")
    if not bool(torch.isfinite(field).all().item()):
        raise InvalidInfrastructure("captured encoder input contains non-finite values")
    return field.float()


# ============================================================================================
# Cylindrical model loading.
# ============================================================================================
def load_cylindrical_model(checkpoint: str, gauge: str):
    """Load the official-weights cylindrical model on CPU, FP32, eager attention.
    ``gauge='cylindrical_xyz'`` -> gauge ON; ``gauge='none'`` -> gauge OFF."""
    from cylindrical_dinov3 import CylindricalDINOv3ViTModel

    model = CylindricalDINOv3ViTModel.from_pretrained(
        checkpoint, gauge=gauge, attn_implementation="eager", dtype=torch.float32
    )
    return model.to(torch.float32).eval()


# ============================================================================================
# A1 static block -- convention facts + quoted live source lines.
# ============================================================================================
def _quote_lines(path: str, start: int, end: int) -> tp.Dict[str, tp.Any]:
    try:
        with open(path, "r") as fh:
            lines = fh.read().splitlines()
        return {"path": path, "start": start, "end": end,
                "text": "\n".join(lines[start - 1:end])}
    except OSError as exc:
        return {"path": path, "start": start, "end": end, "error": repr(exc)}


def _quote_assignments(path: str, names: tp.Sequence[str]) -> tp.Dict[str, tp.Any]:
    """Quote the lines that assign ``names`` (robust to line drift in an external package)."""
    try:
        with open(path, "r") as fh:
            lines = fh.read().splitlines()
        picked = [(i + 1, ln) for i, ln in enumerate(lines)
                  if any(ln.strip().startswith(f"{n} =") for n in names)]
        return {"path": path, "line_numbers": [i for i, _ in picked],
                "text": "\n".join(ln for _, ln in picked)}
    except OSError as exc:
        return {"path": path, "error": repr(exc)}


def build_a1_static(worktree: str) -> tp.Dict[str, tp.Any]:
    """Re-derive and serialise the convention facts, quoting the live source lines (plan sec 1)."""
    import cylindrical_dinov3.gauge as gauge_mod

    ar_md_path = os.path.join(worktree, AR_MD_REL)
    yaw_path = os.path.join(worktree, YAW_ROTATION_REL)
    gauge_path = getattr(gauge_mod, "__file__", "")
    return {
        "convention": {
            "encoder_input": "(source_pos - depth_xyz) / max_value, [B,3,256,512]",
            "channels_xyz": CHANNELS,
            "azimuth": "theta_u = 2*pi*(u+0.5)/W - pi; increasing column = increasing azimuth",
            "elevation": "phi_v = (v+0.5)*pi/H - pi/2",
            "point_cloud": "(x,y,z) = (d*cosphi*costheta, d*cosphi*sintheta, -d*sinphi)",
            "yaw": "Rz(+z) CCW; column roll dj = round(alpha*W/2pi); alpha_eff = dj*2pi/W",
            "gauge_channels": {
                "X_CHANNEL": getattr(gauge_mod, "X_CHANNEL", None),
                "Y_CHANNEL": getattr(gauge_mod, "Y_CHANNEL", None),
                "Z_CHANNEL": getattr(gauge_mod, "Z_CHANNEL", None),
            },
        },
        "source_quotes": {
            "AR_md.convert_equirect_to_camera_coord": _quote_lines(ar_md_path, 58, 66),
            "yaw_rotation.azimuth_rotation_matrix": _quote_lines(yaw_path, 140, 157),
            "yaw_rotation.rotate_scene_metadata.roll_and_rotate": _quote_lines(yaw_path, 198, 216),
            "gauge.channel_contract": (
                _quote_assignments(gauge_path, ["X_CHANNEL", "Y_CHANNEL", "Z_CHANNEL"])
                if gauge_path else None
            ),
        },
    }


# ============================================================================================
# Orchestration.
# ============================================================================================
def angle_spec(width_px: int) -> tp.List[tp.Dict[str, tp.Any]]:
    """Per-angle (alpha_j, dj pixels, token roll) for j in J_VALUES. alpha_j = 2*pi*(16j)/W."""
    spec = []
    for j in J_VALUES:
        dj = ROLL_MULTIPLE * j
        spec.append({
            "j": j,
            "dj_pixels": dj,
            "token_roll": dj // PATCH_SIZE,
            "alpha_rad": dj * 2.0 * math.pi / width_px,
            "degrees": dj * 360.0 / width_px,
        })
    return spec


def run_audit(
    *,
    out_path: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    cyl_repo: str = DEFAULT_CYL_REPO,
    worktree: tp.Optional[str] = None,
    data_roots: tp.Sequence[str] = DEFAULT_DATA_ROOTS,
    sample: tp.Mapping[str, str] = DEFAULT_SAMPLE,
    expect_fingerprint: tp.Optional[str] = None,
) -> tp.Tuple[tp.Dict[str, tp.Any], int]:
    """Execute the full audit on the real sample and write the atomic finite-JSON record.
    Returns ``(record, exit_code)``. Any validity failure yields ``invalid_infrastructure``."""
    worktree = worktree or str(_WORKTREE_ROOT)
    reasons: tp.List[str] = []
    record: tp.Dict[str, tp.Any] = {
        "schema_version": "exp09-stageA-1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parameters": {
            "patch_size": PATCH_SIZE, "roll_multiple": ROLL_MULTIPLE, "j_values": list(J_VALUES),
            "a2a_rtol": A2A_RTOL, "a2b_rtol": A2B_RTOL, "control_min": CONTROL_MIN,
            "geom_max_value": GEOM_MAX_VALUE, "channels": CHANNELS,
        },
        "a1_static": build_a1_static(worktree),
    }

    # These validity sub-flags default False and are only set True on success, so an exception
    # anywhere below correctly yields invalid_infrastructure.
    fingerprint_ok = False
    nondegenerate = False
    capture_ok = False
    provenance_ok = False
    a2a_pass = False
    controls_all_pass = False
    a2b_pass = False

    try:
        # -- provenance (before compute, so a load failure still records what we know) --------
        record["provenance"] = build_provenance(checkpoint, cyl_repo, worktree)
        prov = record["provenance"]
        provenance_ok = (
            prov["cylindrical_dinov3"]["version"] is not None
            and prov["cylindrical_dinov3"]["git"]["sha"] is not None
            and prov["official_weights"]["sha256"] is not None
            and prov["flac_worktree"]["git"]["sha"] is not None
        )
        if not provenance_ok:
            reasons.append("provenance_incomplete")

        # -- real sample -----------------------------------------------------------------------
        data_root = resolve_data_root(data_roots, sample["scene"], sample["sub"], sample["wav"])
        loaded = load_real_sample(worktree, data_root, sample["scene"], sample["sub"], sample["wav"])
        md = loaded["md"]
        ident = {"scene": sample["scene"], "sub": sample["sub"], "wav": sample["wav"],
                 "data_root": data_root}
        fp = sample_fingerprint(md, ident)
        record["sample"] = {**ident, "relpath": loaded["rel"], **fp}
        nondegenerate = bool(fp["nondegenerate"])
        if not nondegenerate:
            reasons.append("sample_degenerate_or_nonfinite")
        if expect_fingerprint is not None:
            fingerprint_ok = (fp["sha256"] == expect_fingerprint)
            if not fingerprint_ok:
                reasons.append("fingerprint_mismatch")
        else:
            fingerprint_ok = True  # no pin provided -> not a mismatch

        height_px, width_px = int(md["depth"].shape[-2]), int(md["depth"].shape[-1])
        record["sample"]["height_px"] = height_px
        record["sample"]["width_px"] = width_px

        # -- capture base + rotated encoder inputs via the REAL conditioner --------------------
        from src.data.yaw_rotation import rotate_scene_metadata

        mc = build_geometry_conditioner(height_px, width_px, GEOM_MAX_VALUE)
        c_base = capture_encoder_input(mc, md)
        capture_ok = True
        record["sample"]["captured_field_shape"] = list(c_base.shape)

        spec = angle_spec(width_px)
        record["a2_params"] = {"angles": spec}

        # -- models (official weights, CPU FP32 eager) -----------------------------------------
        model_on = load_cylindrical_model(checkpoint, "cylindrical_xyz")
        model_off = load_cylindrical_model(checkpoint, "none")

        a2a_results, a2b_results, a2c_results = [], [], []
        a2a_ok, a2b_ok, ctrl_ok = True, True, True
        for ang in spec:
            dj, alpha, troll = ang["dj_pixels"], ang["alpha_rad"], ang["token_roll"]
            md_rot = rotate_scene_metadata(md, alpha, width_px)
            c_rot = capture_encoder_input(mc, md_rot)

            res_a2a = a2a_residual(c_base, c_rot, dj, alpha)
            p_a2a = res_a2a <= A2A_RTOL
            a2a_ok = a2a_ok and p_a2a
            a2a_results.append({"j": ang["j"], "residual": res_a2a, "pass": bool(p_a2a)})

            pooled, patch = a2b_metrics(model_on, c_base, c_rot, troll, height_px, width_px)
            p_a2b = (pooled <= A2B_RTOL) and (patch <= A2B_RTOL)
            a2b_ok = a2b_ok and p_a2b
            a2b_results.append({"j": ang["j"], "pooled_relerr": pooled, "patch_relerr": patch,
                                "pass": bool(p_a2b)})

            controls = a2c_controls(model_on, model_off, c_base, c_rot, troll, height_px, width_px)
            p_ctrl = controls_pass(controls)
            ctrl_ok = ctrl_ok and p_ctrl
            a2c_results.append({"j": ang["j"], **controls, "pass": bool(p_ctrl)})

        record["a2a"] = {"per_angle": a2a_results, "all_pass": bool(a2a_ok)}
        record["a2b"] = {"per_angle": a2b_results, "all_pass": bool(a2b_ok)}
        record["a2c"] = {"per_angle": a2c_results, "all_pass": bool(ctrl_ok)}
        a2a_pass, a2b_pass, controls_all_pass = a2a_ok, a2b_ok, ctrl_ok
        if not a2a_ok:
            reasons.append("a2a_residual_failure")
        if not ctrl_ok:
            reasons.append("negative_control_failure")
        if not a2b_ok:
            reasons.append("a2b_equivariance_failure")

    except InvalidInfrastructure as exc:
        record["error"] = {"type": "InvalidInfrastructure", "message": str(exc)}
        reasons.append(f"invalid_infrastructure_exception:{exc}")
    except Exception as exc:  # noqa: BLE001  -- any unexpected failure is infrastructure-invalid
        record["error"] = {"type": type(exc).__name__, "message": repr(exc)}
        reasons.append(f"unexpected_exception:{type(exc).__name__}")

    # -- finiteness gate (any non-finite metric anywhere => invalid) ---------------------------
    non_finite = collect_non_finite({k: v for k, v in record.items() if k != "a1_static"})
    all_finite = (len(non_finite) == 0)
    if not all_finite:
        reasons.append("non_finite_values")
        record["non_finite_fields"] = non_finite

    other_validity_pass = (
        fingerprint_ok and nondegenerate and capture_ok and provenance_ok and all_finite
    )

    status = classify_audit_status(a2a_pass, controls_all_pass, other_validity_pass, a2b_pass)
    code = exit_code_for(status)

    record["validity"] = {
        "a2a_pass": bool(a2a_pass),
        "controls_all_pass": bool(controls_all_pass),
        "a2b_pass": bool(a2b_pass),
        "fingerprint_ok": bool(fingerprint_ok),
        "nondegenerate": bool(nondegenerate),
        "capture_ok": bool(capture_ok),
        "provenance_ok": bool(provenance_ok),
        "all_finite": bool(all_finite),
        "other_validity_pass": bool(other_validity_pass),
    }
    record["reasons"] = reasons
    record["audit_status"] = status
    record["exit_code"] = code

    # Sanitize (defensive: no NaN/Inf ever written) then atomic write.
    safe_record = sanitize_non_finite(record)
    try:
        atomic_write_json(out_path, safe_record)
    except Exception as exc:  # noqa: BLE001  -- serialisation failure is invalid_infrastructure
        # Re-classify to invalid and retry a minimal record so the run leaves an artifact.
        safe_record["audit_status"] = STATUS_INVALID
        safe_record["exit_code"] = EXIT_CODES[STATUS_INVALID]
        safe_record.setdefault("reasons", []).append(f"serialization_failure:{exc!r}")
        atomic_write_json(out_path, {"audit_status": STATUS_INVALID,
                                     "exit_code": EXIT_CODES[STATUS_INVALID],
                                     "reasons": safe_record["reasons"]})
        return safe_record, EXIT_CODES[STATUS_INVALID]

    return safe_record, code


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="exp-09 Stage A convention audit runner")
    default_out = str(Path(__file__).resolve().parent / "audit_convention.json")
    p.add_argument("--out", default=default_out, help="output JSON path")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--cyl-repo", default=DEFAULT_CYL_REPO)
    p.add_argument("--worktree", default=str(_WORKTREE_ROOT))
    p.add_argument("--data-root", action="append", default=None,
                   help="AcousticRooms root(s); repeatable. Defaults to DEFAULT_DATA_ROOTS.")
    p.add_argument("--scene", default=DEFAULT_SAMPLE["scene"])
    p.add_argument("--sub", default=DEFAULT_SAMPLE["sub"])
    p.add_argument("--wav", default=DEFAULT_SAMPLE["wav"])
    p.add_argument("--expect-fingerprint", default=None,
                   help="pin the sample sha256; a mismatch => invalid_infrastructure")
    return p


def main(argv: tp.Optional[tp.Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(0)
    record, code = run_audit(
        out_path=args.out,
        checkpoint=args.checkpoint,
        cyl_repo=args.cyl_repo,
        worktree=args.worktree,
        data_roots=tuple(args.data_root) if args.data_root else DEFAULT_DATA_ROOTS,
        sample={"scene": args.scene, "sub": args.sub, "wav": args.wav},
        expect_fingerprint=args.expect_fingerprint,
    )
    print(f"audit_status={record['audit_status']} exit_code={code} out={args.out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
