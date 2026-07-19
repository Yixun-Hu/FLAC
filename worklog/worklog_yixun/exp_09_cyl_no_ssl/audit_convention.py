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

* **A1** (static) -- re-derive and serialise the convention facts, QUOTING **and ASSERTING** the
  live source lines for every claimed basis (``AR_md`` theta/phi/xyz; ``yaw_rotation`` Rz matrix
  + depth roll; the ``conditioners`` encoder expression ``(coord[:, i, :, None, None] -
  depth_coord) / self.max_value``; the gauge channel contract). Any unreadable file or absent
  literal (source drift) enters validity => ``invalid_infrastructure``.
* **A2a** (pre-model residual) -- the encoder input captured from the REAL
  ``GeometryConditioner.forward`` on the REAL-``rotate_scene_metadata``-rotated sample equals
  ``Rz(alpha_j) . Roll_{16j}`` of the base captured field, rel-err <= 1e-6. Proves the DATA
  transform is the gauge's assumed physical yaw ``T_k`` before any model runs.
* **A2b** (model equivariance) -- the gauge-ON official-weights model is pooled-invariant AND
  patch roll-equivariant (undo the expected token-column roll of ``16j/16 = j`` columns),
  each rel-err <= 1e-4.
* **A2c** (negative controls) -- gauge-OFF on the same pair, and channel-permuted (y,z,x) input
  with gauge-ON: patch AND pooled residuals each >= 1e-2 (proves the A2b test is sensitive).
* **A2d** (three-way status) -- TOTAL over every gate, at the production boundary (the ENTIRE
  body is exception-wrapped => never an escaping traceback / exit 1). ANY validity failure
  (A2a residual, either control, absent-or-mismatched fingerprint, non-degeneracy, non-finite
  value, capture/hook failure, wrong geometry, A1 source drift, provenance IDENTITY failure --
  any absent OR mismatched registered expectation -- serialisation failure) =>
  ``invalid_infrastructure`` (exit 3). Validity established but A2b failing =>
  ``valid_convention_failure`` (exit 2, selects gauge-OFF). All pass => ``valid_pass`` (exit 0,
  selects gauge-ON). A blessed run is fail-closed: it must register the full expectation set
  (``--expect-*`` flags + ``--expect-fingerprint`` + ``--expect-clean-worktree``).

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

# Registered panorama geometry. The blessed run is PINNED to exactly this H x W; the captured
# encoder field and the per-angle spec are validated against it (never rescaled). A dev run may
# override via ``--geometry H W`` (serialised), but that is not the blessed geometry.
REGISTERED_HEIGHT = 256
REGISTERED_WIDTH = 512

# Known blessed values (for reference / the Planner's records; the runner does NOT default the
# expectation flags to these -- a blessed run must register them explicitly so that ABSENCE of
# any expectation fails closed).
KNOWN_WEIGHTS_SHA256 = "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
KNOWN_BLESSED_FINGERPRINT = "81038cc90f3f295277016e2a8981867ed752b9a081183a8a9541204a892cad5b"

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
CONDITIONERS_REL = "src/models/conditioners.py"
# The exp-09 output directory (relative to the worktree root). Changes under it are treated as
# clean by the ``--expect-clean-worktree`` gate (the audit writes its JSON there).
OUTPUT_RELDIR = "worklog/worklog_yixun/exp_09_cyl_no_ssl"


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


def _worktree_clean_excluding(worktree: str, exclude_reldir: str) -> tp.Optional[bool]:
    """True iff ``git status --porcelain`` for ``worktree`` is empty AFTER dropping every entry
    under ``exclude_reldir`` (the exp-09 output dir, where this audit writes its JSON). Returns
    None if git is unavailable (which the provenance gate then treats as a failed expectation)."""
    try:
        out = subprocess.check_output(
            ["git", "-C", worktree, "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
    except Exception:  # noqa: BLE001
        return None
    exclude = exclude_reldir.strip("/")
    prefix = exclude + "/"
    for line in out.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        if " -> " in path:  # rename entry "old -> new"
            path = path.split(" -> ", 1)[1]
        if path == exclude or path.startswith(prefix):
            continue
        return False
    return True


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
        "flac_worktree": {
            "path": worktree,
            "git": _git_sha(worktree),
            "clean_excluding_output": _worktree_clean_excluding(worktree, OUTPUT_RELDIR),
        },
    }


def evaluate_provenance(
    prov: tp.Mapping[str, tp.Any], expectations: tp.Mapping[str, tp.Any]
) -> tp.Tuple[bool, tp.List[str], tp.List[tp.Dict[str, tp.Any]]]:
    """Gate the built provenance against REGISTERED expectations by IDENTITY, not mere presence
    (finding 1). Fail closed: an expectation that is ABSENT (not registered) is a failure just as
    a MISMATCH is -- a run without the full expectation set is not blessed. Returns
    ``(ok, reasons, checks)`` where ``checks`` is a JSON-serialisable per-field audit trail."""
    prov = prov or {}
    cyl = prov.get("cylindrical_dinov3", {}) or {}
    wt = prov.get("flac_worktree", {}) or {}
    ow = prov.get("official_weights", {}) or {}

    def _eq_check(name: str, expected: tp.Any, actual: tp.Any) -> tp.Dict[str, tp.Any]:
        if expected is None:
            state = "absent"
        else:
            # IDENTITY comparison (mutation M1 target: replacing this with a presence-only
            # ``actual is not None`` re-introduces the finding-1 defect).
            state = "match" if (actual is not None and expected == actual) else "mismatch"
        return {"name": name, "ok": state == "match", "state": state,
                "expected": expected, "actual": actual}

    def _prefix_check(name: str, expected: tp.Any, actual: tp.Any) -> tp.Dict[str, tp.Any]:
        if expected is None:
            state = "absent"
        else:
            state = "match" if (isinstance(actual, str) and actual.startswith(expected)) else "mismatch"
        return {"name": name, "ok": state == "match", "state": state,
                "expected": expected, "actual": actual}

    def _flag_check(name: str, expected_flag: tp.Any, actual: tp.Any) -> tp.Dict[str, tp.Any]:
        if not expected_flag:
            state = "absent"
        else:
            state = "match" if (actual is True) else "mismatch"
        return {"name": name, "ok": state == "match", "state": state,
                "expected": True, "actual": actual}

    checks = [
        _eq_check("package_sha", expectations.get("package_sha"), (cyl.get("git") or {}).get("sha")),
        _prefix_check("package_path_prefix", expectations.get("package_path_prefix"), cyl.get("package_file")),
        _eq_check("worktree_sha", expectations.get("worktree_sha"), (wt.get("git") or {}).get("sha")),
        _flag_check("clean_worktree", expectations.get("expect_clean_worktree"), wt.get("clean_excluding_output")),
        _eq_check("checkpoint_revision", expectations.get("checkpoint_revision"), ow.get("revision")),
        _eq_check("weights_sha256", expectations.get("weights_sha256"), ow.get("sha256")),
    ]
    ok = all(c["ok"] for c in checks)
    reasons = [f"provenance_{c['name']}_{c['state']}" for c in checks if not c["ok"]]
    return ok, reasons, checks


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


def capture_encoder_input(
    multi_conditioner,
    md: tp.Mapping[str, tp.Any],
    expected_hw: tp.Tuple[int, int] = (REGISTERED_HEIGHT, REGISTERED_WIDTH),
) -> torch.Tensor:
    """Run ``md`` through the REAL ``MultiConditioner.forward`` / ``GeometryConditioner.forward``
    and capture the exact encoder input ``c = (coord - depth)/max_value`` fed to ``self.vit``,
    via a forward-pre-hook that aborts the ViT. Returns a ``[1, 3, H, W]`` tensor; raises
    InvalidInfrastructure on any capture/hook failure OR if the captured field is not EXACTLY
    ``[1, 3, expected_h, expected_w]`` (finding 3: the blessed geometry is pinned, never adapted)."""
    exp_h, exp_w = int(expected_hw[0]), int(expected_hw[1])
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
        except Exception as exc:  # noqa: BLE001 -- a conditioner failure before capture => invalid
            if "c" not in captured:
                raise InvalidInfrastructure(f"encoder-input capture failed: {exc!r}") from exc
    finally:
        handle.remove()

    if "c" not in captured:
        # Sentinel: the ViT pre-hook never fired -> the conditioner never produced the encoder
        # input. NEVER proceed with a fabricated field (finding-5 t3).
        raise InvalidInfrastructure("encoder-input capture failed: hook never fired")
    field = captured["c"]
    if field.ndim != 4 or tuple(field.shape) != (1, 3, exp_h, exp_w):
        raise InvalidInfrastructure(
            f"captured field has shape {tuple(field.shape)}, expected [1,3,{exp_h},{exp_w}]")
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
def _quote_basis(path: str, patterns: tp.Sequence[str]) -> tp.Dict[str, tp.Any]:
    """READ ``path`` and, for each required literal ``pattern`` (the stated convention basis),
    locate the source line that contains it. Returns the quoted matching lines plus a ``missing``
    list of any pattern not found and an ``error`` string if the file could not be read. A1
    validity then requires ``error is None and missing == []`` for every basis (finding 4:
    quote AND assert -- any read failure or pattern mismatch enters validity => invalid)."""
    result: tp.Dict[str, tp.Any] = {
        "path": path, "patterns": list(patterns), "matches": {}, "missing": [], "error": None,
    }
    try:
        with open(path, "r") as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        result["error"] = repr(exc)
        result["missing"] = list(patterns)
        result["text"] = ""
        return result
    quoted: tp.List[tp.Tuple[int, str]] = []
    for pat in patterns:
        found = None
        for i, ln in enumerate(lines):
            if pat in ln:
                found = (i + 1, ln)
                break
        if found is None:
            result["missing"].append(pat)
            result["matches"][pat] = None
        else:
            result["matches"][pat] = {"line": found[0], "text": found[1].strip()}
            quoted.append(found)
    quoted.sort()
    result["text"] = "\n".join(ln for _, ln in quoted)
    return result


# The A1 source basis: for every convention fact the audit relies on, the ACTUAL source line(s)
# that establish it, keyed by a name, with the literal pattern(s) that MUST be present.
def _a1_basis_specs(worktree: str, gauge_path: str) -> tp.Dict[str, tp.Tuple[str, tp.Tuple[str, ...]]]:
    ar_md_path = os.path.join(worktree, AR_MD_REL)
    yaw_path = os.path.join(worktree, YAW_ROTATION_REL)
    cond_path = os.path.join(worktree, CONDITIONERS_REL)
    return {
        "AR_md.azimuth_theta": (ar_md_path, ("(theta + 0.5) * 2.0 * np.pi / img_w - np.pi",)),
        "AR_md.elevation_phi": (ar_md_path, ("(phi + 0.5) * np.pi / img_h - np.pi / 2",)),
        "AR_md.point_cloud_xyz": (ar_md_path, (
            "depth_map * cos_phi * cos_theta",
            "depth_map * cos_phi * sin_theta",
            "-depth_map * sin_phi",
        )),
        "yaw_rotation.azimuth_rotation_matrix": (
            yaw_path, ("[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]",)),
        "yaw_rotation.roll_depth": (yaw_path, ("torch.roll(depth, shifts=dj, dims=2)",)),
        "conditioners.encoder_input": (
            cond_path, ("(coord[:, i, :, None, None] - depth_coord) / self.max_value",)),
        "gauge.channel_contract": (
            gauge_path, ("X_CHANNEL = 0", "Y_CHANNEL = 1", "Z_CHANNEL = 2")),
    }


def build_a1_static(worktree: str) -> tp.Dict[str, tp.Any]:
    """Re-derive and serialise the convention facts, QUOTING and ASSERTING the live source lines
    for every claimed basis (plan sec 1; finding 4). Sets ``valid=False`` with ``check_failures``
    if any basis file is unreadable or any expected literal is absent (source drift)."""
    import cylindrical_dinov3.gauge as gauge_mod

    gauge_path = getattr(gauge_mod, "__file__", "") or ""
    source_basis: tp.Dict[str, tp.Any] = {}
    for name, (path, patterns) in _a1_basis_specs(worktree, gauge_path).items():
        source_basis[name] = _quote_basis(path, patterns)

    # Validity (mutation M5 target): a read error OR any missing literal fails the basis.
    check_failures = [name for name, b in source_basis.items() if b["error"] or b["missing"]]
    a1_valid = (len(check_failures) == 0)

    return {
        "valid": bool(a1_valid),
        "check_failures": check_failures,
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
        "source_basis": source_basis,
    }


# ============================================================================================
# Orchestration.
# ============================================================================================
def angle_spec(width_px: int, registered_width: int = REGISTERED_WIDTH) -> tp.List[tp.Dict[str, tp.Any]]:
    """Per-angle (alpha_j, dj pixels, token roll) for j in J_VALUES, pinned to the REGISTERED
    width. alpha_j = 2*pi*(16j)/W_registered. If the runtime ``width_px`` differs from
    ``registered_width`` the angles are NOT rescaled -- this raises InvalidInfrastructure
    (finding 3; mutation M4 target: silently rescaling to ``width_px`` re-introduces the defect)."""
    if width_px != registered_width:
        raise InvalidInfrastructure(
            f"runtime width {width_px} != registered width {registered_width}; "
            "angle spec is pinned and never rescaled"
        )
    spec = []
    for j in J_VALUES:
        dj = ROLL_MULTIPLE * j
        spec.append({
            "j": j,
            "dj_pixels": dj,
            "token_roll": dj // PATCH_SIZE,
            "alpha_rad": dj * 2.0 * math.pi / registered_width,
            "degrees": dj * 360.0 / registered_width,
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
    geometry: tp.Tuple[int, int] = (REGISTERED_HEIGHT, REGISTERED_WIDTH),
    expect_fingerprint: tp.Optional[str] = None,
    expectations: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> tp.Tuple[tp.Dict[str, tp.Any], int]:
    """Execute the full audit on the real sample and write the atomic finite-JSON record.
    Returns ``(record, exit_code)``. The ENTIRE production body is wrapped so ANY unexpected
    exception is classified ``invalid_infrastructure`` (exit 3), never an escaping traceback
    (finding 2). Any validity failure yields ``invalid_infrastructure``."""
    worktree = worktree or str(_WORKTREE_ROOT)
    expectations = dict(expectations or {})
    reg_h, reg_w = int(geometry[0]), int(geometry[1])
    reasons: tp.List[str] = []
    # Minimal always-present record so serialisation can proceed even if construction bails early.
    record: tp.Dict[str, tp.Any] = {
        "schema_version": "exp09-stageA-2",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # These validity sub-flags default False and are only set True on explicit success, so an
    # exception (or an absent expectation) anywhere below correctly yields invalid_infrastructure.
    fingerprint_ok = False
    nondegenerate = False
    capture_ok = False
    provenance_ok = False
    a1_valid = False
    a2a_pass = False
    controls_all_pass = False
    a2b_pass = False

    # Fail closed: a blessed run MUST pin the fingerprint. Recorded up front so absence holds even
    # if the sample never loads (finding 3; mutation M3 target lives in the sample block below).
    if expect_fingerprint is None:
        reasons.append("fingerprint_expectation_absent")

    try:
        record["parameters"] = {
            "patch_size": PATCH_SIZE, "roll_multiple": ROLL_MULTIPLE, "j_values": list(J_VALUES),
            "a2a_rtol": A2A_RTOL, "a2b_rtol": A2B_RTOL, "control_min": CONTROL_MIN,
            "geom_max_value": GEOM_MAX_VALUE, "channels": CHANNELS,
            "geometry": {"height": reg_h, "width": reg_w},
        }

        # -- A1 static (INSIDE the production try: any import/read failure => invalid, exit 3) --
        a1 = build_a1_static(worktree)
        record["a1_static"] = a1
        a1_valid = bool(a1.get("valid"))
        if not a1_valid:
            reasons.append("a1_source_basis_invalid")

        # -- provenance IDENTITY gate (fail closed against registered expectations) ------------
        record["provenance"] = build_provenance(checkpoint, cyl_repo, worktree)
        provenance_ok, prov_reasons, prov_checks = evaluate_provenance(record["provenance"], expectations)
        record["provenance"]["expectation_checks"] = prov_checks
        record["provenance"]["provenance_ok"] = bool(provenance_ok)
        reasons.extend(prov_reasons)

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
        # else: fingerprint_ok stays False (absent expectation already recorded above).

        # -- geometry gate: the captured sample MUST be EXACTLY the registered H x W ------------
        height_px, width_px = int(md["depth"].shape[-2]), int(md["depth"].shape[-1])
        record["sample"]["height_px"] = height_px
        record["sample"]["width_px"] = width_px
        if (height_px, width_px) != (reg_h, reg_w):
            raise InvalidInfrastructure(
                f"sample geometry {height_px}x{width_px} != registered {reg_h}x{reg_w}")

        # -- capture base + rotated encoder inputs via the REAL conditioner --------------------
        from src.data.yaw_rotation import rotate_scene_metadata

        mc = build_geometry_conditioner(height_px, width_px, GEOM_MAX_VALUE)
        c_base = capture_encoder_input(mc, md, expected_hw=(reg_h, reg_w))
        capture_ok = True
        record["sample"]["captured_field_shape"] = list(c_base.shape)

        spec = angle_spec(width_px, registered_width=reg_w)
        record["a2_params"] = {"angles": spec}

        # -- models (official weights, CPU FP32 eager) -----------------------------------------
        model_on = load_cylindrical_model(checkpoint, "cylindrical_xyz")
        model_off = load_cylindrical_model(checkpoint, "none")

        a2a_results, a2b_results, a2c_results = [], [], []
        a2a_ok, a2b_ok, ctrl_ok = True, True, True
        for ang in spec:
            dj, alpha, troll = ang["dj_pixels"], ang["alpha_rad"], ang["token_roll"]
            md_rot = rotate_scene_metadata(md, alpha, width_px)
            c_rot = capture_encoder_input(mc, md_rot, expected_hw=(reg_h, reg_w))

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
        fingerprint_ok and nondegenerate and capture_ok and provenance_ok and a1_valid and all_finite
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
        "a1_valid": bool(a1_valid),
        "all_finite": bool(all_finite),
        "other_validity_pass": bool(other_validity_pass),
    }
    record["reasons"] = reasons
    record["audit_status"] = status
    record["exit_code"] = code

    # -- serialise: the RETURNED record is ALWAYS identical to the PERSISTED record (finding 2) --
    safe_record = sanitize_non_finite(record)
    try:
        atomic_write_json(out_path, safe_record)
        return safe_record, code
    except Exception as exc:  # noqa: BLE001  -- serialisation failure is invalid_infrastructure
        # Build ONE minimal, definitely-serialisable record; force invalid; persist AND return
        # that same record. If even this fallback write fails, still exit 3 with the error on
        # stderr -- never an unhandled traceback (finding 2; mutation M2 target = the inner try).
        minimal = {
            "schema_version": record.get("schema_version"),
            "timestamp_utc": record.get("timestamp_utc"),
            "audit_status": STATUS_INVALID,
            "exit_code": EXIT_CODES[STATUS_INVALID],
            "reasons": list(reasons) + [f"serialization_failure:{exc!r}"],
        }
        try:
            atomic_write_json(out_path, minimal)  # persist the SAME record we return
        except Exception as exc2:  # noqa: BLE001 -- guarded: never re-raise, never exit 1
            print(f"exp-09 audit: serialization fully failed ({exc2!r}); returning "
                  f"{STATUS_INVALID} exit {EXIT_CODES[STATUS_INVALID]}", file=sys.stderr)
        return minimal, EXIT_CODES[STATUS_INVALID]


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
    p.add_argument("--geometry", nargs=2, type=int, metavar=("H", "W"),
                   default=[REGISTERED_HEIGHT, REGISTERED_WIDTH],
                   help="registered panorama geometry H W (blessed run uses 256 512); the "
                        "captured field and angle spec are pinned to it, never rescaled")
    # Registered expectations (finding 1/3). A blessed run MUST pass the FULL set: absence of any
    # of these => provenance/fingerprint gate False => invalid_infrastructure (fail closed).
    p.add_argument("--expect-fingerprint", default=None,
                   help="REQUIRED for a blessed run: pin the sample sha256; absence OR mismatch "
                        "=> invalid_infrastructure")
    p.add_argument("--expect-package-sha", default=None,
                   help="expected cylindrical_dinov3 repo git HEAD sha")
    p.add_argument("--expect-package-path-prefix", default=None,
                   help="expected prefix of cylindrical_dinov3.__file__ (the cyl src dir)")
    p.add_argument("--expect-worktree-sha", default=None,
                   help="expected exp-09 FLAC worktree git HEAD sha")
    p.add_argument("--expect-clean-worktree", action="store_true",
                   help="require the worktree porcelain empty (excluding the exp-09 output dir)")
    p.add_argument("--expect-checkpoint-revision", default=None,
                   help="expected official-weights snapshot revision (checkpoint basename)")
    p.add_argument("--expect-weights-sha256", default=None,
                   help="expected sha256 of model.safetensors (known: %s)" % KNOWN_WEIGHTS_SHA256)
    return p


def main(argv: tp.Optional[tp.Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(0)
    expectations = {
        "package_sha": args.expect_package_sha,
        "package_path_prefix": args.expect_package_path_prefix,
        "worktree_sha": args.expect_worktree_sha,
        "expect_clean_worktree": bool(args.expect_clean_worktree),
        "checkpoint_revision": args.expect_checkpoint_revision,
        "weights_sha256": args.expect_weights_sha256,
    }
    record, code = run_audit(
        out_path=args.out,
        checkpoint=args.checkpoint,
        cyl_repo=args.cyl_repo,
        worktree=args.worktree,
        data_roots=tuple(args.data_root) if args.data_root else DEFAULT_DATA_ROOTS,
        sample={"scene": args.scene, "sub": args.sub, "wav": args.wav},
        geometry=(args.geometry[0], args.geometry[1]),
        expect_fingerprint=args.expect_fingerprint,
        expectations=expectations,
    )
    print(f"audit_status={record['audit_status']} exit_code={code} out={args.out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
