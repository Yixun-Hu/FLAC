"""exp_19 R1 — the FA-on-HAA invariance probe, tested without the HAA dataset.

**Why this gate exists.** The frame-averaging machinery (``fa_invariant``) was
built and validated on AcousticRooms, where the depth panorama is rendered at the
LISTENER. HAA reverses that: the panorama is rendered at the SOURCE, the array is
vertically flipped (``HAA_md.py:46``, ``np.flipud``) and the poses are the
receivers expressed in the source-centred frame. None of that has ever been run
through the C4 orbit. If any of it interacts with the rotation — a sign, an axis,
a flip — the HAA-BF arm would train against conditioning that is *not* invariant
while every log looks normal. Plan §3 R1 makes the probe a HARD launch gate, and
a failure stops the arm rather than being "fixed" by flipping a convention.

**Why the core is a pure function.** The gate itself must run on the real stack
(one HAA batch, the real conditioner, threshold 1e-5), which needs the dataset and
a DINOv3 download. That cannot be a unit test. So the probe is split: a pure
``invariance_gap(cond_fn, md, angles)`` that knows only how to walk an orbit and
measure, and a thin CLI that supplies the real ``cond_fn``. These tests exercise
the pure half on synthetic HAA-SHAPED metadata built with HAA's own metadata
helpers, and — crucially — they check the probe can FAIL: a measurement device
that reports "invariant" for a broken rotation is worse than no gate at all.

Written by the exp_19 coder seat (Claude Opus 5, max effort).
"""
import functools
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.yaw_rotation import (
    DEFAULT_FRAME_ANGLES,
    POSE_KEYS,
    azimuth_rotation_matrix,
    rotate_scene_metadata,
    yaw_column_shift,
)


_REPO = Path(__file__).resolve().parents[2]
PROBE_PATH = _REPO / "worklog/worklog_yixun/exp_19_haa_finetune_claude/probe_haa_fa_invariance.py"
HAA_MD_PATH = _REPO / "src/configs/dataset_configs/custom_metadata/HAA_md.py"


def _load(path, name):
    assert Path(path).is_file(), f"not found: {path}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load(PROBE_PATH, "probe_haa_fa_invariance")
haa_md = _load(HAA_MD_PATH, "HAA_md_for_probe_tests")


# --------------------------------------------------------------------------- #
# synthetic metadata with HAA's shapes, built by HAA's own helpers
# --------------------------------------------------------------------------- #
IMG_H, IMG_W, N_CTX, IR_LEN = 256, 512, 8, 9600


def room_depth_map(img_h=IMG_H, img_w=IMG_W,
                   xmax=3.2, xmin=-1.8, ymax=5.1, ymin=-2.4, ceil=1.6, floor=1.2):
    """An equirectangular depth map of a shoebox room seen from an OFF-CENTRE point.

    The asymmetry is load-bearing, and it was found by measurement, not taste. A
    room centred on the camera is C2-symmetric, and a C4 frame average built on a
    C2-symmetric panorama **cannot see a sign-flipped pose rotation** — the
    discrepancy involves rotations by ``2g``, i.e. only 0 and 180 degrees, which
    the symmetry annihilates (measured: gap 1.2e-7, indistinguishable from float
    noise). Off-centre walls give the profile genuine azimuthal structure and the
    same probe reports 1.9e-2.
    """
    phi_i, theta_i = torch.meshgrid(torch.arange(img_h), torch.arange(img_w), indexing="ij")
    theta = (theta_i + 0.5) * 2.0 * math.pi / img_w - math.pi     # HAA_md's column azimuth
    phi = (phi_i + 0.5) * math.pi / img_h - math.pi / 2
    eps = 1e-6
    ct, st = theta.cos(), theta.sin()
    t_x = torch.where(ct >= 0, xmax / torch.clamp(ct, min=eps), xmin / torch.clamp(ct, max=-eps))
    t_y = torch.where(st >= 0, ymax / torch.clamp(st, min=eps), ymin / torch.clamp(st, max=-eps))
    wall = torch.minimum(t_x, t_y)
    vertical = torch.where(phi >= 0, ceil, floor) / torch.clamp(phi.sin().abs(), min=eps)
    return torch.minimum(wall / torch.clamp(phi.cos().abs(), min=eps), vertical).float()


def haa_metadata(seed=19):
    """One HAA-shaped sample: ``depth`` [3,256,512] + the four pose fields.

    Built through ``HAA_md.convert_equirect_to_camera_coord`` and
    ``HAA_md.get_3d_point_camera_coord`` so the panorama is a genuine
    equirectangular point cloud in HAA's convention (each column's stored vector
    carries that column's azimuth, the map is flipped vertically as HAA_md:46
    does) and the poses are receivers in the source-centred frame. A hand-rolled
    tensor of random numbers would rotate "fine" no matter what the convention is,
    and would prove nothing.
    """
    g = torch.Generator().manual_seed(seed)
    depth_map = room_depth_map()
    depth_map = torch.from_numpy(np.flipud(depth_map.numpy()).copy())     # HAA_md:46
    depth = haa_md.convert_equirect_to_camera_coord(depth_map, IMG_H, IMG_W)
    depth = depth.permute(2, 0, 1).float()                                # [3,H,W]

    speaker_xyz = [1.7, -0.4, 1.5]
    listener_xyz = [4.2, 2.9, 1.2]
    src = torch.tensor(
        np.asarray(haa_md.get_3d_point_camera_coord(speaker_xyz, listener_xyz), dtype=np.float32)
    )
    ctx_xyz = [[3.1, -2.2, 1.1], [0.4, 3.7, 1.3], [-2.6, 1.4, 1.0], [5.0, 0.2, 1.4],
               [-1.1, -3.3, 1.2], [2.2, 2.2, 1.5], [-4.0, -0.7, 1.0], [0.9, -4.4, 1.3]]
    ctx = torch.stack([
        torch.tensor(np.asarray(haa_md.get_3d_point_camera_coord(speaker_xyz, p),
                                dtype=np.float32))
        for p in ctx_xyz[:N_CTX]
    ])                                                                    # [N,3]

    return {
        "scene": "classroomBase",
        "depth": depth,
        "source": src,                       # [3]
        "source_vit": src.unsqueeze(0),      # [1,3]  (HAA_md:33)
        "context_poses": ctx,                # [N,3]
        "context_poses_vit": ctx,            # [N,3]
        "context_audio": torch.randn(N_CTX, 1, IR_LEN, generator=g),
    }


def test_the_synthetic_panorama_is_a_valid_equirectangular_map():
    """Premise check: if the fixture were not a real panorama, nothing below holds.

    Uses the repo's own sign sanity-check — per-column azimuth deviation stays
    ~0 through the C4 orbit only when the roll direction and the rotation-matrix
    sign agree.
    """
    from src.data.yaw_rotation import yaw_transform_consistency
    report = yaw_transform_consistency(haa_metadata()["depth"], IMG_W, DEFAULT_FRAME_ANGLES)
    assert max(report.values()) < 1e-3, report


# --------------------------------------------------------------------------- #
# the conditioning stand-ins
# --------------------------------------------------------------------------- #
def _column_vectors(md):
    """Per-column mean 3-vector of the panorama, [3, W] — rolls with the scene."""
    return md["depth"].mean(dim=1)


def anisotropic_features(md):
    """A deliberately NOT yaw-invariant conditioning function.

    Three parts, and the third exists because of a measurement:

      * ``cols``   — a column ramp over the panorama: moves when the scene rolls.
      * ``raw``    — the pose vectors as they are: move when the scene rotates.
      * ``coupled``— the panorama's per-column vectors projected onto each pose
        direction. **A feature that is LINEAR in the poses cannot detect a
        sign-flipped pose rotation at all**: the C4 average sends
        ``sum_g R(±g) v`` to ``(0, 0, z)`` either way, so the sign is annihilated
        (measured: gap 1.2e-7). The real ViT path is nonlinear and mixes the
        panorama with the poses, so the stand-in must too, or the "can this probe
        fail?" test would be measuring nothing.

    Means, not sums: the panorama has 131,072 pixels, and float32 accumulation
    over that many terms carries far more than the 1e-5 the real gate resolves.
    """
    d = md["depth"]
    col = _column_vectors(md)                                # [3,W]
    ramp = torch.linspace(0.0, 1.0, d.shape[-1], dtype=d.dtype)

    cols = (col * ramp).mean(dim=1)                          # [3]

    src_dir = md["source"] / md["source"].norm().clamp_min(1e-6)
    coupled_src = (torch.einsum("cw,c->w", col, src_dir) * ramp).mean().reshape(1)

    ctx_dir = md["context_poses"] / md["context_poses"].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    coupled_ctx = (torch.einsum("cw,nc->nw", col, ctx_dir) * ramp).mean(dim=1)   # [N]

    raw = torch.cat([md["source"].reshape(-1),
                     md["source_vit"].reshape(-1),
                     md["context_poses"].reshape(-1),
                     md["context_poses_vit"].reshape(-1)])
    return torch.cat([cols, coupled_src, coupled_ctx, raw])


def pose_linear_features(md):
    """The naive stand-in: the panorama ramp plus the raw poses, no coupling.

    Kept so the claim above ("a pose-linear feature cannot see a sign flip") is a
    test rather than a comment.
    """
    d = md["depth"]
    ramp = torch.linspace(0.0, 1.0, d.shape[-1], dtype=d.dtype)
    cols = (_column_vectors(md) * ramp).mean(dim=1)
    poses = torch.cat([md["source"].reshape(-1), md["context_poses"].reshape(-1)])
    return torch.cat([cols, poses])


def orbit_averaged(md, angles=DEFAULT_FRAME_ANGLES, features=anisotropic_features):
    """``(1/|G|) sum_g f(g.x)`` — the frame average, done correctly.

    Exactly the arithmetic ``invariant_conditioning`` performs on the ViT path,
    with the real ``rotate_scene_metadata``. Because G is a group, the result is
    invariant under every element of G.
    """
    img_w = md["depth"].shape[-1]
    parts = [features(rotate_scene_metadata(md, math.radians(a), img_w))
             for a in angles]
    return torch.stack(parts).sum(dim=0) / float(len(parts))


def rotate_wrong_sign(md, alpha_rad, img_w):
    """Rolls the panorama one way and rotates the poses the other.

    The exact failure R1 is about: HAA's source-centred, vertically flipped
    convention makes a sign disagreement between the depth path and the pose path
    plausible, and it would be invisible — both halves still "rotate".
    """
    out = rotate_scene_metadata(md, alpha_rad, img_w, pose_keys=())       # depth only
    alpha_eff = yaw_column_shift(alpha_rad, img_w) * 2.0 * math.pi / img_w
    rot = azimuth_rotation_matrix(-alpha_eff)                            # <-- flipped
    for key in POSE_KEYS:
        if key in md:
            pose = md[key]
            out[key] = torch.einsum("ij,...j->...i", rot.to(pose.dtype), pose)
    return out


def orbit_averaged_wrong_sign(md, angles=DEFAULT_FRAME_ANGLES, features=anisotropic_features):
    """The same frame average, built over a MISMATCHED orbit."""
    img_w = md["depth"].shape[-1]
    parts = [features(rotate_wrong_sign(md, math.radians(a), img_w))
             for a in angles]
    return torch.stack(parts).sum(dim=0) / float(len(parts))


def conditioner_shaped(md):
    """Output shaped like the real conditioner's ``{id: [tensor, mask]}``."""
    feats = orbit_averaged(md)
    return {
        "source_vit": [feats.reshape(1, 1, -1), torch.ones(1, 1)],
        "context_poses_vit": [feats.reshape(1, 1, -1) * 2.0, torch.ones(1, 1)],
    }


# --------------------------------------------------------------------------- #
# 1. the measurement is correct AND non-vacuous
# --------------------------------------------------------------------------- #
def test_a_true_frame_average_is_invariant_on_the_C4_orbit():
    """The positive case, at the threshold the real gate uses.

    Paired with the assertion that the UN-averaged function is wildly
    non-invariant on the same metadata: without that, a probe that always returns
    0.0 (wrong ``img_w``, a no-op rotation, a cond_fn ignoring its argument) would
    pass this test.
    """
    md = haa_metadata()
    gap_avg = probe.invariance_gap(orbit_averaged, md, probe.C4_ANGLES)
    gap_raw = probe.invariance_gap(anisotropic_features, md, probe.C4_ANGLES)

    assert gap_avg < probe.THRESHOLD, f"frame average is not invariant: {gap_avg:.3e}"
    assert gap_raw > 1.0, (
        f"the un-averaged control is only {gap_raw:.3e} off invariance — the "
        "probe would report success for a function that does nothing"
    )
    assert gap_raw / max(gap_avg, 1e-30) > 1e4


def test_the_default_angles_are_the_repo_C4_orbit():
    """One source of truth for the orbit; a private copy could drift off B-F's."""
    assert probe.C4_ANGLES == DEFAULT_FRAME_ANGLES == (0.0, 90.0, 180.0, 270.0)


def test_a_wrong_sign_pose_rotation_is_caught():
    """The R1 failure mode, simulated: depth rolled one way, poses rotated the other.

    The average over a mismatched orbit is not invariant, and the probe must say
    so loudly rather than at the edge of the threshold.
    """
    md = haa_metadata()
    gap = probe.invariance_gap(orbit_averaged_wrong_sign, md, probe.C4_ANGLES)
    assert gap > 1e3 * probe.THRESHOLD, (
        f"a sign-flipped pose rotation produced a gap of only {gap:.3e}; this "
        "probe cannot be used as a launch gate"
    )


def test_a_pose_linear_stand_in_would_hide_the_sign_flip():
    """The limit of what ANY frame-average gate can see — stated as a test.

    With conditioning that is linear in the poses, the C4 average maps
    ``sum_g R(+g) v`` and ``sum_g R(-g) v`` to the same ``(0, 0, z)``: the sign is
    algebraically annihilated and the gate reads clean. The gate is therefore
    informative about the REAL (nonlinear, panorama-coupled) stack and must be run
    on it — not on a simplified stand-in. Recorded here so the CLI's coverage is
    not overclaimed in the closing analysis.
    """
    md = haa_metadata()
    blind = functools.partial(orbit_averaged_wrong_sign, features=pose_linear_features)
    assert probe.invariance_gap(blind, md, probe.C4_ANGLES) < probe.THRESHOLD
    # …and the coupled stand-in, on the same broken rotation, does see it.
    assert probe.invariance_gap(orbit_averaged_wrong_sign, md, probe.C4_ANGLES) > 1e-2


def test_the_conditioner_shaped_output_is_measured_per_id():
    """The real ``cond_fn`` returns ``{id: [tensor, mask]}``; every entry is measured.

    A probe that silently looked at one id would pass a stack whose OTHER ViT
    conditioner is non-invariant.
    """
    md = haa_metadata()
    gaps = probe.invariance_gaps(conditioner_shaped, md, probe.C4_ANGLES)
    assert len(gaps) >= 4, gaps          # 2 ids x (tensor, mask)
    assert any("source_vit" in k for k in gaps)
    assert any("context_poses_vit" in k for k in gaps)
    assert max(gaps.values()) < probe.THRESHOLD
    assert probe.invariance_gap(conditioner_shaped, md, probe.C4_ANGLES) == max(gaps.values())


def test_the_gap_is_measured_across_the_whole_orbit_not_only_against_angle_zero():
    """"Identical whichever orbit element you start from" is a PAIRWISE claim.

    Adversarial by construction: the orbit outputs are offset by (0, +1, -1, 0),
    so the worst deviation from the angle-0 pass is 1.0 while the worst deviation
    between two orbit elements is 2.0. A probe that only compared against the base
    pass would under-report the gap by a factor of two — and near a 1e-5 gate,
    under-reporting is how a broken arm gets launched.
    """
    md = haa_metadata()
    offsets = iter([0.0, 1.0, -1.0, 0.0])

    def drifting(sample):
        return orbit_averaged(sample) + next(offsets)

    gap = probe.invariance_gap(drifting, md, probe.C4_ANGLES)
    assert gap == pytest.approx(2.0, abs=1e-3), gap


# --------------------------------------------------------------------------- #
# 2. the orbit must be a group
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "angles",
    [(0.0, 90.0, 180.0, 270.0), (0.0, 180.0), (0.0,),
     (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)],
)
def test_closed_orbits_are_accepted(angles):
    probe.check_closed_orbit(angles)          # must not raise


@pytest.mark.parametrize(
    "angles, why",
    [
        ((0.0, 90.0, 180.0), "90+180=270 is missing — not closed under composition"),
        ((0.0, 45.0), "45+45=90 is missing"),
        ((0.0, 90.0, 90.0, 180.0, 270.0), "duplicate element"),
        ((90.0, 180.0, 270.0, 0.0), "identity is not first"),
        ((), "empty"),
        ((0.0, 120.0, 240.0, 90.0), "mixed C3/C4 — not a subgroup"),
    ],
)
def test_orbits_that_are_not_closed_are_refused(angles, why):
    """A frame average over a non-group is not invariant under anything.

    Averaging over a set that is not closed gives a function no rotation leaves
    fixed — the probe would then measure a real, meaningless gap and the gate
    would fail for a reason nobody could act on.
    """
    with pytest.raises(ValueError):
        probe.check_closed_orbit(angles)


def test_the_gap_function_refuses_a_non_closed_orbit():
    """The validation is on the callable path, not only in the helper."""
    md = haa_metadata()
    with pytest.raises(ValueError):
        probe.invariance_gap(orbit_averaged, md, (0.0, 90.0, 180.0))


# --------------------------------------------------------------------------- #
# 3. fail-closed on a probe that would measure nothing
# --------------------------------------------------------------------------- #
def test_a_cond_fn_that_returns_nothing_is_refused():
    """The most dangerous pass: 0.0 measured over zero tensors."""
    md = haa_metadata()
    for empty in (lambda m: {}, lambda m: None, lambda m: []):
        with pytest.raises(ValueError, match="no tensors"):
            probe.invariance_gap(empty, md, probe.C4_ANGLES)


def test_a_cond_fn_whose_structure_varies_over_the_orbit_is_refused():
    """Different keys per angle means the comparison is undefined, not zero."""
    md = haa_metadata()
    seen = {"n": 0}

    def unstable(sample):
        seen["n"] += 1
        out = {"a": orbit_averaged(sample)}
        if seen["n"] > 1:
            out["b"] = orbit_averaged(sample)
        return out

    with pytest.raises(ValueError, match="same structure"):
        probe.invariance_gap(unstable, md, probe.C4_ANGLES)


def test_a_cond_fn_whose_shapes_vary_over_the_orbit_is_refused():
    md = haa_metadata()
    seen = {"n": 0}

    def reshaping(sample):
        seen["n"] += 1
        base = orbit_averaged(sample)
        return base if seen["n"] == 1 else base.reshape(-1, 1)

    with pytest.raises(ValueError, match="shape"):
        probe.invariance_gap(reshaping, md, probe.C4_ANGLES)


def test_metadata_without_depth_and_without_an_explicit_width_is_refused():
    """``img_w`` is what turns an angle into an exact column roll; guessing is not an option."""
    md = haa_metadata()
    md.pop("depth")
    with pytest.raises(ValueError, match="img_w"):
        probe.invariance_gap(anisotropic_features, md, probe.C4_ANGLES)


# --------------------------------------------------------------------------- #
# 4. the probe does not disturb what it measures
# --------------------------------------------------------------------------- #
def test_the_callers_metadata_is_not_mutated():
    """The gate runs inside an eval loop; a mutated batch would corrupt the run."""
    md = haa_metadata()
    keys_before = set(md)
    snapshot = {k: v.clone() for k, v in md.items() if torch.is_tensor(v)}
    probe.invariance_gap(orbit_averaged, md, probe.C4_ANGLES)
    assert set(md) == keys_before
    for k, v in snapshot.items():
        assert torch.equal(md[k], v), f"{k} was modified in place"


def test_the_yaw_invariant_fields_are_passed_through_untouched():
    """``context_audio`` and ``scene`` are yaw-invariant by physics.

    Rotating them would be wrong (the reference RIRs do not change when the room
    is spun), so the orbit must leave them identical — asserted on the object the
    probe's own rotation produces.
    """
    md = haa_metadata()
    rotated = rotate_scene_metadata(md, math.radians(90.0), IMG_W)
    assert torch.equal(rotated["context_audio"], md["context_audio"])
    assert rotated["scene"] == md["scene"]


# --------------------------------------------------------------------------- #
# 5. the CLI half exists and is wired for the gate
# --------------------------------------------------------------------------- #
def test_the_cli_defaults_are_the_haa_stack_and_the_registered_threshold():
    """Not exercised end-to-end here (needs the dataset + DINOv3); its wiring is.

    ``--threshold`` defaults to the plan's 1e-5 and the model config defaults to
    the stock HAA finetune config, so an operator running the gate with no flags
    runs the gate that was registered.
    """
    parser = probe.build_parser()
    args = parser.parse_args([])
    assert args.threshold == probe.THRESHOLD == 1e-5
    assert args.model_config.endswith("FLAC/HAA/FLAC_HAA_finetune.json")
    assert args.dataset_config.endswith("HAA/eval/haa_test.json")
    assert tuple(args.angles) == probe.C4_ANGLES


def test_importing_the_probe_pulls_in_no_model_or_dataset_code():
    """The pure half must stay importable without the ViT stack or the dataset.

    Otherwise these tests would silently depend on a DINOv3 download and an HAA
    checkout, and would stop being runnable on a machine that has neither.

    Run in a FRESH interpreter on purpose: ``sys.modules`` is shared across a
    pytest session, so an in-process check would pass or fail according to which
    other test module happened to be collected first — i.e. it would not be
    testing the probe at all.

    ``src.models.factory`` is deliberately not on the list: ``src/__init__.py``
    imports it, so *any* ``src.*`` import pulls that module in. What matters is
    that no ViT is constructed and no weights are fetched.
    """
    import subprocess
    import sys as _sys

    script = (
        "import importlib.util, sys;"
        f"sys.path.insert(0, {str(_REPO)!r});"
        f"spec = importlib.util.spec_from_file_location('p', {str(PROBE_PATH)!r});"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "leaked = [n for n in ('transformers', 'src.models.conditioners',"
        " 'src.data.dataset', 'pytorch_lightning') if n in sys.modules];"
        "print(leaked); sys.exit(1 if leaked else 0)"
    )
    proc = subprocess.run([_sys.executable, "-c", script], cwd=str(_REPO),
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"the probe imported model/dataset code: {proc.stdout}"
