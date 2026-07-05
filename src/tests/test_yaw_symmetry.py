"""Tests for yaw-symmetry helpers in ``src.data.yaw_rotation``.

Covers:
- ``wrap_angle`` — angle wrapping to the half-open interval ``(-pi, pi]``.
- ``cylindrical_pose_features`` — intrinsically yaw-invariant (r, z, dphi) pose
  features w.r.t. the target source azimuth.
- ``rotate_scene_metadata``'s new ``pose_keys`` parameter (default preserves the
  exp_02 semantics; restricted rotates only the listed pose keys).
"""
import copy
import math

import torch

from src.data import yaw_rotation as yr


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _consistent_depth(H: int = 8, W: int = 512) -> torch.Tensor:
    """Build a geometrically consistent equirectangular depth point cloud.

    At column ``j`` the stored 3D vector's horizontal azimuth equals
    ``theta_j = (j + 0.5) * 2*pi / W - pi`` (the convention checked by
    ``yaw_transform_consistency``), so a roll + z-rotation is an exact symmetry
    of the map.
    """
    j = torch.arange(W, dtype=torch.float32)
    theta = (j + 0.5) * 2.0 * math.pi / W - math.pi  # [W]
    i = torch.arange(H, dtype=torch.float32)
    el = (i + 0.5) * math.pi / H - math.pi / 2.0  # elevation [H]
    d = 3.0
    theta_g = theta.view(1, W).expand(H, W)
    el_g = el.view(H, 1).expand(H, W)
    x = d * torch.cos(el_g) * torch.cos(theta_g)
    y = d * torch.cos(el_g) * torch.sin(theta_g)
    z = d * torch.sin(el_g)
    return torch.stack([x, y, z], dim=0).contiguous()  # [3, H, W]


def _make_md(seed: int = 0, source=None) -> dict:
    g = torch.Generator().manual_seed(seed)
    if source is None:
        source = torch.randn(3, generator=g)
    else:
        source = torch.as_tensor(source, dtype=torch.float32)
    return {
        "source": source,
        "source_vit": torch.randn(1, 3, generator=g),
        "context_poses": torch.randn(4, 3, generator=g),
        "context_poses_vit": torch.randn(4, 3, generator=g),
        "depth": _consistent_depth(8, 512),
    }


# --------------------------------------------------------------------------- #
# wrap_angle
# --------------------------------------------------------------------------- #
def test_wrap_angle_scalar_cases():
    pi = math.pi
    assert math.isclose(yr.wrap_angle(0.0), 0.0, abs_tol=1e-12)
    assert math.isclose(yr.wrap_angle(pi), pi, abs_tol=1e-6)
    assert math.isclose(yr.wrap_angle(-pi), pi, abs_tol=1e-6)  # boundary -> +pi
    assert math.isclose(yr.wrap_angle(3 * pi / 2), -pi / 2, abs_tol=1e-6)
    assert math.isclose(yr.wrap_angle(-3 * pi / 2), pi / 2, abs_tol=1e-6)


def test_wrap_angle_tensor():
    pi = math.pi
    inp = torch.tensor([0.0, pi, -pi, 3 * pi / 2, -3 * pi / 2])
    expected = torch.tensor([0.0, pi, pi, -pi / 2, pi / 2])
    out = yr.wrap_angle(inp)
    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# cylindrical_pose_features
# --------------------------------------------------------------------------- #
def test_cylindrical_values():
    md = {
        "source": torch.tensor([1.0, 1.0, 0.5]),
        "context_poses": torch.tensor([[0.0, 2.0, 1.0]]),
    }
    out = yr.cylindrical_pose_features(md)
    src = out["source"]
    ctx = out["context_poses"]
    assert torch.allclose(
        src, torch.tensor([math.sqrt(2.0), 0.5, 0.0]), atol=1e-6
    )
    assert torch.allclose(
        ctx, torch.tensor([[2.0, 1.0, math.pi / 4]]), atol=1e-6
    )


def test_cylindrical_invariance():
    md = _make_md(seed=1)
    base = yr.cylindrical_pose_features(md)
    for deg in (90.0, 180.0, 37.3, -118.0):
        rot = yr.rotate_scene_metadata(md, math.radians(deg), 512)
        rot_feat = yr.cylindrical_pose_features(rot)
        assert torch.allclose(
            rot_feat["source"], base["source"], atol=1e-5
        ), f"source not invariant at {deg} deg"
        assert torch.allclose(
            rot_feat["context_poses"], base["context_poses"], atol=1e-5
        ), f"context_poses not invariant at {deg} deg"


def test_cylindrical_source_dphi_zero():
    for seed in range(5):
        md = _make_md(seed=100 + seed)
        out = yr.cylindrical_pose_features(md)
        assert float(out["source"][2]) == 0.0


def test_cylindrical_degenerate_source():
    # source on the z-axis (r_s = 0); contexts non-degenerate.
    g = torch.Generator().manual_seed(7)
    md = {
        "source": torch.tensor([0.0, 0.0, 1.7]),
        "context_poses": torch.randn(4, 3, generator=g),
        "depth": _consistent_depth(8, 512),
    }
    out = yr.cylindrical_pose_features(md)
    assert not torch.isnan(out["source"]).any()
    assert not torch.isnan(out["context_poses"]).any()
    # invariance must still hold in the fallback (largest-r pose) branch.
    base = yr.cylindrical_pose_features(md)
    for deg in (90.0, 37.3):
        rot = yr.rotate_scene_metadata(md, math.radians(deg), 512)
        rot_feat = yr.cylindrical_pose_features(rot)
        assert torch.allclose(
            rot_feat["source"], base["source"], atol=1e-5
        )
        assert torch.allclose(
            rot_feat["context_poses"], base["context_poses"], atol=1e-5
        )

    # all-degenerate: source and every context on the z-axis -> dphi == 0.
    md_all = {
        "source": torch.tensor([0.0, 0.0, 1.0]),
        "context_poses": torch.tensor(
            [[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]
        ),
    }
    out_all = yr.cylindrical_pose_features(md_all)
    assert not torch.isnan(out_all["context_poses"]).any()
    assert torch.allclose(
        out_all["context_poses"][:, 2], torch.zeros(2), atol=1e-12
    )


def test_cylindrical_nonmutating():
    md = _make_md(seed=3)
    md_ref = copy.deepcopy(md)
    _ = yr.cylindrical_pose_features(md)
    for key in md_ref:
        assert torch.equal(md[key], md_ref[key]), f"{key} was mutated"


def test_cylindrical_no_context():
    md = {"source": torch.tensor([1.0, 1.0, 0.5])}
    out = yr.cylindrical_pose_features(md)
    assert torch.allclose(
        out["source"], torch.tensor([math.sqrt(2.0), 0.5, 0.0]), atol=1e-6
    )
    assert "context_poses" not in out


# --------------------------------------------------------------------------- #
# rotate_scene_metadata pose_keys param
# --------------------------------------------------------------------------- #
def test_rotate_pose_keys_default():
    md = _make_md(seed=5)
    alpha = math.pi / 2.0  # 90 deg == 128 columns of W=512 -> exact
    out = yr.rotate_scene_metadata(md, alpha, 512)
    rot = yr.azimuth_rotation_matrix(alpha)
    for key in yr.POSE_KEYS:
        expected = torch.einsum("ij,...j->...i", rot.to(md[key].dtype), md[key])
        assert torch.allclose(out[key], expected, atol=1e-6), f"{key} mismatch"
    # depth is rotated (rolled + rotated), so it must differ from the input.
    assert not torch.equal(out["depth"], md["depth"])


def test_rotate_pose_keys_restricted():
    md = _make_md(seed=6)
    alpha = math.pi / 2.0
    out = yr.rotate_scene_metadata(
        md, alpha, 512, pose_keys=("source_vit", "context_poses_vit")
    )
    rot = yr.azimuth_rotation_matrix(alpha)
    # non-listed pose keys pass through bit-identical.
    assert torch.equal(out["source"], md["source"])
    assert torch.equal(out["context_poses"], md["context_poses"])
    # listed *_vit keys are rotated.
    for key in ("source_vit", "context_poses_vit"):
        expected = torch.einsum("ij,...j->...i", rot.to(md[key].dtype), md[key])
        assert torch.allclose(out[key], expected, atol=1e-6), f"{key} mismatch"
    # depth handling is unchanged -> still rotated.
    assert not torch.equal(out["depth"], md["depth"])
