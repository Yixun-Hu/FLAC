"""Tests for ``src.data.yaw_rotation.invariant_conditioning`` and the new
``only_ids`` parameter of ``src.models.conditioners.MultiConditioner.forward``.

These exercise the *real* ``MultiConditioner`` populated with small deterministic
fake conditioners (duck-typed ``nn.Module`` s registered under the exact FLAC ids
``source`` / ``source_vit`` / ``context_poses`` / ``context_poses_vit`` /
``context_audio``). The fakes are cheap stand-ins for the DINOv3 geometry stack,
the dist-embedder pose stack and the RIR encoder respectively; they let us assert
the Route-1 frame-averaging contract without loading any pretrained backbone:

- C4 frame closure gives exact end-to-end invariance (findings 2 & 8: depth MUST
  roll together with the ``*_vit`` poses, non-ViT conditioners run once).
- pose entries are invariant at arbitrary angles via the cylindrical features.
- the caller's metadata is never mutated (finding 4).
"""
import copy
import math

import torch
from torch import nn

from src.data import yaw_rotation as yr
from src.models.conditioners import MultiConditioner


DEV = "cpu"


# --------------------------------------------------------------------------- #
# geometrically consistent synthetic depth (adapted from test_yaw_symmetry)
# --------------------------------------------------------------------------- #
def _consistent_depth(H: int = 8, W: int = 512) -> torch.Tensor:
    """Equirectangular depth point cloud whose column-``j`` azimuth equals
    ``theta_j`` — so a roll + z-rotation is an *exact* symmetry of the map."""
    j = torch.arange(W, dtype=torch.float32)
    theta = (j + 0.5) * 2.0 * math.pi / W - math.pi
    i = torch.arange(H, dtype=torch.float32)
    el = (i + 0.5) * math.pi / H - math.pi / 2.0
    d = 3.0
    theta_g = theta.view(1, W).expand(H, W)
    el_g = el.view(H, 1).expand(H, W)
    x = d * torch.cos(el_g) * torch.cos(theta_g)
    y = d * torch.cos(el_g) * torch.sin(theta_g)
    z = d * torch.sin(el_g)
    return torch.stack([x, y, z], dim=0).contiguous()


def _make_md(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "source": torch.randn(3, generator=g),
        "source_vit": torch.randn(1, 3, generator=g),
        "context_poses": torch.randn(4, 3, generator=g),
        "context_poses_vit": torch.randn(4, 3, generator=g),
        "context_audio": torch.randn(4, 1, 256, generator=g),
        "depth": _consistent_depth(8, 512),
    }


def _batch(n: int = 2) -> list:
    return [_make_md(seed=s) for s in range(n)]


# --------------------------------------------------------------------------- #
# fake conditioners (duck-typed nn.Module, keyed exactly like FLAC)
# --------------------------------------------------------------------------- #
class FakeDist(nn.Module):
    """dist_embedder stand-in: deterministic nonlinear function of its pose
    input, with a call counter."""

    def __init__(self, out_dim: int = 8, name: str = "DistEmbedderConditioner"):
        super().__init__()
        self.name = name
        self.out_dim = out_dim
        self.calls = 0

    def forward(self, x_list, device=DEV):
        self.calls += 1
        x = torch.stack(x_list, dim=0).to(device)  # [B, 3] or [B, N, 3]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, 3]
        B, N, _ = x.shape
        feat = torch.tanh(x.sum(dim=-1, keepdim=True))  # [B, N, 1]
        out = feat.expand(B, N, self.out_dim).contiguous()
        return [out, torch.ones(B, N, device=device)]


class FakeGeometry(nn.Module):
    """GeometryConditioner stand-in (name matched so MultiConditioner feeds it
    ``{'coord', 'depth'}``). Deterministic *nonlinear* function of both the
    coord−depth difference field AND the depth roll (via azimuthal column
    weighting) — so a wrong implementation that rotates depth without the poses
    (or vice versa) changes the C4 average and fails the invariance test."""

    def __init__(self, out_dim: int = 6, name: str = "GeometryConditioner"):
        super().__init__()
        self.name = name
        self.out_dim = out_dim
        self.calls = 0
        g = torch.Generator().manual_seed(0)
        self.register_buffer("proj", torch.randn(3, out_dim, generator=g))

    def forward(self, coord_list, device=DEV):
        self.calls += 1
        coords, depths = [], []
        for c in coord_list:
            coords.append(c["coord"].float().to(device))
            depths.append(c["depth"].float().to(device))
        coord = torch.stack(coords, dim=0)  # [B, 3] or [B, N, 3]
        if coord.ndim == 2:
            coord = coord.unsqueeze(1)  # [B, 1, 3]
        depth = torch.stack(depths, dim=0)  # [B, 3, H, W]
        B, N, _ = coord.shape
        W = depth.shape[-1]
        col = torch.arange(W, device=device, dtype=coord.dtype)
        w = torch.cos(2.0 * math.pi * col / W).view(1, 1, 1, W)  # roll-sensitive
        outs = []
        for i in range(N):
            diff = coord[:, i, :, None, None] - depth  # [B, 3, H, W]
            pooled = torch.tanh((diff * w).mean(dim=(2, 3)))  # [B, 3]
            outs.append(pooled)
        stacked = torch.stack(outs, dim=1)  # [B, N, 3]
        out = torch.tanh(stacked @ self.proj)  # [B, N, out_dim]
        return [out, torch.ones(B, 1, device=device)]


class FakeRIR(nn.Module):
    """RIR encoder stand-in: constant-shaped output, a call counter, and an
    internal BatchNorm1d whose ``num_batches_tracked`` must increment exactly
    once per ``invariant_conditioning`` call (finding-2 regression guard)."""

    def __init__(self, out_dim: int = 8, name: str = "RIRConditioner"):
        super().__init__()
        self.name = name
        self.out_dim = out_dim
        self.calls = 0
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, audios_list, device=DEV):
        self.calls += 1
        audios = torch.stack(audios_list, dim=0).to(device)  # [B, N, C, T]
        B, N = audios.shape[0], audios.shape[1]
        flat = audios.reshape(B * N, -1)[:, : self.out_dim]  # [B*N, out_dim]
        feat = self.bn(flat)  # increments num_batches_tracked by 1
        out = feat.reshape(B, N, self.out_dim)
        return [out, torch.ones(B, 1, device=device)]


def _build_cond(with_geometry: bool = True) -> MultiConditioner:
    conditioners = {
        "source": FakeDist(),
        "context_poses": FakeDist(),
        "context_audio": FakeRIR(),
    }
    if with_geometry:
        conditioners["source_vit"] = FakeGeometry()
        conditioners["context_poses_vit"] = FakeGeometry()
    return MultiConditioner(conditioners)


ALL_IDS = {"source", "source_vit", "context_poses", "context_poses_vit", "context_audio"}
VIT_IDS = ("source_vit", "context_poses_vit")


# --------------------------------------------------------------------------- #
# 1. MultiConditioner.only_ids
# --------------------------------------------------------------------------- #
def test_multiconditioner_only_ids():
    cond = _build_cond()
    md = _batch(2)

    out = cond(md, DEV, only_ids=("source_vit",))
    assert set(out.keys()) == {"source_vit"}
    assert cond.conditioners["source_vit"].calls == 1
    for other in ("source", "context_poses", "context_poses_vit", "context_audio"):
        assert cond.conditioners[other].calls == 0, f"{other} ran under only_ids"

    # default None -> full behaviour (regression).
    out_all = cond(md, DEV)
    assert set(out_all.keys()) == ALL_IDS


# --------------------------------------------------------------------------- #
# 2. C4 exact end-to-end invariance (every key)
# --------------------------------------------------------------------------- #
def test_c4_exact_invariance():
    cond = _build_cond()
    md = _batch(2)
    out_0 = yr.invariant_conditioning(cond, md, DEV)
    for deg in (90.0, 180.0, 270.0):
        rot = [yr.rotate_scene_metadata(m, math.radians(deg), 512) for m in md]
        out_g = yr.invariant_conditioning(cond, rot, DEV)
        assert set(out_g.keys()) == ALL_IDS
        for key in ALL_IDS:
            assert torch.allclose(out_g[key][0], out_0[key][0], atol=1e-5), (
                f"{key} not invariant at {deg} deg"
            )


# --------------------------------------------------------------------------- #
# 3. pose entries invariant at an arbitrary (off-C4) angle
# --------------------------------------------------------------------------- #
def test_pose_entries_any_angle():
    cond = _build_cond()
    md = _batch(2)
    out_0 = yr.invariant_conditioning(cond, md, DEV)
    rot = [yr.rotate_scene_metadata(m, math.radians(37.3), 512) for m in md]
    out_g = yr.invariant_conditioning(cond, rot, DEV)
    for key in ("source", "context_poses"):
        assert torch.allclose(out_g[key][0], out_0[key][0], atol=1e-5), (
            f"{key} not invariant at 37.3 deg"
        )


# --------------------------------------------------------------------------- #
# 4. averaging arithmetic (mean over the |G| angle variants)
# --------------------------------------------------------------------------- #
def test_average_correctness():
    cond = _build_cond()
    md = _batch(2)
    angles = yr.DEFAULT_FRAME_ANGLES
    present = list(VIT_IDS)

    md_inv = [yr.cylindrical_pose_features(m) for m in md]
    frames = [md_inv]
    for g in angles[1:]:
        frames.append(
            [
                yr.rotate_scene_metadata(m, math.radians(g), 512, pose_keys=tuple(present))
                for m in md_inv
            ]
        )
    sums = {}
    for fr in frames:
        o = cond(fr, DEV)
        for pid in present:
            sums[pid] = o[pid][0] if pid not in sums else sums[pid] + o[pid][0]
    expected = {pid: sums[pid] / len(angles) for pid in present}

    out = yr.invariant_conditioning(cond, md, DEV, angles)
    for pid in present:
        assert torch.allclose(out[pid][0], expected[pid], atol=1e-5), pid


# --------------------------------------------------------------------------- #
# 5. single pass for non-ViT conditioners; |G| passes for ViT; BN once
# --------------------------------------------------------------------------- #
def test_single_pass_nonvit():
    cond = _build_cond()
    md = _batch(2)
    _ = yr.invariant_conditioning(cond, md, DEV)
    n_angles = len(yr.DEFAULT_FRAME_ANGLES)

    assert cond.conditioners["source"].calls == 1
    assert cond.conditioners["context_poses"].calls == 1
    assert cond.conditioners["context_audio"].calls == 1
    assert cond.conditioners["source_vit"].calls == n_angles
    assert cond.conditioners["context_poses_vit"].calls == n_angles
    assert int(cond.conditioners["context_audio"].bn.num_batches_tracked) == 1


# --------------------------------------------------------------------------- #
# 6. deep non-mutation of the caller's metadata
# --------------------------------------------------------------------------- #
def test_deep_nonmutating():
    cond = _build_cond()
    md = _batch(2)
    md_ref = copy.deepcopy(md)
    _ = yr.invariant_conditioning(cond, md, DEV)
    assert len(md) == len(md_ref)
    for m, m_ref in zip(md, md_ref):
        assert set(m.keys()) == set(m_ref.keys())
        for key in m_ref:
            assert torch.equal(m[key], m_ref[key]), f"{key} was mutated"
    # 'source' still raw xyz (the eval metric callback depends on this).
    for m, m_ref in zip(md, md_ref):
        assert torch.equal(m["source"], m_ref["source"])


# --------------------------------------------------------------------------- #
# 7. no-depth / no-ViT -> single pass, equals plain conditioner(cylindrical(md))
# --------------------------------------------------------------------------- #
def test_no_depth_single_pass():
    cond = _build_cond(with_geometry=False)
    md = [
        {
            "source": torch.randn(3),
            "context_poses": torch.randn(4, 3),
            "context_audio": torch.randn(4, 1, 256),
        }
        for _ in range(2)
    ]
    out = yr.invariant_conditioning(cond, md, DEV)
    plain = cond([yr.cylindrical_pose_features(m) for m in md], DEV)

    assert cond.conditioners["source"].calls == 2  # one here, one for `plain`
    for key in ("source", "context_poses", "context_audio"):
        assert torch.allclose(out[key][0], plain[key][0], atol=1e-6), key


# --------------------------------------------------------------------------- #
# 8. first angle must be the identity (0 deg)
# --------------------------------------------------------------------------- #
def test_angles_first_must_be_zero():
    cond = _build_cond()
    md = _batch(2)
    try:
        yr.invariant_conditioning(cond, md, DEV, angles=(90.0, 180.0))
    except ValueError:
        return
    raise AssertionError("expected ValueError when angles[0] != 0")
