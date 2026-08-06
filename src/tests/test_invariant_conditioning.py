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

import pytest
import torch
from torch import nn

from src.data import yaw_rotation as yr
from src.models.conditioners import MultiConditioner


DEV = "cpu"


# --------------------------------------------------------------------------- #
# geometrically consistent synthetic depth (adapted from test_yaw_symmetry)
# --------------------------------------------------------------------------- #
def _consistent_depth(H: int = 8, W: int = 512) -> torch.Tensor:
    """Geometrically consistent but NON-axisymmetric equirectangular depth
    point cloud.

    At column ``j`` the stored vector's xy-azimuth still equals ``theta_j``
    (the convention checked by ``yaw_transform_consistency``), so roll +
    z-rotation composes exactly. Unlike a constant-radius panorama — for which
    ``rotate_scene_metadata`` is a *fixed point* of the depth map, hiding
    stale-depth bugs — the radial distance varies with azimuth (periods 2*pi
    and pi, so every C4 roll changes the map). This is what lets the
    stale-depth negative test detect a depth/pose co-rotation bug
    (plan-review finding 8 / cycle-3 Codex review finding 1)."""
    j = torch.arange(W, dtype=torch.float32)
    theta = (j + 0.5) * 2.0 * math.pi / W - math.pi
    i = torch.arange(H, dtype=torch.float32)
    el = (i + 0.5) * math.pi / H - math.pi / 2.0
    theta_g = theta.view(1, W).expand(H, W)
    el_g = el.view(H, 1).expand(H, W)
    # Azimuth-dependent radius, strictly positive (minimum 3.0 - 1.5 = 1.5).
    d = 3.0 + 1.0 * torch.sin(theta_g) + 0.5 * torch.sin(2.0 * theta_g)
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
        self.samples = 0
        self.batch_sizes = []

    def forward(self, x_list, device=DEV):
        self.calls += 1
        self.samples += len(x_list)
        self.batch_sizes.append(len(x_list))
        x = torch.stack(x_list, dim=0).to(device)  # [B, 3] or [B, N, 3]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, 3]
        B, N, _ = x.shape
        feat = torch.tanh(x.sum(dim=-1, keepdim=True))  # [B, N, 1]
        out = feat.expand(B, N, self.out_dim).contiguous()
        return [out, torch.ones(B, N, device=device)]


class FakeGeometry(nn.Module):
    """GeometryConditioner stand-in (name matched so MultiConditioner feeds it
    ``{'coord', 'depth'}``). Deterministic *nonlinear* function of the
    coord−depth difference field with a genuinely nonzero-mean coord–depth
    interaction: ``tanh`` is applied per-pixel BEFORE pooling, so the coord
    contribution does not cancel out of the pooled statistic (the cycle-3
    review found that a zero-mean linear weight let it cancel), and the
    azimuthal pooling weight is strictly positive and non-uniform, so the
    statistic is also sensitive to the depth roll offset. A wrong
    implementation that rotates depth without the ``*_vit`` poses (or vice
    versa) therefore changes the C4 average and fails the invariance test."""

    def __init__(self, out_dim: int = 6, name: str = "GeometryConditioner"):
        super().__init__()
        self.name = name
        self.out_dim = out_dim
        self.calls = 0
        self.samples = 0
        self.batch_sizes = []
        g = torch.Generator().manual_seed(0)
        self.register_buffer("proj", torch.randn(3, out_dim, generator=g))

    def forward(self, coord_list, device=DEV):
        self.calls += 1
        self.samples += len(coord_list)
        self.batch_sizes.append(len(coord_list))
        coords, depths = [], []
        for c in coord_list:
            coords.append(c["coord"].to(device))      # dtype-preserving: the fp64
            depths.append(c["depth"].to(device))      # equivalence case runs in double
        coord = torch.stack(coords, dim=0)  # [B, 3] or [B, N, 3]
        if coord.ndim == 2:
            coord = coord.unsqueeze(1)  # [B, 1, 3]
        depth = torch.stack(depths, dim=0)  # [B, 3, H, W]
        B, N, _ = coord.shape
        W = depth.shape[-1]
        col = torch.arange(W, device=device, dtype=coord.dtype)
        # Strictly positive (in [0.5, 1.5]), non-uniform, full-period profile:
        # nonzero mean (== 1) so the coord term survives pooling; period W so
        # every nontrivial C4 roll misaligns depth content against it.
        w = (1.0 + 0.5 * torch.cos(2.0 * math.pi * col / W)).view(1, 1, 1, W)
        outs = []
        for i in range(N):
            diff = coord[:, i, :, None, None] - depth  # [B, 3, H, W]
            pooled = (torch.tanh(diff) * w).mean(dim=(2, 3))  # [B, 3]
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
        self.samples = 0
        self.batch_sizes = []
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, audios_list, device=DEV):
        self.calls += 1
        self.samples += len(audios_list)
        self.batch_sizes.append(len(audios_list))
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
    """Non-ViT conditioners run EXACTLY once; the ViT conditioners see exactly
    the orbit's worth of samples.

    The orbit is now executed as a few BATCHED forwards instead of one forward
    per angle, so the contract is stated in SAMPLES, not calls: the ViT path must
    see the base pass (B) plus (C-1)*B rotated samples, however they are grouped.
    The BatchNorm guard is unchanged — the RIR encoder must still be stepped once."""
    cond = _build_cond()
    batch = 2
    md = _batch(batch)
    _ = yr.invariant_conditioning(cond, md, DEV)
    n_angles = len(yr.DEFAULT_FRAME_ANGLES)

    assert cond.conditioners["source"].calls == 1
    assert cond.conditioners["context_poses"].calls == 1
    assert cond.conditioners["context_audio"].calls == 1
    assert cond.conditioners["source"].samples == batch
    assert cond.conditioners["context_poses"].samples == batch
    assert cond.conditioners["context_audio"].samples == batch
    for vit in VIT_IDS:
        assert cond.conditioners[vit].samples == n_angles * batch, (
            f"{vit} saw {cond.conditioners[vit].samples} samples, expected "
            f"{n_angles} x {batch} (base + orbit)"
        )
        assert cond.conditioners[vit].calls >= 1
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


# --------------------------------------------------------------------------- #
# 9. NEGATIVE: stale (unrotated) depth in the averaging loop breaks invariance
# --------------------------------------------------------------------------- #
def _stale_depth_conditioning(cond, metadata, device, angles=yr.DEFAULT_FRAME_ANGLES,
                              vit_ids=VIT_IDS):
    """Deliberately BROKEN mimic of ``invariant_conditioning``: per frame angle
    it rotates the ``*_vit`` poses but feeds the STALE (unrotated) depth map —
    the exact finding-8 bug class. Test-local only; exists to prove the C4
    invariance assertions in this file can actually catch that bug."""
    md_inv = [yr.cylindrical_pose_features(md) for md in metadata]
    base = cond(md_inv, device)
    present = [i for i in vit_ids if i in base]
    img_w = int(metadata[0]["depth"].shape[-1])
    accum = {i: base[i][0].clone() for i in present}
    for g in angles[1:]:
        variants = []
        for m in md_inv:
            v = yr.rotate_scene_metadata(m, math.radians(g), img_w,
                                         pose_keys=tuple(present))
            v["depth"] = m["depth"]  # BUG under test: depth not co-rotated
            variants.append(v)
        part = cond(variants, device, only_ids=present)
        for i in present:
            accum[i] = accum[i] + part[i][0]
    for i in present:
        base[i][0] = accum[i] / float(len(angles))
    return base


def test_stale_depth_fails_invariance():
    cond = _build_cond()
    md = _batch(2)
    broken_0 = _stale_depth_conditioning(cond, md, DEV)
    correct_0 = yr.invariant_conditioning(cond, md, DEV)
    rot = [yr.rotate_scene_metadata(m, math.radians(90.0), 512) for m in md]
    broken_g = _stale_depth_conditioning(cond, rot, DEV)
    for key in VIT_IDS:
        inv_div = float((broken_g[key][0] - broken_0[key][0]).abs().max())
        avg_div = float((broken_0[key][0] - correct_0[key][0]).abs().max())
        print(
            f"{key}: stale-depth invariance divergence {inv_div:.3e}, "
            f"divergence from correct orbit average {avg_div:.3e}"
        )
        # The broken variant must FAIL the same C4 invariance check the
        # positive test uses...
        assert not torch.allclose(broken_g[key][0], broken_0[key][0], atol=1e-5), (
            f"{key}: stale-depth variant unexpectedly passed the C4 invariance "
            "check — the FakeGeometry/depth fixtures are too weak to pin finding 8"
        )
        # ...and must not reproduce the correct orbit average either.
        assert not torch.allclose(broken_0[key][0], correct_0[key][0], atol=1e-5), (
            f"{key}: stale-depth variant matches the correct orbit average"
        )


# --------------------------------------------------------------------------- #
# 10. finer orbits (exp_11): Cn exact end-to-end invariance for n in {8,16,32}
# --------------------------------------------------------------------------- #
def _orbit(n: int) -> tuple:
    """Uniform Cn yaw orbit in degrees (0.0 first). For W=512 every member is
    an exact, 16-px-patch-aligned column roll (C32 = 16 columns)."""
    return tuple(k * 360.0 / n for k in range(n))


def _small_md(seed: int = 0, n_ctx: int = 2, depth_h: int = 2) -> dict:
    """Same fields as :func:`_make_md`, smaller. The Cn tests below are O(n^2)
    in conditioner passes (n rotations x an n-element orbit average), so the
    synthetic tensors are shrunk to keep this permanent CPU test cheap. The
    panorama width stays 512, so every orbit member is still an exact
    16-px-aligned column roll, and no group element or tolerance is relaxed."""
    g = torch.Generator().manual_seed(seed)
    return {
        "source": torch.randn(3, generator=g),
        "source_vit": torch.randn(1, 3, generator=g),
        "context_poses": torch.randn(n_ctx, 3, generator=g),
        "context_poses_vit": torch.randn(n_ctx, 3, generator=g),
        "context_audio": torch.randn(n_ctx, 1, 256, generator=g),
        "depth": _consistent_depth(depth_h, 512),
    }


def _small_batch(n: int = 1) -> list:
    return [_small_md(seed=s) for s in range(n)]


@pytest.fixture
def single_thread():
    """Pin torch to one intra-op thread for the O(n^2) orbit tests (restored
    afterwards). The synthetic tensors are tiny, so torch's default pool (52
    threads on this login node) spends its time in thread launch/sync: one C16
    orbit average measures ~1.2 s at the default vs ~0.01 s single-threaded.
    Also keeps this permanent CPU test a considerate tenant on a shared node."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


@pytest.mark.parametrize("n", (8, 16, 32))
def test_cn_exact_invariance(n, single_thread):
    cond = _build_cond()
    md = _small_batch(1)
    angles = _orbit(n)
    out_0 = yr.invariant_conditioning(cond, md, DEV, angles)
    for deg in angles[1:]:
        rot = [yr.rotate_scene_metadata(m, math.radians(deg), 512) for m in md]
        out_g = yr.invariant_conditioning(cond, rot, DEV, angles)
        assert set(out_g.keys()) == ALL_IDS
        for key in ALL_IDS:
            assert torch.allclose(out_g[key][0], out_0[key][0], atol=1e-5), (
                f"C{n}: {key} not invariant at {deg} deg"
            )


# --------------------------------------------------------------------------- #
# 11. averaging arithmetic for the non-C4 orbits (catches a wrong divisor)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", (8, 16, 32))
def test_cn_average_correctness(n, single_thread):
    cond = _build_cond()
    md = _small_batch(1)
    angles = _orbit(n)
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
# 12. batched orbit reproduces the pre-batching AVERAGING ARITHMETIC (exp_11 Q5)
#
# Scope of the claim: identical terms, identical accumulation order, identical
# divisor -- proven here on deterministic conditioners. It is NOT a claim that
# the whole training path is unchanged: train-mode DINOv3 draws its random RoPE
# rescale once per forward, so chunked angles share a draw (a disclosed recipe
# change applied identically to every arm; section 14 pins that new contract).
# --------------------------------------------------------------------------- #
def _reference_orbit_average(cond, metadata, device, angles):
    """The PRE-BATCHING algorithm, reproduced here on purpose.

    This is deliberately a copy rather than an import: it is the specification
    the batched implementation must reproduce, so it has to keep working even if
    the library's own reference helper is refactored away."""
    md_inv = [yr.cylindrical_pose_features(m) for m in metadata]
    base = cond(md_inv, device)
    present = [i for i in VIT_IDS if i in base]
    img_w = int(metadata[0]["depth"].shape[-1])
    accum = {i: base[i][0].clone() for i in present}
    for g in angles[1:]:
        variants = [
            yr.rotate_scene_metadata(m, math.radians(g), img_w, pose_keys=tuple(present))
            for m in md_inv
        ]
        part = cond(variants, device, only_ids=present)
        for i in present:
            accum[i] = accum[i] + part[i][0]
    for i in present:
        base[i][0] = accum[i] / float(len(angles))
    return base


@pytest.mark.parametrize("n", (4, 8, 16, 32))
@pytest.mark.parametrize("batch", (1, 2, 3))
def test_batched_orbit_matches_the_sequential_reference(n, batch, single_thread):
    angles = _orbit(n)
    md = _small_batch(batch)
    got = yr.invariant_conditioning(_build_cond(), md, DEV, angles)
    want = _reference_orbit_average(_build_cond(), md, DEV, angles)
    for key in VIT_IDS:
        assert torch.allclose(got[key][0], want[key][0], atol=1e-5), (
            f"C{n} B{batch}: {key} differs from the sequential reference "
            f"(max {float((got[key][0] - want[key][0]).abs().max()):.3e})"
        )
    for key in ("source", "context_poses", "context_audio"):
        assert torch.allclose(got[key][0], want[key][0], atol=1e-5), key


@pytest.mark.parametrize("n", (4, 32))
def test_batched_orbit_matches_reference_in_float64(n, single_thread):
    """In fp64 the two groupings agree far below the fp32 tolerance: the residual
    fp32 gap is kernel-shape rounding, not different averaging arithmetic."""
    angles = _orbit(n)
    md = [
        {k: (v.double() if torch.is_floating_point(v) else v) for k, v in m.items()}
        for m in _small_batch(2)
    ]
    got = yr.invariant_conditioning(_build_cond().double(), md, DEV, angles)
    want = _reference_orbit_average(_build_cond().double(), md, DEV, angles)
    for key in VIT_IDS:
        assert got[key][0].dtype == torch.float64
        assert torch.allclose(got[key][0], want[key][0], atol=1e-7), (
            f"C{n} fp64: {key} max {float((got[key][0] - want[key][0]).abs().max()):.3e}"
        )


def test_orbit_chunking_covers_every_rotated_sample(single_thread):
    """B=3, C=32: 31 rotated angles x 3 = 93 samples with a 64-sample cap, so the
    chunks are 21 and 10 angles (63 + 30 samples) — a boundary that is NOT a
    multiple of the cap. Every sample must still be forwarded exactly once."""
    batch, n = 3, 32
    assert (n - 1) * batch % yr.FRAME_AVG_MAX_FWD_SAMPLES != 0
    cond = _build_cond()
    out = yr.invariant_conditioning(cond, _small_batch(batch), DEV, _orbit(n))
    for vit in VIT_IDS:
        c = cond.conditioners[vit]
        assert c.samples == n * batch, f"{vit} saw {c.samples}, expected {n * batch}"
        assert c.calls < n, f"{vit} used {c.calls} calls; batching should need far fewer than {n}"
        assert all(s <= yr.FRAME_AVG_MAX_FWD_SAMPLES for s in c.batch_sizes[1:]), (
            f"{vit} exceeded the {yr.FRAME_AVG_MAX_FWD_SAMPLES}-sample cap: {c.batch_sizes}"
        )
    assert set(out.keys()) == ALL_IDS


def test_orbit_cap_is_respected_for_a_large_batch(single_thread):
    """With B = 8 the cap allows 8 angles per forward; C16 must therefore split
    the 15 rotated angles into chunks of at most 64 samples."""
    batch, n = 8, 16
    cond = _build_cond()
    yr.invariant_conditioning(cond, _small_batch(batch), DEV, _orbit(n))
    for vit in VIT_IDS:
        c = cond.conditioners[vit]
        assert c.samples == n * batch
        assert max(c.batch_sizes[1:]) <= yr.FRAME_AVG_MAX_FWD_SAMPLES


# --------------------------------------------------------------------------- #
# 13. exact chunk plans, mask provenance, cap boundary (review N8/N9)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,batch,plan", [
    (4, 8, [8, 24]),
    (8, 8, [8, 56]),
    (16, 8, [8, 64, 56]),
    (32, 8, [8, 64, 64, 64, 56]),
    (32, 3, [3, 63, 30]),
])
def test_chunk_plans_are_exact(n, batch, plan, single_thread):
    """Pin the PRODUCTION batch plan, base call included.

    A sample-count contract alone would still pass if the implementation
    regressed to many small forwards — which is exactly the multi-day latency the
    batching exists to remove — so the plan itself is the regression asset."""
    cond = _build_cond()
    yr.invariant_conditioning(cond, _small_batch(batch), DEV, _orbit(n))
    for vit in VIT_IDS:
        assert cond.conditioners[vit].batch_sizes == plan, (
            f"C{n}/B{batch}: {vit} plan {cond.conditioners[vit].batch_sizes} != {plan}"
        )


class MaskSentinelGeometry(FakeGeometry):
    """FakeGeometry whose MASK encodes which call produced it (base call == 1)."""

    def forward(self, coord_list, device=DEV):
        out = super().forward(coord_list, device)
        out[1] = torch.full_like(out[1], float(self.calls))
        return out


def test_masks_come_from_the_base_pass(single_thread):
    """The returned masks must be the angle-0 pass's, not a batched chunk's."""
    cond = MultiConditioner({
        "source": FakeDist(), "context_poses": FakeDist(), "context_audio": FakeRIR(),
        "source_vit": MaskSentinelGeometry(), "context_poses_vit": MaskSentinelGeometry(),
    })
    batch = 8
    out = yr.invariant_conditioning(cond, _small_batch(batch), DEV, _orbit(32))
    for vit in VIT_IDS:
        assert cond.conditioners[vit].calls == 5, cond.conditioners[vit].batch_sizes
        mask = out[vit][1]
        assert mask.shape[0] == batch, f"{vit}: mask batch {mask.shape[0]} != base batch {batch}"
        assert torch.equal(mask, torch.ones_like(mask)), (
            f"{vit}: mask carries call id {float(mask.flatten()[0])}, expected the base pass (1)"
        )


def test_batch_at_the_cap_is_accepted_and_above_it_is_rejected(single_thread):
    """B == cap is the evaluation batch and must work; B > cap cannot be honoured
    (a chunk is whole angles, so it could not be split) and must fail loudly."""
    at_cap = yr.FRAME_AVG_MAX_FWD_SAMPLES
    cond = _build_cond()
    yr.invariant_conditioning(cond, _small_batch(at_cap), DEV, _orbit(4))
    assert cond.conditioners["source_vit"].batch_sizes == [at_cap] + [at_cap] * 3

    with pytest.raises(ValueError) as e:
        yr.invariant_conditioning(_build_cond(), _small_batch(at_cap + 1), DEV, _orbit(4))
    assert str(yr.FRAME_AVG_MAX_FWD_SAMPLES) in str(e.value)


# --------------------------------------------------------------------------- #
# 14. TRAIN-MODE contract: chunk-shared stochastic draws (disclosed recipe
#     change), determinism and gradient flow  (review finding 1, option 2)
# --------------------------------------------------------------------------- #
class StochasticGeometry(FakeGeometry):
    """Stand-in for train-mode DINOv3, whose RoPE rescale draws ONE random value
    per forward and applies it to the whole call.

    Under the batched execution a chunk's angles therefore share a draw where the
    per-angle loop gave them independent ones: the averaging arithmetic is
    unchanged, the augmentation schedule is not. That is the disclosed recipe
    change, applied identically to every arm (C4L included), and these tests pin
    the NEW contract rather than pretending the old one still holds."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.draws = 0
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, coord_list, device=DEV):
        out = super().forward(coord_list, device)
        if self.training:
            self.draws += 1
            rope = torch.rand(1, device=device)          # consumes global RNG, per FORWARD
            out[0] = out[0] * (1.0 + 0.1 * rope) * self.scale
        else:
            out[0] = out[0] * self.scale
        return out


def _stochastic_cond():
    return MultiConditioner({
        "source": FakeDist(), "context_poses": FakeDist(), "context_audio": FakeRIR(),
        "source_vit": StochasticGeometry(), "context_poses_vit": StochasticGeometry(),
    })


@pytest.mark.parametrize("n,batch,expected_draws", [(4, 8, 2), (16, 8, 3), (32, 8, 5), (32, 3, 3)])
def test_train_mode_draws_are_one_per_forward_not_one_per_angle(n, batch, expected_draws,
                                                                single_thread):
    """The batched path consumes exactly one stochastic draw per FORWARD, i.e.
    ``1 + n_chunks`` — the plan pinned in section 13 — instead of one per angle."""
    cond = _stochastic_cond().train()
    torch.manual_seed(0)
    yr.invariant_conditioning(cond, _small_batch(batch), DEV, _orbit(n))
    for vit in VIT_IDS:
        assert cond.conditioners[vit].draws == expected_draws, (
            f"C{n}/B{batch}: {vit} drew {cond.conditioners[vit].draws} times, expected "
            f"{expected_draws} (1 base + {expected_draws - 1} chunk(s))"
        )
        assert len(cond.conditioners[vit].batch_sizes) == expected_draws


def test_train_mode_is_deterministic_under_a_fixed_seed(single_thread):
    """Chunk-shared draws are still reproducible: same seed -> same conditioning."""
    md = _small_batch(8)
    outs = []
    for _ in range(2):
        cond = _stochastic_cond().train()
        torch.manual_seed(1234)
        out = yr.invariant_conditioning(cond, md, DEV, _orbit(16))
        outs.append({k: out[k][0].detach().clone() for k in VIT_IDS})
    for key in VIT_IDS:
        assert torch.equal(outs[0][key], outs[1][key]), f"{key} is not seed-deterministic"


def test_train_mode_gradients_reach_the_conditioner(single_thread):
    """The batched orbit must stay in the autograd graph: every rotated chunk
    contributes gradient to the conditioner parameters."""
    cond = _stochastic_cond().train()
    torch.manual_seed(7)
    out = yr.invariant_conditioning(cond, _small_batch(8), DEV, _orbit(8))
    loss = sum(out[k][0].float().pow(2).mean() for k in VIT_IDS)
    loss.backward()
    for vit in VIT_IDS:
        grad = cond.conditioners[vit].scale.grad
        assert grad is not None, f"{vit}: no gradient reached the conditioner"
        assert torch.isfinite(grad).all(), f"{vit}: non-finite gradient {grad}"
        assert float(grad.abs().max()) > 0.0, f"{vit}: zero gradient — the orbit is detached"
