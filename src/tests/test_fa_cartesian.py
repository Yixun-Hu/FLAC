"""exp_21 (bf_fa_cartesian) — ``src.data.yaw_rotation.fa_cartesian_conditioning``.

The arm's single mechanism change vs the registered B-F (``fa_invariant``) is the
POSE branch's symmetrization scheme: B-F's intrinsically-invariant cylindrical
triplets are replaced by a C4 frame average of the UNCHANGED Cartesian
dist-embedder, so all four pose/geometry conditioners are averaged over the same
orbit and ``context_audio`` still runs once. These tests pin that contract on the
*real* ``MultiConditioner`` populated with small deterministic stand-ins for the
DINOv3 geometry stack, the dist-embedder pose stack and the RIR encoder — the
same fixture strategy as ``test_invariant_conditioning`` /
``test_frame_avg_cap_config``, so no pretrained backbone is loaded.

Fixture requirements this file adds over those two:

* the dist stand-in must be **non-odd**. A plain ``tanh(sum(x))`` pose feature has
  a C4 orbit ``{a, b, -a, -b}``, so its frame average is IDENTICALLY ZERO for
  every pose — the invariance test would pass on a constant and the 45-degree
  negative control could never break. The bias/sigmoid/cos terms below remove
  that accidental symmetry (verified by the magnitude guards in
  :func:`test_45_degree_pre_rotation_breaks_the_average`).
* the geometry stand-in must read BOTH its own ``*_vit`` pose and a
  roll-sensitive reduction of ``depth``, so a frame that rotated one without the
  other is observable (``test_invariant_conditioning``'s finding-8 lesson).
"""
import copy
import math

import pytest
import torch
from torch import nn

from src.data import yaw_rotation as yr
from src.models.conditioners import MultiConditioner

DEV = "cpu"

POSE_KEYS = yr.POSE_KEYS                                   # the exp_21 orbit id set
VIT_IDS = ("source_vit", "context_poses_vit")
DIST_IDS = ("source", "context_poses")
ALL_IDS = set(POSE_KEYS) | {"context_audio"}
C4 = yr.DEFAULT_FRAME_ANGLES                               # (0, 90, 180, 270)

# Panorama geometry for the synthetic scenes. W is a multiple of 8, so every C4
# angle AND the 45-degree negative control quantise to an exact column roll
# (no interpolation, no angle the fixture cannot represent). H and W are kept
# tiny: this is a permanent CPU test on a shared box.
H, W = 2, 16
N_CTX = 3


@pytest.fixture(autouse=True)
def single_thread():
    """One intra-op thread (restored afterwards). The tensors here are tiny, so
    torch's default pool spends its time launching threads; the same trick keeps
    ``test_invariant_conditioning``'s orbit tests at ~10 ms each."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


# --------------------------------------------------------------------------- #
# synthetic scene
# --------------------------------------------------------------------------- #
def _consistent_depth(h: int = H, w: int = W) -> torch.Tensor:
    """Geometrically consistent, NON-axisymmetric equirectangular depth cloud.

    At column ``j`` the stored vector's xy-azimuth equals ``theta_j``, so a roll
    plus a z-rotation composes exactly (the convention
    ``yaw_transform_consistency`` checks). The radius varies with azimuth
    (periods 2*pi and pi), so every C4 roll genuinely changes the map — a
    constant-radius panorama is a fixed point of the rotation and would hide a
    stale-depth bug."""
    j = torch.arange(w, dtype=torch.float32)
    theta = (j + 0.5) * 2.0 * math.pi / w - math.pi
    i = torch.arange(h, dtype=torch.float32)
    el = (i + 0.5) * math.pi / h - math.pi / 2.0
    theta_g = theta.view(1, w).expand(h, w)
    el_g = el.view(h, 1).expand(h, w)
    d = 3.0 + 1.0 * torch.sin(theta_g) + 0.5 * torch.sin(2.0 * theta_g)
    x = d * torch.cos(el_g) * torch.cos(theta_g)
    y = d * torch.cos(el_g) * torch.sin(theta_g)
    z = d * torch.sin(el_g)
    return torch.stack([x, y, z], dim=0).contiguous()


def _md(seed: int = 0, n_ctx: int = N_CTX, width: int = W) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "source": torch.randn(3, generator=g),
        "source_vit": torch.randn(1, 3, generator=g),
        "context_poses": torch.randn(n_ctx, 3, generator=g),
        "context_poses_vit": torch.randn(n_ctx, 3, generator=g),
        "context_audio": torch.randn(n_ctx, 1, 64, generator=g),
        "depth": _consistent_depth(H, width),
        "scene": f"Toy/Toy_idx_{seed}",
    }


def _batch(n: int = 2, width: int = W) -> list:
    return [_md(seed=s, width=width) for s in range(n)]


def _rot(md: list, deg: float, width: int = W) -> list:
    """Pre-rotate a whole scene: depth AND all four pose keys (the default)."""
    return [yr.rotate_scene_metadata(m, math.radians(deg), width) for m in md]


# --------------------------------------------------------------------------- #
# fake conditioners (duck-typed nn.Module, keyed exactly like FLAC)
# --------------------------------------------------------------------------- #
class FakeDist(nn.Module):
    """dist_embedder stand-in: deterministic, rotation-sensitive and **non-odd**.

    ``h = x @ w + b`` mixes the three Cartesian components; the bias plus the
    sigmoid/cos terms make ``f(-u) != -f(u)``, which is what keeps the C4 frame
    average from collapsing to zero (see the module docstring). Records every
    forward's batch size and the exact objects it returned, so the tests can
    assert mask PROVENANCE by identity rather than by value."""

    def __init__(self, out_dim: int = 6, name: str = "DistEmbedderConditioner",
                 seed: int = 0):
        super().__init__()
        self.name = name
        self.out_dim = out_dim
        self.calls = 0
        self.samples = 0
        self.batch_sizes = []
        self.returns = []
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("w", torch.randn(3, out_dim, generator=g))
        self.register_buffer("b", torch.randn(out_dim, generator=g))

    def forward(self, x_list, device=DEV):
        self.calls += 1
        self.samples += len(x_list)
        self.batch_sizes.append(len(x_list))
        x = torch.stack(x_list, dim=0).to(device)      # [B, 3] or [B, N, 3]
        if x.dim() == 2:
            x = x.unsqueeze(1)                         # [B, 1, 3]
        B, N, _ = x.shape
        h = x @ self.w + self.b                        # [B, N, out_dim]
        out = torch.tanh(h) * torch.sigmoid(h - 0.5) + 0.25 * torch.cos(h)
        ret = [out.contiguous(), torch.ones(B, N, device=device)]
        self.returns.append(ret)
        return ret


class FakeGeometry(nn.Module):
    """GeometryConditioner stand-in (name matched so ``MultiConditioner`` feeds it
    ``{'coord', 'depth'}``).

    A nonlinear function of the coord-depth difference field with a nonzero-mean
    coord/depth interaction: ``tanh`` is applied per pixel BEFORE pooling so the
    coord contribution cannot cancel out of the pooled statistic, and the
    azimuthal pooling weight is strictly positive, non-uniform and full-period so
    the statistic also sees the depth ROLL. The output head carries a bias (and a
    sigmoid term), so — as for :class:`FakeDist` — the orbit average is not
    forced to zero by odd symmetry."""

    def __init__(self, out_dim: int = 5, name: str = "GeometryConditioner",
                 seed: int = 0):
        super().__init__()
        self.name = name
        self.out_dim = out_dim
        self.calls = 0
        self.samples = 0
        self.batch_sizes = []
        self.returns = []
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("proj", torch.randn(3, out_dim, generator=g))
        self.register_buffer("head_b", torch.randn(out_dim, generator=g))

    def forward(self, coord_list, device=DEV):
        self.calls += 1
        self.samples += len(coord_list)
        self.batch_sizes.append(len(coord_list))
        coords = [c["coord"].to(device) for c in coord_list]
        depths = [c["depth"].to(device) for c in coord_list]
        coord = torch.stack(coords, dim=0)             # [B, 3] or [B, N, 3]
        if coord.ndim == 2:
            coord = coord.unsqueeze(1)
        depth = torch.stack(depths, dim=0)             # [B, 3, H, W]
        B, N, _ = coord.shape
        width = depth.shape[-1]
        col = torch.arange(width, device=device, dtype=coord.dtype)
        w = (1.0 + 0.5 * torch.cos(2.0 * math.pi * col / width)).view(1, 1, 1, width)
        outs = []
        for i in range(N):
            diff = coord[:, i, :, None, None] - depth  # [B, 3, H, W]
            outs.append((torch.tanh(diff) * w).mean(dim=(2, 3)))
        stacked = torch.stack(outs, dim=1)             # [B, N, 3]
        h = stacked @ self.proj + self.head_b
        out = torch.tanh(h) + 0.3 * torch.sigmoid(2.0 * h)
        ret = [out.contiguous(), torch.ones(B, 1, device=device)]
        self.returns.append(ret)
        return ret


class FakeRIR(nn.Module):
    """RIR encoder stand-in with an internal BatchNorm1d: its
    ``num_batches_tracked`` must increment exactly ONCE per call, which is the
    single-pass contract for ``context_audio`` stated as running state rather
    than as a counter the implementation could satisfy by accident."""

    def __init__(self, out_dim: int = 8, name: str = "RIRConditioner"):
        super().__init__()
        self.name = name
        self.out_dim = out_dim
        self.calls = 0
        self.samples = 0
        self.batch_sizes = []
        self.returns = []
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, audios_list, device=DEV):
        self.calls += 1
        self.samples += len(audios_list)
        self.batch_sizes.append(len(audios_list))
        audios = torch.stack(audios_list, dim=0).to(device)   # [B, N, C, T]
        B, N = audios.shape[0], audios.shape[1]
        flat = audios.reshape(B * N, -1)[:, : self.out_dim]
        out = self.bn(flat).reshape(B, N, self.out_dim)
        ret = [out.contiguous(), torch.ones(B, 1, device=device)]
        self.returns.append(ret)
        return ret


def _build_cond(ids=ALL_IDS, train: bool = False) -> MultiConditioner:
    """The production id set by default. Eval mode unless a test needs the
    BatchNorm bookkeeping — the mocks are otherwise deterministic in both."""
    table = {
        "source": lambda: FakeDist(seed=1),
        "context_poses": lambda: FakeDist(seed=2),
        "source_vit": lambda: FakeGeometry(seed=3),
        "context_poses_vit": lambda: FakeGeometry(seed=4),
        "context_audio": lambda: FakeRIR(),
    }
    cond = MultiConditioner({k: table[k]() for k in table if k in ids})
    return cond.train() if train else cond.eval()


def _plans(cond, ids=POSE_KEYS):
    return {i: list(cond.conditioners[i].batch_sizes) for i in ids}


# --------------------------------------------------------------------------- #
# 1. T-C4-invariance — the WHOLE output dict, every id
# --------------------------------------------------------------------------- #
def test_c4_pre_rotation_leaves_every_output_unchanged():
    """(1/|G|) sum_g f(g h x) = (1/|G|) sum_g f(g x) for h in C4: pre-rotating the
    raw scene permutes the orbit and must not move a single conditioning tensor —
    including ``context_audio``, which never rotates at all."""
    md = _batch(2)
    out_0 = yr.fa_cartesian_conditioning(_build_cond(), md, DEV)
    assert set(out_0) == ALL_IDS
    for deg in (90.0, 180.0, 270.0):
        out_g = yr.fa_cartesian_conditioning(_build_cond(), _rot(md, deg), DEV)
        assert set(out_g) == ALL_IDS
        for key in ALL_IDS:
            assert torch.allclose(out_g[key][0], out_0[key][0], atol=1e-5), (
                f"{key} not invariant at {deg} deg "
                f"(max {float((out_g[key][0] - out_0[key][0]).abs().max()):.3e})"
            )


# --------------------------------------------------------------------------- #
# 2. T-vit-branch-pinned — the ViT branch is numerically the fa_invariant one
# --------------------------------------------------------------------------- #
def test_vit_branch_is_numerically_the_invariant_conditioning_one():
    """The arm changes ONE mechanism. ``GeometryConditioner`` reads only its own
    ``*_vit`` pose and ``depth``, and the cylindrical step touches neither, so the
    averaged ``*_vit`` outputs must agree with ``invariant_conditioning``'s on
    identical input. Conditioner-level pin only: training gradients still differ
    downstream, because the dist branch's inputs changed."""
    md = _batch(2)
    got = yr.fa_cartesian_conditioning(_build_cond(), md, DEV, C4)
    want = yr.invariant_conditioning(_build_cond(), md, DEV, C4)
    for key in VIT_IDS:
        assert torch.allclose(got[key][0], want[key][0], atol=1e-5), (
            f"{key} diverges from invariant_conditioning "
            f"(max {float((got[key][0] - want[key][0]).abs().max()):.3e})"
        )
    # ...and the dist branch must NOT agree: that is the mechanism under test.
    for key in DIST_IDS:
        assert not torch.allclose(got[key][0], want[key][0], atol=1e-5), (
            f"{key} equals the cylindrical result — the fixture cannot see the "
            "pose-representation change this arm exists to make"
        )


# --------------------------------------------------------------------------- #
# 3. T-45-breaks — the negative control is real
# --------------------------------------------------------------------------- #
def test_45_degree_pre_rotation_breaks_the_average():
    """45 deg is not in C4, so the frame average MUST move. A fixture whose orbit
    average were constant (e.g. an odd pose feature) would pass test 1 while
    proving nothing, so the magnitudes are asserted too."""
    md = _batch(2)
    out_0 = yr.fa_cartesian_conditioning(_build_cond(), md, DEV)
    out_45 = yr.fa_cartesian_conditioning(_build_cond(), _rot(md, 45.0), DEV)
    for key in POSE_KEYS:
        scale = float(out_0[key][0].abs().max())
        assert scale > 1e-2, (
            f"{key}: the orbit average is ~0 ({scale:.3e}) — this fixture is "
            "degenerate and cannot support the C4 invariance claim"
        )
        delta = float((out_45[key][0] - out_0[key][0]).abs().max())
        assert delta > 1e-3, f"{key}: 45 deg moved the average by only {delta:.3e}"
        assert not torch.allclose(out_45[key][0], out_0[key][0], atol=1e-5), key
    # context_audio is yaw-invariant at ANY angle: it never enters the orbit.
    assert torch.allclose(out_45["context_audio"][0], out_0["context_audio"][0],
                          atol=1e-6)


# --------------------------------------------------------------------------- #
# 4. T-audio-single-pass — context_audio runs once, orbit ids run the orbit
# --------------------------------------------------------------------------- #
def test_context_audio_runs_exactly_once_per_call():
    """BatchNorm running-stats protection: |G| passes over the RIR encoder would
    step its statistics four times per training step."""
    cond = _build_cond(train=True)          # train mode: BN bookkeeping is live
    batch = 2
    yr.fa_cartesian_conditioning(cond, _batch(batch), DEV)
    rir = cond.conditioners["context_audio"]
    assert rir.calls == 1, f"context_audio ran {rir.calls} times"
    assert rir.samples == batch
    assert int(rir.bn.num_batches_tracked) == 1


def test_every_pose_conditioner_sees_the_whole_orbit():
    """The exp_21 delta vs fa_invariant, stated in samples: ALL FOUR pose ids see
    ``|G| * B`` samples (fa_invariant's dist ids see ``B``), however grouped."""
    batch, cond = 2, _build_cond()
    yr.fa_cartesian_conditioning(cond, _batch(batch), DEV)
    for key in POSE_KEYS:
        c = cond.conditioners[key]
        assert c.samples == len(C4) * batch, (
            f"{key} saw {c.samples} samples, expected {len(C4)} x {batch}")
        assert c.batch_sizes[0] == batch, "the angle-0 base pass is one plain batch"

    ref = _build_cond()
    yr.invariant_conditioning(ref, _batch(batch), DEV)
    for key in DIST_IDS:
        assert ref.conditioners[key].samples == batch, (
            "fa_invariant must still run the dist branch once — this test would "
            "otherwise not be measuring a difference")


# --------------------------------------------------------------------------- #
# 5. T-dist-orbit-correct — the average is the hand-computed one
# --------------------------------------------------------------------------- #
def test_dist_average_equals_the_hand_computed_orbit_mean():
    """Compute the four rotated scenes explicitly, push each through the SAME
    conditioner, average: the divisor, the terms and the id set are all pinned."""
    md = _batch(2)
    hand = _build_cond()
    sums = {}
    for k, deg in enumerate(C4):
        frames = md if k == 0 else _rot(md, deg)
        o = hand(frames, DEV)
        for key in POSE_KEYS:
            sums[key] = o[key][0] if key not in sums else sums[key] + o[key][0]
    expected = {key: sums[key] / float(len(C4)) for key in POSE_KEYS}

    out = yr.fa_cartesian_conditioning(_build_cond(), md, DEV, C4)
    for key in POSE_KEYS:
        assert torch.allclose(out[key][0], expected[key], atol=1e-5), (
            f"{key}: max {float((out[key][0] - expected[key]).abs().max()):.3e}")


# --------------------------------------------------------------------------- #
# 6. T-batched-equals-loop — batch 32, caps {32, 64} (two different partitions)
# --------------------------------------------------------------------------- #
def _loop_reference(cond, md, angles=C4, width: int = W):
    """The pre-batching execution order, via the module's own reference helper
    (the repo's tests treat ``_orbit_average_loop`` as the specification)."""
    base = cond(md, DEV)
    present = list(POSE_KEYS)
    accum = yr._orbit_average_loop(cond, md, base, present, angles, width, DEV)
    for i in present:
        base[i][0] = accum[i] / float(len(angles))
    return base


@pytest.mark.parametrize("cap,plan", [
    (32, [32, 32, 32, 32]),     # 1 angle per chunk — the exp_21 TRAINING plan
    (64, [32, 64, 32]),         # 2 angles per chunk — the module default
])
def test_batched_orbit_equals_the_loop_reference_at_batch_32(cap, plan):
    batch = 32
    md = _batch(batch)
    cond = _build_cond()
    got = yr.fa_cartesian_conditioning(cond, md, DEV, C4, max_fwd_samples=cap)
    want = _loop_reference(_build_cond(), md)

    for key in POSE_KEYS:
        assert cond.conditioners[key].batch_sizes == plan, (
            f"cap {cap}: {key} executed {cond.conditioners[key].batch_sizes}, "
            f"expected {plan}")
        assert torch.allclose(got[key][0], want[key][0], atol=1e-5), (
            f"cap {cap}: {key} max "
            f"{float((got[key][0] - want[key][0]).abs().max()):.3e}")
    assert torch.allclose(got["context_audio"][0], want["context_audio"][0],
                          atol=1e-6)


def test_the_two_caps_really_execute_different_partitions():
    """Guard for test 6: if both caps produced the same plan, the equivalence
    would be tested once and reported twice."""
    plans = []
    for cap in (32, 64):
        cond = _build_cond()
        yr.fa_cartesian_conditioning(cond, _batch(32), DEV, C4, max_fwd_samples=cap)
        plans.append(_plans(cond))
    assert plans[0] != plans[1], plans


# --------------------------------------------------------------------------- #
# 7. T-mask-preserved (+ the POSE_KEYS admission check)
# --------------------------------------------------------------------------- #
def test_masks_are_the_base_pass_objects():
    """Asserted by OBJECT IDENTITY: a value check would still pass if an orbit
    chunk's (identically-shaped, all-ones) mask replaced the base one."""
    cond = _build_cond()
    out = yr.fa_cartesian_conditioning(cond, _batch(4), DEV)
    for key in ALL_IDS:
        base_ret = cond.conditioners[key].returns[0]
        assert out[key][1] is base_ret[1], f"{key}: mask is not the base pass's"
        assert out[key][1].shape[0] == 4, f"{key}: mask batch is not the base batch"


def test_missing_pose_conditioner_id_is_rejected():
    """Fail-closed: this method's contract is that ALL FOUR pose/geometry ids are
    averaged. Averaging whichever subset happened to exist would ship an arm whose
    treatment nobody declared."""
    for dropped in POSE_KEYS:
        cond = _build_cond(ids=ALL_IDS - {dropped})
        with pytest.raises(ValueError, match="fa_cartesian") as e:
            yr.fa_cartesian_conditioning(cond, _batch(2), DEV)
        assert dropped in str(e.value)


# --------------------------------------------------------------------------- #
# 8. T-depth-required
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_index", (0, 1))
def test_missing_depth_is_rejected(bad_index):
    """Stricter than ``invariant_conditioning``, which short-circuits to the base
    pass when depth is absent: here a missing panorama would average four
    differently-posed views of ONE unrotated map — a degraded orbit, not a no-op."""
    md = _batch(2)
    md[bad_index] = {k: v for k, v in md[bad_index].items() if k != "depth"}
    with pytest.raises(ValueError, match="fa_cartesian") as e:
        yr.fa_cartesian_conditioning(_build_cond(), md, DEV)
    assert "depth" in str(e.value)


@pytest.mark.parametrize("bad_index", (0, 1))
def test_missing_depth_is_rejected_even_for_a_single_angle_orbit(bad_index):
    """r1 review nit 2. The depth contract is UNCONDITIONAL, deliberately: a
    one-angle orbit averages nothing, so a reader could reasonably "simplify" the
    check to the multi-angle path. That would make the arm's declared treatment
    depend on ``len(angles)``, and the fail-closed reading of a sample that
    cannot be rotated is the point. Pinned here so the divergence from
    ``invariant_conditioning`` (which short-circuits) survives a refactor."""
    md = _batch(2)
    md[bad_index] = {k: v for k, v in md[bad_index].items() if k != "depth"}
    with pytest.raises(ValueError, match="fa_cartesian") as e:
        yr.fa_cartesian_conditioning(_build_cond(), md, DEV, (0.0,))
    assert "depth" in str(e.value)


@pytest.mark.parametrize("angles", (C4, (0.0,)))
def test_empty_metadata_is_rejected(angles):
    """r1 review nit 2. An empty batch has no panorama to read the orbit's single
    column-shift rule from, so it is rejected rather than silently returning an
    empty (and structurally invalid) conditioning dict — at any orbit size."""
    with pytest.raises(ValueError, match="fa_cartesian") as e:
        yr.fa_cartesian_conditioning(_build_cond(), [], DEV, angles)
    assert "empty" in str(e.value)


def test_mixed_panorama_widths_are_rejected():
    """One orbit is one column-shift rule: a mixed-width batch would rotate
    different samples by different effective angles."""
    md = [_md(0, width=W), _md(1, width=2 * W)]
    with pytest.raises(ValueError, match="fa_cartesian") as e:
        yr.fa_cartesian_conditioning(_build_cond(), md, DEV)
    assert str(W) in str(e.value) and str(2 * W) in str(e.value)


# --------------------------------------------------------------------------- #
# 9. angle validation (identical contract to invariant_conditioning)
# --------------------------------------------------------------------------- #
def test_angles_must_be_non_empty():
    with pytest.raises(ValueError, match="non-empty"):
        yr.fa_cartesian_conditioning(_build_cond(), _batch(2), DEV, angles=())


def test_first_angle_must_be_the_identity():
    with pytest.raises(ValueError, match="0.0"):
        yr.fa_cartesian_conditioning(_build_cond(), _batch(2), DEV,
                                     angles=(90.0, 180.0, 270.0))


# --------------------------------------------------------------------------- #
# 10. cap validation runs even when there is nothing to chunk
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [True, False, 1.0, 32.0, "32", [32], 0, -1])
def test_invalid_cap_is_rejected_even_for_a_single_angle_orbit(bad):
    """A trivial orbit short-circuits before the partition; an invalid cap must
    still abort, or a typo'd arm could train for days without the declared
    treatment ever being read."""
    with pytest.raises(ValueError, match="max_fwd_samples"):
        yr.fa_cartesian_conditioning(_build_cond(), _batch(2), DEV, (0.0,),
                                     max_fwd_samples=bad)


def test_batch_above_the_cap_is_rejected():
    with pytest.raises(ValueError) as e:
        yr.fa_cartesian_conditioning(_build_cond(), _batch(33), DEV, C4,
                                     max_fwd_samples=32)
    msg = str(e.value)
    assert "33" in msg and "32" in msg
    assert "training.frame_avg_max_fwd_samples" in msg


# --------------------------------------------------------------------------- #
# 11. single-angle orbit -> the base pass, untouched
# --------------------------------------------------------------------------- #
def test_single_angle_orbit_returns_the_base_pass_objects():
    """|G| = 1 has nothing to average, and dividing by one must not even replace
    the tensors (identity is asserted, not equality)."""
    cond = _build_cond()
    out = yr.fa_cartesian_conditioning(cond, _batch(2), DEV, (0.0,))
    assert set(out) == ALL_IDS
    for key in ALL_IDS:
        c = cond.conditioners[key]
        assert c.calls == 1, f"{key} ran {c.calls} times for a 1-angle orbit"
        assert out[key][0] is c.returns[0][0], f"{key}: tensor replaced"
        assert out[key][1] is c.returns[0][1], f"{key}: mask replaced"


# --------------------------------------------------------------------------- #
# 12. the caller's metadata is never mutated
# --------------------------------------------------------------------------- #
def test_caller_metadata_is_not_mutated():
    """The downstream metric callback reads ``source`` / ``depth`` off the SAME
    dicts, and unlike ``fa_invariant`` this method hands them to the conditioner
    directly rather than to a derived copy."""
    md = _batch(3)
    ref = copy.deepcopy(md)
    ids = [{k: id(v) for k, v in m.items()} for m in md]
    yr.fa_cartesian_conditioning(_build_cond(), md, DEV)

    assert len(md) == len(ref)
    for k, (m, m_ref) in enumerate(zip(md, ref)):
        assert set(m) == set(m_ref)
        for key, val in m_ref.items():
            if torch.is_tensor(val):
                assert torch.equal(m[key], val), f"sample {k}: {key} was mutated"
                assert id(m[key]) == ids[k][key], f"sample {k}: {key} was replaced"
            else:
                assert m[key] == val


# --------------------------------------------------------------------------- #
# 13. autograd through the Cartesian orbit (r1 review nit 1)
# --------------------------------------------------------------------------- #
class SharedProjDist(nn.Module):
    """dist_embedder stand-in whose projection is a LEARNABLE parameter, and the
    SAME parameter object for both pose ids.

    :class:`FakeDist` keeps its weights in buffers, so no test above can observe
    the backward pass at all — yet the orbit is a training-path construct, and
    what the arm actually changes is the gradient the dist branch delivers. One
    shared parameter makes a single ``.grad`` accumulate every orbit term of BOTH
    ``source`` and ``context_poses``, so a detached frame, a dropped frame, a
    double-counted frame or a wrong divisor each show up as one mismatch against
    the hand-computed loop. Same non-odd nonlinearity as :class:`FakeDist`, for
    the same reason (an odd feature's C4 average is identically zero)."""

    def __init__(self, w: nn.Parameter, b: nn.Parameter,
                 name: str = "DistEmbedderConditioner"):
        super().__init__()
        self.name = name
        self.w = w
        self.b = b

    def forward(self, x_list, device=DEV):
        x = torch.stack(x_list, dim=0).to(device)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, N, _ = x.shape
        h = x @ self.w + self.b
        out = torch.tanh(h) * torch.sigmoid(h - 0.5) + 0.25 * torch.cos(h)
        return [out.contiguous(), torch.ones(B, N, device=device)]


def _grad_cond(out_dim: int = 6, seed: int = 11):
    """The production id set with a differentiable, weight-tied dist branch."""
    g = torch.Generator().manual_seed(seed)
    w = nn.Parameter(torch.randn(3, out_dim, generator=g))
    b = nn.Parameter(torch.randn(out_dim, generator=g))
    cond = MultiConditioner({
        "source": SharedProjDist(w, b),
        "context_poses": SharedProjDist(w, b),
        "source_vit": FakeGeometry(seed=3),
        "context_poses_vit": FakeGeometry(seed=4),
        "context_audio": FakeRIR(),
    }).eval()
    return cond, w, b


def _scalarise(t: torch.Tensor) -> torch.Tensor:
    """A deterministic NON-uniform weighting of every output element.

    A plain ``.sum()`` would give every element, sample and orbit term the same
    upstream gradient, which is exactly the regime in which a permuted or
    misaligned accumulation is invisible."""
    ramp = torch.linspace(0.5, 1.5, t.numel(), dtype=t.dtype).view_as(t)
    return (t * ramp).sum()


def _dist_loss(out) -> torch.Tensor:
    return sum(_scalarise(out[key][0]) for key in DIST_IDS)


def test_orbit_gradients_equal_the_hand_computed_four_term_reference():
    """Backward through the frame average must equal backward through the
    explicit ``(1/|G|) sum_g f(g.x)`` loop — the derivative of the same
    expression, computed two ways."""
    md = _batch(2)

    cond, w, b = _grad_cond()
    _dist_loss(yr.fa_cartesian_conditioning(cond, md, DEV, C4)).backward()

    ref_cond, ref_w, ref_b = _grad_cond()
    total = None
    for k, deg in enumerate(C4):
        o = ref_cond(md if k == 0 else _rot(md, deg), DEV)
        term = _dist_loss(o)
        total = term if total is None else total + term
    (total / float(len(C4))).backward()

    assert w.grad is not None and b.grad is not None, "no gradient reached the orbit"
    assert float(w.grad.abs().max()) > 1e-4, (
        "the reference gradient is ~0 — this fixture cannot support the claim")
    assert torch.allclose(w.grad, ref_w.grad, atol=1e-6, rtol=1e-5), (
        f"weight grad max {float((w.grad - ref_w.grad).abs().max()):.3e}")
    assert torch.allclose(b.grad, ref_b.grad, atol=1e-6, rtol=1e-5), (
        f"bias grad max {float((b.grad - ref_b.grad).abs().max()):.3e}")


def test_the_nonzero_angle_frames_contribute_to_the_gradient():
    """Guard for the test above: if the three rotated frames were detached (or
    never run), the orbit gradient would be exactly the angle-0 gradient divided
    by ``|G|`` — numerically plausible, and wrong."""
    md = _batch(2)

    full, w_full, _ = _grad_cond()
    _dist_loss(yr.fa_cartesian_conditioning(full, md, DEV, C4)).backward()

    base_only, w_base, _ = _grad_cond()
    _dist_loss(yr.fa_cartesian_conditioning(base_only, md, DEV, (0.0,))).backward()

    scaled = w_base.grad / float(len(C4))
    delta = float((w_full.grad - scaled).abs().max())
    assert not torch.allclose(w_full.grad, scaled, atol=1e-5), (
        "the orbit gradient is the angle-0 term alone: the nonzero-angle frames "
        "contribute nothing")
    assert delta > 1e-3, f"nonzero-angle frames moved the gradient by only {delta:.3e}"
