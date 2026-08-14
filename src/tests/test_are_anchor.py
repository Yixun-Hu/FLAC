"""exp_16 (are_port) Phase 1 — the ARE anchor pipeline, tested against ANALYTIC values.

Seat: Opus 5 Coder (SOP §Roles); plan `worklog/worklog_yixun/exp_16_are_port_claude/
plan_are_port.md` §2. Design source: rir2rir `exp_15_anchor` `plan_anchor.md`
§§4.a–4.d (re-implemented against FLAC's stack, not copied).

WHAT THE ANCHOR IS. For one IR path with listener-frame source ``s`` and
``r = ||s||``, the anchor is

    skel(r)[n] = g * hann(n - t*) * sinc(n - t*) / ||hann*sinc||_2 ,  |n - t*| <= H
    t*         = r/343 * fs + delta_hat            (fs from the CONFIG, never a literal)
    g          = A_g / max(r, 0.05)
    A(r)       = Pi_early( Enc_mean(skel) - Enc_mean(0) )      # frames >= 3 zeroed
    A(p)       = 1[LOS(p)] * A(r)                              # depth-panorama gate

and the training target becomes ``z - lambda*A(p)`` with ``+lambda*A_query`` added
back before the decode.

THE CONTRACTS PINNED HERE

1. **Kernel**: exact unit ell-2, so ``||skel||_2 == g`` EXACTLY — an amplitude
   convention that cannot be modulated by the sub-sample phase (the design
   source measured a 3.92 dB peak swing across phase against 1.3e-2 in energy).
2. **Sub-sample precision**: the realised delay is read off the kernel's DFT
   PHASE SLOPE — an analytic measurement, not a re-statement of the formula —
   and must equal ``t*`` to <0.02 samples at every fractional phase.
3. **Silence bias**: ``A`` is ``Enc(skel) - Enc(0)``; the anchor of silence is
   EXACTLY the zero latent. Checked on a stub encoder AND, when the real VAE
   weights are present, on the real frozen encoder.
4. **Frame truncation**: frames >= ``early_frames`` are exactly zero.
5. **LOS gate**: an occluded source bearing yields exactly the zero anchor; the
   bearing->pixel map round-trips against AR's own equirectangular convention
   (`AR_md.convert_equirect_to_camera_coord`).
6. **Determinism**: two calls are BIT-equal, and a cache hit is bit-equal to the
   cache miss that produced it (no RNG anywhere on the path — the anchor uses the
   VAE's MEAN, never ``vae_sample``, which would both randomise the anchor and
   displace the training RNG stream).
7. **Worst case**: AR's largest source-receiver distance must land the direct
   peak (plus the kernel's half-width) inside the kept frame window.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import are_anchor as ar

_REPO = Path(__file__).resolve().parents[2]
VAE_WEIGHTS = _REPO / "weights/FLAC/VAE.safetensors"
CALIB_SCRIPT = _REPO / "worklog/worklog_yixun/exp_16_are_port_claude/calibrate_delta.py"

FS = 22050
SAMPLE_SIZE = 10240
HOP = 1024
A_G = 0.5
H_PANO, W_PANO = 16, 32          # tiny panorama for the gate tests


def _cfg(**over):
    base = dict(sample_rate=FS, sample_size=SAMPLE_SIZE, hop=HOP,
                delta_hat=0.0, a_g=A_G)
    base.update(over)
    return ar.AnchorConfig(**base)


# --------------------------------------------------------------------------- #
# stub encoders (no DINOv3, no VAE weights)
# --------------------------------------------------------------------------- #
class _AffineEncoder:
    """``Enc(x) = pool(x) @ W + b`` — affine, so ``Enc(x) - Enc(0)`` is exactly
    the linear part. Records every waveform it is handed, at its true batch size,
    which is how the tests observe that misses are encoded ONE AT A TIME."""

    def __init__(self, channels=32, latent_len=SAMPLE_SIZE // HOP, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(channels, latent_len, generator=g)
        self.b = torch.randn(channels, latent_len, generator=g) * 3.0
        self.calls = []

    def __call__(self, x):                        # x: [B, 1, T]
        self.calls.append(tuple(x.shape))
        pooled = x.reshape(x.shape[0], 1, -1, HOP).mean(-1)          # [B,1,L]
        return pooled * self.W[None] + self.b[None]                  # [B,C,L]


class _NonEarlyEncoder(_AffineEncoder):
    """Affine, but deliberately spreads energy into EVERY latent frame, so the
    truncation test is measuring the truncation and not the encoder's own decay."""

    def __call__(self, x):
        self.calls.append(tuple(x.shape))
        pooled = x.reshape(x.shape[0], 1, -1, HOP).mean(-1)
        broadcast = pooled.mean(-1, keepdim=True).expand_as(pooled)
        return broadcast * self.W[None] + self.b[None]


def _bank(encoder=None, cfg=None):
    return ar.AnchorBank(encoder or _AffineEncoder(), cfg or _cfg())


# --------------------------------------------------------------------------- #
# 1. kernel: unit ell-2 at every fractional phase
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("frac", [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
def test_kernel_is_exactly_unit_l2(frac):
    offsets = torch.arange(-32, 33, dtype=torch.float64) - frac
    ker = ar.hann_sinc_kernel(offsets, half_width=32)
    assert ker.shape == offsets.shape
    assert float(torch.linalg.vector_norm(ker)) == pytest.approx(1.0, abs=1e-12)


def test_kernel_vanishes_at_the_window_edge():
    """``hann(x) = 0.5(1 + cos(pi x/(H+1)))`` is zero at |x| = H+1, so the kernel
    tapers to zero rather than truncating a sinc mid-lobe."""
    ker = ar.hann_sinc_kernel(torch.tensor([-33.0, 33.0]), half_width=32)
    assert torch.allclose(ker, torch.zeros(2, dtype=torch.float64), atol=1e-12)


# --------------------------------------------------------------------------- #
# 2. skeleton: analytic peak position, amplitude, and SUB-SAMPLE delay
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("frac", [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
def test_skeleton_energy_equals_the_amplitude_law_exactly(frac):
    """``||skel||_2 == A_g / r`` for EVERY fractional phase — the property the
    unit-ell-2 convention exists to buy."""
    cfg = _cfg()
    r = (1000.0 + frac) * ar.C_SOUND / cfg.sample_rate          # t* = 1000 + frac
    skel = ar.direct_path_skeleton(r, cfg)
    assert skel.shape == (1, cfg.sample_size)
    assert float(torch.linalg.vector_norm(skel)) == pytest.approx(A_G / r, rel=1e-9)


@pytest.mark.parametrize("frac,want_peak", [(0.0, 1000), (0.1, 1000), (0.4, 1000),
                                            (0.6, 1001), (0.75, 1001), (0.9, 1001)])
def test_skeleton_peak_lands_on_the_nearest_sample(frac, want_peak):
    cfg = _cfg()
    r = (1000.0 + frac) * ar.C_SOUND / cfg.sample_rate
    skel = ar.direct_path_skeleton(r, cfg)[0]
    assert int(torch.argmax(skel.abs())) == want_peak


def test_half_sample_phase_splits_the_peak_symmetrically():
    """At frac = 0.5 the two samples straddling ``t*`` are equal — the signature
    of a delay realised BETWEEN samples rather than quantised onto one."""
    cfg = _cfg()
    r = 1000.5 * ar.C_SOUND / cfg.sample_rate
    skel = ar.direct_path_skeleton(r, cfg)[0]
    assert float(skel[1000]) == pytest.approx(float(skel[1001]), rel=1e-9)
    assert float(skel[1000]) > float(skel[999])


@pytest.mark.parametrize("frac", [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
def test_realised_delay_matches_t_star_to_a_fiftieth_of_a_sample(frac):
    """The DFT phase slope of the skeleton IS its group delay. This measures the
    realised sub-sample delay analytically instead of re-asserting the formula
    that produced it."""
    cfg = _cfg()
    t_star = 1000.0 + frac
    r = t_star * ar.C_SOUND / cfg.sample_rate
    skel = ar.direct_path_skeleton(r, cfg)[0].double()
    spec = torch.fft.rfft(skel)
    n = skel.numel()
    bins = torch.arange(1, 201, dtype=torch.float64)
    phase = torch.from_numpy(np.unwrap(torch.angle(spec[1:201]).numpy()))
    # least squares through the origin: phase = -2*pi*t*/N * k
    slope = float((phase * bins).sum() / (bins * bins).sum())
    assert -slope * n / (2 * math.pi) == pytest.approx(t_star, abs=0.02)


def test_amplitude_uses_the_distance_guard_at_tiny_r():
    cfg = _cfg(dist_guard=0.05)
    skel = ar.direct_path_skeleton(0.01, cfg)
    assert float(torch.linalg.vector_norm(skel)) == pytest.approx(A_G / 0.05, rel=1e-9)


def test_delta_hat_shifts_the_peak_by_exactly_its_own_samples():
    plain = ar.direct_path_skeleton(1000.0 * ar.C_SOUND / FS, _cfg(delta_hat=0.0))[0]
    shifted = ar.direct_path_skeleton(1000.0 * ar.C_SOUND / FS, _cfg(delta_hat=7.0))[0]
    assert int(torch.argmax(plain.abs())) == 1000
    assert int(torch.argmax(shifted.abs())) == 1007
    assert torch.allclose(plain[900:1100], shifted[907:1107], atol=1e-12)


def test_sample_rate_comes_from_the_config_not_a_literal():
    """Halving fs halves ``t*`` for the same distance. A hardcoded 22050 would
    keep the peak where it was."""
    r = 5.0
    a = ar.direct_path_skeleton(r, _cfg(sample_rate=FS))[0]
    b = ar.direct_path_skeleton(r, _cfg(sample_rate=FS // 2))[0]
    assert int(torch.argmax(a.abs())) == pytest.approx(
        2 * int(torch.argmax(b.abs())), abs=1)


def test_skeleton_out_of_range_is_a_hard_error():
    cfg = _cfg()
    too_far = (cfg.sample_size + 1) * ar.C_SOUND / cfg.sample_rate
    with pytest.raises(ValueError, match="outside"):
        ar.direct_path_skeleton(too_far, cfg)
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            ar.direct_path_skeleton(bad, cfg)


# --------------------------------------------------------------------------- #
# 3. worst-case frame window (plan §2 / deliverable 1)
# --------------------------------------------------------------------------- #
def test_ar_worst_case_distance_lands_inside_the_kept_frames():
    cfg = _cfg()
    rep = ar.compute_worst_case_frames(cfg)
    assert rep["max_distance"] == pytest.approx(ar.AR_MAX_SOURCE_RECEIVER_DISTANCE)
    assert rep["required_frames"] <= cfg.early_frames
    assert rep["ok"] is True
    # the direct peak of the longest AR path, in samples and in frames
    # the loader's largest RandomTimeShift draw is part of the worst case: the
    # augmentation moves the target later, and the anchor moves with it
    assert rep["max_time_shift"] == 10
    assert rep["t_star"] == pytest.approx(
        ar.AR_MAX_SOURCE_RECEIVER_DISTANCE / ar.C_SOUND * FS + 10, rel=1e-9)
    assert rep["peak_frame"] == 1


def test_worst_case_reports_not_ok_when_the_window_is_too_narrow():
    rep = ar.compute_worst_case_frames(_cfg(early_frames=1))
    assert rep["ok"] is False
    assert rep["required_frames"] == 2
    with pytest.raises(ValueError, match="early_frames"):
        ar.assert_worst_case_frames(_cfg(early_frames=1))


def test_worst_case_accounts_for_the_kernel_half_width():
    """A distance whose peak sits just inside frame 2 must still fail when the
    kernel's tail crosses into frame 3."""
    cfg = _cfg(kernel_half_width=32)
    just_inside = (3 * HOP - 16) * ar.C_SOUND / FS
    rep = ar.compute_worst_case_frames(cfg, max_distance=just_inside)
    assert rep["peak_frame"] == 2
    assert rep["required_frames"] == 4
    assert rep["ok"] is False


# --------------------------------------------------------------------------- #
# 4. silence bias + frame truncation
# --------------------------------------------------------------------------- #
def test_anchor_of_silence_is_exactly_zero():
    """``Enc(0) - Enc(0) == 0``: the bias is removed by construction, so a path
    with no skeleton contributes nothing at all."""
    bank = _bank()
    bias = bank.silence_bias(torch.device("cpu"))
    assert bias.shape == (32, SAMPLE_SIZE // HOP)
    assert float(bias.abs().max()) > 0.0, "the stub encoder must have a real bias"
    zero = bank.anchor_from_waveform(torch.zeros(1, SAMPLE_SIZE), torch.device("cpu"))
    assert torch.equal(zero, torch.zeros_like(zero))


def test_anchor_is_the_bias_corrected_encode():
    enc = _AffineEncoder()
    bank = _bank(enc)
    r = 3.0
    got = bank.anchor_for_distance(r, torch.device("cpu"))
    skel = ar.direct_path_skeleton(r, bank.cfg)
    want = enc(skel[None].float()) - enc(torch.zeros(1, 1, SAMPLE_SIZE))
    want[:, :, bank.cfg.early_frames:] = 0.0
    assert torch.equal(got, want[0])


def test_frames_beyond_the_window_are_exactly_zero():
    bank = _bank(_NonEarlyEncoder())
    a = bank.anchor_for_distance(3.0, torch.device("cpu"))
    assert torch.equal(a[:, 3:], torch.zeros_like(a[:, 3:]))
    assert float(a[:, :3].abs().max()) > 0.0


def test_early_frames_is_config_driven():
    bank = _bank(_NonEarlyEncoder(), _cfg(early_frames=5))
    a = bank.anchor_for_distance(3.0, torch.device("cpu"))
    assert float(a[:, 3:5].abs().max()) > 0.0
    assert torch.equal(a[:, 5:], torch.zeros_like(a[:, 5:]))


def test_misses_are_encoded_one_at_a_time():
    """Batch composition perturbs a conv encoder at the 1e-6 level, so every
    anchor is encoded at B=1 and is therefore a pure function of ``r``."""
    enc = _AffineEncoder()
    bank = _bank(enc)
    dev = torch.device("cpu")
    for r in (1.0, 2.0, 3.0):
        bank.anchor_for_distance(r, dev)
    assert all(shape[0] == 1 for shape in enc.calls), enc.calls


# --------------------------------------------------------------------------- #
# 5. the LOS gate
# --------------------------------------------------------------------------- #
def _direction(i, j, img_h=H_PANO, img_w=W_PANO):
    """The unit direction AR's own equirectangular convention assigns to pixel
    (i, j) — see ``AR_md.convert_equirect_to_camera_coord``."""
    theta = (j + 0.5) * 2.0 * math.pi / img_w - math.pi
    phi = (i + 0.5) * math.pi / img_h - math.pi / 2
    return torch.tensor([math.cos(phi) * math.cos(theta),
                         math.cos(phi) * math.sin(theta),
                         -math.sin(phi)], dtype=torch.float32)


def _depth_pano(value, img_h=H_PANO, img_w=W_PANO):
    """A [3, H, W] point cloud in AR's layout whose every pixel has range ``value``."""
    pano = torch.stack([_direction(i, j, img_h, img_w) * value
                        for i in range(img_h) for j in range(img_w)])
    return pano.reshape(img_h, img_w, 3).permute(2, 0, 1).contiguous()


def test_bearing_pixel_roundtrips_on_every_pixel():
    for i in range(H_PANO):
        for j in range(W_PANO):
            s = _direction(i, j) * 4.0
            assert ar.source_bearing_pixel(s, H_PANO, W_PANO) == (i, j)


def test_bearing_pixel_wraps_the_seam_and_clamps_the_poles():
    # a bearing exactly on the -pi seam maps to column 0, never to -1 or W
    i, j = ar.source_bearing_pixel(torch.tensor([-1.0, 0.0, 0.0]), H_PANO, W_PANO)
    assert 0 <= j < W_PANO
    for z in (+1.0, -1.0):                                   # straight up / down
        i, j = ar.source_bearing_pixel(torch.tensor([0.0, 0.0, z]), H_PANO, W_PANO)
        assert 0 <= i < H_PANO and 0 <= j < W_PANO


def test_los_gate_open_when_the_room_is_further_than_the_source():
    depth = _depth_pano(10.0)
    assert ar.line_of_sight(depth, _direction(5, 7) * 4.0, _cfg()) is True


def test_los_gate_closed_when_the_source_bearing_is_occluded():
    depth = _depth_pano(10.0)
    s = _direction(5, 7) * 4.0
    i, j = ar.source_bearing_pixel(s, H_PANO, W_PANO)
    depth[:, i, j] *= 0.1                                    # a wall at 1 m
    assert ar.line_of_sight(depth, s, _cfg()) is False


def test_los_gate_uses_a_three_by_three_minimum():
    """A one-pixel bearing quantisation error must not open the gate: the
    neighbourhood minimum is what makes the error direction safe."""
    depth = _depth_pano(10.0)
    s = _direction(5, 7) * 4.0
    depth[:, 6, 8] *= 0.1                                    # diagonal neighbour
    assert ar.line_of_sight(depth, s, _cfg()) is False


def test_los_threshold_is_config_driven():
    depth = _depth_pano(10.0)
    s = _direction(5, 7) * 10.4                              # depth = 0.96 * r
    assert ar.line_of_sight(depth, s, _cfg(los_threshold=0.95)) is True
    assert ar.line_of_sight(depth, s, _cfg(los_threshold=0.99)) is False


def test_occluded_sample_gets_exactly_the_zero_anchor():
    bank = _bank()
    dev = torch.device("cpu")
    depth = _depth_pano(10.0)
    s = _direction(5, 7) * 4.0
    open_anchor = bank.anchor_for_sample(s, depth, dev)
    assert float(open_anchor.abs().max()) > 0.0

    i, j = ar.source_bearing_pixel(s, H_PANO, W_PANO)
    depth[:, i, j] *= 0.1
    gated = bank.anchor_for_sample(s, depth, dev)
    assert torch.equal(gated, torch.zeros_like(gated))


# --------------------------------------------------------------------------- #
# 6. determinism (no RNG on the anchor path, cache-transparent)
# --------------------------------------------------------------------------- #
def test_two_calls_are_bit_equal():
    bank = _bank()
    dev = torch.device("cpu")
    a = bank.anchor_for_distance(2.5, dev)
    b = bank.anchor_for_distance(2.5, dev)
    assert torch.equal(a, b)


def test_a_cache_hit_is_bit_equal_to_its_miss():
    bank = _bank()
    dev = torch.device("cpu")
    miss = bank.anchor_for_distance(2.5, dev).clone()
    assert bank.cache_stats()["misses"] == 1
    hit = bank.anchor_for_distance(2.5, dev)
    assert bank.cache_stats()["hits"] == 1
    assert torch.equal(miss, hit)
    bank.clear_cache()
    again = bank.anchor_for_distance(2.5, dev)
    assert torch.equal(miss, again)


def test_encode_mean_applies_the_pretransform_scale():
    """``AutoencoderPretransform.encode`` divides by ``scale`` and ``decode``
    multiplies by it. The anchor lives in the same latent space as ``z``, so it
    must carry the same convention. The repo's configs happen to leave scale at
    1.0; that must be honoured rather than assumed."""
    from src.models.bottleneck import VAEBottleneck

    class _Model:
        bottleneck = VAEBottleneck()

        def encode(self, x, skip_bottleneck=False):
            b = x.shape[0]
            pooled = x.reshape(b, 1, -1, HOP).mean(-1)
            return torch.cat([pooled.expand(b, 32, SAMPLE_SIZE // HOP),
                              torch.zeros(b, 32, SAMPLE_SIZE // HOP)], dim=1)

    class _Pre:
        model = _Model()
        scale = 4.0
        downsampling_ratio = HOP

    enc = ar.make_encode_mean(_Pre())
    x = torch.full((1, 1, SAMPLE_SIZE), 8.0)
    assert torch.allclose(enc(x), torch.full((1, 32, SAMPLE_SIZE // HOP), 2.0))


def test_the_anchor_path_consumes_no_global_rng():
    """The VAE bottleneck's ``vae_sample`` draws ``randn_like``. Routing the
    anchor through it would randomise the anchor AND displace the training noise
    stream, so the anchor must use the encoder MEAN and touch no generator."""
    bank = _bank()
    dev = torch.device("cpu")
    torch.manual_seed(1234)
    before = torch.randn(4)
    torch.manual_seed(1234)
    bank.anchor_for_distance(2.5, dev)
    bank.anchor_for_sample(_direction(3, 3) * 2.0, _depth_pano(10.0), dev)
    after = torch.randn(4)
    assert torch.equal(before, after)


def test_cache_is_bounded():
    bank = _bank(cfg=_cfg(cache_size=4))
    dev = torch.device("cpu")
    for k in range(10):
        bank.anchor_for_distance(1.0 + k, dev)
    assert bank.cache_stats()["size"] == 4


# --------------------------------------------------------------------------- #
# 7. batch entry point
# --------------------------------------------------------------------------- #
def _md(r, occluded=False, img_h=H_PANO, img_w=W_PANO, time_shift=0):
    depth = _depth_pano(10.0, img_h, img_w)
    s = _direction(4, 9, img_h, img_w) * r
    if occluded:
        i, j = ar.source_bearing_pixel(s, img_h, img_w)
        depth[:, i, j] *= 0.01
    return {"source": s, "depth": depth, "time_shift": time_shift}


def test_compute_are_anchors_shapes_and_gating():
    bank = _bank()
    md = [_md(2.0), _md(3.0, occluded=True), _md(4.0)]
    out = ar.compute_are_anchors(bank, md, torch.device("cpu"))
    assert out.shape == (3, 32, SAMPLE_SIZE // HOP)
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert float(out[0].abs().max()) > 0.0


def test_compute_are_anchors_requires_source_depth_and_time_shift():
    bank = _bank()
    md = _md(2.0)
    with pytest.raises(ValueError, match="source"):
        ar.compute_are_anchors(bank, [{k: v for k, v in md.items() if k != "source"}],
                               torch.device("cpu"))
    with pytest.raises(ValueError, match="depth"):
        ar.compute_are_anchors(bank, [{k: v for k, v in md.items() if k != "depth"}],
                               torch.device("cpu"))
    # fail-closed on the augmentation draw: assuming 0 would be systematically
    # wrong on half the AR training set
    with pytest.raises(ValueError, match="time_shift"):
        ar.compute_are_anchors(bank, [{k: v for k, v in md.items() if k != "time_shift"}],
                               torch.device("cpu"))


def test_compute_are_anchors_applies_the_loader_time_shift():
    """``RandomTimeShift`` moves the TARGET forward while the geometry stays put;
    the anchor must move with the target, not with the geometry."""
    bank = _bank()
    dev = torch.device("cpu")
    # a distance whose t* sits just inside frame 0, so a 7-sample shift really
    # moves energy across the latent frame boundary (the stub encoder pools per
    # frame and would otherwise be blind to an intra-frame move)
    r = 1020.0 * ar.C_SOUND / FS
    room = _depth_pano(40.0)                       # far enough to keep the gate open
    src = _direction(4, 9) * r

    def md(shift):
        return {"source": src, "depth": room, "time_shift": shift}

    plain = ar.compute_are_anchors(bank, [md(0)], dev)
    shifted = ar.compute_are_anchors(bank, [md(7)], dev)
    assert not torch.equal(plain, shifted)
    # ...and it is exactly the anchor of a skeleton placed 7 samples later
    direct = bank.anchor_for_distance(r, dev, time_shift=7)
    assert torch.equal(shifted[0], direct)


def test_the_cache_is_keyed_on_the_shift_as_well_as_the_distance():
    bank = _bank()
    dev = torch.device("cpu")
    bank.anchor_for_distance(4.0, dev, time_shift=0)
    misses = bank.cache_stats()["misses"]
    bank.anchor_for_distance(4.0, dev, time_shift=3)
    assert bank.cache_stats()["misses"] == misses + 1, (
        "a different time shift is a different anchor and must not hit the cache")


def test_time_shift_moves_the_skeleton_peak_by_exactly_its_own_samples():
    cfg = _cfg()
    r = 1000.0 * ar.C_SOUND / cfg.sample_rate
    plain = ar.direct_path_skeleton(r, cfg)[0]
    shifted = ar.direct_path_skeleton(r, cfg, shift_samples=4)[0]
    assert int(torch.argmax(plain.abs())) == 1000
    assert int(torch.argmax(shifted.abs())) == 1004
    assert torch.allclose(plain[900:1100], shifted[904:1104], atol=1e-12)


def test_occluded_samples_are_never_encoded():
    """The gate runs before the encoder, so an NLOS sample costs nine depth
    lookups and no VAE forward at all."""
    enc = _AffineEncoder()
    bank = _bank(enc)
    ar.compute_are_anchors(bank, [_md(2.0, occluded=True)] * 4, torch.device("cpu"))
    # exactly one encode: the silence bias. No skeleton was encoded.
    assert len(enc.calls) == 1


def test_compute_are_anchors_accepts_the_vit_shaped_source():
    """``AR_md`` also publishes ``source_vit`` as ``[1, 3]``; a ``[1, 3]``
    ``source`` must not be read as three separate coordinates."""
    bank = _bank()
    md = _md(2.0)
    flat = ar.compute_are_anchors(bank, [md], torch.device("cpu"))
    nested = ar.compute_are_anchors(
        bank, [{"source": md["source"][None], "depth": md["depth"],
                "time_shift": md["time_shift"]}],
        torch.device("cpu"))
    assert torch.equal(flat, nested)


# --------------------------------------------------------------------------- #
# 8. the real frozen VAE (skipped when the weights are absent)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not VAE_WEIGHTS.is_file(), reason="weights/FLAC/VAE.safetensors absent")
def test_real_vae_anchor_is_bias_free_deterministic_and_early():
    from src.models.factory import create_pretransform_from_config
    from src.models.utils import load_ckpt_state_dict

    cfg_path = _REPO / "worklog/worklog_yixun/exp_07_fa_scratch_claude/FLAC_AR_BVp1.json"
    model_config = json.loads(cfg_path.read_text())
    pre = create_pretransform_from_config(model_config["model"]["pretransform"],
                                          model_config["sample_rate"])
    pre.load_state_dict(load_ckpt_state_dict(str(VAE_WEIGHTS)))
    pre.eval()

    bank = ar.build_anchor_bank(pre, model_config["sample_rate"],
                                model_config["sample_size"],
                                {"delta_hat": 0.0, "a_g": A_G})
    dev = torch.device("cpu")

    # (a) the silence bias is real and large — the correction is not cosmetic
    bias = bank.silence_bias(dev)
    assert float(torch.linalg.matrix_norm(bias)) > 1.0

    # (b) the anchor of silence is exactly zero
    zero = bank.anchor_from_waveform(torch.zeros(1, model_config["sample_size"]), dev)
    assert torch.equal(zero, torch.zeros_like(zero))

    # (c) bit-equal repeats, and no RNG consumed
    torch.manual_seed(7)
    ref = torch.randn(3)
    torch.manual_seed(7)
    a1 = bank.anchor_for_distance(3.0, dev)
    assert torch.equal(ref, torch.randn(3))
    bank.clear_cache()
    assert torch.equal(a1, bank.anchor_for_distance(3.0, dev))

    # (d) the corrected anchor is naturally early-supported: after truncation it
    #     still carries essentially all of the untruncated anchor's energy
    full = bank.anchor_from_waveform(ar.direct_path_skeleton(3.0, bank.cfg), dev,
                                     truncate=False)
    kept = float(torch.linalg.matrix_norm(full[:, :bank.cfg.early_frames]) ** 2)
    total = float(torch.linalg.matrix_norm(full) ** 2)
    assert kept / total > 0.90


# --------------------------------------------------------------------------- #
# 9. the delta-hat calibration script, on a synthetic fixture
# --------------------------------------------------------------------------- #
def _write_fixture(root, delay_samples, n=6, a_g=0.4):
    """A miniature AR tree with a KNOWN answer.

    Each path's distance is chosen so that ``r/343*fs`` is an exact integer
    number of samples, and its impulse is placed exactly ``delay_samples`` later.
    The calibration's ``delta_hat_raw`` must then recover ``delay_samples`` with
    no rounding residual to hide behind.
    """
    import torchaudio

    ir_root = root / "AcousticRooms" / "single_channel_ir_1" / "Toy" / "Toy_idx_0"
    md_root = root / "AcousticRooms" / "metadata" / "Toy" / "Toy_idx_0"
    ir_root.mkdir(parents=True)
    md_root.mkdir(parents=True)
    rel = []
    for k in range(n):
        n_analytic = 200 + 137 * k
        r = n_analytic * ar.C_SOUND / FS
        (md_root / f"S00{k}_R000.json").write_text(json.dumps(
            {"src_loc": [r, 0.0, 0.0], "rec_loc": [0.0, 0.0, 0.0]}))
        wav = torch.zeros(1, SAMPLE_SIZE)
        wav[0, n_analytic + delay_samples] = a_g / r
        name = f"S00{k}_R000_hybrid_IR.wav"
        torchaudio.save(str(ir_root / name), wav, FS)
        rel.append(f"Toy/Toy_idx_0/{name}")

    # the nested AR split shape (scene -> scene_id -> filenames), so the test
    # exercises the same branch production runs on
    split = root / "train.json"
    split.write_text(json.dumps({"Toy": {"Toy_idx_0": [p.split("/")[-1] for p in rel]}}))
    return split


@pytest.mark.parametrize("delay", [0, 5])
def test_calibration_script_recovers_a_known_offset(tmp_path, delay):
    split = _write_fixture(tmp_path, delay)
    out = tmp_path / "calibration.json"
    rc = subprocess.run(
        [sys.executable, str(CALIB_SCRIPT),
         "--audio-root", str(tmp_path / "AcousticRooms"),
         "--folder-name", "single_channel_ir_1",
         "--split-json", str(split),
         "--sample-rate", str(FS), "--sample-size", str(SAMPLE_SIZE),
         "--n-paths", "6", "--no-los-filter", "--out", str(out)],
        cwd=str(_REPO), capture_output=True, text=True)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    rec = json.loads(out.read_text())
    assert rec["delta_hat_raw"] == pytest.approx(float(delay), abs=1e-3)
    # R1: an offset under half a sample is snapped to exactly 0
    assert rec["delta_hat"] == pytest.approx(0.0 if delay == 0 else float(delay), abs=1e-3)
    assert rec["n_paths"] == 6
    assert rec["a_g"] == pytest.approx(0.4, rel=5e-3)
    assert rec["escalate_r2"] is False and rec["escalate_r4"] is False


def test_calibration_snaps_a_sub_half_sample_offset_to_zero(tmp_path):
    """Rule R1 of the design source: an offset under half a sample is rounding,
    not a fit, and is recorded as exactly zero."""
    split = _write_fixture(tmp_path, 0)
    out = tmp_path / "calibration.json"
    rc = subprocess.run(
        [sys.executable, str(CALIB_SCRIPT),
         "--audio-root", str(tmp_path / "AcousticRooms"),
         "--folder-name", "single_channel_ir_1",
         "--split-json", str(split),
         "--sample-rate", str(FS), "--sample-size", str(SAMPLE_SIZE),
         "--n-paths", "6", "--no-los-filter", "--out", str(out)],
        cwd=str(_REPO), capture_output=True, text=True)
    assert rc.returncode == 0, rc.stdout + rc.stderr
    rec = json.loads(out.read_text())
    assert rec["delta_hat"] == 0.0
    assert abs(rec["delta_hat_raw"]) < 0.5
    assert rec["r1_fired"] is True


def test_calibration_is_deterministic(tmp_path):
    split = _write_fixture(tmp_path, 3)
    outs = []
    for k in range(2):
        out = tmp_path / f"c{k}.json"
        rc = subprocess.run(
            [sys.executable, str(CALIB_SCRIPT),
             "--audio-root", str(tmp_path / "AcousticRooms"),
             "--folder-name", "single_channel_ir_1",
             "--split-json", str(split),
             "--sample-rate", str(FS), "--sample-size", str(SAMPLE_SIZE),
             "--n-paths", "4", "--no-los-filter", "--out", str(out)],
            cwd=str(_REPO), capture_output=True, text=True)
        assert rc.returncode == 0, rc.stdout + rc.stderr
        rec = json.loads(out.read_text())
        rec.pop("created", None)
        outs.append(rec)
    assert outs[0] == outs[1]


# --------------------------------------------------------------------------- #
# 10. the time-shift publication, through the REAL augmentation and the REAL
#     dataset (r1 code review, finding 5)
# --------------------------------------------------------------------------- #
def test_random_time_shift_publishes_the_displacement_it_applied():
    """The module contract: ``last_shift`` is the number of samples the waveform
    actually moved, not a nominal parameter."""
    from src.data.utils import RandomTimeShift

    shifter = RandomTimeShift(max_shift=10, p=1.0)
    signal = torch.arange(1, 65, dtype=torch.float32).reshape(1, 64)
    torch.manual_seed(0)
    out = shifter(signal)
    k = shifter.last_shift
    assert 1 <= k <= 10
    assert torch.equal(out[:, k:], signal[:, :-k]), "published shift != applied shift"
    assert torch.equal(out[:, :k], torch.zeros(1, k))


def test_random_time_shift_publishes_zero_when_it_declines():
    from src.data.utils import RandomTimeShift

    shifter = RandomTimeShift(max_shift=10, p=0.0)
    signal = torch.arange(1, 33, dtype=torch.float32).reshape(1, 32)
    out = shifter(signal)
    assert shifter.last_shift == 0
    assert torch.equal(out, signal)


def test_publishing_the_shift_consumes_no_extra_rng():
    """Recording a draw that already happened must not BE a draw: the ARE arm and
    its control have to walk the same python RNG stream."""
    import random

    from src.data.utils import RandomTimeShift

    shifter = RandomTimeShift(max_shift=10, p=0.5)
    signal = torch.ones(1, 64)

    random.seed(7)
    for _ in range(50):
        shifter(signal)
    after_forward = random.getstate()

    # the same number of draws, made by hand: one random(), and a randint() only
    # when that random() cleared the probability gate
    random.seed(7)
    for _ in range(50):
        if not (random.random() > 0.5):
            random.randint(1, 10)
    assert random.getstate() == after_forward


def _tiny_ar_tree(root, n=4):
    """A miniature AR tree the real ``SampleDataset`` can walk."""
    import torchaudio

    ir_root = root / "AcousticRooms" / "single_channel_ir_1" / "Toy" / "Toy_idx_0"
    ir_root.mkdir(parents=True)
    names = []
    for k in range(n):
        wav = torch.zeros(1, SAMPLE_SIZE)
        wav[0, 500 + k] = 0.5                      # a single locatable impulse
        name = f"S00{k}_R000_hybrid_IR.wav"
        torchaudio.save(str(ir_root / name), wav, FS)
        names.append(name)
    split = root / "train.json"
    split.write_text(json.dumps({"Toy": {"Toy_idx_0": names}}))
    return split


def _dataset(root, split, augs):
    from src.data.dataset import LocalDatasetConfig, SampleDataset

    cfg = LocalDatasetConfig(
        id="Toy", path=str(root / "AcousticRooms"), json_file_path=str(split),
        folder_name="single_channel_ir_1", conditioning={})
    return SampleDataset([cfg], sample_size=SAMPLE_SIZE, sample_rate=FS,
                         random_crop=False, force_channels="mono", augs=augs)


def test_dataset_publishes_the_shift_that_actually_displaced_the_waveform(tmp_path):
    """Integration, through the real loader and the real augmentation: the
    metadata value must equal the displacement the returned audio really carries.

    Round 1's tests supplied ``time_shift`` by hand, so nothing connected the
    number to the waveform (r1 review finding 5).
    """
    import random

    split = _tiny_ar_tree(tmp_path)
    ds = _dataset(tmp_path, split, augs=True)
    assert ds._time_shifter is not None

    seen_shifted = False
    for seed in range(40):
        random.seed(seed)
        audio, info = ds[0]
        shift = info["time_shift"]
        peak = int(torch.argmax(audio.abs()))
        assert peak == 500 + shift, (
            f"metadata says time_shift={shift} but the impulse moved to {peak}")
        seen_shifted = seen_shifted or shift > 0
    assert seen_shifted, "no seed in the sweep exercised a non-zero shift"


def test_dataset_publishes_zero_when_there_is_no_augmentation(tmp_path):
    split = _tiny_ar_tree(tmp_path)
    ds = _dataset(tmp_path, split, augs=False)
    assert ds._time_shifter is None
    _, info = ds[0]
    assert info["time_shift"] == 0


def test_dataset_refuses_to_build_on_a_shifter_that_does_not_publish(tmp_path, monkeypatch):
    """r1 review finding 5, fixed at the only place it CAN be fixed.

    Round 1 defaulted a missing ``last_shift`` to 0 inside ``__getitem__``. That
    is fail-open, and raising there would have been no better: ``__getitem__``
    catches every exception and substitutes a random item, so the error would have
    become an unbounded retry loop instead of a stop. The contract is therefore
    proven at CONSTRUCTION, outside that handler.
    """
    from torch import nn

    import src.data.dataset as dsmod

    class _MuteShifter(nn.Module):
        """A time shifter that applies a shift and tells nobody."""

        def __init__(self, max_shift=10, p=0.5):
            super().__init__()

        def forward(self, signal):
            return signal

    monkeypatch.setattr(dsmod, "RandomTimeShift", _MuteShifter)
    split = _tiny_ar_tree(tmp_path)
    with pytest.raises(RuntimeError, match="publishes no 'last_shift'"):
        _dataset(tmp_path, split, augs=True)


def test_dataset_refuses_two_time_shifters_in_one_pipeline(tmp_path, monkeypatch):
    """The published value would describe only one of them."""
    import src.data.dataset as dsmod
    from src.data.utils import RandomTimeShift

    class _ExtraShifter(RandomTimeShift):
        """Stands in for AddNoise's slot, but is itself a time shifter."""

        def __init__(self, snr_db_range=None, noise_type=None, p=0.5):
            super().__init__(max_shift=4, p=p)

    monkeypatch.setattr(dsmod, "AddNoise", _ExtraShifter)
    split = _tiny_ar_tree(tmp_path)
    with pytest.raises(RuntimeError, match="RandomTimeShift modules"):
        _dataset(tmp_path, split, augs=True)


def test_a_reordered_pipeline_still_finds_the_shifter(tmp_path, monkeypatch):
    """Resolution is BY TYPE, so moving the shifter off index 0 -- the exact
    regression round 1's ``self.augs[0]`` would have missed -- changes nothing."""
    import random

    from torch import nn

    import src.data.dataset as dsmod
    from src.data.utils import AddNoise, RandomTimeShift

    real_sequential = nn.Sequential

    def _reversed_sequential(*mods):
        return real_sequential(*reversed(mods))

    monkeypatch.setattr(dsmod.torch.nn, "Sequential", _reversed_sequential)
    split = _tiny_ar_tree(tmp_path)
    ds = _dataset(tmp_path, split, augs=True)
    assert isinstance(ds.augs[0], AddNoise)                 # not the shifter
    assert isinstance(ds._time_shifter, RandomTimeShift)    # found anyway

    for seed in range(20):
        random.seed(seed)
        audio, info = ds[0]
        assert int(torch.argmax(audio.abs())) == 500 + info["time_shift"]
