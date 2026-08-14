"""exp_16 (are_port) — ARE anchors: an analytic direct-sound prior in the VAE latent.

Seat: Opus 5 Coder (SOP §Roles). Plan:
``worklog/worklog_yixun/exp_16_are_port_claude/plan_are_port.md`` §2. Design
source: rir2rir ``exp_15_anchor/plan_anchor.md`` §§4.a-4.d — re-implemented here
against FLAC's stack (its config-driven sample rate, its 1024-sample VAE hop, its
listener-frame ``source`` pose and its equirectangular depth panorama), not copied.

WHAT AN ANCHOR IS
-----------------
For one IR path with listener-frame source ``s`` and ``r = ||s||``::

    t*      = r / 343 * fs + delta_hat            # fs from the model config
    g       = A_g / max(r, 0.05)                  # A_g calibrated on AR TRAIN
    skel[n] = g * hann(n - t*) * sinc(n - t*)     # |n - t*| <= H, unit-l2 scaled
    A(r)    = Pi_early( Enc_mean(skel) - Enc_mean(0) )
    A(p)    = 1[LOS(p)] * A(r)

and FLAC's rectified flow is trained to ``noise -> (z - lambda*A(p))``, with
``+lambda*A_query`` added back before the single decode.

FIVE PROPERTIES THIS MODULE OWES ITS CALLERS, AND WHY
-----------------------------------------------------
1. **Unit-l2 amplitude convention.** The sampled peak of a windowed sinc swings
   3.92 dB across sub-sample phase while its l2 energy varies by 1.3e-2. Keying
   the amplitude to energy therefore removes a phase artefact of the same order
   as the physical 1/r effect under test. ``||skel||_2 == g`` EXACTLY, at every
   phase and at every distance (the normalisation is applied to the *placed*
   skeleton, so a window clipped by the array boundary cannot change the scale).

2. **The silence bias is subtracted, once.** ``Enc(0)`` is not the zero latent —
   it is a large, information-free constant. ``A`` is always
   ``Enc(skel) - Enc(0)``; there is exactly one function that subtracts it
   (:meth:`AnchorBank.anchor_from_waveform`) and every other entry point routes
   through it.

3. **No RNG, ever.** FLAC's VAE bottleneck samples (``vae_sample`` draws
   ``randn_like``). Routing the anchor through it would both randomise the anchor
   and *displace the training noise stream*, which would silently break the
   comparison against P1 — exp_16's lambda=0 control is a run that already
   happened, so its RNG stream is not re-runnable. The anchor therefore uses the
   encoder MEAN (``skip_bottleneck=True``), and touches no generator anywhere.

4. **Determinism, batch-composition included.** A conv encoder is not
   batch-invariant (~2.4e-6 between B=1 and B=16 on this codec), so every cache
   miss is encoded at **B=1**: ``A`` is a pure function of ``r``, independent of
   who else was in the batch and of what the cache happened to hold.

5. **Fail closed.** Distances outside the representable window, a missing pose or
   panorama, a latent-length mismatch, a config whose kept-frame window cannot
   hold AR's longest path — all raise. None of them degrade to a silent zero.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch

# Speed of sound, pinned (the AR metadata carries metres; the loader carries
# 22,050 Hz). Not re-derived anywhere.
C_SOUND = 343.0

# The largest source-receiver distance in AcousticRooms, measured over all
# 302,925 metadata records under ``AcousticRooms/metadata`` on 2026-08-14
# (max 27.104669 m at Cafe/Cafe_idx_0/S0045_R0034; min 0.500087 m). It is the
# worst case ``compute_worst_case_frames`` is asked about, and it is a MEASURED
# constant rather than a quoted one so that a future dataset change fails the
# check instead of quietly moving the direct peak out of the kept frames.
AR_MAX_SOURCE_RECEIVER_DISTANCE = 27.104669
AR_MIN_SOURCE_RECEIVER_DISTANCE = 0.500087

DEFAULT_KERNEL_HALF_WIDTH = 32       # H: fractional-delay kernel half-width
DEFAULT_EARLY_FRAMES = 3             # latent frames kept (0, 1, 2)
DEFAULT_LOS_THRESHOLD = 0.95         # depth(bearing) >= 0.95*r  =>  line of sight
DEFAULT_DIST_GUARD = 0.05            # metres; floors the 1/r gain
DEFAULT_P_EXP = 1.0                  # the 1/r^p law; p pinned to 1
DEFAULT_CACHE_SIZE = 65536           # ~25 MB of [32, 3] float32 anchors

# ``RandomTimeShift(max_shift=10, p=0.5)`` in the AR training loader
# (``src/data/dataset.py``) moves the TARGET forward by up to this many samples
# while the geometry stays put. The per-sample draw is published as
# ``metadata['time_shift']`` and applied to ``t*``; this constant only bounds the
# worst-case frame check.
DEFAULT_MAX_TIME_SHIFT = 10

# Keys a config's ``training.are_anchor`` block must state, may state, and may
# never state (the last are the MODEL config's, and restating them would let the
# anchor's time base disagree with the audio the run trains on).
ANCHOR_REQUIRED_KEYS = ("delta_hat", "a_g")
ANCHOR_OPTIONAL_KEYS = ("kernel_half_width", "early_frames", "los_threshold",
                        "dist_guard", "p_exp", "cache_size", "max_distance",
                        "max_time_shift")
ANCHOR_MODEL_KEYS = ("sample_rate", "sample_size")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnchorConfig:
    """Everything the anchor needs, all of it declared rather than derived.

    ``sample_rate`` and ``sample_size`` come from the MODEL config and ``hop``
    from the pretransform's ``downsampling_ratio``; ``delta_hat`` and ``a_g``
    come from the calibration record. Nothing here has a "sensible" numeric
    default that could stand in for a missing calibration.
    """

    sample_rate: int
    sample_size: int
    hop: int
    delta_hat: float
    a_g: float
    kernel_half_width: int = DEFAULT_KERNEL_HALF_WIDTH
    early_frames: int = DEFAULT_EARLY_FRAMES
    los_threshold: float = DEFAULT_LOS_THRESHOLD
    dist_guard: float = DEFAULT_DIST_GUARD
    p_exp: float = DEFAULT_P_EXP
    cache_size: int = DEFAULT_CACHE_SIZE
    max_distance: float = AR_MAX_SOURCE_RECEIVER_DISTANCE
    max_time_shift: int = DEFAULT_MAX_TIME_SHIFT

    def __post_init__(self):
        if isinstance(self.max_time_shift, bool) or not isinstance(self.max_time_shift, int):
            raise ValueError(
                f"AnchorConfig.max_time_shift must be an int, got {self.max_time_shift!r}")
        if self.max_time_shift < 0:
            raise ValueError(
                f"AnchorConfig.max_time_shift must be >= 0, got {self.max_time_shift}")
        for name in ("sample_rate", "sample_size", "hop", "kernel_half_width",
                     "early_frames", "cache_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"AnchorConfig.{name} must be an int, got {value!r}")
            if value <= 0:
                raise ValueError(f"AnchorConfig.{name} must be > 0, got {value}")
        for name in ("delta_hat", "a_g", "los_threshold", "dist_guard", "p_exp",
                     "max_distance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"AnchorConfig.{name} must be a number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"AnchorConfig.{name} must be finite, got {value!r}")
        if self.a_g <= 0:
            raise ValueError(f"AnchorConfig.a_g must be > 0, got {self.a_g}")
        if self.dist_guard <= 0:
            raise ValueError(f"AnchorConfig.dist_guard must be > 0, got {self.dist_guard}")
        if self.sample_size % self.hop:
            raise ValueError(
                f"AnchorConfig.sample_size ({self.sample_size}) must be a multiple of "
                f"the latent hop ({self.hop})")
        if self.early_frames > self.latent_len:
            raise ValueError(
                f"AnchorConfig.early_frames ({self.early_frames}) exceeds the latent "
                f"length ({self.latent_len})")

    @property
    def latent_len(self) -> int:
        return self.sample_size // self.hop


def anchor_config_from_dict(block, hop) -> AnchorConfig:
    """Build an :class:`AnchorConfig` from a validated ``training.are_anchor`` block.

    Fail-closed on the RAW values, in the exp_14/exp_15 style: an unknown key is
    a typo that would otherwise be silently ignored, a missing calibrated
    constant would let the anchor run on an assumed one, and
    ``sample_rate``/``sample_size`` are refused inside the block because they are
    properties of the model config, not of the anchor.
    """
    if not isinstance(block, dict):
        raise ValueError(
            f"are_anchor must be an object with keys {list(ANCHOR_REQUIRED_KEYS)}, "
            f"got {type(block).__name__}")
    allowed = set(ANCHOR_REQUIRED_KEYS) | set(ANCHOR_OPTIONAL_KEYS) | set(ANCHOR_MODEL_KEYS)
    unknown = sorted(k for k in block if k not in allowed)
    if unknown:
        raise ValueError(
            f"are_anchor has unknown key(s) {unknown}; allowed keys are "
            f"{sorted(allowed)}")
    for key in ANCHOR_REQUIRED_KEYS:
        if key not in block:
            raise ValueError(
                f"are_anchor requires '{key}' (no default is assumed: it is a "
                "calibrated constant and must be stated by the config)")
    for key in ANCHOR_MODEL_KEYS:
        if key not in block:
            raise ValueError(f"are_anchor is missing '{key}' (supplied by the model config)")
    return AnchorConfig(hop=int(hop), **{k: block[k] for k in block})


# --------------------------------------------------------------------------- #
# the analytic skeleton
# --------------------------------------------------------------------------- #
def _hann_sinc_raw(offsets: torch.Tensor, half_width: int) -> torch.Tensor:
    """``hann(x) * sinc(x)`` on ``|x| <= H``, unnormalised.

    ``hann(x) = 0.5 * (1 + cos(pi x / (H + 1)))`` reaches exactly zero at
    ``|x| = H + 1`` and is clamped to zero beyond it (it turns positive again
    further out, which would re-introduce the tail the window exists to remove).
    """
    offsets = torch.as_tensor(offsets, dtype=torch.float64)
    inside = offsets.abs() <= float(half_width)
    window = 0.5 * (1.0 + torch.cos(math.pi * offsets / (half_width + 1.0)))
    return torch.where(inside, window * torch.sinc(offsets), torch.zeros_like(offsets))


def hann_sinc_kernel(offsets, half_width: int = DEFAULT_KERNEL_HALF_WIDTH) -> torch.Tensor:
    """The unit-l2 Hann-windowed sinc fractional-delay kernel, sampled at ``offsets``.

    ``offsets[k]`` is ``n_k - t*`` in samples, so a non-integer ``t*`` is realised
    at sub-sample precision rather than quantised onto a grid point (one sample at
    22,050 Hz is 0.045 ms, which is material for EDT/C50).

    An all-zero support (every offset outside the window) cannot be normalised and
    is returned as-is rather than as NaN.
    """
    ker = _hann_sinc_raw(offsets, half_width)
    norm = torch.linalg.vector_norm(ker)
    if float(norm) == 0.0:
        return ker
    return ker / norm


def direct_path_skeleton(r, cfg: AnchorConfig, shift_samples=0.0) -> torch.Tensor:
    """The analytic direct-sound waveform for distance ``r``: ``[1, sample_size]``.

    Returned in float64 so that the amplitude law holds to machine precision; the
    encoder path casts to float32 at the boundary.

    ``shift_samples`` is the loader's ``RandomTimeShift`` draw for this sample.
    The augmentation moves the TARGET forward while the geometry stays put, so an
    anchor placed at the geometric ``t*`` would be early by exactly this much on
    the shifted half of the training set — a 1-10 sample (0.045-0.45 ms) error on
    precisely the quantity the experiment is about. It is added to ``t*``, never
    silently ignored.

    The l2 normalisation is applied to the PLACED skeleton, not to the kernel in
    isolation, so ``||skel||_2 == g`` holds exactly even for a ``t*`` close enough
    to sample 0 that the window is clipped by the array boundary.
    """
    r = float(r)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError(f"source-receiver distance must be finite and > 0, got {r!r}")
    shift_samples = float(shift_samples)
    if not math.isfinite(shift_samples):
        raise ValueError(f"time shift must be finite, got {shift_samples!r}")

    t_star = r / C_SOUND * cfg.sample_rate + cfg.delta_hat + shift_samples
    half = cfg.kernel_half_width
    if t_star < 0.0 or t_star + half >= cfg.sample_size:
        raise ValueError(
            f"direct arrival t*={t_star:.3f} samples (r={r:.4f} m, fs={cfg.sample_rate}, "
            f"delta_hat={cfg.delta_hat}, time_shift={shift_samples}) is outside the "
            f"{cfg.sample_size}-sample window the anchor can represent")

    n0 = int(math.floor(t_star))
    idx = torch.arange(n0 - half, n0 + half + 1, dtype=torch.long)
    raw = _hann_sinc_raw(idx.to(torch.float64) - t_star, half)

    valid = (idx >= 0) & (idx < cfg.sample_size)
    skel = torch.zeros(1, cfg.sample_size, dtype=torch.float64)
    skel[0, idx[valid]] = raw[valid]

    norm = float(torch.linalg.vector_norm(skel))
    if norm == 0.0:
        raise ValueError(
            f"the skeleton for r={r:.4f} m placed no energy inside the window "
            f"(t*={t_star:.3f}); refusing to emit a degenerate anchor")
    gain = cfg.a_g / (max(r, cfg.dist_guard) ** cfg.p_exp)
    return skel * (gain / norm)


def compute_worst_case_frames(cfg: AnchorConfig, max_distance: Optional[float] = None) -> dict:
    """Does the longest path in the dataset still land inside the kept frames?

    The check the plan makes mandatory: at FLAC's latent hop the anchor keeps
    frames ``0 .. early_frames-1``, and a direct peak past that window would be
    truncated away — the anchor would then carry no timing information at all for
    the very paths whose timing is hardest.

    The kernel's half-width is included: the peak may sit inside frame 2 while its
    tail crosses into frame 3. So is the loader's largest ``RandomTimeShift``
    draw, which moves the anchor later by up to ``max_time_shift`` samples.
    """
    max_distance = float(cfg.max_distance if max_distance is None else max_distance)
    t_star = (max_distance / C_SOUND * cfg.sample_rate + cfg.delta_hat
              + cfg.max_time_shift)
    last_sample = t_star + cfg.kernel_half_width
    peak_frame = int(math.floor(t_star / cfg.hop))
    required_frames = int(math.floor(last_sample / cfg.hop)) + 1
    kept_samples = cfg.early_frames * cfg.hop
    return {
        "max_distance": max_distance,
        "sample_rate": cfg.sample_rate,
        "hop": cfg.hop,
        "delta_hat": cfg.delta_hat,
        "kernel_half_width": cfg.kernel_half_width,
        "max_time_shift": cfg.max_time_shift,
        "early_frames": cfg.early_frames,
        "t_star": t_star,
        "last_sample": last_sample,
        "peak_frame": peak_frame,
        "required_frames": required_frames,
        "kept_samples": kept_samples,
        # the largest distance the kept window could hold, as reported headroom
        "max_representable_distance": max(
            0.0, (kept_samples - cfg.kernel_half_width - cfg.delta_hat
                  - cfg.max_time_shift) * C_SOUND / cfg.sample_rate),
        "ok": required_frames <= cfg.early_frames,
    }


def assert_worst_case_frames(cfg: AnchorConfig, max_distance: Optional[float] = None) -> dict:
    """:func:`compute_worst_case_frames`, as a hard gate. Returns the report."""
    report = compute_worst_case_frames(cfg, max_distance)
    if not report["ok"]:
        raise ValueError(
            "ARE anchor window too narrow: the worst-case source-receiver distance "
            f"{report['max_distance']:.4f} m puts the direct peak at sample "
            f"{report['t_star']:.1f} (frame {report['peak_frame']}), and with the "
            f"kernel half-width the anchor needs {report['required_frames']} latent "
            f"frames — but early_frames is {report['early_frames']}. Widen "
            f"are_anchor.early_frames to {report['required_frames']} (and record the "
            "deviation) or shorten the worst-case distance.")
    return report


# --------------------------------------------------------------------------- #
# geometry: the line-of-sight gate
# --------------------------------------------------------------------------- #
def as_source_vector(source) -> torch.Tensor:
    """A listener-frame source pose as a flat float64 ``[3]``.

    ``AR_md`` publishes the same position twice — ``source`` as ``[3]`` and
    ``source_vit`` as ``[1, 3]`` — so a ``[1, 3]`` input must not be read as three
    separate coordinates.
    """
    s = torch.as_tensor(source, dtype=torch.float64).reshape(-1)
    if s.numel() != 3:
        raise ValueError(
            f"source pose must hold exactly 3 coordinates, got {tuple(torch.as_tensor(source).shape)}")
    if not bool(torch.isfinite(s).all()):
        raise ValueError(f"source pose must be finite, got {s.tolist()}")
    return s


def source_bearing_pixel(source, img_h: int, img_w: int) -> Tuple[int, int]:
    """Which panorama pixel contains the source bearing: ``(row, col)``.

    Inverse of ``AR_md.convert_equirect_to_camera_coord``, which places pixel
    ``(i, j)`` at ``phi = (i+0.5)*pi/H - pi/2`` and
    ``theta = (j+0.5)*2pi/W - pi`` with
    ``p = d * [cos(phi)cos(theta), cos(phi)sin(theta), -sin(phi)]``. "Which pixel
    CONTAINS this bearing" is therefore a ``floor``, not a ``round``.

    The horizontal seam WRAPS (``mod W``) and the vertical boundary CLAMPS (no
    pole wrap) — the same conventions the design source pinned.
    """
    s = as_source_vector(source)
    r = float(torch.linalg.vector_norm(s))
    if r <= 0.0:
        raise ValueError("source pose has zero length; its bearing is undefined")
    phi = math.asin(min(1.0, max(-1.0, float(-s[2]) / r)))
    theta = math.atan2(float(s[1]), float(s[0]))
    row = int(math.floor((phi + math.pi / 2.0) * img_h / math.pi))
    col = int(math.floor((theta + math.pi) * img_w / (2.0 * math.pi)))
    return min(max(row, 0), img_h - 1), col % img_w


def line_of_sight(depth, source, cfg: AnchorConfig) -> bool:
    """Is the source bearing unoccluded? Geometry only — no waveform, no GT.

    ``depth(bearing) >= los_threshold * ||s||``, evaluated as the MINIMUM over the
    3x3 pixel neighbourhood (wrapping the seam, clamping the poles). The minimum
    is the safe direction of error: a one-pixel bearing quantisation on a grazing
    wall closes the gate rather than injecting direct-path energy the target does
    not contain.

    Only the nine pixels are read. Norming the whole 256x512 panorama would cost
    ~400k operations per sample inside ``training_step`` for an answer that
    depends on nine of them.
    """
    d = torch.as_tensor(depth)
    if d.ndim != 3 or d.shape[0] < 3:
        raise ValueError(
            f"depth panorama must have shape [3, H, W], got {tuple(d.shape)}")
    img_h, img_w = int(d.shape[1]), int(d.shape[2])
    s = as_source_vector(source)
    r = float(torch.linalg.vector_norm(s))
    row, col = source_bearing_pixel(s, img_h, img_w)
    rows = [min(max(row + dr, 0), img_h - 1) for dr in (-1, 0, 1)]
    cols = [(col + dc) % img_w for dc in (-1, 0, 1)]
    patch = d[:3][:, rows][:, :, cols].to(torch.float64)          # [3, 3, 3]
    ranges = torch.linalg.vector_norm(patch, dim=0)               # [3, 3]
    return bool(float(ranges.min()) >= cfg.los_threshold * r)


# --------------------------------------------------------------------------- #
# the frozen-encoder seam
# --------------------------------------------------------------------------- #
def make_encode_mean(pretransform) -> Callable[[torch.Tensor], torch.Tensor]:
    """A deterministic ``[B, 1, T] -> [B, C, L]`` mean encode for a VAE pretransform.

    NOT ``pretransform.encode``: that runs ``VAEBottleneck.encode``, whose
    ``vae_sample`` draws ``randn_like``. Using it would (a) randomise the anchor
    and (b) advance the global RNG inside ``training_step``, changing the noise
    every subsequent sample sees. The mean is taken from the pre-bottleneck
    latents instead, with autocast explicitly disabled so the anchor does not
    silently change precision with the ambient AMP state.
    """
    if pretransform is None:
        raise ValueError(
            "ARE needs the frozen VAE pretransform to encode its skeleton, but the "
            "model has pretransform=None")
    explicit = getattr(pretransform, "encode_mean", None)
    if callable(explicit):
        return explicit

    model = getattr(pretransform, "model", None)
    if model is None or not hasattr(model, "encode"):
        raise ValueError(
            f"pretransform {type(pretransform).__name__} exposes neither 'encode_mean' "
            "nor a '.model.encode'; ARE cannot obtain a deterministic mean encode")

    from ..models.bottleneck import VAEBottleneck        # local: avoid an import cycle

    bottleneck = getattr(model, "bottleneck", None)
    if not isinstance(bottleneck, VAEBottleneck):
        raise ValueError(
            f"ARE expects a VAE bottleneck (got {type(bottleneck).__name__}); the "
            "mean/scale split of the pre-bottleneck latents is what makes the anchor "
            "deterministic")
    scale = float(getattr(pretransform, "scale", 1.0) or 1.0)

    def _encode_mean(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.amp.autocast(x.device.type, enabled=False):
            latents = model.encode(x.float(), skip_bottleneck=True)
        if latents.shape[1] % 2:
            raise ValueError(
                f"pre-bottleneck latents have an odd channel count {latents.shape[1]}; "
                "they cannot be split into (mean, scale)")
        mean, _scale = latents.chunk(2, dim=1)
        return (mean / scale).float()

    return _encode_mean


class AnchorBank:
    """Deterministic, LRU-cached anchors for one (frozen VAE, :class:`AnchorConfig`).

    The cache is keyed on the exact ``(distance, time_shift)`` pair — no
    quantisation, so there is no bin-width error to justify — and every miss is
    encoded at ``B=1`` so a hit and a miss are bit-equal and ``A`` is a pure
    function of its key, independent of who else was in the batch.

    A batch's misses are nevertheless encoded and transferred as ONE group
    (:meth:`anchors_for`): encoding at B=1 costs 8.2 ms/sample on an A6000, but
    copying each result back to the host separately costs a further ~17 ms
    because every copy synchronises the device. Measured on the real VAE, that is
    the difference between 21 % and ~7 % of a training step — so the transfers are
    batched even though the encodes deliberately are not.
    """

    def __init__(self, encode_mean_fn: Callable[[torch.Tensor], torch.Tensor],
                 cfg: AnchorConfig):
        if not callable(encode_mean_fn):
            raise ValueError("encode_mean_fn must be callable")
        if not isinstance(cfg, AnchorConfig):
            raise ValueError(f"cfg must be an AnchorConfig, got {type(cfg).__name__}")
        # Fail closed at construction: a window that cannot hold the dataset's
        # longest path must never reach a six-day run.
        self.worst_case = assert_worst_case_frames(cfg)
        self.cfg = cfg
        self._encode = encode_mean_fn
        self._cache: "OrderedDict[float, torch.Tensor]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._bias: Optional[torch.Tensor] = None
        self._bias_by_device: dict = {}

    # -- bookkeeping -------------------------------------------------------- #
    def cache_stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses,
                "size": len(self._cache), "capacity": self.cfg.cache_size}

    def clear_cache(self) -> None:
        """Drop the anchor cache and its counters. The silence bias is NOT
        dropped: it is a property of the frozen encoder, not of the cache, and
        re-encoding it would be a second, redundant definition of the bias."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    # -- the one place the bias is computed, and the one place it is removed -- #
    def silence_bias(self, device) -> torch.Tensor:
        """``Enc_mean(0)`` — encoded once, then held resident on each device.

        The per-device copy is not a micro-optimisation: a fresh ``.to(device)``
        on every anchor is a pageable host-to-device copy, which SYNCHRONISES the
        stream and measured ~7 ms per sample on an A6000 — more than the encode
        it precedes. The returned tensor is read-only by contract; every consumer
        below either clones or only reads its shape.
        """
        key = str(device)
        resident = self._bias_by_device.get(key)
        if resident is not None:
            return resident
        if self._bias is None:
            zeros = torch.zeros(1, 1, self.cfg.sample_size, dtype=torch.float32,
                                device=device)
            self._bias = self._encode(zeros)[0].detach().to("cpu", torch.float32)
        resident = self._bias.to(device)
        self._bias_by_device[key] = resident
        return resident

    def _bias_correct(self, latents: torch.Tensor, device,
                      truncate: bool = True) -> torch.Tensor:
        """``Pi_early(latents - Enc_mean(0))`` for a ``[B, C, L]`` batch.

        The ONLY place the silence bias is subtracted and the ONLY place the
        frame window is applied. Every entry point below routes through it, so
        neither correction can be applied twice or forgotten in one path.
        """
        out = (latents - self.silence_bias(device)[None]).clone()
        if truncate:
            out[:, :, self.cfg.early_frames:] = 0.0
        return out

    def anchor_from_waveform(self, waveform, device, truncate: bool = True) -> torch.Tensor:
        """``Pi_early(Enc_mean(w) - Enc_mean(0))`` for one waveform: ``[C, L]``.

        ``truncate=False`` is a diagnostic (how much of the corrected anchor was
        already early), never the training path.
        """
        w = torch.as_tensor(waveform)
        x = w.reshape(1, 1, -1).to(device=device, dtype=torch.float32)
        if x.shape[-1] != self.cfg.sample_size:
            raise ValueError(
                f"anchor waveform has {x.shape[-1]} samples, expected "
                f"{self.cfg.sample_size}")
        return self._bias_correct(self._encode(x), device, truncate)[0]

    # -- the cached entry points -------------------------------------------- #
    def anchors_for(self, keys, device) -> torch.Tensor:
        """``[N, C, L]`` on ``device`` for a list of ``(distance, time_shift)`` keys.

        Order is preserved and duplicates are encoded once. Cache hits are
        resolved BEFORE any eviction can run, so a cache smaller than the batch
        cannot drop a row that is still needed.
        """
        keys = [(float(r), float(shift)) for r, shift in keys]
        resolved, missing, pending = {}, [], set()
        for key in keys:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                resolved[key] = self._cache[key]
            else:
                self._misses += 1
                if key not in pending:
                    pending.add(key)
                    missing.append(key)

        if missing:
            # Encode at B=1 -- so A stays a pure function of its key, independent
            # of batch composition -- but make exactly ONE host->device and ONE
            # device->host transfer for the whole group. A per-item copy in either
            # direction synchronises the stream and costs more than the encode.
            host_in = torch.empty(len(missing), 1, self.cfg.sample_size,
                                  dtype=torch.float32)
            for i, (r, shift) in enumerate(missing):
                host_in[i, 0] = direct_path_skeleton(r, self.cfg, shift_samples=shift)[0]
            skeletons = host_in.to(device)
            computed = [self._bias_correct(self._encode(skeletons[i:i + 1]), device)[0]
                        for i in range(len(missing))]
            host = torch.stack(computed, dim=0).detach().to("cpu", torch.float32)
            for i, key in enumerate(missing):
                row = host[i].clone()
                resolved[key] = row
                self._cache[key] = row
            while len(self._cache) > self.cfg.cache_size:
                self._cache.popitem(last=False)

        return torch.stack([resolved[key] for key in keys], dim=0).to(device)

    def anchor_for_distance(self, r, device, time_shift=0.0) -> torch.Tensor:
        return self.anchors_for([(r, time_shift)], device)[0]

    def anchor_for_sample(self, source, depth, device, time_shift=0.0) -> torch.Tensor:
        """The gated anchor for one sample: ``1[LOS] * A(||s||, shift)``, ``[C, L]``."""
        s = as_source_vector(source)
        if not line_of_sight(depth, s, self.cfg):
            return torch.zeros_like(self.silence_bias(device))
        return self.anchor_for_distance(float(torch.linalg.vector_norm(s)), device,
                                        time_shift)


def build_anchor_bank(pretransform, sample_rate: int, sample_size: int,
                      anchor_cfg: dict) -> AnchorBank:
    """Assemble an :class:`AnchorBank` from a model's pretransform + its config block.

    ``hop`` is read off the pretransform (``downsampling_ratio``) and the time base
    off the model config — never hardcoded, so a different VAE or a different
    sample rate changes the anchor rather than silently mis-placing it.
    """
    hop = getattr(pretransform, "downsampling_ratio", None)
    if hop is None:
        raise ValueError(
            "ARE needs the pretransform's downsampling_ratio to place the anchor in "
            "latent frames")
    block = dict(anchor_cfg or {})
    block["sample_rate"] = int(sample_rate)
    block["sample_size"] = int(sample_size)
    cfg = anchor_config_from_dict(block, int(hop))
    return AnchorBank(make_encode_mean(pretransform), cfg)


# --------------------------------------------------------------------------- #
# batch entry points (the two things the training/eval sites call)
# --------------------------------------------------------------------------- #
def compute_are_anchors(bank: AnchorBank, metadata, device) -> torch.Tensor:
    """Per-sample gated anchors for one batch: ``[B, C, L]``.

    Every input is already in the batch — ``AR_md`` publishes ``source`` (the
    listener-frame pose) and ``depth`` (the listener's panorama as a point cloud),
    and the loader publishes ``time_shift`` (the augmentation draw it already
    made) — so nothing here touches the filesystem, the dataloader's RNG or the
    batch contract. A sample missing any of them is a hard error: silently
    emitting a zero anchor, or silently assuming an unshifted target, would train
    a mixture of two methods.

    The gate is evaluated first and only the LOS samples are encoded, so an
    occluded sample costs nine depth lookups and nothing else.
    """
    if not isinstance(metadata, (list, tuple)):
        raise ValueError(
            f"metadata must be a list of per-sample dicts, got {type(metadata).__name__}")
    if len(metadata) == 0:
        raise ValueError("metadata is empty; there is no batch to anchor")

    keys, live_index = [], []
    for i, md in enumerate(metadata):
        if not isinstance(md, dict):
            raise ValueError(f"metadata[{i}] must be a dict, got {type(md).__name__}")
        if "source" not in md:
            raise ValueError(
                f"metadata[{i}] has no 'source' pose; ARE cannot compute r = ||s||")
        if "depth" not in md:
            raise ValueError(
                f"metadata[{i}] has no 'depth' panorama; ARE cannot evaluate the "
                "line-of-sight gate")
        if "time_shift" not in md:
            raise ValueError(
                f"metadata[{i}] has no 'time_shift'; the loader's RandomTimeShift "
                "moves the target without moving the geometry, so an anchor placed "
                "without it would be systematically early on half the training set")
        shift = md["time_shift"]
        if torch.is_tensor(shift):
            shift = float(shift.reshape(-1)[0])
        if isinstance(shift, bool) or not isinstance(shift, (int, float)):
            raise ValueError(
                f"metadata[{i}]['time_shift'] must be a number of samples, got {shift!r}")
        s = as_source_vector(md["source"])
        if line_of_sight(md["depth"], s, bank.cfg):
            keys.append((float(torch.linalg.vector_norm(s)), float(shift)))
            live_index.append(i)

    zero = torch.zeros(len(metadata), *bank.silence_bias(device).shape,
                       device=device, dtype=torch.float32)
    if not keys:
        return zero
    anchors = bank.anchors_for(keys, device)
    zero[live_index] = anchors.to(zero.dtype)
    return zero


def apply_anchor_addback(latents: torch.Tensor, anchors: torch.Tensor,
                         are_lambda: float) -> torch.Tensor:
    """``z_hat + lambda * A_query`` — the inference half of the reparameterisation.

    Applied to the LATENT, before the single decode, so the K-reference averaging
    (which happens inside the model's cross-attention, not over decoded outputs)
    is untouched.
    """
    if anchors.shape != latents.shape:
        raise ValueError(
            f"anchor shape {tuple(anchors.shape)} does not match the sampled latent "
            f"shape {tuple(latents.shape)}")
    return latents + float(are_lambda) * anchors.to(latents.dtype)
