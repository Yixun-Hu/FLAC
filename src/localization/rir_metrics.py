"""Non-AGREE waveform / acoustic scorers for exp_18 R4 (exploratory, firewalled).

Implements the five pre-declared metric families of ``plan_loc_invert_R4.md`` §1
plus the metric-matched retrieval control of §2. Everything here is a pure
function over torch/numpy arrays: the driver calls :func:`compute_metrics` inside
the replay loop, so no generation happens twice.

Integrity rules this module is written to satisfy:

* Every formula constant is a module-level REGISTERABLE, collected in
  :data:`REGISTERABLE` so the metric-registration manifest can freeze it. Nothing
  is chosen here -- ``delta_max`` in particular is PASSED IN, selected on the R1
  seen prefix from the pre-listed grid :data:`M1_DELTA_GRID`.
* Repo estimators are reused, never re-implemented (``RT60``, ``EDT``, ``C50``,
  ``Env``, ``l1_stft_multires``); where a repo helper is device-locked, the
  deviation is stated at the call site.
* Both directions -- generated-vs-obs and context-vs-obs -- run through the SAME
  function objects with the same window/alignment/amplitude policy.
"""
import torch

from src.metrics.modules.l1_stft_multires import safe_log  # noqa: F401  (M2 reuses it)

#: audio conventions (repo-wide).
SAMPLE_RATE = 22050
#: common analysis window for M1/M2/M3/M5: the repo's context ``max_len``.
WINDOW_SAMPLES = 9600
#: acoustic-parameter window for M4: the repo's AR ``max_len``.
PARAM_WINDOW_SAMPLES = 8000
EPS = 1e-8

#: M1/M5: the pre-listed alignment grid. The ONE calibrated constant is which of
#: these is registered, and that choice is made on the R1 seen prefix elsewhere.
M1_DELTA_GRID = (0, 8, 32, 128)

#: M2: the repo's multiscale_log_l1 scale set and epsilons (pinned by test).
M2_FFT_SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
M2_STFT_EPS = 1e-6
M2_SAFE_LOG_EPS = 1e-7
M2_LAMBDA = 1.0

#: M3: decay region (dB, defined on the OBSERVED curve) and secondary bands.
M3_REGION_DB = (0.0, -30.0)
M3_OCTAVE_BANDS_HZ = (500, 1000, 2000, 4000)

#: M4: the fixed feature vector and its estimator conventions.
M4_FEATURES = ("arrival_time", "drr", "c50", "c80", "edt", "t30", "early_late_50ms",
               "t30_500", "t30_1k", "t30_2k")
M4_ARRIVAL_THRESHOLD_DB = -20.0
M4_DIRECT_HALF_WIDTH_MS = 2.5
M4_EARLY_LATE_MS = 50.0
M4_T30_DECAY_DB = 30.0
M4_EDT_DECAY_DB = 10.0

#: M5 secondary.
M5_SECONDARY = "gcc_phat"

#: K aggregation and prediction rule (plan §1).
K_AGGREGATION_PRIMARY = "mean"
K_AGGREGATION_SECONDARIES = ("min", "median", "lme")
LME_TAU = 0.02
PREDICTION_TIE_BREAK = "lowest_index"

REGISTERABLE = {
    "sample_rate": SAMPLE_RATE,
    "window_samples": WINDOW_SAMPLES,
    "param_window_samples": PARAM_WINDOW_SAMPLES,
    "eps": EPS,
    "m1_delta_grid": M1_DELTA_GRID,
    "m2_fft_sizes": M2_FFT_SIZES,
    "m2_stft_eps": M2_STFT_EPS,
    "m2_safe_log_eps": M2_SAFE_LOG_EPS,
    "m2_lambda": M2_LAMBDA,
    "m3_region_db": M3_REGION_DB,
    "m3_octave_bands_hz": M3_OCTAVE_BANDS_HZ,
    "m4_features": M4_FEATURES,
    "m4_arrival_threshold_db": M4_ARRIVAL_THRESHOLD_DB,
    "m4_direct_half_width_ms": M4_DIRECT_HALF_WIDTH_MS,
    "m4_early_late_ms": M4_EARLY_LATE_MS,
    "m4_t30_decay_db": M4_T30_DECAY_DB,
    "m4_edt_decay_db": M4_EDT_DECAY_DB,
    "m5_secondary": M5_SECONDARY,
    "k_aggregation_primary": K_AGGREGATION_PRIMARY,
    "k_aggregation_secondaries": K_AGGREGATION_SECONDARIES,
    "lme_tau": LME_TAU,
    "prediction_tie_break": PREDICTION_TIE_BREAK,
}


def registerable_payload():
    """JSON-serializable view of :data:`REGISTERABLE` for the metric manifest."""
    def _plain(value):
        if isinstance(value, tuple):
            return list(value)
        return value

    return {key: _plain(value) for key, value in REGISTERABLE.items()}


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #
def _to_window(x, samples):
    x = torch.as_tensor(x).float()
    if x.shape[-1] >= samples:
        return x[..., :samples].contiguous()
    pad = torch.zeros(*x.shape[:-1], samples - x.shape[-1], dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1)


def common_window(x, samples=WINDOW_SAMPLES):
    """Crop or zero-pad to the common analysis window (M1/M2/M3/M5)."""
    return _to_window(x, samples)


def param_window(x, samples=PARAM_WINDOW_SAMPLES):
    """Crop or zero-pad to the acoustic-parameter window (M4)."""
    return _to_window(x, samples)


# --------------------------------------------------------------------------- #
# shared lag machinery for M1 and M5 (zero-padded shifts, never wraparound)
# --------------------------------------------------------------------------- #
def shift(x, delta):
    """``out[t] = x[t - delta]``, zero outside -- no wraparound."""
    out = torch.zeros_like(x)
    length = x.shape[-1]
    if delta >= length or -delta >= length:
        return out
    if delta >= 0:
        out[..., delta:] = x[..., : length - delta]
    else:
        out[..., : length + delta] = x[..., -delta:]
    return out


def lag_products(x, y, delta_max):
    """``(dots, energies)`` over lags ``-delta_max .. +delta_max``.

    ``dots[..., i] = <y, shift(x, delta_i)>`` and
    ``energies[..., i] = ||shift(x, delta_i)||^2``. Both are computed in closed
    form from one FFT correlation and one prefix-energy scan, so the cost does not
    grow with the number of lags -- but the values are exactly those of the naive
    zero-padded shift (asserted against a loop in the tests).
    """
    # float64 internally: an FFT correlation over ~10k samples loses too much in
    # float32 for "a scaled, shifted copy scores exactly 0" to hold.
    x = torch.as_tensor(x).double()
    y = torch.as_tensor(y).double().reshape(-1)
    length = x.shape[-1]
    if y.shape[-1] != length:
        raise ValueError(f"x and y must share the window length, got {length} and {y.shape[-1]}")
    delta_max = int(delta_max)
    if delta_max < 0 or delta_max >= length:
        raise ValueError(f"delta_max must be in [0, {length}), got {delta_max}")

    size = 1
    while size < 2 * length:
        size *= 2
    fx = torch.fft.rfft(x, n=size)
    fy = torch.fft.rfft(y, n=size)
    correlation = torch.fft.irfft(fy * torch.conj(fx), n=size)   # index d -> <y, shift(x, d)>

    lags = torch.arange(-delta_max, delta_max + 1, device=x.device)
    dots = correlation.index_select(-1, lags % size)

    prefix = torch.cumsum(x ** 2, dim=-1)
    total = prefix[..., -1:]
    energies = torch.empty(*x.shape[:-1], lags.numel(), dtype=x.dtype, device=x.device)
    for i, delta in enumerate(lags.tolist()):
        if delta >= 0:
            energies[..., i] = prefix[..., length - delta - 1] if delta < length else 0.0
        else:
            energies[..., i] = (total.squeeze(-1) - prefix[..., -delta - 1])
    return dots, energies


def _validate_delta_max(delta_max):
    if int(delta_max) not in M1_DELTA_GRID:
        raise ValueError(f"delta_max={delta_max} is not in the pre-listed grid {M1_DELTA_GRID}; "
                         "the alignment bound is a registered constant, not a free parameter")
    return int(delta_max)


# --------------------------------------------------------------------------- #
# M1 — aligned, scale-invariant waveform residual
# --------------------------------------------------------------------------- #
def m1_distance(pred, obs, delta_max, eps=EPS):
    """``min_delta ||obs - alpha*(delta) shift(pred, delta)||^2 / (||obs||^2 + eps)``.

    With the analytic optimal gain ``alpha* = <obs, pred_d> / (||pred_d||^2 + eps)``
    this is exactly ``1 - max_delta rho^2(delta)``, so no search over gains is
    needed. Scale-invariant by construction (that is the point: amplitude lives
    here, and M2/M3 are explicitly amplitude-policy-fixed instead).
    """
    delta_max = _validate_delta_max(delta_max)
    pred = torch.as_tensor(pred)
    obs = torch.as_tensor(obs).double().reshape(-1)
    flat = pred.reshape(-1, pred.shape[-1])

    dots, energies = lag_products(flat, obs, delta_max)
    obs_energy = (obs ** 2).sum()
    rho2 = dots ** 2 / ((energies + eps) * (obs_energy + eps))
    distance = 1.0 - rho2.max(dim=-1).values
    return distance.clamp(min=0.0).reshape(pred.shape[:-1]).float()


# --------------------------------------------------------------------------- #
# M5 — normalized cross-correlation (+ GCC-PHAT secondary)
# --------------------------------------------------------------------------- #
def m5_distance(pred, obs, delta_max, eps=EPS):
    """``(1 - max_delta NCC, peak lag)`` under the same bound and pad convention.

    Related to M1 by disclosure, not by derivation: M1 squares the correlation
    (gain-fitted energy residual), M5 does not (sign-sensitive similarity). Both
    are pre-declared families; neither may be promoted post hoc.
    """
    delta_max = _validate_delta_max(delta_max)
    pred = torch.as_tensor(pred)
    obs = torch.as_tensor(obs).double().reshape(-1)
    flat = pred.reshape(-1, pred.shape[-1])

    dots, energies = lag_products(flat, obs, delta_max)
    obs_norm = torch.sqrt((obs ** 2).sum() + eps)
    ncc = dots / (torch.sqrt(energies + eps) * obs_norm)
    best = ncc.max(dim=-1)
    lags = torch.arange(-delta_max, delta_max + 1, device=flat.device)
    return ((1.0 - best.values).reshape(pred.shape[:-1]).float(),
            lags[best.indices].reshape(pred.shape[:-1]))


def gcc_phat_lag(pred, obs, delta_max, eps=EPS):
    """Declared M5 secondary: GCC-PHAT peak lag over the same lag bound."""
    delta_max = int(delta_max)
    pred = torch.as_tensor(pred)
    obs = torch.as_tensor(obs).double().reshape(-1)
    flat = pred.reshape(-1, pred.shape[-1]).double()
    length = flat.shape[-1]

    size = 1
    while size < 2 * length:
        size *= 2
    fx = torch.fft.rfft(flat, n=size)
    fy = torch.fft.rfft(obs, n=size)
    cross = fy * torch.conj(fx)
    phat = torch.fft.irfft(cross / (cross.abs() + eps), n=size)
    lags = torch.arange(-delta_max, delta_max + 1, device=flat.device)
    windowed = phat.index_select(-1, lags % size)
    return lags[windowed.argmax(dim=-1)].reshape(pred.shape[:-1])
