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
from dataclasses import dataclass, field

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


# --------------------------------------------------------------------------- #
# M2 — multi-resolution STFT distance
# --------------------------------------------------------------------------- #
def _stft_amplitude(x, n_fft, eps=M2_STFT_EPS):
    """Amplitude spectrogram in the repo's convention.

    Deviation, stated: ``l1_stft_multires.get_stft`` hardcodes
    ``torch.hann_window(n_fft).cuda()``, so it cannot run on CPU. Everything else
    -- window type, hop default, the ``sqrt(re^2 + im^2 + eps)`` amplitude and the
    imported ``safe_log`` -- is the repo's, and the scale set is pinned by test.
    """
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(x, n_fft=n_fft, hop_length=None, window=window, return_complex=False)
    return torch.sqrt(spec[..., 0] ** 2 + spec[..., 1] ** 2 + eps)


def m2_terms(pred, obs, eps=M2_STFT_EPS, fft_sizes=M2_FFT_SIZES):
    """Per-scale ``{log_l1, convergence}`` terms (both ``[B]``)."""
    pred = torch.as_tensor(pred).float()
    obs = torch.as_tensor(obs).float().reshape(-1)
    flat = pred.reshape(-1, pred.shape[-1])
    reference = obs.unsqueeze(0).expand_as(flat)

    terms = {}
    for n_fft in fft_sizes:
        est_amp = _stft_amplitude(flat, n_fft, eps=eps)
        ref_amp = _stft_amplitude(reference, n_fft, eps=eps)
        log_l1 = torch.abs(safe_log(est_amp) - safe_log(ref_amp)).mean(dim=(-2, -1))
        numerator = torch.linalg.norm((est_amp - ref_amp).reshape(flat.shape[0], -1), dim=-1)
        denominator = torch.linalg.norm(ref_amp.reshape(flat.shape[0], -1), dim=-1)
        terms[n_fft] = {"log_l1": log_l1, "convergence": numerator / (denominator + EPS)}
    return terms


def m2_distance(pred, obs, lam=M2_LAMBDA, eps=M2_STFT_EPS, fft_sizes=M2_FFT_SIZES):
    """``sum_r [ mean|log|X_r| - log|Y_r|| + lam * ||X_r|-|Y_r||_F / (||Y_r||_F+eps) ]``.

    Raw amplitudes, no per-pair gain: the amplitude question is M1's by design
    (plan §1), so a scaled copy is deliberately NOT free here.
    """
    pred = torch.as_tensor(pred).float()
    terms = m2_terms(pred, obs, eps=eps, fft_sizes=fft_sizes)
    total = sum(terms[n]["log_l1"] + float(lam) * terms[n]["convergence"] for n in fft_sizes)
    return total.reshape(pred.shape[:-1])


# --------------------------------------------------------------------------- #
# M3 — envelope / energy-decay distance
# --------------------------------------------------------------------------- #
def schroeder_edc(x, eps=1e-10):
    """Normalized log Schroeder energy-decay curve, in dB.

    The repo's integration (``RT60._measure_rt60_torch``): reversed cumulative
    power, 10*log10, normalized to 0 dB at the window start. Amplitude-blind by
    construction -- decay SHAPE is the semantics here, amplitude is M1's job.
    """
    x = torch.as_tensor(x).float()
    power = x ** 2
    energy = torch.flip(torch.cumsum(torch.flip(power, dims=[-1]), dim=-1), dims=[-1])
    energy_db = 10.0 * torch.log10(energy + eps)
    return energy_db - energy_db[..., :1]


def m3_region_mask(obs, region_db=M3_REGION_DB):
    """The ``[0 dB, -30 dB]`` region of the OBSERVED curve.

    Fixed by the observation so every candidate of a query is scored over the
    same samples; a candidate cannot move its own goalposts.
    """
    edc = schroeder_edc(torch.as_tensor(obs).float().reshape(1, -1))[0]
    upper, lower = float(region_db[0]), float(region_db[1])
    return (edc <= upper) & (edc >= lower)


def m3_distance(pred, obs, region_db=M3_REGION_DB):
    """L1 between normalized log Schroeder EDCs over the observation's region."""
    pred = torch.as_tensor(pred).float()
    obs = torch.as_tensor(obs).float().reshape(-1)
    flat = pred.reshape(-1, pred.shape[-1])

    mask = m3_region_mask(obs, region_db=region_db)
    if int(mask.sum()) == 0:
        return torch.full(pred.shape[:-1], float("nan"))
    obs_edc = schroeder_edc(obs.unsqueeze(0))[0][mask]
    pred_edc = schroeder_edc(flat)[:, mask]
    return torch.abs(pred_edc - obs_edc).mean(dim=-1).reshape(pred.shape[:-1])


def _octave_band(x, centre_hz, sample_rate=SAMPLE_RATE):
    """One octave band via FFT masking (no filter-design dependency)."""
    low, high = centre_hz / (2 ** 0.5), centre_hz * (2 ** 0.5)
    spectrum = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / sample_rate).to(x.device)
    keep = ((freqs >= low) & (freqs < high)).to(spectrum.dtype)
    return torch.fft.irfft(spectrum * keep, n=x.shape[-1], dim=-1)


def _short_time_rms(x, frame=256):
    length = (x.shape[-1] // frame) * frame
    frames = x[..., :length].reshape(*x.shape[:-1], length // frame, frame)
    return torch.sqrt((frames ** 2).mean(dim=-1) + EPS)


def m3_band_envelope_distance(pred, obs, bands=M3_OCTAVE_BANDS_HZ, sample_rate=SAMPLE_RATE):
    """Declared secondary: 4-band short-time-RMS envelope L1, per-band peak-normalized."""
    pred = torch.as_tensor(pred).float()
    obs = torch.as_tensor(obs).float().reshape(-1)
    flat = pred.reshape(-1, pred.shape[-1])

    total = torch.zeros(flat.shape[0], dtype=torch.float32, device=flat.device)
    for centre in bands:
        pred_env = _short_time_rms(_octave_band(flat, centre, sample_rate))
        obs_env = _short_time_rms(_octave_band(obs.unsqueeze(0), centre, sample_rate))
        pred_env = pred_env / (pred_env.max(dim=-1, keepdim=True).values + EPS)
        obs_env = obs_env / (obs_env.max(dim=-1, keepdim=True).values + EPS)
        total = total + torch.abs(pred_env - obs_env).mean(dim=-1)
    return (total / len(bands)).reshape(pred.shape[:-1])


def m3_hilbert_envelope_distance(pred, obs):
    """Declared secondary: full-band Hilbert-envelope L1 in ``Env.env_loss``'s
    convention (|hilbert| envelopes, absolute difference normalized by the
    observation's peak, x100)."""
    import numpy as np
    from scipy.signal import hilbert

    pred = torch.as_tensor(pred).float()
    obs = torch.as_tensor(obs).float().reshape(-1)
    flat = pred.reshape(-1, pred.shape[-1]).cpu().numpy()
    obs_env = np.abs(hilbert(obs.cpu().numpy()))
    pred_env = np.abs(hilbert(flat, axis=-1))
    distance = np.mean(np.abs(obs_env[None, :] - pred_env), axis=-1) / (obs_env.max() + EPS)
    return torch.from_numpy(distance * 100.0).float().reshape(pred.shape[:-1])


# --------------------------------------------------------------------------- #
# M4 — acoustic-parameter distance (repo estimators; CPU loop)
# --------------------------------------------------------------------------- #
def _arrival_index(x, threshold_db=M4_ARRIVAL_THRESHOLD_DB):
    """First sample crossing ``threshold_db`` of the window peak (deterministic)."""
    peak = float(torch.max(torch.abs(x)))
    if peak <= 0.0:
        return None
    level = peak * (10.0 ** (threshold_db / 20.0))
    hits = torch.nonzero(torch.abs(x) >= level, as_tuple=False)
    return int(hits[0]) if hits.numel() else None


def m4_features(x, sample_rate=SAMPLE_RATE, t30_backend="pyroomacoustics"):
    """The fixed acoustic feature set of one RIR, via the repo's own estimators.

    Windowed to ``PARAM_WINDOW_SAMPLES`` (the repo's AR convention). Every
    estimator is imported, never re-implemented: ``C50._measure_clarity`` (C50 and
    C80 by its ``time`` argument), ``EDT._edt``, and RT60's pyroomacoustics path
    with the torch Schroeder implementation as the registered fallback -- the
    choice is a config value, decided at seen calibration, never after.
    """
    import numpy as np
    from src.metrics.modules.C50 import _measure_clarity
    from src.metrics.modules.EDT import _edt
    from src.metrics.modules.RT60 import _measure_rt60_torch, _mesure_rt60_pyroomacoustics

    x = param_window(torch.as_tensor(x).float().reshape(-1))
    shaped = x.reshape(1, 1, -1)
    nan = float("nan")

    arrival = _arrival_index(x)
    features = {}
    features["arrival_time"] = nan if arrival is None else arrival / float(sample_rate)

    half = int(round(M4_DIRECT_HALF_WIDTH_MS * 1e-3 * sample_rate))
    if arrival is None:
        features["drr"] = nan
    else:
        low, high = max(0, arrival - half), min(x.shape[-1], arrival + half + 1)
        direct = float((x[low:high] ** 2).sum())
        reverberant = float((x ** 2).sum()) - direct
        features["drr"] = 10.0 * np.log10((direct + EPS) / (reverberant + EPS))

    features["c50"] = float(_measure_clarity(shaped, time=50, fs=sample_rate)[0, 0])
    features["c80"] = float(_measure_clarity(shaped, time=80, fs=sample_rate)[0, 0])
    features["edt"] = float(_edt(x.numpy(), fs=sample_rate, decay_db=M4_EDT_DECAY_DB))

    def _t30(signal):
        shaped_signal = signal.reshape(1, 1, -1)
        if t30_backend == "torch":
            return float(_measure_rt60_torch(shaped_signal, fs=sample_rate,
                                             decay_db=M4_T30_DECAY_DB)[0, 0])
        return float(_mesure_rt60_pyroomacoustics(shaped_signal, fs=sample_rate,
                                                  decay_db=int(M4_T30_DECAY_DB))[0, 0])

    features["t30"] = _t30(x)

    cut = int(round(M4_EARLY_LATE_MS * 1e-3 * sample_rate))
    early = float((x[:cut] ** 2).sum())
    late = float((x[cut:] ** 2).sum())
    features["early_late_50ms"] = 10.0 * np.log10((early + EPS) / (late + EPS))

    for centre, name in zip((500, 1000, 2000), ("t30_500", "t30_1k", "t30_2k")):
        features[name] = _t30(_octave_band(x.reshape(1, -1), centre, sample_rate)[0])
    return features


def m4_feature_vector(x, sample_rate=SAMPLE_RATE, t30_backend="pyroomacoustics"):
    """``(vector [F], names)`` in the registered :data:`M4_FEATURES` order."""
    import numpy as np
    features = m4_features(x, sample_rate=sample_rate, t30_backend=t30_backend)
    return np.asarray([features[name] for name in M4_FEATURES], dtype=np.float64), M4_FEATURES


def m4_validity_mask(candidate_features, obs_features):
    """Per-(query, feature) validity, UNIFORM across the query's candidates.

    Plan §1: a feature is kept iff it is finite for the observation AND for every
    candidate-sample of that query. Dropping per candidate would let candidates be
    scored on different feature sets.
    """
    import numpy as np
    candidate_features = np.asarray(candidate_features, dtype=np.float64)
    obs_features = np.asarray(obs_features, dtype=np.float64).reshape(-1)
    finite_candidates = np.isfinite(candidate_features).all(axis=tuple(
        range(candidate_features.ndim - 1)))
    return finite_candidates & np.isfinite(obs_features)


def m4_distance(candidate_features, obs_features, mu, sigma, mask, eps=EPS):
    """L1 over the valid features after the FROZEN z-normalization."""
    import numpy as np
    candidate_features = np.asarray(candidate_features, dtype=np.float64)
    obs_features = np.asarray(obs_features, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(candidate_features.shape[:-1], np.nan)
    scale = np.asarray(sigma, dtype=np.float64) + eps
    mu = np.asarray(mu, dtype=np.float64)
    z_candidates = (candidate_features - mu) / scale
    z_obs = (obs_features - mu) / scale
    return np.abs(z_candidates[..., mask] - z_obs[mask]).mean(axis=-1)


# --------------------------------------------------------------------------- #
# aggregation, prediction and the §2 metric-matched retrieval control
# --------------------------------------------------------------------------- #
def aggregate_over_k(distances, how=K_AGGREGATION_PRIMARY, tau=LME_TAU):
    """Collapse ``[M, K]`` to ``[M]``: mean is primary, min/median/LME declared."""
    distances = torch.as_tensor(distances).float()
    if how == "mean":
        return distances.mean(dim=-1)
    if how == "min":
        return distances.min(dim=-1).values
    if how == "median":
        # AMBIGUITY (plan §1 does not fix the even-K convention): use the
        # np.median convention -- the mean of the two middle values -- because
        # that is what exp_18's own scoring.summarize reduces to, so "median"
        # means one thing across the experiment. torch.median would take the
        # LOWER middle value instead, which at K=8 is a different statistic.
        ordered = distances.sort(dim=-1).values
        k = ordered.shape[-1]
        if k % 2:
            return ordered[..., k // 2]
        return 0.5 * (ordered[..., k // 2 - 1] + ordered[..., k // 2])
    if how == "lme":
        # distances are "lower is better", so LME is applied to the negated score
        return -float(tau) * (torch.logsumexp(-distances / float(tau), dim=-1)
                              - torch.log(torch.tensor(float(distances.shape[-1]))))
    raise ValueError(f"unknown K aggregation {how!r}; registered: "
                     f"{K_AGGREGATION_PRIMARY} + {K_AGGREGATION_SECONDARIES}")


def predict_from_distances(scores):
    """argmin with the registered lowest-index tie-break."""
    scores = torch.as_tensor(scores).reshape(-1)
    winners = torch.nonzero(scores == scores.min(), as_tuple=False)
    return int(winners[0].item())


def metric_matched_retrieval(cand_xyz, ctx_xyz, ctx_distances, eligible_mask=None):
    """Plan §2: closest context RIR UNDER THE METRIC, then the nearest candidate.

    Identical geometry to the registered control -- it delegates to
    ``scoring.nearest_context_baseline`` -- with the single difference the plan
    specifies: the context is chosen by the metric's distance (lower is better)
    instead of by AGREE similarity. Negating turns "closest" into the
    "highest similarity" that helper expects, so the tie-break stays the
    registered lowest-index one.
    """
    from src.localization.scoring import nearest_context_baseline

    ctx_distances = torch.as_tensor(ctx_distances).float().reshape(-1)
    return nearest_context_baseline(cand_xyz, ctx_xyz, -ctx_distances,
                                    eligible_mask=eligible_mask)


@dataclass
class MetricConfig:
    """Everything the metric pass is allowed to know.

    ``delta_max`` is the ONE calibrated constant and is passed IN, chosen on the
    R1 seen prefix from :data:`M1_DELTA_GRID`. ``m4_mu``/``m4_sigma`` are the
    frozen z-normalization statistics -- also seen-only. Defaults here are inert
    placeholders (zero mean, unit scale), never a tuned value.
    """
    delta_max: int = 0
    window_samples: int = WINDOW_SAMPLES
    param_window_samples: int = PARAM_WINDOW_SAMPLES
    lam: float = M2_LAMBDA
    t30_backend: str = "pyroomacoustics"
    sample_rate: int = SAMPLE_RATE
    m4_mu: object = None
    m4_sigma: object = None
    families: tuple = ("m1", "m2", "m3", "m4", "m5")
    secondaries: bool = False

    def payload(self):
        """JSON-serializable record of what this pass actually applied."""
        import numpy as np
        return {"delta_max": int(self.delta_max), "window_samples": int(self.window_samples),
                "param_window_samples": int(self.param_window_samples), "lam": float(self.lam),
                "t30_backend": self.t30_backend, "sample_rate": int(self.sample_rate),
                "families": list(self.families), "secondaries": bool(self.secondaries),
                "m4_mu": None if self.m4_mu is None else np.asarray(self.m4_mu).tolist(),
                "m4_sigma": None if self.m4_sigma is None else np.asarray(self.m4_sigma).tolist()}


def compute_metrics(pred, obs, ctx, config):
    """All five families for one query, candidates AND context, in one pass.

    ``pred`` is ``[M, K, T]``, ``obs`` ``[T]``, ``ctx`` ``[N, T]`` -- the
    exactly-as-scored tensors. Returns per-family ``[M, K]`` candidate distances,
    ``[N]`` context distances (for the §2 metric-matched control) and the
    diagnostics the plan asks to record (M4 features + validity mask, M5 lags).

    Both directions go through the SAME function objects with the same window,
    alignment and amplitude policy -- that equality is what makes the
    metric-matched control comparable to the candidate scores.
    """
    import numpy as np

    pred = common_window(pred, config.window_samples)
    obs = common_window(torch.as_tensor(obs).reshape(-1), config.window_samples)
    ctx = common_window(torch.as_tensor(ctx), config.window_samples)
    num_candidates, num_samples = pred.shape[0], pred.shape[1]

    candidates, context, diagnostics = {}, {}, {}
    if "m1" in config.families:
        candidates["m1"] = m1_distance(pred, obs, config.delta_max)
        context["m1"] = m1_distance(ctx, obs, config.delta_max)
    if "m2" in config.families:
        candidates["m2"] = m2_distance(pred, obs, lam=config.lam)
        context["m2"] = m2_distance(ctx, obs, lam=config.lam)
    if "m3" in config.families:
        candidates["m3"] = m3_distance(pred, obs)
        context["m3"] = m3_distance(ctx, obs)
        if config.secondaries:
            diagnostics["m3_band_candidates"] = m3_band_envelope_distance(pred, obs)
            diagnostics["m3_hilbert_candidates"] = m3_hilbert_envelope_distance(pred, obs)
    if "m5" in config.families:
        candidates["m5"], diagnostics["m5_lags"] = m5_distance(pred, obs, config.delta_max)
        context["m5"], diagnostics["m5_context_lags"] = m5_distance(ctx, obs, config.delta_max)
        if config.secondaries:
            diagnostics["m5_gcc_phat_lags"] = gcc_phat_lag(pred, obs, config.delta_max)

    if "m4" in config.families:
        obs_features, _ = m4_feature_vector(obs, config.sample_rate, config.t30_backend)
        candidate_features = np.stack([
            np.stack([m4_feature_vector(pred[m, k], config.sample_rate,
                                        config.t30_backend)[0]
                      for k in range(num_samples)])
            for m in range(num_candidates)])
        context_features = np.stack([m4_feature_vector(ctx[n], config.sample_rate,
                                                       config.t30_backend)[0]
                                     for n in range(ctx.shape[0])])
        # the validity mask is per QUERY: it must see the context rows too, so the
        # candidate and control numbers are computed over the same feature set
        stacked = np.concatenate([candidate_features.reshape(-1, len(M4_FEATURES)),
                                  context_features], axis=0)
        mask = m4_validity_mask(stacked, obs_features)
        mu = np.zeros(len(M4_FEATURES)) if config.m4_mu is None else np.asarray(config.m4_mu)
        sigma = np.ones(len(M4_FEATURES)) if config.m4_sigma is None else np.asarray(
            config.m4_sigma)
        candidates["m4"] = torch.from_numpy(
            m4_distance(candidate_features, obs_features, mu, sigma, mask)).float()
        context["m4"] = torch.from_numpy(
            m4_distance(context_features, obs_features, mu, sigma, mask)).float()
        diagnostics["m4_features"] = candidate_features
        diagnostics["m4_context_features"] = context_features
        diagnostics["m4_obs_features"] = obs_features
        diagnostics["m4_mask"] = mask

    return {"candidates": candidates, "context": context, "diagnostics": diagnostics,
            "config": config.payload()}
