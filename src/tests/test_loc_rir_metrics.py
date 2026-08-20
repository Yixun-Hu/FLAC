"""Tests for ``src.localization.rir_metrics`` (exp_18 R4, round R4-r1).

Contract: ``plan_loc_invert_R4.md`` §1 (the five families), §2 (metric-matched
retrieval control) and §3 (controls). Every constant is a module-level
REGISTERABLE so the metric-registration manifest can freeze it; nothing here may
be tuned on anything but the R1 seen prefix, so the tests pin definitions and
conventions rather than thresholds.
"""
import inspect

import numpy as np
import pytest
import torch

from src.localization import rir_metrics as rm


# --------------------------------------------------------------------------- #
# conventions + the registerable constant set
# --------------------------------------------------------------------------- #
def test_registerable_collects_every_constant_the_manifest_must_freeze():
    reg = rm.REGISTERABLE
    for key in ("sample_rate", "window_samples", "param_window_samples", "eps",
                "m1_delta_grid", "m2_fft_sizes", "m2_stft_eps", "m2_safe_log_eps", "m2_lambda",
                "m3_region_db", "m3_octave_bands_hz", "m4_features", "m4_arrival_threshold_db",
                "m4_direct_half_width_ms", "m4_early_late_ms", "m4_t30_decay_db",
                "m5_secondary", "k_aggregation_primary", "k_aggregation_secondaries",
                "prediction_tie_break"):
        assert key in reg, f"REGISTERABLE is missing {key!r}"
    assert reg["sample_rate"] == 22050
    assert reg["window_samples"] == 9600 and reg["param_window_samples"] == 8000
    assert reg["m1_delta_grid"] == (0, 8, 32, 128)
    assert reg["m2_lambda"] == 1.0
    assert reg["m3_region_db"] == (0.0, -30.0)
    assert reg["k_aggregation_primary"] == "mean"
    assert reg["prediction_tie_break"] == "lowest_index"
    # a frozen manifest must be able to serialize it
    import json
    assert json.loads(json.dumps(rm.registerable_payload()))["m2_lambda"] == 1.0


def test_m2_scale_set_is_pinned_to_the_repo_function():
    """The scale set is the repo's, not ours: read it out of
    ``multiscale_log_l1``'s source so a change there fails here."""
    from src.metrics.modules import l1_stft_multires
    source = inspect.getsource(l1_stft_multires.multiscale_log_l1)
    repo_sizes = tuple(int(n) for n in
                       __import__("re").findall(r"n_fft=(\d+)", source))
    assert rm.M2_FFT_SIZES == repo_sizes
    assert rm.M2_SAFE_LOG_EPS == inspect.signature(
        l1_stft_multires.safe_log).parameters["eps"].default
    assert rm.M2_STFT_EPS == inspect.signature(
        l1_stft_multires.log_L1_STFT).parameters["eps"].default


@pytest.mark.parametrize("length,expected", [(100, 9600), (9600, 9600), (20000, 9600)])
def test_common_window_crops_or_zero_pads(length, expected):
    x = torch.arange(length, dtype=torch.float32)
    out = rm.common_window(x)
    assert out.shape[-1] == expected
    if length < expected:
        assert torch.all(out[length:] == 0.0)
    assert torch.equal(out[: min(length, expected)], x[: min(length, expected)])


def test_common_window_is_batched_and_dtype_stable():
    x = torch.rand(3, 4, 12000, dtype=torch.float64)
    out = rm.common_window(x)
    assert out.shape == (3, 4, 9600) and out.dtype == torch.float32


def test_param_window_uses_the_repo_acoustic_convention():
    x = torch.rand(9600)
    assert rm.param_window(x).shape[-1] == 8000 == rm.PARAM_WINDOW_SAMPLES


# --------------------------------------------------------------------------- #
# shared lag machinery (M1 + M5): zero-padded shifts, no wraparound
# --------------------------------------------------------------------------- #
def _naive_shift(x, delta):
    """Reference: ``out[t] = x[t - delta]``, zero outside."""
    out = torch.zeros_like(x)
    if delta >= 0:
        if delta < x.shape[-1]:
            out[..., delta:] = x[..., : x.shape[-1] - delta]
    else:
        out[..., : x.shape[-1] + delta] = x[..., -delta:]
    return out


def test_shift_matches_the_naive_definition_and_never_wraps():
    x = torch.arange(8, dtype=torch.float32)
    for delta in (-3, -1, 0, 1, 5, 9):
        assert torch.equal(rm.shift(x, delta), _naive_shift(x, delta))
    assert torch.all(rm.shift(x, 8) == 0)                    # shifted entirely out


def test_lag_products_match_a_naive_loop():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 64, generator=g)
    y = torch.randn(64, generator=g)
    dots, energies = rm.lag_products(x, y, delta_max=7)
    assert dots.shape == (3, 15) and energies.shape == (3, 15)
    for i, delta in enumerate(range(-7, 8)):
        shifted = _naive_shift(x, delta)
        assert torch.allclose(dots[:, i].float(), (shifted * y).sum(-1), atol=1e-4)
        assert torch.allclose(energies[:, i].float(), (shifted ** 2).sum(-1), atol=1e-4)


# --------------------------------------------------------------------------- #
# M1 — aligned, scale-invariant residual
# --------------------------------------------------------------------------- #
def test_m1_is_zero_for_a_scaled_and_shifted_copy_within_delta_max():
    g = torch.Generator().manual_seed(1)
    # silent head AND tail: a zero-padded shift then loses no energy, so the
    # analytic case is exactly 0 (a shift that drops real samples cannot be)
    obs = torch.randn(2048, generator=g)
    obs[:64] = 0.0
    obs[-64:] = 0.0
    for delta, gain in ((0, 1.0), (5, 3.0), (-4, -0.5), (8, 1e-3)):
        pred = _naive_shift(obs, delta) * gain
        d = rm.m1_distance(pred.unsqueeze(0), obs, delta_max=8)
        # the plan's eps guard in alpha* biases a VANISHING-gain copy slightly, so
        # the tolerance is loosened for the 1e-3 case only
        tol = 1e-6 if gain > 1e-2 else 1e-4
        assert float(d[0]) == pytest.approx(0.0, abs=tol), (delta, gain)


def test_m1_is_positive_when_the_shift_exceeds_delta_max():
    g = torch.Generator().manual_seed(2)
    obs = torch.randn(2048, generator=g)
    obs[:256] = 0.0
    obs[-256:] = 0.0
    pred = _naive_shift(obs, 40)
    assert float(rm.m1_distance(pred.unsqueeze(0), obs, delta_max=8)[0]) > 0.1
    assert float(rm.m1_distance(pred.unsqueeze(0), obs, delta_max=128)[0]) == pytest.approx(
        0.0, abs=1e-6)


def test_m1_equals_one_minus_max_squared_correlation():
    g = torch.Generator().manual_seed(3)
    obs = torch.randn(512, generator=g)
    pred = torch.randn(4, 512, generator=g)
    d = rm.m1_distance(pred, obs, delta_max=8)
    best = torch.zeros(4)
    for delta in range(-8, 9):
        shifted = _naive_shift(pred, delta)
        rho2 = (shifted * obs).sum(-1) ** 2 / ((shifted ** 2).sum(-1) * (obs ** 2).sum(-1))
        best = torch.maximum(best, rho2)
    assert torch.allclose(d, 1.0 - best, atol=1e-5)


def test_m1_delta_zero_is_the_no_alignment_sensitivity_row():
    g = torch.Generator().manual_seed(4)
    obs = torch.randn(1024, generator=g)
    obs[:64] = 0.0
    obs[-64:] = 0.0
    pred = _naive_shift(obs, 3).unsqueeze(0)
    assert float(rm.m1_distance(pred, obs, delta_max=0)[0]) > 0.0
    assert float(rm.m1_distance(pred, obs, delta_max=8)[0]) == pytest.approx(0.0, abs=1e-6)


def test_m1_is_batch_invariant_and_deterministic():
    g = torch.Generator().manual_seed(5)
    obs = torch.randn(1024, generator=g)
    pred = torch.randn(6, 1024, generator=g)
    whole = rm.m1_distance(pred, obs, delta_max=8)
    piecewise = torch.cat([rm.m1_distance(pred[i:i + 1], obs, delta_max=8) for i in range(6)])
    assert torch.allclose(whole, piecewise, atol=1e-6)
    assert torch.equal(whole, rm.m1_distance(pred, obs, delta_max=8))


def test_m1_handles_a_silent_prediction_without_nan():
    obs = torch.randn(256, generator=torch.Generator().manual_seed(6))
    d = rm.m1_distance(torch.zeros(1, 256), obs, delta_max=8)
    assert torch.isfinite(d).all() and float(d[0]) == pytest.approx(1.0, abs=1e-6)


def test_m1_rejects_a_delta_max_outside_the_registered_grid():
    obs = torch.randn(128)
    for off_grid in (7, 4, 64, 200):
        with pytest.raises(ValueError):
            rm.m1_distance(torch.randn(1, 128), obs, delta_max=off_grid)


# --------------------------------------------------------------------------- #
# M5 — normalized cross-correlation (same lag bound and pad convention as M1)
# --------------------------------------------------------------------------- #
def test_m5_recovers_the_peak_lag_and_scores_zero_for_a_shifted_copy():
    g = torch.Generator().manual_seed(7)
    obs = torch.randn(2048, generator=g)
    obs[:64] = 0.0
    obs[-64:] = 0.0
    for delta in (-6, 0, 5):
        pred = _naive_shift(obs, delta).unsqueeze(0)
        d, lag = rm.m5_distance(pred, obs, delta_max=8)
        assert float(d[0]) == pytest.approx(0.0, abs=1e-6)
        # the recorded lag is the shift applied to the PREDICTION to align it to
        # obs, i.e. the negative of the shift that produced pred
        assert int(lag[0]) == -delta


def test_m5_is_gain_invariant_but_sign_sensitive():
    g = torch.Generator().manual_seed(8)
    obs = torch.randn(512, generator=g)
    positive, _ = rm.m5_distance((3.0 * obs).unsqueeze(0), obs, delta_max=8)
    negative, _ = rm.m5_distance((-3.0 * obs).unsqueeze(0), obs, delta_max=8)
    assert float(positive[0]) == pytest.approx(0.0, abs=1e-6)
    # a sign-flipped copy cannot score 0: its best NCC over the bound is well below 1
    assert float(negative[0]) > 0.9


def test_m5_relation_to_m1_is_disclosed_and_holds_numerically():
    """M1 = 1 - max rho^2 (gain-squared), M5 = 1 - max rho: both pre-declared."""
    g = torch.Generator().manual_seed(9)
    obs = torch.randn(512, generator=g)
    pred = torch.randn(3, 512, generator=g)
    m1 = rm.m1_distance(pred, obs, delta_max=8)
    m5, _ = rm.m5_distance(pred, obs, delta_max=8)
    rho_max = 1.0 - m5
    assert torch.all(m1 <= 1.0 - rho_max ** 2 + 1e-5)      # rho^2 max >= (max rho)^2


def test_m5_gcc_phat_secondary_recovers_the_lag():
    g = torch.Generator().manual_seed(10)
    obs = torch.randn(1024, generator=g)
    obs[:64] = 0.0
    obs[-64:] = 0.0
    pred = _naive_shift(obs, 7).unsqueeze(0)
    lag = rm.gcc_phat_lag(pred, obs, delta_max=32)
    assert int(lag[0]) == -7           # shift applied to pred, as in m5_distance


def test_m5_never_touches_the_global_rng():
    before = torch.random.get_rng_state()
    obs = torch.randn(256, generator=torch.Generator().manual_seed(11))
    rm.m5_distance(torch.randn(2, 256, generator=torch.Generator().manual_seed(12)), obs,
                   delta_max=8)
    rm.m1_distance(torch.randn(2, 256, generator=torch.Generator().manual_seed(13)), obs,
                   delta_max=8)
    assert torch.equal(torch.random.get_rng_state(), before)


# --------------------------------------------------------------------------- #
# M2 — multi-resolution STFT distance (repo scale set + spectral convergence)
# --------------------------------------------------------------------------- #
def test_m2_is_zero_for_an_identical_signal_and_positive_otherwise():
    g = torch.Generator().manual_seed(20)
    obs = torch.randn(rm.WINDOW_SAMPLES, generator=g)
    same = rm.m2_distance(obs.unsqueeze(0), obs)
    assert float(same[0]) == pytest.approx(0.0, abs=1e-6)
    other = rm.m2_distance(torch.randn(1, rm.WINDOW_SAMPLES, generator=g), obs)
    assert float(other[0]) > 0.0


def test_m2_uses_raw_amplitudes_with_no_per_pair_gain():
    """Amplitude policy is FIXED (plan §1): the gain question lives in M1, so a
    scaled copy is NOT free under M2."""
    g = torch.Generator().manual_seed(21)
    obs = torch.randn(rm.WINDOW_SAMPLES, generator=g)
    assert float(rm.m2_distance((3.0 * obs).unsqueeze(0), obs)[0]) > 0.1


def test_m2_sums_the_repo_scale_set_and_adds_spectral_convergence():
    g = torch.Generator().manual_seed(22)
    obs = torch.randn(rm.WINDOW_SAMPLES, generator=g)
    pred = torch.randn(rm.WINDOW_SAMPLES, generator=g)
    parts = rm.m2_terms(pred.unsqueeze(0), obs)
    assert sorted(parts) == sorted(rm.M2_FFT_SIZES)
    total = sum(float(parts[n]["log_l1"][0]) + rm.M2_LAMBDA * float(parts[n]["convergence"][0])
                for n in rm.M2_FFT_SIZES)
    assert float(rm.m2_distance(pred.unsqueeze(0), obs)[0]) == pytest.approx(total, rel=1e-5)


def test_m2_log_term_matches_the_repo_formula_on_cpu():
    """Same amplitude/eps/safe_log convention as log_L1_STFT; the only deviation
    is the window device (the repo's get_stft hardcodes .cuda())."""
    from src.metrics.modules.l1_stft_multires import safe_log as repo_safe_log
    g = torch.Generator().manual_seed(23)
    obs = torch.randn(rm.WINDOW_SAMPLES, generator=g)
    pred = torch.randn(rm.WINDOW_SAMPLES, generator=g)
    n_fft = 256
    window = torch.hann_window(n_fft)
    est = torch.stft(pred, n_fft=n_fft, hop_length=None, window=window, return_complex=False)
    ref = torch.stft(obs, n_fft=n_fft, hop_length=None, window=window, return_complex=False)
    est_amp = torch.sqrt(est[..., 0] ** 2 + est[..., 1] ** 2 + rm.M2_STFT_EPS)
    ref_amp = torch.sqrt(ref[..., 0] ** 2 + ref[..., 1] ** 2 + rm.M2_STFT_EPS)
    expected = torch.mean(torch.abs(repo_safe_log(est_amp) - repo_safe_log(ref_amp)))
    got = rm.m2_terms(pred.unsqueeze(0), obs)[n_fft]["log_l1"][0]
    assert float(got) == pytest.approx(float(expected), rel=1e-5)


def test_m2_is_batch_invariant():
    g = torch.Generator().manual_seed(24)
    obs = torch.randn(rm.WINDOW_SAMPLES, generator=g)
    pred = torch.randn(5, rm.WINDOW_SAMPLES, generator=g)
    whole = rm.m2_distance(pred, obs)
    piecewise = torch.cat([rm.m2_distance(pred[i:i + 1], obs) for i in range(5)])
    assert torch.allclose(whole, piecewise, atol=1e-5)


# --------------------------------------------------------------------------- #
# M3 — energy-decay distance
# --------------------------------------------------------------------------- #
def _exponential_rir(length, tau_samples, seed=0):
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(length, generator=g)
    envelope = torch.exp(-torch.arange(length, dtype=torch.float32) / tau_samples)
    return noise * envelope


def test_m3_edc_of_an_exponential_decay_is_linear_in_db():
    rir = _exponential_rir(8192, 1200, seed=30)
    edc = rm.schroeder_edc(rir.unsqueeze(0))[0]
    assert float(edc[0]) == pytest.approx(0.0, abs=1e-6)          # normalized to 0 dB
    assert torch.all(edc[1:] <= edc[:-1] + 1e-6)                  # monotonically decaying
    early = edc[:4000]
    slope = (early[-1] - early[0]) / 4000.0
    fitted = early[0] + slope * torch.arange(4000, dtype=torch.float32)
    assert float(torch.abs(early - fitted).mean()) < 1.5          # near-linear in dB


def test_m3_is_zero_for_the_same_decay_shape_and_amplitude_blind():
    rir = _exponential_rir(8192, 1200, seed=31)
    d_same = rm.m3_distance(rir.unsqueeze(0), rir)
    assert float(d_same[0]) == pytest.approx(0.0, abs=1e-6)
    # amplitude is M1's job: a globally scaled copy has the SAME decay shape
    d_scaled = rm.m3_distance((7.0 * rir).unsqueeze(0), rir)
    assert float(d_scaled[0]) == pytest.approx(0.0, abs=1e-5)


def test_m3_separates_different_decay_rates():
    obs = _exponential_rir(8192, 1200, seed=32)
    near = _exponential_rir(8192, 1250, seed=33)
    far = _exponential_rir(8192, 400, seed=34)
    d_near = float(rm.m3_distance(near.unsqueeze(0), obs)[0])
    d_far = float(rm.m3_distance(far.unsqueeze(0), obs)[0])
    assert 0.0 < d_near < d_far


def test_m3_region_is_defined_by_the_observation_only():
    """All candidates of a query share the region, so a candidate cannot move it."""
    obs = _exponential_rir(8192, 1200, seed=35)
    mask = rm.m3_region_mask(obs)
    assert mask.dtype == torch.bool and int(mask.sum()) > 100
    edc = rm.schroeder_edc(obs.unsqueeze(0))[0]
    inside = edc[mask]
    assert float(inside.max()) <= rm.M3_REGION_DB[0] + 1e-6
    assert float(inside.min()) >= rm.M3_REGION_DB[1] - 1e-6
    # the same mask is used for every candidate
    preds = torch.stack([_exponential_rir(8192, t, seed=36) for t in (300, 1200, 3000)])
    d_batch = rm.m3_distance(preds, obs)
    d_single = torch.cat([rm.m3_distance(preds[i:i + 1], obs) for i in range(3)])
    assert torch.allclose(d_batch, d_single, atol=1e-6)


def test_m3_secondaries_are_available_and_finite():
    obs = _exponential_rir(8192, 1200, seed=37)
    pred = _exponential_rir(8192, 900, seed=38).unsqueeze(0)
    band = rm.m3_band_envelope_distance(pred, obs)
    hilbert = rm.m3_hilbert_envelope_distance(pred, obs)
    assert band.shape == (1,) and hilbert.shape == (1,)
    assert torch.isfinite(band).all() and torch.isfinite(hilbert).all()
    assert float(rm.m3_band_envelope_distance(obs.unsqueeze(0), obs)[0]) == pytest.approx(
        0.0, abs=1e-6)
    assert float(rm.m3_hilbert_envelope_distance(obs.unsqueeze(0), obs)[0]) == pytest.approx(
        0.0, abs=1e-6)


def test_m3_handles_a_silent_prediction_without_nan():
    obs = _exponential_rir(4096, 800, seed=39)
    d = rm.m3_distance(torch.zeros(1, 4096), obs)
    assert torch.isfinite(d).all() and float(d[0]) > 0.0


# --------------------------------------------------------------------------- #
# M4 — acoustic-parameter distance (repo estimators; CPU)
# --------------------------------------------------------------------------- #
def _two_impulse_rir(length=8000, direct_at=100, reflection_at=1000, direct_gain=1.0,
                     reflection_gain=0.3, tail_tau=800, seed=40):
    """Direct impulse + one reflection + an exponential noise tail: C50, DRR and
    the arrival time are known by construction."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(length)
    x[direct_at] = direct_gain
    x[reflection_at] = reflection_gain
    tail = torch.randn(length, generator=g) * 0.01
    tail[:reflection_at] = 0.0
    x = x + tail * torch.exp(-torch.arange(length, dtype=torch.float32) / tail_tau)
    return x


def test_m4_arrival_time_is_the_first_minus_20db_crossing():
    rir = _two_impulse_rir(direct_at=250)
    features = rm.m4_features(rir)
    assert features["arrival_time"] == pytest.approx(250 / rm.SAMPLE_RATE, abs=1e-6)


def test_m4_drr_uses_a_2p5ms_direct_window_and_is_higher_for_a_dominant_direct_path():
    strong = rm.m4_features(_two_impulse_rir(direct_gain=1.0, reflection_gain=0.05))
    weak = rm.m4_features(_two_impulse_rir(direct_gain=1.0, reflection_gain=0.9))
    assert strong["drr"] > weak["drr"]
    assert np.isfinite(strong["drr"])


def test_m4_c50_matches_the_repo_estimator():
    from src.metrics.modules.C50 import _measure_clarity
    rir = _two_impulse_rir()
    expected = float(_measure_clarity(rm.param_window(rir).reshape(1, 1, -1),
                                      time=50, fs=rm.SAMPLE_RATE)[0, 0])
    assert rm.m4_features(rir)["c50"] == pytest.approx(expected, rel=1e-6)


def test_m4_edt_and_t30_use_the_repo_estimators():
    from src.metrics.modules.EDT import _edt
    rir = _two_impulse_rir(tail_tau=1500)
    windowed = rm.param_window(rir).numpy()
    assert rm.m4_features(rir)["edt"] == pytest.approx(
        float(_edt(windowed, fs=rm.SAMPLE_RATE, decay_db=rm.M4_EDT_DECAY_DB)), rel=1e-6)
    torch_t30 = rm.m4_features(rir, t30_backend="torch")["t30"]
    assert np.isfinite(torch_t30) and torch_t30 > 0.0


def test_m4_feature_vector_has_the_registered_order_and_length():
    vec, names = rm.m4_feature_vector(_two_impulse_rir())
    assert names == rm.M4_FEATURES and vec.shape == (len(rm.M4_FEATURES),)


def test_m4_validity_mask_is_per_query_and_uniform_across_candidates():
    """Plan §1: a feature is kept iff finite for obs AND every candidate-sample of
    the query -- never dropped per candidate."""
    obs = np.array([1.0, 2.0, 3.0])
    feats = np.array([[[1.0, 2.0, 3.0], [1.0, np.nan, 3.0]],
                      [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])          # [M=2, K=2, F=3]
    mask = rm.m4_validity_mask(feats, obs)
    assert mask.tolist() == [True, False, True]                      # feature 1 dropped for ALL
    assert rm.m4_validity_mask(feats, np.array([1.0, 2.0, np.inf])).tolist() == \
        [True, False, False]


def test_m4_distance_is_l1_over_the_valid_features_after_z_norm():
    obs = np.array([0.0, 10.0, 100.0])
    feats = np.array([[[1.0, 12.0, 100.0]]])                         # [1, 1, 3]
    mu, sigma = np.array([0.0, 10.0, 0.0]), np.array([1.0, 2.0, 50.0])
    mask = np.array([True, True, False])
    d = rm.m4_distance(feats, obs, mu=mu, sigma=sigma, mask=mask)
    assert d.shape == (1, 1)
    assert float(d[0, 0]) == pytest.approx((abs(1.0 - 0.0) / 1.0 + abs(12.0 - 10.0) / 2.0) / 2)


def test_m4_distance_is_zero_for_the_observation_itself():
    obs = np.array([1.0, 2.0, 3.0])
    feats = obs.reshape(1, 1, 3)
    mu, sigma = np.zeros(3), np.ones(3)
    d = rm.m4_distance(feats, obs, mu=mu, sigma=sigma, mask=np.ones(3, dtype=bool))
    assert float(d[0, 0]) == pytest.approx(0.0)


def test_m4_features_are_finite_on_a_realistic_rir_and_deterministic():
    rir = _two_impulse_rir(tail_tau=1500)
    first, _ = rm.m4_feature_vector(rir, t30_backend="torch")
    second, _ = rm.m4_feature_vector(rir, t30_backend="torch")
    assert np.array_equal(first, second)
    assert np.isfinite(first[:7]).all()          # the non-band features


# --------------------------------------------------------------------------- #
# compute_metrics — the driver-facing entry point
# --------------------------------------------------------------------------- #
def _query_tensors(m=3, k=2, length=None, seed=50):
    length = length or rm.WINDOW_SAMPLES
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(length, generator=g) * 0.1
    pred = torch.randn(m, k, length, generator=g) * 0.1
    pred[1, 0] = obs                                   # one exact hit
    ctx = torch.randn(4, length, generator=g) * 0.1
    return pred, obs, ctx


def test_compute_metrics_returns_every_family_for_candidates_and_context():
    pred, obs, ctx = _query_tensors()
    out = rm.compute_metrics(pred, obs, ctx, config=rm.MetricConfig(delta_max=8,
                                                                    t30_backend="torch"))
    for family in ("m1", "m2", "m3", "m5"):
        assert out["candidates"][family].shape == (3, 2)
        assert out["context"][family].shape == (4,)
        assert torch.isfinite(out["candidates"][family]).all()
    assert out["candidates"]["m4"].shape == (3, 2) and out["context"]["m4"].shape == (4,)
    assert out["diagnostics"]["m4_features"].shape == (3, 2, len(rm.M4_FEATURES))
    assert out["diagnostics"]["m4_mask"].shape == (len(rm.M4_FEATURES),)
    assert out["diagnostics"]["m5_lags"].shape == (3, 2)
    assert out["config"]["delta_max"] == 8


def test_compute_metrics_scores_an_exact_copy_best_on_every_waveform_family():
    pred, obs, ctx = _query_tensors()
    out = rm.compute_metrics(pred, obs, ctx, config=rm.MetricConfig(delta_max=8,
                                                                    t30_backend="torch"))
    for family in ("m1", "m2", "m3", "m5"):
        distances = out["candidates"][family]
        assert float(distances[1, 0]) == pytest.approx(float(distances.min()), abs=1e-6)


def test_compute_metrics_uses_the_same_function_objects_for_pred_and_context():
    """Preprocessing equality (plan §1): both directions go through the same
    windowing and the same distance callables."""
    pred, obs, ctx = _query_tensors()
    config = rm.MetricConfig(delta_max=8, t30_backend="torch")
    out = rm.compute_metrics(pred, obs, ctx, config=config)
    # feeding a context RIR as a candidate must reproduce its context distance
    as_candidate = rm.compute_metrics(ctx[2].reshape(1, 1, -1), obs, ctx, config=config)
    for family in ("m1", "m2", "m3", "m5"):
        assert float(as_candidate["candidates"][family][0, 0]) == pytest.approx(
            float(out["context"][family][2]), abs=1e-6)


def test_compute_metrics_windows_every_input_identically():
    pred, obs, ctx = _query_tensors(length=12000)
    out = rm.compute_metrics(pred, obs, ctx, config=rm.MetricConfig(delta_max=8,
                                                                    t30_backend="torch"))
    cropped = rm.compute_metrics(pred[..., :rm.WINDOW_SAMPLES], obs[:rm.WINDOW_SAMPLES],
                                 ctx[:, :rm.WINDOW_SAMPLES],
                                 config=rm.MetricConfig(delta_max=8, t30_backend="torch"))
    for family in ("m1", "m2", "m3", "m5"):
        assert torch.allclose(out["candidates"][family], cropped["candidates"][family],
                              atol=1e-6)


def test_compute_metrics_does_not_touch_the_global_rng():
    pred, obs, ctx = _query_tensors()
    before = torch.random.get_rng_state()
    rm.compute_metrics(pred, obs, ctx, config=rm.MetricConfig(delta_max=8, t30_backend="torch"))
    assert torch.equal(torch.random.get_rng_state(), before)


def test_aggregate_over_k_defaults_to_the_mean_with_declared_secondaries():
    distances = torch.tensor([[1.0, 3.0], [0.5, 0.5]])
    assert torch.allclose(rm.aggregate_over_k(distances), torch.tensor([2.0, 0.5]))
    assert torch.allclose(rm.aggregate_over_k(distances, "min"), torch.tensor([1.0, 0.5]))
    assert torch.allclose(rm.aggregate_over_k(distances, "median"), torch.tensor([2.0, 0.5]))
    assert torch.isfinite(rm.aggregate_over_k(distances, "lme")).all()
    with pytest.raises(ValueError):
        rm.aggregate_over_k(distances, "max")          # not a declared secondary


def test_predict_from_distances_is_argmin_with_lowest_index_tie_break():
    assert rm.predict_from_distances(torch.tensor([0.5, 0.2, 0.2])) == 1
    assert rm.predict_from_distances(torch.tensor([0.2, 0.2, 0.2])) == 0


# --------------------------------------------------------------------------- #
# §2 metric-matched retrieval control
# --------------------------------------------------------------------------- #
def test_metric_matched_retrieval_picks_the_candidate_nearest_the_closest_context():
    cand_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    ctx_xyz = np.array([[2.9, 0.0, 0.0], [0.1, 0.0, 0.0]])
    ctx_distances = torch.tensor([0.9, 0.2])            # context 1 is the closest under m
    assert rm.metric_matched_retrieval(cand_xyz, ctx_xyz, ctx_distances) == 0
    assert rm.metric_matched_retrieval(cand_xyz, ctx_xyz, torch.tensor([0.2, 0.9])) == 2


def test_metric_matched_retrieval_has_a_masked_variant_and_reuses_scoring_geometry():
    from src.localization.scoring import nearest_context_baseline
    cand_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    ctx_xyz = np.array([[2.9, 0.0, 0.0], [0.1, 0.0, 0.0]])
    ctx_distances = torch.tensor([0.9, 0.2])
    raw = rm.metric_matched_retrieval(cand_xyz, ctx_xyz, ctx_distances)
    masked = rm.metric_matched_retrieval(cand_xyz, ctx_xyz, ctx_distances,
                                         eligible_mask=[False, True, True])
    # identical geometry to the registered control, only the context CHOICE differs
    assert raw == nearest_context_baseline(cand_xyz, ctx_xyz, torch.tensor([-0.9, -0.2]))
    assert masked == 1


def test_metric_matched_retrieval_refuses_bad_inputs():
    cand_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ctx_xyz = np.array([[0.5, 0.0, 0.0]])
    with pytest.raises(ValueError):
        rm.metric_matched_retrieval(cand_xyz, ctx_xyz, torch.tensor([0.1, 0.2]))
    with pytest.raises(ValueError):
        rm.metric_matched_retrieval(cand_xyz, ctx_xyz, torch.tensor([0.1]),
                                    eligible_mask=[False, False])


# --------------------------------------------------------------------------- #
# R4-r2 (Planner ruling on flag 4): the uniform-drop rule STAYS, plus a per-query
# diagnostic saying how many features it removed and who caused each drop --
# reporting, so the analysis can quantify whether it ever mattered.
# --------------------------------------------------------------------------- #
def test_m4_dropped_diagnostic_names_the_causing_candidate():
    obs = np.array([1.0, 2.0, 3.0])
    feats = np.array([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                      [[1.0, np.nan, 3.0], [1.0, 2.0, 3.0]]])          # [M=2, K=2, F=3]
    diag = rm.m4_dropped_diagnostic(feats, obs, names=("a", "b", "c"))
    assert diag["n_dropped"] == 1 and diag["dropped"] == ["b"]
    assert diag["causes"]["b"]["candidate"] == [1, 0]                   # (m, k)
    assert diag["causes"]["b"]["obs_invalid"] is False
    assert diag["n_features"] == 3 and diag["n_kept"] == 2


def test_m4_dropped_diagnostic_reports_an_invalid_observation_and_context():
    obs = np.array([1.0, np.inf, 3.0])
    feats = np.ones((1, 1, 3))
    ctx = np.array([[1.0, 2.0, np.nan]])
    diag = rm.m4_dropped_diagnostic(feats, obs, context_features=ctx, names=("a", "b", "c"))
    assert sorted(diag["dropped"]) == ["b", "c"]
    assert diag["causes"]["b"]["obs_invalid"] is True
    assert diag["causes"]["c"]["context"] == 0
    assert diag["n_dropped"] == 2


def test_m4_dropped_diagnostic_is_empty_on_a_healthy_query():
    diag = rm.m4_dropped_diagnostic(np.ones((2, 2, 3)), np.ones(3), names=("a", "b", "c"))
    assert diag["n_dropped"] == 0 and diag["dropped"] == [] and diag["causes"] == {}


def test_compute_metrics_reports_the_dropped_feature_diagnostic():
    pred, obs, ctx = _query_tensors()
    out = rm.compute_metrics(pred, obs, ctx, config=rm.MetricConfig(delta_max=8,
                                                                    t30_backend="torch"))
    diag = out["diagnostics"]["m4_dropped"]
    assert set(diag) >= {"n_dropped", "dropped", "causes", "n_features", "n_kept"}
    assert diag["n_features"] == len(rm.M4_FEATURES)
    assert diag["n_kept"] == int(out["diagnostics"]["m4_mask"].sum())
