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
