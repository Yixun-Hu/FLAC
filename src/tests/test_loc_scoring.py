"""Tests for ``src.localization.scoring`` (exp_18 loc_invert, round 1).

Written test-first (announcement 02). Contracts: ``loc_invert_impl_contracts.md``
§4.3 plus the Rev 3 §4 deltas -- LME aggregation through ``torch.logsumexp``
(tau = 0.02 stability), the information-matched context-conditioned baseline with
its explicit GT-only edge case, the non-generative nearest-context control, the
room-clustered (paired) bootstrap statistics, the wiring power statistic and the
cross-platform-stable noise key.
"""
import math

import pytest
import torch

from src.localization.scoring import (
    aggregate,
    cosine_sims,
    localization_error,
    predict_index,
    softmax_map,
    success_within,
)


def _unit(*values):
    v = torch.tensor(values, dtype=torch.float32)
    return v / v.norm()


# --------------------------------------------------------------------------- #
# cosine_sims
# --------------------------------------------------------------------------- #
def test_cosine_sims_hand_built_vectors():
    obs = _unit(1.0, 0.0, 0.0)
    gen = torch.stack([
        torch.stack([_unit(1.0, 0.0, 0.0), _unit(-1.0, 0.0, 0.0)]),
        torch.stack([_unit(1.0, 1.0, 0.0), _unit(0.0, 1.0, 0.0)]),
    ])                                                    # [M=2, K=2, D=3]
    sims = cosine_sims(obs, gen)
    assert tuple(sims.shape) == (2, 2)
    expected = torch.tensor([[1.0, -1.0], [1.0 / math.sqrt(2.0), 0.0]])
    assert torch.allclose(sims, expected, atol=1e-6)


def test_cosine_sims_rejects_unnormalized_obs():
    gen = torch.stack([torch.stack([_unit(1.0, 0.0, 0.0)])])
    with pytest.raises(ValueError):
        cosine_sims(torch.tensor([1.0, 1.0, 0.0]), gen)


def test_cosine_sims_rejects_unnormalized_gen():
    obs = _unit(1.0, 0.0, 0.0)
    gen = torch.stack([torch.stack([torch.tensor([0.5, 0.0, 0.0])])])
    with pytest.raises(ValueError):
        cosine_sims(obs, gen)


def test_cosine_sims_norm_guard_tolerance_is_1e_4():
    """Deviations within 1e-4 of unit norm pass; a 1e-3 deviation does not."""
    obs = _unit(1.0, 0.0, 0.0)
    gen_ok = (torch.stack([torch.stack([_unit(1.0, 0.0, 0.0)])]) * (1.0 + 5e-5))
    cosine_sims(obs, gen_ok)
    gen_bad = (torch.stack([torch.stack([_unit(1.0, 0.0, 0.0)])]) * (1.0 + 1e-3))
    with pytest.raises(ValueError):
        cosine_sims(obs, gen_bad)


def test_cosine_sims_rejects_bad_shapes():
    obs = _unit(1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        cosine_sims(obs, _unit(1.0, 0.0, 0.0))                       # gen not [M,K,D]
    with pytest.raises(ValueError):
        cosine_sims(obs.unsqueeze(0), torch.stack([torch.stack([obs])]))   # obs not [D]
    with pytest.raises(ValueError):
        cosine_sims(obs, torch.stack([torch.stack([_unit(1.0, 0.0)])]))    # D mismatch


# --------------------------------------------------------------------------- #
# aggregate -- LME via torch.logsumexp (O18)
# --------------------------------------------------------------------------- #
_SIMS = torch.tensor([[0.9, 0.1, 0.5], [-0.2, -0.4, 0.0]])


def test_aggregate_mean_and_max():
    assert torch.allclose(aggregate(_SIMS, "mean"), _SIMS.mean(dim=-1))
    assert torch.allclose(aggregate(_SIMS, "max"), _SIMS.max(dim=-1).values)


def test_aggregate_lme_tends_to_max_as_tau_goes_to_zero():
    got = aggregate(_SIMS, "lme", tau=1e-3)
    assert torch.allclose(got, _SIMS.max(dim=-1).values, atol=1e-2)


def test_aggregate_lme_tends_to_mean_as_tau_grows():
    """float64 so the tau * (logsumexp - log K) cancellation is not the limit."""
    sims = _SIMS.double()
    got = aggregate(sims, "lme", tau=1e6)
    assert torch.allclose(got, sims.mean(dim=-1), atol=1e-6)


def test_aggregate_lme_k1_equals_the_single_sample():
    sims = torch.tensor([[0.3], [-0.7]])
    assert torch.allclose(aggregate(sims, "lme", tau=0.02), sims[:, 0], atol=1e-6)


def test_aggregate_lme_is_between_mean_and_max():
    got = aggregate(_SIMS, "lme", tau=0.05)
    assert torch.all(got >= _SIMS.mean(dim=-1) - 1e-6)
    assert torch.all(got <= _SIMS.max(dim=-1).values + 1e-6)


def test_aggregate_lme_tau_002_is_numerically_stable():
    """tau = 0.02 divides by 50; the naive sum-of-exp overflows where logsumexp
    does not. Registered scale (cosines) plus an extreme case, both finite."""
    extreme = torch.tensor([[100.0, 99.0, 98.0]])
    assert not torch.isfinite(torch.exp(extreme / 0.02).sum())
    got = aggregate(extreme, "lme", tau=0.02)
    assert torch.isfinite(got).all()
    expected = 100.0 + 0.02 * math.log(
        1.0 + math.exp(-50.0) + math.exp(-100.0)) - 0.02 * math.log(3.0)
    assert abs(float(got[0]) - expected) < 1e-4

    cos_scale = torch.tensor([[0.98, 0.10, -0.30]])
    ref = 0.02 * (torch.logsumexp(cos_scale.double() / 0.02, dim=-1) - math.log(3.0))
    assert torch.allclose(aggregate(cos_scale, "lme", tau=0.02).double(), ref, atol=1e-6)


@pytest.mark.parametrize("tau", [0.0, -0.02])
def test_aggregate_lme_rejects_nonpositive_tau(tau):
    with pytest.raises(ValueError):
        aggregate(_SIMS, "lme", tau=tau)


def test_aggregate_lme_requires_tau():
    with pytest.raises(ValueError):
        aggregate(_SIMS, "lme")


def test_aggregate_rejects_unknown_method_and_bad_shape():
    with pytest.raises(ValueError):
        aggregate(_SIMS, "median", tau=0.02)
    with pytest.raises(ValueError):
        aggregate(torch.tensor([0.1, 0.2]), "mean")


def test_aggregate_preserves_dtype():
    assert aggregate(_SIMS, "lme", tau=0.05).dtype == torch.float32
    assert aggregate(_SIMS.double(), "lme", tau=0.05).dtype == torch.float64


# --------------------------------------------------------------------------- #
# predict_index / softmax_map
# --------------------------------------------------------------------------- #
def test_predict_index_argmax():
    assert predict_index(torch.tensor([0.1, 0.9, 0.5])) == 1
    assert isinstance(predict_index(torch.tensor([0.1, 0.9])), int)


def test_predict_index_lowest_index_tie_break():
    assert predict_index(torch.tensor([0.5, 0.9, 0.9])) == 1
    assert predict_index(torch.tensor([0.9, 0.9, 0.9])) == 0


def test_predict_index_rejects_bad_shape():
    with pytest.raises(ValueError):
        predict_index(torch.tensor([[0.1, 0.9]]))
    with pytest.raises(ValueError):
        predict_index(torch.tensor([]))


def test_softmax_map_sums_to_one_and_ranks_scores():
    p = softmax_map(torch.tensor([0.1, 0.9, 0.5]), T=0.2)
    assert tuple(p.shape) == (3,)
    assert abs(float(p.sum()) - 1.0) < 1e-6
    assert int(torch.argmax(p)) == 1
    assert float(p[1]) > float(p[2]) > float(p[0])


def test_softmax_map_is_shift_invariant():
    scores = torch.tensor([0.1, 0.9, 0.5])
    a = softmax_map(scores, T=0.3)
    b = softmax_map(scores + 5.0, T=0.3)
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("bad_T", [0.0, -0.3])
def test_softmax_map_requires_positive_temperature(bad_T):
    with pytest.raises(ValueError):
        softmax_map(torch.tensor([0.1, 0.9]), T=bad_T)


# --------------------------------------------------------------------------- #
# localization_error / success_within
# --------------------------------------------------------------------------- #
def test_localization_error_is_l2():
    assert localization_error([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]) == pytest.approx(5.0)
    assert localization_error(torch.tensor([1.0, 2.0, 3.0]),
                              torch.tensor([1.0, 2.0, 3.0])) == pytest.approx(0.0)
    assert isinstance(localization_error([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), float)


def test_localization_error_rejects_bad_shape():
    with pytest.raises(ValueError):
        localization_error([0.0, 0.0], [1.0, 0.0, 0.0])


def test_success_within_boundary_is_inclusive():
    assert success_within(1.0, 1.0) is True          # e_loc <= r counts as success
    assert success_within(0.9999, 1.0) is True
    assert success_within(1.0001, 1.0) is False


def test_success_within_rejects_negative_radius():
    with pytest.raises(ValueError):
        success_within(0.5, -1.0)
