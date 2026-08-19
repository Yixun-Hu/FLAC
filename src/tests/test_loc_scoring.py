"""Tests for ``src.localization.scoring`` (exp_18 loc_invert, round 1).

Written test-first (announcement 02). Contracts: ``loc_invert_impl_contracts.md``
§4.3 plus the Rev 3 §4 deltas -- LME aggregation through ``torch.logsumexp``
(tau = 0.02 stability), the information-matched context-conditioned baseline with
its explicit GT-only edge case, the non-generative nearest-context control, the
room-clustered (paired) bootstrap statistics, the wiring power statistic and the
cross-platform-stable noise key.
"""
import math
import os

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.localization.scoring import (
    aggregate,
    clustered_bootstrap_ci,
    context_conditioned_baseline,
    cosine_sims,
    localization_error,
    nearest_context_baseline,
    noise_key,
    paired_room_clustered_test,
    power_statistic,
    predict_index,
    softmax_map,
    success_within,
    summarize,
    uniform_baseline,
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


# --------------------------------------------------------------------------- #
# uniform_baseline (spec's literal lower bound)
# --------------------------------------------------------------------------- #
_CAND = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
_GT = torch.tensor([0.0, 0.0, 0.0])                       # candidate 0; distances 0/1/3/4


def test_uniform_baseline_hand_example():
    out = uniform_baseline(_CAND, _GT, radii=(0.5, 1.0))
    assert out["n_candidates"] == 4 and out["n_eligible"] == 4
    assert out["distances"] == pytest.approx([0.0, 1.0, 3.0, 4.0])
    assert out["mean_error"] == pytest.approx(2.0)
    assert out["success"][0.5] == pytest.approx(0.25)
    assert out["success"][1.0] == pytest.approx(0.5)      # boundary counts
    assert out["top1"] == pytest.approx(0.25)


def test_uniform_baseline_matches_monte_carlo():
    """Exact expectations must agree with a seeded uniform draw over C (1e-3)."""
    import numpy as np
    out = uniform_baseline(_CAND, _GT, radii=(0.5, 1.0))
    rng = np.random.default_rng(18)
    draws = rng.integers(0, 4, size=4000000)   # sd(p_hat) ~ 2e-4, so 1e-3 is ~5 sigma
    dists = np.asarray(out["distances"])[draws]
    assert dists.mean() == pytest.approx(out["mean_error"], abs=1e-2)
    assert (dists <= 1.0).mean() == pytest.approx(out["success"][1.0], abs=1e-3)
    assert (dists <= 0.5).mean() == pytest.approx(out["success"][0.5], abs=1e-3)


def test_uniform_baseline_accepts_numpy_inputs():
    import numpy as np
    out = uniform_baseline(np.asarray(_CAND.numpy(), dtype=np.float64), np.zeros(3))
    assert out["mean_error"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# context_conditioned_baseline -- REGISTERED comparison target (C1)
# --------------------------------------------------------------------------- #
def test_context_conditioned_baseline_hand_example():
    mask = [False, True, False, True]                     # candidates 1 and 3 are context
    out = context_conditioned_baseline(_CAND, _GT, mask, radii=(0.5, 1.0))
    assert out["n_candidates"] == 4 and out["n_eligible"] == 2
    assert out["distances"] == pytest.approx([0.0, 3.0])
    assert out["mean_error"] == pytest.approx(1.5)
    assert out["success"][0.5] == pytest.approx(0.5)
    assert out["success"][1.0] == pytest.approx(0.5)
    assert out["top1"] == pytest.approx(0.5)
    assert out["gt_only"] is False


def test_context_conditioned_baseline_gt_only_edge_case():
    """LivingRoomsWithHallway_idx_30: 9 sources, 8 in context => eligible == {GT},
    so elimination alone names the target (baseline top-1 = 1.0, zero headroom).
    The flag lets callers exclude/label the room instead of silently averaging it."""
    cand = torch.cat([_GT.unsqueeze(0), torch.arange(1.0, 9.0).reshape(8, 1) * torch.tensor([[1.0, 0.0, 0.0]])])
    mask = [False] + [True] * 8
    out = context_conditioned_baseline(cand, _GT, mask, radii=(0.5, 1.0))
    assert out["gt_only"] is True
    assert out["n_eligible"] == 1
    assert out["distances"] == pytest.approx([0.0])
    assert out["mean_error"] == pytest.approx(0.0)
    assert out["success"][0.5] == pytest.approx(1.0)
    assert out["success"][1.0] == pytest.approx(1.0)
    assert out["top1"] == pytest.approx(1.0)


def test_context_conditioned_baseline_gt_must_be_eligible():
    """GT is excluded from the context draw by construction; a mask that claims
    otherwise is a wiring bug, not a baseline to average over."""
    with pytest.raises(ValueError):
        context_conditioned_baseline(_CAND, _GT, [True, False, False, False])


def test_context_conditioned_baseline_requires_gt_in_candidates():
    with pytest.raises(ValueError):
        context_conditioned_baseline(_CAND, torch.tensor([9.0, 9.0, 9.0]),
                                     [False, True, False, True])


def test_context_conditioned_baseline_rejects_bad_mask():
    with pytest.raises(ValueError):
        context_conditioned_baseline(_CAND, _GT, [False, True, False])


def test_context_conditioned_baseline_all_eligible_equals_uniform():
    """Identical conventions: with an empty context the two baselines coincide."""
    a = uniform_baseline(_CAND, _GT, radii=(0.5, 1.0))
    b = context_conditioned_baseline(_CAND, _GT, [False] * 4, radii=(0.5, 1.0))
    assert b["distances"] == pytest.approx(a["distances"])
    assert b["mean_error"] == pytest.approx(a["mean_error"])
    assert b["success"] == pytest.approx(a["success"])
    assert b["top1"] == pytest.approx(a["top1"])


# --------------------------------------------------------------------------- #
# nearest_context_baseline -- non-generative control (O10)
# --------------------------------------------------------------------------- #
_CTX = torch.tensor([[2.9, 0.0, 0.0], [0.1, 0.0, 0.0]])


def test_nearest_context_baseline_hand_example():
    assert nearest_context_baseline(_CAND, _CTX, torch.tensor([0.2, 0.8])) == 0
    assert nearest_context_baseline(_CAND, _CTX, torch.tensor([0.9, 0.2])) == 2
    assert isinstance(nearest_context_baseline(_CAND, _CTX, torch.tensor([0.9, 0.2])), int)


def test_nearest_context_baseline_lowest_index_tie_breaks():
    assert nearest_context_baseline(_CAND, _CTX, torch.tensor([0.5, 0.5])) == 2   # ctx tie -> ctx 0
    mid = torch.tensor([[0.5, 0.0, 0.0]])
    assert nearest_context_baseline(_CAND, mid, torch.tensor([1.0])) == 0         # candidate tie


def test_nearest_context_baseline_rejects_bad_shapes():
    with pytest.raises(ValueError):
        nearest_context_baseline(_CAND, _CTX, torch.tensor([0.5]))                # len mismatch
    with pytest.raises(ValueError):
        nearest_context_baseline(_CAND, torch.zeros(0, 3), torch.zeros(0))        # empty context


# --------------------------------------------------------------------------- #
# noise_key -- deterministic, cross-platform-stable generator seeds (C10)
# --------------------------------------------------------------------------- #
def test_noise_key_pinned_golden_values():
    """Pinned by the canonical payload ``["loc_invert_noise_key", seed, query_id, k]``
    hashed with sha256 (first 8 bytes, top bit cleared). These literals are the
    cross-platform contract: any change to the derivation breaks resumability."""
    assert noise_key(42, "Cafe/Cafe_idx_1/S008_R089", 0) == 4131827329579807174
    assert noise_key(42, "Cafe/Cafe_idx_1/S008_R089", 7) == 1203058009045468154
    assert noise_key(0, "a", 0) == 6625936037822441059


def test_noise_key_is_stable_across_interpreter_hash_seeds():
    """Not Python's ``hash()``: PYTHONHASHSEED must not change the key."""
    import subprocess
    import sys
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from src.localization.scoring import noise_key;"
        "print(noise_key(42, 'Cafe/Cafe_idx_1/S008_R089', 3))" % _REPO_ROOT
    )
    outs = []
    for hash_seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, check=True,
                                   capture_output=True, text=True).stdout.strip())
    assert len(set(outs)) == 1
    assert int(outs[0]) == noise_key(42, "Cafe/Cafe_idx_1/S008_R089", 3)


def test_noise_key_no_collisions_over_a_grid():
    keys = [noise_key(seed, qid, k)
            for seed in (0, 1, 42, 43, 44)
            for qid in ("a", "b", "1", "12", "S008_R089", "Cafe/Cafe_idx_1/S008_R089")
            for k in range(8)]
    assert len(set(keys)) == len(keys)


def test_noise_key_depends_on_k_and_query_id_independently():
    assert noise_key(42, "q", 1) != noise_key(42, "q", 2)
    assert noise_key(42, "q1", 1) != noise_key(42, "q2", 1)
    assert noise_key(42, "q", 1) != noise_key(43, "q", 1)
    # no field-boundary aliasing: ("1", 2) must not collide with ("1|2", 0) etc.
    assert noise_key(42, "1", 2) != noise_key(42, "12", 0)
    assert noise_key(42, "1", 2) != noise_key(42, "1|2", 0)


def test_noise_key_is_usable_as_a_torch_generator_seed():
    key = noise_key(42, "Cafe/Cafe_idx_1/S008_R089", 3)
    assert isinstance(key, int) and 0 <= key < 2 ** 63
    a = torch.randn(4, generator=torch.Generator().manual_seed(key))
    b = torch.randn(4, generator=torch.Generator().manual_seed(key))
    assert torch.equal(a, b)


def test_noise_key_normalizes_argument_types():
    assert noise_key(42, "q", 1) == noise_key(42, "q", True)      # bool -> int 1
    assert noise_key(True, "q", 1) == noise_key(1, "q", 1)


# --------------------------------------------------------------------------- #
# power_statistic -- wiring control (plan §2.8.2)
# --------------------------------------------------------------------------- #
def test_power_statistic_hand_example():
    """var_m(mean_k s) / mean_m(var_k s) with unbiased variances:
    means [0.1, 1.2] -> var 0.605; per-candidate vars [0.02, 0.08] -> mean 0.05."""
    sims = torch.tensor([[0.0, 0.2], [1.0, 1.4]])
    assert power_statistic(sims) == pytest.approx(0.605 / 0.05, rel=1e-6)


def test_power_statistic_is_large_when_candidate_identity_dominates():
    identity_driven = torch.tensor([[0.90, 0.91], [0.10, 0.11]])
    noise_driven = torch.tensor([[0.90, 0.10], [0.91, 0.11]])
    assert power_statistic(identity_driven) > 100.0
    assert power_statistic(noise_driven) < 1e-2


def test_power_statistic_zero_within_variance_is_infinite():
    assert math.isinf(power_statistic(torch.tensor([[0.5, 0.5], [0.9, 0.9]])))


def test_power_statistic_requires_two_candidates_and_two_samples():
    with pytest.raises(ValueError):
        power_statistic(torch.tensor([[0.1, 0.2]]))          # M = 1
    with pytest.raises(ValueError):
        power_statistic(torch.tensor([[0.1], [0.2]]))        # K = 1
    with pytest.raises(ValueError):
        power_statistic(torch.tensor([0.1, 0.2]))            # not [M, K]


# --------------------------------------------------------------------------- #
# summarize -- pooled primary (C3) + labelled per-room secondaries
# --------------------------------------------------------------------------- #
_RECORDS = [
    {"query_id": "q0", "room_id": "A/A_idx_0", "e_loc": 0.0, "top1": 1.0, "rr": 1.0},
    {"query_id": "q1", "room_id": "A/A_idx_0", "e_loc": 2.0, "top1": 0.0, "rr": 0.5},
    {"query_id": "q2", "room_id": "B/B_idx_1", "e_loc": 3.0, "top1": 0.0, "rr": 0.25},
]


def test_summarize_pooled_primary_and_secondaries():
    out = summarize(_RECORDS, radii=(0.5, 1.0))
    assert out["primary_name"] == "pooled_median_e_loc"
    assert out["primary"] == pytest.approx(2.0)
    assert out["n_queries"] == 3 and out["n_rooms"] == 2

    pooled = out["pooled"]
    assert pooled["median_e_loc"] == pytest.approx(2.0)
    assert pooled["mean_e_loc"] == pytest.approx(5.0 / 3.0)
    assert pooled["success"][0.5] == pytest.approx(1.0 / 3.0)
    assert pooled["success"][1.0] == pytest.approx(1.0 / 3.0)
    assert pooled["top1"] == pytest.approx(1.0 / 3.0)
    assert pooled["mrr"] == pytest.approx(1.75 / 3.0)


def test_summarize_per_room_and_macro():
    out = summarize(_RECORDS, radii=(0.5, 1.0))
    room_a = out["per_room"]["A/A_idx_0"]
    assert room_a["n_queries"] == 2
    assert room_a["median_e_loc"] == pytest.approx(1.0)       # median of [0, 2]
    assert room_a["mean_e_loc"] == pytest.approx(1.0)
    assert room_a["success"][1.0] == pytest.approx(0.5)
    assert room_a["top1"] == pytest.approx(0.5)
    assert room_a["mrr"] == pytest.approx(0.75)
    assert out["per_room"]["B/B_idx_1"]["median_e_loc"] == pytest.approx(3.0)

    macro = out["macro"]
    assert macro["n_rooms"] == 2
    assert macro["mean_of_room_medians"] == pytest.approx(2.0)
    assert macro["mean_of_room_means"] == pytest.approx(2.0)
    assert macro["success"][1.0] == pytest.approx(0.25)
    assert macro["top1"] == pytest.approx(0.25)
    assert macro["mrr"] == pytest.approx(0.5)


def test_summarize_median_matches_numpy_for_even_counts():
    import numpy as np
    values = [1.0, 2.0, 3.0, 4.0]
    recs = [{"room_id": "A/A_idx_0", "e_loc": v} for v in values]
    assert summarize(recs)["primary"] == pytest.approx(float(np.median(values)))


def test_summarize_weights_baseline_distance_lists_by_one_over_m():
    """A baseline record carries the whole per-candidate distance list; each
    candidate gets weight 1/M so the query still counts once (identical
    weighting to a FLAC record, which is the degenerate M=1 case)."""
    recs = [
        {"query_id": "b0", "room_id": "A/A_idx_0", "distances": [0.0, 2.0, 4.0]},
        {"query_id": "b1", "room_id": "A/A_idx_0", "e_loc": 1.0},
    ]
    out = summarize(recs, radii=(1.0,))
    assert out["primary"] == pytest.approx(1.0)
    assert out["pooled"]["mean_e_loc"] == pytest.approx(1.5)
    assert out["pooled"]["success"][1.0] == pytest.approx(2.0 / 3.0)


def test_summarize_singleton_distance_list_equals_scalar_record():
    a = summarize([{"room_id": "A/A_idx_0", "distances": [2.5]}], radii=(1.0, 5.0))
    b = summarize([{"room_id": "A/A_idx_0", "e_loc": 2.5}], radii=(1.0, 5.0))
    assert a["pooled"] == b["pooled"] and a["primary"] == b["primary"]


def test_summarize_optional_fields_absent_report_none():
    recs = [{"room_id": "A/A_idx_0", "e_loc": 1.0}, {"room_id": "B/B_idx_1", "e_loc": 2.0}]
    out = summarize(recs)
    assert out["pooled"]["top1"] is None and out["pooled"]["mrr"] is None
    assert out["macro"]["top1"] is None and out["per_room"]["A/A_idx_0"]["top1"] is None


def test_summarize_rejects_partially_present_optional_fields():
    recs = [{"room_id": "A/A_idx_0", "e_loc": 1.0, "top1": 1.0},
            {"room_id": "A/A_idx_0", "e_loc": 2.0}]
    with pytest.raises(ValueError):
        summarize(recs)


@pytest.mark.parametrize("bad", [
    [],                                                             # no records
    [{"room_id": "A/A_idx_0"}],                                     # neither e_loc nor distances
    [{"room_id": "A/A_idx_0", "e_loc": 1.0, "distances": [1.0]}],   # ambiguous
    [{"e_loc": 1.0}],                                               # no room key
    [{"room_id": "A/A_idx_0", "distances": []}],                    # empty list
])
def test_summarize_rejects_malformed_records(bad):
    with pytest.raises(ValueError):
        summarize(bad)


# --------------------------------------------------------------------------- #
# clustered_bootstrap_ci -- 17-room clustered CI on the primary (C3)
# --------------------------------------------------------------------------- #
def _room_records(n_rooms=4, per_room=5, offset=0.0):
    return [{"query_id": f"r{r}q{i}", "room_id": f"S{r}/S{r}_idx_0",
             "e_loc": offset + r + 0.1 * i}
            for r in range(n_rooms) for i in range(per_room)]


def test_clustered_bootstrap_ci_reports_the_primary_and_brackets_it():
    recs = _room_records()
    out = clustered_bootstrap_ci(recs, by="room_id", n=500, seed=42)
    assert out["stat"] == "pooled_median_e_loc"
    assert out["point"] == pytest.approx(summarize(recs)["primary"])
    assert out["n_clusters"] == 4 and out["n_boot"] == 500 and out["by"] == "room_id"
    assert out["lo"] <= out["point"] <= out["hi"]
    assert out["lo"] < out["hi"]                       # rooms differ -> non-degenerate


def test_clustered_bootstrap_ci_degenerate_single_room():
    """One cluster: every resample is that same room, so the CI collapses onto
    the point estimate rather than pretending to have between-room information."""
    recs = _room_records(n_rooms=1, per_room=7)
    out = clustered_bootstrap_ci(recs, n=200, seed=1)
    assert out["n_clusters"] == 1
    assert out["lo"] == pytest.approx(out["point"])
    assert out["hi"] == pytest.approx(out["point"])


def test_clustered_bootstrap_ci_is_reproducible_with_seed():
    recs = _room_records()
    a = clustered_bootstrap_ci(recs, n=300, seed=7)
    b = clustered_bootstrap_ci(recs, n=300, seed=7)
    assert a == b
    # the seed must actually drive the resampling: with 20 clusters the
    # percentile endpoints are no longer a coarse discrete grid.
    many = _room_records(n_rooms=20, per_room=5)
    d = clustered_bootstrap_ci(many, n=300, seed=7)
    e = clustered_bootstrap_ci(many, n=300, seed=8)
    assert d["point"] == pytest.approx(e["point"])
    assert (d["lo"], d["hi"]) != (e["lo"], e["hi"])


def test_clustered_bootstrap_ci_resamples_rooms_not_queries():
    """Clustering must be by room: 300 queries in 2 rooms still give 2 clusters."""
    recs = [{"query_id": f"q{i}", "room_id": "A/A_idx_0" if i < 150 else "B/B_idx_0",
             "e_loc": 0.0 if i < 150 else 10.0} for i in range(300)]
    out = clustered_bootstrap_ci(recs, n=400, seed=3)
    assert out["n_clusters"] == 2
    assert out["lo"] == pytest.approx(0.0) and out["hi"] == pytest.approx(10.0)


def test_clustered_bootstrap_ci_rejects_bad_arguments():
    recs = _room_records()
    with pytest.raises(ValueError):
        clustered_bootstrap_ci(recs, n=0, seed=1)
    with pytest.raises(ValueError):
        clustered_bootstrap_ci(recs, n=100, seed=1, alpha=0.0)
    with pytest.raises(ValueError):
        clustered_bootstrap_ci(recs, by="scene", n=100, seed=1)


# --------------------------------------------------------------------------- #
# paired_room_clustered_test -- pre-registered paired comparison (O12)
# --------------------------------------------------------------------------- #
def _paired(a_values, b_values):
    a = [{"query_id": f"q{i}", "room_id": f"S{i % 3}/S{i % 3}_idx_0", "e_loc": v}
         for i, v in enumerate(a_values)]
    b = [{"query_id": f"q{i}", "room_id": f"S{i % 3}/S{i % 3}_idx_0", "e_loc": v}
         for i, v in enumerate(b_values)]
    return a, b


def test_paired_room_clustered_test_constant_difference():
    a, b = _paired([2.0] * 9, [1.0] * 9)
    out = paired_room_clustered_test(a, b, n=200, seed=5)
    assert out["stat"] == "median_paired_difference"
    assert out["point"] == pytest.approx(1.0)            # a - b
    assert out["lo"] == pytest.approx(1.0) and out["hi"] == pytest.approx(1.0)
    assert out["p_value"] == pytest.approx(0.0)
    assert out["n_queries"] == 9 and out["n_clusters"] == 3


def test_paired_room_clustered_test_sign_convention_and_null():
    a, b = _paired([2.0] * 9, [1.0] * 9)
    assert paired_room_clustered_test(b, a, n=200, seed=5)["point"] == pytest.approx(-1.0)
    same = paired_room_clustered_test(a, list(a), n=200, seed=5)
    assert same["point"] == pytest.approx(0.0) and same["p_value"] == pytest.approx(1.0)


def test_paired_room_clustered_test_is_reproducible_and_supports_mean():
    a, b = _paired([3.0, 1.0, 2.0, 5.0, 0.5, 1.5, 2.5, 4.0, 0.0],
                   [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    first = paired_room_clustered_test(a, b, n=300, seed=11)
    assert first == paired_room_clustered_test(a, b, n=300, seed=11)
    assert 0.0 <= first["p_value"] <= 1.0 and first["lo"] < first["hi"]
    mean_out = paired_room_clustered_test(a, b, n=300, seed=11, stat="mean")
    assert mean_out["stat"] == "mean_paired_difference"
    assert mean_out["point"] == pytest.approx(
        sum(x["e_loc"] for x in a) / 9.0 - sum(x["e_loc"] for x in b) / 9.0)


def test_paired_room_clustered_test_pairs_baseline_distance_lists_by_mean():
    a = [{"query_id": "q0", "room_id": "A/A_idx_0", "e_loc": 1.0}]
    b = [{"query_id": "q0", "room_id": "A/A_idx_0", "distances": [0.0, 2.0, 4.0]}]
    out = paired_room_clustered_test(a, b, n=50, seed=0)
    assert out["point"] == pytest.approx(1.0 - 2.0)     # baseline expectation = mean


def test_paired_room_clustered_test_rejects_unpaired_inputs():
    a, b = _paired([1.0] * 3, [1.0] * 3)
    with pytest.raises(ValueError):
        paired_room_clustered_test(a, b[:2], n=10, seed=0)
    mismatched = [dict(rec, query_id="other") for rec in b[:1]] + b[1:]
    with pytest.raises(ValueError):
        paired_room_clustered_test(a, mismatched, n=10, seed=0)
    moved = [dict(b[0], room_id="Z/Z_idx_0")] + b[1:]
    with pytest.raises(ValueError):
        paired_room_clustered_test(a, moved, n=10, seed=0)
    with pytest.raises(ValueError):
        paired_room_clustered_test([{"room_id": "A/A_idx_0", "e_loc": 1.0}],
                                   [{"room_id": "A/A_idx_0", "e_loc": 1.0}], n=10, seed=0)


# --------------------------------------------------------------------------- #
# r1 fix F1 (review finding 1): non-finite values must fail closed everywhere.
# `NaN > tol` is False, so a bare threshold check lets NaN through and the
# pipeline then reports superficially valid success/top-1 numbers next to NaN
# errors. Every entry point rejects NaN and +/-Inf instead.
# --------------------------------------------------------------------------- #
_NONFINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("bad", _NONFINITE)
def test_cosine_sims_rejects_nonfinite_obs_and_gen(bad):
    obs = _unit(1.0, 0.0, 0.0)
    gen = torch.stack([torch.stack([_unit(1.0, 0.0, 0.0)])])
    with pytest.raises(ValueError):
        cosine_sims(torch.tensor([bad, 0.0, 0.0]), gen)
    bad_gen = gen.clone()
    bad_gen[0, 0, 1] = bad
    with pytest.raises(ValueError):
        cosine_sims(obs, bad_gen)


@pytest.mark.parametrize("bad", _NONFINITE)
@pytest.mark.parametrize("method", ["lme", "mean", "max"])
def test_aggregate_rejects_nonfinite_sims(bad, method):
    with pytest.raises(ValueError):
        aggregate(torch.tensor([[0.5, bad]]), method, tau=0.02)


@pytest.mark.parametrize("bad", _NONFINITE)
def test_aggregate_rejects_nonfinite_tau(bad):
    with pytest.raises(ValueError):
        aggregate(_SIMS, "lme", tau=bad)


@pytest.mark.parametrize("bad", _NONFINITE)
def test_softmax_map_and_predict_index_reject_nonfinite(bad):
    with pytest.raises(ValueError):
        softmax_map(torch.tensor([0.1, bad]), T=0.2)
    with pytest.raises(ValueError):
        softmax_map(torch.tensor([0.1, 0.9]), T=bad)
    with pytest.raises(ValueError):
        predict_index(torch.tensor([0.1, bad]))


@pytest.mark.parametrize("bad", _NONFINITE)
def test_power_statistic_rejects_nonfinite_sims(bad):
    with pytest.raises(ValueError):
        power_statistic(torch.tensor([[0.1, 0.2], [0.3, bad]]))


@pytest.mark.parametrize("bad", _NONFINITE)
def test_localization_error_and_success_within_reject_nonfinite(bad):
    with pytest.raises(ValueError):
        localization_error([0.0, 0.0, 0.0], [1.0, bad, 0.0])
    with pytest.raises(ValueError):
        success_within(bad, 1.0)
    with pytest.raises(ValueError):
        success_within(0.5, bad)


@pytest.mark.parametrize("bad", _NONFINITE)
def test_baselines_reject_nonfinite_coordinates(bad):
    """A NaN candidate used to give success@1 = 0.5 with a NaN mean error."""
    cand = _CAND.clone()
    cand[2, 0] = bad
    with pytest.raises(ValueError):
        uniform_baseline(cand, _GT)
    with pytest.raises(ValueError):
        context_conditioned_baseline(cand, _GT, [False, True, False, True])
    with pytest.raises(ValueError):
        uniform_baseline(_CAND, torch.tensor([bad, 0.0, 0.0]))
    with pytest.raises(ValueError):
        nearest_context_baseline(cand, _CTX, torch.tensor([0.2, 0.8]))
    with pytest.raises(ValueError):
        nearest_context_baseline(_CAND, _CTX, torch.tensor([0.2, bad]))


@pytest.mark.parametrize("bad", _NONFINITE)
def test_summarize_rejects_nonfinite_records(bad):
    """A NaN e_loc used to be silently counted as a failure rather than refused."""
    with pytest.raises(ValueError):
        summarize([{"room_id": "A/A_idx_0", "e_loc": bad}])
    with pytest.raises(ValueError):
        summarize([{"room_id": "A/A_idx_0", "distances": [1.0, bad]}])
    with pytest.raises(ValueError):
        summarize([{"room_id": "A/A_idx_0", "e_loc": 1.0, "top1": bad, "rr": 1.0}])
    with pytest.raises(ValueError):
        summarize([{"room_id": "A/A_idx_0", "e_loc": 1.0, "rr": bad}])
    with pytest.raises(ValueError):
        summarize([{"room_id": "A/A_idx_0", "e_loc": 1.0}], radii=(bad,))


@pytest.mark.parametrize("bad", _NONFINITE)
def test_bootstrap_helpers_reject_nonfinite_records(bad):
    recs = [{"query_id": "q0", "room_id": "A/A_idx_0", "e_loc": bad}]
    with pytest.raises(ValueError):
        clustered_bootstrap_ci(recs, n=10, seed=0)
    other = [{"query_id": "q0", "room_id": "A/A_idx_0", "e_loc": 1.0}]
    with pytest.raises(ValueError):
        paired_room_clustered_test(recs, other, n=10, seed=0)
