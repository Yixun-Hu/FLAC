"""exp_22 I1 -- the frozen mesh-grid localization engine (inherited plan §1.4/§1.5).

The engine is the only place where the frozen checkpoint, the D1 context draws
and the G1 candidate geometry meet, so almost every test here is a *binding*
test: the noise a candidate is generated from, the conditioning it is cached
under, the manifest it is admitted against, and the artifact it is resumed from
all have to be derivable from registered quantities and nothing else.

The generation stack is injected (``MeshEngine``), so the whole per-query and
per-receiver layout is exercised on CPU with recording fakes; the real stack is
built by ``build_mesh_engine`` and covered by the driver-level smoke.
"""
import json
import math
import os

import numpy as np
import pytest
import torch

from src.localization import meshgrid_engine as me
from src.localization import scoring as sc


# --------------------------------------------------------------------------- #
# the noise key: (seed, query_id, candidate_index, k)
# --------------------------------------------------------------------------- #
def test_noise_key_is_deterministic_and_separates_every_coordinate():
    base = me.noise_key(42, "q|a.wav", 7, 3)
    assert base == me.noise_key(42, "q|a.wav", 7, 3)
    assert 0 <= base < (1 << 63)
    assert base != me.noise_key(43, "q|a.wav", 7, 3)
    assert base != me.noise_key(42, "q|b.wav", 7, 3)
    assert base != me.noise_key(42, "q|a.wav", 8, 3)
    assert base != me.noise_key(42, "q|a.wav", 7, 4)


def test_noise_key_is_not_pythons_salted_hash_and_is_pinned():
    # a literal pin: a resumed or re-run pass on another machine must draw the
    # same noise, so this value is part of the protocol, not an implementation
    # detail.
    assert me.noise_key(42, "0|single_channel_ir_1/Cafe/Cafe_idx_1/S006_R008_hybrid_IR.wav",
                        0, 0) == me.noise_key(
        42, "0|single_channel_ir_1/Cafe/Cafe_idx_1/S006_R008_hybrid_IR.wav", 0, 0)
    assert me.noise_key(42, "q", 0, 0) != abs(hash(("q", 0, 0)))


def test_shared_policy_delegates_to_the_exp18_query_level_key():
    assert me.noise_key_for("shared_across_candidates", 42, "q", 5, 2) == sc.noise_key(42, "q", 2)
    assert me.noise_key_for("per_candidate", 42, "q", 5, 2) == me.noise_key(42, "q", 5, 2)
    with pytest.raises(ValueError, match="noise policy"):
        me.noise_key_for("whatever", 42, "q", 5, 2)


def test_the_registered_default_policy_is_recorded_not_assumed():
    assert me.NOISE_KEY_POLICY == "per_candidate"
    assert set(me.NOISE_KEY_POLICIES) == {"per_candidate", "shared_across_candidates"}


# --------------------------------------------------------------------------- #
# the noise block: candidate-major, chunk invariant
# --------------------------------------------------------------------------- #
def test_noise_block_is_candidate_major_and_shaped_by_the_latent():
    block = me.noise_block(42, "q", [3, 9], num_samples=4, latent_shape=(2, 5))
    assert tuple(block.shape) == (8, 2, 5)
    # row m * K + k is candidate m, draw k
    single = me.noise_block(42, "q", [9], num_samples=4, latent_shape=(2, 5))
    assert torch.equal(block[4:8], single)


def test_noise_block_is_invariant_to_how_the_rows_are_chunked():
    whole = me.noise_block(42, "q", [0, 1, 2, 3], num_samples=8, latent_shape=(2, 5))
    halves = torch.cat([me.noise_block(42, "q", [0, 1], num_samples=8, latent_shape=(2, 5)),
                        me.noise_block(42, "q", [2, 3], num_samples=8, latent_shape=(2, 5))])
    assert torch.equal(whole, halves)


def test_per_candidate_noise_differs_by_candidate_while_shared_does_not():
    per = me.noise_block(42, "q", [0, 1], num_samples=2, latent_shape=(2, 5))
    assert not torch.equal(per[0:2], per[2:4])
    shared = me.noise_block(42, "q", [0, 1], num_samples=2, latent_shape=(2, 5),
                            policy="shared_across_candidates")
    assert torch.equal(shared[0:2], shared[2:4])


def test_noise_block_refuses_a_bad_shape_or_sample_count():
    with pytest.raises(ValueError, match="num_samples"):
        me.noise_block(42, "q", [0], num_samples=0, latent_shape=(2, 5))
    with pytest.raises(ValueError, match="latent_shape"):
        me.noise_block(42, "q", [0], num_samples=1, latent_shape=(2,))
    with pytest.raises(ValueError, match="candidate_indices"):
        me.noise_block(42, "q", [], num_samples=1, latent_shape=(2, 5))
    with pytest.raises(ValueError, match="candidate_indices"):
        me.noise_block(42, "q", [0, 0], num_samples=1, latent_shape=(2, 5))


# --------------------------------------------------------------------------- #
# the nested K prefixes
# --------------------------------------------------------------------------- #
def _sims(seed=0, m=6, k=8):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((m, k), generator=generator).float() * 0.4 + 0.3


def test_the_registered_prefixes_and_tau_are_pinned():
    assert me.K_PREFIXES == (1, 4, 8)
    assert me.NUM_SAMPLES == 8
    assert me.TAU == 0.1
    assert me.SEED == 42


def test_nested_scores_are_exactly_the_prefix_aggregations():
    sims = _sims()
    scores = me.nested_scores(sims, tau=0.1)
    for k in (1, 4, 8):
        assert torch.equal(scores[k]["scores"],
                           sc.aggregate(sims[:, :k].contiguous(), method="lme", tau=0.1))
        assert torch.equal(scores[k]["mean_scores"], sims[:, :k].mean(dim=-1))


def test_k1_is_the_head_of_k8_not_a_separate_draw():
    sims = _sims()
    scores = me.nested_scores(sims, tau=0.1)
    # K=1's log-mean-exp of one sample is that sample
    assert torch.allclose(scores[1]["scores"], sims[:, 0], atol=1e-6)
    assert torch.equal(scores[1]["mean_scores"], sims[:, 0])


def test_lme_matches_the_registered_closed_form():
    sims = _sims()
    got = me.nested_scores(sims, tau=0.1)[4]["scores"]
    want = 0.1 * (torch.logsumexp(sims[:, :4] / 0.1, dim=-1) - math.log(4))
    assert torch.allclose(got, want, atol=0, rtol=0)


def test_nested_scores_refuse_a_short_or_non_finite_sims_block():
    with pytest.raises(ValueError, match="samples"):
        me.nested_scores(_sims(k=4), tau=0.1)
    bad = _sims()
    bad[2, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        me.nested_scores(bad, tau=0.1)
    with pytest.raises(ValueError, match="tau"):
        me.nested_scores(_sims(), tau=0.0)


# --------------------------------------------------------------------------- #
# prediction: lexicographically stable over the GLOBAL candidate index
# --------------------------------------------------------------------------- #
def test_prediction_breaks_ties_by_the_smallest_global_candidate_index():
    scores = torch.tensor([0.5, 0.9, 0.9, 0.2])
    # the manifest's indices are ascending, so row 1 wins
    assert me.argmax_by_global_index(scores, [4, 11, 17, 23]) == 1
    # ... but the rule is on the GLOBAL index, so a permuted set picks row 2
    assert me.argmax_by_global_index(scores, [4, 17, 11, 23]) == 2


def test_prediction_refuses_a_mismatched_or_non_finite_input():
    with pytest.raises(ValueError, match="candidate_indices"):
        me.argmax_by_global_index(torch.tensor([0.1, 0.2]), [1, 2, 3])
    with pytest.raises(ValueError, match="finite"):
        me.argmax_by_global_index(torch.tensor([0.1, float("inf")]), [1, 2])


def test_scored_query_carries_every_prefix_prediction_and_the_mean_diagnostic():
    sims = _sims(m=5)
    indices = [10, 3, 7, 90, 40]
    coordinates = np.arange(15, dtype=np.float64).reshape(5, 3)
    scored = me.score_query(sims, indices, coordinates, tau=0.1)
    assert sorted(scored["by_k"]) == [1, 4, 8]
    for k, block in scored["by_k"].items():
        expected = me.argmax_by_global_index(
            sc.aggregate(sims[:, :k].contiguous(), method="lme", tau=0.1), indices)
        assert block["prediction_row"] == expected
        assert block["prediction_index"] == indices[expected]
        assert block["prediction_xyz"] == coordinates[expected].tolist()
        # full float32 precision, losslessly recoverable
        assert len(block["scores_hex"]) == 5
        assert block["mean_prediction_index"] == indices[
            me.argmax_by_global_index(sims[:, :k].mean(dim=-1), indices)]
    assert scored["n_candidates"] == 5
    assert scored["num_samples"] == 8


# --------------------------------------------------------------------------- #
# the conditioning split: context per query, source per (receiver, candidate)
# --------------------------------------------------------------------------- #
class FakeConditioner:
    """A deterministic stand-in for ``MultiConditioner`` with its ``only_ids`` seam.

    Every branch's value is a function of the metadata it is supposed to read --
    the context ids of the context tensors, the source ids of ``source`` and
    ``depth`` -- so an incorrectly assembled batch cannot coincidentally match.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, batch_metadata, device, only_ids=None):
        self.calls.append({"n": len(batch_metadata),
                           "ids": None if only_ids is None else sorted(only_ids)})
        out = {}
        for key in me.CONTEXT_COND_IDS + me.SOURCE_COND_IDS:
            if only_ids is not None and key not in only_ids:
                continue
            rows = []
            for md in batch_metadata:
                if key in me.CONTEXT_COND_IDS:
                    seed = float(torch.as_tensor(md["context_audio"]).sum()) + len(key)
                else:
                    seed = (float(torch.as_tensor(md["source"]).sum())
                            + float(torch.as_tensor(md["depth"]).sum()) + len(key))
                rows.append(torch.full((3,), seed, dtype=torch.float32))
            out[key] = [torch.stack(rows).to(device),
                        torch.ones(len(batch_metadata), dtype=torch.bool)]
        return out


def _md(source=(0.0, 0.0, 0.0), depth_value=1.0, context_value=2.0):
    return {"source": torch.tensor(source, dtype=torch.float32),
            "source_vit": torch.tensor(source, dtype=torch.float32).unsqueeze(0),
            "depth": torch.full((2, 4), float(depth_value)),
            "context_audio": torch.full((8, 1, 16), float(context_value)),
            "context_poses": torch.arange(24, dtype=torch.float32).reshape(8, 3),
            "context_poses_vit": torch.arange(24, dtype=torch.float32).reshape(8, 3)}


def test_the_two_branches_partition_the_registered_conditioning_ids():
    from src.models import conditioners as _c            # the id contract is the config's
    config = json.load(open("src/configs/model_configs/FLAC/AR/FLAC_AR.json"))
    declared = {entry["id"] for entry in config["model"]["conditioning"]["configs"]}
    assert set(me.CONTEXT_COND_IDS) | set(me.SOURCE_COND_IDS) == declared
    assert not set(me.CONTEXT_COND_IDS) & set(me.SOURCE_COND_IDS)
    assert "only_ids" in _c.MultiConditioner.forward.__code__.co_varnames


def test_context_conditioning_runs_once_over_one_row():
    conditioner = FakeConditioner()
    cached = me.context_conditioning(conditioner, _md(), "cpu")
    assert sorted(cached) == sorted(me.CONTEXT_COND_IDS)
    assert conditioner.calls == [{"n": 1, "ids": sorted(me.CONTEXT_COND_IDS)}]
    assert cached["context_audio"][0].shape[0] == 1


def test_source_conditioning_covers_the_union_and_is_chunk_invariant():
    conditioner = FakeConditioner()
    positions = np.arange(12, dtype=np.float64).reshape(4, 3)
    whole = me.source_conditioning(conditioner, _md(), positions, "cpu", chunk=64)
    chunked = me.source_conditioning(conditioner, _md(), positions, "cpu", chunk=1)
    for key in me.SOURCE_COND_IDS:
        assert torch.equal(whole[key][0], chunked[key][0])
        assert whole[key][0].shape[0] == 4
    assert [call["n"] for call in conditioner.calls[1:]] == [1, 1, 1, 1]


def test_conditioning_refuses_a_branch_the_conditioner_did_not_return():
    class Partial(FakeConditioner):
        def __call__(self, batch_metadata, device, only_ids=None):
            out = super().__call__(batch_metadata, device, only_ids=only_ids)
            out.pop(me.CONTEXT_COND_IDS[0], None)
            return out

    with pytest.raises(ValueError, match="context_poses_vit"):
        me.context_conditioning(Partial(), _md(), "cpu")


def test_expanding_the_cache_selects_one_row_per_generated_row():
    conditioner = FakeConditioner()
    positions = np.arange(9, dtype=np.float64).reshape(3, 3)
    source = me.source_conditioning(conditioner, _md(), positions, "cpu")
    context = me.context_conditioning(conditioner, _md(), "cpu")
    merged = me.expand_conditioning(context, source, torch.tensor([2, 0, 2]), "cpu")
    assert sorted(merged) == sorted(me.CONTEXT_COND_IDS + me.SOURCE_COND_IDS)
    assert torch.equal(merged["source"][0][0], source["source"][0][2])
    assert torch.equal(merged["source"][0][1], source["source"][0][0])
    # the context row is the SAME for every generated row
    assert torch.equal(merged["context_audio"][0][0], merged["context_audio"][0][2])
    assert merged["context_audio"][0].shape[0] == 3


# --------------------------------------------------------------------------- #
# the receiver cache: bit-identity against the uncached path
# --------------------------------------------------------------------------- #
def test_cached_conditioning_is_bit_identical_to_the_uncached_path():
    from src.localization.candidates import candidate_metadata

    conditioner = FakeConditioner()
    base = _md()
    positions = np.arange(15, dtype=np.float64).reshape(5, 3)
    indices = [4, 9, 11, 30, 31]

    uncached = conditioner([candidate_metadata(base, positions[m]) for m in range(5)], "cpu")

    cache = me.ReceiverCache.build(conditioner, "R", base, indices, positions, "cpu", chunk=2)
    context = me.context_conditioning(conditioner, base, "cpu")
    rows = cache.rows_for([4, 9, 11, 30, 31])
    cached = me.expand_conditioning(context, cache.conditioning, rows, "cpu")

    for key in me.CONTEXT_COND_IDS + me.SOURCE_COND_IDS:
        assert torch.equal(cached[key][0], uncached[key][0]), key


def test_the_cache_serves_a_query_subset_in_the_querys_own_order():
    conditioner = FakeConditioner()
    positions = np.arange(15, dtype=np.float64).reshape(5, 3)
    cache = me.ReceiverCache.build(conditioner, "R", _md(), [4, 9, 11, 30, 31], positions, "cpu")
    assert cache.rows_for([30, 4]).tolist() == [3, 0]
    assert cache.n_candidates == 5
    with pytest.raises(ValueError, match="not in the receiver union"):
        cache.rows_for([4, 12])


def test_the_cache_refuses_a_query_whose_depth_is_not_the_receivers():
    conditioner = FakeConditioner()
    positions = np.arange(6, dtype=np.float64).reshape(2, 3)
    cache = me.ReceiverCache.build(conditioner, "R", _md(depth_value=1.0), [0, 1], positions, "cpu")
    cache.assert_same_depth(_md(depth_value=1.0))       # the receiver's own panorama
    with pytest.raises(ValueError, match="depth"):
        cache.assert_same_depth(_md(depth_value=7.0))


def test_the_union_is_deduplicated_and_ascending():
    conditioner = FakeConditioner()
    union = me.receiver_union([[5, 1], [1, 9], [5]])
    assert union == [1, 5, 9]
    positions = np.arange(9, dtype=np.float64).reshape(3, 3)
    cache = me.ReceiverCache.build(conditioner, "R", _md(), union, positions, "cpu")
    # one conditioner call per (receiver, candidate) in the union -- the cost the
    # G1 gate counted, not one per (query, candidate)
    assert cache.n_conditioner_rows == 3
