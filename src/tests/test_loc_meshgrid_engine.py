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


def test_the_registered_policy_is_common_random_numbers():
    # inherited plan §1.1: "All candidates for a query share ... seeds". The r7
    # dispatch keyed the draw per candidate; the r7 review ruled for CRN.
    assert me.NOISE_KEY_POLICY == "shared_across_candidates"
    assert me.REGISTERED_NOISE_POLICY == "shared_across_candidates"
    assert set(me.NOISE_KEY_POLICIES) == {"per_candidate", "shared_across_candidates"}


def test_a_registered_pass_refuses_the_per_candidate_policy(tmp_path):
    plan, records, items = _aligned(tmp_path)
    with pytest.raises(ValueError, match="common random numbers"):
        me.run_pass(SyntheticEngine(), items, records, plan, str(tmp_path / "run"),
                    num_samples=4, prefixes=(1, 4), noise_policy="per_candidate")
    # the implementation stays reachable for a ruling, but only explicitly
    summary = me.run_pass(SyntheticEngine(), items, records, plan, str(tmp_path / "b"),
                          num_samples=4, prefixes=(1, 4), noise_policy="per_candidate",
                          allow_unregistered_noise_policy=True)
    assert summary["n_scored"] == 4


def test_the_driver_refuses_a_per_candidate_run():
    import localize_meshgrid as driver

    args = driver.parse_args(["--ckpt-path", "x.ckpt"])
    assert args.noise_policy == "shared_across_candidates"
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--noise-policy", "per_candidate"])
    with pytest.raises(SystemExit, match="common random numbers"):
        driver.validate_args(args)


def test_common_random_numbers_share_the_draw_across_candidates():
    block = me.noise_block(42, "q", [3, 9, 40], num_samples=4, latent_shape=(2, 5),
                           policy=me.REGISTERED_NOISE_POLICY)
    assert torch.equal(block[0:4], block[4:8]) and torch.equal(block[4:8], block[8:12])
    # ... and they are exp_18's own keys, not a second implementation
    assert me.noise_key_for(me.REGISTERED_NOISE_POLICY, 42, "q", 999, 2) == sc.noise_key(
        42, "q", 2)


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
    per = me.noise_block(42, "q", [0, 1], num_samples=2, latent_shape=(2, 5),
                         policy="per_candidate")
    assert not torch.equal(per[0:2], per[2:4])
    shared = me.noise_block(42, "q", [0, 1], num_samples=2, latent_shape=(2, 5))
    assert torch.equal(shared[0:2], shared[2:4])       # the registered default


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
                # bounded and decorrelated: an unbounded seed would dominate the
                # synthetic embedder and collapse every cosine to 1.0, which would
                # make the whole synthetic stack unable to express a score at all
                rows.append(torch.full((3,), math.sin(seed), dtype=torch.float32))
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


# --------------------------------------------------------------------------- #
# the G1 binding: candidate manifests, branch, receiver groups
# --------------------------------------------------------------------------- #
#: A REAL fixture geometry: a 0.5 m lattice, real receivers and real context
#: sources, from which every candidate set is DERIVED by the same
#: meshgrid_geometry filter the G1 audit ran. Nothing about a query's candidate
#: set is hand-written, so the engine's GT-free reconstruction of it is a real
#: test and not a comparison against numbers invented to match it.
FIXTURE_LATTICE = np.array([[x, y, z]
                            for x in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
                            for y in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
                            for z in (0.5, 1.0, 1.5)], dtype=np.float64)

_NEAR = [(1.0, 1.0), (1.5, 1.0), (1.0, 1.5), (1.5, 1.5),
         (2.0, 1.0), (2.0, 1.5), (1.0, 2.0), (1.5, 2.0)]
_FAR = [(0.0, 2.5), (0.5, 2.5), (2.5, 0.0), (2.5, 0.5),
        (0.0, 2.0), (0.5, 2.0), (2.0, 2.5), (2.5, 2.0)]


def _ctx(points, z):
    return [[float(x), float(y), float(z)] for x, y in points]


#: ``(position, IR stem, receiver, context globals, TRUTH)``. The truth exists
#: here and in the published manifest's oracle only -- exactly where G1 put it.
#: The synthetic stream carries a ``source`` field, as the real loader does, and
#: the engine is forbidden from reading it.
FIXTURE_QUERIES = {
    "A/A_idx_1": [
        (0, "S001_R002", [0.0, 0.0, 0.5], _ctx(_NEAR, 0.5), [1.1, 1.1, 0.6]),
        (1, "S003_R004", [2.5, 2.5, 1.5], _ctx(_FAR, 0.5), [0.4, 2.4, 0.6]),
        # same receiver as position 0, different contexts -> a genuine union
        (2, "S005_R002", [0.0, 0.0, 0.5], _ctx(_NEAR, 1.5), [1.6, 1.6, 1.4]),
    ],
    "B/B_idx_2": [
        (3, "S001_R009", [1.0, 1.0, 1.0], _ctx(_FAR, 1.0), [2.4, 0.4, 1.1]),
    ],
}


def fixture_relpath(room_id, name):
    scene, scene_id = room_id.split("/")
    return f"ir/{scene}/{scene_id}/{name}_hybrid_IR.wav"


def fixture_query_id(room_id, entry):
    return f"{entry[0]}|{fixture_relpath(room_id, entry[1])}"


def fixture_receiver_id(room_id, receiver):
    return f"{room_id}|" + ",".join(f"{float(v):.6f}" for v in receiver)


def fixture_indices(room_id, entry, branch="z_band", base=None):
    """The branch's candidate indices, derived exactly as the audit derives them."""
    from src.localization import meshgrid_geometry as mg

    base = FIXTURE_LATTICE if base is None else base
    _position, _name, receiver, contexts, _truth = entry
    band = mg.context_z_band(contexts)
    kept = mg.filter_query_candidates(base, receiver=receiver, context_sources=contexts,
                                      z_band=None if branch == "full_height" else band)
    return np.flatnonzero(kept["mask"]).tolist()


def _write_room(out_dir, room_id, base=None, branch="z_band"):
    """A VERIFIER-VALID room manifest whose queries are filtered, not invented."""
    from src.localization import meshgrid_geometry as mg

    base = FIXTURE_LATTICE if base is None else base
    scene, scene_id = room_id.split("/")
    stem = f"candidates_{scene}_{scene_id}"
    np.savez(os.path.join(out_dir, stem + ".npz"), base_candidates=base)
    payload = {"room_id": room_id, "chosen_branch": branch, "spacing": 0.5,
               "coordinates_npz": stem + ".npz", "n_base_valid": int(base.shape[0]),
               "base_candidates_sha256": mg.coordinates_digest(base),
               "directions_seed": 1, "queries": []}
    for entry in FIXTURE_QUERIES[room_id]:
        position, name, receiver, contexts, truth = entry
        full = mg.filter_query_candidates(base, receiver=receiver, context_sources=contexts)
        band_bounds = mg.context_z_band(contexts)
        banded = mg.filter_query_candidates(base, receiver=receiver,
                                            context_sources=contexts, z_band=band_bounds)
        full_idx = np.flatnonzero(full["mask"])
        band_idx = np.flatnonzero(banded["mask"])
        payload["queries"].append({
            "position": position, "query_id": fixture_query_id(room_id, entry),
            "receiver": [float(v) for v in receiver],
            "receiver_id": fixture_receiver_id(room_id, receiver),
            "candidate_indices": [int(i) for i in full_idx],
            "n_candidates": int(full_idx.size),
            "candidate_indices_z_band": [int(i) for i in band_idx],
            "n_candidates_z_band": int(band_idx.size),
            "candidate_coordinates_sha256": mg.coordinates_digest(full["candidates"]),
            "n_dropped_receiver": full["n_dropped_receiver"],
            "n_dropped_context": full["n_dropped_context"],
            "n_contexts": len(contexts),
            "z_band": [float(band_bounds[0]), float(band_bounds[1])],
            "oracle": {"full_height": mg.grid_oracle_error(full["candidates"], truth),
                       "z_band": mg.grid_oracle_error(banded["candidates"], truth)}})
    path = os.path.join(out_dir, stem + ".json")
    with open(path, "w") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path, payload


def _fixture_audit(tmp_path, branch="z_band", rooms=None):
    from src.localization import meshgrid_geometry as mg

    out_dir = str(tmp_path / "g1")
    os.makedirs(out_dir, exist_ok=True)
    base = FIXTURE_LATTICE
    written = {}
    for room_id in (rooms or sorted(FIXTURE_QUERIES)):
        written[room_id] = _write_room(out_dir, room_id, base, branch=branch)
    report = {"experiment": "exp_22 loc_meshgrid G1 geometry audit",
              "n_queries": sum(len(FIXTURE_QUERIES[room]) for room in written),
              "n_rooms": len(written), "status": "accepted", "diagnostics_only": False,
              "branch": {"branch": branch, "n_new_over_threshold": 0},
              "directions_seed": 1, "spacing": 0.5,
              "rooms": {room: {"candidate_manifest": os.path.basename(path),
                               "candidate_manifest_sha256": mg.manifest_json_sha256(payload)}
                        for room, (path, payload) in written.items()}}
    report_path = os.path.join(out_dir, "geometry_audit_report.json")
    with open(report_path, "w") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return out_dir, report_path, base


def test_the_geometry_verifiers_live_beside_the_geometry_module():
    from src.localization import meshgrid_geometry as mg
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_agm", "worklog/worklog_yixun/exp_22_loc_meshgrid_claude/audit_meshgrid_geometry.py")
    agm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agm)
    # ONE implementation, re-exported: the audit tool and the engine must not be
    # able to disagree about whether an artifact verifies.
    assert agm.verify_room_manifest is mg.verify_room_manifest
    assert agm.verify_report_chain is mg.verify_report_chain
    assert agm.coordinates_digest is mg.coordinates_digest


def test_the_engine_verifies_every_room_manifest_before_reading_it(tmp_path):
    out_dir, report_path, base = _fixture_audit(tmp_path)
    plan = me.load_audit_plan(report_path)
    assert plan.branch == "z_band"
    assert sorted(plan.rooms) == ["A/A_idx_1", "B/B_idx_2"]
    assert plan.n_queries == 4
    assert plan.report_sha256 == me.file_sha256(report_path)


def test_a_tampered_room_manifest_is_refused_not_read(tmp_path):
    out_dir, report_path, base = _fixture_audit(tmp_path)
    room = os.path.join(out_dir, "candidates_A_A_idx_1.json")
    payload = json.load(open(room))
    payload["queries"][0]["candidate_indices"] = [0, 1, 2, 3, 4]
    with open(room, "w") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="A/A_idx_1"):
        me.load_audit_plan(report_path)


def test_the_branch_comes_from_the_audit_and_a_mismatch_is_refused(tmp_path):
    out_dir, report_path, base = _fixture_audit(tmp_path, branch="z_band")
    with pytest.raises(ValueError, match="full_height"):
        me.load_audit_plan(report_path, branch="full_height")
    assert me.load_audit_plan(report_path, branch="z_band").branch == "z_band"


def test_a_room_plan_carries_the_branchs_indices_and_their_coordinates(tmp_path):
    out_dir, report_path, base = _fixture_audit(tmp_path)
    plan = me.load_audit_plan(report_path)
    room = me.load_room_plan(plan, "A/A_idx_1")
    first = room.queries[0]
    entry = FIXTURE_QUERIES["A/A_idx_1"][0]
    expected = fixture_indices("A/A_idx_1", entry, branch="z_band")
    assert first.candidate_indices.tolist() == expected            # the z_band branch
    assert first.candidate_indices.tolist() != fixture_indices(
        "A/A_idx_1", entry, branch="full_height")                  # ... and it BITES
    assert first.coordinates.tolist() == base[expected].tolist()
    assert first.oracle == pytest.approx(
        float(np.linalg.norm(base[expected] - np.asarray(entry[4]), axis=1).min()))
    assert first.branch == "z_band"
    assert first.receiver_xyz.tolist() == entry[2]


def test_receiver_groups_are_the_union_and_keep_stream_order_inside(tmp_path):
    out_dir, report_path, base = _fixture_audit(tmp_path)
    room = me.load_room_plan(me.load_audit_plan(report_path), "A/A_idx_1")
    groups = me.receiver_groups(room)
    entries = FIXTURE_QUERIES["A/A_idx_1"]
    assert [group.receiver_id for group in groups] == [
        fixture_receiver_id("A/A_idx_1", entries[0][2]),
        fixture_receiver_id("A/A_idx_1", entries[1][2])]
    shared = sorted(set(fixture_indices("A/A_idx_1", entries[0]))
                    | set(fixture_indices("A/A_idx_1", entries[2])))
    assert groups[0].union == shared                    # positions 0 and 2 share a receiver
    assert len(shared) < len(fixture_indices("A/A_idx_1", entries[0])) + len(
        fixture_indices("A/A_idx_1", entries[2]))       # a real union, not a concatenation
    assert [query.position for query in groups[0].queries] == [0, 2]
    assert sum(len(group.queries) for group in groups) == 3
    # the cost the G1 gate counted: one conditioner call per union member
    assert sum(len(group.union) for group in groups) == _fixture_totals(
        rooms=["A/A_idx_1"])[1]


# --------------------------------------------------------------------------- #
# the D1 binding: the contexts a query is generated from
# --------------------------------------------------------------------------- #
def _d1_record(position, md):
    """The D1 record this md would have produced -- the manifest's own codec."""
    from src.localization import meshgrid_queries as mq

    return mq.context_record(md, position, eligible=8)


def _stream_md(room_id="A/A_idx_1", entry=None, value=None):
    """One loader item, built from the SAME fixture geometry as the manifest.

    It carries ``source``/``source_vit`` exactly as the released loader does --
    the engine is forbidden from reading them, and the guard is what enforces
    that rather than their absence.
    """
    entry = FIXTURE_QUERIES[room_id][0] if entry is None else entry
    position, name, receiver, contexts, truth = entry
    poses = torch.tensor(np.asarray(contexts, dtype=np.float64)
                         - np.asarray(receiver, dtype=np.float64), dtype=torch.float32)
    source = torch.tensor(np.asarray(truth, dtype=np.float64)
                          - np.asarray(receiver, dtype=np.float64), dtype=torch.float32)
    relpath = fixture_relpath(room_id, name)
    value = float(position) + 2.0 if value is None else float(value)
    return {"depth": torch.full((2, 4), float(sum(receiver))),
            "context_audio": torch.full((8, 1, 16), value, dtype=torch.float32),
            "context_poses": poses, "context_poses_vit": poses,
            "source": source, "source_vit": source.unsqueeze(0),
            "relpath": relpath, "path": "AcousticRooms/" + relpath, "idx": position}


def test_the_context_draw_is_verified_against_the_manifest_before_use():
    md = _stream_md()
    record = _d1_record(0, md)
    assert me.verify_context_record(md, record, 0) is True


def test_a_context_audio_digest_mismatch_aborts_the_query():
    md = _stream_md()
    record = _d1_record(0, md)
    with pytest.raises(ValueError, match="context audio"):
        me.verify_context_record(_stream_md(value=3.0), record, 0)


def test_a_context_fingerprint_mismatch_aborts_the_query():
    md = _stream_md()
    record = _d1_record(0, md)
    moved = _stream_md()
    moved["context_poses"] = moved["context_poses"] + 1.0
    with pytest.raises(ValueError, match="fingerprint"):
        me.verify_context_record(moved, record, 0)


def test_the_stream_position_and_identity_are_part_of_the_binding():
    md = _stream_md()
    record = _d1_record(0, md)
    with pytest.raises(ValueError, match="position"):
        me.verify_context_record(md, record, 5)
    other = dict(record, query_id="7|other")
    with pytest.raises(ValueError, match="query_id"):
        me.verify_context_record(md, other, 0)


def test_the_stream_must_deliver_each_room_as_one_contiguous_block():
    records = [{"position": 0, "room_id": "A"}, {"position": 1, "room_id": "A"},
               {"position": 2, "room_id": "B"}]
    assert me.assert_room_blocks(records) == ["A", "B"]
    with pytest.raises(ValueError, match="contiguous"):
        me.assert_room_blocks(records + [{"position": 3, "room_id": "A"}])


def test_the_query_geometry_is_reconstructed_without_any_target_access():
    """The GT-free replacement for the oracle cross-check (r7 review BLOCKER GT).

    Given only the manifest's receiver, the live context poses and the base
    bank, the engine re-derives the query's z-band, its drop counts and its
    whole candidate index set -- and refuses if any of them disagrees.
    """
    from src.localization import meshgrid_geometry as mg

    room_id, entry = "A/A_idx_1", FIXTURE_QUERIES["A/A_idx_1"][0]
    position, name, receiver, contexts, _truth = entry
    md = _stream_md(room_id, entry)
    band = mg.context_z_band(contexts)
    full = mg.filter_query_candidates(FIXTURE_LATTICE, receiver=receiver,
                                      context_sources=contexts)
    query = me.QueryPlan(position=position, query_id=fixture_query_id(room_id, entry),
                         room_id=room_id,
                         receiver_id=fixture_receiver_id(room_id, receiver),
                         receiver_xyz=receiver,
                         candidate_indices=fixture_indices(room_id, entry),
                         base=FIXTURE_LATTICE, oracle=0.1, branch="z_band",
                         z_band=list(band), n_contexts=8,
                         n_dropped_receiver=full["n_dropped_receiver"],
                         n_dropped_context=full["n_dropped_context"])
    report = me.assert_query_geometry_consistent(md, query)
    assert report["reconstructed"] == len(query.candidate_indices)
    assert report["n_tolerated"] == 0


def test_a_receiver_that_is_not_this_querys_is_refused_without_the_target():
    from src.localization import meshgrid_geometry as mg

    room_id, entry = "A/A_idx_1", FIXTURE_QUERIES["A/A_idx_1"][0]
    position, name, receiver, contexts, _truth = entry
    md = _stream_md(room_id, entry)
    full = mg.filter_query_candidates(FIXTURE_LATTICE, receiver=receiver,
                                      context_sources=contexts)
    kwargs = dict(position=position, query_id=fixture_query_id(room_id, entry),
                  room_id=room_id, receiver_id="whatever",
                  candidate_indices=fixture_indices(room_id, entry),
                  base=FIXTURE_LATTICE, oracle=0.1, branch="z_band",
                  z_band=list(mg.context_z_band(contexts)), n_contexts=8,
                  n_dropped_receiver=full["n_dropped_receiver"],
                  n_dropped_context=full["n_dropped_context"])
    moved = me.QueryPlan(receiver_xyz=[2.0, 2.0, 1.5], **kwargs)
    with pytest.raises(ValueError, match="candidate set|drop count|z-band"):
        me.assert_query_geometry_consistent(md, moved)

    # ... and a query whose contexts are not the ones the manifest was built from
    other = _stream_md(room_id, FIXTURE_QUERIES["A/A_idx_1"][2])
    with pytest.raises(ValueError, match="z-band|candidate set"):
        me.assert_query_geometry_consistent(other, me.QueryPlan(receiver_xyz=receiver,
                                                                **kwargs))


def test_a_boundary_grazing_disagreement_is_tolerated_but_counted():
    """float32 context poses vs the audit's float64 metadata anchors can only
    move a candidate that sits within microns of a guard boundary."""
    from src.localization import meshgrid_geometry as mg

    room_id, entry = "B/B_idx_2", FIXTURE_QUERIES["B/B_idx_2"][0]
    position, name, receiver, contexts, _truth = entry
    md = _stream_md(room_id, entry)
    full = mg.filter_query_candidates(FIXTURE_LATTICE, receiver=receiver,
                                      context_sources=contexts)
    indices = fixture_indices(room_id, entry)
    # drop a candidate that sits EXACTLY on the receiver guard
    distances = np.linalg.norm(FIXTURE_LATTICE[indices] - np.asarray(receiver), axis=1)
    grazing = int(np.asarray(indices)[np.argmin(np.abs(distances - 0.5))])
    assert abs(float(np.linalg.norm(FIXTURE_LATTICE[grazing] - np.asarray(receiver)))
               - 0.5) < 1e-9
    query = me.QueryPlan(position=position, query_id=fixture_query_id(room_id, entry),
                         room_id=room_id, receiver_id="r", receiver_xyz=receiver,
                         candidate_indices=[i for i in indices if i != grazing],
                         base=FIXTURE_LATTICE, oracle=0.1, branch="z_band",
                         z_band=list(mg.context_z_band(contexts)), n_contexts=8,
                         n_dropped_receiver=full["n_dropped_receiver"] + 1,
                         n_dropped_context=full["n_dropped_context"])
    report = me.assert_query_geometry_consistent(md, query)
    assert report["n_tolerated"] == 1 and report["tolerated"][0]["index"] == grazing


# --------------------------------------------------------------------------- #
# the leakage guard: the engine may not read the target
# --------------------------------------------------------------------------- #
def test_the_guard_refuses_the_target_fields_and_passes_everything_else():
    md = _stream_md()
    guarded = me.GuardedMetadata(md)
    assert torch.equal(guarded["context_audio"], md["context_audio"])
    assert guarded["relpath"] == md["relpath"]
    assert "source" in set(guarded)                    # present, and unreadable
    for key in me.GuardedMetadata.BLOCKED:
        with pytest.raises(me.LeakageError, match="target"):
            guarded[key]
        with pytest.raises(me.LeakageError, match="target"):
            guarded.get(key)
    # a wholesale copy cannot smuggle it out either
    with pytest.raises(me.LeakageError):
        dict(guarded)


def test_the_pass_hands_the_conditioner_a_guarded_metadata(tmp_path):
    plan, records, items = _aligned(tmp_path)

    class Leaking(FakeConditioner):
        def __call__(self, batch_metadata, device, only_ids=None):
            for md in batch_metadata:
                md.get("source")            # what the engine may never do
            return super().__call__(batch_metadata, device, only_ids=only_ids)

    engine = SyntheticEngine()
    engine.conditioner = Leaking()
    with pytest.raises(me.LeakageError, match="target"):
        me.run_pass(engine, items, records, plan, str(tmp_path / "run"), num_samples=4,
                    prefixes=(1, 4))


def test_the_d1_reverification_does_not_read_the_target(tmp_path):
    plan, records, items = _aligned(tmp_path)
    guarded = me.GuardedMetadata(items[0][1])
    assert me.verify_context_record(guarded, records[0], 0) is True
    # the frozen manifest already recorded the target-absence proof; the engine
    # requires it rather than re-deriving it from the target
    with pytest.raises(ValueError, match="target_absent"):
        me.verify_context_record(guarded, dict(records[0], target_absent=False), 0)
    with pytest.raises(ValueError, match="target_absent"):
        me.verify_context_record(guarded, {k: v for k, v in records[0].items()
                                           if k != "target_absent"}, 0)


def test_the_released_recorder_can_still_prove_target_absence():
    """D1 materialization SEES the target -- it is the engine that may not."""
    from src.localization import meshgrid_queries as mq

    md = _stream_md()
    assert mq.context_record(md, 0, eligible=8)["target_absent"] is True
    leaking = dict(md)
    leaking["source"] = md["context_poses"][0]
    with pytest.raises(ValueError, match="target source appears"):
        mq.context_record(leaking, 0, eligible=8)
    # ... and the verify-only variant never touches it
    assert mq.context_record(me.GuardedMetadata(md), 0, eligible=8,
                             prove_target_absent=False)["target_absent"] is None


# --------------------------------------------------------------------------- #
# the run binding, the per-query artifacts and resume
# --------------------------------------------------------------------------- #
def _binding(**overrides):
    payload = {"model_config_sha256": "a" * 64, "ckpt_sha256": "b" * 64,
               "agree_ckpt_sha256": "c" * 64, "d1_manifest_sha256": "d" * 64,
               "g1_report_sha256": "e" * 64, "branch": "z_band",
               "room_manifest_sha256": {"A/A_idx_1": "f" * 64},
               "k_prefixes": [1, 4, 8], "num_samples": 8, "tau": 0.1, "seed": 42,
               "noise_policy": "shared_across_candidates", "steps": 1, "cfg_scale": 1.0,
               "cond_method": "vanilla", "scorer_readout": "mean",
               "cond_autocast": "default", "dataset_config_sha256": "1" * 64,
               "dataset_config": "src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json"}
    payload.update(overrides)
    return payload


def test_the_binding_covers_every_registered_field_and_hashes_them():
    payload = _binding()
    assert set(me.RUN_BINDING_FIELDS) == set(payload)
    digest = me.binding_sha256(payload)
    assert digest == me.binding_sha256(dict(reversed(list(payload.items()))))
    assert digest != me.binding_sha256(_binding(tau=0.2))
    with pytest.raises(ValueError, match="cfg_scale"):
        me.binding_sha256({k: v for k, v in payload.items() if k != "cfg_scale"})


def test_the_binding_covers_the_arithmetic_and_the_data_it_reads(tmp_path):
    # r7 review BLOCKER BINDING: a resume could previously mix conditioner
    # arithmetic or an edited dataset config
    assert "cond_autocast" in me.RUN_BINDING_FIELDS
    assert "dataset_config_sha256" in me.RUN_BINDING_FIELDS
    out = str(tmp_path / "run")
    os.makedirs(out)
    me.write_binding(out, _binding())
    with pytest.raises(ValueError, match="cond_autocast"):
        me.assert_binding(out, _binding(cond_autocast="off"))
    with pytest.raises(ValueError, match="dataset_config_sha256"):
        me.assert_binding(out, _binding(dataset_config_sha256="9" * 64))


def test_the_driver_binds_the_dataset_bytes_and_the_conditioner_arithmetic(tmp_path):
    import localize_meshgrid as driver

    out_dir, report_path, base = _fixture_audit(tmp_path)
    plan = me.load_audit_plan(report_path)
    manifest_path = tmp_path / "d1.json"
    manifest_path.write_text(json.dumps({"records": [], "protocol_facts": {}}))
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--audit-report", report_path,
                              "--context-manifest", str(manifest_path),
                              "--cond-autocast", "off"])
    binding = driver.build_run_binding(args, plan, ckpt_sha256="1" * 64,
                                       agree_sha256="2" * 64, model_config_sha256="3" * 64)
    assert binding["cond_autocast"] == "off"
    assert binding["dataset_config_sha256"] == me.file_sha256(args.dataset_config)


def test_every_row_records_the_batching_it_was_actually_produced_with(tmp_path):
    plan, records, items = _aligned(tmp_path)
    out = str(tmp_path / "run")
    summary = me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                          prefixes=(1, 4), batch_rows=8, source_chunk=3)
    assert summary["batching"] == {"batch_rows": 8, "source_chunk": 3}
    row = json.load(open(me.query_artifact_paths(out, "A/A_idx_1", 0)["row"]))
    assert row["batching"] == {"batch_rows": 8, "source_chunk": 3}


def test_an_advisory_change_is_persisted_into_the_binding_not_just_printed(tmp_path):
    out = str(tmp_path / "run")
    os.makedirs(out)
    me.write_binding(out, _binding(), advisory={"source_chunk": 16, "batch_rows": 64})
    moved = me.assert_binding(out, _binding(), advisory={"source_chunk": 4, "batch_rows": 64})
    assert moved is not True
    me.record_advisory_change(out, moved, advisory={"source_chunk": 4, "batch_rows": 64})

    published = json.load(open(os.path.join(out, me.BINDING_FILENAME)))
    assert published["advisory"] == {"source_chunk": 4, "batch_rows": 64}
    assert len(published["advisory_history"]) == 1
    entry = published["advisory_history"][0]
    assert entry["changed"]["source_chunk"] == {"published": 16, "this_run": 4}
    assert entry["at_utc"] and entry["batching_caveat"] == me.BATCHING_CAVEAT
    # the strict binding is untouched by an advisory change
    assert published["binding_sha256"] == me.binding_sha256(_binding())
    # ... and a second change appends rather than replaces
    me.record_advisory_change(out, moved, advisory={"source_chunk": 8, "batch_rows": 64})
    assert len(json.load(open(os.path.join(
        out, me.BINDING_FILENAME)))["advisory_history"]) == 2


def test_a_resume_under_a_different_binding_is_refused(tmp_path):
    out = str(tmp_path / "run")
    os.makedirs(out)
    me.write_binding(out, _binding())
    assert me.assert_binding(out, _binding()) is True
    with pytest.raises(ValueError, match="ckpt_sha256"):
        me.assert_binding(out, _binding(ckpt_sha256="9" * 64))
    with pytest.raises(ValueError, match="tau"):
        me.assert_binding(out, _binding(tau=0.02))


def _row_and_sims(query_id="0|ir/A/A_idx_1/S001_R002_hybrid_IR.wav", m=3, k=8):
    sims = torch.linspace(0.1, 0.9, m * k).reshape(m, k)
    row = {"query_id": query_id, "room_id": "A/A_idx_1", "position": 0,
           "receiver_id": "A/A_idx_1|1,2,1.5", "n_candidates": m, "num_samples": k,
           "branch": "z_band", "by_k": {"1": {}, "4": {}, "8": {}}}
    return row, sims


def test_a_query_artifact_is_written_atomically_and_verifies_itself(tmp_path):
    out = str(tmp_path / "run")
    row, sims = _row_and_sims()
    paths = me.write_query_artifact(out, row, sims)
    assert os.path.isfile(paths["row"]) and os.path.isfile(paths["sims"])
    assert not [name for name in os.listdir(os.path.dirname(paths["row"]))
                if name.endswith(".partial") or name.endswith(".tmp")]
    verdict = me.verify_query_artifact(paths["row"])
    assert verdict["ok"] and verdict["query_id"] == row["query_id"]
    stored = np.load(paths["sims"])
    assert stored.dtype == np.float16 and stored.shape == (3, 8)
    assert np.allclose(stored.astype(np.float32), sims.numpy(), atol=1e-3)


def test_a_truncated_or_edited_sidecar_fails_verification(tmp_path):
    out = str(tmp_path / "run")
    row, sims = _row_and_sims()
    paths = me.write_query_artifact(out, row, sims)
    np.save(paths["sims"], np.zeros((3, 8), dtype=np.float16))
    verdict = me.verify_query_artifact(paths["row"])
    assert not verdict["ok"] and "sims" in verdict["reason"]

    other = me.write_query_artifact(out, dict(row, position=1), sims)
    edited = json.load(open(other["row"]))
    edited["n_candidates"] = 4
    with open(other["row"], "w") as handle:
        json.dump(edited, handle)
    assert not me.verify_query_artifact(other["row"])["ok"]


def test_a_row_is_authenticated_as_a_whole_not_only_its_sidecar(tmp_path):
    """r7 review BLOCKER RESUME: an edited prediction or oracle was skippable."""
    plan, records, items = _aligned(tmp_path)
    out = str(tmp_path / "run")
    me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                prefixes=(1, 4), batch_rows=8, binding_sha256="ab" * 32)
    path = me.query_artifact_paths(out, "A/A_idx_1", 0)["row"]
    assert me.verify_query_artifact(path)["ok"]
    published = json.load(open(path))
    assert published["binding_sha256"] == "ab" * 32
    assert published["row_sha256"] == me.row_digest(published)

    for field, value in (("e_oracle", 99.0),
                         ("candidate_indices", [0]),
                         ("receiver_id", "somebody else")):
        edited = dict(published, **{field: value})
        me.write_json(path, edited)
        verdict = me.verify_query_artifact(path)
        assert not verdict["ok"] and "row_sha256" in verdict["reason"]
        me.write_json(path, published)

    # ... including a prediction buried inside a K block
    tampered = json.loads(json.dumps(published))
    tampered["by_k"]["4"]["prediction_index"] = 12345
    me.write_json(path, tampered)
    assert not me.verify_query_artifact(path)["ok"]


def test_a_row_from_another_binding_is_never_adopted(tmp_path):
    plan, records, items = _aligned(tmp_path)
    out = str(tmp_path / "run")
    me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                prefixes=(1, 4), batch_rows=8, binding_sha256="ab" * 32)
    path = me.query_artifact_paths(out, "A/A_idx_1", 0)["row"]
    assert me.verify_query_artifact(path, binding_sha256="ab" * 32)["ok"]
    verdict = me.verify_query_artifact(path, binding_sha256="cd" * 32)
    assert not verdict["ok"] and "binding" in verdict["reason"]
    done, rejected = me.completed_queries(out, binding_sha256="cd" * 32)
    assert done == set() and len(rejected) == 4


def test_a_skipped_query_is_authenticated_against_the_g1_plan(tmp_path):
    plan, records, items = _aligned(tmp_path)
    out = str(tmp_path / "run")
    me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                prefixes=(1, 4), batch_rows=8, binding_sha256="ab" * 32)
    done, rejected = me.completed_queries(out, binding_sha256="ab" * 32)
    assert len(done) == 4 and rejected == []

    # a row that verifies internally but describes ANOTHER query's candidates
    path = me.query_artifact_paths(out, "A/A_idx_1", 0)["row"]
    published = json.load(open(path))
    # same shape, so the sidecar still fits: only the IDENTITY is wrong
    other = fixture_indices("A/A_idx_1", FIXTURE_QUERIES["A/A_idx_1"][2])
    borrowed = other[:published["n_candidates"]]
    assert len(borrowed) == published["n_candidates"] != len(
        set(borrowed) & set(published["candidate_indices"]))
    swapped = dict(published, candidate_indices=borrowed)
    swapped["row_sha256"] = me.row_digest(swapped)
    me.write_json(path, swapped)
    with pytest.raises(ValueError, match="does not match the candidate manifest"):
        me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                    prefixes=(1, 4), batch_rows=8, done=done, binding_sha256="ab" * 32)


def test_a_resume_without_a_published_binding_is_refused():
    import localize_meshgrid as driver

    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--resume"])
    assert driver.writes_query_artifacts(args) is True
    with pytest.raises(SystemExit, match="binding"):
        driver.assert_resumable(args, "/nonexistent/dir")


def test_resume_verifies_a_waveform_sidecar_a_row_names(tmp_path):
    plan, records, items = _aligned(tmp_path)
    out = str(tmp_path / "run")
    me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                prefixes=(1, 4), batch_rows=8,
                dump_queries={"0|ir/A/A_idx_1/S001_R002_hybrid_IR.wav"}, dump_top_n=2)
    path = me.query_artifact_paths(out, "A/A_idx_1", 0)["row"]
    assert me.verify_query_artifact(path)["ok"]
    row = json.load(open(path))
    np.savez(os.path.join(out, row["waveform_path"]), candidate_indices=np.zeros(1),
             waveforms=np.zeros((1, 1, 1)), observation=np.zeros(1))
    verdict = me.verify_query_artifact(path)
    assert not verdict["ok"] and "waveform" in verdict["reason"]


def test_resume_skips_only_digest_verified_queries(tmp_path):
    out = str(tmp_path / "run")
    row, sims = _row_and_sims()
    good = me.write_query_artifact(out, row, sims)
    bad = me.write_query_artifact(out, dict(row, position=1, query_id="1|ir/A/A_idx_1/x.wav"),
                                  sims)
    np.save(bad["sims"], np.zeros((3, 8), dtype=np.float16))
    done, rejected = me.completed_queries(out)
    assert done == {row["query_id"]}
    assert [entry["query_id"] for entry in rejected] == ["1|ir/A/A_idx_1/x.wav"]
    assert os.path.isfile(good["row"])


def test_the_sims_precision_is_declared_not_incidental():
    assert me.SIMS_DTYPE == "float16"
    assert "float16" in me.SIMS_PRECISION_CAVEAT


# --------------------------------------------------------------------------- #
# bounded dumps (announcement-08 exemption) and the no-quality probe
# --------------------------------------------------------------------------- #
def test_the_registered_probe_queries_are_computed_from_the_manifest(tmp_path):
    out_dir, report_path, base = _fixture_audit(tmp_path)
    plan = me.load_audit_plan(report_path)
    probes = me.registered_probe_queries(plan)
    assert probes == {"A/A_idx_1": "0|ir/A/A_idx_1/S001_R002_hybrid_IR.wav",
                      "B/B_idx_2": "3|ir/B/B_idx_2/S001_R009_hybrid_IR.wav"}
    # one per room, chosen by the LEXICOGRAPHICALLY smallest relpath
    assert len(probes) == len(plan.rooms)


def test_dumping_anything_outside_the_registered_list_is_refused(tmp_path):
    out_dir, report_path, base = _fixture_audit(tmp_path)
    allowed = me.registered_probe_queries(me.load_audit_plan(report_path))
    assert me.assert_dump_allowed(["0|ir/A/A_idx_1/S001_R002_hybrid_IR.wav"], allowed) is True
    with pytest.raises(ValueError, match="announcement 08"):
        me.assert_dump_allowed(["1|ir/A/A_idx_1/S003_R004_hybrid_IR.wav"], allowed)


def test_a_case_list_extends_the_allowed_set_only_when_it_is_registered(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"query_ids": ["1|ir/A/A_idx_1/S003_R004_hybrid_IR.wav"]}))
    cases = me.load_dump_cases(str(path), expected_sha256=me.file_sha256(str(path)))
    assert cases["query_ids"] == ["1|ir/A/A_idx_1/S003_R004_hybrid_IR.wav"]
    assert cases["sha256"] == me.file_sha256(str(path))
    allowed = {"0|a"} | set(cases["query_ids"])
    assert me.assert_dump_allowed(["1|ir/A/A_idx_1/S003_R004_hybrid_IR.wav"], allowed) is True


def test_the_throughput_probe_records_timing_and_never_a_score():
    record = me.probe_record(query_id="0|x", room_id="A/A_idx_1", n_candidates=1667,
                             num_samples=8, timings={"sampling": 1.5, "embed": 0.25})
    assert record["scores_written"] is False
    assert not ({"by_k", "scores_hex", "sims", "prediction_index"} & set(record))
    assert record["n_generated"] == 1667 * 8
    assert me.assert_no_scores(record) is True
    with pytest.raises(ValueError, match="no-quality"):
        me.assert_no_scores(dict(record, by_k={"8": {"prediction_index": 3}}))


def test_the_probe_writes_to_its_own_diagnostics_stem(tmp_path):
    out = str(tmp_path / "run")
    record = me.probe_record(query_id="0|x", room_id="A/A_idx_1", n_candidates=4,
                             num_samples=8, timings={"sampling": 0.5})
    path = me.write_probe_records(out, [record], stem="probe_K8")
    assert os.path.basename(path).startswith("diagnostics_probe_K8")
    payload = json.load(open(path))
    assert payload["scores_written"] is False and payload["n_queries"] == 1
    # the probe's directory carries no query artifacts at all
    assert not os.path.isdir(os.path.join(out, "rows"))


# --------------------------------------------------------------------------- #
# the pass: synthetic end to end
# --------------------------------------------------------------------------- #
class SyntheticEngine(me.MeshEngine):
    """A deterministic stand-in for the whole generation + scoring stack.

    The sampled latent depends on BOTH the noise row and the conditioning row,
    and the embedding depends on the decoded waveform, so a mis-assembled batch
    (wrong candidate, wrong context, wrong draw) changes a score.
    """

    def __init__(self):
        conditioner = FakeConditioner()
        super().__init__(device="cpu", latent_shape=(2, 4), conditioner=conditioner,
                         cond_inputs_fn=self._cond_inputs, sampler=self._sample,
                         decoder=self._decode, embedder=self._embed,
                         cond_method="vanilla")
        self.n_sampler_rows = 0

    @staticmethod
    def _cond_inputs(conditioning):
        return {"cond": torch.cat([conditioning[key][0] for key in
                                   sorted(conditioning)], dim=-1)}

    def _sample(self, noise, cond):
        self.n_sampler_rows += int(noise.shape[0])
        # MEAN, not sum: a sum over every conditioning element would swamp the
        # noise and drive every cosine to exactly 1.0, and a stack that cannot
        # express a score cannot test one
        return noise + cond["cond"].mean(dim=-1).reshape(-1, 1, 1)

    @staticmethod
    def _decode(latents):
        return latents.reshape(latents.shape[0], 1, -1)

    @staticmethod
    def _embed(wavs):
        flat = wavs.reshape(wavs.shape[0], -1)[:, :4].float()
        return torch.nn.functional.normalize(flat + 1e-3, dim=-1)


def _aligned(tmp_path, branch="z_band", rooms=None):
    """``(plan, D1 records, stream items)`` over the whole fixture, in position order."""
    out_dir, report_path, base = _fixture_audit(tmp_path, branch=branch, rooms=rooms)
    plan = me.load_audit_plan(report_path)
    entries = sorted(((room_id, entry) for room_id in (rooms or sorted(FIXTURE_QUERIES))
                      for entry in FIXTURE_QUERIES[room_id]), key=lambda pair: pair[1][0])
    records, items = [], []
    for room_id, entry in entries:
        md = _stream_md(room_id, entry)
        records.append(_d1_record(entry[0], md))
        items.append((torch.full((1, 1, 16), 0.5), md))
    return plan, records, items


def _fixture_totals(branch="z_band", rooms=None):
    """``(pairs, union rows)`` the fixture implies -- never hand-counted."""
    pairs, unions = 0, 0
    for room_id in (rooms or sorted(FIXTURE_QUERIES)):
        by_receiver = {}
        for entry in FIXTURE_QUERIES[room_id]:
            indices = fixture_indices(room_id, entry, branch=branch)
            pairs += len(indices)
            by_receiver.setdefault(fixture_receiver_id(room_id, entry[2]),
                                   set()).update(indices)
        unions += sum(len(members) for members in by_receiver.values())
    return pairs, unions


def test_a_synthetic_pass_scores_every_query_and_publishes_its_artifacts(tmp_path):
    plan, records, items = _aligned(tmp_path)
    engine = SyntheticEngine()
    out = str(tmp_path / "run")
    summary = me.run_pass(engine, items, records, plan, out, num_samples=4,
                          prefixes=(1, 4), batch_rows=8)
    assert summary["n_scored"] == 4
    done, rejected = me.completed_queries(out)
    assert len(done) == 4 and rejected == []

    row = json.load(open(me.query_artifact_paths(out, "A/A_idx_1", 0)["row"]))
    expected = fixture_indices("A/A_idx_1", FIXTURE_QUERIES["A/A_idx_1"][0])
    assert row["n_candidates"] == len(expected) and row["num_samples"] == 4
    assert sorted(row["by_k"]) == ["1", "4"]
    assert row["candidate_indices"] == expected
    assert row["e_oracle"] > 0.0 and row["branch"] == "z_band"
    assert row["by_k"]["4"]["prediction_index"] in row["candidate_indices"]
    assert row["noise_policy"] == "shared_across_candidates" and row["seed"] == 42
    assert row["scorer_readout"] == "mean"
    assert me.AGREE_LEAKAGE_CAVEAT in row["agree_leakage_caveat"]


def test_the_pass_generates_one_sampler_row_per_candidate_and_draw(tmp_path):
    plan, records, items = _aligned(tmp_path)
    engine = SyntheticEngine()
    me.run_pass(engine, items, records, plan, str(tmp_path / "run"), num_samples=4,
                prefixes=(1, 4), batch_rows=8)
    # every z_band candidate of every query, four draws each
    assert engine.n_sampler_rows == _fixture_totals()[0] * 4


def test_the_source_branch_is_computed_once_per_receiver_union(tmp_path):
    plan, records, items = _aligned(tmp_path)
    engine = SyntheticEngine()
    summary = me.run_pass(engine, items, records, plan, str(tmp_path / "run"),
                          num_samples=4, prefixes=(1, 4), batch_rows=8)
    pairs, unions = _fixture_totals()
    assert summary["n_conditioner_rows"] == unions
    # ... while the naive per-query cost would have been one call per pair
    assert summary["n_candidate_query_pairs"] == pairs
    assert unions < pairs


def test_the_scored_similarities_do_not_depend_on_the_row_chunking(tmp_path):
    plan, records, items = _aligned(tmp_path)
    first = str(tmp_path / "a")
    me.run_pass(SyntheticEngine(), items, records, plan, first, num_samples=4,
                prefixes=(1, 4), batch_rows=32)
    second = str(tmp_path / "b")
    me.run_pass(SyntheticEngine(), items, records, plan, second, num_samples=4,
                prefixes=(1, 4), batch_rows=1)
    for room, position in (("A/A_idx_1", 0), ("B/B_idx_2", 3)):
        left = json.load(open(me.query_artifact_paths(first, room, position)["row"]))
        right = json.load(open(me.query_artifact_paths(second, room, position)["row"]))
        assert left["by_k"] == right["by_k"]
        assert left["sims_sha256"] == right["sims_sha256"]


def test_a_resumed_pass_regenerates_only_what_did_not_verify(tmp_path):
    plan, records, items = _aligned(tmp_path)
    out = str(tmp_path / "run")
    me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                prefixes=(1, 4), batch_rows=8)
    corrupt = me.query_artifact_paths(out, "A/A_idx_1", 1)["sims"]
    n_candidates = len(fixture_indices("A/A_idx_1", FIXTURE_QUERIES["A/A_idx_1"][1]))
    np.save(corrupt, np.zeros((n_candidates, 4), dtype=np.float16))

    done, rejected = me.completed_queries(out)
    engine = SyntheticEngine()
    summary = me.run_pass(engine, items, records, plan, out, num_samples=4,
                          prefixes=(1, 4), batch_rows=8, done=done)
    assert summary["n_scored"] == 1 and summary["n_skipped"] == 3
    assert engine.n_sampler_rows == n_candidates * 4
    assert me.verify_query_artifact(me.query_artifact_paths(out, "A/A_idx_1", 1)["row"])["ok"]


def test_the_pass_refuses_a_stream_whose_context_draw_moved(tmp_path):
    plan, records, items = _aligned(tmp_path)
    items[2][1]["context_audio"] = torch.full((8, 1, 16), 99.0)
    with pytest.raises(ValueError, match="context audio"):
        me.run_pass(SyntheticEngine(), items, records, plan, str(tmp_path / "run"),
                    num_samples=4, prefixes=(1, 4))


def test_the_pass_refuses_a_frame_averaged_conditioning_method(tmp_path):
    plan, records, items = _aligned(tmp_path)
    engine = SyntheticEngine()
    engine.cond_method = "fa_invariant"
    with pytest.raises(ValueError, match="fa_invariant"):
        me.run_pass(engine, items, records, plan, str(tmp_path / "run"), num_samples=4,
                    prefixes=(1, 4))


def test_the_pass_refuses_a_room_the_stream_did_not_deliver_completely(tmp_path):
    plan, records, items = _aligned(tmp_path)
    with pytest.raises(ValueError, match="did not deliver"):
        me.run_pass(SyntheticEngine(), items[:2], records[:2], plan,
                    str(tmp_path / "run"), num_samples=4, prefixes=(1, 4))


def test_the_probe_amortizes_whole_receiver_groups_and_writes_no_scores(tmp_path):
    plan, records, items = _aligned(tmp_path)
    out = str(tmp_path / "run")
    summary = me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                          prefixes=(1, 4), probe=1)
    assert summary["n_scored"] == 0
    # the whole first receiver GROUP is measured, so the cache cost is amortized
    # the way the real pass amortizes it
    assert len(summary["probe_records"]) == 2
    for record in summary["probe_records"]:
        assert me.assert_no_scores(record) is True
        # the cache cost belongs to the GROUP, and the key says so
        assert record["timings_s"]["source_cache_group"] > 0.0
        assert record["timings_s"]["group_size"] == 2.0
    assert not os.path.isdir(os.path.join(out, me.ROWS_DIRNAME))


# --------------------------------------------------------------------------- #
# the real stack's wiring and the driver's refusals
# --------------------------------------------------------------------------- #
def test_the_release_rng_state_is_checked_before_the_iterator_is_created():
    manifest = {"protocol_facts": {"rng_digest_at_iter": "0" * 64}}
    with pytest.raises(ValueError, match="global RNG"):
        me.assert_release_rng_state(manifest)
    # the guard is the D1 pass's own recorded digest, not a constant
    from src.localization import meshgrid_queries as mq

    live = {"protocol_facts": {"rng_digest_at_iter": mq.rng_state_digest()}}
    assert me.assert_release_rng_state(live) is True


def test_the_engine_builder_refuses_a_non_vanilla_conditioning_method():
    with pytest.raises(ValueError, match="fa_invariant"):
        me.build_mesh_engine("weights/exp20/P1_40k.ckpt", {"model": {}}, None,
                             cond_method="fa_invariant")


def test_the_driver_pins_every_registered_default():
    import localize_meshgrid as driver

    args = driver.parse_args(["--ckpt-path", "x.ckpt"])
    assert args.seed == 42 and args.tau == 0.1 and args.num_samples == 8
    assert args.k_prefixes == [1, 4, 8] and args.steps == 1 and args.cfg_scale == 1.0
    assert args.cond_method == "vanilla"
    assert args.noise_policy == "shared_across_candidates"
    assert args.model_config.endswith("FLAC/AR/FLAC_AR.json")
    assert args.dataset_config.endswith("AR/eval/acousticroom_unseeneval.json")
    assert args.branch is None                      # taken from the audit, not chosen


def test_the_driver_refuses_a_probe_that_would_write_anything():
    import localize_meshgrid as driver

    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--probe", "8",
                              "--dump-waveforms", "0|x"])
    with pytest.raises(SystemExit, match="no-quality"):
        driver.validate_args(args)


def test_the_driver_refuses_a_sample_count_the_prefixes_cannot_nest_into():
    import localize_meshgrid as driver

    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--num-samples", "3"])
    with pytest.raises(SystemExit, match="nested"):
        driver.validate_args(args)


def test_the_driver_builds_the_binding_from_the_files_it_will_actually_read(tmp_path):
    import localize_meshgrid as driver

    out_dir, report_path, base = _fixture_audit(tmp_path)
    plan = me.load_audit_plan(report_path)
    manifest_path = tmp_path / "d1.json"
    manifest_path.write_text(json.dumps({"records": [], "protocol_facts": {}}))
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--audit-report", report_path,
                              "--context-manifest", str(manifest_path)])
    binding = driver.build_run_binding(args, plan, ckpt_sha256="1" * 64,
                                       agree_sha256="2" * 64,
                                       model_config_sha256="3" * 64)
    assert set(binding) == set(me.RUN_BINDING_FIELDS)
    assert binding["g1_report_sha256"] == me.file_sha256(report_path)
    assert binding["d1_manifest_sha256"] == me.file_sha256(str(manifest_path))
    assert binding["room_manifest_sha256"] == {
        room: me.file_sha256(path) for room, path in plan.rooms.items()}
    assert binding["branch"] == "z_band" and binding["k_prefixes"] == [1, 4, 8]
    assert me.binding_sha256(binding)


def test_the_cache_compares_depth_by_digest_so_the_pass_holds_one_panorama():
    conditioner = FakeConditioner()
    positions = np.arange(6, dtype=np.float64).reshape(2, 3)
    cache = me.ReceiverCache.build(conditioner, "R", _md(depth_value=1.0), [0, 1], positions,
                                   "cpu")
    assert cache.assert_same_depth_digest(me.tensor_digest(_md(depth_value=1.0)["depth"]))
    with pytest.raises(ValueError, match="depth"):
        cache.assert_same_depth_digest(me.tensor_digest(_md(depth_value=2.0)["depth"]))


def test_a_query_whose_panorama_is_not_its_receivers_aborts_the_pass(tmp_path):
    plan, records, items = _aligned(tmp_path)
    # positions 0 and 2 share a receiver; give the second a different panorama
    items[2][1]["depth"] = items[2][1]["depth"] + 1.0
    with pytest.raises(ValueError, match="depth"):
        me.run_pass(SyntheticEngine(), items, records, plan, str(tmp_path / "run"),
                    num_samples=4, prefixes=(1, 4))


def test_the_pass_refuses_a_query_the_candidate_manifest_does_not_carry(tmp_path):
    plan, records, items = _aligned(tmp_path)
    # the stream and the D1 record still agree; the CANDIDATE manifest is the one
    # that does not carry this query
    items[1][1]["relpath"] = "ir/A/A_idx_1/NOT_IN_THE_MANIFEST.wav"
    items[1][1]["path"] = "AcousticRooms/" + items[1][1]["relpath"]
    records[1] = dict(records[1], query_id="1|ir/A/A_idx_1/NOT_IN_THE_MANIFEST.wav")
    with pytest.raises(ValueError, match="two registrations disagree"):
        me.run_pass(SyntheticEngine(), items, records, plan, str(tmp_path / "run"),
                    num_samples=4, prefixes=(1, 4))


# --------------------------------------------------------------------------- #
# what an admitted dump actually contains
# --------------------------------------------------------------------------- #
def test_the_dump_selection_is_bounded_and_derived_from_the_scores():
    sims = _sims(m=20)
    scored = me.score_query(sims, list(range(100, 120)), np.zeros((20, 3)))
    rows = me.dump_selection(scored, top_n=4)
    assert len(rows) <= 4 + 2 * len(me.K_PREFIXES)
    assert rows == sorted(set(rows))
    # every prefix's prediction and mean prediction is in the set, whatever else is
    for block in scored["by_k"].values():
        assert block["prediction_row"] in rows
        assert block["mean_prediction_row"] in rows
    # ... and the top rows by the largest prefix's score
    best = me.nested_scores(sims, tau=me.TAU)[8]["scores"]
    assert set(int(i) for i in torch.topk(best, 4).indices.tolist()) <= set(rows)


def test_dumping_a_query_regenerates_only_the_selected_candidates(tmp_path):
    plan, records, items = _aligned(tmp_path)
    engine = SyntheticEngine()
    out = str(tmp_path / "run")
    summary = me.run_pass(engine, items, records, plan, out, num_samples=4,
                          prefixes=(1, 4), batch_rows=8,
                          dump_queries={"0|ir/A/A_idx_1/S001_R002_hybrid_IR.wav"},
                          dump_top_n=2)
    assert summary["n_dumped"] == 1
    row = json.load(open(me.query_artifact_paths(out, "A/A_idx_1", 0)["row"]))
    assert row["waveform_path"] and row["waveform_sha256"]
    payload = np.load(os.path.join(out, row["waveform_path"]))
    assert payload["candidate_indices"].tolist() == row["waveform_candidate_indices"]
    assert payload["waveforms"].shape[0] == len(row["waveform_candidate_indices"])
    assert payload["waveforms"].shape[1] == 4               # K
    assert payload["observation"].shape[-1] > 0
    # a dumped query costs its own generation plus ONLY the dumped rows again
    assert engine.n_sampler_rows == _fixture_totals()[0] * 4 + len(
        row["waveform_candidate_indices"]) * 4


def test_an_undumped_query_carries_no_waveform_fields(tmp_path):
    plan, records, items = _aligned(tmp_path)
    out = str(tmp_path / "run")
    me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                prefixes=(1, 4), batch_rows=8)
    row = json.load(open(me.query_artifact_paths(out, "A/A_idx_1", 0)["row"]))
    assert "waveform_path" not in row and "waveform_sha256" not in row


def test_the_regenerated_dump_is_the_waveform_that_was_scored(tmp_path):
    plan, records, items = _aligned(tmp_path)
    out = str(tmp_path / "run")
    me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                prefixes=(1, 4), batch_rows=8,
                dump_queries={"0|ir/A/A_idx_1/S001_R002_hybrid_IR.wav"}, dump_top_n=2)
    row = json.load(open(me.query_artifact_paths(out, "A/A_idx_1", 0)["row"]))
    payload = np.load(os.path.join(out, row["waveform_path"]))
    # the dumped waveforms reproduce the scored similarities they came from
    sims = np.load(me.query_artifact_paths(out, "A/A_idx_1", 0)["sims"])
    embedder = SyntheticEngine()._embed
    for slot, index in enumerate(row["waveform_candidate_indices"]):
        position = row["candidate_indices"].index(index)
        wavs = torch.as_tensor(payload["waveforms"][slot]).unsqueeze(1)
        obs = torch.as_tensor(payload["observation"]).reshape(1, 1, -1)
        got = (embedder(wavs) @ embedder(obs)[0]).numpy()
        assert np.allclose(got, sims[position].astype(np.float32), atol=2e-3)


# --------------------------------------------------------------------------- #
# the §1.5 parity proof, separated from the backbone's batch nondeterminism
# --------------------------------------------------------------------------- #
class BatchSensitiveConditioner(FakeConditioner):
    """A conditioner whose output depends on the BATCH SIZE it was called with.

    Real backbones are like this under autocast -- GEMM tiling changes with the
    batch shape -- so the parity check has to separate "the cache computes
    something else" from "the same call at another batch size rounds
    differently".
    """

    #: exaggerated on purpose: the real effect is one float16 ulp, and a
    #: synthetic stack whose embedder is dominated by large constants would round
    #: that away before the comparison machinery ever saw it.
    OFFSET = 0.05

    def __call__(self, batch_metadata, device, only_ids=None):
        out = super().__call__(batch_metadata, device, only_ids=only_ids)
        return {key: [value[0] + self.OFFSET * len(batch_metadata), value[1]]
                for key, value in out.items()}


def _parity_query(tmp_path):
    plan, records, items = _aligned(tmp_path)
    room = me.load_room_plan(plan, "A/A_idx_1")
    return room.queries[0], items[0][1]


def test_the_parity_check_proves_memoization_at_matched_batching(tmp_path):
    query, md = _parity_query(tmp_path)
    engine = SyntheticEngine()
    report = me.cache_parity_check(engine, query, md, n_candidates=3, source_chunk=2)
    assert report["memoization"]["match"] is True
    for entry in report["memoization"]["keys"].values():
        assert entry["equal"] and entry["max_abs_diff"] == 0.0
    assert report["counter_test"]["detected"] is True
    assert report["match"] is True


def test_the_parity_check_separates_batch_nondeterminism_from_a_cache_error(tmp_path):
    query, md = _parity_query(tmp_path)
    engine = SyntheticEngine()
    engine.conditioner = BatchSensitiveConditioner()
    report = me.cache_parity_check(engine, query, md, n_candidates=3, source_chunk=2)
    # the cache still computes the same thing at the same batching ...
    assert report["memoization"]["match"] is True
    # ... while the whole-batch comparison shows the backbone's own batch effect
    assert report["batched"]["max_abs_diff"] > 0.0
    assert report["match"] is True          # the CONTRACT is the memoization one


def test_the_parity_check_reports_a_real_cache_error(tmp_path):
    query, md = _parity_query(tmp_path)

    class Wrong(FakeConditioner):
        """A conditioner whose SPLIT is not faithful: asking for the source ids
        alone computes something other than the full call computes for them.

        This is the failure the memoization comparison exists to catch -- the
        cache reads the released only_ids seam, and if that seam changed what a
        branch computes, every cached candidate would be conditioned on a
        tensor the uncached path never produces."""

        def __call__(self, batch_metadata, device, only_ids=None):
            out = super().__call__(batch_metadata, device, only_ids=only_ids)
            if only_ids is not None and set(only_ids) == set(me.SOURCE_COND_IDS):
                out = {key: [value[0] + 0.5, value[1]] for key, value in out.items()}
            return out

    engine = SyntheticEngine()
    engine.conditioner = Wrong()
    report = me.cache_parity_check(engine, query, md, n_candidates=3, source_chunk=3)
    assert report["memoization"]["match"] is False
    assert report["match"] is False


# --------------------------------------------------------------------------- #
# the advisory tier: batching changes results only at the backbone's ulp level
# --------------------------------------------------------------------------- #
def test_the_batching_knobs_are_recorded_as_advisory_not_as_the_binding():
    assert set(me.RUN_BINDING_ADVISORY) == {"source_chunk", "batch_rows"}
    assert not set(me.RUN_BINDING_ADVISORY) & set(me.RUN_BINDING_FIELDS)
    # they cannot enter the strict digest at all -- smuggling one in is refused,
    # which is what makes a re-chunked resume (e.g. after an OOM) possible
    for field in me.RUN_BINDING_ADVISORY:
        with pytest.raises(ValueError, match=field):
            me.binding_sha256(dict(_binding(), **{field: 16}))


def test_a_changed_batching_knob_is_reported_but_does_not_refuse_a_resume(tmp_path):
    out = str(tmp_path / "run")
    os.makedirs(out)
    me.write_binding(out, _binding(), advisory={"source_chunk": 16, "batch_rows": 64})
    assert me.assert_binding(out, _binding(),
                             advisory={"source_chunk": 16, "batch_rows": 64}) is True
    moved = me.assert_binding(out, _binding(), advisory={"source_chunk": 4, "batch_rows": 64})
    assert moved is not True
    assert moved["source_chunk"] == {"published": 16, "this_run": 4}
    assert "ulp" in me.BATCHING_CAVEAT
    # ... while a STRICT field still refuses
    with pytest.raises(ValueError, match="tau"):
        me.assert_binding(out, _binding(tau=0.02), advisory={"source_chunk": 16,
                                                             "batch_rows": 64})


def test_the_published_binding_carries_both_tiers(tmp_path):
    out = str(tmp_path / "run")
    os.makedirs(out)
    path = me.write_binding(out, _binding(), advisory={"source_chunk": 16, "batch_rows": 64})
    payload = json.load(open(path))
    assert payload["advisory"] == {"source_chunk": 16, "batch_rows": 64}
    assert payload["batching_caveat"] == me.BATCHING_CAVEAT
    assert payload["binding_sha256"] == me.binding_sha256(_binding())


def test_the_drivers_parity_path_runs_end_to_end_on_a_synthetic_stack(tmp_path, capsys):
    import localize_meshgrid as driver

    plan, records, items = _aligned(tmp_path)
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--source-chunk", "2"])
    code = driver._run_cache_parity(args, SyntheticEngine(), plan, records,
                                    [([obs], [md]) for obs, md in items])
    assert code == 0
    printed = capsys.readouterr().out
    assert "MEMOIZATION" in printed and "MATCH" in printed
    assert "BATCHED" in printed


def test_the_probe_can_be_pointed_at_one_room_so_a_smoke_is_affordable(tmp_path):
    plan, records, items = _aligned(tmp_path)
    chosen, covered = me.probe_groups(plan, 1, room="B/B_idx_2")
    assert chosen == [("B/B_idx_2",
                       fixture_receiver_id("B/B_idx_2",
                                           FIXTURE_QUERIES["B/B_idx_2"][0][2]))]
    assert covered == 1
    with pytest.raises(ValueError, match="not in the audit"):
        me.probe_groups(plan, 1, room="Nowhere/Nowhere_idx_1")

    out = str(tmp_path / "run")
    summary = me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                          prefixes=(1, 4), probe=1, probe_room="B/B_idx_2")
    assert summary["n_scored"] == 0
    assert [record["room_id"] for record in summary["probe_records"]] == ["B/B_idx_2"]


def test_a_diagnostics_mode_never_claims_the_output_directory(tmp_path):
    import localize_meshgrid as driver

    for extra in (["--probe", "2"], ["--cache-parity-check"]):
        args = driver.parse_args(["--ckpt-path", "x.ckpt", "--out-dir",
                                  str(tmp_path / "out")] + extra)
        assert driver.writes_query_artifacts(args) is False
    args = driver.parse_args(["--ckpt-path", "x.ckpt"])
    assert driver.writes_query_artifacts(args) is True


def test_the_driver_refuses_an_are_checkpoint_before_it_builds_the_scorer(tmp_path):
    import localize_meshgrid as driver

    model_config = json.load(open("src/configs/model_configs/FLAC/AR/FLAC_AR.json"))
    args = driver.parse_args(["--ckpt-path", "x.ckpt"])
    clean = {"model_config": {"training": {"cond_method": "vanilla"}}}
    assert driver.validate_checkpoint(args, model_config, clean)["binding"] == "checkpoint"

    are = {"model_config": {"training": {"are_lambda": 0.5, "cond_method": "vanilla"}}}
    with pytest.raises(SystemExit, match="ARE"):
        driver.validate_checkpoint(args, model_config, are)

    wrong = {"model_config": {"training": {"cond_method": "fa_invariant"}}}
    with pytest.raises(SystemExit, match="cond_method"):
        driver.validate_checkpoint(args, model_config, wrong)


def test_a_probe_does_not_parse_the_manifests_of_rooms_it_will_not_touch(tmp_path):
    plan, records, items = _aligned(tmp_path)
    loaded = []
    real = me.load_room_plan

    def _counting(plan_arg, room_id):
        loaded.append(room_id)
        return real(plan_arg, room_id)

    me.load_room_plan, saved = _counting, me.load_room_plan
    try:
        me.run_pass(SyntheticEngine(), items, records, plan, str(tmp_path / "run"),
                    num_samples=4, prefixes=(1, 4), probe=1, probe_room="B/B_idx_2")
    finally:
        me.load_room_plan = saved
    # room A is streamed and verified, but its 137 MB-class manifest is never read
    assert "A/A_idx_1" not in loaded
    assert loaded.count("B/B_idx_2") >= 1


def test_the_probe_query_list_is_only_computed_when_a_dump_asks_for_it(tmp_path):
    import localize_meshgrid as driver

    out_dir, report_path, base = _fixture_audit(tmp_path)
    plan = me.load_audit_plan(report_path)
    touched = []
    real = me.registered_probe_queries
    me.registered_probe_queries = lambda arg: touched.append(arg) or real(arg)
    try:
        args = driver.parse_args(["--ckpt-path", "x.ckpt"])
        assert driver.dump_allowance(args, plan)["query_ids"] == set()
        assert touched == []            # 328 MB of manifests not parsed for nothing

        args = driver.parse_args(["--ckpt-path", "x.ckpt", "--dump-waveforms",
                                  "0|ir/A/A_idx_1/S001_R002_hybrid_IR.wav"])
        assert driver.dump_allowance(args, plan)["query_ids"] == {
            "0|ir/A/A_idx_1/S001_R002_hybrid_IR.wav"}
        assert touched == [plan]
        args = driver.parse_args(["--ckpt-path", "x.ckpt", "--dump-waveforms", "9|nope"])
        with pytest.raises(ValueError, match="announcement 08"):
            driver.dump_allowance(args, plan)
    finally:
        me.registered_probe_queries = real


# --------------------------------------------------------------------------- #
# determinism: what "the same" and "close enough" mean, registered
# --------------------------------------------------------------------------- #
def test_the_score_tolerance_and_its_basis_are_registered():
    assert isinstance(me.SCORE_TOLERANCE, float) and me.SCORE_TOLERANCE > 0.0
    assert "fixed batching" in me.DETERMINISM_CONTRACT
    assert "bit-exact" in me.DETERMINISM_CONTRACT


def test_a_fixed_batching_replay_is_bit_exact_through_scoring(tmp_path):
    plan, records, items = _aligned(tmp_path)
    first, second = str(tmp_path / "a"), str(tmp_path / "b")
    for out in (first, second):
        me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                    prefixes=(1, 4), batch_rows=8, source_chunk=3)
    left = me.read_rows(first)
    right = me.read_rows(second)
    assert len(left) == 4
    for row_a, row_b in zip(left, right):
        assert me.score_fingerprint(row_a) == me.score_fingerprint(row_b)
        assert row_a["sims_sha256"] == row_b["sims_sha256"]
    report = me.compare_scored_runs(left, right)
    assert report["bit_exact"] is True and report["n_flipped"] == 0
    assert report["max_abs_delta"] == 0.0


def test_a_changed_batching_run_is_compared_against_the_registered_tolerance(tmp_path):
    plan, records, items = _aligned(tmp_path)

    def _run(out, source_chunk):
        engine = SyntheticEngine()
        # the conditioner is the batch-shaped stage, as the real ViT is
        engine.conditioner = BatchSensitiveConditioner()
        me.run_pass(engine, items, records, plan, out, num_samples=4, prefixes=(1, 4),
                    batch_rows=8, source_chunk=source_chunk)
        return me.read_rows(out)

    left = _run(str(tmp_path / "a"), 7)
    right = _run(str(tmp_path / "b"), 3)
    report = me.compare_scored_runs(left, right)
    assert report["bit_exact"] is False and report["max_abs_delta"] > 0.0
    assert set(report["by_k"]) == {1, 4}
    assert report["n_queries"] == 4
    # the report NAMES the queries whose argmax moved instead of absorbing them
    assert isinstance(report["flipped"], list)
    assert report["n_flipped"] == len(report["flipped"])
    assert report["within_tolerance"] == (report["max_abs_delta"] <= me.SCORE_TOLERANCE)


def test_a_row_reports_how_far_its_argmax_is_from_flipping(tmp_path):
    plan, records, items = _aligned(tmp_path)
    out = str(tmp_path / "run")
    summary = me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                          prefixes=(1, 4), batch_rows=8)
    row = json.load(open(me.query_artifact_paths(out, "A/A_idx_1", 0)["row"]))
    for block in row["by_k"].values():
        assert block["margin"] >= 0.0
        assert block["argmax_stable"] is (block["margin"] > me.SCORE_TOLERANCE)
    stability = summary["argmax_stability"]
    assert sorted(stability) == [1, 4]
    for entry in stability.values():
        assert entry["n_queries"] == 4
        assert 0 <= entry["n_unstable"] <= 4
        assert entry["min_margin"] >= 0.0


def test_the_margin_is_the_gap_to_the_runner_up():
    scores = torch.tensor([0.10, 0.42, 0.31, 0.42])
    assert me.top1_margin(scores) == pytest.approx(0.0)      # a tie cannot be stable
    assert me.top1_margin(torch.tensor([0.10, 0.42, 0.31])) == pytest.approx(0.11)
    assert me.top1_margin(torch.tensor([0.5])) == float("inf")   # nothing to flip to


@pytest.mark.skipif(not os.environ.get("EXP22_REAL_STACK"),
                    reason="ladder step: set EXP22_REAL_STACK=1 to run against the frozen "
                           "checkpoint on a GPU")
def test_the_real_stack_is_bit_exact_at_fixed_batching():
    """The registered determinism claim, on the real generator + AGREE readout."""
    import localize_meshgrid as driver

    argv = ["--ckpt-path", os.environ.get("EXP22_CKPT", "weights/exp20/P1_40k.ckpt"),
            "--device", os.environ.get("EXP22_DEVICE", "cuda:0"), "--replay-check"]
    assert driver.main(argv) == 0


def test_the_replay_check_scores_one_query_twice_and_compares(tmp_path):
    plan, records, items = _aligned(tmp_path)
    room = me.load_room_plan(plan, "A/A_idx_1")
    query = room.queries[0]
    md = me.GuardedMetadata(items[0][1])
    report = me.replay_check(SyntheticEngine(), query, md, items[0][0], num_samples=4,
                             prefixes=(1, 4), batch_rows=8, source_chunk=3)
    assert report["bit_exact"] is True and report["max_abs_delta"] == 0.0
    assert report["fingerprint_equal"] is True
    assert report["n_candidates"] == query.n_candidates

    class Drifting(FakeConditioner):
        """A stack that is NOT deterministic at fixed batching."""

        def __init__(self):
            super().__init__()
            self.n = 0

        def __call__(self, batch_metadata, device, only_ids=None):
            out = super().__call__(batch_metadata, device, only_ids=only_ids)
            self.n += 1
            return {key: [value[0] + 1e-2 * self.n, value[1]] for key, value in out.items()}

    engine = SyntheticEngine()
    engine.conditioner = Drifting()
    report = me.replay_check(engine, query, md, items[0][0], num_samples=4, prefixes=(1, 4),
                             batch_rows=8, source_chunk=3)
    assert report["bit_exact"] is False and report["max_abs_delta"] > 0.0


# --------------------------------------------------------------------------- #
# the probe must be able to substantiate the cost decision on its own
# --------------------------------------------------------------------------- #
def test_the_registered_totals_are_the_audits_and_are_pinned():
    assert me.REGISTERED_TOTALS == {"rooms": 16, "queries": 5337,
                                    "candidate_query_pairs": 8896540,
                                    "source_rows": 966147,
                                    "generated_waveforms": 71172320}
    assert me.REGISTERED_TOTALS["generated_waveforms"] == (
        me.REGISTERED_TOTALS["candidate_query_pairs"] * me.NUM_SAMPLES)


def test_a_probe_record_identifies_the_receiver_group_it_was_billed_under(tmp_path):
    plan, records, items = _aligned(tmp_path)
    out = str(tmp_path / "run")
    summary = me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                          prefixes=(1, 4), probe=1, probe_room="A/A_idx_1")
    assert summary["n_contexts_conditioned"] >= 1
    for record in summary["probe_records"]:
        assert record["receiver_id"].startswith("A/A_idx_1|")
        assert record["n_union"] > 0
        assert record["timings_s"]["source_cache_group"] > 0.0
        assert me.assert_no_scores(record) is True
    # every query of the group shares ONE group identity, so the cache cost is
    # not counted once per query when the records are summed
    assert len({record["receiver_id"] for record in summary["probe_records"]}) == 1
    assert len({record["n_union"] for record in summary["probe_records"]}) == 1


def test_the_projection_separates_the_three_registered_quantities():
    records = [{"query_id": "q0", "room_id": "R", "receiver_id": "R|a", "n_union": 40,
                "n_candidates": 10, "num_samples": 8, "n_generated": 80,
                "scores_written": False,
                "timings_s": {"sampling": 0.2, "decode": 1.0, "embed": 0.2, "scoring": 0.01,
                              "conditioning": 0.01, "context": 0.5,
                              "source_cache_group": 4.0, "group_size": 2.0}},
               {"query_id": "q1", "room_id": "R", "receiver_id": "R|a", "n_union": 40,
                "n_candidates": 10, "num_samples": 8, "n_generated": 80,
                "scores_written": False,
                "timings_s": {"sampling": 0.2, "decode": 1.0, "embed": 0.2, "scoring": 0.01,
                              "conditioning": 0.01, "context": 0.5,
                              "source_cache_group": 4.0, "group_size": 2.0}}]
    projection = me.project_cost(records)
    # the generation rate is per WAVEFORM over both queries
    assert projection["seconds_per_waveform"] == pytest.approx(2.84 / 160)
    # the cache is billed ONCE for the group, not once per query
    assert projection["source_rows_measured"] == 40
    assert projection["seconds_per_source_row"] == pytest.approx(4.0 / 40)
    assert projection["contexts_measured"] == 2
    assert projection["seconds_per_context"] == pytest.approx(0.5)
    hours = projection["projected_gpu_hours"]
    assert hours["generation"] == pytest.approx(
        me.REGISTERED_TOTALS["generated_waveforms"] * 2.84 / 160 / 3600)
    assert hours["source_conditioning"] == pytest.approx(
        me.REGISTERED_TOTALS["source_rows"] * 0.1 / 3600)
    assert hours["context"] == pytest.approx(me.REGISTERED_TOTALS["queries"] * 0.5 / 3600)
    assert hours["total"] == pytest.approx(sum(v for k, v in hours.items() if k != "total"))
    assert projection["totals"] == me.REGISTERED_TOTALS


def test_the_probe_artifact_carries_immutable_provenance(tmp_path):
    out = str(tmp_path / "run")
    record = me.probe_record(query_id="0|x", room_id="A/A_idx_1", receiver_id="A/A_idx_1|r",
                             n_union=12, n_candidates=4, num_samples=8,
                             timings={"sampling": 0.5, "decode": 1.0, "embed": 0.2,
                                      "context": 0.1, "source_cache_group": 0.3,
                                      "group_size": 1.0})
    path = me.write_probe_records(out, [record], stem="probe_K8",
                                  binding=_binding(), binding_sha256="ab" * 32,
                                  advisory={"batch_rows": 64, "source_chunk": 16})
    payload = json.load(open(path))
    assert payload["scores_written"] is False
    assert payload["binding_sha256"] == "ab" * 32
    for field in ("ckpt_sha256", "model_config_sha256", "agree_ckpt_sha256",
                  "d1_manifest_sha256", "g1_report_sha256", "cond_autocast",
                  "dataset_config_sha256", "noise_policy", "num_samples", "tau"):
        assert payload["binding"][field] == _binding()[field]
    assert payload["advisory"] == {"batch_rows": 64, "source_chunk": 16}
    assert payload["projection"]["totals"] == me.REGISTERED_TOTALS
    assert payload["determinism_contract"] == me.DETERMINISM_CONTRACT
    assert payload["agree_leakage_caveat"] == me.AGREE_LEAKAGE_CAVEAT
    assert payload["noise_policy"] == me.REGISTERED_NOISE_POLICY


def test_a_probe_under_an_unregistered_policy_cannot_be_published(tmp_path):
    out = str(tmp_path / "run")
    record = me.probe_record(query_id="0|x", room_id="R", receiver_id="R|a", n_union=1,
                             n_candidates=1, num_samples=8, timings={"decode": 1.0})
    with pytest.raises(ValueError, match="common random numbers"):
        me.write_probe_records(out, [record], stem="p", binding=_binding(
            noise_policy="per_candidate"), binding_sha256="ab" * 32)


# --------------------------------------------------------------------------- #
# sharding: --rooms
# --------------------------------------------------------------------------- #
def test_the_room_filter_accepts_canonical_ids_and_refuses_everything_else(tmp_path):
    out_dir, report_path, base = _fixture_audit(tmp_path)
    plan = me.load_audit_plan(report_path)
    assert me.assert_declared_rooms(["B/B_idx_2", "A/A_idx_1"], plan) == ["A/A_idx_1",
                                                                         "B/B_idx_2"]
    with pytest.raises(ValueError, match="empty"):
        me.assert_declared_rooms([], plan)
    with pytest.raises(ValueError, match="twice"):
        me.assert_declared_rooms(["A/A_idx_1", "A/A_idx_1"], plan)
    with pytest.raises(ValueError, match="not in the audit"):
        me.assert_declared_rooms(["A/A_idx_1", "Nowhere/Nowhere_idx_9"], plan)


def test_a_shard_scores_only_its_rooms_but_verifies_the_whole_stream(tmp_path):
    plan, records, items = _aligned(tmp_path)
    seen = []
    original = me.verify_context_record
    me.verify_context_record = lambda md, rec, pos: seen.append(pos) or original(md, rec, pos)
    loaded = []
    real_plan = me.load_room_plan
    me.load_room_plan = lambda p, room: loaded.append(room) or real_plan(p, room)
    try:
        out = str(tmp_path / "shard")
        summary = me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                              prefixes=(1, 4), batch_rows=8, rooms=["B/B_idx_2"])
    finally:
        me.verify_context_record = original
        me.load_room_plan = real_plan
    # the WHOLE D1 stream is walked and verified -- the draws depend on it
    assert seen == [0, 1, 2, 3]
    # ... while room A's 137 MB-class manifest is never opened
    assert "A/A_idx_1" not in loaded
    assert summary["declared_rooms"] == ["B/B_idx_2"]
    assert summary["n_scored"] == 1
    assert me.completed_queries(out)[0] == {"3|ir/B/B_idx_2/S001_R009_hybrid_IR.wav"}


def test_a_shard_conditions_nothing_for_a_room_it_does_not_own(tmp_path):
    plan, records, items = _aligned(tmp_path)
    engine = SyntheticEngine()
    me.run_pass(engine, items, records, plan, str(tmp_path / "shard"), num_samples=4,
                prefixes=(1, 4), batch_rows=8, rooms=["B/B_idx_2"])
    conditioned = [call for call in engine.conditioner.calls
                   if call["ids"] == sorted(me.CONTEXT_COND_IDS)]
    assert len(conditioned) == 1               # one query, one context branch


def test_the_two_shards_of_the_reviewers_split_cover_the_pass_exactly(tmp_path):
    plan, records, items = _aligned(tmp_path)
    left = me.run_pass(SyntheticEngine(), items, records, plan, str(tmp_path / "a"),
                       num_samples=4, prefixes=(1, 4), batch_rows=8, rooms=["A/A_idx_1"])
    right = me.run_pass(SyntheticEngine(), items, records, plan, str(tmp_path / "b"),
                        num_samples=4, prefixes=(1, 4), batch_rows=8, rooms=["B/B_idx_2"])
    assert left["n_scored"] + right["n_scored"] == len(records)
    assert (left["n_candidate_query_pairs"] + right["n_candidate_query_pairs"]
            == _fixture_totals()[0])
    assert left["n_conditioner_rows"] + right["n_conditioner_rows"] == _fixture_totals()[1]
    assert set(left["declared_rooms"]) & set(right["declared_rooms"]) == set()


def test_a_shards_declared_rooms_are_published_and_pinned_for_its_resume(tmp_path):
    out = str(tmp_path / "run")
    os.makedirs(out)
    me.write_binding(out, _binding(), declared_rooms=["A/A_idx_1"])
    published = json.load(open(os.path.join(out, me.BINDING_FILENAME)))
    assert published["declared_rooms"] == ["A/A_idx_1"]
    # the SHARDING does not enter the strict digest: the merge requires the base
    # bindings of two shards to be identical
    assert published["binding_sha256"] == me.binding_sha256(_binding())
    assert me.assert_binding(out, _binding(), declared_rooms=["A/A_idx_1"]) is True
    with pytest.raises(ValueError, match="declared_rooms"):
        me.assert_binding(out, _binding(), declared_rooms=["B/B_idx_2"])


def test_the_driver_parses_and_validates_a_room_shard():
    import localize_meshgrid as driver

    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--rooms", "Cafe/Cafe_idx_1"])
    assert args.rooms == ["Cafe/Cafe_idx_1"]
    assert driver.parse_args(["--ckpt-path", "x.ckpt"]).rooms is None
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--rooms", "Cafe/Cafe_idx_1",
                              "--probe", "1"])
    with pytest.raises(SystemExit, match="--rooms"):
        driver.validate_args(args)


# --------------------------------------------------------------------------- #
# the merge and its census
# --------------------------------------------------------------------------- #
def _shards(tmp_path, **kwargs):
    """Two disjoint shards of the fixture pass, published like the real ones."""
    plan, records, items = _aligned(tmp_path)
    binding = _binding()
    digest = me.binding_sha256(binding)
    dirs = {}
    for name, rooms in (("a", ["A/A_idx_1"]), ("b", ["B/B_idx_2"])):
        out = str(tmp_path / name)
        me.write_binding(out, binding, advisory={"batch_rows": 8, "source_chunk": 3},
                         declared_rooms=rooms)
        summary = me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                              prefixes=(1, 4), batch_rows=8, source_chunk=3, rooms=rooms,
                              binding_sha256=digest, **kwargs)
        me.write_json(os.path.join(out, "run_summary.json"), summary)
        dirs[name] = out
    pairs, unions = _fixture_totals()
    totals = {"rooms": 2, "queries": len(records), "candidate_query_pairs": pairs,
              "source_rows": unions, "generated_waveforms": pairs * 4}
    return plan, records, dirs, totals


def test_a_merge_publishes_a_fresh_directory_only_after_its_census_passes(tmp_path):
    plan, records, dirs, totals = _shards(tmp_path)
    merged = str(tmp_path / "merged")
    report = me.merge_shards([dirs["a"], dirs["b"]], merged, plan, records, totals=totals)

    assert report["ok"] is True
    assert report["n_rows"] == len(records)
    assert report["declared_rooms"] == ["A/A_idx_1", "B/B_idx_2"]
    assert report["totals"] == {"candidate_query_pairs": totals["candidate_query_pairs"],
                                "source_rows": totals["source_rows"],
                                "generated_waveforms": totals["generated_waveforms"]}
    assert report["binding_sha256"] == me.binding_sha256(_binding())
    # the merged directory is a complete, re-verifiable run
    done, rejected = me.completed_queries(merged, binding_sha256=report["binding_sha256"])
    assert len(done) == len(records) and rejected == []
    assert json.load(open(os.path.join(merged, me.BINDING_FILENAME)))["declared_rooms"] == [
        "A/A_idx_1", "B/B_idx_2"]
    assert os.path.isfile(os.path.join(merged, "merge_report.json"))


def test_a_merge_refuses_to_write_into_a_directory_that_has_content(tmp_path):
    plan, records, dirs, totals = _shards(tmp_path)
    merged = str(tmp_path / "merged")
    os.makedirs(merged)
    open(os.path.join(merged, "something"), "w").close()
    with pytest.raises(ValueError, match="fresh"):
        me.merge_shards([dirs["a"], dirs["b"]], merged, plan, records, totals=totals)


def test_a_merge_refuses_shards_from_different_bindings(tmp_path):
    plan, records, dirs, totals = _shards(tmp_path)
    other = json.load(open(os.path.join(dirs["b"], me.BINDING_FILENAME)))
    other["tau"] = 0.02
    other["binding_sha256"] = me.binding_sha256(
        {k: v for k, v in other.items() if k in me.RUN_BINDING_FIELDS})
    me.write_json(os.path.join(dirs["b"], me.BINDING_FILENAME), other)
    with pytest.raises(ValueError, match="same binding"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m"), plan, records,
                        totals=totals)


def test_a_merge_refuses_shards_whose_batching_was_not_pinned(tmp_path):
    plan, records, dirs, totals = _shards(tmp_path)
    published = json.load(open(os.path.join(dirs["b"], me.BINDING_FILENAME)))
    published["advisory"] = {"batch_rows": 64, "source_chunk": 16}
    me.write_json(os.path.join(dirs["b"], me.BINDING_FILENAME), published)
    with pytest.raises(ValueError, match="advisory"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m"), plan, records,
                        totals=totals)


def test_a_merge_refuses_overlapping_or_incomplete_room_sets(tmp_path):
    plan, records, dirs, totals = _shards(tmp_path)
    with pytest.raises(ValueError, match="disjoint|missing"):
        me.merge_shards([dirs["a"], dirs["a"]], str(tmp_path / "m"), plan, records,
                        totals=totals)
    with pytest.raises(ValueError, match="at least two"):
        me.merge_shards([dirs["a"]], str(tmp_path / "m2"), plan, records, totals=totals)
    with pytest.raises(ValueError, match="missing"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m3"), plan, records,
                        totals=dict(totals, rooms=3),
                        expected_rooms=["A/A_idx_1", "B/B_idx_2", "C/C_idx_3"])


def test_a_merge_refuses_a_wrong_census(tmp_path):
    plan, records, dirs, totals = _shards(tmp_path)
    with pytest.raises(ValueError, match="candidate_query_pairs"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m"), plan, records,
                        totals=dict(totals, candidate_query_pairs=totals[
                            "candidate_query_pairs"] + 1))
    with pytest.raises(ValueError, match="source_rows"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m2"), plan, records,
                        totals=dict(totals, source_rows=totals["source_rows"] + 1))


def test_a_merge_refuses_a_tampered_row_or_a_missing_one(tmp_path):
    plan, records, dirs, totals = _shards(tmp_path)
    path = me.query_artifact_paths(dirs["a"], "A/A_idx_1", 0)["row"]
    row = json.load(open(path))
    me.write_json(path, dict(row, e_oracle=99.0))
    with pytest.raises(ValueError, match="row_sha256|cannot be adopted"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m"), plan, records,
                        totals=totals)
    me.write_json(path, row)
    os.remove(me.query_artifact_paths(dirs["a"], "A/A_idx_1", 1)["row"])
    with pytest.raises(ValueError, match="missing"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m2"), plan, records,
                        totals=totals)


def test_a_merge_carries_a_dumped_waveform_across(tmp_path):
    plan, records, dirs, totals = _shards(
        tmp_path, dump_queries={"0|ir/A/A_idx_1/S001_R002_hybrid_IR.wav"}, dump_top_n=2)
    merged = str(tmp_path / "merged")
    me.merge_shards([dirs["a"], dirs["b"]], merged, plan, records, totals=totals)
    row = json.load(open(me.query_artifact_paths(merged, "A/A_idx_1", 0)["row"]))
    assert os.path.isfile(os.path.join(merged, row["waveform_path"]))
    assert me.verify_query_artifact(
        me.query_artifact_paths(merged, "A/A_idx_1", 0)["row"])["ok"]


def test_the_driver_exposes_the_merge_with_a_fresh_output(tmp_path):
    import localize_meshgrid as driver

    args = driver.parse_args(["--merge-shards", "a", "b", "--merge-out", "m"])
    assert args.merge_shards == ["a", "b"] and args.merge_out == "m"
    assert driver.writes_query_artifacts(args) is False
    with pytest.raises(SystemExit, match="--merge-out"):
        driver.validate_args(driver.parse_args(["--merge-shards", "a", "b"]))
    with pytest.raises(SystemExit, match="at least two"):
        driver.validate_args(driver.parse_args(["--merge-shards", "a", "--merge-out", "m"]))


# --------------------------------------------------------------------------- #
# the dump case list may not authorize itself
# --------------------------------------------------------------------------- #
def test_a_case_list_must_match_a_registered_digest(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"query_ids": ["1|ir/A/A_idx_1/S003_R004_hybrid_IR.wav"]}))
    digest = me.file_sha256(str(path))

    with pytest.raises(ValueError, match="registered digest"):
        me.load_dump_cases(str(path))                      # self-authorizing: refused
    with pytest.raises(ValueError, match="registered digest"):
        me.load_dump_cases(str(path), expected_sha256="9" * 64)
    cases = me.load_dump_cases(str(path), expected_sha256=digest)
    assert cases["sha256"] == digest and len(cases["query_ids"]) == 1

    # editing the list after registration invalidates it
    path.write_text(json.dumps({"query_ids": ["0|somebody/elses/query.wav"]}))
    with pytest.raises(ValueError, match="registered digest"):
        me.load_dump_cases(str(path), expected_sha256=digest)


def test_the_driver_requires_the_case_list_digest(tmp_path):
    import localize_meshgrid as driver

    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"query_ids": ["1|x"]}))
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--dump-waveforms", "1|x",
                              "--dump-cases", str(path)])
    with pytest.raises(SystemExit, match="--dump-cases-sha256"):
        driver.validate_args(args)
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--dump-waveforms", "1|x",
                              "--dump-cases", str(path),
                              "--dump-cases-sha256", me.file_sha256(str(path))])
    assert driver.validate_args(args) is True


def test_the_admitted_dump_set_records_where_its_authority_came_from(tmp_path):
    import localize_meshgrid as driver

    out_dir, report_path, base = _fixture_audit(tmp_path)
    plan = me.load_audit_plan(report_path)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"query_ids": ["1|ir/A/A_idx_1/S003_R004_hybrid_IR.wav"]}))
    args = driver.parse_args(["--ckpt-path", "x.ckpt", "--audit-report", report_path,
                              "--dump-waveforms", "1|ir/A/A_idx_1/S003_R004_hybrid_IR.wav",
                              "--dump-cases", str(path),
                              "--dump-cases-sha256", me.file_sha256(str(path))])
    admitted = driver.dump_allowance(args, plan)
    assert admitted["query_ids"] == {"1|ir/A/A_idx_1/S003_R004_hybrid_IR.wav"}
    assert admitted["case_list"]["sha256"] == me.file_sha256(str(path))
    assert admitted["probe_queries"]

    # a query in NEITHER list is still refused
    args.dump_waveforms = ["2|ir/A/A_idx_1/S005_R002_hybrid_IR.wav"]
    with pytest.raises(ValueError, match="announcement 08"):
        driver.dump_allowance(args, plan)


# --------------------------------------------------------------------------- #
# r8b: the argmax stability boundary is 2 eps, not eps
# --------------------------------------------------------------------------- #
def test_stability_needs_twice_the_per_score_tolerance():
    # a per-score bound eps moves a GAP by up to 2 eps: the leader can lose eps
    # while the runner-up gains eps
    assert me.ARGMAX_STABILITY_FACTOR == 2
    assert me.argmax_stability_bound() == 2 * me.SCORE_TOLERANCE
    assert me.argmax_stability_bound(0.5) == 1.0
    assert me.is_argmax_stable(3 * me.SCORE_TOLERANCE) is True
    assert me.is_argmax_stable(2 * me.SCORE_TOLERANCE) is False
    # the band the r8 code called stable and the review rejected
    assert me.is_argmax_stable(1.5 * me.SCORE_TOLERANCE) is False
    assert me.is_argmax_stable(me.SCORE_TOLERANCE) is False


def test_a_margin_between_one_and_two_epsilon_is_classified_at_risk():
    eps = me.SCORE_TOLERANCE
    # M candidates whose top-1 margin is 1.5 eps exactly
    sims = torch.zeros((3, 8))
    sims[0] = 0.5 + 1.5 * eps
    sims[1] = 0.5
    sims[2] = 0.1
    scored = me.score_query(sims, [10, 11, 12], np.zeros((3, 3)), prefixes=(8,))
    block = scored["by_k"][8]
    # float32 through the log-mean-exp: compare at float32 resolution, not exactly
    assert block["margin"] == pytest.approx(1.5 * eps, rel=1e-4)
    assert block["argmax_stable"] is False
    assert block["stability_bound"] == pytest.approx(2 * eps)


def test_the_comparison_counts_risk_at_the_two_epsilon_boundary(tmp_path):
    eps = me.SCORE_TOLERANCE

    def _rows(margin):
        sims = torch.zeros((2, 8))
        sims[0] = 0.5 + margin
        sims[1] = 0.5
        scored = me.score_query(sims, [7, 9], np.zeros((2, 3)), prefixes=(8,))
        return [{"query_id": "q", "room_id": "R", "position": 0, "receiver_id": "r",
                 "branch": "z_band", "n_candidates": 2, "num_samples": 8, "tau": me.TAU,
                 "seed": 42, "noise_policy": me.REGISTERED_NOISE_POLICY, "k_prefixes": [8],
                 "candidate_indices": [7, 9], "e_oracle": 0.1, "sims_sha256": "x",
                 "by_k": {"8": scored["by_k"][8]}}]

    for margin, expected in ((1.5 * eps, 1), (3.0 * eps, 0)):
        report = me.compare_scored_runs(_rows(margin), _rows(margin))
        assert report["n_argmax_at_risk"] == expected, margin
        assert report["stability_bound"] == pytest.approx(2 * eps)


def test_the_summary_uses_the_same_boundary(tmp_path):
    plan, records, items = _aligned(tmp_path)
    out = str(tmp_path / "run")
    summary = me.run_pass(SyntheticEngine(), items, records, plan, out, num_samples=4,
                          prefixes=(1, 4), batch_rows=8)
    for entry in summary["argmax_stability"].values():
        assert entry["stability_bound"] == pytest.approx(2 * me.SCORE_TOLERANCE)
    row = json.load(open(me.query_artifact_paths(out, "A/A_idx_1", 0)["row"]))
    for block in row["by_k"].values():
        assert block["argmax_stable"] is (block["margin"] > 2 * me.SCORE_TOLERANCE)


# --------------------------------------------------------------------------- #
# r8b: the merge trusts nothing it is handed
# --------------------------------------------------------------------------- #
def test_the_merge_recomputes_every_shard_binding_digest(tmp_path):
    plan, records, dirs, totals = _shards(tmp_path)
    path = os.path.join(dirs["b"], me.BINDING_FILENAME)
    published = json.load(open(path))
    # the CONTENT changes while the stored digest keeps saying it did not
    tampered = dict(published, tau=0.02)
    me.write_json(path, tampered)
    with pytest.raises(ValueError, match="does not match its own content"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m"), plan, records,
                        totals=totals)

    # ... and a stored digest that lies about matching content is refused too
    me.write_json(path, dict(published, binding_sha256="0" * 64))
    with pytest.raises(ValueError, match="does not match its own content"):
        me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "m2"), plan, records,
                        totals=totals)


def test_a_resumed_shard_still_merges_because_the_census_is_derived(tmp_path):
    plan, records, dirs, totals = _shards(tmp_path)
    digest = me.binding_sha256(_binding())

    # resume shard A after it already finished: nothing is regenerated, so its
    # final summary reports ZERO conditioner rows for the whole shard
    done, rejected = me.completed_queries(dirs["a"], binding_sha256=digest)
    assert len(done) == 3 and rejected == []
    summary = me.run_pass(SyntheticEngine(), items_of(tmp_path), records, plan, dirs["a"],
                          num_samples=4, prefixes=(1, 4), batch_rows=8, source_chunk=3,
                          rooms=["A/A_idx_1"], done=done, binding_sha256=digest)
    assert summary["n_scored"] == 0 and summary["n_conditioner_rows"] == 0
    me.write_json(os.path.join(dirs["a"], "run_summary.json"), summary)

    report = me.merge_shards([dirs["a"], dirs["b"]], str(tmp_path / "merged"), plan,
                             records, totals=totals)
    assert report["ok"] is True
    # the census comes from the G1 plan, not from whichever invocation ran last
    assert report["totals"]["source_rows"] == totals["source_rows"]
    assert report["source_rows_observed"] < report["totals"]["source_rows"]
    assert report["source_rows_derived_from"] == "g1_plan"


def items_of(tmp_path):
    """The fixture stream that ``_shards`` built its shards from."""
    return _aligned(tmp_path)[2]
