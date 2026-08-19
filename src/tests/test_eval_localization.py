"""Tests for ``eval_localization.py`` (exp_18 loc_invert, round 3 driver).

Written test-first (announcement 02). Contracts: ``loc_invert_impl_contracts.md``
§4.5, serving plan Rev 3 §2. The driver owns the fail-closed identity audit
(§2.1), the keyed noise bank (§2.3), the candidate-major query layout, the JSONL
row/summary records and the ``eval_FLAC`` parity harness (C8); everything the
``src.localization`` modules already provide is reused, never re-derived.
"""
import hashlib

import pytest

import eval_localization as el


# --------------------------------------------------------------------------- #
# fixtures: a split whose identities can be built without loading any audio
# --------------------------------------------------------------------------- #
class _FakeDataset:
    """Stand-in for ``SampleDataset``: the ordered file list plus its roots."""

    def __init__(self, filenames, root_paths):
        self.filenames = list(filenames)
        self.root_paths = list(root_paths)

    def __len__(self):
        return len(self.filenames)


_ROOT = "/data/AcousticRooms"
_FILES = [
    f"{_ROOT}/single_channel_ir_1/Cafe/Cafe_idx_1/S00{s}_R003_hybrid_IR.wav" for s in (0, 7)
] + [f"{_ROOT}/single_channel_ir_1/Bedrooms/Bedrooms_idx_2/S002_R0011_hybrid_IR.wav"]


def _md(idx, filename, root=_ROOT):
    import os
    return {"idx": idx, "path": filename, "relpath": os.path.relpath(filename, root)}


def _loader(dataset, batch_size=2):
    """Mimic the eval dataloader: yields ``(reals, [md, ...])`` batches in order."""
    metadata = [_md(i, f) for i, f in enumerate(dataset.filenames)]
    for start in range(0, len(metadata), batch_size):
        chunk = metadata[start:start + batch_size]
        yield (None, chunk)


# --------------------------------------------------------------------------- #
# expected_split_identities / split_hash / audit_split_identities  (unit a)
# --------------------------------------------------------------------------- #
def test_expected_split_identities_mirrors_the_loader_without_reading_audio():
    dataset = _FakeDataset(_FILES, [_ROOT])
    expected = el.expected_split_identities(dataset)
    assert expected == [
        "0|single_channel_ir_1/Cafe/Cafe_idx_1/S000_R003_hybrid_IR.wav",
        "1|single_channel_ir_1/Cafe/Cafe_idx_1/S007_R003_hybrid_IR.wav",
        "2|single_channel_ir_1/Bedrooms/Bedrooms_idx_2/S002_R0011_hybrid_IR.wav",
    ]


def test_expected_split_identities_falls_back_to_the_absolute_path():
    dataset = _FakeDataset(_FILES[:1], ["/somewhere/else"])
    assert el.expected_split_identities(dataset) == [f"0|{_FILES[0]}"]


def test_split_hash_is_the_pinned_serialization():
    """LF-joined identities, UTF-8, sha256 -- pinned so the recorded split hash is
    comparable across machines and runs."""
    ids = ["0|a.wav", "1|b.wav"]
    assert el.split_hash(ids) == hashlib.sha256("0|a.wav\n1|b.wav".encode("utf-8")).hexdigest()
    assert el.split_hash(ids) != el.split_hash(list(reversed(ids)))       # order matters


def test_audit_split_identities_clean_pass_returns_the_split_hash():
    dataset = _FakeDataset(_FILES, [_ROOT])
    expected = el.expected_split_identities(dataset)
    assert el.audit_split_identities(_loader(dataset), expected) == el.split_hash(expected)


def test_audit_split_identities_aborts_on_an_injected_substitution():
    """SampleDataset silently substitutes a random other item on a load failure;
    the first such position must abort the run (plan §2.1)."""
    dataset = _FakeDataset(_FILES, [_ROOT])
    expected = el.expected_split_identities(dataset)

    def substituted():
        for reals, batch in _loader(dataset):
            patched = [dict(md) for md in batch]
            for md in patched:
                if md["idx"] == 1:
                    md["idx"], md["relpath"] = 2, "single_channel_ir_1/X/X_idx_0/S001_R001_hybrid_IR.wav"
            yield (reals, patched)

    with pytest.raises(SystemExit):
        el.audit_split_identities(substituted(), expected)


def test_audit_split_identities_aborts_on_a_short_or_long_stream():
    dataset = _FakeDataset(_FILES, [_ROOT])
    expected = el.expected_split_identities(dataset)
    short = [(None, [_md(0, _FILES[0])])]
    with pytest.raises(SystemExit):
        el.audit_split_identities(short, expected)
    with pytest.raises(SystemExit):
        el.audit_split_identities(_loader(dataset), expected[:2])


def test_audit_split_identities_accepts_a_plain_metadata_iterable():
    dataset = _FakeDataset(_FILES, [_ROOT])
    expected = el.expected_split_identities(dataset)
    stream = [_md(i, f) for i, f in enumerate(dataset.filenames)]
    assert el.audit_split_identities(stream, expected) == el.split_hash(expected)


# --------------------------------------------------------------------------- #
# build_noise_bank  (unit b) -- plan §2.3: deterministic, keyed by
# (seed, query_id, k), shared across candidates (common random numbers, C10),
# and never drawn from the global stream.
# --------------------------------------------------------------------------- #
import torch                                    # noqa: E402
from src.localization.scoring import noise_key  # noqa: E402

_LATENT = (4, 16)


def test_build_noise_bank_shape_and_dtype():
    bank = el.build_noise_bank(42, "q0", 8, _LATENT)
    assert tuple(bank.shape) == (8, 4, 16)
    assert bank.dtype == torch.float32 and bank.device.type == "cpu"


def test_build_noise_bank_is_keyed_by_scoring_noise_key():
    """Pins the bank to scoring.noise_key: sample k is exactly what a generator
    seeded with noise_key(seed, query_id, k) draws."""
    bank = el.build_noise_bank(42, "Cafe/Cafe_idx_1/S008_R089", 3, _LATENT)
    for k in range(3):
        g = torch.Generator().manual_seed(noise_key(42, "Cafe/Cafe_idx_1/S008_R089", k))
        assert torch.equal(bank[k], torch.randn(_LATENT, generator=g))


def test_build_noise_bank_is_reproducible_and_query_specific():
    a = el.build_noise_bank(42, "q0", 4, _LATENT)
    assert torch.equal(a, el.build_noise_bank(42, "q0", 4, _LATENT))      # resume-safe
    assert not torch.equal(a, el.build_noise_bank(42, "q1", 4, _LATENT))
    assert not torch.equal(a, el.build_noise_bank(43, "q0", 4, _LATENT))
    assert not torch.equal(a[0], a[1])                                    # k is load-bearing


def test_build_noise_bank_never_touches_the_global_rng():
    before = torch.random.get_rng_state()
    el.build_noise_bank(42, "q0", 8, _LATENT)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_build_noise_bank_prefix_is_stable_in_k():
    """A K=8 bank starts with the K=4 bank: enlarging K cannot renumber samples."""
    small = el.build_noise_bank(42, "q0", 4, _LATENT)
    large = el.build_noise_bank(42, "q0", 8, _LATENT)
    assert torch.equal(large[:4], small)


def test_build_noise_bank_rejects_bad_arguments():
    with pytest.raises(ValueError):
        el.build_noise_bank(42, "q0", 0, _LATENT)
    with pytest.raises(ValueError):
        el.build_noise_bank(42, "q0", 4, (4,))
    with pytest.raises(ValueError):
        el.build_noise_bank(42, "q0", 4, (4, 0))


# --------------------------------------------------------------------------- #
# run_query  (unit c) -- candidate-major layout [m*K + k], one conditioner call
# over the M candidate metadata dicts, common random numbers across candidates.
# The engine is the seam: real callables in a run, recording fakes here.
# --------------------------------------------------------------------------- #
import numpy as np                                                    # noqa: E402
from src.localization.candidates import CandidateSet, project_to_camera  # noqa: E402

_REC = np.array([1.0, 2.0, 0.5])
_NODES = [0, 3, 7]
_XYZ = np.array([[0.0, 0.0, 1.0], [2.0, -1.0, 1.5], [-3.0, 4.0, 0.75]])


def _cand_set(nodes=None, xyz=None, gt_node=3):
    nodes = _NODES if nodes is None else nodes
    xyz = _XYZ if xyz is None else xyz
    return CandidateSet(nodes=nodes, xyz_world=xyz, rec_loc=_REC,
                        gt_node=gt_node, gt_xyz=xyz[nodes.index(gt_node)])


def _base_md(cand_set=None):
    cand_set = cand_set or _cand_set()
    source = torch.as_tensor(project_to_camera(cand_set.rec_loc, cand_set.gt_xyz),
                             dtype=torch.float32)
    return {"scene": "Cafe", "idx": 7, "source": source, "source_vit": source.unsqueeze(0),
            "depth": torch.zeros(3, 4, 8), "context_audio": torch.zeros(2, 1, 16)}


class _RecordingEngine:
    """Fake generation stack that keeps candidate identity and noise identity
    distinguishable in the output, and records exactly what each row received."""

    def __init__(self, latent=(2, 8), embed_dim=6):
        self.latent, self.embed_dim = latent, embed_dim
        self.device, self.io_channels, self.latent_samples = "cpu", latent[0], latent[1]
        self.calls, self.seen_metadata = [], []

    def conditioner(self, metadata, device):
        self.seen_metadata.append(metadata)
        return {"source": (torch.stack([md["source"] for md in metadata]), None)}

    def cond_inputs_fn(self, conditioning):
        return {"global_cond": conditioning["source"][0], "cross_attn_cond": None}

    def sampler(self, noise, cond_inputs):
        self.calls.append({"noise": noise.clone(), "global_cond": cond_inputs["global_cond"].clone()})
        return noise + cond_inputs["global_cond"].sum(-1)[:, None, None]

    def decoder(self, latents):
        return latents.reshape(latents.shape[0], 1, -1)

    def embedder(self, wavs):
        feats = wavs.reshape(wavs.shape[0], -1)[:, : self.embed_dim]
        return torch.nn.functional.normalize(feats.float(), dim=-1)


def _engine_kwargs(rec):
    return dict(device=rec.device, io_channels=rec.io_channels, latent_samples=rec.latent_samples,
                conditioner=rec.conditioner, cond_inputs_fn=rec.cond_inputs_fn,
                sampler=rec.sampler, decoder=rec.decoder, embedder=rec.embedder)


def _engine():
    rec = _RecordingEngine()
    return rec, el.Engine(**_engine_kwargs(rec))


_OBS = torch.ones(1, 1, 16) * 0.1


def test_run_query_layout_is_candidate_major():
    """Row m*K + k must carry candidate m's conditioning and noise draw k."""
    rec, engine = _engine()
    cand = _cand_set()
    noise = el.build_noise_bank(42, "q0", 4, (2, 8))
    out = el.run_query(engine, _base_md(cand), cand, noise, _OBS, batch_size=64)

    assert tuple(out["sims"].shape) == (3, 4) and out["sims"].dtype == torch.float32
    assert len(rec.calls) == 1 and len(rec.seen_metadata) == 1        # one conditioner call
    seen_noise = rec.calls[0]["noise"]
    seen_cond = rec.calls[0]["global_cond"]
    cams = torch.as_tensor(out["cand_cam_xyz"], dtype=torch.float32)
    for m in range(3):
        for k in range(4):
            row = m * 4 + k
            assert torch.equal(seen_noise[row], noise[k])
            assert torch.equal(seen_cond[row], cams[m])


def test_run_query_is_invariant_to_batch_splitting():
    rec_a, engine_a = _engine()
    rec_b, engine_b = _engine()
    cand, noise = _cand_set(), el.build_noise_bank(42, "q0", 4, (2, 8))
    whole = el.run_query(engine_a, _base_md(cand), cand, noise, _OBS, batch_size=64)
    split = el.run_query(engine_b, _base_md(cand), cand, noise, _OBS, batch_size=5)
    assert len(rec_b.calls) > len(rec_a.calls)                        # really did split
    assert torch.equal(whole["sims"], split["sims"])


def test_run_query_per_candidate_results_do_not_depend_on_the_other_candidates():
    """Common random numbers per (query, k): dropping a candidate must not change
    what the retained candidates generate.

    The generated waveforms are compared BITWISE -- that is the pipeline claim.
    The similarities are compared at atol 1e-7 because ``cosine_sims`` reduces a
    [M, K, D] batched matmul, which reassociates differently at M=3 and M=2
    (measured 6e-08); that is a float32 property of the scoring reduction, not a
    dependence of one candidate on another.
    """
    _rec, engine = _engine()
    noise = el.build_noise_bank(42, "q0", 3, (2, 8))
    full = el.run_query(engine, _base_md(), _cand_set(), noise, _OBS, return_wavs=True)
    subset_nodes, subset_xyz = [0, 3], _XYZ[:2]
    subset = _cand_set(nodes=subset_nodes, xyz=subset_xyz)
    _rec2, engine2 = _engine()
    part = el.run_query(engine2, _base_md(subset), subset, noise, _OBS, return_wavs=True)
    assert torch.equal(full["wavs"][: 2 * 3], part["wavs"])           # 2 candidates x K=3
    assert torch.allclose(full["sims"][:2], part["sims"], rtol=0, atol=1e-7)


def test_run_query_permuting_the_noise_bank_permutes_the_sample_axis():
    _rec, engine = _engine()
    cand = _cand_set()
    noise = el.build_noise_bank(42, "q0", 4, (2, 8))
    base = el.run_query(engine, _base_md(cand), cand, noise, _OBS)
    order = [2, 0, 3, 1]
    _rec2, engine2 = _engine()
    permuted = el.run_query(engine2, _base_md(cand), cand, noise[order], _OBS)
    assert torch.equal(permuted["sims"], base["sims"][:, order])


def test_run_query_constant_source_control_passes_one_identical_position():
    """§2.8.1: every candidate is conditioned on the SAME centroid position, so a
    working pipeline must collapse to the context-conditioned baseline."""
    rec, engine = _engine()
    cand = _cand_set()
    noise = el.build_noise_bank(42, "q0", 2, (2, 8))
    out = el.run_query(engine, _base_md(cand), cand, noise, _OBS, control="constant_source")

    passed = torch.stack([md["source"] for md in rec.seen_metadata[0]])
    centroid = torch.as_tensor(
        np.stack([project_to_camera(cand.rec_loc, xyz) for xyz in cand.xyz_world]).mean(axis=0),
        dtype=torch.float32)
    assert torch.allclose(passed, centroid.expand(3, 3), atol=0)
    assert out["control"] == "constant_source"
    assert torch.equal(out["sims"][0], out["sims"][1])                # identical conditioning


def test_run_query_enforces_the_gt_geometry_invariant():
    _rec, engine = _engine()
    cand = _cand_set()
    md = _base_md(cand)
    md["source"] = md["source"] + 1e-3
    with pytest.raises(AssertionError):
        el.run_query(engine, md, cand, el.build_noise_bank(42, "q0", 2, (2, 8)), _OBS)


def test_run_query_does_not_mutate_the_base_metadata():
    _rec, engine = _engine()
    cand = _cand_set()
    md = _base_md(cand)
    original = md["source"].clone()
    el.run_query(engine, md, cand, el.build_noise_bank(42, "q0", 2, (2, 8)), _OBS)
    assert torch.equal(md["source"], original)
    assert set(md) >= {"scene", "depth", "context_audio"}


def test_run_query_returns_candidate_camera_positions():
    _rec, engine = _engine()
    cand = _cand_set()
    out = el.run_query(engine, _base_md(cand), cand, el.build_noise_bank(1, "q", 2, (2, 8)), _OBS)
    expected = np.stack([project_to_camera(cand.rec_loc, xyz) for xyz in cand.xyz_world])
    np.testing.assert_array_equal(out["cand_cam_xyz"], expected)


# --------------------------------------------------------------------------- #
# row helpers  (unit d, part 1): exact sims serialization, room key, context
# membership by the eval_FLAC fingerprint rendering, GT reciprocal rank.
# --------------------------------------------------------------------------- #
from eval_FLAC import sample_context_ids                              # noqa: E402


def test_encode_decode_sims_round_trips_bitwise():
    """float.hex() is exact for a float32 value widened to float64, so an offline
    re-aggregation reproduces the online scores exactly (O18)."""
    g = torch.Generator().manual_seed(3)
    sims = (torch.rand(4, 8, generator=g) * 2 - 1).float()
    payload = el.encode_sims(sims)
    assert isinstance(payload, list) and isinstance(payload[0][0], str)
    assert torch.equal(el.decode_sims(payload), sims)


def test_encode_sims_survives_a_json_round_trip():
    import json
    sims = torch.tensor([[0.1, -0.7], [1.0, -1.0]], dtype=torch.float32)
    restored = el.decode_sims(json.loads(json.dumps(el.encode_sims(sims))))
    assert torch.equal(restored, sims)
    assert restored.dtype == torch.float32


def test_encode_sims_captures_awkward_values_exactly():
    sims = torch.tensor([[float(np.float32(1 / 3)), -0.0, 1.0]], dtype=torch.float32)
    assert torch.equal(el.decode_sims(el.encode_sims(sims)), sims)


def test_room_id_from_relpath():
    assert el.room_id_from_relpath(
        "single_channel_ir_1/Cafe/Cafe_idx_1/S000_R003_hybrid_IR.wav") == "Cafe/Cafe_idx_1"
    with pytest.raises(ValueError):
        el.room_id_from_relpath("S000_R003_hybrid_IR.wav")


def test_context_membership_mask_matches_the_eval_flac_fingerprint():
    """The context sources are candidates too; membership is decided by rendering
    each candidate's camera-frame position with the SAME 6-decimal rule
    eval_FLAC.sample_context_ids applies to context_poses."""
    cams = np.array([[0.5, -1.25, 2.0], [-3.0, 0.0, 1.0], [7.125, 2.5, -0.5]])
    poses = torch.as_tensor(cams[[0, 2]], dtype=torch.float32)
    context_ids = sample_context_ids({"context_poses": poses})
    mask = el.context_membership_mask(cams, context_ids)
    assert list(mask) == [True, False, True]
    assert el.context_membership_mask(cams, []) == [False, False, False]


def test_context_membership_mask_normalizes_negative_zero():
    cams = np.array([[-0.0, 0.0, 1.0]])
    poses = torch.as_tensor(np.array([[0.0, -0.0, 1.0]]), dtype=torch.float32)
    assert el.context_membership_mask(cams, sample_context_ids({"context_poses": poses})) == [True]


def test_gt_reciprocal_rank_uses_lowest_index_tie_break():
    scores = torch.tensor([0.9, 0.5, 0.7])
    assert el.gt_reciprocal_rank(scores, 0) == pytest.approx(1.0)
    assert el.gt_reciprocal_rank(scores, 2) == pytest.approx(0.5)
    assert el.gt_reciprocal_rank(scores, 1) == pytest.approx(1 / 3)
    tied = torch.tensor([0.9, 0.9, 0.9])
    assert el.gt_reciprocal_rank(tied, 0) == pytest.approx(1.0)     # lowest index wins ties
    assert el.gt_reciprocal_rank(tied, 2) == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# build_row / write_row / read_rows  (unit d, part 2)
# --------------------------------------------------------------------------- #
def _row_kwargs(**over):
    cand = _cand_set()
    cams = el.candidate_camera_positions(cand)
    kwargs = dict(
        query_id="7|single_channel_ir_1/Cafe/Cafe_idx_1/S003_R011_hybrid_IR.wav",
        room_id="Cafe/Cafe_idx_1",
        relpath="single_channel_ir_1/Cafe/Cafe_idx_1/S003_R011_hybrid_IR.wav",
        receiver_node=11,
        cand_set=cand,
        cam_xyz=cams,
        sims=torch.tensor([[0.10, 0.20], [0.90, 0.80], [0.30, 0.30]], dtype=torch.float32),
        context_mask=[True, False, False],
        noise_keys=[111, 222],
        tau=0.02, agg="lme", control="none", score_source="flac", smoke=False,
    )
    kwargs.update(over)
    return kwargs


def test_build_row_has_the_full_schema_and_correct_derived_fields():
    row = el.build_row(**_row_kwargs())
    assert row["query_id"].endswith("S003_R011_hybrid_IR.wav")
    assert row["room_id"] == "Cafe/Cafe_idx_1" and row["receiver_node"] == 11
    assert row["gt_node"] == 3 and row["gt_index"] == 1
    assert row["candidate_nodes"] == [0, 3, 7] and row["n_candidates"] == 3
    assert row["n_samples"] == 2 and row["noise_keys"] == [111, 222]
    assert row["context_member"] == [True, False, False]
    assert row["n_eligible"] == 2 and row["gt_only"] is False
    assert row["pred_index"] == 1 and row["pred_node"] == 3       # candidate 3 scores highest
    assert row["e_loc"] == pytest.approx(0.0) and row["top1"] == 1.0
    assert row["rr"] == pytest.approx(1.0)
    assert row["tau"] == 0.02 and row["agg"] == "lme"
    assert row["control"] == "none" and row["score_source"] == "flac" and row["smoke"] is False
    assert row["substituted"] is False and row["candidate_available"] == [True] * 3
    assert len(row["candidate_xyz_world"]) == 3 and len(row["candidate_xyz_cam"][0]) == 3
    assert row["gt_xyz_world"] == pytest.approx(list(_XYZ[1]))


def test_build_row_scores_and_error_follow_the_registered_aggregation():
    from src.localization.scoring import aggregate, localization_error
    kwargs = _row_kwargs(sims=torch.tensor([[0.9, 0.9], [0.1, 0.1], [0.5, 0.5]],
                                           dtype=torch.float32))
    row = el.build_row(**kwargs)
    expected = aggregate(kwargs["sims"], "lme", 0.02)
    assert torch.equal(el.decode_scores(row["scores_hex"]), expected)
    assert row["pred_index"] == 0 and row["pred_node"] == 0
    assert row["e_loc"] == pytest.approx(localization_error(_XYZ[0], _XYZ[1]))
    # GT (0.1) ranks behind candidate 0 (0.9) and candidate 2 (0.5) -> rank 3
    assert row["top1"] == 0.0 and row["rr"] == pytest.approx(1 / 3)


def test_build_row_gt_only_and_substitution_flags():
    row = el.build_row(**_row_kwargs(context_mask=[True, False, True], substituted=True))
    assert row["n_eligible"] == 1 and row["gt_only"] is True and row["substituted"] is True


def test_build_row_restricts_the_prediction_to_available_candidates():
    """gt_rir mode: a missing measured file shrinks the ORACLE's eligibility, never
    the candidate set (plan §2.2)."""
    kwargs = _row_kwargs(sims=torch.tensor([[0.99], [0.10], [0.50]], dtype=torch.float32),
                         noise_keys=[], available=[False, True, True],
                         score_source="gt_rir", identity_index=1)
    row = el.build_row(**kwargs)
    assert row["candidate_available"] == [False, True, True] and row["n_available"] == 2
    assert row["n_candidates"] == 3                       # candidate set NOT shrunk
    assert row["pred_index"] == 2                         # candidate 0 excluded from the argmax
    assert row["identity_index"] == 1 and row["score_source"] == "gt_rir"


def test_build_row_rejects_an_empty_available_set():
    with pytest.raises(ValueError):
        el.build_row(**_row_kwargs(available=[False, False, False]))


def test_jsonl_round_trip_is_bitwise_exact(tmp_path):
    rows = [el.build_row(**_row_kwargs()),
            el.build_row(**_row_kwargs(query_id="8|other.wav", smoke=True))]
    path = tmp_path / "rows.jsonl"
    with open(path, "w") as handle:
        for row in rows:
            el.write_row(handle, row)
    restored = el.read_rows(path)
    assert restored == rows
    for original, back in zip(rows, restored):
        assert torch.equal(el.decode_sims(back["sims_hex"]), el.decode_sims(original["sims_hex"]))
    assert torch.equal(el.decode_sims(restored[0]["sims_hex"]), _row_kwargs()["sims"])


def test_write_row_appends_and_flushes_per_row(tmp_path):
    path = tmp_path / "rows.jsonl"
    with open(path, "a") as handle:
        el.write_row(handle, el.build_row(**_row_kwargs()))
        assert len(el.read_rows(path)) == 1        # readable before the file is closed
        el.write_row(handle, el.build_row(**_row_kwargs(query_id="9|x.wav")))
    assert len(el.read_rows(path)) == 2


# --------------------------------------------------------------------------- #
# summarize_run  (unit e, part 1) -- scoring.summarize on the rows, both random
# baselines under IDENTICAL conventions, and the non-generative control.
# --------------------------------------------------------------------------- #
from src.localization.scoring import (context_conditioned_baseline,  # noqa: E402
                                      nearest_context_baseline, summarize,
                                      uniform_baseline)


def _rows_fixture():
    """Two rooms, three queries; query 2 is the GT-only (zero-headroom) case."""
    cand = _cand_set()
    cams = el.candidate_camera_positions(cand)
    common = dict(cand_set=cand, cam_xyz=cams, receiver_node=11, noise_keys=[1, 2],
                  tau=0.02, agg="lme", control="none", score_source="flac", smoke=False)
    specs = [
        ("q0", "Cafe/Cafe_idx_1", [[0.1, 0.1], [0.9, 0.9], [0.2, 0.2]], [True, False, False]),
        ("q1", "Cafe/Cafe_idx_1", [[0.9, 0.9], [0.1, 0.1], [0.2, 0.2]], [False, False, True]),
        ("q2", "Bedrooms/Bedrooms_idx_2", [[0.2, 0.2], [0.5, 0.5], [0.9, 0.9]], [True, False, True]),
    ]
    rows = []
    for query_id, room_id, sims, mask in specs:
        rows.append(el.build_row(
            query_id=query_id, room_id=room_id,
            relpath=f"single_channel_ir_1/{room_id}/S003_R011_hybrid_IR.wav",
            sims=torch.tensor(sims, dtype=torch.float32), context_mask=mask, **common))
    return rows


def test_summarize_run_flac_block_equals_scoring_summarize_on_the_same_rows():
    rows = _rows_fixture()
    summary = el.summarize_run(rows, radii=(0.5, 1.0))
    expected = summarize([{"query_id": r["query_id"], "room_id": r["room_id"],
                           "e_loc": r["e_loc"], "top1": r["top1"], "rr": r["rr"]} for r in rows],
                         radii=(0.5, 1.0))
    assert summary["flac"] == expected
    assert summary["n_queries"] == 3 and summary["n_rooms"] == 2


def test_summarize_run_baselines_use_identical_weighting():
    rows = _rows_fixture()
    summary = el.summarize_run(rows, radii=(0.5, 1.0))

    uniform_records, context_records = [], []
    for row in rows:
        cand = np.asarray(row["candidate_xyz_world"])
        gt = np.asarray(row["gt_xyz_world"])
        u = uniform_baseline(cand, gt, radii=(0.5, 1.0))
        c = context_conditioned_baseline(cand, gt, row["context_member"], radii=(0.5, 1.0))
        uniform_records.append({"query_id": row["query_id"], "room_id": row["room_id"],
                                "distances": u["distances"], "top1": u["top1"]})
        context_records.append({"query_id": row["query_id"], "room_id": row["room_id"],
                                "distances": c["distances"], "top1": c["top1"]})
    assert summary["baselines"]["uniform"] == summarize(uniform_records, radii=(0.5, 1.0))
    assert summary["baselines"]["context_conditioned"] == summarize(context_records,
                                                                    radii=(0.5, 1.0))


def test_summarize_run_reports_the_gt_only_room_separately_and_excludes_it():
    rows = _rows_fixture()
    summary = el.summarize_run(rows)
    assert summary["gt_only"]["n_queries"] == 1
    assert summary["gt_only"]["rooms"] == ["Bedrooms/Bedrooms_idx_2"]
    excl = summary["baselines"]["context_conditioned_excl_gt_only"]
    assert excl["n_queries"] == 2                      # the zero-headroom query is dropped
    assert summary["baselines"]["context_conditioned"]["n_queries"] == 3


def test_summarize_run_reports_eligible_sizes_and_context_prediction_rate():
    rows = _rows_fixture()
    summary = el.summarize_run(rows)
    assert summary["eligible_set_sizes"]["histogram"] == {"1": 1, "2": 2}
    assert summary["eligible_set_sizes"]["min"] == 1 and summary["eligible_set_sizes"]["max"] == 2
    # q0 predicts candidate 1 (non-context), q1 predicts candidate 0 (non-context),
    # q2 predicts candidate 2, which IS a context member
    assert summary["context_member_prediction_rate"] == pytest.approx(1 / 3)


def test_summarize_run_nearest_context_control_when_rows_carry_context_evidence():
    rows = _rows_fixture()
    cams = el.candidate_camera_positions(_cand_set())
    for row in rows:
        row["context_xyz_cam"] = [list(cams[0]), list(cams[2])]
        row["context_sims_hex"] = el.encode_sims(torch.tensor([[0.2, 0.8]],
                                                              dtype=torch.float32))[0]
    summary = el.summarize_run(rows)

    raw_records, masked_records = [], []
    for row in rows:
        cand = np.asarray(row["candidate_xyz_world"])
        cam = np.asarray(row["candidate_xyz_cam"])
        ctx = np.asarray(row["context_xyz_cam"])
        sims = el.decode_scores(row["context_sims_hex"])
        raw = nearest_context_baseline(cam, ctx, sims)
        eligible = [not m for m in row["context_member"]]
        masked = nearest_context_baseline(cam, ctx, sims, eligible_mask=eligible)
        gt = np.asarray(row["gt_xyz_world"])
        raw_records.append({"query_id": row["query_id"], "room_id": row["room_id"],
                            "e_loc": float(np.linalg.norm(cand[raw] - gt)),
                            "top1": 1.0 if raw == row["gt_index"] else 0.0})
        masked_records.append({"query_id": row["query_id"], "room_id": row["room_id"],
                               "e_loc": float(np.linalg.norm(cand[masked] - gt)),
                               "top1": 1.0 if masked == row["gt_index"] else 0.0})
    assert summary["controls"]["nearest_context_raw"] == summarize(raw_records)
    assert summary["controls"]["nearest_context_masked"] == summarize(masked_records)


def test_summarize_run_control_is_none_without_context_evidence():
    summary = el.summarize_run(_rows_fixture())
    assert summary["controls"]["nearest_context_raw"] is None
    assert summary["controls"]["nearest_context_masked"] is None


def test_summarize_run_refuses_empty_rows():
    with pytest.raises(ValueError):
        el.summarize_run([])
