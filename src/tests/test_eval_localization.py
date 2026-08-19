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
from src.localization.candidates import (CandidateSet, candidate_metadata,  # noqa: E402
                                         project_to_camera)

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
    """Pure-aggregation behaviour for rows that carry no evidence (e.g. an offline
    re-aggregation of foreign rows). A RUN can never produce such rows: the driver
    validates the context per query and gates publication on full-length evidence
    (r4 item 3 / full-review F3).
    """
    summary = el.summarize_run(_rows_fixture())
    assert summary["controls"]["nearest_context_raw"] is None
    assert summary["controls"]["nearest_context_masked"] is None


def test_summarize_run_refuses_empty_rows():
    with pytest.raises(ValueError):
        el.summarize_run([])


# --------------------------------------------------------------------------- #
# provenance / output paths / summary writer  (unit e, part 2)
# --------------------------------------------------------------------------- #
import os     # noqa: E402
import types  # noqa: E402


def _args(**over):
    base = dict(model_config="src/configs/model_configs/m.json",
                dataset_config="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
                ckpt_path="weights/FLAC/FLAC_EMA.ckpt", agree_ckpt="weights/AGREE/AGREE_AR.pt",
                num_samples=8, tau=0.02, agg="lme", steps=1, cfg_scale=1.0, seed=42,
                cond_method="vanilla", frame_avg_angles=None, rotate_deg=0.0,
                cond_autocast="default", score_source="flac", control="none",
                batch_size=64, num_workers=6, out_dir="out", eval_name="exp18_R2",
                smoke=False, max_queries=None, registration_sha=None, parity_check=False,
                device="cpu")
    base.update(over)
    return types.SimpleNamespace(**base)


_REQUIRED_PROVENANCE_KEYS = {
    "experiment", "source_sha", "model_config", "dataset_config", "ckpt_path", "ckpt_sha256",
    "agree_ckpt", "agree_sha256", "split_hash", "n_queries", "weights_source", "readout",
    "num_samples", "tau", "agg", "steps", "cfg_scale", "seed", "cond_method",
    "frame_avg_angles", "rotate_deg", "cond_autocast", "orbit_execution", "frame_avg_fwd_cap",
    "score_source", "control", "batch_size", "num_workers", "smoke", "max_queries",
    "eval_name", "torch_version", "created_utc",
}


def test_build_provenance_records_every_protocol_quantity():
    from eval_FLAC import orbit_provenance
    record = el.build_provenance(_args(), ckpt_sha256="a" * 64, agree_sha256="b" * 64,
                                 split_hash="c" * 64, weights_source="ema", n_queries=6337)
    assert _REQUIRED_PROVENANCE_KEYS <= set(record)
    assert record["experiment"] == "exp_18_loc_invert"
    assert record["readout"] == "mean" and record["weights_source"] == "ema"
    assert record["num_samples"] == 8 and record["tau"] == 0.02 and record["agg"] == "lme"
    assert record["steps"] == 1 and record["cfg_scale"] == 1.0 and record["seed"] == 42
    assert record["batch_size"] == 64 and record["num_workers"] == 6
    assert record["rotate_deg"] == 0.0 and record["cond_autocast"] == "default"
    assert record["frame_avg_angles"] == "n/a"                       # vanilla: no orbit
    assert (record["orbit_execution"], record["frame_avg_fwd_cap"]) == orbit_provenance("vanilla")
    assert record["ckpt_sha256"] == "a" * 64 and record["split_hash"] == "c" * 64
    assert record["smoke"] is False and len(record["source_sha"]) > 0


def test_build_provenance_records_frame_avg_angles_when_symmetrized():
    record = el.build_provenance(_args(cond_method="fa_invariant", frame_avg_angles=[0.0, 90.0]),
                                 ckpt_sha256="a", agree_sha256="b", split_hash="c",
                                 weights_source="online", n_queries=1)
    assert record["frame_avg_angles"] == [0.0, 90.0]
    assert record["orbit_execution"] != "n/a"


def test_build_provenance_stamps_smoke():
    record = el.build_provenance(_args(smoke=True, max_queries=4), ckpt_sha256="a",
                                 agree_sha256="b", split_hash="c", weights_source="ema",
                                 n_queries=4)
    assert record["smoke"] is True and record["max_queries"] == 4


def test_artifact_stem_carries_every_cell_defining_field(tmp_path):
    args = _run_args(tmp_path, **{"--eval-name": "exp18_R2", "--tau": "0.05", "--seed": "43"})
    stem = el.artifact_stem(args)
    for token in ("exp18_R2", "flac", "vanilla", "ac-default", "lme", "tau0.05", "K2", "seed43",
                  "scorer-a"):
        assert token in stem, f"{token} missing from {stem}"


def test_artifact_stems_are_unique_across_differing_cells(tmp_path):
    variants = [
        {}, {"--seed": "43"}, {"--tau": "0.05"}, {"--agg": "mean"},
        {"--control": "constant_source"}, {"--cond-autocast": "off"},
        {"--cond-method": "fa_invariant"}, {"--num-samples": "4"},
        {"--eval-name": "other"}, {"--smoke": True},
    ]
    stems = {el.artifact_stem(_run_args(tmp_path, **v)) for v in variants}
    assert len(stems) == len(variants)
    oracle = el.artifact_stem(_run_args(tmp_path, **{"--score-source": "gt_rir"}))
    assert "gt_rir" in oracle and "K1" in oracle and oracle not in stems


def test_artifact_stem_marks_smoke_registered_and_dev(tmp_path):
    assert el.artifact_stem(_run_args(tmp_path)).endswith("_dev")
    assert el.artifact_stem(_run_args(tmp_path, **{"--smoke": True})).endswith("_smoke")
    registered = _run_args(tmp_path, **{"--registration-manifest": "reg.json"})
    assert el.artifact_stem(registered).endswith("_registered")


def test_artifact_paths_refuse_existing_targets_unless_overwrite(tmp_path):
    args = _run_args(tmp_path)
    paths = el.artifact_paths(args)
    assert set(paths) == {"rows", "summary", "manifest"}
    assert os.path.isdir(os.path.dirname(paths["rows"]))

    open(paths["rows"], "w").close()
    with pytest.raises(SystemExit):
        el.artifact_paths(args)
    assert el.artifact_paths(_run_args(tmp_path, **{"--overwrite": True}))["rows"] == paths["rows"]

    os.remove(paths["rows"])
    open(paths["summary"] + ".partial", "w").close()
    with pytest.raises(SystemExit):
        el.artifact_paths(args)                      # a stale .partial is a refusal too


def test_write_summary_round_trips(tmp_path):
    import json
    rows = _rows_fixture()
    summary = el.summarize_run(rows)
    provenance = el.build_provenance(_args(), ckpt_sha256="a", agree_sha256="b",
                                     split_hash="c", weights_source="ema", n_queries=len(rows))
    path = tmp_path / "summary.json"
    el.write_summary(path, summary, provenance)
    payload = json.loads(open(path).read())
    # json has no numeric keys, so the success-radius keys are rendered as strings
    # by an explicit normalizer rather than silently by the encoder.
    assert payload["summary"] == el.jsonable(summary)
    assert payload["provenance"] == provenance
    assert payload["summary"]["flac"]["pooled"]["success"]["1.0"] == pytest.approx(
        summary["flac"]["pooled"]["success"][1.0])
    assert payload["summary"]["flac"]["primary_name"] == "pooled_median_e_loc"
    assert el.jsonable(summary)["flac"]["primary"] == summary["flac"]["primary"]


# --------------------------------------------------------------------------- #
# gt_rir mode  (unit f) -- the checkpoint-free oracle: score the MEASURED RIRs
# --------------------------------------------------------------------------- #
import torchaudio  # noqa: E402


def _write_rir(room_dir, src, rec, value, length=9000, rate=22050, name=None):
    os.makedirs(room_dir, exist_ok=True)
    wav = torch.full((1, length), float(value))
    wav[0, 0] = float(value) * 0.5                     # make rows distinguishable
    path = os.path.join(room_dir, name or f"S00{src}_R00{rec}_hybrid_IR.wav")
    torchaudio.save(path, wav, rate)
    return path


def test_measured_rir_paths_match_numerically_and_report_gaps(tmp_path):
    room = str(tmp_path / "Cafe_idx_1")
    _write_rir(room, 0, 11, 0.1)
    _write_rir(room, 3, 11, 0.2, name="S003_R011_hybrid_IR.wav")     # zero-padded variant
    paths = el.measured_rir_paths(room, _cand_set(), receiver_node=11)
    assert [p is not None for p in paths] == [True, True, False]     # node 7 absent
    assert os.path.basename(paths[1]) == "S003_R011_hybrid_IR.wav"


def test_run_query_gt_rir_scores_measured_files_and_marks_the_identity(tmp_path):
    room = str(tmp_path / "Cafe_idx_1")
    _write_rir(room, 0, 11, 0.1)
    _write_rir(room, 3, 11, 0.2)                                     # the GT candidate
    _rec, engine = _engine()
    cand = _cand_set()
    obs = el.load_measured_rirs(room, cand, 11)[0][1:2]              # the GT file itself

    out = el.run_query_gt_rir(engine, cand, room, receiver_node=11, obs_wav=obs)
    assert tuple(out["sims"].shape) == (3, 1)
    assert out["available"] == [True, True, False]
    assert out["identity_index"] == cand.gt_index == 1
    assert out["sims"][1, 0] == pytest.approx(1.0, abs=1e-6)         # identity scores 1
    assert out["sims"][2, 0] == 0.0                                  # missing file placeholder
    np.testing.assert_array_equal(out["cand_cam_xyz"], el.candidate_camera_positions(cand))


def test_run_query_gt_rir_row_keeps_the_candidate_set_and_shrinks_only_eligibility(tmp_path):
    room = str(tmp_path / "Cafe_idx_1")
    _write_rir(room, 0, 11, 0.1)
    _write_rir(room, 3, 11, 0.9)
    _rec, engine = _engine()
    cand = _cand_set()
    obs = el.load_measured_rirs(room, cand, 11)[0][1:2]
    out = el.run_query_gt_rir(engine, cand, room, 11, obs)
    row = el.build_row(query_id="q", room_id="Cafe/Cafe_idx_1", relpath="a/Cafe/Cafe_idx_1/f.wav",
                       receiver_node=11, cand_set=cand, cam_xyz=out["cand_cam_xyz"],
                       sims=out["sims"], context_mask=[False] * 3, noise_keys=[],
                       tau=None, agg="max", control="none", score_source="gt_rir", smoke=False,
                       available=out["available"], identity_index=out["identity_index"])
    assert row["n_candidates"] == 3 and row["n_available"] == 2
    assert row["pred_index"] == 1 and row["identity_index"] == 1


def test_load_measured_rirs_refuses_a_wrong_sample_rate(tmp_path):
    room = str(tmp_path / "Cafe_idx_1")
    _write_rir(room, 0, 11, 0.1, rate=16000)
    _write_rir(room, 3, 11, 0.2)
    with pytest.raises(ValueError):
        el.load_measured_rirs(room, _cand_set(), 11)


def test_load_measured_rirs_pads_and_truncates_to_max_len(tmp_path):
    from src.localization.agree_embed import MAX_LEN
    room = str(tmp_path / "Cafe_idx_1")
    _write_rir(room, 0, 11, 0.1, length=100)
    _write_rir(room, 3, 11, 0.2, length=20000)
    wavs, available, _paths = el.load_measured_rirs(room, _cand_set(), 11)
    assert tuple(wavs.shape) == (2, 1, MAX_LEN) and available == [True, True, False]
    assert torch.all(wavs[0, 0, 100:] == 0.0)


# --------------------------------------------------------------------------- #
# CLI + startup guards  (unit h, part 1) -- every fail-closed rule must fire
# before any file/model/GPU work (announcement 05).
# --------------------------------------------------------------------------- #
_CLI = ["--model-config", "m.json", "--dataset-config", "d.json",
        "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt", "--num-samples", "8"]


def test_parse_args_defaults_match_the_registered_protocol():
    args = el.parse_args(_CLI)
    assert args.num_samples == 8 and args.agg == "lme"
    assert args.steps == 1 and args.cfg_scale == 1.0 and args.seed == 42
    assert args.cond_method == "vanilla" and args.rotate_deg == 0.0
    assert args.cond_autocast == "default" and args.score_source == "flac"
    assert args.control == "none" and args.smoke is False and args.max_queries is None
    assert args.batch_size > 0 and args.num_workers >= 0


def test_num_samples_is_required_for_generation_and_optional_for_the_oracle():
    """The oracle scores one measured RIR per candidate, so K is meaningless
    there; generation cannot proceed without it."""
    with pytest.raises(SystemExit):
        el.validate_args(el.parse_args(_CLI[:-2]))
    oracle = ["--model-config", "m.json", "--dataset-config", "d.json", "--agree-ckpt", "a.pt",
              "--score-source", "gt_rir"]
    el.validate_args(el.parse_args(oracle))


@pytest.mark.parametrize("flag,value", [("--agg", "median"), ("--cond-method", "canon"),
                                        ("--score-source", "measured"), ("--control", "shuffle"),
                                        ("--cond-autocast", "fp8")])
def test_parse_args_rejects_unknown_choices(flag, value):
    with pytest.raises(SystemExit):
        el.parse_args(_CLI + [flag, value])


def test_validate_args_refuses_nonzero_rotation():
    """Rotation is unimplemented here by design: silently ignoring it would put a
    rotated-conditioning number under an unrotated protocol label."""
    with pytest.raises(SystemExit):
        el.validate_args(el.parse_args(_CLI + ["--rotate-deg", "90"]))


def test_validate_args_refuses_max_queries_without_smoke():
    with pytest.raises(SystemExit):
        el.validate_args(el.parse_args(_CLI + ["--max-queries", "4"]))
    el.validate_args(el.parse_args(_CLI + ["--max-queries", "4", "--smoke"]))


@pytest.mark.parametrize("tau", ["0", "-0.02"])
def test_validate_args_refuses_nonpositive_tau_for_lme(tau):
    with pytest.raises(SystemExit):
        el.validate_args(el.parse_args(_CLI + ["--tau", tau]))
    el.validate_args(el.parse_args(_CLI + ["--tau", tau, "--agg", "max"]))   # tau unused


def test_validate_args_accepts_the_registered_configuration():
    args = el.parse_args(_CLI + ["--tau", "0.05", "--eval-name", "exp18_R2"])
    assert el.validate_args(args) is args


def test_assert_rectified_flow_guard():
    el.assert_rectified_flow({"model": {"diffusion": {"diffusion_objective": "rectified_flow"}}})
    for bad in ({"model": {"diffusion": {"diffusion_objective": "v"}}},
                {"model": {"diffusion": {}}}, {}):
        with pytest.raises(SystemExit):
            el.assert_rectified_flow(bad)


def test_assert_no_are_guard():
    """An ARE checkpoint carries a residual objective; scoring it as if it were
    vanilla FLAC would compare two different generative processes."""
    plain = {"model": {"diffusion": {"diffusion_objective": "rectified_flow"}}, "training": {}}
    el.assert_no_are(None, plain)
    el.assert_no_are(plain, plain)
    # ARE is declared under training.are_lambda / training.are_anchor (eval_FLAC:195)
    are_config = {"model": {"diffusion": {"diffusion_objective": "rectified_flow"}},
                  "training": {"are_lambda": 0.5, "are_anchor": {"early_frames": 4}}}
    with pytest.raises(SystemExit):
        el.assert_no_are(are_config, are_config)
    with pytest.raises(SystemExit):           # declared in the file config alone
        el.assert_no_are(None, are_config)


# --------------------------------------------------------------------------- #
# prepare_state_dict  (unit h, part 2) -- evaluate_model's EMA lines of record
# (eval_FLAC.py:1146-1167), factored out so they are testable without a ckpt.
# --------------------------------------------------------------------------- #
def _fake_ckpt(with_ema=True):
    """Shaped like a real training checkpoint (verified against weights/FLAC/FLAC.ckpt):
    the online weight is ``diffusion.model.<X>`` and its EMA shadow is
    ``diffusion_ema.ema_model.<X>``, so both land on ``model.<X>`` after the remap."""
    state = {"diffusion.model.w": torch.ones(2), "diffusion.pretransform.p": torch.zeros(2)}
    if with_ema:
        state["diffusion_ema.ema_model.w"] = torch.full((2,), 9.0)
    return {"state_dict": state}


def test_prepare_state_dict_strips_the_diffusion_prefix_and_folds_ema():
    state, source = el.prepare_state_dict(_fake_ckpt(), {"use_ema": True})
    assert source == "ema"
    assert "diffusion.model.w" not in state and "diffusion_ema.ema_model.w" not in state
    assert torch.equal(state["model.w"], torch.full((2,), 9.0))       # EMA weights won
    assert "pretransform.p" in state


def test_prepare_state_dict_keeps_online_weights_when_no_ema_present():
    state, source = el.prepare_state_dict(_fake_ckpt(with_ema=False), {"use_ema": True})
    assert source == "online"
    assert torch.equal(state["model.w"], torch.ones(2))


def test_prepare_state_dict_keeps_online_weights_when_config_disables_ema():
    state, source = el.prepare_state_dict(_fake_ckpt(), {"use_ema": False})
    assert source == "online"
    assert torch.equal(state["model.w"], torch.ones(2))


def test_prepare_state_dict_reports_the_resolved_source_like_eval_flac():
    from eval_FLAC import resolve_weights_source
    ckpt = _fake_ckpt()
    keys = [k.replace("diffusion.", "", 1) if k.startswith("diffusion.") else k
            for k in ckpt["state_dict"]]
    assert el.prepare_state_dict(ckpt, {"use_ema": True})[1] == resolve_weights_source(
        {"use_ema": True}, keys)


def test_prepare_state_dict_on_the_released_export_shape_reports_online():
    """weights/FLAC/FLAC_EMA.ckpt ships ALREADY flattened (keys start with model./
    conditioner./pretransform.) and carries no EMA shadow, so the honest resolved
    source is 'online' even though the weights themselves are the EMA export --
    exactly the distinction resolve_weights_source exists to record (O9)."""
    released = {"state_dict": {"model.model.w": torch.ones(2), "conditioner.c": torch.zeros(2)}}
    state, source = el.prepare_state_dict(released, {"use_ema": True})
    assert source == "online"
    assert set(state) == {"model.model.w", "conditioner.c"}


def test_prepare_state_dict_flips_use_ema_off_in_the_training_config():
    """evaluate_model sets training_config['use_ema'] = False after folding EMA in,
    so the training wrapper does not rebuild an EMA shadow at eval time."""
    training = {"use_ema": True}
    el.prepare_state_dict(_fake_ckpt(), training)
    assert training["use_ema"] is False


# --------------------------------------------------------------------------- #
# INTEGRATION: build_engine + parity_check_one_query on the REAL checkpoint (C8).
# Repo-root-anchored asset detection (r2 review finding 1): present assets must
# make the test RUN, whatever the working directory is. CPU, one candidate, K=1.
# --------------------------------------------------------------------------- #
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FLAC_CKPT = os.path.join(_REPO_ROOT, "weights", "FLAC", "FLAC_EMA.ckpt")
_FLAC_CONFIG = os.path.join(_REPO_ROOT, "src", "configs", "model_configs", "FLAC", "AR",
                            "FLAC_AR.json")
_DATASET_CONFIG = os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs", "AR", "eval",
                               "acousticroom_unseeneval.json")


def _dinov3_cache_present():
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    return os.path.isdir(os.path.join(hub, "models--facebook--dinov3-vits16-pretrain-lvd1689m"))


def _gen_assets_present():
    return (os.path.isfile(_FLAC_CKPT) and os.path.isfile(_FLAC_CONFIG)
            and _dinov3_cache_present())


_HAVE_GEN_ASSETS = _gen_assets_present()
gen_integration = pytest.mark.skipif(
    not _HAVE_GEN_ASSETS, reason="FLAC checkpoint/config or the gated DINOv3 HF cache are absent")


def test_generation_asset_detection_is_repo_root_anchored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _gen_assets_present() is _HAVE_GEN_ASSETS
    if (os.path.isfile(_FLAC_CKPT) and os.path.isfile(_FLAC_CONFIG) and _dinov3_cache_present()):
        assert _HAVE_GEN_ASSETS is True


def _synthetic_metadata(seed=0):
    """One query's conditioning without touching the dataset: the shapes the AR
    loader produces (depth [3,256,512], 8 context RIRs of 9600 samples, poses)."""
    g = torch.Generator().manual_seed(seed)
    source = torch.randn(3, generator=g)
    return {"scene": "Cafe", "source": source, "source_vit": source.unsqueeze(0),
            "context_poses": torch.randn(8, 3, generator=g),
            "context_poses_vit": torch.randn(8, 3, generator=g),
            "context_audio": torch.randn(8, 1, 9600, generator=g) * 0.1,
            "depth": torch.rand(3, 256, 512, generator=g) * 3.0}


@pytest.fixture(scope="module")
def real_engine():
    previous = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        args = el.parse_args([
            "--model-config", _FLAC_CONFIG, "--dataset-config", _DATASET_CONFIG,
            "--ckpt-path", _FLAC_CKPT, "--agree-ckpt", "unused-for-parity",
            "--num-samples", "1", "--device", "cpu"])
        el.validate_args(args)
        engine, context = el.build_engine(args, agree=None, device="cpu")
        yield args, engine, context
    finally:
        os.chdir(previous)


@gen_integration
def test_integration_build_engine_follows_the_reference_construction(real_engine):
    args, engine, context = real_engine
    assert context["weights_source"] == "online"      # the released EMA export ships flattened
    assert context["latent_shape"] == (engine.io_channels, engine.latent_samples)
    assert engine.latent_samples == 10240 // context["module"].diffusion.pretransform.downsampling_ratio
    assert context["module"].training is False
    assert not any(p.requires_grad for p in context["module"].parameters())


@gen_integration
def test_integration_parity_one_query_matches_the_eval_flac_path(real_engine):
    """C8: same ckpt + metadata + noise through the driver path and a straight-line
    replay of evaluate_model's calls must give IDENTICAL waveforms."""
    args, engine, context = real_engine
    noise = el.build_noise_bank(42, "parity_query", 1, context["latent_shape"])
    result = el.parity_check_one_query(args, engine, context, [_synthetic_metadata()], noise)
    assert result["match"] is True
    assert result["max_abs_diff"] == 0.0
    assert result["shape"] == [1, 1, 10240]


@gen_integration
def test_integration_generation_is_reproducible_for_a_fixed_noise_key(real_engine):
    args, engine, context = real_engine
    metadata = [_synthetic_metadata(seed=3)]
    noise = el.build_noise_bank(42, "repeat_query", 1, context["latent_shape"])
    first = engine.decoder(engine.sampler(
        noise, engine.cond_inputs_fn(engine.conditioner(metadata, engine.device))))
    second = engine.decoder(engine.sampler(
        noise, engine.cond_inputs_fn(engine.conditioner(metadata, engine.device))))
    assert torch.equal(first, second)


# --------------------------------------------------------------------------- #
# per-query wiring  (unit h, part 3): dataset-folder derivation, candidate set,
# context evidence for the O10 control.
# --------------------------------------------------------------------------- #
import json as _json  # noqa: E402


def _dataset_tree(tmp_path, sources=((0, (0.0, 0.0, 1.0)), (3, (2.0, -1.0, 1.5)),
                                     (7, (-3.0, 4.0, 0.75))), receiver=11,
                  rec_loc=(1.0, 2.0, 0.5), with_depth=True):
    """A miniature AcousticRooms tree: metadata pair JSONs + IR wavs (+ depth maps)."""
    root = tmp_path / "AcousticRooms"
    meta_room = root / "metadata" / "Cafe" / "Cafe_idx_1"
    wav_room = root / "single_channel_ir_1" / "Cafe" / "Cafe_idx_1"
    meta_room.mkdir(parents=True, exist_ok=True)
    wav_room.mkdir(parents=True, exist_ok=True)
    if with_depth:
        depth_room = root / "depth_map" / "Cafe" / "Cafe_idx_1"
        depth_room.mkdir(parents=True, exist_ok=True)
        np.save(str(depth_room / f"{receiver}.npy"), np.zeros((4, 8), dtype=np.float32))
    for node, xyz in sources:
        (meta_room / f"S00{node}_R00{receiver}.json").write_text(_json.dumps(
            {"src_loc": list(xyz), "rec_loc": list(rec_loc), "IR_norm": 1.0}))
        _write_rir(str(wav_room), node, receiver, 0.1 * (node + 1))
    return root, wav_room


_ROOM_SOURCES = {0: (0.0, 0.0, 1.0), 3: (2.0, -1.0, 1.5), 7: (-3.0, 4.0, 0.75)}


def _query_md(root, wav_room, src=3, receiver=11, rec_loc=(1.0, 2.0, 0.5),
              src_loc=None, context_nodes=None):
    """One loader item. The context poses are the camera-frame positions of OTHER
    real sources in the room -- what AR_md actually produces -- so the fail-closed
    membership resolution (r3 fix F7) has something to resolve against."""
    from src.localization.candidates import project_to_camera
    src_loc = _ROOM_SOURCES[src] if src_loc is None else src_loc
    path = str(wav_room / f"S00{src}_R00{receiver}_hybrid_IR.wav")
    source = torch.as_tensor(project_to_camera(np.asarray(rec_loc), np.asarray(src_loc)),
                             dtype=torch.float32)
    # every OTHER source in the room: a constant context size, as a real room gives
    context = [n for n in (sorted(_ROOM_SOURCES) if context_nodes is None else context_nodes)
               if n != src]
    poses = torch.as_tensor(
        np.stack([project_to_camera(np.asarray(rec_loc), np.asarray(_ROOM_SOURCES[n]))
                  for n in context]), dtype=torch.float32)
    audio = torch.stack([torch.full((1, 9600), 0.4 - 0.2 * i) for i in range(len(context))])
    return {"idx": 0, "path": path, "relpath": os.path.relpath(path, str(root)),
            "scene": "Cafe", "source": source, "source_vit": source.unsqueeze(0),
            "context_poses": poses, "context_audio": audio,
            "depth": torch.zeros(3, 4, 8)}


def test_dataset_folder_from_md_mirrors_the_release_derivation(tmp_path):
    root, wav_room = _dataset_tree(tmp_path)
    md = _query_md(root, wav_room)
    assert el.dataset_folder_from_md(md).rstrip("/") == str(root)


def test_query_candidate_set_builds_from_the_metadata_authority(tmp_path):
    root, wav_room = _dataset_tree(tmp_path)
    cand = el.query_candidate_set(_query_md(root, wav_room))
    assert cand.nodes == [0, 3, 7] and cand.gt_node == 3
    np.testing.assert_allclose(cand.rec_loc, [1.0, 2.0, 0.5])


def test_context_evidence_scores_the_measured_context_rirs(tmp_path):
    """The O10 control needs cos(h_obs, context RIR) -- available from the
    metadata the loader already carries, with no extra file reads."""
    root, wav_room = _dataset_tree(tmp_path)
    md = _query_md(root, wav_room)
    md["context_poses"] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32)
    md["context_audio"] = torch.stack([torch.full((1, 9600), 0.3), torch.full((1, 9600), -0.2)])
    _rec, engine = _engine()
    obs = torch.full((1, 1, 9600), 0.3)

    evidence = el.context_evidence(engine, md, obs)
    assert evidence["context_xyz_cam"] == [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    sims = el.decode_scores(evidence["context_sims_hex"])
    assert tuple(sims.shape) == (2,)
    assert sims[0] == pytest.approx(1.0, abs=1e-6)     # identical waveform to the observation
    assert sims[1] < sims[0]


def test_context_evidence_is_none_without_context_metadata(tmp_path):
    root, wav_room = _dataset_tree(tmp_path)
    md = _query_md(root, wav_room)
    md.pop("context_audio")
    _rec, engine = _engine()
    assert el.context_evidence(engine, md, torch.zeros(1, 1, 9600)) is None


def test_build_row_carries_optional_context_evidence():
    cand = _cand_set()
    row = el.build_row(**_row_kwargs(context_xyz_cam=[[0.0, 0.0, 0.0]],
                                     context_sims_hex=el.encode_sims(
                                         torch.tensor([[0.5]], dtype=torch.float32))[0]))
    assert row["context_xyz_cam"] == [[0.0, 0.0, 0.0]]
    assert el.decode_scores(row["context_sims_hex"])[0] == pytest.approx(0.5)
    assert "context_xyz_cam" not in el.build_row(**_row_kwargs())


# --------------------------------------------------------------------------- #
# run_evaluation / main  (unit h, part 4) -- end to end on a miniature dataset
# with a stub engine: audit -> per-query rows -> summary, plus the smoke rules.
# --------------------------------------------------------------------------- #
class _FakeLoader:
    def __init__(self, batches, dataset):
        self.batches, self.dataset = batches, dataset

    def __iter__(self):
        return iter(self.batches)


def _fake_run(tmp_path, nodes=(3, 0), batch_size=2):
    """Two queries in one room, delivered as the eval loader would deliver them."""
    root, wav_room = _dataset_tree(tmp_path)
    srcs = {0: (0.0, 0.0, 1.0), 3: (2.0, -1.0, 1.5), 7: (-3.0, 4.0, 0.75)}
    metadata, filenames = [], []
    for node in nodes:
        md = _query_md(root, wav_room, src=node, src_loc=srcs[node])
        md["idx"] = len(metadata)
        metadata.append(md)
        filenames.append(md["path"])
    dataset = _FakeDataset(filenames, [str(root)])
    reals = torch.stack([torch.full((1, 9600), 0.2 + 0.1 * i) for i in range(len(metadata))])
    batches = [(reals[i:i + batch_size], metadata[i:i + batch_size])
               for i in range(0, len(metadata), batch_size)]
    return _FakeLoader(batches, dataset), root


def _run_args(tmp_path, **over):
    oracle = over.get("--score-source") == "gt_rir"
    argv = ["--model-config", "m.json", "--dataset-config", "d.json",
            "--agree-ckpt", "a.pt", "--num-samples", "2",
            "--out-dir", str(tmp_path / "out"), "--eval-name", "unit", "--device", "cpu"]
    if not oracle:      # the oracle refuses a checkpoint: no generation happens (r3 fix F8)
        argv += ["--ckpt-path", "c.ckpt"]
    for flag, value in over.items():
        argv += [flag] if value is True else [flag, str(value)]
    return el.validate_args(el.parse_args(argv))


def _stub_context(root=None, split=None):
    """The run context. A frozen candidate manifest is mandatory on the query path
    (r4 item 1), so the fixtures build one from the miniature dataset tree."""
    context = {"weights_source": "ema", "latent_shape": (2, 8), "device": "cpu"}
    if root is not None:
        context["manifest"] = el.build_room_manifest(str(root), split or _split_dict())
    return context


def test_run_evaluation_end_to_end_writes_rows_and_summary(tmp_path):
    loader, _root = _fake_run(tmp_path)
    _rec, engine = _engine()
    args = _run_args(tmp_path)
    result = el.run_evaluation(args, loader, engine, _stub_context(_root), "ck" * 32, "ag" * 32,
                               expected=el.expected_split_identities(loader.dataset))

    rows = el.read_rows(result["rows_path"])
    assert len(rows) == 2 and os.path.exists(result["summary_path"])
    assert {r["room_id"] for r in rows} == {"Cafe/Cafe_idx_1"}
    assert rows[0]["gt_node"] == 3 and rows[1]["gt_node"] == 0
    assert rows[0]["candidate_nodes"] == [0, 3, 7] and rows[0]["n_samples"] == 2
    assert rows[0]["noise_keys"] == [noise_key(args.seed, rows[0]["query_id"], k) for k in range(2)]
    assert "context_xyz_cam" in rows[0]                       # O10 evidence recorded
    assert result["summary"] == el.summarize_run(rows)
    assert result["provenance"]["split_hash"] == el.split_hash(
        el.expected_split_identities(loader.dataset))
    assert result["provenance"]["n_queries"] == 2
    assert result["summary"]["controls"]["nearest_context_raw"] is not None


def test_run_evaluation_row_is_reproducible_across_runs(tmp_path):
    loader, _root = _fake_run(tmp_path)
    _rec, engine = _engine()
    first = el.run_evaluation(_run_args(tmp_path / "a"), loader, engine, _stub_context(_root), "c", "a",
                              expected=el.expected_split_identities(loader.dataset))
    loader2, _root2 = _fake_run(tmp_path)
    _rec2, engine2 = _engine()
    second = el.run_evaluation(_run_args(tmp_path / "b"), loader2, engine2, _stub_context(_root2), "c", "a",
                               expected=el.expected_split_identities(loader2.dataset))
    assert [r["sims_hex"] for r in first["rows"]] == [r["sims_hex"] for r in second["rows"]]


def test_run_evaluation_smoke_truncates_after_auditing_the_truncated_enumeration(tmp_path):
    loader, _root = _fake_run(tmp_path)
    _rec, engine = _engine()
    args = _run_args(tmp_path, **{"--smoke": True, "--max-queries": 1})
    result = el.run_evaluation(args, loader, engine, _stub_context(_root), "c", "a",
                               expected=el.expected_split_identities(loader.dataset))
    rows = el.read_rows(result["rows_path"])
    assert len(rows) == 1 and rows[0]["smoke"] is True
    assert "_smoke_" in os.path.basename(result["rows_path"])
    assert result["provenance"]["smoke"] is True and result["provenance"]["n_queries"] == 1
    assert result["provenance"]["split_hash"] == el.split_hash(
        el.expected_split_identities(loader.dataset)[:1])


def test_run_evaluation_aborts_before_writing_when_the_audit_fails(tmp_path):
    loader, _root = _fake_run(tmp_path)
    loader.batches[0][1][1]["idx"] = 99                        # silent substitution
    _rec, engine = _engine()
    args = _run_args(tmp_path)
    with pytest.raises(SystemExit):
        el.run_evaluation(args, loader, engine, _stub_context(_root), "c", "a",
                          expected=el.expected_split_identities(loader.dataset))
    assert not os.path.exists(el.artifact_paths(args, overwrite=True)["rows"])  # nothing written


def test_run_evaluation_constant_source_control_is_recorded(tmp_path):
    loader, _root = _fake_run(tmp_path)
    _rec, engine = _engine()
    args = _run_args(tmp_path, **{"--control": "constant_source"})
    result = el.run_evaluation(args, loader, engine, _stub_context(_root), "c", "a",
                               expected=el.expected_split_identities(loader.dataset))
    assert all(row["control"] == "constant_source" for row in result["rows"])
    assert result["provenance"]["control"] == "constant_source"


def test_run_evaluation_gt_rir_mode_scores_measured_files(tmp_path):
    loader, _root = _fake_run(tmp_path)
    _rec, engine = _engine()
    args = _run_args(tmp_path, **{"--score-source": "gt_rir", "--agg": "max"})
    result = el.run_evaluation(args, loader, engine, _stub_context(_root), "c", "a",
                               expected=el.expected_split_identities(loader.dataset))
    rows = result["rows"]
    assert all(row["score_source"] == "gt_rir" and row["n_samples"] == 1 for row in rows)
    assert all(row["noise_keys"] == [] for row in rows)
    assert rows[0]["identity_index"] == rows[0]["gt_index"]


def test_main_refuses_a_missing_checkpoint_only_for_the_generative_mode():
    argv = ["--model-config", "m.json", "--dataset-config", "d.json",
            "--agree-ckpt", "a.pt", "--num-samples", "2"]
    with pytest.raises(SystemExit):
        el.validate_args(el.parse_args(argv))                  # flac mode needs a ckpt
    el.validate_args(el.parse_args(argv + ["--score-source", "gt_rir"]))


def test_scoring_only_engine_embeds_but_refuses_to_generate():
    """R-1 runs the measured-RIR oracle before any checkpoint exists, so the
    generation callables must be absent-by-construction rather than silently
    returning something."""
    class _Agree:
        model = None

    engine = el.scoring_only_engine(_Agree(), "cpu")
    for call in (engine.conditioner, engine.cond_inputs_fn, engine.sampler, engine.decoder):
        with pytest.raises(ValueError):
            call(None)


def test_main_is_wired_and_validates_before_touching_assets(tmp_path):
    """main() refuses an invalid protocol before it opens a config or a ckpt."""
    with pytest.raises(SystemExit):
        el.main(["--model-config", "nope.json", "--dataset-config", "nope.json",
                 "--ckpt-path", "nope.ckpt", "--agree-ckpt", "nope.pt",
                 "--num-samples", "2", "--rotate-deg", "45"])
    with pytest.raises(SystemExit):
        el.main(["--model-config", "nope.json", "--dataset-config", "nope.json",
                 "--ckpt-path", "nope.ckpt", "--agree-ckpt", "nope.pt",
                 "--num-samples", "2", "--max-queries", "3"])


# --------------------------------------------------------------------------- #
# r3 fix F3 (review finding 3): the constant-source control must NOT overwrite
# the candidate geometry that rows, membership and the baselines depend on.
# --------------------------------------------------------------------------- #
def test_run_query_control_keeps_candidate_geometry_and_only_moves_conditioning():
    rec, engine = _engine()
    cand = _cand_set()
    noise = el.build_noise_bank(42, "q0", 2, (2, 8))
    out = el.run_query(engine, _base_md(cand), cand, noise, _OBS, control="constant_source")

    true_positions = el.candidate_camera_positions(cand)
    np.testing.assert_array_equal(out["cand_cam_xyz"], true_positions)       # untouched
    centroid = true_positions.mean(axis=0)
    np.testing.assert_allclose(out["conditioning_xyz_cam"], np.repeat(centroid[None], 3, axis=0))
    passed = torch.stack([md["source"] for md in rec.seen_metadata[0]])
    assert torch.allclose(passed, torch.as_tensor(centroid, dtype=torch.float32).expand(3, 3),
                          atol=0)


def test_run_query_control_leaves_context_membership_intact():
    """With centroid geometry in the row, every candidate would look absent from
    the context and the information-matched comparison would be meaningless."""
    _rec, engine = _engine()
    cand = _cand_set()
    cams = el.candidate_camera_positions(cand)
    context_ids = [el.render_position_id(cams[0]), el.render_position_id(cams[2])]
    noise = el.build_noise_bank(42, "q0", 2, (2, 8))

    plain = el.run_query(engine, _base_md(cand), cand, noise, _OBS)
    controlled = el.run_query(engine, _base_md(cand), cand, noise, _OBS,
                              control="constant_source")
    assert el.context_membership_mask(controlled["cand_cam_xyz"], context_ids) == \
        el.context_membership_mask(plain["cand_cam_xyz"], context_ids) == [True, False, True]


def test_run_query_without_control_reports_identical_position_arrays():
    _rec, engine = _engine()
    cand = _cand_set()
    out = el.run_query(engine, _base_md(cand), cand, el.build_noise_bank(1, "q", 2, (2, 8)), _OBS)
    np.testing.assert_array_equal(out["cand_cam_xyz"], out["conditioning_xyz_cam"])


# --------------------------------------------------------------------------- #
# r3 fix F7 (review finding 7): membership must fail CLOSED. A projection or
# receiver mismatch silently enlarged the eligible set, which inflates the
# information-matched baseline's headroom.
# --------------------------------------------------------------------------- #
def test_context_membership_mask_aborts_on_an_unresolvable_context_id():
    cams = np.array([[0.5, -1.25, 2.0], [-3.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        el.context_membership_mask(cams, [el.render_position_id([9.0, 9.0, 9.0])])


def test_context_membership_mask_aborts_on_candidate_fingerprint_collision():
    cams = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        el.context_membership_mask(cams, [el.render_position_id(cams[0])])


def test_context_membership_mask_aborts_when_the_gt_is_a_context_member():
    """Context is drawn from OTHER sources by construction; GT appearing in it is
    a wiring bug that would hand the answer to the baseline."""
    cams = np.array([[0.5, -1.25, 2.0], [-3.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        el.context_membership_mask(cams, [el.render_position_id(cams[1])], gt_index=1)
    assert el.context_membership_mask(cams, [el.render_position_id(cams[0])], gt_index=1) == \
        [True, False]


def test_context_membership_mask_allows_a_context_drawn_with_replacement():
    """AR_md falls back to np.random.choice(replace=True) when a room has fewer
    sources than the context size, so repeated ids are legitimate."""
    cams = np.array([[0.5, -1.25, 2.0], [-3.0, 0.0, 1.0]])
    ids = [el.render_position_id(cams[0])] * 3
    assert el.context_membership_mask(cams, ids) == [True, False]


def test_process_query_aborts_before_generation_on_a_bad_context(tmp_path):
    root, wav_room = _dataset_tree(tmp_path)
    md = _query_md(root, wav_room)
    md["context_poses"] = torch.tensor([[99.0, 99.0, 99.0]], dtype=torch.float32)
    md["context_audio"] = torch.full((1, 1, 9600), 0.3)
    rec, engine = _engine()
    args = _run_args(tmp_path)
    with pytest.raises(ValueError):
        el.process_query(args, engine, _stub_context(root), md, torch.full((1, 1, 9600), 0.2))
    assert rec.calls == []                       # nothing was generated


# --------------------------------------------------------------------------- #
# r3 fix F9 (review finding 9): numeric guards must reject non-finite and
# degenerate values, and the artifact validation must happen on CPU BEFORE any
# model is constructed or moved to a device.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("flag,value", [
    ("--tau", "nan"), ("--tau", "inf"),
    ("--cfg-scale", "nan"), ("--cfg-scale", "-inf"),
    ("--steps", "0"), ("--steps", "-1"),
    ("--num-workers", "-1"), ("--num-workers", "0"),
    ("--batch-size", "0"),
    ("--num-samples", "0"),
])
def test_validate_args_rejects_degenerate_numbers(flag, value):
    with pytest.raises(SystemExit):
        el.validate_args(el.parse_args(_CLI + [flag, value]))


def test_validate_args_rejects_a_non_positive_smoke_limit():
    with pytest.raises(SystemExit):
        el.validate_args(el.parse_args(_CLI + ["--smoke", "--max-queries", "0"]))
    el.validate_args(el.parse_args(_CLI + ["--smoke", "--max-queries", "1"]))


def test_load_and_validate_artifacts_refuses_a_foreign_objective(tmp_path):
    config = tmp_path / "m.json"
    config.write_text(_json.dumps({"model": {"diffusion": {"diffusion_objective": "v"}},
                                   "sample_size": 10240, "sample_rate": 22050}))
    args = el.parse_args(["--model-config", str(config), "--dataset-config", "d.json",
                          "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt", "--num-samples", "2"])
    with pytest.raises(SystemExit):
        el.load_and_validate_artifacts(args)


def test_load_and_validate_artifacts_runs_before_the_scorer_is_built(tmp_path, monkeypatch):
    """A wrong objective or an ARE artifact must be refused on CPU, before the
    AGREE scorer is constructed or anything is moved to a device."""
    config = tmp_path / "m.json"
    config.write_text(_json.dumps({"model": {"diffusion": {"diffusion_objective": "v"}},
                                   "sample_size": 10240, "sample_rate": 22050}))

    def _never(*args, **kwargs):
        raise AssertionError("the scorer must not be built before the artifacts validate")

    monkeypatch.setattr(el, "load_agree_audio", _never)
    monkeypatch.setattr(el, "build_engine", _never)
    with pytest.raises(SystemExit):
        el.main(["--model-config", str(config), "--dataset-config", "d.json",
                 "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt", "--num-samples", "2",
                 "--device", "cpu"])


def test_load_and_validate_artifacts_skips_the_checkpoint_for_the_oracle(tmp_path):
    config = tmp_path / "m.json"
    config.write_text(_json.dumps({"model": {"diffusion": {"diffusion_objective": "rectified_flow"}},
                                   "sample_size": 10240, "sample_rate": 22050}))
    args = el.validate_args(el.parse_args(
        ["--model-config", str(config), "--dataset-config", "d.json", "--agree-ckpt", "a.pt",
         "--score-source", "gt_rir"]))
    model_config, ckpt = el.load_and_validate_artifacts(args)
    assert ckpt is None and model_config["sample_size"] == 10240


# --------------------------------------------------------------------------- #
# r3 fix F8 (review finding 8): gt_rir must be unambiguous and must not record a
# protocol it did not run.
# --------------------------------------------------------------------------- #
def test_measured_rir_paths_reject_duplicate_numeric_matches(tmp_path):
    room = str(tmp_path / "Cafe_idx_1")
    _write_rir(room, 3, 11, 0.2)
    _write_rir(room, 3, 11, 0.3, name="S003_R011_hybrid_IR.wav")      # same (3, 11)
    with pytest.raises(ValueError):
        el.measured_rir_paths(room, _cand_set(), receiver_node=11)


def test_run_query_gt_rir_aborts_when_the_identity_candidate_is_missing(tmp_path):
    room = str(tmp_path / "Cafe_idx_1")
    _write_rir(room, 0, 11, 0.1)                                      # GT (node 3) absent
    _rec, engine = _engine()
    cand = _cand_set()
    with pytest.raises(ValueError):
        el.run_query_gt_rir(engine, cand, room, 11, torch.full((1, 1, 8000), 0.1))


def test_gt_reciprocal_rank_ranks_over_available_candidates_only():
    scores = torch.tensor([0.99, 0.50, 0.10])
    assert el.gt_reciprocal_rank(scores, 1) == pytest.approx(0.5)
    # candidate 0 has no measured file: it must not out-rank the GT
    assert el.gt_reciprocal_rank(scores, 1, available=[False, True, True]) == pytest.approx(1.0)
    assert el.gt_reciprocal_rank(scores, 2, available=[False, True, True]) == pytest.approx(0.5)


def test_build_row_rank_uses_the_available_set(tmp_path):
    kwargs = _row_kwargs(sims=torch.tensor([[0.99], [0.10], [0.50]], dtype=torch.float32),
                         noise_keys=[], available=[False, True, True],
                         score_source="gt_rir", identity_index=1, agg="max", tau=None)
    row = el.build_row(**kwargs)
    assert row["rr"] == pytest.approx(0.5)          # GT second among the two available
    assert row["pred_index"] == 2


@pytest.mark.parametrize("extra", [["--ckpt-path", "c.ckpt"],
                                   ["--control", "constant_source"],
                                   ["--parity-check"]])
def test_validate_args_refuses_irrelevant_flags_in_oracle_mode(extra):
    argv = ["--model-config", "m.json", "--dataset-config", "d.json", "--agree-ckpt", "a.pt",
            "--score-source", "gt_rir"] + extra
    with pytest.raises(SystemExit):
        el.validate_args(el.parse_args(argv))


def test_provenance_records_the_oracle_as_k1(tmp_path):
    args = el.validate_args(el.parse_args(
        ["--model-config", "m.json", "--dataset-config", "d.json", "--agree-ckpt", "a.pt",
         "--score-source", "gt_rir", "--out-dir", str(tmp_path), "--eval-name", "R_minus_1"]))
    assert el.effective_num_samples(args) == 1
    stem = el.artifact_stem(args)
    assert stem.startswith("R_minus_1_gt_rir") and "_K1_" in stem
    record = el.build_provenance(args, ckpt_sha256="n/a", agree_sha256="b", split_hash="c",
                                 weights_source="n/a", n_queries=3)
    assert record["num_samples"] == 1 and record["score_source"] == "gt_rir"


# --------------------------------------------------------------------------- #
# r3 fix F5 (review finding 5, O16): smoke and parity runs must read a SEEN
# split. Probing the unseen split repeatedly with a debug tool is exactly the
# pre-registration leak the protocol forbids.
# --------------------------------------------------------------------------- #
_SEEN_CONFIG = os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs", "AR", "eval",
                            "acousticroom_seeneval.json")
_UNSEEN_CONFIG = os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs", "AR", "eval",
                              "acousticroom_unseeneval.json")


def _split_args(dataset_config, *extra):
    return el.parse_args(["--model-config", "m.json", "--dataset-config", dataset_config,
                          "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt",
                          "--num-samples", "2", *extra])


def test_validate_dataset_split_refuses_smoke_on_the_unseen_split():
    with pytest.raises(SystemExit):
        el.validate_dataset_split(_split_args(_UNSEEN_CONFIG, "--smoke", "--max-queries", "2"))
    with pytest.raises(SystemExit):
        el.validate_dataset_split(_split_args(_UNSEEN_CONFIG, "--parity-check"))


def test_validate_dataset_split_allows_smoke_and_parity_on_the_seen_split():
    el.validate_dataset_split(_split_args(_SEEN_CONFIG, "--smoke", "--max-queries", "2"))
    el.validate_dataset_split(_split_args(_SEEN_CONFIG, "--parity-check"))


def test_validate_dataset_split_leaves_registered_unseen_runs_alone():
    el.validate_dataset_split(_split_args(_UNSEEN_CONFIG))
    oracle = el.parse_args(["--model-config", "m.json", "--dataset-config", _UNSEEN_CONFIG,
                            "--agree-ckpt", "a.pt", "--score-source", "gt_rir"])
    el.validate_dataset_split(oracle)                       # registered R-1 oracle run


def test_validate_dataset_split_refuses_a_split_that_declares_neither(tmp_path):
    config = tmp_path / "d.json"
    config.write_text(_json.dumps({"dataset_type": "audio_dir", "datasets": []}))
    with pytest.raises(SystemExit):
        el.validate_dataset_split(_split_args(str(config), "--smoke"))


# --------------------------------------------------------------------------- #
# r3 fix F1 (review finding 1, BLOCKER): the identity audit had a TOCTOU hole --
# a loader that was clean during the audit pass could substitute during the
# scoring pass and the wrong query entered the artifact. The expectation is now
# derived from the SPLIT JSON (not from the object being audited), checked per
# row BEFORE generation, gated on count+rooms at the end, hashed over the SCORED
# stream, and published atomically.
# --------------------------------------------------------------------------- #
def _split_dataset_config(tmp_path, root, order=("S003_R0011_hybrid_IR.wav",
                                                 "S000_R0011_hybrid_IR.wav")):
    split = tmp_path / "split.json"
    split.write_text(_json.dumps({"Cafe": {"Cafe_idx_1": list(order)}}))
    return {"dataset_type": "audio_dir", "seeneval": True,
            "datasets": [{"id": "AcousticRooms", "path": str(root),
                          "json_file_path": str(split),
                          "folder_name": "single_channel_ir_1"}]}


def test_expected_split_identities_come_from_the_split_json_not_the_loader(tmp_path):
    root, _wav_room = _dataset_tree(tmp_path)
    config = _split_dataset_config(tmp_path, root)
    identities = el.expected_split_identities_from_config(config)
    assert identities == [
        "0|single_channel_ir_1/Cafe/Cafe_idx_1/S003_R0011_hybrid_IR.wav",
        "1|single_channel_ir_1/Cafe/Cafe_idx_1/S000_R0011_hybrid_IR.wav",
    ]


def test_expected_split_identities_follow_the_split_order(tmp_path):
    root, _wav_room = _dataset_tree(tmp_path)
    flipped = _split_dataset_config(tmp_path, root, order=("S000_R0011_hybrid_IR.wav",
                                                           "S003_R0011_hybrid_IR.wav"))
    assert el.expected_split_identities_from_config(flipped)[0].endswith("S000_R0011_hybrid_IR.wav")


def test_assert_scored_stream_gates_count_and_rooms():
    expected = ["0|ir/Cafe/Cafe_idx_1/a.wav", "1|ir/Bed/Bed_idx_0/b.wav"]
    assert el.assert_scored_stream(expected, expected) == el.split_hash(expected)
    with pytest.raises(SystemExit):
        el.assert_scored_stream(expected[:1], expected)                  # short
    with pytest.raises(SystemExit):
        el.assert_scored_stream(["0|ir/Cafe/Cafe_idx_1/a.wav"] * 2, expected)   # wrong rooms


class _TOCTOULoader:
    """Clean on the first full iteration, corrupt on every later one."""

    def __init__(self, batches, dataset, mode):
        self.batches, self.dataset, self.mode, self.passes = batches, dataset, mode, 0

    def __iter__(self):
        self.passes += 1
        if self.passes == 1:
            yield from self.batches
            return
        if self.mode == "truncate":
            yield (self.batches[0][0][:1], self.batches[0][1][:1])
            return
        reals, metadata = self.batches[0]
        poisoned = [dict(md) for md in metadata]
        poisoned[-1]["idx"] = 99                                 # silent substitution
        yield (reals, poisoned)


def test_run_evaluation_aborts_when_the_scoring_pass_substitutes(tmp_path):
    """The exact TOCTOU attack: audit pass clean, scoring pass substituted."""
    loader, _root = _fake_run(tmp_path)
    expected = el.expected_split_identities(loader.dataset)
    attacker = _TOCTOULoader(loader.batches, loader.dataset, mode="substitute")
    assert el.audit_split_identities(attacker, expected) == el.split_hash(expected)  # pass 1 clean

    _rec, engine = _engine()
    args = _run_args(tmp_path)
    with pytest.raises(SystemExit):
        el.run_evaluation(args, attacker, engine, _stub_context(_root), "c", "a", expected=expected)
    paths = el.artifact_paths(args, overwrite=True)
    assert not os.path.exists(paths["rows"]) and not os.path.exists(paths["summary"])


def test_run_evaluation_aborts_at_the_end_gate_when_the_scoring_pass_truncates(tmp_path):
    loader, _root = _fake_run(tmp_path)
    expected = el.expected_split_identities(loader.dataset)
    attacker = _TOCTOULoader(loader.batches, loader.dataset, mode="truncate")
    el.audit_split_identities(attacker, expected)                        # pass 1 clean
    _rec, engine = _engine()
    args = _run_args(tmp_path)
    with pytest.raises(SystemExit):
        el.run_evaluation(args, attacker, engine, _stub_context(_root), "c", "a", expected=expected)
    assert not os.path.exists(el.artifact_paths(args, overwrite=True)["rows"])


def test_run_evaluation_hashes_the_scored_stream_and_publishes_atomically(tmp_path):
    loader, _root = _fake_run(tmp_path)
    _rec, engine = _engine()
    args = _run_args(tmp_path)
    expected = el.expected_split_identities(loader.dataset)
    result = el.run_evaluation(args, loader, engine, _stub_context(_root), "c", "a", expected=expected)

    scored = [row["query_id"] for row in result["rows"]]
    assert result["provenance"]["split_hash"] == el.split_hash(scored)
    assert os.path.exists(result["rows_path"]) and os.path.exists(result["summary_path"])
    assert not os.path.exists(result["rows_path"] + ".partial")
    assert not os.path.exists(result["summary_path"] + ".partial")


def test_run_evaluation_checks_identity_before_generating(tmp_path):
    loader, _root = _fake_run(tmp_path)
    loader.batches[0][1][0]["idx"] = 42                    # first item already substituted
    rec, engine = _engine()
    expected = el.expected_split_identities(loader.dataset)
    with pytest.raises(SystemExit):
        el.run_evaluation(_run_args(tmp_path), loader, engine, _stub_context(_root), "c", "a",
                          expected=expected)
    assert rec.calls == []                                 # aborted before any generation


# --------------------------------------------------------------------------- #
# r3 fix F6 (review finding 6): the provenance record must pin the registration
# commit (O17), the config CONTENTS, the context draw's parameters (O8) and the
# numerics of the box the run happened on.
# --------------------------------------------------------------------------- #
_R3FIX_PROVENANCE_KEYS = {
    "registration_sha", "model_config_sha256", "dataset_config_sha256", "context_k",
    "loader_shuffle", "loader_drop_last", "device_name", "float32_matmul_precision",
    "torch_version", "cuda_version", "cudnn_version", "torchaudio_version",
    "transformers_version", "tf32_matmul", "tf32_cudnn", "cudnn_deterministic",
    "flash_attn_available", "context_stream_digest",
}


def _real_dataset_config():
    return _json.loads(open(_UNSEEN_CONFIG).read())


def test_build_provenance_records_the_environment_and_config_contents(tmp_path):
    import hashlib as _h
    model_config = tmp_path / "m.json"
    model_config.write_text(_json.dumps({"model": {}}))
    args = _args(model_config=str(model_config), dataset_config=_UNSEEN_CONFIG,
                 registration_sha="deadbeef")
    record = el.build_provenance(args, ckpt_sha256="a", agree_sha256="b", split_hash="c",
                                 weights_source="ema", n_queries=3,
                                 dataset_config=_real_dataset_config(),
                                 context_digest="ctxdigest")
    assert _R3FIX_PROVENANCE_KEYS <= set(record)
    assert record["registration_sha"] == "deadbeef"
    assert record["model_config_sha256"] == _h.sha256(
        open(model_config, "rb").read()).hexdigest()
    assert record["dataset_config_sha256"] == _h.sha256(open(_UNSEEN_CONFIG, "rb").read()).hexdigest()
    assert record["context_k"] == 8                      # modalities.acoustic_context.max_context
    assert record["loader_shuffle"] is False and record["loader_drop_last"] is False
    assert record["float32_matmul_precision"] == torch.get_float32_matmul_precision()
    assert record["flash_attn_available"] in (True, False)
    assert record["context_stream_digest"] == "ctxdigest"
    assert isinstance(record["device_name"], str) and record["device_name"]


def test_build_provenance_defaults_unavailable_fields_to_na(tmp_path):
    record = el.build_provenance(_args(model_config="missing.json"), ckpt_sha256="a",
                                 agree_sha256="b", split_hash="c", weights_source="ema",
                                 n_queries=1)
    assert record["registration_sha"] == "n/a"
    assert record["model_config_sha256"] == "n/a"          # the file does not exist
    assert len(record["dataset_config_sha256"]) == 64      # the default config does
    assert record["context_k"] is None and record["context_stream_digest"] == "n/a"


def test_assert_registration_sha_is_required_for_a_registered_unseen_run():
    unseen = _real_dataset_config()
    args = _split_args(_UNSEEN_CONFIG)
    with pytest.raises(SystemExit):
        el.assert_registration_sha(args, unseen)
    # a bare SHA is no longer enough: the locked protocol must be machine-checkable
    with pytest.raises(SystemExit):
        el.assert_registration_sha(_split_args(_UNSEEN_CONFIG, "--registration-sha", "abc123"),
                                   unseen)
    el.assert_registration_sha(
        _split_args(_UNSEEN_CONFIG, "--registration-sha", "abc123",
                    "--registration-manifest", "reg.json"), unseen)


def test_assert_registration_sha_is_not_required_for_smoke_seen_or_oracle_runs():
    unseen, seen = _real_dataset_config(), _json.loads(open(_SEEN_CONFIG).read())
    el.assert_registration_sha(_split_args(_SEEN_CONFIG), seen)
    el.assert_registration_sha(_split_args(_SEEN_CONFIG, "--smoke", "--max-queries", "2"), seen)
    oracle = el.parse_args(["--model-config", "m.json", "--dataset-config", _UNSEEN_CONFIG,
                            "--agree-ckpt", "a.pt", "--score-source", "gt_rir"])
    el.assert_registration_sha(oracle, unseen)           # registered checkpoint-free R-1 run


def test_context_stream_digest_is_order_sensitive_and_uses_the_fingerprints():
    from eval_FLAC import canonical_stream_hash
    rows = [{"context_xyz_cam": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]},
            {"context_xyz_cam": [[3.0, 0.0, 0.0]]}]
    expected = canonical_stream_hash([
        tuple(el.render_position_id(xyz) for xyz in row["context_xyz_cam"]) for row in rows])
    assert el.context_stream_digest(rows) == expected
    assert el.context_stream_digest(rows) != el.context_stream_digest(list(reversed(rows)))
    assert el.context_stream_digest([{"e_loc": 1.0}]) == "n/a"


# --------------------------------------------------------------------------- #
# r3 fix F4 (review finding 4): the summary must carry the registered
# comparisons -- FLAC on the SAME retained subset as the information-matched
# baseline, the 17-room clustered CI, the paired room-clustered tests, and the
# §2.8.2 power statistic.
# --------------------------------------------------------------------------- #
from src.localization.scoring import (clustered_bootstrap_ci,  # noqa: E402
                                      paired_room_clustered_test, power_statistic)


def _rows_with_context():
    rows = _rows_fixture()
    cams = el.candidate_camera_positions(_cand_set())
    for row in rows:
        row["context_xyz_cam"] = [list(cams[0]), list(cams[2])]
        row["context_sims_hex"] = el.encode_sims(torch.tensor([[0.2, 0.8]],
                                                              dtype=torch.float32))[0]
    return rows


def test_build_row_records_the_per_query_power_statistic():
    row = el.build_row(**_row_kwargs())
    sims = _row_kwargs()["sims"]
    assert row["power_statistic"] == pytest.approx(power_statistic(sims))
    single = el.build_row(**_row_kwargs(sims=torch.tensor([[0.5], [0.9], [0.1]],
                                                          dtype=torch.float32),
                                        noise_keys=[1]))
    assert single["power_statistic"] is None            # K = 1: no within-candidate variance


def test_summarize_run_matches_flac_to_the_same_retained_subset():
    rows = _rows_fixture()
    summary = el.summarize_run(rows, bootstrap_n=50, bootstrap_seed=1)
    kept = [r for r in rows if not r["gt_only"]]
    expected = summarize([{"query_id": r["query_id"], "room_id": r["room_id"],
                           "e_loc": r["e_loc"], "top1": r["top1"], "rr": r["rr"]} for r in kept])
    assert summary["flac_excl_gt_only"] == expected
    assert summary["flac_excl_gt_only"]["n_queries"] == \
        summary["baselines"]["context_conditioned_excl_gt_only"]["n_queries"]


def test_summarize_run_includes_the_clustered_ci_on_the_primary():
    rows = _rows_fixture()
    summary = el.summarize_run(rows, bootstrap_n=50, bootstrap_seed=1)
    records = [{"query_id": r["query_id"], "room_id": r["room_id"], "e_loc": r["e_loc"]}
               for r in rows]
    assert summary["statistics"]["clustered_ci"] == clustered_bootstrap_ci(
        records, n=50, seed=1)


def test_summarize_run_includes_the_paired_room_clustered_tests():
    rows = _rows_with_context()
    summary = el.summarize_run(rows, bootstrap_n=50, bootstrap_seed=1)
    flac = [{"query_id": r["query_id"], "room_id": r["room_id"], "e_loc": r["e_loc"]}
            for r in rows]
    context = []
    for row in rows:
        out = context_conditioned_baseline(np.asarray(row["candidate_xyz_world"]),
                                           np.asarray(row["gt_xyz_world"]),
                                           row["context_member"])
        context.append({"query_id": row["query_id"], "room_id": row["room_id"],
                        "distances": out["distances"]})
    assert summary["statistics"]["paired_vs_context_conditioned"] == paired_room_clustered_test(
        flac, context, n=50, seed=1)
    assert summary["statistics"]["paired_vs_nearest_context_masked"] is not None


def test_summarize_run_paired_control_is_none_without_context_evidence():
    summary = el.summarize_run(_rows_fixture(), bootstrap_n=20, bootstrap_seed=1)
    assert summary["statistics"]["paired_vs_nearest_context_masked"] is None
    assert summary["controls"]["nearest_context_masked_excl_gt_only"] is None


def test_summarize_run_aggregates_the_power_statistic_distribution():
    rows = _rows_fixture()
    summary = el.summarize_run(rows, bootstrap_n=20, bootstrap_seed=1)
    values = [row["power_statistic"] for row in rows]
    block = summary["power_statistic"]
    assert block["n_queries"] == 3
    assert block["mean"] == pytest.approx(float(np.mean(values)))
    assert block["median"] == pytest.approx(float(np.median(values)))
    assert block["min"] == pytest.approx(min(values)) and block["max"] == pytest.approx(max(values))


# --------------------------------------------------------------------------- #
# r3 fix F2 (review finding 2): parity must exercise the DEFINING M x K path --
# candidate tiling, cross-attention conditioning and masks, common noise --
# not just a single-item generation. (a) always-running synthetic model.
# --------------------------------------------------------------------------- #
from src.inference.sampling import sample_discrete_euler          # noqa: E402
from src.models.factory import create_model_from_config           # noqa: E402
from src.training.factory import create_training_wrapper_from_config  # noqa: E402


def _tiny_module():
    """A random-init diffusion_cond model that produces the FULL conditioning the
    real config produces: cross_attn_cond + cross_attn_mask + global_cond."""
    cfg = {
        "model_type": "diffusion_cond", "sample_size": 64, "sample_rate": 22050,
        "audio_channels": 1,
        "model": {
            "conditioning": {"configs": [
                {"id": "source", "type": "dist_embedder",
                 "config": {"num_freqs": 4, "max_freq": 4, "ch_dim": 1, "include_in": True}},
                {"id": "context_poses", "type": "dist_embedder",
                 "config": {"num_freqs": 4, "max_freq": 4, "ch_dim": 1, "include_in": True}}],
                "cond_dim": 32},
            "diffusion": {"cross_attention_cond_ids": ["context_poses"],
                          "global_cond_ids": ["source"], "type": "dit",
                          "diffusion_objective": "rectified_flow",
                          "config": {"io_channels": 4, "embed_dim": 64, "depth": 1,
                                     "num_heads": 2, "cond_token_dim": 32,
                                     "global_cond_dim": 32,
                                     "transformer_type": "continuous_transformer",
                                     "global_cond_type": "adaLN"}},
            "io_channels": 4},
        "training": {"timestep_sampler": "uniform", "cfg_dropout_prob": 0.0, "use_ema": False,
                     "optimizer_configs": {"diffusion": {"optimizer": {
                         "type": "AdamW", "config": {"lr": 5e-6, "betas": [0.9, 0.999],
                                                     "weight_decay": 1e-3}}}}}}
    module = create_training_wrapper_from_config(cfg, create_model_from_config(cfg))
    module.eval().requires_grad_(False)
    return module


def _tiny_engine(module, recorder=None):
    dit = module.diffusion.model

    def sampler(noise, cond_inputs):
        if recorder is not None:
            recorder.append({k: (v.clone() if torch.is_tensor(v) else v)
                             for k, v in cond_inputs.items()})
        with torch.no_grad():
            return sample_discrete_euler(dit, noise, 1, **cond_inputs, cfg_scale=1.0,
                                         dist_shift=module.diffusion.dist_shift,
                                         batch_cfg=True, disable_tqdm=True)

    def embedder(wavs):
        feats = wavs.reshape(wavs.shape[0], -1)[:, :8]
        return torch.nn.functional.normalize(feats.float(), dim=-1)

    return el.Engine(device="cpu", io_channels=4, latent_samples=64,
                     conditioner=lambda mds, dev: module.diffusion.conditioner(mds, "cpu"),
                     cond_inputs_fn=module.diffusion.get_conditioning_inputs,
                     sampler=sampler,
                     decoder=lambda latents: latents.reshape(latents.shape[0], 1, -1),
                     embedder=embedder)


def _tiny_query(module, cand):
    """Base metadata for the tiny model: real GT projection + context poses."""
    source = torch.as_tensor(project_to_camera(cand.rec_loc, cand.gt_xyz), dtype=torch.float32)
    g = torch.Generator().manual_seed(5)
    return {"source": source, "source_vit": source.unsqueeze(0),
            "context_poses": torch.randn(2, 3, generator=g)}


@pytest.mark.parametrize("batch_size", [64, 4])
def test_run_query_matches_a_candidate_major_replay_with_full_conditioning(batch_size):
    """M=3, K=2 through run_query vs an explicit per-(m, k) replay: identical
    waveforms, and every conditioning key -- including the cross-attention mask --
    on row m*K+k is candidate m's."""
    module = _tiny_module()
    cand = _cand_set()
    base_md = _tiny_query(module, cand)
    noise = el.build_noise_bank(42, "mk_parity", 2, (4, 64))
    seen = []
    engine = _tiny_engine(module, recorder=seen)

    out = el.run_query(engine, base_md, cand, noise, _OBS, batch_size=batch_size,
                       return_wavs=True)

    cams = el.candidate_camera_positions(cand)
    replay = []
    with torch.no_grad():
        for m in range(3):
            md_m = candidate_metadata(base_md, cams[m])
            cond_m = module.diffusion.get_conditioning_inputs(
                module.diffusion.conditioner([md_m], "cpu"))
            for k in range(2):
                latents = sample_discrete_euler(
                    module.diffusion.model, noise[k:k + 1], 1, **cond_m, cfg_scale=1.0,
                    dist_shift=module.diffusion.dist_shift, batch_cfg=True, disable_tqdm=True)
                replay.append(latents.reshape(1, 1, -1).clamp(-1.0, 1.0))
    assert torch.equal(out["wavs"], torch.cat(replay))

    # every conditioning key, per row, is candidate m's -- masks included
    batched = module.diffusion.get_conditioning_inputs(module.diffusion.conditioner(
        [candidate_metadata(base_md, cams[m]) for m in range(3)], "cpu"))
    recorded = {}
    for chunk in seen:
        for key, value in chunk.items():
            if torch.is_tensor(value):
                recorded.setdefault(key, []).append(value)
    assert set(recorded) >= {"cross_attn_cond", "cross_attn_mask", "global_cond"}
    for key, chunks in recorded.items():
        rows = torch.cat(chunks)
        assert rows.shape[0] == 6
        for m in range(3):
            for k in range(2):
                assert torch.equal(rows[m * 2 + k], batched[key][m]), f"{key} row {m}*2+{k}"


def test_run_query_mk_replay_is_stable_across_the_batch_split():
    module = _tiny_module()
    cand = _cand_set()
    base_md = _tiny_query(module, cand)
    noise = el.build_noise_bank(42, "mk_parity", 2, (4, 64))
    whole = el.run_query(_tiny_engine(module), base_md, cand, noise, _OBS, batch_size=64,
                         return_wavs=True)
    split = el.run_query(_tiny_engine(module), base_md, cand, noise, _OBS, batch_size=4,
                         return_wavs=True)
    assert torch.equal(whole["wavs"], split["wavs"])
    assert torch.equal(whole["sims"], split["sims"])


# --------------------------------------------------------------------------- #
# r3 fix F2(b): the same M x K parity on REAL assets -- a seen-split query, real
# metadata (depth + drawn context), the released checkpoint. The dataset is
# mid-download, so the room probe skips with a precise reason if nothing local is
# complete.
# --------------------------------------------------------------------------- #
_SEEN_SPLIT_JSON = os.path.join(_REPO_ROOT, "data", "AR", "seen_eval.json")
_AR_ROOT = os.path.join(_REPO_ROOT, "AcousticRooms")
_AR_MD_PATH = os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs", "custom_metadata",
                           "AR_md.py")


def _find_complete_seen_query(min_candidates=3):
    """First seen-split query whose wav, pair JSONs and depth map are all local."""
    if not (os.path.isfile(_SEEN_SPLIT_JSON) and os.path.isdir(_AR_ROOT)):
        return None
    from src.localization.candidates import build_candidate_set, parse_ir_filename
    split = _json.loads(open(_SEEN_SPLIT_JSON).read())
    metadata_root = os.path.join(_AR_ROOT, "metadata")
    for scene in sorted(split):
        for room in sorted(split[scene]):
            wav_dir = os.path.join(_AR_ROOT, "single_channel_ir_1", scene, room)
            depth_dir = os.path.join(_AR_ROOT, "depth_map", scene, room)
            if not (os.path.isdir(wav_dir) and os.path.isdir(depth_dir)):
                continue
            for fname in sorted(split[scene][room]):
                wav = os.path.join(wav_dir, fname)
                if not os.path.isfile(wav):
                    continue
                try:
                    _src, receiver = parse_ir_filename(fname)
                except ValueError:
                    continue
                if not os.path.isfile(os.path.join(depth_dir, f"{receiver}.npy")):
                    continue
                try:
                    cand = build_candidate_set(wav, metadata_root)
                except (ValueError, OSError):
                    continue
                if len(cand.nodes) >= min_candidates:
                    return {"wav": wav, "cand": cand, "scene": scene, "room": room}
    return None


_SEEN_QUERY = _find_complete_seen_query() if _HAVE_GEN_ASSETS else None
real_mk_integration = pytest.mark.skipif(
    not (_HAVE_GEN_ASSETS and _SEEN_QUERY is not None),
    reason="no SEEN-split room with wav + metadata pair JSONs + depth_map is complete locally "
           "(AcousticRooms is mid-download), or the FLAC/DINOv3 assets are absent")


def _real_query_metadata(entry):
    """The loader's own metadata for that query (AR_md, importlib-loaded by path)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("metadata_module", _AR_MD_PATH)
    ar_md = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ar_md)
    info = {"path": entry["wav"], "relpath": os.path.relpath(entry["wav"], _AR_ROOT),
            "modalities": {"acoustic_context": {"load": True, "max_context": 8, "max_len": 9600},
                           "depth": {"load": True}, "poses": {"load": True}}}
    return ar_md.get_custom_metadata(info, None)


def _real_mk_replay(args, engine, context, md, cand, noise, autocast_mode):
    """Explicit per-(m, k) replay of evaluate_model's calls for the same query."""
    import contextlib
    module, model = context["module"], context["model"]
    cams = el.candidate_camera_positions(cand)
    replay = []
    with torch.no_grad():
        for m in range(len(cand.nodes)):
            cond_ctx = (torch.amp.autocast(context["device"]) if autocast_mode == "default"
                        else contextlib.nullcontext())
            with cond_ctx:
                conditioning = module.diffusion.conditioner([candidate_metadata(md, cams[m])],
                                                            module.device)
            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)
            for k in range(int(noise.shape[0])):
                latents = sample_discrete_euler(
                    model, noise[k:k + 1].to(engine.device), args.steps, **cond_inputs,
                    cfg_scale=args.cfg_scale, dist_shift=module.diffusion.dist_shift,
                    batch_cfg=True, disable_tqdm=True)
                replay.append(module.diffusion.pretransform.decode(latents).clamp(-1.0, 1.0))
    return torch.cat(replay)


def _real_mk_setup(entry, min_keep=3):
    md = _real_query_metadata(entry)
    full = entry["cand"]
    keep = sorted({full.gt_node} | set(full.nodes[:min_keep]))[:min_keep]
    rows = [full.nodes.index(n) for n in keep]
    cand = CandidateSet(nodes=keep, xyz_world=full.xyz_world[rows], rec_loc=full.rec_loc,
                        gt_node=full.gt_node, gt_xyz=full.gt_xyz)
    return md, cand


def _light_embedder(wavs):
    """Keeps the 350 MB AGREE load out of a GENERATION-parity test."""
    return torch.nn.functional.normalize(
        wavs.reshape(wavs.shape[0], -1)[:, :8].float() + 1e-3, dim=-1)


@real_mk_integration
def test_integration_real_mk_parity_is_exact_without_conditioning_autocast():
    """M=3 real candidates, K=2, released checkpoint, --cond-autocast off: the
    driver's M x K path IS the candidate-major replay, bit for bit."""
    import dataclasses
    previous = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        args = el.validate_args(el.parse_args([
            "--model-config", _FLAC_CONFIG, "--dataset-config", _SEEN_CONFIG,
            "--ckpt-path", _FLAC_CKPT, "--agree-ckpt", "unused", "--num-samples", "2",
            "--device", "cpu", "--cond-autocast", "off"]))
        engine, context = el.build_engine(args, agree=None, device="cpu")
        engine = dataclasses.replace(engine, embedder=_light_embedder)
        md, cand = _real_mk_setup(_SEEN_QUERY)
        noise = el.build_noise_bank(42, "real_mk", 2, context["latent_shape"])
        obs = torch.full((1, 1, 9600), 0.1)

        out = el.run_query(engine, md, cand, noise, obs, batch_size=64, return_wavs=True)
        assert tuple(out["sims"].shape) == (len(cand.nodes), 2)
        assert torch.equal(out["wavs"], _real_mk_replay(args, engine, context, md, cand, noise,
                                                        "off"))
    finally:
        os.chdir(previous)


@real_mk_integration
def test_integration_real_mk_batch_split_is_bitwise_irrelevant(real_engine):
    """At the REGISTERED --cond-autocast default, splitting the M x K rows into two
    chunks changes nothing bitwise; the replay agrees to ~1e-3 of full scale
    because that autocast conditions in bf16, whose reduction order depends on how
    many candidates are conditioned together -- a property of the conditioner's
    precision, not of the driver's wiring (the fp32 test above is exact)."""
    import dataclasses
    args, engine, context = real_engine
    engine = dataclasses.replace(engine, embedder=_light_embedder)
    md, cand = _real_mk_setup(_SEEN_QUERY)
    noise = el.build_noise_bank(42, "real_mk", 2, context["latent_shape"])
    obs = torch.full((1, 1, 9600), 0.1)

    whole = el.run_query(engine, md, cand, noise, obs, batch_size=64, return_wavs=True)
    chunked = el.run_query(engine, md, cand, noise, obs, batch_size=3, return_wavs=True)
    assert torch.equal(whole["wavs"], chunked["wavs"])
    assert torch.equal(whole["sims"], chunked["sims"])

    replay = _real_mk_replay(args, engine, context, md, cand, noise, "default")
    assert torch.allclose(whole["wavs"], replay, rtol=0, atol=2e-3)


def test_module_runs_as_a_script_with_every_symbol_defined():
    """Regression: the __main__ guard once sat mid-file, so anything defined after
    it did not exist when the driver was RUN (imports in tests still passed). The
    CLI must reach its own refusal, not a NameError."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, os.path.join(_REPO_ROOT, "eval_localization.py"),
         "--model-config", "missing.json", "--dataset-config", "missing.json",
         "--ckpt-path", "missing.ckpt", "--agree-ckpt", "missing.pt",
         "--num-samples", "2", "--rotate-deg", "45"],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode != 0
    assert "REFUSED" in (result.stderr + result.stdout)
    assert "NameError" not in result.stderr


# --------------------------------------------------------------------------- #
# r4 item 1 (full-review F1, plan Rev 3.1 §2): the candidate authority is frozen
# ONCE per run into a room manifest, hashed into provenance and consumed from
# memory. Per-query disk re-enumeration could change M -- and therefore the
# autocast conditioning batch composition -- between seeds.
# --------------------------------------------------------------------------- #
def _split_dict():
    return {"Cafe": {"Cafe_idx_1": ["S003_R0011_hybrid_IR.wav", "S000_R0011_hybrid_IR.wav"]}}


def test_build_room_manifest_freezes_nodes_coordinates_and_wav_availability(tmp_path):
    root, _wav_room = _dataset_tree(tmp_path)
    manifest = el.build_room_manifest(str(root), _split_dict())
    room = manifest["rooms"]["Cafe/Cafe_idx_1"]
    assert room["nodes"] == [0, 3, 7]
    assert room["xyz_world"][1] == [2.0, -1.0, 1.5]
    assert room["wav_nodes"] == [0, 3, 7]
    assert room["receivers"]["11"] == [1.0, 2.0, 0.5]         # rec_loc frozen per receiver
    assert room["n_metadata_sources"] == 3 and room["n_wav_sources"] == 3
    assert manifest["folder_name"] == "single_channel_ir_1"


def test_build_room_manifest_is_deterministic_and_hashable(tmp_path):
    root, _wav_room = _dataset_tree(tmp_path)
    first = el.build_room_manifest(str(root), _split_dict())
    second = el.build_room_manifest(str(root), _split_dict())
    assert first == second
    assert el.manifest_sha256(first) == el.manifest_sha256(second)
    moved = _json.loads(_json.dumps(first))
    moved["rooms"]["Cafe/Cafe_idx_1"]["xyz_world"][0][0] += 0.001
    assert el.manifest_sha256(moved) != el.manifest_sha256(first)


def test_build_room_manifest_is_json_serializable(tmp_path):
    root, _wav_room = _dataset_tree(tmp_path)
    manifest = el.build_room_manifest(str(root), _split_dict())
    assert _json.loads(_json.dumps(manifest)) == manifest


def test_build_room_manifest_aborts_on_a_missing_room(tmp_path):
    root, _wav_room = _dataset_tree(tmp_path)
    split = {"Cafe": {"Cafe_idx_1": ["S003_R0011_hybrid_IR.wav"]},
             "Ghost": {"Ghost_idx_0": ["S000_R000_hybrid_IR.wav"]}}
    with pytest.raises(SystemExit):
        el.build_room_manifest(str(root), split)


def test_candidate_set_from_manifest_equals_the_on_disk_construction(tmp_path):
    root, wav_room = _dataset_tree(tmp_path)
    manifest = el.build_room_manifest(str(root), _split_dict())
    md = _query_md(root, wav_room)
    from_disk = el.query_candidate_set(md)
    frozen = el.candidate_set_from_manifest(manifest, "Cafe/Cafe_idx_1", gt_node=3, rec_node=11)
    assert frozen.nodes == from_disk.nodes and frozen.gt_node == from_disk.gt_node
    np.testing.assert_array_equal(frozen.xyz_world, from_disk.xyz_world)
    np.testing.assert_array_equal(frozen.rec_loc, from_disk.rec_loc)
    np.testing.assert_array_equal(frozen.gt_xyz, from_disk.gt_xyz)


def test_candidate_set_from_manifest_aborts_on_an_unknown_room_or_receiver(tmp_path):
    root, _wav_room = _dataset_tree(tmp_path)
    manifest = el.build_room_manifest(str(root), _split_dict())
    with pytest.raises(ValueError):
        el.candidate_set_from_manifest(manifest, "Ghost/Ghost_idx_0", 3, 11)
    with pytest.raises(ValueError):
        el.candidate_set_from_manifest(manifest, "Cafe/Cafe_idx_1", 3, 99)
    with pytest.raises(ValueError):
        el.candidate_set_from_manifest(manifest, "Cafe/Cafe_idx_1", 99, 11)


# --- real unseen split (Rev 3.1 §1: the corrected LRH fact) ------------------ #
_UNSEEN_SPLIT_JSON = os.path.join(_REPO_ROOT, "data", "AR", "unseen_eval.json")
_HAVE_UNSEEN_ROOMS = (os.path.isfile(_UNSEEN_SPLIT_JSON) and os.path.isdir(_AR_ROOT)
                      and os.path.isdir(os.path.join(_AR_ROOT, "metadata")))
unseen_rooms = pytest.mark.skipif(
    not _HAVE_UNSEEN_ROOMS, reason="the unseen split's AcousticRooms metadata is not local")


@unseen_rooms
def test_integration_every_unseen_room_has_ten_metadata_sources():
    """Rev 3.1 §1: all 17 unseen rooms have M=10; LivingRoomsWithHallway_idx_30's
    source 10 has metadata but no wavs, so its eligible set is {GT, S10} = 2 --
    NOT the retired M=9 / GT-only case."""
    split = _json.loads(open(_UNSEEN_SPLIT_JSON).read())
    manifest = el.build_room_manifest(_AR_ROOT, split)
    assert len(manifest["rooms"]) == 17
    assert {len(room["nodes"]) for room in manifest["rooms"].values()} == {10}
    lrh = manifest["rooms"]["LivingRoomsWithHallway/LivingRoomsWithHallway_idx_30"]
    assert lrh["nodes"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert lrh["wav_nodes"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert sorted(set(lrh["nodes"]) - set(lrh["wav_nodes"])) == [10]
    for room_id, room in manifest["rooms"].items():
        if room_id != "LivingRoomsWithHallway/LivingRoomsWithHallway_idx_30":
            assert room["wav_nodes"] == room["nodes"], room_id


def test_run_evaluation_records_the_frozen_manifest_hash(tmp_path):
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    context = _stub_context(root)
    result = el.run_evaluation(_run_args(tmp_path), loader, engine, context, "c", "a",
                               expected=el.expected_split_identities(loader.dataset))
    assert result["provenance"]["candidate_manifest_sha256"] == el.manifest_sha256(
        context["manifest"])
    assert len(result["provenance"]["candidate_manifest_sha256"]) == 64


def test_process_query_refuses_to_run_without_a_frozen_manifest(tmp_path):
    root, wav_room = _dataset_tree(tmp_path)
    md = _query_md(root, wav_room)
    _rec, engine = _engine()
    with pytest.raises(SystemExit):
        el.process_query(_run_args(tmp_path), engine, {"latent_shape": (2, 8)}, md,
                         torch.full((1, 1, 9600), 0.2))


# --------------------------------------------------------------------------- #
# r4 items 6+7 (full-review F6, F7 and the Part-1 #9 leftover)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_frame_avg_angles_reject_non_finite_values_at_parse_time(bad):
    with pytest.raises(SystemExit):
        el.parse_args(_CLI + ["--cond-method", "fa_invariant", "--frame-avg-angles", "0", bad])
    el.parse_args(_CLI + ["--cond-method", "fa_invariant", "--frame-avg-angles", "0", "90"])


def test_device_provenance_block_on_cpu():
    block = el.device_provenance("cpu")
    assert block["device_requested"] == "cpu" and block["device_name"] == "cpu"
    assert block["device_index"] is None and block["device_capability"] is None
    assert block["device_uuid"] == "n/a"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_device_provenance_resolves_the_requested_cuda_index():
    index = min(1, torch.cuda.device_count() - 1)
    block = el.device_provenance(f"cuda:{index}")
    assert block["device_index"] == index
    assert block["device_name"] == torch.cuda.get_device_name(index)
    assert tuple(block["device_capability"]) == torch.cuda.get_device_capability(index)
    assert isinstance(block["device_uuid"], str) and block["device_uuid"]


def test_provenance_carries_the_resolved_device_block():
    record = el.build_provenance(_args(), ckpt_sha256="a", agree_sha256="b", split_hash="c",
                                 weights_source="ema", n_queries=1)
    for key in ("device_requested", "device_index", "device_name", "device_capability",
                "device_uuid"):
        assert key in record


def test_main_validates_registration_before_loading_any_model(tmp_path, monkeypatch):
    """A registered unseen run without --registration-sha must be refused before
    the checkpoint is read and before AGREE is constructed (F7)."""
    def _never(*args, **kwargs):
        raise AssertionError("nothing may be loaded before registration validates")

    monkeypatch.setattr(el, "load_agree_audio", _never)
    monkeypatch.setattr(el, "build_engine", _never)
    monkeypatch.setattr(el, "load_and_validate_artifacts",
                        lambda args: ({"model": {"diffusion": {
                            "diffusion_objective": "rectified_flow"}},
                            "sample_size": 10240, "sample_rate": 22050}, None))
    with pytest.raises(SystemExit):
        el.main(["--model-config", "m.json", "--dataset-config", _UNSEEN_CONFIG,
                 "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt", "--num-samples", "8",
                 "--device", "cpu"])


# --------------------------------------------------------------------------- #
# r4 item 3 (full-review F3): the nearest-context control is a REGISTERED success
# criterion, so a run that cannot compute it must abort -- not publish with the
# control set to None. No leniency flag: a half-registered artifact is worse than
# a refusal.
# --------------------------------------------------------------------------- #
def test_resolve_context_k_reads_the_registered_context_size():
    assert el.resolve_context_k(_json.loads(open(_UNSEEN_CONFIG).read())) == 8
    one = _json.loads(open(os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs", "AR",
                                        "eval", "acousticroom_unseeneval_1.json")).read())
    assert el.resolve_context_k(one) == 1


@pytest.mark.parametrize("config", [
    {"modalities": {}},
    {"modalities": {"acoustic_context": {"load": False, "max_context": 8}}},
    {"modalities": {"acoustic_context": {"load": True}}},
    {"modalities": {"acoustic_context": {"load": True, "max_context": 0}}},
])
def test_resolve_context_k_refuses_a_run_that_cannot_produce_the_control(config):
    with pytest.raises(SystemExit):
        el.resolve_context_k(config)


def test_assert_query_context_accepts_the_loader_shapes(tmp_path):
    root, wav_room = _dataset_tree(tmp_path)
    el.assert_query_context(_query_md(root, wav_room), 2)


def test_assert_query_context_aborts_on_missing_or_mis_shaped_context(tmp_path):
    root, wav_room = _dataset_tree(tmp_path)
    base = _query_md(root, wav_room)
    for mutate in (
        lambda md: md.pop("context_poses"),
        lambda md: md.pop("context_audio"),
        lambda md: md.__setitem__("context_poses", md["context_poses"][:1]),
        lambda md: md.__setitem__("context_audio", md["context_audio"][:1]),
        lambda md: md.__setitem__("context_poses", md["context_poses"].double()),
        lambda md: md.__setitem__("context_poses", torch.zeros(2, 4)),
        lambda md: md.__setitem__("context_audio", torch.zeros(2)),
    ):
        md = dict(base)
        mutate(md)
        with pytest.raises(ValueError):
            el.assert_query_context(md, 2)


def test_process_query_aborts_before_generation_when_context_is_short(tmp_path):
    root, wav_room = _dataset_tree(tmp_path)
    md = _query_md(root, wav_room)
    md["context_poses"] = md["context_poses"][:1]
    md["context_audio"] = md["context_audio"][:1]
    rec, engine = _engine()
    context = _stub_context(root)
    context["context_k"] = 2
    with pytest.raises(ValueError):
        el.process_query(_run_args(tmp_path), engine, context, md,
                         torch.full((1, 1, 9600), 0.2))
    assert rec.calls == []


def test_assert_context_evidence_complete_gates_publication():
    good = [{"context_xyz_cam": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "query_id": "q0"}]
    el.assert_context_evidence_complete(good, 2)
    with pytest.raises(SystemExit):
        el.assert_context_evidence_complete([{"query_id": "q0"}], 2)
    with pytest.raises(SystemExit):
        el.assert_context_evidence_complete(good, 8)          # short context


def test_run_evaluation_publishes_a_full_length_context_digest(tmp_path):
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    context = _stub_context(root)
    context["context_k"] = 2
    result = el.run_evaluation(_run_args(tmp_path), loader, engine, context, "c", "a",
                               expected=el.expected_split_identities(loader.dataset))
    assert result["provenance"]["context_stream_digest"] != "n/a"
    assert all(len(row["context_xyz_cam"]) == 2 for row in result["rows"])
    assert result["summary"]["controls"]["nearest_context_masked"] is not None


# --------------------------------------------------------------------------- #
# r4 item 4 (full-review F4): a registered unseen run must be locked to a
# COMMITTED manifest. The old check accepted any non-empty string, so K, tau,
# agg, the checkpoint or the scorer could all be overridden while the artifact
# still looked headline-shaped.
# --------------------------------------------------------------------------- #
def _locked(**over):
    locked = {"model_config_sha256": "m" * 64, "dataset_config_sha256": "d" * 64,
              "ckpt_sha256": "c" * 64, "agree_sha256": "a" * 64, "num_samples": 8,
              "tau": 0.02, "agg": "lme", "cond_method": "vanilla", "cond_autocast": "default",
              "steps": 1, "cfg_scale": 1.0, "seeds": [42, 43, 44], "readout": "mean",
              "candidate_manifest_sha256": "f" * 64}
    locked.update(over)
    return locked


def _resolved(**over):
    resolved = {"model_config_sha256": "m" * 64, "dataset_config_sha256": "d" * 64,
                "ckpt_sha256": "c" * 64, "agree_sha256": "a" * 64, "num_samples": 8,
                "tau": 0.02, "agg": "lme", "cond_method": "vanilla", "cond_autocast": "default",
                "steps": 1, "cfg_scale": 1.0, "seed": 43, "readout": "mean",
                "candidate_manifest_sha256": "f" * 64}
    resolved.update(over)
    return resolved


def _git_repo_with_manifest(tmp_path, manifest, name="registration.json"):
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    path = tmp_path / name
    path.write_text(_json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    subprocess.run(["git", "add", name], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "-m", "register"], cwd=tmp_path, check=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
                         capture_output=True, text=True).stdout.strip()
    return str(path), sha


def test_verify_registration_commit_accepts_a_committed_manifest(tmp_path):
    path, sha = _git_repo_with_manifest(tmp_path, _locked())
    assert el.verify_registration_commit(path, sha, repo_root=str(tmp_path)) == _locked()


def test_verify_registration_commit_refuses_a_sha_that_is_not_a_commit(tmp_path):
    path, _sha = _git_repo_with_manifest(tmp_path, _locked())
    with pytest.raises(SystemExit):
        el.verify_registration_commit(path, "0" * 40, repo_root=str(tmp_path))
    with pytest.raises(SystemExit):
        el.verify_registration_commit(path, "not-a-sha", repo_root=str(tmp_path))


def test_verify_registration_commit_refuses_content_drift(tmp_path):
    path, sha = _git_repo_with_manifest(tmp_path, _locked())
    drifted = _locked(tau=0.5)
    open(path, "w").write(_json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    with pytest.raises(SystemExit):
        el.verify_registration_commit(path, sha, repo_root=str(tmp_path))


def test_verify_registration_commit_refuses_an_uncommitted_manifest(tmp_path):
    path, sha = _git_repo_with_manifest(tmp_path, _locked())
    other = os.path.join(str(tmp_path), "other.json")
    open(other, "w").write(_json.dumps(_locked()))
    with pytest.raises(SystemExit):
        el.verify_registration_commit(other, sha, repo_root=str(tmp_path))


def test_check_registration_fields_passes_on_an_exact_match():
    el.check_registration_fields(_locked(), _resolved(), registered=True)


@pytest.mark.parametrize("drift", [
    {"num_samples": 4}, {"tau": 0.05}, {"agg": "mean"}, {"cond_method": "fa_invariant"},
    {"cond_autocast": "off"}, {"steps": 2}, {"cfg_scale": 2.0}, {"readout": "sample"},
    {"ckpt_sha256": "x" * 64}, {"agree_sha256": "x" * 64}, {"model_config_sha256": "x" * 64},
    {"dataset_config_sha256": "x" * 64}, {"candidate_manifest_sha256": "x" * 64},
])
def test_check_registration_fields_refuses_any_protocol_override(drift):
    with pytest.raises(SystemExit):
        el.check_registration_fields(_locked(), _resolved(**drift), registered=True)


def test_check_registration_fields_refuses_a_seed_outside_the_registered_list():
    with pytest.raises(SystemExit):
        el.check_registration_fields(_locked(), _resolved(seed=99), registered=True)
    el.check_registration_fields(_locked(seeds=[99]), _resolved(seed=99), registered=True)


def test_check_registration_fields_refuses_a_missing_lock():
    incomplete = _locked()
    del incomplete["agg"]
    with pytest.raises(SystemExit):
        el.check_registration_fields(incomplete, _resolved(), registered=True)


def test_check_registration_fields_allows_tbd_manifest_hash_only_outside_registered_mode():
    el.check_registration_fields(_locked(candidate_manifest_sha256="tbd"), _resolved(),
                                 registered=False)
    with pytest.raises(SystemExit):
        el.check_registration_fields(_locked(candidate_manifest_sha256="tbd"), _resolved(),
                                     registered=True)


def test_registered_unseen_run_requires_both_registration_flags():
    unseen = _json.loads(open(_UNSEEN_CONFIG).read())
    with pytest.raises(SystemExit):
        el.assert_registration_sha(_split_args(_UNSEEN_CONFIG, "--registration-sha", "abc"),
                                   unseen)                        # manifest missing
    with pytest.raises(SystemExit):
        el.assert_registration_sha(_split_args(_UNSEEN_CONFIG, "--registration-manifest", "r.json"),
                                   unseen)                        # sha missing
    el.assert_registration_sha(
        _split_args(_UNSEEN_CONFIG, "--registration-sha", "abc",
                    "--registration-manifest", "r.json"), unseen)


def test_verify_registration_end_to_end_on_a_real_commit(tmp_path):
    path, sha = _git_repo_with_manifest(tmp_path, _locked())
    args = _split_args(_UNSEEN_CONFIG, "--registration-sha", sha,
                       "--registration-manifest", path, "--seed", "43", "--num-samples", "8")
    unseen = _json.loads(open(_UNSEEN_CONFIG).read())
    assert el.verify_registration(args, unseen, _resolved(), repo_root=str(tmp_path)) is True
    bad = _split_args(_UNSEEN_CONFIG, "--registration-sha", sha,
                      "--registration-manifest", path, "--seed", "43", "--num-samples", "4")
    with pytest.raises(SystemExit):
        el.verify_registration(bad, unseen, _resolved(num_samples=4), repo_root=str(tmp_path))


# --------------------------------------------------------------------------- #
# r4 item 2a (full-review F2): --mode readback, R-1's gate. crosscheck_sources_vs_files
# had no caller at all; the rung-4 readback was being done by hand.
# --------------------------------------------------------------------------- #
def _readback_args(tmp_path, root, split_path, **over):
    config = tmp_path / "d.json"
    config.write_text(_json.dumps({
        "dataset_type": "audio_dir", "seeneval": True,
        "modalities": {"acoustic_context": {"load": True, "max_context": 2}},
        "datasets": [{"id": "AcousticRooms", "path": str(root),
                      "json_file_path": str(split_path),
                      "folder_name": "single_channel_ir_1"}]}))
    argv = ["--mode", "readback", "--model-config", "m.json", "--dataset-config", str(config),
            "--out-dir", str(tmp_path / "out"), "--eval-name", "R_minus_1"]
    for flag, value in over.items():
        argv += [flag] if value is True else [flag, str(value)]
    return el.validate_args(el.parse_args(argv))


def _write_split(tmp_path, files=("S003_R0011_hybrid_IR.wav", "S000_R0011_hybrid_IR.wav")):
    path = tmp_path / "split.json"
    path.write_text(_json.dumps({"Cafe": {"Cafe_idx_1": list(files)}}))
    return path


def test_readback_mode_reports_a_clean_room(tmp_path):
    root, _wav_room = _dataset_tree(tmp_path)
    args = _readback_args(tmp_path, root, _write_split(tmp_path))
    report = el.run_readback(args)
    assert report["ok"] is True and report["failures"] == []
    room = report["rooms"]["Cafe/Cafe_idx_1"]
    assert room["metadata_nodes"] == [0, 3, 7] and room["wav_nodes"] == [0, 3, 7]
    assert room["split_sources"] == [0, 3] and room["split_files"] == 2
    assert room["sample_rate"] == 22050 and room["wav_readback_ok"] is True
    assert room["depth_present"] == 1 and room["depth_missing"] == []
    assert len(report["manifest_sha256"]) == 64
    assert os.path.exists(report["report_path"])
    assert _json.loads(open(report["report_path"]).read())["ok"] is True


def test_readback_mode_records_a_metadata_only_source_as_a_warning(tmp_path):
    """LivingRoomsWithHallway_idx_30 is exactly this shape (Rev 3.1 §1): a node
    with metadata and no wavs shrinks the oracle, it does not fail the gate."""
    root, wav_room = _dataset_tree(tmp_path)
    os.remove(str(wav_room / "S007_R0011_hybrid_IR.wav"))
    report = el.run_readback(_readback_args(tmp_path, root, _write_split(tmp_path)))
    assert report["ok"] is True
    room = report["rooms"]["Cafe/Cafe_idx_1"]
    assert room["metadata_only_nodes"] == [7]
    assert any("metadata-only" in w for w in report["warnings"])


def test_readback_mode_fails_on_a_wav_without_metadata(tmp_path):
    root, wav_room = _dataset_tree(tmp_path)
    _write_rir(str(wav_room), 42, 11, 0.5)                       # no pair JSON for node 42
    with pytest.raises(SystemExit):
        el.run_readback(_readback_args(tmp_path, root, _write_split(tmp_path)))
    report = _json.loads(open(os.path.join(str(tmp_path / "out"),
                                           "R_minus_1_readback.json")).read())
    assert report["ok"] is False
    assert any("42" in failure for failure in report["failures"])


def test_readback_mode_fails_on_a_missing_split_file_or_depth_map(tmp_path):
    root, wav_room = _dataset_tree(tmp_path)
    os.remove(str(wav_room / "S003_R0011_hybrid_IR.wav"))        # a SPLIT file is gone
    with pytest.raises(SystemExit):
        el.run_readback(_readback_args(tmp_path, root, _write_split(tmp_path)))

    root2, _wav2 = _dataset_tree(tmp_path / "b", with_depth=False)
    with pytest.raises(SystemExit):
        el.run_readback(_readback_args(tmp_path / "b", root2, _write_split(tmp_path / "b")))


def test_readback_mode_needs_no_checkpoint_or_scorer(tmp_path):
    """R-1 runs the moment the dataset lands: no ckpt, no AGREE, no GPU."""
    root, _wav_room = _dataset_tree(tmp_path)
    args = _readback_args(tmp_path, root, _write_split(tmp_path))
    assert args.ckpt_path is None and args.agree_ckpt is None
    assert el.run_readback(args)["ok"] is True
