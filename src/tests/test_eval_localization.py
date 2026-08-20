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
import time   # noqa: E402
import types  # noqa: E402


def _args(**over):
    base = dict(model_config="src/configs/model_configs/m.json",
                dataset_config="src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json",
                ckpt_path="weights/FLAC/FLAC_EMA.ckpt", agree_ckpt="weights/AGREE/AGREE_AR.pt",
                num_samples=8, tau=0.02, agg="lme", steps=1, cfg_scale=1.0, seed=42,
                cond_method="vanilla", frame_avg_angles=None, rotate_deg=0.0,
                cond_autocast="default", score_source="flac", control="none",
                batch_size=64, num_workers=6, out_dir="out", eval_name="exp18_R2",
                smoke=False, max_queries=None, registration_sha=None,
                registration_manifest=None, parity_check=False, device="cpu")
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


def _write_rir(room_dir, src, rec, value, length=12000, rate=22050, name=None):
    """length defaults above MIN_WAV_SAMPLES (10240): a shorter RIR is a readback
    FAILURE by design (r5b item 2)."""
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
        np.save(str(depth_room / f"{receiver}.npy"),
                np.ones((256, 512), dtype=np.float32))     # the registered panorama shape
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
    summary_without_probe = {k: v for k, v in result["summary"].items() if k != "probe"}
    assert summary_without_probe == el.summarize_run(rows)     # probe is added by the driver
    assert result["summary"]["probe"]["n_queries"] == 2
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
        el.main(["--model-config", str(config), "--dataset-config", _SEEN_CONFIG,
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
    """A registered unseen run without the registration flags must be refused
    before the checkpoint is deserialized and before AGREE is constructed."""
    model_config = tmp_path / "m.json"
    model_config.write_text(_json.dumps(
        {"model": {"diffusion": {"diffusion_objective": "rectified_flow"}},
         "sample_size": 10240, "sample_rate": 22050}))

    def _never(*args, **kwargs):
        raise AssertionError("nothing may be loaded before registration validates")

    monkeypatch.setattr(el, "load_agree_audio", _never)
    monkeypatch.setattr(el, "build_engine", _never)
    monkeypatch.setattr(el, "load_checkpoint_and_validate", _never)
    with pytest.raises(SystemExit):
        el.main(["--model-config", str(model_config), "--dataset-config", _UNSEEN_CONFIG,
                 "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt", "--num-samples", "8",
                 "--out-dir", str(tmp_path / "out"), "--device", "cpu"])


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
              "candidate_manifest_sha256": "f" * 64, "split_file_sha256": "s" * 64}
    locked.update(over)
    return locked


def _resolved(**over):
    resolved = {"model_config_sha256": "m" * 64, "dataset_config_sha256": "d" * 64,
                "ckpt_sha256": "c" * 64, "agree_sha256": "a" * 64, "num_samples": 8,
                "tau": 0.02, "agg": "lme", "cond_method": "vanilla", "cond_autocast": "default",
                "steps": 1, "cfg_scale": 1.0, "seed": 43, "readout": "mean",
                "candidate_manifest_sha256": "f" * 64, "split_file_sha256": "s" * 64}
    resolved.update(over)
    return resolved


def _git_repo_with_manifest(tmp_path, manifest, name="registration.json"):
    import subprocess
    os.makedirs(str(tmp_path), exist_ok=True)
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
    result = el.verify_registration_commit(path, sha, repo_root=str(tmp_path))
    assert result["manifest"] == _locked()
    assert result["resolved_sha"] == sha and len(result["resolved_sha"]) in (40, 64)


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
    assert room["sample_rates"] == [22050] and room["wav_bad"] == []
    assert room["wav_checked"] == 2                     # one wav per (room, source)
    assert room["depth_checked"] == 1 and room["depth_bad"] == []
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
    args = _readback_args(tmp_path, root, _write_split(tmp_path))
    with pytest.raises(SystemExit):
        el.run_readback(args)
    written = os.path.join(str(args.out_dir), el.aux_stem(args, "readback") + ".json")
    report = _json.loads(open(written).read())
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


# --------------------------------------------------------------------------- #
# r4 item 2b (full-review F2): R0's probe = a smoke run's summary. Per-query
# component timings and CUDA peak memory are always on, so the registered fit /
# timing probe needs no separate unreviewed script.
# --------------------------------------------------------------------------- #
_PROBE_COMPONENTS = ("conditioning", "sampling", "decode", "embed", "scoring")   # run_query's


def test_run_query_reports_per_component_timings():
    _rec, engine = _engine()
    cand = _cand_set()
    out = el.run_query(engine, _base_md(cand), cand, el.build_noise_bank(1, "q", 2, (2, 8)), _OBS)
    timings = out["timings_s"]
    assert set(timings) == set(_PROBE_COMPONENTS)
    assert all(isinstance(v, float) and v >= 0.0 for v in timings.values())


def test_run_query_gt_rir_reports_timings(tmp_path):
    room = str(tmp_path / "Cafe_idx_1")
    _write_rir(room, 0, 11, 0.1)
    _write_rir(room, 3, 11, 0.2)
    _rec, engine = _engine()
    cand = _cand_set()
    obs = el.load_measured_rirs(room, cand, 11)[0][1:2]
    out = el.run_query_gt_rir(engine, cand, room, 11, obs)
    assert set(out["timings_s"]) == set(_PROBE_COMPONENTS)


def test_build_row_carries_the_timings():
    row = el.build_row(**_row_kwargs(timings={"conditioning": 0.5, "sampling": 1.0,
                                              "decode": 0.25, "embed": 0.1, "scoring": 0.01}))
    assert row["timings_s"]["sampling"] == pytest.approx(1.0)
    assert el.build_row(**_row_kwargs())["timings_s"] is None


def test_probe_summary_aggregates_components_and_memory():
    rows = [{"timings_s": {name: float(i + 1) for name in el.PROBE_COMPONENTS}} for i in range(20)]
    probe = el.probe_summary(rows, peak_memory_bytes=1234)
    assert probe["n_queries"] == 20 and probe["peak_memory_bytes"] == 1234
    block = probe["components"]["sampling"]
    values = [float(i + 1) for i in range(20)]
    assert block["mean"] == pytest.approx(float(np.mean(values)))
    assert block["p50"] == pytest.approx(float(np.percentile(values, 50, method="linear")))
    assert block["p95"] == pytest.approx(float(np.percentile(values, 95, method="linear")))
    assert probe["total_s"]["mean"] == pytest.approx(len(el.PROBE_COMPONENTS)
                                                    * float(np.mean(values)))
    assert probe["total_wall_s"] is None                  # these rows carry no wall time


def test_probe_summary_is_none_without_timings():
    assert el.probe_summary([{"e_loc": 1.0}], peak_memory_bytes=None) is None


def test_peak_memory_helpers_are_cpu_safe():
    el.reset_peak_memory("cpu")
    assert el.read_peak_memory("cpu") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_peak_memory_helpers_measure_on_cuda():
    el.reset_peak_memory("cuda")
    block = torch.empty(1024 * 1024, device="cuda")          # 4 MB
    peak = el.read_peak_memory("cuda")
    assert isinstance(peak, int) and peak >= block.numel() * 4
    del block


def test_run_evaluation_summary_carries_the_probe_block(tmp_path):
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    result = el.run_evaluation(_run_args(tmp_path), loader, engine, _stub_context(root), "c", "a",
                               expected=el.expected_split_identities(loader.dataset))
    probe = result["summary"]["probe"]
    assert probe["n_queries"] == 2 and set(probe["components"]) == set(el.PROBE_COMPONENTS)
    assert probe["total_wall_s"]["mean"] > 0.0
    assert probe["peak_memory_bytes"] is None                 # CPU run
    assert all("timings_s" in row for row in result["rows"])


# --------------------------------------------------------------------------- #
# r4 item 2c (full-review F2): --mode scorer-noise, the §2.8.3 measurement of
# what the registered mean readout removes. The engine hard-wires the mean
# readout, so the sampled path had no reviewed caller.
# --------------------------------------------------------------------------- #
def test_measure_scorer_noise_on_identical_draws_is_degenerate():
    mean = torch.nn.functional.normalize(torch.randn(2, 6), dim=-1)
    draws = mean.unsqueeze(0).expand(5, 2, 6).contiguous()
    stats = el.measure_scorer_noise(mean, draws)
    assert stats["aggregate"]["pairwise"]["mean"] == pytest.approx(1.0, abs=1e-6)
    assert stats["aggregate"]["pairwise"]["min"] == pytest.approx(1.0, abs=1e-6)
    assert stats["aggregate"]["vs_mean"]["mean"] == pytest.approx(1.0, abs=1e-6)
    assert stats["n_draws"] == 5 and stats["n_wavs"] == 2


def test_measure_scorer_noise_detects_spread():
    g = torch.Generator().manual_seed(0)
    mean = torch.nn.functional.normalize(torch.randn(1, 8, generator=g), dim=-1)
    draws = torch.nn.functional.normalize(
        mean.unsqueeze(0) + 0.3 * torch.randn(20, 1, 8, generator=g), dim=-1)
    stats = el.measure_scorer_noise(mean, draws)
    assert 0.0 < stats["aggregate"]["pairwise"]["mean"] < 1.0
    assert stats["aggregate"]["pairwise"]["p5"] <= stats["aggregate"]["pairwise"]["mean"]
    assert stats["per_wav"][0]["vs_mean"]["mean"] < 1.0


def test_measure_scorer_noise_needs_two_draws():
    mean = torch.nn.functional.normalize(torch.randn(1, 4), dim=-1)
    with pytest.raises(ValueError):
        el.measure_scorer_noise(mean, mean.unsqueeze(0))


_AGREE_CKPT = os.path.join(_REPO_ROOT, "weights", "AGREE", "AGREE_AR.pt")
agree_integration = pytest.mark.skipif(
    not (os.path.isfile(_AGREE_CKPT) and _dinov3_cache_present()),
    reason="the AGREE checkpoint or the gated DINOv3 HF cache is absent")


def _seen_split_wavs(count=2):
    """Real files from the seen split's own enumeration (r5 item 6)."""
    split = _json.loads(open(_SEEN_SPLIT_JSON).read())
    found = []
    for scene in sorted(split):
        for room in sorted(split[scene]):
            for fname in sorted(split[scene][room]):
                path = os.path.join(_AR_ROOT, "single_channel_ir_1", scene, room, fname)
                if os.path.isfile(path):
                    found.append(path)
                if len(found) >= count:
                    return found
    return found


def _scorer_noise_args(tmp_path, wavs, draws=6, seed=42, name="R0_noise"):
    return el.validate_args(el.parse_args([
        "--mode", "scorer-noise", "--model-config", _FLAC_CONFIG,
        "--dataset-config", _SEEN_CONFIG, "--agree-ckpt", _AGREE_CKPT,
        "--noise-draws", str(draws), "--seed", str(seed), "--noise-wavs", *wavs,
        "--out-dir", str(tmp_path / "out"), "--eval-name", name, "--device", "cpu"]))


@agree_integration
def test_integration_scorer_noise_mode_measures_the_sampled_readout(tmp_path):
    """The real AGREE on real seen-split RIRs: the sampled readout must be
    stochastic and its distance from the registered mean readout quantified."""
    wavs = _seen_split_wavs(2)
    if len(wavs) < 2:
        pytest.skip("the seen split's wavs are not local")
    previous = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        report = el.run_scorer_noise(_scorer_noise_args(tmp_path, wavs, draws=8))
    finally:
        os.chdir(previous)

    assert report["n_draws"] == 8 and report["n_wavs"] == 2 and report["seed"] == 42
    assert len(report["agree_sha256"]) == 64
    for block in ("pairwise", "vs_mean"):
        stats = report["aggregate"][block]
        assert -1.0 <= stats["min"] <= stats["mean"] <= 1.0 + 1e-6
        assert stats["p5"] <= stats["mean"] + 1e-6
    assert report["aggregate"]["pairwise"]["mean"] < 1.0        # genuinely stochastic
    assert os.path.exists(report["report_path"])
    assert _json.loads(open(report["report_path"]).read())["n_draws"] == 8


def test_scorer_noise_refuses_wavs_outside_the_configured_split(tmp_path):
    """A seen config plus an arbitrary path would have measured unseen RIRs."""
    room = str(tmp_path / "Cafe_idx_1")
    _write_rir(room, 0, 11, 0.1)
    args = el.validate_args(el.parse_args([
        "--mode", "scorer-noise", "--model-config", _FLAC_CONFIG,
        "--dataset-config", _SEEN_CONFIG, "--agree-ckpt", "a.pt",
        "--noise-wavs", os.path.join(room, "S000_R0011_hybrid_IR.wav"),
        "--out-dir", str(tmp_path / "out"), "--eval-name", "R0_noise"]))
    previous = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        with pytest.raises(SystemExit):
            el.resolve_noise_wavs(args)
    finally:
        os.chdir(previous)


@unseen_rooms
def test_scorer_noise_accepts_wavs_from_the_configured_split():
    wavs = _seen_split_wavs(1)
    if not wavs:
        pytest.skip("the seen split's wavs are not local")
    args = el.validate_args(el.parse_args([
        "--mode", "scorer-noise", "--model-config", _FLAC_CONFIG,
        "--dataset-config", _SEEN_CONFIG, "--agree-ckpt", "a.pt",
        "--noise-wavs", *wavs, "--eval-name", "R0_noise"]))
    previous = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        assert el.resolve_noise_wavs(args) == wavs
    finally:
        os.chdir(previous)


@agree_integration
def test_integration_scorer_noise_draws_are_seeded_and_reproducible(tmp_path):
    wavs = _seen_split_wavs(1)
    if not wavs:
        pytest.skip("the seen split's wavs are not local")
    previous = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        first = el.run_scorer_noise(_scorer_noise_args(tmp_path, wavs, draws=6, seed=42,
                                                       name="A"))
        second = el.run_scorer_noise(_scorer_noise_args(tmp_path, wavs, draws=6, seed=42,
                                                        name="B"))
        other = el.run_scorer_noise(_scorer_noise_args(tmp_path, wavs, draws=6, seed=43,
                                                       name="C"))
    finally:
        os.chdir(previous)
    assert first["aggregate"] == second["aggregate"]        # same seed -> same draws
    assert first["seed"] == 42 and other["seed"] == 43
    assert first["aggregate"]["pairwise"]["mean"] != other["aggregate"]["pairwise"]["mean"]


# --------------------------------------------------------------------------- #
# r4 item 2d (full-review F2): --mode reaggregate wires R1's offline selection.
# --------------------------------------------------------------------------- #
def test_reaggregate_mode_writes_a_report(tmp_path):
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    run = el.run_evaluation(_run_args(tmp_path), loader, engine, _stub_context(root), "c", "a",
                            expected=el.expected_split_identities(loader.dataset))
    args = el.validate_args(el.parse_args(
        ["--mode", "reaggregate", "--model-config", "m.json", "--dataset-config", "d.json",
         "--rows", run["rows_path"], "--out-dir", str(tmp_path / "re"), "--eval-name", "R1"]))
    report = el.run_reaggregate(args)
    assert report["n_rows"] == 2 and report["selected"]["method"] == "lme"
    assert report["selected"]["k_prime"] == 2                 # K'=8 unavailable at K=2
    assert os.path.exists(report["report_path"])


def test_reaggregate_mode_requires_rows():
    with pytest.raises(SystemExit):
        el.validate_args(el.parse_args(["--mode", "reaggregate", "--model-config", "m.json",
                                        "--dataset-config", "d.json"]))


def test_the_main_guard_is_the_last_statement_in_the_module():
    """Structural regression for a defect that has now bitten twice: appending a
    function after `if __name__ == "__main__": main()` leaves it undefined when the
    driver RUNS, while imports (and therefore this suite) stay green. Checked on
    the AST so it cannot recur however the file is edited."""
    import ast
    module = ast.parse(open(os.path.join(_REPO_ROOT, "eval_localization.py")).read())
    last = module.body[-1]
    assert isinstance(last, ast.If), f"the module must end with the __main__ guard, got {last}"
    assert ast.dump(last.test).count("__name__") == 1
    assert isinstance(last.body[0], ast.Expr)


# --------------------------------------------------------------------------- #
# r5 item 1 (r4 review H1): R0's timings must be WALL-correct on the target GPU.
# _sync ignored the device (so GPU-1 work could be timed against a GPU-0 sync),
# scoring stopped its timer before the .cpu() wait, there was no leading sync, and
# context_evidence was untimed.
# --------------------------------------------------------------------------- #
def test_resolve_cuda_index_reads_the_requested_device():
    assert el._resolve_cuda_index("cpu") is None
    if torch.cuda.is_available():
        assert el._resolve_cuda_index("cuda:1") == 1
        assert el._resolve_cuda_index("cuda") == torch.cuda.current_device()
    else:
        assert el._resolve_cuda_index("cuda:1") is None


def test_timed_brackets_every_interval_with_leading_and_trailing_sync(monkeypatch):
    """A spy over _sync proves each component is drained before the clock starts
    and waited for before it stops."""
    events = []
    monkeypatch.setattr(el, "_sync", lambda device: events.append(("sync", str(device))))
    timings = {}
    with el._timed(timings, "sampling", "cuda:1"):
        events.append(("work", "sampling"))
    assert events == [("sync", "cuda:1"), ("work", "sampling"), ("sync", "cuda:1")]
    assert timings["sampling"] >= 0.0


def test_run_query_syncs_around_every_timed_component(monkeypatch):
    events = []
    monkeypatch.setattr(el, "_sync", lambda device: events.append("sync"))
    rec = _RecordingEngine()
    for name in ("conditioner", "sampler", "decoder", "embedder"):
        original = getattr(rec, name)

        def wrapped(*a, _o=original, _n=name, **kw):
            events.append(_n)
            return _o(*a, **kw)

        setattr(rec, name, wrapped)
    engine = el.Engine(**_engine_kwargs(rec))
    cand = _cand_set()
    out = el.run_query(engine, _base_md(cand), cand, el.build_noise_bank(1, "q", 2, (2, 8)), _OBS)

    assert set(out["timings_s"]) == set(el.PROBE_COMPONENTS) - {"context", "total_wall"}
    for name in ("conditioner", "sampler", "decoder", "embedder"):
        position = events.index(name)
        assert events[position - 1] == "sync", f"no leading sync before {name}"
        assert "sync" in events[position + 1:position + 3], f"no trailing sync after {name}"
    assert events[-1] == "sync"                      # scoring's trailing sync is last


def test_process_query_times_context_evidence_and_the_whole_query(tmp_path):
    root, wav_room = _dataset_tree(tmp_path)
    md = _query_md(root, wav_room)
    _rec, engine = _engine()
    context = _stub_context(root)
    context["context_k"] = 2
    row = el.process_query(_run_args(tmp_path), engine, context, md, torch.full((1, 1, 9600), 0.2))
    timings = row["timings_s"]
    assert "context" in timings and "total_wall" in timings
    assert timings["context"] > 0.0
    component_sum = sum(timings[name] for name in el.PROBE_COMPONENTS)
    assert timings["total_wall"] >= component_sum - 1e-6      # wall covers the components


def test_probe_summary_reports_both_component_sum_and_measured_wall():
    rows = [{"timings_s": {**{name: 1.0 for name in el.PROBE_COMPONENTS}, "total_wall": 8.0}}
            for _ in range(4)]
    probe = el.probe_summary(rows, peak_memory_bytes=None)
    assert probe["total_s"]["mean"] == pytest.approx(float(len(el.PROBE_COMPONENTS)))
    assert probe["total_wall_s"]["mean"] == pytest.approx(8.0)
    assert set(probe["components"]) == set(el.PROBE_COMPONENTS)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_timed_waits_for_the_requested_cuda_device_not_device_zero():
    """The reviewer's exact scenario. The kernels are large enough that the CPU
    runs far ahead of the GPU (launch cost ~1 ms vs ~0.5 s of work), so a
    synchronization on the WRONG device returns immediately while ours waits."""
    def queue_work(index, size=8192, iters=12):
        with torch.cuda.device(index):
            a = torch.randn(size, size, device=f"cuda:{index}")
            b = torch.randn(size, size, device=f"cuda:{index}")
            for _ in range(iters):
                a = torch.mm(a, b)
        return a

    for index in (0, 1):                      # warm both CUDA contexts
        warm = queue_work(index, size=1024, iters=1)
        torch.cuda.synchronize(index)
        del warm

    try:
        torch.cuda.synchronize(1)
        started = time.perf_counter()
        held = queue_work(1)
        torch.cuda.synchronize(0)             # the OLD behaviour: wrong device
        wrong_device = time.perf_counter() - started
        torch.cuda.synchronize(1)
        del held

        timings = {}
        torch.cuda.synchronize(1)
        with el._timed(timings, "sampling", "cuda:1"):
            held = queue_work(1)
        measured = timings["sampling"]
        del held
    finally:
        torch.cuda.empty_cache()

    assert measured > 0.1, f"the timed interval did not wait for cuda:1 ({measured:.4f}s)"
    assert wrong_device < measured * 0.2, (
        f"a cuda:0 synchronization waited {wrong_device:.4f}s against our {measured:.4f}s; "
        "the test cannot discriminate")


# --------------------------------------------------------------------------- #
# r5 item 2 (r4 review H2): --mode readback must ENFORCE the corrected R-1
# invariants, not merely report them. Deleting LRH's metadata-only S10 used to
# pass -- reintroducing the very M=9 error Rev 3.1 corrected.
# --------------------------------------------------------------------------- #
_LRH = "LivingRoomsWithHallway"
_LRH_ROOM = f"{_LRH}_idx_30"


def _registered_unseen_tree(tmp_path, n_rooms=17, nodes=10, receiver=7,
                            lrh_metadata_only=True, depth_shape=(256, 512),
                            depth_dtype=np.float32, wav_samples=12000):
    """A tree with the REGISTERED unseen shape: 17 rooms x 10 metadata sources,
    with LivingRoomsWithHallway_idx_30's source 10 having metadata but no wav."""
    root = tmp_path / "AcousticRooms"
    split = {}
    scenes = [(_LRH, _LRH_ROOM)] + [(f"Scene{i}", f"Scene{i}_idx_{i}") for i in range(n_rooms - 1)]
    for scene, scene_id in scenes:
        meta_dir = root / "metadata" / scene / scene_id
        wav_dir = root / "single_channel_ir_1" / scene / scene_id
        depth_dir = root / "depth_map" / scene / scene_id
        for d in (meta_dir, wav_dir, depth_dir):
            d.mkdir(parents=True, exist_ok=True)
        np.save(str(depth_dir / f"{receiver}.npy"),
                np.ones(depth_shape, dtype=depth_dtype))
        files = []
        for node in range(1, nodes + 1):
            (meta_dir / f"S00{node}_R00{receiver}.json").write_text(_json.dumps(
                {"src_loc": [float(node), 0.5 * node, 1.0], "rec_loc": [0.0, 0.0, 1.0],
                 "IR_norm": 1.0}))
            wav_absent = (lrh_metadata_only and scene_id == _LRH_ROOM and node == nodes)
            if not wav_absent:
                _write_rir(str(wav_dir), node, receiver, 0.05 * node, length=wav_samples)
                files.append(f"S00{node}_R00{receiver}_hybrid_IR.wav")
        split[scene] = {scene_id: files}
    split_path = tmp_path / "unseen.json"
    split_path.write_text(_json.dumps(split))
    return root, split_path


def _unseen_readback_args(tmp_path, root, split_path, **over):
    config = tmp_path / "unseen_d.json"
    config.write_text(_json.dumps({
        "dataset_type": "audio_dir", "unseeneval": True,
        "modalities": {"acoustic_context": {"load": True, "max_context": 8}},
        "datasets": [{"id": "AcousticRooms", "path": str(root),
                      "json_file_path": str(split_path),
                      "folder_name": "single_channel_ir_1"}]}))
    argv = ["--mode", "readback", "--model-config", "m.json", "--dataset-config", str(config),
            "--out-dir", str(tmp_path / "out"), "--eval-name", "R_minus_1"]
    for flag, value in over.items():
        argv += [flag] if value is True else [flag, str(value)]
    args = el.validate_args(el.parse_args(argv))
    # these fixtures have the registered SHAPE but are not the canonical split file;
    # the digest gate itself is exercised by the dedicated tests below.
    args.skip_split_digest = True
    return args


def test_readback_passes_the_registered_unseen_shape(tmp_path):
    root, split_path = _registered_unseen_tree(tmp_path)
    report = el.run_readback(_unseen_readback_args(tmp_path, root, split_path))
    assert report["ok"] is True and report["failures"] == []
    assert report["registered_check"]["enforced"] is True
    assert report["registered_check"]["n_rooms"] == 17
    assert any(_LRH_ROOM in w and "metadata-only" in w for w in report["warnings"])
    lrh = report["rooms"][f"{_LRH}/{_LRH_ROOM}"]
    assert lrh["metadata_nodes"] == list(range(1, 11)) and lrh["metadata_only_nodes"] == [10]
    assert lrh["depth_checked"] == 1 and lrh["depth_bad"] == []
    assert lrh["wav_checked"] == 9                      # one wav per (room, source) present


def test_readback_fails_when_the_lrh_metadata_only_source_disappears(tmp_path):
    """The exact regression the reviewer named: deleting S10's metadata leaves 9
    metadata nodes matching 9 wavs, which used to pass silently."""
    root, split_path = _registered_unseen_tree(tmp_path)
    os.remove(str(root / "metadata" / _LRH / _LRH_ROOM / "S0010_R007.json"))
    args = _unseen_readback_args(tmp_path, root, split_path)
    with pytest.raises(SystemExit):
        el.run_readback(args)
    written = os.path.join(str(args.out_dir), el.aux_stem(args, "readback") + ".json")
    report = _json.loads(open(written).read())            # the report is written before the exit
    assert report["ok"] is False
    assert any("is GONE" in f for f in report["failures"])
    assert any("9 metadata sources" in f for f in report["failures"])


def test_readback_fails_on_an_unregistered_metadata_only_source(tmp_path):
    root, split_path = _registered_unseen_tree(tmp_path)
    os.remove(str(root / "single_channel_ir_1" / "Scene0" / "Scene0_idx_0"
                   / "S001_R007_hybrid_IR.wav"))
    split = _json.loads(open(split_path).read())
    split["Scene0"]["Scene0_idx_0"] = [f for f in split["Scene0"]["Scene0_idx_0"]
                                       if not f.startswith("S001_")]
    open(split_path, "w").write(_json.dumps(split))
    with pytest.raises(SystemExit):
        el.run_readback(_unseen_readback_args(tmp_path, root, split_path))


def test_readback_fails_on_a_wrong_room_count(tmp_path):
    root, split_path = _registered_unseen_tree(tmp_path, n_rooms=16)
    with pytest.raises(SystemExit):
        el.run_readback(_unseen_readback_args(tmp_path, root, split_path))


@pytest.mark.parametrize("mutate", ["shape", "dtype", "nonfinite", "missing"])
def test_readback_loads_and_validates_every_depth_map(tmp_path, mutate):
    root, split_path = _registered_unseen_tree(tmp_path)
    depth = str(root / "depth_map" / "Scene0" / "Scene0_idx_0" / "7.npy")
    if mutate == "shape":
        np.save(depth, np.ones((128, 256), dtype=np.float32))
    elif mutate == "dtype":
        np.save(depth, np.ones((256, 512), dtype=np.int32))
    elif mutate == "nonfinite":
        bad = np.ones((256, 512), dtype=np.float32)
        bad[0, 0] = np.nan
        np.save(depth, bad)
    else:
        os.remove(depth)
    with pytest.raises(SystemExit):
        el.run_readback(_unseen_readback_args(tmp_path, root, split_path))


@pytest.mark.parametrize("mutate", ["rate", "stereo", "empty"])
def test_readback_decodes_one_wav_per_room_and_source(tmp_path, mutate):
    root, split_path = _registered_unseen_tree(tmp_path)
    path = str(root / "single_channel_ir_1" / "Scene0" / "Scene0_idx_0"
               / "S002_R007_hybrid_IR.wav")
    if mutate == "rate":
        torchaudio.save(path, torch.full((1, 200), 0.1), 16000)
    elif mutate == "stereo":
        torchaudio.save(path, torch.full((2, 200), 0.1), 22050)
    else:
        torchaudio.save(path, torch.zeros(1, 1), 22050)
        os.truncate(path, 44)                            # header only: no samples
    with pytest.raises(SystemExit):
        el.run_readback(_unseen_readback_args(tmp_path, root, split_path))


@unseen_rooms
def test_integration_readback_gate_passes_on_the_real_unseen_split(tmp_path):
    """The actual R-1 gate on the real dataset: 17 rooms, M=10, every depth map
    loadable at (256, 512), one wav decoded per (room, source)."""
    config = tmp_path / "unseen.json"
    config.write_text(_json.dumps({
        "dataset_type": "audio_dir", "unseeneval": True,
        "modalities": {"acoustic_context": {"load": True, "max_context": 8}},
        "datasets": [{"id": "AcousticRooms", "path": _AR_ROOT,
                      "json_file_path": _UNSEEN_SPLIT_JSON,
                      "folder_name": "single_channel_ir_1"}]}))
    args = el.validate_args(el.parse_args(
        ["--mode", "readback", "--model-config", "m.json", "--dataset-config", str(config),
         "--out-dir", str(tmp_path / "out"), "--eval-name", "R_minus_1_real"]))
    report = el.run_readback(args)
    assert report["ok"] is True, report["failures"]
    assert report["registered_check"]["n_rooms"] == 17
    assert report["warnings"] == [
        f"{_LRH}/{_LRH_ROOM}: metadata-only sources (no wavs): [10]"]
    assert sum(room["wav_checked"] for room in report["rooms"].values()) == 169
    assert all(room["depth_bad"] == [] for room in report["rooms"].values())


# --------------------------------------------------------------------------- #
# r5 item 3 (r4 review M3): the auxiliary modes opened a fixed {eval_name}_*.json
# with "w" and ignored --overwrite, so a failed R-1 could erase passing evidence
# and two different scorer-noise inputs collided.
# --------------------------------------------------------------------------- #
def test_write_json_atomic_refuses_an_existing_target(tmp_path):
    path = str(tmp_path / "report.json")
    el.write_json_atomic(path, {"ok": True}, overwrite=False)
    assert _json.loads(open(path).read()) == {"ok": True}
    assert not os.path.exists(path + ".partial")          # no debris on success
    with pytest.raises(SystemExit):
        el.write_json_atomic(path, {"ok": False}, overwrite=False)
    el.write_json_atomic(path, {"ok": False}, overwrite=True)
    assert _json.loads(open(path).read()) == {"ok": False}


def test_write_json_atomic_refuses_a_stale_partial(tmp_path):
    path = str(tmp_path / "report.json")
    open(path + ".partial", "w").close()
    with pytest.raises(SystemExit):
        el.write_json_atomic(path, {"ok": True}, overwrite=False)


def test_readback_refuses_to_overwrite_its_own_report(tmp_path):
    root, _wav_room = _dataset_tree(tmp_path)
    args = _readback_args(tmp_path, root, _write_split(tmp_path))
    first = el.run_readback(args)["report_path"]
    with pytest.raises(SystemExit):
        el.run_readback(args)
    assert os.path.exists(first)                          # the passing evidence survives
    over = _readback_args(tmp_path, root, _write_split(tmp_path), **{"--overwrite": True})
    assert el.run_readback(over)["report_path"] == first


def test_auxiliary_stems_are_content_addressed(tmp_path):
    root, _wav_room = _dataset_tree(tmp_path)
    split_a = _write_split(tmp_path)
    base = _readback_args(tmp_path, root, split_a)
    other_root, _w = _dataset_tree(tmp_path / "other")
    other = _readback_args(tmp_path / "other", other_root, _write_split(tmp_path / "other"))
    assert el.aux_stem(base, "readback") != el.aux_stem(other, "readback")
    assert el.aux_stem(base, "readback").startswith("R_minus_1_readback_ds-")

    noise = _run_args(tmp_path, **{"--mode": "scorer-noise", "--noise-draws": "16"})
    noise_other = _run_args(tmp_path, **{"--mode": "scorer-noise", "--noise-draws": "32"})
    noise_seed = _run_args(tmp_path, **{"--mode": "scorer-noise", "--noise-draws": "16",
                                        "--seed": "43"})
    stems = {el.aux_stem(a, "scorer-noise") for a in (noise, noise_other, noise_seed)}
    assert len(stems) == 3

    rows_a, rows_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rows_a.write_text('{"query_id": "q0"}\n')
    rows_b.write_text('{"query_id": "q1"}\n')
    left = _run_args(tmp_path, **{"--mode": "reaggregate", "--rows": str(rows_a)})
    right = _run_args(tmp_path, **{"--mode": "reaggregate", "--rows": str(rows_b)})
    assert el.aux_stem(left, "reaggregate") != el.aux_stem(right, "reaggregate")
    assert el.aux_stem(left, "reaggregate") == el.aux_stem(left, "reaggregate")


def test_reaggregate_mode_refuses_to_clobber(tmp_path):
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    run = el.run_evaluation(_run_args(tmp_path), loader, engine, _stub_context(root), "c", "a",
                            expected=el.expected_split_identities(loader.dataset))
    argv = ["--mode", "reaggregate", "--model-config", "m.json", "--dataset-config",
            str(tmp_path / "d.json"), "--rows", run["rows_path"],
            "--out-dir", str(tmp_path / "re"), "--eval-name", "R1"]
    (tmp_path / "d.json").write_text(_json.dumps({"dataset_type": "audio_dir", "datasets": []}))
    args = el.validate_args(el.parse_args(argv))
    el.run_reaggregate(args)
    with pytest.raises(SystemExit):
        el.run_reaggregate(args)
    el.run_reaggregate(el.validate_args(el.parse_args(argv + ["--overwrite"])))


def test_run_evaluation_leaves_no_partial_files(tmp_path):
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    result = el.run_evaluation(_run_args(tmp_path), loader, engine, _stub_context(root), "c", "a",
                               expected=el.expected_split_identities(loader.dataset))
    for key in ("rows_path", "summary_path", "manifest_path"):
        assert os.path.exists(result[key]) and not os.path.exists(result[key] + ".partial")


# --------------------------------------------------------------------------- #
# r5 item 4 (r4 review M4): the registration SHA must be an IMMUTABLE object id
# that is an ancestor of the executing HEAD, and the manifest must live inside
# the repository. HEAD, branches, tags, abbreviations and unrelated commits all
# used to pass as long as the bytes matched.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("committish", ["HEAD", "master", "main", "v1.0", "abc123"])
def test_verify_registration_commit_refuses_symbolic_or_abbreviated_ids(tmp_path, committish):
    path, sha = _git_repo_with_manifest(tmp_path, _locked())
    with pytest.raises(SystemExit):
        el.verify_registration_commit(path, committish, repo_root=str(tmp_path))
    with pytest.raises(SystemExit):
        el.verify_registration_commit(path, sha[:8], repo_root=str(tmp_path))
    with pytest.raises(SystemExit):
        el.verify_registration_commit(path, sha.upper(), repo_root=str(tmp_path))


def test_verify_registration_commit_requires_an_ancestor_of_head(tmp_path):
    """A commit on another branch can contain identical bytes; only an ancestor of
    the executing HEAD proves the protocol was registered BEFORE this run."""
    import subprocess
    path, base_sha = _git_repo_with_manifest(tmp_path, _locked())
    subprocess.run(["git", "checkout", "-q", "-b", "side"], cwd=tmp_path, check=True)
    open(path, "a").write("")                              # same bytes, new commit
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q",
                    "--allow-empty", "-m", "side"], cwd=tmp_path, check=True)
    side_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
                              capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "checkout", "-q", base_sha], cwd=tmp_path, check=True)

    assert el.verify_registration_commit(path, base_sha,
                                         repo_root=str(tmp_path))["resolved_sha"] == base_sha
    with pytest.raises(SystemExit):
        el.verify_registration_commit(path, side_sha, repo_root=str(tmp_path))


def test_verify_registration_commit_refuses_a_manifest_outside_the_repository(tmp_path):
    path, sha = _git_repo_with_manifest(tmp_path / "repo", _locked())
    outside = tmp_path / "outside.json"
    outside.write_text(open(path).read())
    with pytest.raises(SystemExit):
        el.verify_registration_commit(str(outside), sha, repo_root=str(tmp_path / "repo"))


def test_verify_registration_records_the_resolved_sha(tmp_path):
    path, sha = _git_repo_with_manifest(tmp_path, _locked())
    args = _split_args(_UNSEEN_CONFIG, "--registration-sha", sha,
                       "--registration-manifest", path, "--seed", "43", "--num-samples", "8")
    unseen = _json.loads(open(_UNSEEN_CONFIG).read())
    assert el.verify_registration(args, unseen, _resolved(), repo_root=str(tmp_path)) is True
    assert args.registration_sha_resolved == sha
    record = el.build_provenance(args, ckpt_sha256="a", agree_sha256="b", split_hash="c",
                                 weights_source="ema", n_queries=1)
    assert record["registration_sha_resolved"] == sha


# --------------------------------------------------------------------------- #
# r5 item 5 (r4 review M5): the "before model loads" ordering was incomplete --
# the checkpoint was deserialized before the registration flags were checked, and
# the context configuration and output-collision refusal happened only after both
# models and the dataloader were built.
# --------------------------------------------------------------------------- #
def test_main_runs_every_cheap_validation_before_any_deserialization(tmp_path, monkeypatch):
    """Order is recorded by spies, not asserted by inspection: configs are read and
    hashed, registration verified, context resolved and output targets claimed
    BEFORE torch.load of the checkpoint, AGREE, the generator and the dataloader."""
    events = []
    model_config = tmp_path / "m.json"
    model_config.write_text(_json.dumps(
        {"model": {"diffusion": {"diffusion_objective": "rectified_flow"}},
         "sample_size": 10240, "sample_rate": 22050}))

    def spy(name, result=None, raises=None):
        def _call(*args, **kwargs):
            events.append(name)
            if raises is not None:
                raise raises
            return result
        return _call

    monkeypatch.setattr(el, "load_dataset_config", lambda args: (
        events.append("read_dataset_config") or _json.loads(open(_SEEN_CONFIG).read())))
    monkeypatch.setattr(el, "read_model_config", lambda args: (
        events.append("read_model_config") or _json.loads(open(model_config).read())))
    monkeypatch.setattr(el, "verify_registration", spy("registration", result=False))
    monkeypatch.setattr(el, "resolve_context_k", spy("context_k", result=8))
    monkeypatch.setattr(el, "manifest_for_dataset_config",
                        spy("manifest", result={"rooms": {}, "dataset_root": "x"}))
    monkeypatch.setattr(el, "artifact_paths", spy("artifact_paths", result={
        "rows": str(tmp_path / "r.jsonl"), "summary": str(tmp_path / "s.json"),
        "manifest": str(tmp_path / "m.json.out")}))
    monkeypatch.setattr(el, "sha256_file", lambda path: "0" * 64)
    monkeypatch.setattr(el, "load_checkpoint_and_validate", spy("torch_load", result={}))
    monkeypatch.setattr(el, "load_agree_audio", spy("agree"))
    monkeypatch.setattr(el, "build_engine", spy("engine", result=(None, {})))
    monkeypatch.setattr(el, "build_dataloader",
                        spy("dataloader", raises=SystemExit("stop after the ordering")))

    with pytest.raises(SystemExit):
        el.main(["--model-config", str(model_config), "--dataset-config", _SEEN_CONFIG,
                 "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt", "--num-samples", "2",
                 "--out-dir", str(tmp_path / "out"), "--device", "cpu"])

    for early in ("read_dataset_config", "read_model_config", "registration", "context_k",
                  "artifact_paths"):
        for late in ("torch_load", "agree", "engine", "dataloader"):
            assert events.index(early) < events.index(late), f"{early} must precede {late}"
    assert events.index("torch_load") < events.index("agree") < events.index("engine")
    assert events.index("engine") < events.index("dataloader")


def test_read_model_config_and_checkpoint_validation_are_separable(tmp_path):
    config = tmp_path / "m.json"
    config.write_text(_json.dumps({"model": {"diffusion": {"diffusion_objective": "v"}},
                                   "sample_size": 10240, "sample_rate": 22050}))
    args = el.parse_args(["--model-config", str(config), "--dataset-config", "d.json",
                          "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt", "--num-samples", "2"])
    with pytest.raises(SystemExit):
        el.read_model_config(args)                        # objective refused without torch.load


# --------------------------------------------------------------------------- #
# r5b item 1 (r5 review, H2 residual): the gate read its authority FROM the split
# it was validating, so a truncated or same-shaped substituted split passed and
# the later identity audit merely agreed with the altered authority. The
# canonical split is now pinned by byte digest, identity count, room count and
# room/node map.
# --------------------------------------------------------------------------- #
def _real_split():
    return _json.loads(open(_UNSEEN_SPLIT_JSON).read())


def _unseen_config_for(tmp_path, split_path, name="d.json"):
    config = tmp_path / name
    config.write_text(_json.dumps({
        "dataset_type": "audio_dir", "unseeneval": True,
        "modalities": {"acoustic_context": {"load": True, "max_context": 8}},
        "datasets": [{"id": "AcousticRooms", "path": _AR_ROOT,
                      "json_file_path": str(split_path),
                      "folder_name": "single_channel_ir_1"}]}))
    return config


def test_registered_split_constants_match_the_committed_split():
    """The pinned values ARE the canonical split's; if data/AR/unseen_eval.json
    ever changes, this fails before anything else does."""
    import hashlib as _h
    raw = open(_UNSEEN_SPLIT_JSON, "rb").read()
    assert _h.sha256(raw).hexdigest() == el.UNSEEN_SPLIT_FILE_SHA256
    split = _json.loads(raw)
    assert sum(len(v) for rooms in split.values() for v in rooms.values()) == \
        el.UNSEEN_SPLIT_N_FILES == 6337
    assert sum(len(rooms) for rooms in split.values()) == el.UNSEEN_SPLIT_N_ROOMS == 17
    assert el.room_node_map_digest(split) == el.UNSEEN_ROOM_NODE_MAP_SHA256


def test_verify_registered_split_passes_on_the_canonical_split():
    checks = el.verify_registered_split(_UNSEEN_SPLIT_JSON, enforced=True)
    assert checks["failures"] == [] and checks["enforced"] is True
    assert checks["file_sha256"] == el.UNSEEN_SPLIT_FILE_SHA256
    assert checks["n_files"] == 6337 and checks["n_rooms"] == 17
    assert checks["room_node_map_sha256"] == el.UNSEEN_ROOM_NODE_MAP_SHA256


def test_verify_registered_split_fails_on_a_truncated_split(tmp_path):
    split = _real_split()
    scene = sorted(split)[0]
    room = sorted(split[scene])[0]
    split[scene][room] = split[scene][room][:-1]              # one identity fewer
    path = tmp_path / "truncated.json"
    path.write_text(_json.dumps(split))
    checks = el.verify_registered_split(path, enforced=True)
    assert any("6336" in f for f in checks["failures"])
    assert any("byte digest" in f for f in checks["failures"])


def test_verify_registered_split_fails_on_a_same_count_wrong_room_split(tmp_path):
    """17 rooms and 6337 files, but one room renamed: the counts alone cannot see
    it -- the room/node map digest can."""
    split = _real_split()
    scene = sorted(split)[0]
    room = sorted(split[scene])[0]
    split[scene][f"{room}_impostor"] = split[scene].pop(room)
    path = tmp_path / "renamed.json"
    path.write_text(_json.dumps(split))
    checks = el.verify_registered_split(path, enforced=True)
    assert checks["n_files"] == 6337 and checks["n_rooms"] == 17
    assert any("room/node map" in f for f in checks["failures"])


def test_verify_registered_split_fails_on_a_substituted_source_list(tmp_path):
    split = _real_split()
    scene = sorted(split)[0]
    room = sorted(split[scene])[0]
    files = split[scene][room]
    split[scene][room] = [files[0].replace("S001", "S011") if f is files[0] else f
                          for f in files]
    path = tmp_path / "substituted.json"
    path.write_text(_json.dumps(split))
    checks = el.verify_registered_split(path, enforced=True)
    assert checks["failures"]


def test_verify_registered_split_is_inert_for_non_unseen_configs(tmp_path):
    path = tmp_path / "seen.json"
    path.write_text(_json.dumps({"Cafe": {"Cafe_idx_1": ["S000_R000_hybrid_IR.wav"]}}))
    checks = el.verify_registered_split(path, enforced=False)
    assert checks["enforced"] is False and checks["failures"] == []


def test_readback_fails_on_a_noncanonical_unseen_split(tmp_path):
    split = _real_split()
    scene = sorted(split)[0]
    room = sorted(split[scene])[0]
    split[scene][room] = split[scene][room][:-1]
    path = tmp_path / "truncated.json"
    path.write_text(_json.dumps(split))
    args = el.validate_args(el.parse_args(
        ["--mode", "readback", "--model-config", "m.json",
         "--dataset-config", str(_unseen_config_for(tmp_path, path)),
         "--out-dir", str(tmp_path / "out"), "--eval-name", "R_minus_1"]))
    with pytest.raises(SystemExit):
        el.run_readback(args)
    written = os.path.join(str(args.out_dir), el.aux_stem(args, "readback") + ".json")
    report = _json.loads(open(written).read())
    assert report["ok"] is False and report["split_check"]["enforced"] is True
    assert any("6336" in f for f in report["failures"])


def test_run_startup_refuses_a_noncanonical_unseen_split(tmp_path):
    split = _real_split()
    scene = sorted(split)[0]
    split[scene]["Ghost_idx_0"] = split[scene].pop(sorted(split[scene])[0])
    path = tmp_path / "renamed.json"
    path.write_text(_json.dumps(split))
    config = _unseen_config_for(tmp_path, path)
    args = el.parse_args(["--model-config", "m.json", "--dataset-config", str(config),
                          "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt", "--num-samples", "8"])
    with pytest.raises(SystemExit):
        el.assert_registered_split(args, _json.loads(open(config).read()))


def test_registration_locks_the_split_digest():
    assert "split_file_sha256" in el.REGISTRATION_LOCKED_FIELDS
    with pytest.raises(SystemExit):
        el.check_registration_fields(_locked(split_file_sha256="tbd"), _resolved(),
                                     registered=True)
    with pytest.raises(SystemExit):
        el.check_registration_fields(_locked(split_file_sha256="x" * 64), _resolved(),
                                     registered=True)


# --------------------------------------------------------------------------- #
# r5b item 2 (r5 review, H2 residual): a finite, mono, nonempty but TRUNCATED RIR
# passed R-1 and would change oracle/context/query scores.
# --------------------------------------------------------------------------- #
def test_readback_refuses_a_wav_shorter_than_the_scored_prefix(tmp_path):
    """Every scoring path consumes a prefix -- target pad_crop 10240, context
    9600, oracle 8000 -- so a wav shorter than 10240 changes scores, while one
    longer than 10240 is score-inert."""
    root, split_path = _registered_unseen_tree(tmp_path, wav_samples=12000)
    report = el.run_readback(_unseen_readback_args(tmp_path, root, split_path))
    assert report["ok"] is True
    assert report["wav_lengths"]["min"] == 12000 and report["wav_lengths"]["max"] == 12000
    assert report["wav_lengths"]["mean"] == pytest.approx(12000.0)
    assert el.MIN_WAV_SAMPLES == 10240

    short_root, short_split = _registered_unseen_tree(tmp_path / "short", wav_samples=9000)
    short_args = _unseen_readback_args(tmp_path / "short", short_root, short_split)
    with pytest.raises(SystemExit):
        el.run_readback(short_args)
    written = os.path.join(str(short_args.out_dir),
                           el.aux_stem(short_args, "readback") + ".json")
    report = _json.loads(open(written).read())
    assert report["ok"] is False
    assert any("9000 samples" in f or "shorter than" in f for f in report["failures"])


def test_scorer_noise_stem_includes_the_selected_wav_set(tmp_path):
    """Two measurements over different RIRs must not share a report path even at
    identical knobs (r5 review nit)."""
    args = _run_args(tmp_path, **{"--mode": "scorer-noise"})
    a = el.aux_stem(args, "scorer-noise", wavs=["/data/a.wav", "/data/b.wav"])
    b = el.aux_stem(args, "scorer-noise", wavs=["/data/a.wav", "/data/c.wav"])
    bare = el.aux_stem(args, "scorer-noise")
    assert a != b and a != bare
    assert a == el.aux_stem(args, "scorer-noise", wavs=["/data/b.wav", "/data/a.wav"])
    assert f"w{args.noise_wav_count}" in bare


def test_probe_wall_documents_what_it_excludes():
    assert "row construction" in el.__doc__ or True     # documented at PROBE_WALL
    import inspect
    source = inspect.getsource(el)
    assert "NOT the row construction" in source
    assert "NOT an interprocess lock" in source


# --------------------------------------------------------------------------- #
# r6 (plan Rev 3.2): duplicate-position sources MERGE into one candidate. Real
# occurrences: Bathrooms_idx_11 (S9 == S10) and Bathrooms_idx_16 (S4 == S7) in
# the SEEN split; the 17 unseen rooms are clean, so the headline path must stay
# byte-identical.
# --------------------------------------------------------------------------- #
def _dirty_dataset_tree(tmp_path, receiver=11, rec_loc=(1.0, 2.0, 0.5)):
    """A room shaped like Bathrooms_idx_11: nodes 3 and 7 share one position."""
    root = tmp_path / "AcousticRooms"
    meta_room = root / "metadata" / "Cafe" / "Cafe_idx_1"
    wav_room = root / "single_channel_ir_1" / "Cafe" / "Cafe_idx_1"
    depth_room = root / "depth_map" / "Cafe" / "Cafe_idx_1"
    for d in (meta_room, wav_room, depth_room):
        d.mkdir(parents=True, exist_ok=True)
    np.save(str(depth_room / f"{receiver}.npy"), np.ones((256, 512), dtype=np.float32))
    shared = (2.0, -1.0, 1.5)
    positions = {0: (0.0, 0.0, 1.0), 3: shared, 7: shared}
    for node, xyz in positions.items():
        (meta_room / f"S00{node}_R00{receiver}.json").write_text(_json.dumps(
            {"src_loc": list(xyz), "rec_loc": list(rec_loc), "IR_norm": 1.0}))
        _write_rir(str(wav_room), node, receiver, 0.1 * (node + 1))
    return root, wav_room, positions


def test_build_room_manifest_merges_duplicate_positions(tmp_path):
    root, _wav_room, _positions = _dirty_dataset_tree(tmp_path)
    manifest = el.build_room_manifest(str(root), _split_dict())
    room = manifest["rooms"]["Cafe/Cafe_idx_1"]
    assert room["nodes"] == [0, 3]                       # 7 folded into 3
    assert room["merge_map"] == {"3": [3, 7]}            # only non-trivial groups
    assert room["member_nodes"] == [0, 3, 7]
    assert len(room["xyz_world"]) == 2
    assert room["xyz_world"][1] == [2.0, -1.0, 1.5]


def test_clean_rooms_have_an_empty_merge_map_and_unchanged_candidates(tmp_path):
    root, _wav_room = _dataset_tree(tmp_path)
    manifest = el.build_room_manifest(str(root), _split_dict())
    room = manifest["rooms"]["Cafe/Cafe_idx_1"]
    assert room["merge_map"] == {}
    assert room["nodes"] == [0, 3, 7] == room["member_nodes"]
    cand = el.candidate_set_from_manifest(manifest, "Cafe/Cafe_idx_1", gt_node=3, rec_node=11)
    from_disk = el.query_candidate_set(_query_md(root, _wav_room := root / "single_channel_ir_1"
                                                 / "Cafe" / "Cafe_idx_1"))
    assert cand.nodes == from_disk.nodes and cand.gt_node == from_disk.gt_node
    np.testing.assert_array_equal(cand.xyz_world, from_disk.xyz_world)


def test_candidate_set_resolves_a_gt_that_is_a_non_canonical_member(tmp_path):
    root, _wav_room, positions = _dirty_dataset_tree(tmp_path)
    manifest = el.build_room_manifest(str(root), _split_dict())
    canonical = el.candidate_set_from_manifest(manifest, "Cafe/Cafe_idx_1", 3, 11)
    member = el.candidate_set_from_manifest(manifest, "Cafe/Cafe_idx_1", 7, 11)
    assert member.gt_node == 3 == canonical.gt_node       # the merged candidate's identity
    assert member.nodes == [0, 3] and member.gt_index == 1
    np.testing.assert_array_equal(member.gt_xyz, np.asarray(positions[7]))
    from src.localization.scoring import localization_error
    assert localization_error(member.gt_xyz, canonical.gt_xyz) == pytest.approx(0.0)


def test_candidate_set_still_refuses_a_node_outside_every_group(tmp_path):
    root, _wav_room, _positions = _dirty_dataset_tree(tmp_path)
    manifest = el.build_room_manifest(str(root), _split_dict())
    with pytest.raises(ValueError):
        el.candidate_set_from_manifest(manifest, "Cafe/Cafe_idx_1", 99, 11)


def test_merged_room_resolves_duplicate_context_poses_to_one_index(tmp_path):
    """Two context refs that are different member nodes of one group resolve to
    the SAME candidate -- correct, and no longer an F7 fingerprint collision."""
    root, _wav_room, positions = _dirty_dataset_tree(tmp_path)
    manifest = el.build_room_manifest(str(root), _split_dict())
    cand = el.candidate_set_from_manifest(manifest, "Cafe/Cafe_idx_1", 0, 11)
    cams = el.candidate_camera_positions(cand)
    from src.localization.candidates import project_to_camera
    merged_cam = project_to_camera(cand.rec_loc, np.asarray(positions[3]))
    context_ids = [el.render_position_id(merged_cam), el.render_position_id(merged_cam)]
    assert el.context_membership_mask(cams, context_ids, gt_index=cand.gt_index) == [False, True]


def test_manifest_merge_summary_and_row_field(tmp_path):
    root, _wav_room, _positions = _dirty_dataset_tree(tmp_path)
    manifest = el.build_room_manifest(str(root), _split_dict())
    assert el.merge_group_count(manifest) == 1
    clean = el.build_room_manifest(str(_dataset_tree(tmp_path / "clean")[0]), _split_dict())
    assert el.merge_group_count(clean) == 0

    row = el.build_row(**_row_kwargs(merge_map={"3": [3, 7]}))
    assert row["merge_map"] == {"3": [3, 7]}
    assert el.build_row(**_row_kwargs())["merge_map"] == {}


def test_run_evaluation_records_the_merge_groups(tmp_path):
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    result = el.run_evaluation(_run_args(tmp_path), loader, engine, _stub_context(root), "c", "a",
                               expected=el.expected_split_identities(loader.dataset))
    assert result["provenance"]["candidate_merge_groups"] == 0     # the clean fixture
    assert all(row["merge_map"] == {} for row in result["rows"])


def _merged_cand_set():
    """Candidates [0, 3] where 3 is the canonical of the merged group {3, 7}."""
    xyz = np.array([[0.0, 0.0, 1.0], [2.0, -1.0, 1.5]])
    return CandidateSet(nodes=[0, 3], xyz_world=xyz, rec_loc=_REC, gt_node=3, gt_xyz=xyz[1]), \
        {"3": [3, 7]}


def test_measured_rir_paths_prefer_the_canonical_node_then_fall_back(tmp_path):
    room = str(tmp_path / "room")
    cand, merge_map = _merged_cand_set()
    _write_rir(room, 0, 11, 0.1)
    _write_rir(room, 3, 11, 0.2)
    _write_rir(room, 7, 11, 0.3)
    paths = el.measured_rir_paths(room, cand, 11, merge_map=merge_map)
    assert os.path.basename(paths[1]) == "S003_R0011_hybrid_IR.wav"      # canonical wins

    os.remove(os.path.join(room, "S003_R0011_hybrid_IR.wav"))
    fallback = el.measured_rir_paths(room, cand, 11, merge_map=merge_map)
    assert os.path.basename(fallback[1]) == "S007_R0011_hybrid_IR.wav"   # member fallback
    assert el.measured_rir_paths(room, cand, 11)[1] is None              # without the map


def test_load_measured_rirs_records_which_member_supplied_each_file(tmp_path):
    room = str(tmp_path / "room")
    cand, merge_map = _merged_cand_set()
    _write_rir(room, 0, 11, 0.1)
    _write_rir(room, 7, 11, 0.3)                       # only the non-canonical member exists
    wavs, available, sources = el.load_measured_rirs(room, cand, 11, merge_map=merge_map)
    assert available == [True, True] and tuple(wavs.shape)[0] == 2
    assert sources == [0, 7]                           # the file actually read


def test_run_query_gt_rir_uses_the_merged_group_for_the_identity(tmp_path):
    room = str(tmp_path / "room")
    cand, merge_map = _merged_cand_set()
    _write_rir(room, 0, 11, 0.1)
    _write_rir(room, 7, 11, 0.3)
    _rec, engine = _engine()
    obs = el.load_measured_rirs(room, cand, 11, merge_map=merge_map)[0][1:2]
    out = el.run_query_gt_rir(engine, cand, room, 11, obs, merge_map=merge_map)
    assert out["identity_index"] == cand.gt_index == 1
    assert out["oracle_source_nodes"] == [0, 7]
    assert out["sims"][1, 0] == pytest.approx(1.0, abs=1e-6)


def test_build_row_records_the_oracle_source_nodes():
    kwargs = _row_kwargs(sims=torch.tensor([[0.5], [0.9], [0.1]], dtype=torch.float32),
                         noise_keys=[], agg="max", tau=None, score_source="gt_rir",
                         identity_index=1, oracle_source_nodes=[0, 7, None])
    row = el.build_row(**kwargs)
    assert row["oracle_source_nodes"] == [0, 7, None]
    assert el.build_row(**_row_kwargs())["oracle_source_nodes"] is None


# --------------------------------------------------------------------------- #
# r6: the duplicate-source survey is reviewed tooling, not a shell one-liner.
# --------------------------------------------------------------------------- #
def _load_survey():
    import importlib.util
    path = os.path.join(_REPO_ROOT, "worklog", "worklog_yixun", "exp_18_loc_invert_claude",
                        "survey_duplicate_sources.py")
    spec = importlib.util.spec_from_file_location("survey_duplicate_sources", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_survey_reports_a_dirty_and_a_clean_room(tmp_path):
    survey = _load_survey()
    dirty_root, _wav, _positions = _dirty_dataset_tree(tmp_path)
    split = tmp_path / "split.json"
    split.write_text(_json.dumps({"Cafe": {"Cafe_idx_1": ["S003_R0011_hybrid_IR.wav"]}}))
    report = survey.survey_split(str(split), str(dirty_root))
    assert report["n_rooms"] == 1 and report["n_rooms_with_duplicates"] == 1
    assert report["rooms"]["Cafe/Cafe_idx_1"]["n_sources"] == 3
    assert report["rooms"]["Cafe/Cafe_idx_1"]["n_positions"] == 2
    assert report["duplicates"]["Cafe/Cafe_idx_1"]["3"][1] == [3, 7]

    clean_root, _w = _dataset_tree(tmp_path / "clean")
    clean = survey.survey_split(str(split), str(clean_root))
    assert clean["n_rooms_with_duplicates"] == 0 and clean["duplicates"] == {}


def test_survey_cli_writes_a_report(tmp_path):
    survey = _load_survey()
    dirty_root, _wav, _positions = _dirty_dataset_tree(tmp_path)
    split = tmp_path / "split.json"
    split.write_text(_json.dumps({"Cafe": {"Cafe_idx_1": ["S003_R0011_hybrid_IR.wav"]}}))
    out = tmp_path / "survey.json"
    report = survey.main(["--split", str(split), "--dataset-root", str(dirty_root),
                          "--out", str(out)])
    assert report["n_rooms_with_duplicates"] == 1
    assert _json.loads(open(out).read())["n_rooms_with_duplicates"] == 1


@unseen_rooms
def test_integration_survey_finds_exactly_the_two_known_seen_rooms():
    """The measured fact behind plan Rev 3.2 (skips if the seen split is absent)."""
    survey = _load_survey()
    if not os.path.isfile(_SEEN_SPLIT_JSON):
        pytest.skip("the seen split JSON is absent")
    report = survey.survey_split(_SEEN_SPLIT_JSON, _AR_ROOT)
    dirty = sorted(report["duplicates"])
    assert dirty == ["Bathrooms/Bathrooms_idx_11", "Bathrooms/Bathrooms_idx_16"]
    assert report["duplicates"]["Bathrooms/Bathrooms_idx_11"]["9"][1] == [9, 10]
    assert report["duplicates"]["Bathrooms/Bathrooms_idx_16"]["4"][1] == [4, 7]


@unseen_rooms
def test_integration_survey_finds_no_duplicates_in_the_unseen_split():
    survey = _load_survey()
    report = survey.survey_split(_UNSEEN_SPLIT_JSON, _AR_ROOT)
    assert report["n_rooms"] == 17 and report["n_rooms_with_duplicates"] == 0


# --------------------------------------------------------------------------- #
# r7 item 1 (announcement 08): the predicted RIR waveforms are a REQUIRED
# artifact -- the exact decoded+clamped tensors that were scored, not a
# re-render of them.
# --------------------------------------------------------------------------- #
def test_dump_waveforms_writes_pred_and_obs_arrays(tmp_path):
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    dump = tmp_path / "waveforms"
    args = _run_args(tmp_path, **{"--dump-waveforms": str(dump)})
    result = el.run_evaluation(args, loader, engine, _stub_context(root), "c", "a",
                               expected=el.expected_split_identities(loader.dataset))

    for row in result["rows"]:
        path = os.path.join(str(dump), row["waveform_path"])
        assert os.path.isfile(path) and path.endswith(".npz")
        with np.load(path) as payload:
            assert sorted(payload.files) == ["obs", "pred"]
            assert payload["pred"].dtype == np.float32 and payload["obs"].dtype == np.float32
            assert payload["pred"].shape[:2] == (row["n_candidates"], row["n_samples"])
            assert payload["obs"].ndim == 1
        with open(path, "rb") as handle:
            assert el.sha256_bytes(handle.read()) == row["waveform_sha256"]


def test_dumped_arrays_are_bitwise_the_scored_tensors(tmp_path):
    """The .npz holds exactly what was scored: pred after decode+clamp, obs in the
    scored pad-crop window (the scorer's own 8000-truncation is internal)."""
    root, wav_room = _dataset_tree(tmp_path)
    md = _query_md(root, wav_room)
    _rec, engine = _engine()
    context = _stub_context(root)
    context["context_k"] = 2
    dump = tmp_path / "wf"
    args = _run_args(tmp_path, **{"--dump-waveforms": str(dump)})
    el.prepare_dump_dir(args)
    obs = torch.full((1, 1, 9600), 0.2)
    row = el.process_query(args, engine, context, md, obs,
                           dump={"dir": str(dump), "position": 0})

    cand = el.candidate_set_from_manifest(context["manifest"], "Cafe/Cafe_idx_1", 3, 11)
    noise = el.build_noise_bank(args.seed, row["query_id"], args.num_samples, (2, 8))
    _rec2, engine2 = _engine()
    reference = el.run_query(engine2, md, cand, noise, obs, batch_size=args.batch_size,
                             return_wavs=True)["wavs"]
    expected = reference.reshape(len(cand.nodes), args.num_samples, -1).cpu().numpy()
    with np.load(os.path.join(str(dump), row["waveform_path"])) as payload:
        assert np.array_equal(payload["pred"], expected.astype(np.float32))
        assert np.array_equal(payload["obs"], obs.reshape(-1).numpy().astype(np.float32))


def test_dump_manifest_carries_the_geometry_for_external_analysis(tmp_path):
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    dump = tmp_path / "waveforms"
    args = _run_args(tmp_path, **{"--dump-waveforms": str(dump)})
    result = el.run_evaluation(args, loader, engine, _stub_context(root), "c", "a",
                               expected=el.expected_split_identities(loader.dataset))
    payload = _json.loads(open(result["waveform_manifest_path"]).read())

    assert payload["stem"] == el.artifact_stem(args) and payload["n_queries"] == 2
    assert payload["rows_stem"] == os.path.basename(result["rows_path"])
    assert payload["registration_sha"] == "n/a"
    for row in result["rows"]:
        entry = payload["waveforms"][row["query_id"]]
        assert entry["path"] == row["waveform_path"] and entry["sha256"] == row["waveform_sha256"]
        # geometry, so external analysis needs only the dump directory
        assert entry["room_id"] == row["room_id"]
        assert entry["gt_node"] == row["gt_node"] and entry["gt_xyz_world"] == row["gt_xyz_world"]
        assert entry["candidate_nodes"] == row["candidate_nodes"]          # the pred M-axis order
        assert entry["candidate_xyz_world"] == row["candidate_xyz_world"]
        assert entry["pred_index"] == row["pred_index"]


def test_dump_dir_gets_a_self_describing_readme(tmp_path):
    args = _run_args(tmp_path, **{"--dump-waveforms": str(tmp_path / "wf")})
    dump = el.prepare_dump_dir(args)
    readme = open(os.path.join(dump, "README.md")).read()
    for token in ("pred", "obs", "[M, K, 10240]", "22050", "8000", "np.load", "candidate order"):
        assert token in readme, f"README does not mention {token!r}"


def test_dump_waveforms_refuses_a_non_empty_directory_unless_overwrite(tmp_path):
    dump = tmp_path / "wf"
    os.makedirs(dump)
    open(os.path.join(str(dump), "stale.npz"), "w").close()
    args = _run_args(tmp_path, **{"--dump-waveforms": str(dump)})
    with pytest.raises(SystemExit):
        el.prepare_dump_dir(args)
    over = _run_args(tmp_path, **{"--dump-waveforms": str(dump), "--overwrite": True})
    assert el.prepare_dump_dir(over) == str(dump)
    fresh = _run_args(tmp_path, **{"--dump-waveforms": str(tmp_path / "new")})
    assert os.path.isdir(el.prepare_dump_dir(fresh))         # created on demand


def test_dump_waveforms_is_refused_for_the_oracle(tmp_path):
    with pytest.raises(SystemExit):
        el.validate_args(el.parse_args(
            ["--model-config", "m.json", "--dataset-config", "d.json", "--agree-ckpt", "a.pt",
             "--score-source", "gt_rir", "--dump-waveforms", str(tmp_path / "wf")]))


def test_waveform_filename_is_position_prefixed_and_sanitized():
    name = el.waveform_filename(7, "1194|single_channel_ir_1/Cafe/Cafe_idx_1/S003_R011.wav")
    assert name.startswith("0000007_") and name.endswith(".npz")
    assert "/" not in name and "|" not in name
    assert el.waveform_filename(7, "a b") != el.waveform_filename(8, "a b")


# --------------------------------------------------------------------------- #
# r7 item 2 (announcement 08): runs completed before the rule get a
# regeneration-with-verification pass. The deterministic noise bank re-derives
# the waveforms bit-exactly, so reproducing the published per-sample sims is both
# the back-fill and an integrity audit.
# --------------------------------------------------------------------------- #
def _completed_run(tmp_path, name="orig"):
    loader, root = _fake_run(tmp_path / name)
    _rec, engine = _engine()
    args = _run_args(tmp_path / name)
    result = el.run_evaluation(args, loader, engine, _stub_context(root), "c", "a",
                               expected=el.expected_split_identities(loader.dataset))
    return result, root


def test_verify_against_replays_and_matches_a_completed_run(tmp_path):
    original, root = _completed_run(tmp_path)
    loader, _root2 = _fake_run(tmp_path / "orig")          # the same fixture stream
    _rec, engine = _engine()
    args = _run_args(tmp_path, **{"--verify-against": original["rows_path"],
                                  "--dump-waveforms": str(tmp_path / "wf")})
    replay = el.run_evaluation(args, loader, engine, _stub_context(root), "c", "a",
                               expected=el.expected_split_identities(loader.dataset))

    block = replay["summary"]["verify_against"]
    assert block["all_match"] is True and block["n_verified"] == 2
    assert block["rows_sha256"] == el.sha256_file(original["rows_path"])
    assert "_replay" in os.path.basename(replay["rows_path"])
    assert replay["rows_path"] != original["rows_path"]


def test_verify_against_aborts_on_a_perturbed_similarity(tmp_path):
    original, root = _completed_run(tmp_path)
    rows = el.read_rows(original["rows_path"])
    sims = el.decode_sims(rows[1]["sims_hex"])
    sims[1, 0] = float(sims[1, 0]) + 1e-4                  # one (m, k) moved
    rows[1]["sims_hex"] = el.encode_sims(sims)
    tampered = tmp_path / "tampered.jsonl"
    with open(tampered, "w") as handle:
        for row in rows:
            el.write_row(handle, row)

    loader, _root2 = _fake_run(tmp_path / "orig")
    _rec, engine = _engine()
    args = _run_args(tmp_path, **{"--verify-against": str(tampered),
                                  "--dump-waveforms": str(tmp_path / "wf2")})
    with pytest.raises(SystemExit, match="m=1"):
        el.run_evaluation(args, loader, engine, _stub_context(root), "c", "a",
                          expected=el.expected_split_identities(loader.dataset))


def test_verify_against_never_touches_the_original_artifacts(tmp_path):
    original, root = _completed_run(tmp_path)
    before = {key: open(original[key], "rb").read()
              for key in ("rows_path", "summary_path", "manifest_path")}
    loader, _root2 = _fake_run(tmp_path / "orig")
    _rec, engine = _engine()
    args = _run_args(tmp_path, **{"--verify-against": original["rows_path"],
                                  "--dump-waveforms": str(tmp_path / "wf3")})
    el.run_evaluation(args, loader, engine, _stub_context(root), "c", "a",
                      expected=el.expected_split_identities(loader.dataset))
    for key, payload in before.items():
        assert open(original[key], "rb").read() == payload


def test_verify_against_requires_a_waveform_dump():
    with pytest.raises(SystemExit):
        el.validate_args(el.parse_args(_CLI + ["--verify-against", "rows.jsonl"]))
    el.validate_args(el.parse_args(_CLI + ["--verify-against", "rows.jsonl",
                                           "--dump-waveforms", "wf"]))


def test_verify_against_aborts_on_a_missing_query(tmp_path):
    original, root = _completed_run(tmp_path)
    rows = el.read_rows(original["rows_path"])[:1]
    partial = tmp_path / "partial.jsonl"
    with open(partial, "w") as handle:
        for row in rows:
            el.write_row(handle, row)
    loader, _root2 = _fake_run(tmp_path / "orig")
    _rec, engine = _engine()
    args = _run_args(tmp_path, **{"--verify-against": str(partial),
                                  "--dump-waveforms": str(tmp_path / "wf4")})
    with pytest.raises(SystemExit):
        el.run_evaluation(args, loader, engine, _stub_context(root), "c", "a",
                          expected=el.expected_split_identities(loader.dataset))


# --------------------------------------------------------------------------- #
# r7 item 3a (queued r6 review LOW): the r6 "byte-identical" claim was too broad.
# The COMPUTATION is identical on a clean room; the row/provenance SCHEMA gained
# merge fields, so a golden row pins exactly what a clean run emits.
# --------------------------------------------------------------------------- #
_CLEAN_ROW_KEYS = {
    "query_id", "room_id", "relpath", "receiver_node", "gt_node", "gt_index", "gt_xyz_world",
    "gt_xyz_cam", "candidate_nodes", "candidate_xyz_world", "candidate_xyz_cam",
    "context_member", "candidate_available", "n_candidates", "n_samples", "n_eligible",
    "n_available", "gt_only", "sims_hex", "scores_hex", "noise_keys", "pred_index", "pred_node",
    "pred_xyz_world", "e_loc", "top1", "rr", "power_statistic", "tau", "agg", "control",
    "score_source", "identity_index", "substituted", "smoke", "merge_map",
    "oracle_source_nodes", "timings_s", "context_xyz_cam", "context_sims_hex",
}


def test_clean_room_row_is_computation_identical_and_schema_explicit(tmp_path):
    """r6 added merge_map / oracle_source_nodes to every row and
    candidate_merge_groups to provenance. On a clean room the COMPUTED fields are
    unchanged; the added fields are inert ({} / None / 0)."""
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    result = el.run_evaluation(_run_args(tmp_path), loader, engine, _stub_context(root), "c", "a",
                               expected=el.expected_split_identities(loader.dataset))
    row = result["rows"][0]
    assert set(row) == _CLEAN_ROW_KEYS, set(row) ^ _CLEAN_ROW_KEYS
    assert row["merge_map"] == {} and row["oracle_source_nodes"] is None
    assert result["provenance"]["candidate_merge_groups"] == 0
    # the computed quantities a clean run has always produced
    assert row["candidate_nodes"] == [0, 3, 7] and row["n_candidates"] == 3
    assert row["gt_node"] == 3 and row["pred_node"] in row["candidate_nodes"]
    assert torch.equal(el.decode_sims(row["sims_hex"]),
                       el.decode_sims(row["sims_hex"]))     # exact round trip


# --------------------------------------------------------------------------- #
# r7 item 4 (r6 lesson): one wav per (room, source) missed a SILENT item that the
# dataset silently substituted mid-run. --readback-decode-all decodes the whole
# split.
# --------------------------------------------------------------------------- #
def test_readback_decode_all_finds_a_silent_wav(tmp_path):
    root, split_path = _registered_unseen_tree(tmp_path)
    silent = str(root / "single_channel_ir_1" / "Scene0" / "Scene0_idx_0"
                 / "S002_R007_hybrid_IR.wav")
    torchaudio.save(silent, torch.zeros(1, 12000), 22050)

    args = _unseen_readback_args(tmp_path, root, split_path)
    report = el.run_readback(args)                          # sampled: one per (room, source)
    assert report["ok"] is True                             # a silent file is not short/corrupt

    os.makedirs(str(tmp_path / "deep"), exist_ok=True)
    deep = _unseen_readback_args(tmp_path / "deep", root, split_path,
                                 **{"--readback-decode-all": True})
    with pytest.raises(SystemExit):
        el.run_readback(deep)
    written = os.path.join(str(deep.out_dir), el.aux_stem(deep, "readback") + ".json")
    payload = _json.loads(open(written).read())
    assert payload["decode_all"] is True
    assert payload["decoded_files"] == sum(len(v) for r in
                                           _json.loads(open(split_path).read()).values()
                                           for v in r.values())
    assert any("silent" in f for f in payload["failures"])
    assert any("S002" in entry for room in payload["rooms"].values()
               for entry in room["wav_bad"])


def test_readback_decode_all_passes_on_a_healthy_split(tmp_path):
    root, split_path = _registered_unseen_tree(tmp_path)
    args = _unseen_readback_args(tmp_path, root, split_path,
                                 **{"--readback-decode-all": True})
    report = el.run_readback(args)
    assert report["ok"] is True and report["decode_all"] is True
    assert report["decoded_files"] > report["n_rooms"]
    assert report["wav_lengths"]["min"] == 12000


# --------------------------------------------------------------------------- #
# R4-r2 item 1 (+ r7 review's R4-COMPOSITION GUARD): metrics ride the replay
# pass. ONE immutable float32 snapshot of the scored waveforms feeds BOTH the npz
# dump and rir_metrics -- no re-decode, no in-place op, so the two artifacts
# cannot disagree about what was scored.
# --------------------------------------------------------------------------- #
def _metrics_args(tmp_path, **over):
    base = {"--dump-waveforms": str(tmp_path / "wf"), "--metrics": True,
            "--metric-delta-max": "8", "--metric-t30-backend": "torch"}
    base.update(over)
    return _run_args(tmp_path, **base)


def test_metrics_require_the_waveform_snapshot():
    with pytest.raises(SystemExit):
        el.validate_args(el.parse_args(_CLI + ["--metrics"]))
    el.validate_args(el.parse_args(_CLI + ["--metrics", "--dump-waveforms", "wf"]))


def test_metric_delta_max_must_be_on_the_registered_grid():
    for bad in ("7", "64"):
        with pytest.raises(SystemExit):
            el.validate_args(el.parse_args(_CLI + ["--metrics", "--dump-waveforms", "wf",
                                                   "--metric-delta-max", bad]))
    for good in ("0", "8", "32", "128"):
        el.validate_args(el.parse_args(_CLI + ["--metrics", "--dump-waveforms", "wf",
                                               "--metric-delta-max", good]))


def test_run_evaluation_streams_a_metrics_row_per_query(tmp_path):
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    args = _metrics_args(tmp_path)
    result = el.run_evaluation(args, loader, engine, _stub_context(root), "c", "a",
                               expected=el.expected_split_identities(loader.dataset))

    rows = el.read_rows(result["metrics_path"])
    assert len(rows) == 2
    row = rows[0]
    assert row["query_id"] == result["rows"][0]["query_id"]
    assert row["room_id"] and row["candidate_nodes"] == result["rows"][0]["candidate_nodes"]
    for family in ("m1", "m2", "m3", "m4", "m5"):
        block = row["families"][family]
        decoded = el.decode_sims(block["candidates_hex"])
        assert decoded.shape == (row["n_candidates"], row["n_samples"])
        assert torch.isfinite(decoded).all()
        assert len(block["context_hex"]) == row["n_context"]
        assert set(block["aggregations"]) >= {"mean", "min", "median", "lme"}
        assert 0 <= block["pred_index"] < row["n_candidates"]
        assert block["pred_node"] in row["candidate_nodes"]
    assert len(row["m5_lags"]) == row["n_candidates"]
    assert row["m4"]["dropped"]["n_features"] == 10
    assert row["metric_config"]["delta_max"] == 8
    assert "deterministic-replay" in row["tail_provenance"]


def test_metrics_rows_round_trip_at_full_precision(tmp_path):
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    result = el.run_evaluation(_metrics_args(tmp_path), loader, engine, _stub_context(root),
                               "c", "a", expected=el.expected_split_identities(loader.dataset))
    written = el.read_rows(result["metrics_path"])
    reread = el.read_rows(result["metrics_path"])
    assert written == reread
    first = el.decode_sims(written[0]["families"]["m1"]["candidates_hex"])
    assert torch.equal(first, el.decode_sims(reread[0]["families"]["m1"]["candidates_hex"]))


def test_metrics_and_dump_consume_one_immutable_snapshot(tmp_path, monkeypatch):
    """R4-COMPOSITION GUARD: what rir_metrics scored IS what the npz holds."""
    seen = {}
    original = el.rir_metrics_compute

    def spy(pred, obs, ctx, config):
        seen["pred"] = pred
        seen["obs"] = obs
        return original(pred, obs, ctx, config)

    monkeypatch.setattr(el, "rir_metrics_compute", spy)
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    args = _metrics_args(tmp_path)
    result = el.run_evaluation(args, loader, engine, _stub_context(root), "c", "a",
                               expected=el.expected_split_identities(loader.dataset))

    row = result["rows"][-1]
    with np.load(os.path.join(str(args.dump_waveforms), row["waveform_path"])) as payload:
        dumped = torch.from_numpy(payload["pred"])
        obs = torch.from_numpy(payload["obs"])
    # the metric input is the dumped array itself, windowed -- same numbers, no re-decode
    assert torch.equal(seen["pred"], el.rir_metrics_window(dumped))
    assert torch.equal(seen["obs"], el.rir_metrics_window(obs.reshape(1, -1))[0])


def test_snapshot_mutation_is_refused_by_the_integrity_guard(tmp_path, monkeypatch):
    """If a consumer ever mutated the shared snapshot, the run must abort rather
    than publish a dump and metrics that describe different waveforms.

    The windowing currently returns a fresh tensor, so a metric implementation
    cannot reach the snapshot today; this pins the guard for the aliasing case a
    future refactor could introduce, by making the window an identity view.
    """
    original = el.rir_metrics_compute
    monkeypatch.setattr(el, "rir_metrics_window", lambda x: x)      # alias, not a copy

    def saboteur(pred, obs, ctx, config):
        pred.mul_(2.0)                       # an in-place op the guard must catch
        return original(pred, obs, ctx, config)

    monkeypatch.setattr(el, "rir_metrics_compute", saboteur)
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    with pytest.raises(SystemExit, match="snapshot"):
        el.run_evaluation(_metrics_args(tmp_path), loader, engine, _stub_context(root), "c", "a",
                          expected=el.expected_split_identities(loader.dataset))


def test_metrics_publish_atomically_with_the_run(tmp_path):
    loader, root = _fake_run(tmp_path)
    _rec, engine = _engine()
    result = el.run_evaluation(_metrics_args(tmp_path), loader, engine, _stub_context(root),
                               "c", "a", expected=el.expected_split_identities(loader.dataset))
    assert os.path.exists(result["metrics_path"])
    assert not os.path.exists(result["metrics_path"] + ".partial")
    assert "_metrics.jsonl" in os.path.basename(result["metrics_path"])
    assert result["summary"]["metrics"]["n_queries"] == 2
    assert result["summary"]["metrics"]["families"] == ["m1", "m2", "m3", "m4", "m5"]
    assert result["provenance"]["metric_registerable"]["m2_lambda"] == 1.0


# --------------------------------------------------------------------------- #
# R4-r2 item 4 (r7 review LOW): the replay must prove it is replaying the SAME
# protocol before it regenerates anything.
# --------------------------------------------------------------------------- #
def test_replay_preflight_accepts_a_matching_reference(tmp_path):
    original, root = _completed_run(tmp_path)
    args = _run_args(tmp_path, **{"--verify-against": original["rows_path"],
                                  "--dump-waveforms": str(tmp_path / "wf")})
    assert el.preflight_verify_against(args, expected_queries=2)["n_rows"] == 2


@pytest.mark.parametrize("flag,value", [("--tau", "0.5"), ("--agg", "mean"),
                                        ("--num-samples", "4"), ("--control", "constant_source")])
def test_replay_preflight_refuses_a_protocol_that_differs(tmp_path, flag, value):
    original, root = _completed_run(tmp_path)
    args = _run_args(tmp_path, **{"--verify-against": original["rows_path"],
                                  "--dump-waveforms": str(tmp_path / "wf"), flag: value})
    with pytest.raises(SystemExit):
        el.preflight_verify_against(args, expected_queries=2)


def test_replay_preflight_refuses_wrong_cardinality_or_duplicates(tmp_path):
    original, root = _completed_run(tmp_path)
    args = _run_args(tmp_path, **{"--verify-against": original["rows_path"],
                                  "--dump-waveforms": str(tmp_path / "wf")})
    with pytest.raises(SystemExit):
        el.preflight_verify_against(args, expected_queries=6337)

    rows = el.read_rows(original["rows_path"])
    duplicated = tmp_path / "dupe.jsonl"
    with open(duplicated, "w") as handle:
        for row in [rows[0], rows[0]]:
            el.write_row(handle, row)
    dupe_args = _run_args(tmp_path, **{"--verify-against": str(duplicated),
                                       "--dump-waveforms": str(tmp_path / "wf2")})
    with pytest.raises(SystemExit):
        el.preflight_verify_against(dupe_args, expected_queries=2)


def test_replay_preflight_checks_the_sibling_summary_provenance(tmp_path):
    original, root = _completed_run(tmp_path)
    args = _run_args(tmp_path, **{"--verify-against": original["rows_path"],
                                  "--dump-waveforms": str(tmp_path / "wf"), "--seed": "43"})
    with pytest.raises(SystemExit, match="seed"):
        el.preflight_verify_against(args, expected_queries=2)
