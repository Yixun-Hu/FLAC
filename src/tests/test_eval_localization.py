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
                smoke=False, max_queries=None)
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


def test_output_paths_stamp_smoke_seed_and_k(tmp_path):
    rows, summary = el.output_paths(tmp_path / "out", "exp18_R2", num_samples=8, seed=43,
                                    smoke=False)
    assert os.path.basename(rows) == "exp18_R2_K8_seed43_rows.jsonl"
    assert os.path.basename(summary) == "exp18_R2_K8_seed43_summary.json"
    assert os.path.isdir(os.path.dirname(rows))                      # created on demand

    smoke_rows, smoke_summary = el.output_paths(tmp_path / "out", "exp18_R2", num_samples=8,
                                                seed=43, smoke=True)
    assert os.path.basename(smoke_rows) == "exp18_R2_K8_seed43_smoke_rows.jsonl"
    assert "_smoke_" in os.path.basename(smoke_summary)


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


def test_parse_args_requires_num_samples():
    with pytest.raises(SystemExit):
        el.parse_args(_CLI[:-2])


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
                  rec_loc=(1.0, 2.0, 0.5)):
    """A miniature AcousticRooms tree: metadata pair JSONs + IR wavs."""
    root = tmp_path / "AcousticRooms"
    meta_room = root / "metadata" / "Cafe" / "Cafe_idx_1"
    wav_room = root / "single_channel_ir_1" / "Cafe" / "Cafe_idx_1"
    meta_room.mkdir(parents=True, exist_ok=True)
    wav_room.mkdir(parents=True, exist_ok=True)
    for node, xyz in sources:
        (meta_room / f"S00{node}_R00{receiver}.json").write_text(_json.dumps(
            {"src_loc": list(xyz), "rec_loc": list(rec_loc), "IR_norm": 1.0}))
        _write_rir(str(wav_room), node, receiver, 0.1 * (node + 1))
    return root, wav_room


def _query_md(root, wav_room, src=3, receiver=11, rec_loc=(1.0, 2.0, 0.5),
              src_loc=(2.0, -1.0, 1.5)):
    from src.localization.candidates import project_to_camera
    path = str(wav_room / f"S00{src}_R00{receiver}_hybrid_IR.wav")
    source = torch.as_tensor(project_to_camera(np.asarray(rec_loc), np.asarray(src_loc)),
                             dtype=torch.float32)
    return {"idx": 0, "path": path, "relpath": os.path.relpath(path, str(root)),
            "scene": "Cafe", "source": source, "source_vit": source.unsqueeze(0),
            "context_poses": torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                                          dtype=torch.float32),
            "context_audio": torch.stack([torch.full((1, 9600), 0.4),
                                          torch.full((1, 9600), -0.15)]),
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
    argv = ["--model-config", "m.json", "--dataset-config", "d.json",
            "--ckpt-path", "c.ckpt", "--agree-ckpt", "a.pt", "--num-samples", "2",
            "--out-dir", str(tmp_path / "out"), "--eval-name", "unit", "--device", "cpu"]
    for flag, value in over.items():
        argv += [flag] if value is True else [flag, str(value)]
    return el.validate_args(el.parse_args(argv))


def _stub_context():
    return {"weights_source": "ema", "latent_shape": (2, 8), "device": "cpu"}


def test_run_evaluation_end_to_end_writes_rows_and_summary(tmp_path):
    loader, _root = _fake_run(tmp_path)
    _rec, engine = _engine()
    args = _run_args(tmp_path)
    result = el.run_evaluation(args, loader, engine, _stub_context(), "ck" * 32, "ag" * 32)

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
    first = el.run_evaluation(_run_args(tmp_path / "a"), loader, engine, _stub_context(), "c", "a")
    loader2, _root2 = _fake_run(tmp_path)
    _rec2, engine2 = _engine()
    second = el.run_evaluation(_run_args(tmp_path / "b"), loader2, engine2, _stub_context(), "c", "a")
    assert [r["sims_hex"] for r in first["rows"]] == [r["sims_hex"] for r in second["rows"]]


def test_run_evaluation_smoke_truncates_after_auditing_the_truncated_enumeration(tmp_path):
    loader, _root = _fake_run(tmp_path)
    _rec, engine = _engine()
    args = _run_args(tmp_path, **{"--smoke": True, "--max-queries": 1})
    result = el.run_evaluation(args, loader, engine, _stub_context(), "c", "a")
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
        el.run_evaluation(args, loader, engine, _stub_context(), "c", "a")
    rows_path, _summary = el.output_paths(args.out_dir, args.eval_name, args.num_samples,
                                          args.seed, args.smoke)
    assert not os.path.exists(rows_path)                       # nothing was written


def test_run_evaluation_constant_source_control_is_recorded(tmp_path):
    loader, _root = _fake_run(tmp_path)
    _rec, engine = _engine()
    args = _run_args(tmp_path, **{"--control": "constant_source"})
    result = el.run_evaluation(args, loader, engine, _stub_context(), "c", "a")
    assert all(row["control"] == "constant_source" for row in result["rows"])
    assert result["provenance"]["control"] == "constant_source"


def test_run_evaluation_gt_rir_mode_scores_measured_files(tmp_path):
    loader, _root = _fake_run(tmp_path)
    _rec, engine = _engine()
    args = _run_args(tmp_path, **{"--score-source": "gt_rir", "--agg": "max"})
    result = el.run_evaluation(args, loader, engine, _stub_context(), "c", "a")
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
