import torch

from src.data.yaw_rotation import invariant_conditioning
from src.localization.engine import (
    CONTEXT_CONDITIONING_IDS,
    FA_CONTEXT_CONDITIONING_IDS,
    FA_DYNAMIC_CONDITIONING_IDS,
    SCORE_SAMPLE_COUNTS,
    SOURCE_CONDITIONING_IDS,
    cache_conditioning_branch,
    cache_invariant_conditioning_branch,
    candidate_seed,
    deterministic_noise,
    encode_audio_features,
    load_frozen_query,
    merge_cached_conditioning,
    prepare_generated_audio,
    project_runtime_seconds,
)


class FakeConditioner:
    def __call__(self, metadata, device, only_ids=None):
        keys = tuple(only_ids) if only_ids is not None else (
            *SOURCE_CONDITIONING_IDS,
            *CONTEXT_CONDITIONING_IDS,
        )
        output = {}
        for key in keys:
            if key.startswith("source"):
                values = torch.stack([item[key].float() for item in metadata]).to(device)
                if values.ndim == 2:
                    values = values.unsqueeze(1)
            else:
                values = torch.stack([item[key].float() for item in metadata]).to(device)
                if values.ndim == 2:
                    values = values.unsqueeze(1)
            output[key] = [values, torch.ones(len(metadata), 1, device=device)]
        return output


def _metadata(source):
    source = torch.tensor(source, dtype=torch.float32)
    return {
        "source_vit": source.unsqueeze(0),
        "source": source,
        "context_poses_vit": torch.arange(24, dtype=torch.float32).reshape(8, 3),
        "context_poses": torch.arange(24, dtype=torch.float32).reshape(8, 3) + 1,
        "context_audio": torch.arange(32, dtype=torch.float32).reshape(8, 4),
    }


def test_cached_and_uncached_conditioning_are_bit_identical():
    conditioner = FakeConditioner()
    candidates = [_metadata([1, 2, 3]), _metadata([4, 5, 6])]
    uncached = conditioner(candidates, "cpu")
    context = cache_conditioning_branch(
        conditioner, [candidates[0]], CONTEXT_CONDITIONING_IDS, "cpu"
    )
    source = cache_conditioning_branch(
        conditioner, candidates, SOURCE_CONDITIONING_IDS, "cpu"
    )
    cached = merge_cached_conditioning(source, context, batch_size=2)
    assert tuple(cached) == tuple(uncached)
    for key in uncached:
        assert torch.equal(cached[key][0], uncached[key][0])
        assert torch.equal(cached[key][1], uncached[key][1])


def test_cached_fa_conditioning_matches_released_full_c4_path():
    conditioner = FakeConditioner()
    candidates = [_metadata([1, 2, 3]), _metadata([4, 5, 6])]
    for item in candidates:
        item["depth"] = torch.zeros(3, 4, 8)
    uncached = invariant_conditioning(conditioner, candidates, "cpu")
    context = cache_invariant_conditioning_branch(
        conditioner, [candidates[0]], FA_CONTEXT_CONDITIONING_IDS, "cpu"
    )
    source = cache_invariant_conditioning_branch(
        conditioner, candidates, SOURCE_CONDITIONING_IDS, "cpu"
    )
    dynamic = cache_invariant_conditioning_branch(
        conditioner, candidates, FA_DYNAMIC_CONDITIONING_IDS, "cpu"
    )
    cached = merge_cached_conditioning(
        source, context, batch_size=2, dynamic_branch=dynamic
    )
    for key in uncached:
        assert torch.equal(cached[key][0], uncached[key][0])
        assert torch.equal(cached[key][1], uncached[key][1])


class FakeRetrieval:
    def __init__(self):
        self.seen = None

    def compute_audio_features(self, waveforms):
        self.seen = waveforms
        return waveforms.mean(dim=-1).squeeze()


def test_audio_features_delegate_to_retrieval_and_keep_batch_axis():
    retrieval = FakeRetrieval()
    waveform = torch.arange(10240, dtype=torch.float32).reshape(1, 1, 10240)
    actual = encode_audio_features(retrieval, waveform)
    assert retrieval.seen is waveform
    assert actual.shape == (1, 1)


def test_frozen_query_preserves_context_channel_axis(monkeypatch, tmp_path):
    waveform = torch.linspace(-0.5, 0.5, 12000).reshape(1, -1)
    monkeypatch.setattr(
        "src.localization.engine.torchaudio.load",
        lambda _path: (waveform.clone(), 22050),
    )
    monkeypatch.setattr(
        "src.localization.engine.np.load",
        lambda _path: torch.zeros(8, 16).numpy(),
    )
    monkeypatch.setattr(
        "src.localization.engine.convert_equirect_to_camera_coord",
        lambda _depth, height, width: torch.zeros(height, width, 3),
    )
    record = {
        "query_id": "single_channel_ir_1/Cafe/Cafe_idx_1/S002_R068_hybrid_IR.wav",
        "filename": "S002_R068_hybrid_IR.wav",
        "scene": "Cafe",
        "room": "Cafe_idx_1",
        "receiver_global": [1.0, 2.0, 3.0],
        "source_global": [2.0, 4.0, 6.0],
        "context_sources_global": [[1.0, 2.0, 3.0]] * 8,
        "contexts": [f"context_{index}.wav" for index in range(8)],
    }
    observed, metadata = load_frozen_query(record, tmp_path)
    assert observed.shape == (1, 10240)
    assert metadata["context_audio"].shape == (8, 1, 9600)
    assert metadata["context_poses"].shape == (8, 3)
    assert metadata["depth"].shape == (3, 256, 512)


def test_generated_audio_clamp_matches_eval_contract():
    waveform = torch.tensor([[[-2.0, -1.0, 0.25, 1.0, 2.0]]], dtype=torch.float32)
    assert torch.equal(
        prepare_generated_audio(waveform, sample_size=5), waveform.clamp(-1.0, 1.0)
    )
    try:
        prepare_generated_audio(waveform.double(), sample_size=5)
    except ValueError as error:
        assert "float32" in str(error)
    else:
        raise AssertionError("non-float32 waveform must fail")


def test_candidate_seeds_and_noise_are_batch_partition_invariant():
    seeds = [candidate_seed(42, 7, candidate, 0) for candidate in range(5)]
    assert len(set(seeds)) == 5
    all_noise = deterministic_noise(seeds, (2, 3), device="cpu", dtype=torch.float32)
    split_noise = torch.cat(
        [
            deterministic_noise(seeds[:2], (2, 3), device="cpu", dtype=torch.float32),
            deterministic_noise(seeds[2:], (2, 3), device="cpu", dtype=torch.float32),
        ]
    )
    assert torch.equal(all_noise, split_noise)


def test_runtime_projection_adds_cache_and_generation_components():
    assert SCORE_SAMPLE_COUNTS == (1, 4, 8)
    projected = project_runtime_seconds(
        query_count=10,
        receiver_candidate_count=100,
        query_candidate_count=1000,
        query_io_seconds_per_query=0.05,
        context_seconds_per_query=0.2,
        observation_seconds_per_query=0.1,
        source_candidates_per_second=20,
        generated_scores_per_second=5,
        score_samples=2,
    )
    assert projected == {
        "query_io_seconds": 0.5,
        "context_cache_seconds": 2.0,
        "observation_seconds": 1.0,
        "source_cache_seconds": 5.0,
        "generation_score_seconds": 400.0,
        "total_seconds": 408.5,
    }
