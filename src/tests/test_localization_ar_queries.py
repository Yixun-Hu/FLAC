import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.localization.ar_queries import (
    ContextProtocol,
    attach_context_selections,
    clone_with_candidate,
    context_availability_histogram,
    filter_materialized_scope,
    load_context_manifest,
    parse_split_queries,
    save_context_manifest,
)


def _fake_dataset(tmp_path: Path):
    root = tmp_path / "AcousticRooms"
    room = root / "single_channel_ir_1" / "Cafe" / "Cafe_idx_1"
    room.mkdir(parents=True)
    names = [f"S{i:03d}_R001_hybrid_IR.wav" for i in range(1, 11)]
    for name in names:
        (room / name).touch()
    split = {"Cafe": {"Cafe_idx_1": [names[2], names[9]]}}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    return root, split_path


def test_parse_order_and_released_s010_pool_quirk(tmp_path):
    root, split_path = _fake_dataset(tmp_path)
    queries = parse_split_queries(split_path, root)

    assert [q.filename for q in queries] == [
        "S003_R001_hybrid_IR.wav",
        "S010_R001_hybrid_IR.wav",
    ]
    # Released f"S00{node}" resolves 1..9, but node 10 becomes S0010.
    assert queries[0].eligible_context_count == 8
    assert queries[1].eligible_context_count == 9
    assert all("S010_" not in p for p in queries[0].eligible_context_relpaths)


def test_manifest_requires_full_order_then_filters(tmp_path):
    root, split_path = _fake_dataset(tmp_path)
    queries = parse_split_queries(split_path, root)
    protocol = ContextProtocol(seed=42, batch_size=64, num_workers=4)
    selections = [
        [queries[0].eligible_context_relpaths[0]] * 8,
        list(queries[1].eligible_context_relpaths[:8]),
    ]
    manifest = attach_context_selections(queries, selections, protocol)

    with pytest.raises(ValueError, match="full split order"):
        attach_context_selections(queries[1:], selections[1:], protocol)

    path = tmp_path / "manifest.json"
    save_context_manifest(manifest, path)
    assert load_context_manifest(path) == manifest
    assert json.loads(path.read_text())["sha256"] == manifest["sha256"]

    filtered = filter_materialized_scope(
        manifest, excluded_room="Cafe_idx_1", expected_excluded=2
    )
    assert filtered["records"] == []
    assert filtered["excluded_count"] == 2


def test_context_histogram_and_candidate_clone(tmp_path):
    root, split_path = _fake_dataset(tmp_path)
    queries = parse_split_queries(split_path, root)
    assert context_availability_histogram(queries) == {8: 1, 9: 1}

    md = {
        "source": torch.tensor([1.0, 2.0, 3.0]),
        "source_vit": torch.tensor([[1.0, 2.0, 3.0]]),
        "context_audio": torch.ones(8, 1, 16),
        "context_poses": torch.ones(8, 3),
        "scene": "Cafe",
    }
    out = clone_with_candidate(md, np.array([5.0, 7.0, 11.0]), np.array([1.0, 2.0, 3.0]))
    assert torch.equal(out["source"], torch.tensor([4.0, 5.0, 8.0]))
    assert torch.equal(out["source_vit"], torch.tensor([[4.0, 5.0, 8.0]]))
    for key in ("context_audio", "context_poses", "scene"):
        if isinstance(md[key], torch.Tensor):
            assert torch.equal(out[key], md[key])
        else:
            assert out[key] == md[key]


def test_real_split_context_census_if_assets_available():
    root = Path("/home/zhixuanzhao/projects/rir2rir/FLAC/AcousticRooms")
    if not root.exists():
        pytest.skip("AcousticRooms assets are external to the repository")
    queries = parse_split_queries(Path("data/AR/unseen_eval.json"), root)
    assert len(queries) == 6337
    assert context_availability_histogram(queries) == {6: 91, 7: 429, 8: 5263, 9: 554}
    kept = [q for q in queries if q.room != "ListeningRoom_idx_2"]
    assert len(kept) == 5337
    assert context_availability_histogram(kept) == {6: 91, 7: 429, 8: 4363, 9: 454}
