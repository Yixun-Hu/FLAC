"""Stream-audit identity for RAF (exp_19 r2, Codex R3 / contracts Amendment 2).

RAF receivers sit inside one 1.46 m array, so two different context draws can
render to the SAME six-decimal position fingerprint. The C7 provenance fix
(`context_capture_ids`) therefore has to reach `eval_FLAC`'s stream recorder,
which previously fingerprinted `context_poses` only.

The change is strictly additive and the AR/HAA path must stay BYTE-identical: a
peer session imports `sample_context_ids` and `canonical_stream_hash` directly and
pairs cells by those digests. Every golden literal in the AR section below was
produced by the PRE-change code and is pinned here as a regression fence.
"""
import os
import sys

import numpy as np
import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import eval_FLAC  # noqa: E402

from test_raf_md import (  # noqa: E402  (pytest prepends src/tests)
    GROUP_KEYS, N_PER_GROUP, N_SUPPORT, ROOM, load_raf_md, runtime_root,  # noqa: F401
)


def _ar_md(idx=3, relpath="sceneA/binaural_rirs/x.wav"):
    """Metadata shaped like AR_md's output: context_poses, no capture ids."""
    return {
        "idx": idx,
        "relpath": relpath,
        "context_poses": torch.tensor([[1.5, -2.25, 0.125], [0.0, -0.0, 3.0]],
                                      dtype=torch.float32),
    }


def _raf_md(ids=(17, 3), scene=ROOM, poses=None, idx=0,
            relpath="EmptyRoom/mono_rirs_22050Hz/000000.wav"):
    md = {
        "idx": idx,
        "relpath": relpath,
        "scene": scene,
        "context_poses": torch.zeros(len(ids), 3, dtype=torch.float32) if poses is None
        else poses,
        "context_capture_ids": torch.tensor(list(ids), dtype=torch.int64),
    }
    return md


# --------------------------------------------------------------------------- #
# AR / HAA path: byte-identical (goldens from the pre-change code)
# --------------------------------------------------------------------------- #
def test_ar_context_fingerprints_are_unchanged():
    assert eval_FLAC.sample_context_ids(_ar_md()) == [
        "1.500000,-2.250000,0.125000", "0.000000,0.000000,3.000000"]


def test_ar_stream_hashes_are_unchanged():
    stream = eval_FLAC.RotationStream()
    stream.record(_ar_md(), 7, 512)
    stream.record(_ar_md(idx=4, relpath="sceneA/binaural_rirs/y.wav"), 11, 512)
    assert stream.input_hash() == \
        "6821c1052864ecf877695ae5c96d1bf83fcce77eaebb72fa31ad1f710be92e31"
    assert stream.assignment_hash() == \
        "0d40a09e4363afcc59ba833688ce67f5c96cce9816d484be16a0477317a2fd93"


def test_canonical_stream_hash_is_unchanged():
    assert eval_FLAC.canonical_stream_hash(
        [[0, "a", ["1.000000,2.000000,3.000000"], 512]]) == \
        "dd37f0cb1739e0f62ea64e3e09eb3d89ee901a595e5dc1c12323d12e0d452397"


def test_ar_metadata_without_context_is_still_empty():
    assert eval_FLAC.sample_context_ids({"idx": 0}) == []


def test_pose_fingerprint_schema_constant_is_unchanged():
    """The AR rendering rule did not change, so its schema id may not move either:
    the peer's collector validates AR sidecars against the literal 1."""
    assert eval_FLAC.CONTEXT_FINGERPRINT_SCHEMA == 1
    assert eval_FLAC.CONTEXT_ID_FINGERPRINT_SCHEMA == 2


def test_ar_stream_record_declares_the_pose_schema():
    stream = eval_FLAC.RotationStream()
    stream.record(_ar_md(), 7, 512)
    record = eval_FLAC.build_stream_record(
        eval_FLAC.resolve_rotation_plan('fixed', 0.0, None), stream)
    assert record["fingerprint_schema"] == 1


# --------------------------------------------------------------------------- #
# RAF path: room-qualified, zero-padded capture ids
# --------------------------------------------------------------------------- #
def test_raf_fingerprints_use_room_qualified_zero_padded_ids():
    assert eval_FLAC.sample_context_ids(_raf_md(ids=(17, 3))) == [
        "EmptyRoom|000017", "EmptyRoom|000003"]


def test_raf_fingerprints_preserve_draw_order():
    assert eval_FLAC.sample_context_ids(_raf_md(ids=(3, 17))) == [
        "EmptyRoom|000003", "EmptyRoom|000017"]


def test_raf_fingerprints_separate_rooms():
    a = eval_FLAC.sample_context_ids(_raf_md(ids=(17, 3), scene="EmptyRoom"))
    b = eval_FLAC.sample_context_ids(_raf_md(ids=(17, 3), scene="FurnishedRoom"))
    assert a != b


def test_raf_ids_discriminate_draws_that_positions_cannot():
    """The whole point of R3: identical six-decimal poses, different captures.

    RAF placements are re-occupied to sub-centimetre precision, so this is the
    real failure mode, not a contrived one."""
    poses = torch.tensor([[0.123456, -1.0, 0.5], [0.2, 0.3, 0.4]], dtype=torch.float32)
    a = _raf_md(ids=(17, 3), poses=poses)
    b = _raf_md(ids=(18, 4), poses=poses.clone())
    assert eval_FLAC.sample_context_ids(a) != eval_FLAC.sample_context_ids(b)
    # ... while the pose-only rendering could not tell them apart
    pose_only = {k: v for k, v in a.items() if k != "context_capture_ids"}
    pose_only_b = {k: v for k, v in b.items() if k != "context_capture_ids"}
    assert eval_FLAC.sample_context_ids(pose_only) == \
        eval_FLAC.sample_context_ids(pose_only_b)


def test_raf_stream_record_declares_the_id_schema():
    stream = eval_FLAC.RotationStream()
    stream.record(_raf_md(), 0, 512)
    record = eval_FLAC.build_stream_record(
        eval_FLAC.resolve_rotation_plan('fixed', 0.0, None), stream)
    assert record["fingerprint_schema"] == 2
    assert record["input_tuples"][0][2] == ["EmptyRoom|000017", "EmptyRoom|000003"]


def test_a_stream_may_not_mix_fingerprint_schemas():
    """One run, one identity rule: a mixed stream's hash would be uninterpretable."""
    stream = eval_FLAC.RotationStream()
    stream.record(_raf_md(), 0, 512)
    stream.record(_ar_md(), 1, 512)
    with pytest.raises(ValueError):
        eval_FLAC.build_stream_record(
            eval_FLAC.resolve_rotation_plan('fixed', 0.0, None), stream)


# --------------------------------------------------------------------------- #
# fail-closed validation of the id tensor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ids", [
    torch.tensor([17, 3], dtype=torch.int32),
    torch.tensor([17.0, 3.0], dtype=torch.float32),
    np.array([17, 3]),
    [17, 3],
])
def test_capture_ids_must_be_an_int64_tensor(ids):
    md = _raf_md()
    md["context_capture_ids"] = ids
    with pytest.raises(ValueError):
        eval_FLAC.sample_context_ids(md)


@pytest.mark.parametrize("ids", [
    torch.zeros(0, dtype=torch.int64),
    torch.tensor([[17, 3]], dtype=torch.int64),
    torch.tensor(17, dtype=torch.int64),
])
def test_capture_ids_must_be_a_non_empty_vector(ids):
    md = _raf_md()
    md["context_capture_ids"] = ids
    with pytest.raises(ValueError):
        eval_FLAC.sample_context_ids(md)


def test_capture_ids_length_must_match_context_poses():
    md = _raf_md(ids=(17, 3, 5))
    md["context_poses"] = torch.zeros(2, 3, dtype=torch.float32)
    with pytest.raises(ValueError) as exc:
        eval_FLAC.sample_context_ids(md)
    assert "context_poses" in str(exc.value)


def test_capture_ids_require_context_poses_to_verify_against():
    md = _raf_md()
    del md["context_poses"]
    with pytest.raises(ValueError):
        eval_FLAC.sample_context_ids(md)


def test_capture_ids_still_validate_the_pose_tensor():
    md = _raf_md()
    md["context_poses"] = torch.zeros(2, 3, dtype=torch.float64)
    with pytest.raises(ValueError):
        eval_FLAC.sample_context_ids(md)


def test_negative_capture_ids_are_rejected():
    md = _raf_md(ids=(17, -3))
    with pytest.raises(ValueError):
        eval_FLAC.sample_context_ids(md)


def test_duplicate_capture_ids_are_rejected():
    """A repeated capture means the draw collapsed: K distinct references were
    promised, so recording it as a valid identity would hide a real defect."""
    md = _raf_md(ids=(17, 17))
    with pytest.raises(ValueError):
        eval_FLAC.sample_context_ids(md)


@pytest.mark.parametrize("scene", [None, "", "Empty|Room", 5])
def test_room_qualification_is_fail_closed(scene):
    md = _raf_md()
    if scene is None:
        del md["scene"]
    else:
        md["scene"] = scene
    with pytest.raises(ValueError):
        eval_FLAC.sample_context_ids(md)


# --------------------------------------------------------------------------- #
# end to end: real loader -> real collation -> stream record
# --------------------------------------------------------------------------- #
def test_raf_stream_end_to_end(runtime_root, tmp_path):
    import json

    from src.data.dataset import LocalDatasetConfig, SampleDataset, collation_fn

    split = {ROOM: [f"{i:06d}.wav" for i in range(4)]}
    split_path = tmp_path / "test_base.json"
    with open(split_path, "w") as f:
        json.dump(split, f)
    config = LocalDatasetConfig(
        id="RAF", path=str(runtime_root),
        custom_metadata_fn=load_raf_md().get_custom_metadata,
        json_file_path=str(split_path), folder_name="mono_rirs_22050Hz",
        conditioning={
            "acoustic_context": {"load": True, "max_context": 8, "max_len": 9600,
                                 "deterministic": True},
            "depth": {"load": True}, "poses": {"load": True}},
    )
    dataset = SampleDataset([config], sample_size=10240, sample_rate=22050,
                            random_crop=False, force_channels="mono", augs=False)
    _, metadata = collation_fn([dataset[i] for i in range(4)])

    stream = eval_FLAC.RotationStream()
    for md in metadata:
        stream.record(md, None, 512)

    record = eval_FLAC.build_stream_record(
        eval_FLAC.resolve_rotation_plan('fixed', 0.0, None), stream)
    assert record["fingerprint_schema"] == 2
    assert record["stream_count"] == 4

    support = {f"{ROOM}|{i:06d}" for i in range(N_SUPPORT)}
    for position, (row, md) in enumerate(zip(record["input_tuples"], metadata)):
        ids = row[2]
        assert len(ids) == 8
        assert set(ids).issubset(support)
        assert len(set(ids)) == 8
        # the fingerprint names the captures the loader actually drew
        assert ids == [f"{ROOM}|{int(c):06d}" for c in md["context_capture_ids"]]
        assert f"{ROOM}|{int(md['sample_target_id']):06d}" not in ids
    # hashes are recomputable from the published preimages
    assert eval_FLAC.canonical_stream_hash(record["input_tuples"]) == record["input_hash"]
    assert eval_FLAC.canonical_stream_hash(record["assignment_tuples"]) == \
        record["assignment_hash"]
