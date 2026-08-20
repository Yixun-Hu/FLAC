"""Tests for ``src/configs/dataset_configs/custom_metadata/RAF_md.py``.

exp_19 (RAF finetune), contract section D. TDD cycles:

* cycle 9  — scene / poses / depth (no flipud) + the bounded per-worker caches
* cycle 10 — acoustic context (stochastic train vs deterministic eval), provenance
  tensors, and the shape contract through the REAL collation path

The hook is loaded exactly the way ``src/data/dataset.py`` loads it (dynamically,
by file path), so each test gets a module with fresh module-level caches.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import raf_common  # noqa: E402

from src.data.dataset import RAFPublicationError  # noqa: E402

_RAF_MD_PATH = os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs",
                            "custom_metadata", "RAF_md.py")

ROOM = "EmptyRoom"
N_SUPPORT = 12
N_TEST = 4
N_PER_GROUP = N_SUPPORT + N_TEST
GROUP_KEYS = ["g0" + "0" * 14, "g1" + "0" * 14, "g2" + "0" * 14]


def load_raf_md(test_mode=True):
    """Load RAF_md.py the way ``create_dataloader_from_config`` does.

    ``test_mode`` sets the module's test-only opt-out so synthetic fixtures -- which
    are not published trees -- can be read. It is a constant on a freshly exec'd
    module object, reachable only from here: no config, flag or environment
    variable can disable the production gate (r5 finding 1).
    """
    spec = importlib.util.spec_from_file_location("raf_metadata_module", _RAF_MD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._RAF_MD_TEST_MODE = bool(test_mode)
    return mod


@pytest.fixture
def raf_md():
    return load_raf_md()


# --------------------------------------------------------------------------- #
# synthetic runtime dataset
# --------------------------------------------------------------------------- #
def _depth_map(group_index, h=256, w=512):
    """A distinguishable, strictly positive depth map per group."""
    rows = np.linspace(1.0, 2.0, h, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 0.5, w, dtype=np.float32)[None, :]
    return (rows + cols + float(group_index)).astype(np.float32)


def _wave(seed, n=12000):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=n) * 0.01).astype(np.float32)


@pytest.fixture
def runtime_root(tmp_path):
    """<root>/RAF/<Room>/{mono_rirs_22050Hz, metadata, depth_images}."""
    root = tmp_path / "RAF"
    room = root / ROOM
    (room / "mono_rirs_22050Hz").mkdir(parents=True)
    (room / "metadata").mkdir()
    (room / "depth_images").mkdir()

    poses, groups_meta = {}, {}
    cid = 0
    for gi, gk in enumerate(GROUP_KEYS):
        tx = [float(gi), 0.5 * gi, 1.5]
        ids = []
        for m in range(N_PER_GROUP):
            capture_id = f"{cid:06d}"
            ids.append(capture_id)
            sf.write(str(room / "mono_rirs_22050Hz" / f"{capture_id}.wav"),
                     _wave(cid), 22050, subtype="FLOAT")
            poses[capture_id] = {
                "tx_xyz_p": tx,
                "quat_raw": [0.1 * (gi + 1), 0.9, 0.0, 0.1],
                "rx_p": [float(m) * 0.1, 1.0 + 0.05 * m, 0.6 + 0.01 * m],
                "group_key": gk,
                "split_role": "train" if m < N_SUPPORT else "test",
            }
            cid += 1
        groups_meta[gk] = {
            "tx_xyz_p": tx,
            "depth_file": f"{gk}_depth_image.npy",
            "train_ids": ids[:N_SUPPORT],
            "role": "train_test",
        }
        np.save(str(room / "depth_images" / f"{gk}_depth_image.npy"), _depth_map(gi))

    with open(room / "metadata" / "poses_metadata.json", "w") as f:
        json.dump(poses, f)
    with open(room / "metadata" / "groups_metadata.json", "w") as f:
        json.dump(groups_meta, f)
    return root


def _modalities(max_context=8, max_len=9600, deterministic=False, poses=True,
                depth=True, context=True):
    return {
        "acoustic_context": {"load": context, "max_context": max_context,
                             "max_len": max_len, "deterministic": deterministic},
        "depth": {"load": depth},
        "poses": {"load": poses},
    }


def _info(runtime_root, capture_id, modalities=None):
    rel = os.path.join(ROOM, "mono_rirs_22050Hz", f"{capture_id}.wav")
    return {
        "path": os.path.join(str(runtime_root), rel),
        "relpath": rel,
        "modalities": _modalities() if modalities is None else modalities,
    }


# --------------------------------------------------------------------------- #
# scene + poses (cycle 9)
# --------------------------------------------------------------------------- #
def test_scene_is_the_room_name(raf_md, runtime_root):
    md = raf_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert md["scene"] == ROOM


def test_source_is_the_source_centred_receiver_position(raf_md, runtime_root):
    """Mapping H: the frame is centred on the source, the receiver fills the
    ``source``/``source_vit`` slots (HAA's sim2real naming is kept deliberately)."""
    md = raf_md.get_custom_metadata(_info(runtime_root, "000003"), None)
    # capture 000003 -> group 0 (tx = [0, 0, 1.5]), mic 3 (rx = [0.3, 1.15, 0.63])
    expected = np.array([0.3, 1.15, 0.63]) - np.array([0.0, 0.0, 1.5])
    assert md["source"].shape == (3,)
    assert md["source"].dtype == torch.float32
    np.testing.assert_allclose(md["source"].numpy(), expected, atol=1e-6)
    assert md["source_vit"].shape == (1, 3)
    np.testing.assert_allclose(md["source_vit"].numpy()[0], expected, atol=1e-6)


def test_second_group_uses_its_own_tx(raf_md, runtime_root):
    cid = f"{N_PER_GROUP:06d}"          # first capture of group 1
    md = raf_md.get_custom_metadata(_info(runtime_root, cid), None)
    expected = np.array([0.0, 1.0, 0.6]) - np.array([1.0, 0.5, 1.5])
    np.testing.assert_allclose(md["source"].numpy(), expected, atol=1e-6)


def test_poses_can_be_switched_off(raf_md, runtime_root):
    md = raf_md.get_custom_metadata(
        _info(runtime_root, "000000", _modalities(poses=False)), None)
    assert "source" not in md and "source_vit" not in md


def test_unknown_capture_fails_closed(raf_md, runtime_root):
    with pytest.raises(KeyError):
        raf_md.get_custom_metadata(_info(runtime_root, "999999"), None)


# --------------------------------------------------------------------------- #
# depth (cycle 9)
# --------------------------------------------------------------------------- #
def test_depth_shape_dtype_and_pixel_to_ray_convention(raf_md, runtime_root):
    md = raf_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    depth = md["depth"]
    assert depth.shape == (3, 256, 512)
    assert depth.dtype == torch.float32
    assert torch.isfinite(depth).all()
    raw = _depth_map(0)
    expected = raw[..., None] * raf_common.equirect_directions()   # [H, W, 3]
    np.testing.assert_allclose(depth.numpy(), np.transpose(expected, (2, 0, 1)),
                               rtol=1e-6, atol=0.0)


def test_depth_is_not_flipped(raf_md, runtime_root):
    """HAA_md applies ``np.flipud``; the RAF renderer already emits rows in the
    pipeline's row order, so applying it here would put the ceiling underfoot."""
    md = raf_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    raw = _depth_map(0)
    flipped = np.flipud(raw)[..., None] * raf_common.equirect_directions()
    assert not np.allclose(md["depth"].numpy(), np.transpose(flipped, (2, 0, 1)))
    # row 0 of the map must be the zenith: +z points up there
    assert md["depth"][2, 0, :].min() > 0
    assert md["depth"][2, -1, :].max() < 0


def test_depth_uses_the_group_map(raf_md, runtime_root):
    md0 = raf_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    md1 = raf_md.get_custom_metadata(_info(runtime_root, f"{N_PER_GROUP:06d}"), None)
    assert not torch.allclose(md0["depth"], md1["depth"])


def test_depth_can_be_switched_off(raf_md, runtime_root):
    md = raf_md.get_custom_metadata(
        _info(runtime_root, "000000", _modalities(depth=False)), None)
    assert "depth" not in md


def test_missing_depth_map_fails_closed(raf_md, runtime_root):
    os.remove(runtime_root / ROOM / "depth_images" / f"{GROUP_KEYS[0]}_depth_image.npy")
    with pytest.raises(FileNotFoundError):
        raf_md.get_custom_metadata(_info(runtime_root, "000000"), None)


# --------------------------------------------------------------------------- #
# caches (cycle 9)
# --------------------------------------------------------------------------- #
def test_metadata_json_is_loaded_once_per_worker(raf_md, runtime_root, monkeypatch):
    calls = []
    real_load = json.load

    def counting_load(fp, *a, **kw):
        calls.append(getattr(fp, "name", "?"))
        return real_load(fp, *a, **kw)

    monkeypatch.setattr(raf_md.json, "load", counting_load)
    for cid in ("000000", "000001", "000002"):
        raf_md.get_custom_metadata(_info(runtime_root, cid), None)
    assert len(calls) == 2   # poses_metadata.json + groups_metadata.json, once each


def test_depth_cache_avoids_reloading_the_same_map(raf_md, runtime_root, monkeypatch):
    calls = []
    real_np_load = np.load

    def counting_load(path, *a, **kw):
        calls.append(str(path))
        return real_np_load(path, *a, **kw)

    monkeypatch.setattr(raf_md.np, "load", counting_load)
    for cid in ("000000", "000001", f"{N_PER_GROUP:06d}"):
        raf_md.get_custom_metadata(_info(runtime_root, cid), None)
    assert len(calls) == 2   # two distinct groups


def test_depth_cache_is_bounded_and_evicts_least_recently_used(raf_md, runtime_root,
                                                               monkeypatch):
    monkeypatch.setattr(raf_md, "_DEPTH_CACHE_MAX", 2)
    for gi in range(3):
        raf_md.get_custom_metadata(_info(runtime_root, f"{gi * N_PER_GROUP:06d}"), None)
    assert len(raf_md._DEPTH_CACHE) == 2
    assert all(GROUP_KEYS[0] not in k for k in raf_md._DEPTH_CACHE)


def test_depth_cache_hands_out_independent_tensors(raf_md, runtime_root):
    a = raf_md.get_custom_metadata(_info(runtime_root, "000000"), None)["depth"]
    a += 1000.0
    b = raf_md.get_custom_metadata(_info(runtime_root, "000001"), None)["depth"]
    assert b.max() < 100.0


def test_default_depth_cache_bound_is_64(raf_md):
    assert raf_md._DEPTH_CACHE_MAX == 64


# --------------------------------------------------------------------------- #
# acoustic context (cycle 10)
# --------------------------------------------------------------------------- #
def _context_ids(md):
    return [f"{int(i):06d}" for i in md["context_capture_ids"].tolist()]


def test_context_shapes_and_dtypes(raf_md, runtime_root):
    md = raf_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert md["context_audio"].shape == (8, 1, 9600)
    assert md["context_audio"].dtype == torch.float32
    assert md["context_poses"].shape == (8, 3)
    assert torch.equal(md["context_poses_vit"], md["context_poses"])
    assert md["context_capture_ids"].shape == (8,)
    assert md["context_capture_ids"].dtype == torch.int64
    assert md["sample_target_id"].dtype == torch.int64
    assert md["sample_target_id"].item() == 0
    assert md["sample_target_id"].shape == ()


def test_context_pool_is_the_group_support_minus_the_target(raf_md, runtime_root):
    md = raf_md.get_custom_metadata(_info(runtime_root, "000003"), None)
    ids = _context_ids(md)
    support = [f"{i:06d}" for i in range(N_SUPPORT)]
    assert set(ids).issubset(set(support))
    assert "000003" not in ids
    assert len(set(ids)) == 8


def test_context_pool_of_a_test_item_is_the_full_support(raf_md, runtime_root):
    md = raf_md.get_custom_metadata(_info(runtime_root, f"{N_SUPPORT:06d}"), None)
    support = {f"{i:06d}" for i in range(N_SUPPORT)}
    assert set(_context_ids(md)).issubset(support)


def test_context_poses_are_source_centred(raf_md, runtime_root):
    md = raf_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    for slot, cid in enumerate(_context_ids(md)):
        m = int(cid)
        expected = np.array([m * 0.1, 1.0 + 0.05 * m, 0.6 + 0.01 * m]) - np.array([0.0, 0.0, 1.5])
        np.testing.assert_allclose(md["context_poses"][slot].numpy(), expected, atol=1e-6)


def test_context_audio_is_cropped_and_padded_to_max_len(raf_md, runtime_root):
    long_md = raf_md.get_custom_metadata(
        _info(runtime_root, "000000", _modalities(max_len=9600)), None)
    assert long_md["context_audio"].shape[-1] == 9600
    pad_md = raf_md.get_custom_metadata(
        _info(runtime_root, "000000", _modalities(max_len=20000)), None)
    assert pad_md["context_audio"].shape[-1] == 20000
    assert torch.all(pad_md["context_audio"][:, :, 12000:] == 0)
    assert torch.any(pad_md["context_audio"][:, :, :12000] != 0)


def test_train_mode_context_is_stochastic_but_seed_reproducible(raf_md, runtime_root):
    info = _info(runtime_root, "000000", _modalities(deterministic=False, max_context=4))
    np.random.seed(0)
    a = _context_ids(raf_md.get_custom_metadata(info, None))
    np.random.seed(0)
    b = _context_ids(raf_md.get_custom_metadata(info, None))
    np.random.seed(1)
    c = _context_ids(raf_md.get_custom_metadata(info, None))
    assert a == b
    assert a != c


def test_eval_mode_context_ignores_every_ambient_rng(raf_md, runtime_root):
    """Deterministic draws must not depend on worker topology, item order, or the
    diffusion seed: only on (room, capture id)."""
    info = _info(runtime_root, "000003", _modalities(deterministic=True))
    np.random.seed(0)
    torch.manual_seed(0)
    first = _context_ids(raf_md.get_custom_metadata(info, None))
    np.random.seed(7)
    torch.manual_seed(1234)
    for other in ("000001", "000002"):   # other items drawn in between
        raf_md.get_custom_metadata(_info(runtime_root, other,
                                         _modalities(deterministic=True)), None)
    second = _context_ids(raf_md.get_custom_metadata(info, None))
    assert first == second


def test_eval_mode_context_survives_a_fresh_module_load(runtime_root):
    """A second worker process re-executes the hook; it must draw the same set."""
    info = _info(runtime_root, "000005", _modalities(deterministic=True))
    a = _context_ids(load_raf_md().get_custom_metadata(info, None))
    b = _context_ids(load_raf_md().get_custom_metadata(info, None))
    assert a == b


def test_eval_mode_context_differs_between_targets_and_rooms(raf_md, runtime_root):
    a = _context_ids(raf_md.get_custom_metadata(
        _info(runtime_root, "000000", _modalities(deterministic=True)), None))
    b = _context_ids(raf_md.get_custom_metadata(
        _info(runtime_root, "000001", _modalities(deterministic=True)), None))
    assert a != b
    # the seed is (room, capture id): a different room name must change the draw
    pool = [f"{i:06d}" for i in range(N_SUPPORT)]
    assert (raf_md.select_context_ids("EmptyRoom", "000000", pool, 8, True)
            != raf_md.select_context_ids("FurnishedRoom", "000000", pool, 8, True))


def test_deterministic_draw_ignores_the_pool_order(raf_md):
    pool = [f"{i:06d}" for i in range(N_SUPPORT)]
    a = raf_md.select_context_ids(ROOM, "000000", pool, 8, True)
    b = raf_md.select_context_ids(ROOM, "000000", list(reversed(pool)), 8, True)
    assert a == b


def test_context_pool_too_small_fails_closed(raf_md, runtime_root):
    with pytest.raises(ValueError):
        raf_md.get_custom_metadata(
            _info(runtime_root, "000000", _modalities(max_context=N_SUPPORT)), None)


def test_context_can_be_switched_off(raf_md, runtime_root):
    md = raf_md.get_custom_metadata(
        _info(runtime_root, "000000", _modalities(context=False)), None)
    for key in ("context_audio", "context_poses", "context_capture_ids",
                "sample_target_id"):
        assert key not in md


def test_cached_base_metadata_is_never_mutated(raf_md, runtime_root):
    path = os.path.join(str(runtime_root), ROOM, "metadata", "groups_metadata.json")
    with open(path) as f:
        on_disk = json.load(f)
    for cid in ("000000", "000005", f"{N_PER_GROUP:06d}"):
        raf_md.get_custom_metadata(_info(runtime_root, cid), None)
    assert raf_md.load_json_cached(path) == on_disk


# --------------------------------------------------------------------------- #
# shape contract through the REAL collation path (cycle 10)
# --------------------------------------------------------------------------- #
def _dataset(runtime_root, tmp_path, deterministic=False):
    from src.data.dataset import LocalDatasetConfig, SampleDataset

    split = {ROOM: [f"{i:06d}.wav" for i in range(8)]}
    split_path = tmp_path / "train_base.json"
    with open(split_path, "w") as f:
        json.dump(split, f)

    config = LocalDatasetConfig(
        id="RAF",
        path=str(runtime_root),
        custom_metadata_fn=load_raf_md().get_custom_metadata,
        json_file_path=str(split_path),
        folder_name="mono_rirs_22050Hz",
        conditioning=_modalities(deterministic=deterministic),
    )
    return SampleDataset([config], sample_size=10240, sample_rate=22050,
                         random_crop=False, force_channels="mono", augs=False)


def test_batch_shapes_through_the_real_collation_path(runtime_root, tmp_path):
    from src.data.dataset import collation_fn

    dataset = _dataset(runtime_root, tmp_path)
    assert len(dataset) == 8
    audio, metadata = collation_fn([dataset[i] for i in range(4)])
    assert audio.shape == (4, 1, 10240)
    assert audio.dtype == torch.float32
    assert len(metadata) == 4
    for md in metadata:
        assert md["scene"] == ROOM
        assert md["source"].shape == (3,)
        assert md["source_vit"].shape == (1, 3)
        assert md["context_poses"].shape == (8, 3)
        assert md["context_poses_vit"].shape == (8, 3)
        assert md["context_audio"].shape == (8, 1, 9600)
        assert md["depth"].shape == (3, 256, 512)
        assert md["depth"].dtype == torch.float32
        assert torch.isfinite(md["depth"]).all()
        assert md["context_capture_ids"].shape == (8,)
        assert md["sample_target_id"].dtype == torch.int64


def test_provenance_identifies_the_item_after_collation(runtime_root, tmp_path):
    from src.data.dataset import collation_fn

    dataset = _dataset(runtime_root, tmp_path, deterministic=True)
    _, metadata = collation_fn([dataset[i] for i in range(4)])
    assert [int(md["sample_target_id"]) for md in metadata] == [0, 1, 2, 3]
    for md in metadata:
        target = int(md["sample_target_id"])
        assert target not in md["context_capture_ids"].tolist()


# --------------------------------------------------------------------------- #
# r2 R5: the loader validates the depth map it is handed
# --------------------------------------------------------------------------- #
def _corrupt_depth(runtime_root, transform):
    path = runtime_root / ROOM / "depth_images" / f"{GROUP_KEYS[0]}_depth_image.npy"
    np.save(str(path), transform(np.load(str(path))))


def test_depth_load_rejects_a_wrong_shape(raf_md, runtime_root):
    _corrupt_depth(runtime_root, lambda a: a[:128])
    with pytest.raises(ValueError) as exc:
        raf_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert "RAF depth map contract" in str(exc.value)


def test_depth_load_rejects_a_wrong_dtype(raf_md, runtime_root):
    _corrupt_depth(runtime_root, lambda a: a.astype(np.float64))
    with pytest.raises(ValueError) as exc:
        raf_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert "RAF depth map contract" in str(exc.value)


@pytest.mark.parametrize("bad", [np.inf, np.nan, -1.0, 0.0])
def test_depth_load_rejects_non_finite_or_non_positive_values(raf_md, runtime_root, bad):
    def corrupt(a):
        a = a.copy()
        a[7, 9] = bad
        return a

    _corrupt_depth(runtime_root, corrupt)
    with pytest.raises(ValueError) as exc:
        raf_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert "RAF depth map contract" in str(exc.value)


def test_depth_validation_failure_is_distinctive_enough_to_survive_substitution():
    """SampleDataset swallows loader exceptions and substitutes a random item, so
    the message is the only trace left in the log: it must name the contract."""
    import inspect

    source = inspect.getsource(load_raf_md().validate_depth_map)
    assert "RAF depth map contract" in source


# --------------------------------------------------------------------------- #
# r2 R9: a real batched conditioner pass over the real collated batch
# --------------------------------------------------------------------------- #
def _raf_conditioning_config(cond_dim=256):
    """The RAF model config's conditioning block with a LIGHTWEIGHT ViT injected.

    Only the ViT backbone is substituted (permitted by contracts Amendment 2 R9):
    the DINOv3 weights are a network download and 21M parameters, neither of which
    a unit test may depend on. Everything else -- conditioner ids, types, cond_dim,
    the dist_embedder frequencies, the RIR encoder's STFT settings -- is the real
    config, so the shapes asserted below are the shapes the run will produce.
    """
    path = os.path.join(_REPO_ROOT, "src", "configs", "model_configs", "FLAC", "RAF",
                        "FLAC_RAF_finetune.json")
    with open(path) as f:
        model_config = json.load(f)
    conditioning = model_config["model"]["conditioning"]
    assert conditioning["cond_dim"] == cond_dim
    for entry in conditioning["configs"]:
        if entry["type"] == "ViTCoordinates":
            entry["config"]["ViT"] = {          # SimpleViT branch: built locally
                "ch_dim": 3, "img_h": 256, "img_w": 512,
                "patch_h": 64, "patch_w": 64,
                "dim": 32, "depth": 1, "heads": 2, "mlp_dim": 32,
            }
    return model_config, conditioning


def _raf_batch(runtime_root, tmp_path, n=2):
    from src.data.dataset import LocalDatasetConfig, SampleDataset, collation_fn

    split = {ROOM: [f"{i:06d}.wav" for i in range(n)]}
    split_path = tmp_path / "batch.json"
    with open(split_path, "w") as f:
        json.dump(split, f)
    config = LocalDatasetConfig(
        id="RAF", path=str(runtime_root),
        custom_metadata_fn=load_raf_md().get_custom_metadata,
        json_file_path=str(split_path), folder_name="mono_rirs_22050Hz",
        conditioning=_modalities(deterministic=True),
    )
    dataset = SampleDataset([config], sample_size=10240, sample_rate=22050,
                            random_crop=False, force_channels="mono", augs=False)
    return collation_fn([dataset[i] for i in range(n)])


def test_real_multiconditioner_consumes_the_real_raf_batch(runtime_root, tmp_path):
    """C10's consumer-level contract: the batch RAF_md emits must survive the
    actual MultiConditioner built from the actual RAF conditioning config."""
    from src.models.conditioners import create_multi_conditioner_from_conditioning_config

    _, conditioning = _raf_conditioning_config()
    conditioner = create_multi_conditioner_from_conditioning_config(conditioning)
    _, metadata = _raf_batch(runtime_root, tmp_path, n=2)

    with torch.no_grad():
        out = conditioner(list(metadata), device="cpu")

    assert set(out) == {"source", "source_vit", "context_poses_vit", "context_poses",
                        "context_audio"}
    for key, (tensor, mask) in out.items():
        assert tensor.dtype == torch.float32, key
        assert torch.isfinite(tensor).all(), key
        assert tensor.shape[0] == 2, key
        assert tensor.shape[-1] == 256, key          # cond_dim
        # Upstream shape: every conditioner emits a per-ITEM mask [B, 1], not one
        # entry per token. Shared with the AR/HAA path; recorded, not changed here.
        assert mask.shape == (2, 1), key
    # one token per reference for the context conditioners, one for the source
    assert out["source"][0].shape == (2, 1, 256)
    assert out["source_vit"][0].shape == (2, 1, 256)
    assert out["context_poses"][0].shape == (2, 8, 256)
    assert out["context_poses_vit"][0].shape == (2, 8, 256)
    assert out["context_audio"][0].shape == (2, 8, 256)


def test_conditioning_assembles_into_cross_attention_and_global_inputs(runtime_root,
                                                                       tmp_path):
    """Run the REAL get_conditioning_inputs over the REAL id lists: this is what
    the DiT is handed, and it is where a wrong per-conditioner shape shows up."""
    import types

    from src.models.conditioners import create_multi_conditioner_from_conditioning_config
    from src.models.diffusion import ConditionedDiffusionModelWrapper

    model_config, conditioning = _raf_conditioning_config()
    diffusion_config = model_config["model"]["diffusion"]
    conditioner = create_multi_conditioner_from_conditioning_config(conditioning)
    _, metadata = _raf_batch(runtime_root, tmp_path, n=2)

    with torch.no_grad():
        tensors = conditioner(list(metadata), device="cpu")

    stub = types.SimpleNamespace(
        cross_attn_cond_ids=diffusion_config["cross_attention_cond_ids"],
        global_cond_ids=diffusion_config["global_cond_ids"],
        input_concat_ids=[], prepend_cond_ids=[])
    inputs = ConditionedDiffusionModelWrapper.get_conditioning_inputs(stub, tensors)

    cross = inputs["cross_attn_cond"]
    assert cross.dtype == torch.float32
    assert cross.shape[0] == 2 and cross.shape[2] == 256
    # context_poses_vit (8) + context_poses (8) + context_audio tokens
    assert cross.shape[1] == (tensors["context_poses_vit"][0].shape[1]
                              + tensors["context_poses"][0].shape[1]
                              + tensors["context_audio"][0].shape[1])
    assert cross.shape == (2, 24, 256)      # 8 + 8 + 8 reference tokens
    # the mask concatenates the three per-item masks, one column per conditioner
    assert inputs["cross_attn_mask"].shape == (2, 3)
    # adaLN global conditioning: source + source_vit concatenated on the channel dim
    assert inputs["global_cond"].shape == (2, 512)
    assert inputs["global_cond"].dtype == torch.float32
    assert torch.isfinite(inputs["global_cond"]).all()
    assert diffusion_config["config"]["global_cond_dim"] == 512


# --------------------------------------------------------------------------- #
# r4 T3: the loader verifies the publication once per process
# --------------------------------------------------------------------------- #
def _publish_tree(tmp_path, runtime_root, rooms=None, canonical=True,
                  parameters_ok=True, digest=None, with_depth=True,
                  output_dir=None, parameters=None, marker_digest=None):
    """Publish the synthetic tree the way prepare_data + render_depth do:
    a prepare marker in the SPLIT directory and a depth marker per room."""
    import publish as raf_publish
    import readback_audit as raf_readback

    rooms = list(raf_publish.CANONICAL_ROOMS) if rooms is None else list(rooms)
    split_dir = tmp_path / "splits"
    pointer = {
        "split_dir": str(split_dir.resolve()),
        "output_dir": str((output_dir or runtime_root).resolve()),
        "rooms": list(rooms),
        "canonical": canonical,
        "taint": [],
        "parameters": {"n_groups": 16},
        "canonical_parameters": parameters_ok,
        "readback_record": {
            "sha256": raf_readback.CANONICAL_RECORD_SHA256 if digest is None else digest},
    }
    prepare_extra = {
        "canonical": canonical,
        "canonical_parameters": parameters_ok,
        "taint": [],
        "parameters": dict(raf_publish.CANONICAL_PREPARE_PARAMS, rooms=list(rooms),
                           amplitude_derivation_sha256="a" * 64,
                           **(parameters or {})),
        "readback_record": {"sha256": marker_digest or pointer["readback_record"]["sha256"]},
    }
    depth_extra = {
        "canonical": canonical,
        "canonical_parameters": parameters_ok,
        "taint": [],
        "parameters": dict(raf_publish.CANONICAL_RENDER_PARAMS, rooms=list(rooms)),
        "readback_record": {"sha256": marker_digest or pointer["readback_record"]["sha256"]},
    }
    with raf_publish.PublishTransaction(str(split_dir), kind="prepare") as txn:
        runtime = txn.stage(str(runtime_root))
        splits = txn.stage(str(split_dir))
        with open(runtime.path("raf_publication.json"), "w") as f:
            json.dump(pointer, f)
        with open(splits.path("train_base.json"), "w") as f:
            json.dump({ROOM: []}, f)
        txn.commit(extra=prepare_extra)
    if with_depth:
        with raf_publish.PublishTransaction(str(runtime_root), kind="depth") as txn:
            for room in rooms:
                staged = txn.stage(str(runtime_root / room / "depth_images"))
                with open(staged.path("attested.txt"), "w") as f:
                    f.write("depth")
            txn.commit(extra=depth_extra)
    return pointer


@pytest.fixture
def gated_md(runtime_root, tmp_path):
    """RAF_md with the production gate ACTIVE (no test-mode opt-out)."""
    return load_raf_md(test_mode=False)


def test_the_gate_is_mandatory_and_has_no_environment_switch():
    """r5 finding 1: an operator who forgets an env var would silently train on an
    unpublished tree, so there is no env var to forget."""
    import inspect

    source = inspect.getsource(load_raf_md(test_mode=False))
    assert "RAF_REQUIRE_PUBLICATION" not in source
    assert "os.environ" not in source
    assert load_raf_md(test_mode=False)._RAF_MD_TEST_MODE is False


def test_gate_accepts_a_fully_published_tree(gated_md, runtime_root, tmp_path):
    _publish_tree(tmp_path, runtime_root)
    md = gated_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert md["scene"] == ROOM


def test_gate_refuses_a_tree_that_was_never_published(gated_md, runtime_root):
    with pytest.raises(RAFPublicationError) as exc:
        gated_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert "raf_publication.json" in str(exc.value)


def test_gate_refuses_a_prepared_tree_whose_depth_was_never_published(
        gated_md, runtime_root, tmp_path):
    """The r4 gate checked only the prepare marker; depth maps published under a
    different generation (or not at all) went unnoticed."""
    _publish_tree(tmp_path, runtime_root, with_depth=False)
    with pytest.raises(RAFPublicationError) as exc:
        gated_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert "depth" in str(exc.value)


def test_gate_refuses_a_tree_whose_payload_changed_after_publication(
        gated_md, runtime_root, tmp_path):
    _publish_tree(tmp_path, runtime_root)
    (runtime_root / "raf_publication.json").write_text('{"tampered": true}')
    with pytest.raises(RAFPublicationError):
        gated_md.get_custom_metadata(_info(runtime_root, "000000"), None)


def test_gate_finds_the_prepare_marker_in_the_split_directory(
        gated_md, runtime_root, tmp_path):
    """r5 finding 1's core defect: production writes that marker under split_dir,
    so a gate looking beside the data could never verify a real publication."""
    import publish as raf_publish

    _publish_tree(tmp_path, runtime_root)
    assert (tmp_path / "splits" / raf_publish.marker_name("prepare")).exists()
    assert not (runtime_root / raf_publish.marker_name("prepare")).exists()
    gated_md.get_custom_metadata(_info(runtime_root, "000000"), None)


def test_canonical_tree_must_carry_the_registered_identity(
        gated_md, runtime_root, tmp_path):
    _publish_tree(tmp_path, runtime_root, digest="0" * 64)
    with pytest.raises(RAFPublicationError) as exc:
        gated_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert "pinned" in str(exc.value)


def test_canonical_tree_must_have_registered_parameters(
        gated_md, runtime_root, tmp_path):
    """The consumer checks the PAYLOAD, not the producer's boolean (F3): the
    marker here claims canonical_parameters yet carries a wrong n_groups."""
    _publish_tree(tmp_path, runtime_root, parameters={"n_groups": 4})
    with pytest.raises(RAFPublicationError) as exc:
        gated_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert "n_groups" in str(exc.value)


def test_canonical_tree_must_cover_both_registered_rooms(
        gated_md, runtime_root, tmp_path):
    _publish_tree(tmp_path, runtime_root, rooms=(ROOM,))
    with pytest.raises(RAFPublicationError) as exc:
        gated_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert "EmptyRoom" in str(exc.value) or "rooms" in str(exc.value)


def test_publication_check_runs_once_per_process(gated_md, runtime_root, tmp_path,
                                                 monkeypatch):
    """Bounded by cost: per-process, not per item."""
    _publish_tree(tmp_path, runtime_root)
    calls = []
    real_verify = gated_md._verify_publication

    def counting(root):
        calls.append(root)
        return real_verify(root)

    monkeypatch.setattr(gated_md, "_verify_publication", counting)
    for cid in ("000000", "000001", "000002", "000003"):
        gated_md.get_custom_metadata(_info(runtime_root, cid), None)
    assert len(calls) == 1
    assert gated_md._PUBLICATION_CHECKED


def test_a_second_worker_process_repeats_the_check(runtime_root, tmp_path):
    _publish_tree(tmp_path, runtime_root)
    first, second = load_raf_md(test_mode=False), load_raf_md(test_mode=False)
    assert first._PUBLICATION_CHECKED == {}
    first.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert first._PUBLICATION_CHECKED != {}
    # a freshly exec'd module -- i.e. another dataloader worker -- has its own cache
    assert second._PUBLICATION_CHECKED == {}




# --------------------------------------------------------------------------- #
# r6 F2: the failure reaches the caller through the REAL loader path
# --------------------------------------------------------------------------- #
def _raf_dataset_config(runtime_root, split_path):
    return {
        "dataset_type": "audio_dir",
        "datasets": [{
            "id": "RAF",
            "path": str(runtime_root),
            "json_file_path": str(split_path),
            "custom_metadata_module": _RAF_MD_PATH,
            "folder_name": "mono_rirs_22050Hz",
        }],
        "random_crop": False, "augs": False, "force_channels": "mono",
        "is_eval": True, "drop_last": False,
        "modalities": _modalities(deterministic=True),
    }


def _split_file(tmp_path, n=4):
    path = tmp_path / "eval_base.json"
    with open(path, "w") as f:
        json.dump({ROOM: [f"{i:06d}.wav" for i in range(n)]}, f)
    return path


def test_publication_failure_propagates_through_sample_dataset(runtime_root, tmp_path):
    """F2: the r5 gate raised inside __getitem__'s catch-all, so a RAF-only config
    recursed into an unrelated RecursionError and a mixed config silently served
    another dataset's items. The dedicated type is re-raised instead."""
    from src.data.dataset import create_dataloader_from_config

    config = _raf_dataset_config(runtime_root, _split_file(tmp_path))
    loader = create_dataloader_from_config(config, batch_size=2, sample_size=10240,
                                           sample_rate=22050, audio_channels=1,
                                           num_workers=1, shuffle=False)
    with pytest.raises(RAFPublicationError) as exc:
        loader.dataset[0]
    assert "raf_publication.json" in str(exc.value)


def test_publication_failure_is_not_swallowed_by_the_substitution_handler(
        runtime_root, tmp_path):
    """An invalid POINTER (not a missing one) on the production path."""
    from src.data.dataset import create_dataloader_from_config

    (runtime_root / "raf_publication.json").write_text('{"not": "a pointer"}')
    config = _raf_dataset_config(runtime_root, _split_file(tmp_path))
    loader = create_dataloader_from_config(config, batch_size=2, sample_size=10240,
                                           sample_rate=22050, audio_channels=1,
                                           num_workers=1, shuffle=False)
    # the dataset is what the training loop indexes; a worker would re-raise the
    # same type, but this keeps the assertion in-process and deterministic
    with pytest.raises(RAFPublicationError) as exc:
        loader.dataset[0]
    assert "not a valid publication pointer" in str(exc.value)


def test_the_handler_still_substitutes_for_ordinary_load_failures(runtime_root,
                                                                  tmp_path):
    """The re-raise is narrow: everything else keeps the upstream behaviour, which
    is what keeps AR/HAA untouched.

    The corrupted capture is a TEST-role one (>= N_SUPPORT), so it is nobody's
    acoustic context. Corrupting a support capture instead poisons every item that
    draws it -- upstream behaviour, and how this test first found its own footgun.
    """
    from src.data.dataset import LocalDatasetConfig, SampleDataset

    _publish_tree(tmp_path, runtime_root)
    ids = [f"{i:06d}" for i in range(N_SUPPORT, N_SUPPORT + 4)]
    split_path = tmp_path / "eval_base.json"
    with open(split_path, "w") as f:
        json.dump({ROOM: [f"{cid}.wav" for cid in ids]}, f)
    config = LocalDatasetConfig(
        id="RAF", path=str(runtime_root),
        custom_metadata_fn=load_raf_md(test_mode=False).get_custom_metadata,
        json_file_path=str(split_path), folder_name="mono_rirs_22050Hz",
        conditioning=_modalities(deterministic=True))
    dataset = SampleDataset([config], sample_size=10240, sample_rate=22050,
                            random_crop=False, force_channels="mono", augs=False)
    (runtime_root / ROOM / "mono_rirs_22050Hz" / f"{ids[0]}.wav").write_bytes(b"nope")
    audio, info = dataset[0]                      # substituted, not raised
    assert audio.shape == (1, 10240)
    assert info["relpath"].endswith(".wav")
    assert not info["relpath"].endswith(f"{ids[0]}.wav")


def test_the_exception_type_is_raised_only_by_the_raf_hook():
    """AR/HAA cannot reach it: nothing else in the tree raises it."""
    import subprocess

    hits = subprocess.run(
        ["grep", "-rln", "RAFPublicationError", "--include=*.py",
         os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "data")],
        capture_output=True, text=True).stdout.split()
    non_test = sorted(h for h in hits if "/tests/" not in h)
    assert [os.path.basename(h) for h in non_test] == ["RAF_md.py", "dataset.py"]
    # dataset.py DEFINES and re-raises it; RAF_md is the only producer
    from src.data.dataset import RAFPublicationError as ExcType

    assert issubclass(ExcType, Exception)
    md = load_raf_md(test_mode=False)
    assert md._publication_error_type() is ExcType


# --------------------------------------------------------------------------- #
# r6 F1: the pointer must vouch for THIS tree, and only for a canonical one
# --------------------------------------------------------------------------- #
def test_a_non_canonical_pointer_is_refused_in_production(gated_md, runtime_root,
                                                          tmp_path):
    """r5 finding 1: canonical was read FROM the pointer, so a registered RAF
    config would consume a publication that declared itself non-canonical -- and
    the r5 positive test exercised exactly that fail-open case."""
    _publish_tree(tmp_path, runtime_root, canonical=False)
    with pytest.raises(RAFPublicationError) as exc:
        gated_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert "NON-CANONICAL" in str(exc.value)


def test_a_pointer_copied_from_another_valid_tree_is_refused(gated_md, runtime_root,
                                                             tmp_path):
    """The stale/copied-pointer redirect: tree B is genuinely published, and its
    pointer is dropped into tree A. Verification must not authenticate B while the
    loader goes on reading A."""
    other = tmp_path / "other_runtime"
    (other / ROOM / "metadata").mkdir(parents=True)
    _publish_tree(tmp_path / "other", other)          # a genuinely published tree B

    with open(other / "raf_publication.json") as f:
        stolen = json.load(f)
    _publish_tree(tmp_path, runtime_root)             # tree A, published properly
    with open(runtime_root / "raf_publication.json", "w") as f:
        json.dump(stolen, f)                          # ... now carrying B's pointer

    with pytest.raises(RAFPublicationError) as exc:
        gated_md.get_custom_metadata(_info(runtime_root, "000000"), None)
    message = str(exc.value)
    assert "not the tree being loaded" in message or "manifest" in message


def test_the_output_dir_check_is_by_inode_not_by_string(gated_md, runtime_root,
                                                        tmp_path):
    """A symlinked runtime root is the same directory; a different path is not."""
    _publish_tree(tmp_path, runtime_root)
    link = tmp_path / "linked_runtime"
    link.symlink_to(runtime_root, target_is_directory=True)
    info = _info(runtime_root, "000000")
    info["path"] = info["path"].replace(str(runtime_root), str(link))
    gated_md.get_custom_metadata(info, None)          # same inode -> accepted
