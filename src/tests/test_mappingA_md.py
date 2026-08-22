"""RAF_A_md: AR_md semantics on RAF (exp_21, contract D, cycles 8-9).

The formulas are AR_md's, verbatim -- listener-centred frame, PER-CONTEXT own-rx
subtraction, depth at the target receiver. The per-context part is the one a
"nominal microphone position" shortcut gets quietly wrong, so it is checked against
a hand computation that uses each context's own recorded receiver.
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

from src.data.dataset import RAFPublicationError  # noqa: E402

_MD_PATH = os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs",
                        "custom_metadata", "RAF_A_md.py")
ROOM = "EmptyRoom"
K = 8


def load_md(test_mode=True):
    """Load RAF_A_md the way create_dataloader_from_config does."""
    spec = importlib.util.spec_from_file_location("mappingA_metadata_module", _MD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._RAF_A_MD_TEST_MODE = bool(test_mode)
    return mod


@pytest.fixture
def md_module():
    return load_md()


def _depth_map(seed=0, h=256, w=512):
    rows = np.linspace(1.0, 2.0, h, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 0.5, w, dtype=np.float32)[None, :]
    return (rows + cols + float(seed)).astype(np.float32)


def _wave(seed, n=12000):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=n) * 0.01).astype(np.float32)


@pytest.fixture
def runtime_root(tmp_path):
    """<root>/mappingA/<Room>/{mono_rirs_22050Hz, metadata, depth_images}."""
    root = tmp_path / "mappingA"
    room = root / ROOM
    (room / "mono_rirs_22050Hz").mkdir(parents=True)
    (room / "metadata").mkdir()
    (room / "depth_images").mkdir()

    items = {}
    for slot in range(3):
        target = f"{slot:06d}"
        # every capture the item needs, written as audio
        context = []
        for j in range(K):
            capture = f"{100 + slot * K + j:06d}"
            sf.write(str(room / "mono_rirs_22050Hz" / f"{capture}.wav"),
                     _wave(1000 + slot * K + j), 22050, subtype="FLOAT")
            context.append({
                "capture_id": capture, "group_key": f"gC{j}",
                "xyz_key": f"{j}.000000,1.500000,0.000000",
                # each context has its OWN receiver, a few mm from the target's
                "tx_p": [float(j), 0.5 * j, 1.5],
                "rx_p": [0.10 + 0.001 * j, 0.20, 1.00 + 0.002 * j],
                "rx_displacement_m": 0.002,
            })
        sf.write(str(room / "mono_rirs_22050Hz" / f"{target}.wav"), _wave(slot),
                 22050, subtype="FLOAT")
        depth_file = f"{ROOM}_p000_slot{slot:02d}_depth_image.npy"
        np.save(str(room / "depth_images" / depth_file), _depth_map(slot))
        items[target] = {
            "item_id": f"{ROOM}/p000/slot{slot:02d}",
            "room": ROOM, "placement_id": "p000", "mic_slot": slot,
            "target_capture_id": target, "target_group_key": "gT",
            "target_xyz_key": "99.000000,1.500000,99.000000",
            "tx_p": [9.0, 8.0, 1.5],
            "rx_target_p": [0.1, 0.2, 1.0],
            "rx_target_height_raf_m": 1.0,
            "depth_file": depth_file,
            "context": context,
            "match": {"p95_m": 0.004, "max_m": 0.008, "min_ambiguity_margin": 40.0},
        }
    with open(room / "metadata" / "mappingA_metadata.json", "w") as f:
        json.dump(items, f)
    return root


def _modalities(k=K, max_len=9600, poses=True, depth=True, context=True):
    return {
        "acoustic_context": {"load": context, "max_context": k, "max_len": max_len,
                             "deterministic": True},
        "depth": {"load": depth},
        "poses": {"load": poses},
    }


def _info(runtime_root, capture_id, modalities=None):
    rel = os.path.join(ROOM, "mono_rirs_22050Hz", f"{capture_id}.wav")
    return {"path": os.path.join(str(runtime_root), rel), "relpath": rel,
            "modalities": _modalities() if modalities is None else modalities}


# --------------------------------------------------------------------------- #
# AR_md semantics
# --------------------------------------------------------------------------- #
def test_source_is_the_listener_centred_source_position(md_module, runtime_root):
    """AR_md: source = tx - rx_target (the frame is on the LISTENER)."""
    md = md_module.get_custom_metadata(_info(runtime_root, "000000"), None)
    expected = np.array([9.0, 8.0, 1.5]) - np.array([0.1, 0.2, 1.0])
    assert md["source"].shape == (3,) and md["source"].dtype == torch.float32
    np.testing.assert_allclose(md["source"].numpy(), expected, atol=1e-6)
    assert md["source_vit"].shape == (1, 3)
    np.testing.assert_allclose(md["source_vit"].numpy()[0], expected, atol=1e-6)


def test_each_context_subtracts_its_own_receiver(md_module, runtime_root):
    """The M3 formula, and the one a nominal-mic shortcut gets wrong: context j
    uses ITS OWN capture's rx, which differs from the target's by millimetres."""
    md = md_module.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert md["context_poses"].shape == (K, 3)
    for j in range(K):
        own_rx = np.array([0.10 + 0.001 * j, 0.20, 1.00 + 0.002 * j])
        expected = np.array([float(j), 0.5 * j, 1.5]) - own_rx
        np.testing.assert_allclose(md["context_poses"][j].numpy(), expected, atol=1e-6)
    # ... and that is NOT what subtracting the target's receiver would give
    target_rx = np.array([0.1, 0.2, 1.0])
    naive = np.array([[float(j), 0.5 * j, 1.5] for j in range(K)]) - target_rx
    assert not np.allclose(md["context_poses"].numpy(), naive)


def test_context_poses_vit_mirrors_context_poses(md_module, runtime_root):
    md = md_module.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert torch.equal(md["context_poses_vit"], md["context_poses"])


def test_context_audio_shape_and_crop(md_module, runtime_root):
    md = md_module.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert md["context_audio"].shape == (K, 1, 9600)
    assert md["context_audio"].dtype == torch.float32
    padded = md_module.get_custom_metadata(
        _info(runtime_root, "000000", _modalities(max_len=20000)), None)
    assert padded["context_audio"].shape == (K, 1, 20000)
    assert torch.all(padded["context_audio"][:, :, 12000:] == 0)


def test_the_context_is_the_manifest_not_a_draw(md_module, runtime_root):
    """Every arm and seed must condition on identical references, so the context is
    fixed in the manifest rather than drawn at load time."""
    np.random.seed(0)
    torch.manual_seed(0)
    first = md_module.get_custom_metadata(_info(runtime_root, "000001"), None)
    np.random.seed(7)
    torch.manual_seed(1234)
    second = load_md().get_custom_metadata(_info(runtime_root, "000001"), None)
    assert torch.equal(first["context_capture_ids"], second["context_capture_ids"])
    expected = [100 + 1 * K + j for j in range(K)]
    assert first["context_capture_ids"].tolist() == expected


def test_depth_is_the_target_receiver_panorama(md_module, runtime_root):
    """AR renders at the RECEIVER; Mapping A follows, one map per target mic."""
    import raf_common

    md = md_module.get_custom_metadata(_info(runtime_root, "000002"), None)
    assert md["depth"].shape == (3, 256, 512)
    raw = _depth_map(2)
    expected = raw[..., None] * raf_common.equirect_directions()
    np.testing.assert_allclose(md["depth"].numpy(), np.transpose(expected, (2, 0, 1)),
                               rtol=1e-6, atol=0.0)
    # no flipud: row 0 is the zenith
    assert md["depth"][2, 0, :].min() > 0
    assert md["depth"][2, -1, :].max() < 0


def test_provenance_identifies_the_item(md_module, runtime_root):
    md = md_module.get_custom_metadata(_info(runtime_root, "000001"), None)
    assert md["sample_target_id"].dtype == torch.int64
    assert int(md["sample_target_id"]) == 1
    assert md["context_capture_ids"].shape == (K,)
    assert int(md["sample_target_id"]) not in md["context_capture_ids"].tolist()
    assert md["scene"] == ROOM


def test_a_context_count_mismatch_fails_closed(md_module, runtime_root):
    with pytest.raises(ValueError) as exc:
        md_module.get_custom_metadata(
            _info(runtime_root, "000000", _modalities(k=4)), None)
    assert "manifest is the contract" in str(exc.value)


def test_an_unknown_capture_fails_closed(md_module, runtime_root):
    with pytest.raises(KeyError):
        md_module.get_custom_metadata(_info(runtime_root, "999999"), None)


def test_a_malformed_depth_map_fails_closed(md_module, runtime_root):
    path = runtime_root / ROOM / "depth_images" / f"{ROOM}_p000_slot00_depth_image.npy"
    np.save(str(path), _depth_map(0)[:128])
    with pytest.raises(ValueError) as exc:
        md_module.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert "RAF depth map contract" in str(exc.value)


# --------------------------------------------------------------------------- #
# the publication gate (cycle 9)
# --------------------------------------------------------------------------- #
def _publish_mappingA(tmp_path, runtime_root, flavor="mappingA", canonical=True):
    import publish as raf_publish

    split_dir = tmp_path / "splits_mappingA"
    # r2 N5: every registered digest except the audio union is PINNED, and the
    # pointer must agree with the markers it points at.
    parameters = dict(raf_publish.CANONICAL_MAPPINGA_PREPARE_PARAMS,
                      audio_union_sha256="b" * 64)
    pointer = {"split_dir": str(split_dir.resolve()),
               "output_dir": str(runtime_root.resolve()),
               "rooms": list(raf_publish.CANONICAL_ROOMS),
               "flavor": flavor, "canonical": canonical, "taint": [],
               "parameters": parameters,
               "readback_record": {"sha256": raf_publish.canonical_record_digest()}}
    extra = {"canonical": canonical, "taint": [], "canonical_parameters": True,
             "parameters": parameters,
             "readback_record": {"sha256": raf_publish.canonical_record_digest()}}
    depth_extra = dict(extra,
                       parameters=dict(raf_publish.CANONICAL_MAPPINGA_DEPTH_PARAMS))
    with raf_publish.PublishTransaction(str(split_dir), kind="mappingA_prepare") as txn:
        runtime = txn.stage(str(runtime_root))
        splits = txn.stage(str(split_dir))
        with open(runtime.path("raf_publication.json"), "w") as f:
            json.dump(pointer, f)
        with open(splits.path("mappingA_eval.json"), "w") as f:
            json.dump({ROOM: []}, f)
        txn.commit(extra=extra)
    with raf_publish.PublishTransaction(str(runtime_root), kind="mappingA_depth") as txn:
        for room in raf_publish.CANONICAL_ROOMS:
            staged = txn.stage(str(runtime_root / room / "depth_images"))
            with open(staged.path("attested.txt"), "w") as f:
                f.write("depth")
        txn.commit(extra=depth_extra)


def test_the_gate_is_mandatory_and_has_no_environment_switch():
    import inspect

    source = inspect.getsource(load_md(test_mode=False))
    assert "os.environ" not in source
    assert load_md(test_mode=False)._RAF_A_MD_TEST_MODE is False


def test_the_gate_refuses_an_unpublished_tree(runtime_root):
    gated = load_md(test_mode=False)
    with pytest.raises(RAFPublicationError) as exc:
        gated.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert "raf_publication.json" in str(exc.value)


def test_the_gate_accepts_a_published_mappingA_tree(tmp_path, runtime_root):
    _publish_mappingA(tmp_path, runtime_root)
    gated = load_md(test_mode=False)
    assert gated.get_custom_metadata(_info(runtime_root, "000000"), None)["scene"] == ROOM


def test_the_gate_refuses_a_mapping_h_tree(tmp_path, runtime_root):
    """A Mapping-A config may not consume the Mapping-H publication: the items,
    the conditioning and the depth convention are all different."""
    _publish_mappingA(tmp_path, runtime_root, flavor="mappingH")
    gated = load_md(test_mode=False)
    with pytest.raises(RAFPublicationError) as exc:
        gated.get_custom_metadata(_info(runtime_root, "000000"), None)
    assert "mappingA" in str(exc.value)


def test_the_gate_refuses_a_non_canonical_tree(tmp_path, runtime_root):
    _publish_mappingA(tmp_path, runtime_root, canonical=False)
    gated = load_md(test_mode=False)
    with pytest.raises(RAFPublicationError):
        gated.get_custom_metadata(_info(runtime_root, "000000"), None)


def test_the_gate_runs_once_per_process(tmp_path, runtime_root, monkeypatch):
    _publish_mappingA(tmp_path, runtime_root)
    gated = load_md(test_mode=False)
    calls = []
    real = gated._verify_publication
    monkeypatch.setattr(gated, "_verify_publication",
                        lambda root: (calls.append(root), real(root))[1])
    for capture in ("000000", "000001", "000002"):
        gated.get_custom_metadata(_info(runtime_root, capture), None)
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# cycle 9: the real conditioner, vanilla and fa_invariant, on Mapping-A metadata
# --------------------------------------------------------------------------- #
def _raf_conditioning_config():
    """The shipped RAF conditioning block with a LIGHTWEIGHT ViT injected.

    Only the DINOv3 backbone is swapped for the in-tree SimpleViT branch (a network
    download and 21M parameters have no place in a unit test); ids, cond_dim,
    dist_embedder frequencies and the RIR encoder are the real config, so the shapes
    asserted are the shapes the run produces.
    """
    path = os.path.join(_REPO_ROOT, "src", "configs", "model_configs", "FLAC", "RAF",
                        "FLAC_RAF_finetune.json")
    with open(path) as f:
        model_config = json.load(f)
    conditioning = model_config["model"]["conditioning"]
    for entry in conditioning["configs"]:
        if entry["type"] == "ViTCoordinates":
            entry["config"]["ViT"] = {"ch_dim": 3, "img_h": 256, "img_w": 512,
                                      "patch_h": 64, "patch_w": 64, "dim": 32,
                                      "depth": 1, "heads": 2, "mlp_dim": 32}
    return model_config, conditioning


def _mappingA_batch(runtime_root, tmp_path, n=2):
    from src.data.dataset import LocalDatasetConfig, SampleDataset, collation_fn

    split = {ROOM: [f"{i:06d}.wav" for i in range(n)]}
    split_path = tmp_path / "mappingA_eval.json"
    with open(split_path, "w") as f:
        json.dump(split, f)
    config = LocalDatasetConfig(
        id="RAF", path=str(runtime_root), custom_metadata_fn=load_md().get_custom_metadata,
        json_file_path=str(split_path), folder_name="mono_rirs_22050Hz",
        conditioning=_modalities())
    dataset = SampleDataset([config], sample_size=10240, sample_rate=22050,
                            random_crop=False, force_channels="mono", augs=False)
    return collation_fn([dataset[i] for i in range(n)])


def test_batch_shapes_through_the_real_collation_path(runtime_root, tmp_path):
    audio, metadata = _mappingA_batch(runtime_root, tmp_path, n=3)
    assert audio.shape == (3, 1, 10240)
    for md in metadata:
        assert md["source"].shape == (3,) and md["source_vit"].shape == (1, 3)
        assert md["context_poses"].shape == (K, 3)
        assert md["context_audio"].shape == (K, 1, 9600)
        assert md["depth"].shape == (3, 256, 512)
        assert torch.isfinite(md["depth"]).all()
    # the ids prove no silent substitution occurred
    assert [int(md["sample_target_id"]) for md in metadata] == [0, 1, 2]


def test_the_real_conditioner_consumes_a_mappingA_batch_vanilla(runtime_root, tmp_path):
    from src.models.conditioners import create_multi_conditioner_from_conditioning_config

    _, conditioning = _raf_conditioning_config()
    conditioner = create_multi_conditioner_from_conditioning_config(conditioning)
    _, metadata = _mappingA_batch(runtime_root, tmp_path, n=2)
    with torch.no_grad():
        out = conditioner(list(metadata), device="cpu")
    assert set(out) == {"source", "source_vit", "context_poses_vit", "context_poses",
                        "context_audio"}
    assert out["source"][0].shape == (2, 1, 256)
    assert out["context_poses"][0].shape == (2, K, 256)
    assert out["context_audio"][0].shape == (2, K, 256)
    for tensor, _mask in out.values():
        assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()


def test_the_real_conditioner_consumes_a_mappingA_batch_fa_invariant(runtime_root,
                                                                     tmp_path):
    """BF's arm runs cond_method=fa_invariant, so the FA machinery has to accept
    Mapping-A metadata -- listener-centred panoramas are its AR-native case."""
    from src.data import yaw_rotation as yr
    from src.models.conditioners import create_multi_conditioner_from_conditioning_config

    _, conditioning = _raf_conditioning_config()
    conditioner = create_multi_conditioner_from_conditioning_config(conditioning)
    _, metadata = _mappingA_batch(runtime_root, tmp_path, n=2)
    with torch.no_grad():
        out = yr.invariant_conditioning(conditioner, list(metadata), "cpu")
    assert set(out) >= {"source", "source_vit", "context_poses_vit", "context_poses",
                        "context_audio"}
    assert out["source_vit"][0].shape == (2, 1, 256)
    for key in ("source_vit", "context_poses_vit"):
        assert torch.isfinite(out[key][0]).all()


def test_fa_conditioning_is_c4_invariant_on_mappingA_metadata(runtime_root, tmp_path):
    """The invariance that licenses the BF arm: rotating the whole scene by a
    multiple of 90 degrees must leave the frame-averaged conditioning unchanged."""
    import math

    from src.data import yaw_rotation as yr
    from src.models.conditioners import create_multi_conditioner_from_conditioning_config

    _, conditioning = _raf_conditioning_config()
    conditioner = create_multi_conditioner_from_conditioning_config(conditioning)
    _, metadata = _mappingA_batch(runtime_root, tmp_path, n=2)
    with torch.no_grad():
        base = yr.invariant_conditioning(conditioner, list(metadata), "cpu")
        for degrees in (90.0, 180.0, 270.0):
            rotated = [yr.rotate_scene_metadata(dict(md), math.radians(degrees), 512)
                       for md in metadata]
            out = yr.invariant_conditioning(conditioner, rotated, "cpu")
            for key in ("source_vit", "context_poses_vit"):
                torch.testing.assert_close(out[key][0], base[key][0],
                                           rtol=1e-4, atol=1e-5)
