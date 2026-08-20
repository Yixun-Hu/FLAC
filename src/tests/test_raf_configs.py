"""Tests for the RAF model + dataset configs (exp_19, contract section E, cycle 11).

The RAF configs are clones of the HAA ones. "Clone" is enforced literally: each
test diffs the RAF config against its HAA template and asserts that the set of
differing key paths is EXACTLY the registered whitelist — an accidental extra
delta (a changed lr, a dropped conditioner) would otherwise ride along invisibly
into a finetune whose whole claim is HAA-protocol parity.
"""
import json
import os

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_DIR = os.path.join(_REPO_ROOT, "src", "configs", "model_configs", "FLAC")
_DATA_DIR = os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs")


def _load(path):
    with open(path) as f:
        return json.load(f)


def _flatten(obj, prefix=""):
    """Flatten nested dicts/lists into {dotted path: leaf value}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _diff(template, candidate):
    a, b = _flatten(template), _flatten(candidate)
    return {
        "added": sorted(set(b) - set(a)),
        "removed": sorted(set(a) - set(b)),
        "changed": sorted(k for k in set(a) & set(b) if a[k] != b[k]),
    }


# --------------------------------------------------------------------------- #
# model config
# --------------------------------------------------------------------------- #
_HAA_MODEL = os.path.join(_MODEL_DIR, "HAA", "FLAC_HAA_finetune.json")
_RAF_MODEL = os.path.join(_MODEL_DIR, "RAF", "FLAC_RAF_finetune.json")


def test_model_config_differs_from_the_haa_template_only_by_the_whitelist():
    diff = _diff(_load(_HAA_MODEL), _load(_RAF_MODEL))
    assert diff["added"] == ["training.cond_method"]
    assert diff["removed"] == []
    assert diff["changed"] == [
        "training.metrics.AGREE_ckpt",
        "training.metrics.dataset_name",
        "training.metrics.eval_FD",
        "training.metrics.eval_retrieval",
    ]


def test_model_config_delta_values():
    cfg = _load(_RAF_MODEL)
    assert cfg["training"]["cond_method"] == "vanilla"
    metrics = cfg["training"]["metrics"]
    assert metrics["dataset_name"] == "RAF"
    assert metrics["eval_FD"] is False
    assert metrics["eval_retrieval"] is False
    # No AGREE model was ever trained on RAF, so FD/Recall are unavailable — never
    # reported as zero (plan Rev 2 section 7, C14).
    assert metrics["AGREE_ckpt"] is None


def test_model_config_keeps_the_finetune_recipe():
    cfg = _load(_RAF_MODEL)
    opt = cfg["training"]["optimizer_configs"]["diffusion"]
    assert opt["optimizer"]["config"]["lr"] == 5e-6
    assert opt["scheduler"]["type"] == "InverseLR"
    assert cfg["training"]["cfg_dropout_prob"] == 0.1
    assert cfg["training"]["use_ema"] is True
    assert cfg["sample_size"] == 10240 and cfg["sample_rate"] == 22050


# --------------------------------------------------------------------------- #
# dataset configs
# --------------------------------------------------------------------------- #
_PAIRS = [
    ("train/raf_train.json", "train/haa_train.json", "data/RAF/train_base.json", False),
    ("eval/raf_val.json", "eval/haa_val.json", "data/RAF/val_base.json", True),
    ("eval/raf_test.json", "eval/haa_test.json", "data/RAF/test_base.json", True),
    # the HAA-parity diagnostic row (r2 R2): its own manifest, same eval protocol
    ("eval/raf_diagnostic.json", "eval/haa_test.json",
     "data/RAF/diagnostic_base.json", True),
]

_DATASET_WHITELIST = [
    "datasets[0].custom_metadata_module",
    "datasets[0].id",
    "datasets[0].json_file_path",
    "datasets[0].path",
]


def test_dataset_configs_differ_from_their_haa_templates_only_by_the_whitelist():
    for raf_rel, haa_rel, _, _ in _PAIRS:
        raf = _load(os.path.join(_DATA_DIR, "RAF", raf_rel))
        haa = _load(os.path.join(_DATA_DIR, "HAA", haa_rel))
        diff = _diff(haa, raf)
        assert diff["added"] == ["modalities.acoustic_context.deterministic"], raf_rel
        assert diff["removed"] == [], raf_rel
        assert diff["changed"] == _DATASET_WHITELIST, raf_rel


def test_dataset_config_values():
    for raf_rel, _, json_file_path, deterministic in _PAIRS:
        cfg = _load(os.path.join(_DATA_DIR, "RAF", raf_rel))
        entry = cfg["datasets"][0]
        assert entry["id"] == "RAF"
        assert entry["path"] == "RAF"
        assert entry["json_file_path"] == json_file_path
        assert entry["folder_name"] == "mono_rirs_22050Hz"
        assert entry["custom_metadata_module"] == \
            "src/configs/dataset_configs/custom_metadata/RAF_md.py"
        assert cfg["modalities"]["acoustic_context"]["max_context"] == 8
        assert cfg["modalities"]["acoustic_context"]["max_len"] == 9600
        assert cfg["modalities"]["acoustic_context"]["deterministic"] is deterministic
        assert cfg["force_channels"] == "mono"
        assert cfg["augs"] is False and cfg["random_crop"] is False


def test_eval_configs_keep_the_haa_eval_flags():
    val = _load(os.path.join(_DATA_DIR, "RAF", "eval/raf_val.json"))
    test = _load(os.path.join(_DATA_DIR, "RAF", "eval/raf_test.json"))
    diagnostic = _load(os.path.join(_DATA_DIR, "RAF", "eval/raf_diagnostic.json"))
    for cfg in (val, test, diagnostic):
        assert cfg["is_eval"] is True
        assert cfg["drop_last"] is False
    assert val["seeneval"] is True
    assert test["unseeneval"] is True


def test_referenced_metadata_module_exists():
    for raf_rel, _, _, _ in _PAIRS:
        cfg = _load(os.path.join(_DATA_DIR, "RAF", raf_rel))
        module = cfg["datasets"][0]["custom_metadata_module"]
        assert os.path.isfile(os.path.join(_REPO_ROOT, module))


# --------------------------------------------------------------------------- #
# r2 R11: structural clone check (deep-copy the template, patch, compare exactly)
# --------------------------------------------------------------------------- #
import copy  # noqa: E402


def assert_same_structure(a, b, path="$"):
    """Recursive node-by-node comparison: type, container length, and leaf value.

    Stronger than comparing flattened dotted paths, which R11 showed is
    incomplete: a flatten emits nothing for an empty dict or list, so a stray
    ``"extra": {}`` rides along invisibly, and a JSON key containing a dot or a
    bracket can collide with a generated path. Walking the nodes also pins
    container TYPES (``[] != {}``) and numeric types (``1`` is not ``1.0``, which
    plain ``==`` accepts).
    """
    assert type(a) is type(b), f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        assert set(a) == set(b), (
            f"{path}: keys differ (only expected: {sorted(set(a) - set(b))}, "
            f"only actual: {sorted(set(b) - set(a))})")
        for key in a:
            assert_same_structure(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same_structure(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _expected_model_config():
    """The HAA finetune config plus EXACTLY the registered RAF deltas."""
    cfg = copy.deepcopy(_load(_HAA_MODEL))
    cfg["training"]["cond_method"] = "vanilla"            # added
    cfg["training"]["metrics"]["dataset_name"] = "RAF"    # changed
    cfg["training"]["metrics"]["eval_FD"] = False         # changed
    cfg["training"]["metrics"]["eval_retrieval"] = False  # changed
    cfg["training"]["metrics"]["AGREE_ckpt"] = None       # changed (no AGREE-RAF)
    return cfg


def _expected_dataset_config(haa_rel, json_file_path, deterministic):
    cfg = copy.deepcopy(_load(os.path.join(_DATA_DIR, "HAA", haa_rel)))
    entry = cfg["datasets"][0]
    entry["id"] = "RAF"
    entry["path"] = "RAF"
    entry["json_file_path"] = json_file_path
    entry["custom_metadata_module"] = \
        "src/configs/dataset_configs/custom_metadata/RAF_md.py"
    cfg["modalities"]["acoustic_context"]["deterministic"] = deterministic
    return cfg


def test_model_config_is_structurally_the_patched_haa_template():
    assert_same_structure(_expected_model_config(), _load(_RAF_MODEL))


def test_dataset_configs_are_structurally_the_patched_haa_templates():
    for raf_rel, haa_rel, json_file_path, deterministic in _PAIRS:
        assert_same_structure(
            _expected_dataset_config(haa_rel, json_file_path, deterministic),
            _load(os.path.join(_DATA_DIR, "RAF", raf_rel)),
            path=f"${raf_rel}")


def test_structural_comparison_catches_what_a_flatten_misses():
    """The two holes R11 named, on the real template."""
    with pytest.raises(AssertionError):     # an empty container rides in
        smuggled = _expected_model_config()
        smuggled["training"]["extra"] = {}
        assert_same_structure(_expected_model_config(), smuggled)
    with pytest.raises(AssertionError):     # a container type changes
        retyped = _expected_model_config()
        retyped["model"]["conditioning"]["configs"] = {}
        assert_same_structure(_expected_model_config(), retyped)
    with pytest.raises(AssertionError):     # a list gets longer
        longer = _expected_model_config()
        longer["model"]["diffusion"]["global_cond_ids"].append("smuggled")
        assert_same_structure(_expected_model_config(), longer)
    with pytest.raises(AssertionError):     # an int silently becomes a float
        retyped_leaf = _expected_model_config()
        retyped_leaf["sample_size"] = float(retyped_leaf["sample_size"])
        assert_same_structure(_expected_model_config(), retyped_leaf)


def test_referenced_split_files_are_declared_but_not_required_yet():
    """Split manifests are generated at the run rung (after the readback gate), so
    their absence before a prep run is legitimate; what must hold now is that every
    config points into data/RAF and that the four paths are distinct."""
    paths = []
    for raf_rel, _, json_file_path, _ in _PAIRS:
        cfg = _load(os.path.join(_DATA_DIR, "RAF", raf_rel))
        assert cfg["datasets"][0]["json_file_path"] == json_file_path
        assert json_file_path.startswith("data/RAF/")
        paths.append(json_file_path)
    assert len(set(paths)) == len(paths)
