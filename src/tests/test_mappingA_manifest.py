"""Mapping-A item manifest + static validator (exp_21, contract B3/B4, cycle 5).

M5: the manifest is what every arm and seed conditions on, so its invariants are
checked statically before publication rather than hoped for at eval time. Each
condition below corresponds to a way the experiment could quietly stop measuring
what it claims: a repeated target, a context that contains the answer, a context
source at the target's own position, or a "same microphone" that is not.
"""
import json
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import mappingA_common as mac  # noqa: E402
import prepare_mappingA as prep_a  # noqa: E402


def _context(i, xyz=None):
    # r2 N6: the record is now RECOMPUTED, so the fixture has to be self-consistent
    # -- rx_p is 2 mm from the target mic and rx_displacement_m says so.
    return {"capture_id": f"{900 + i:06d}", "group_key": f"gC{i}",
            "xyz_key": mac.source_xyz_key(xyz or (float(i), 1.5, 0.0)),
            "tx_p": [float(i), 0.0, 1.5], "rx_p": [0.1, 0.202, 1.0],
            "rx_row": i, "rx_displacement_m": 0.002,
            "match": {"p95_m": 0.003, "max_m": 0.006, "min_ambiguity_margin": 30.0}}


def _valid_item(room="EmptyRoom", placement="p000", slot=0, target="000001", k=8):
    return {
        "item_id": f"{room}/{placement}/slot{slot:02d}",
        "room": room, "placement_id": placement, "mic_slot": slot,
        "target_capture_id": target, "target_group_key": "gT",
        "target_xyz_key": mac.source_xyz_key((99.0, 1.5, 99.0)),
        "tx_p": [99.0, 99.0, 1.5], "rx_target_p": [0.1, 0.2, 1.0],
        "rx_target_row": slot, "rx_target_height_raf_m": 1.0,
        "depth_file": f"{room}_{placement}_slot{slot:02d}_depth_image.npy",
        "context": [_context(i) for i in range(k)],
        "match": {"p95_m": 0.004, "max_m": 0.008, "min_ambiguity_margin": 40.0},
    }


def _manifest(n_items=4, k=8):
    items = []
    for slot in range(n_items):
        items.append(_valid_item(slot=slot, target=f"{slot + 1:06d}", k=k))
    return {"items": items, "k": k, "n_items": len(items)}


# --------------------------------------------------------------------------- #
# the validator's accept case
# --------------------------------------------------------------------------- #
def test_a_well_formed_manifest_validates():
    report = prep_a.validate_manifest(_manifest(), expected_items=4, k=8)
    assert report["passed"] is True
    assert report["n_items"] == 4
    assert report["n_unique_targets"] == 4
    assert report["violations"] == []
    json.dumps(report)


def test_the_expected_item_count_is_enforced():
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(_manifest(n_items=3), expected_items=4, k=8)
    assert "3" in str(exc.value) and "4" in str(exc.value)


def test_the_canonical_shape_is_the_planned_one():
    assert prep_a.CANONICAL_N_ITEMS == 1152
    assert prep_a.CANONICAL_N_PLACEMENTS == 16
    assert prep_a.CANONICAL_K == 8
    assert prep_a.CANONICAL_N_PLACEMENTS * 36 * 2 == prep_a.CANONICAL_N_ITEMS


# --------------------------------------------------------------------------- #
# M5 conditions, one failure mode each
# --------------------------------------------------------------------------- #
def test_a_repeated_target_capture_is_refused():
    manifest = _manifest()
    manifest["items"][1]["target_capture_id"] = manifest["items"][0]["target_capture_id"]
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    assert "duplicate_target" in str(exc.value)


def test_a_repeated_item_slot_is_refused():
    manifest = _manifest()
    manifest["items"][1]["mic_slot"] = manifest["items"][0]["mic_slot"]
    manifest["items"][1]["item_id"] = manifest["items"][0]["item_id"]
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    assert "duplicate_item" in str(exc.value)


def test_the_wrong_number_of_contexts_is_refused():
    manifest = _manifest(k=8)
    manifest["items"][2]["context"] = manifest["items"][2]["context"][:6]
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    assert "6" in str(exc.value)


def test_a_repeated_context_capture_within_an_item_is_refused():
    """Eight slots holding six distinct references is not K=8 conditioning."""
    manifest = _manifest()
    manifest["items"][0]["context"][3] = dict(manifest["items"][0]["context"][2])
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    assert "distinct" in str(exc.value)


def test_the_target_capture_inside_its_own_context_is_refused():
    manifest = _manifest()
    manifest["items"][0]["context"][0]["capture_id"] = \
        manifest["items"][0]["target_capture_id"]
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    assert "own context" in str(exc.value)


def test_a_context_source_at_the_target_position_is_refused():
    """M5: unseen source POSITION -- a quaternion-only duplicate is the same
    loudspeaker in the same place, and its presence falsifies the whole row."""
    manifest = _manifest()
    manifest["items"][0]["context"][4]["xyz_key"] = \
        manifest["items"][0]["target_xyz_key"]
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    assert "source position" in str(exc.value)


def test_a_context_recorded_at_a_different_microphone_is_refused():
    """The item claims one microphone heard everything; a 5 cm displacement means
    it did not."""
    manifest = _manifest()
    # a genuinely distant microphone: the poses AND the record say 0.05 m
    manifest["items"][1]["context"][2]["rx_p"] = [0.1, 0.25, 1.0]
    manifest["items"][1]["context"][2]["rx_displacement_m"] = 0.05
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    assert "displacement" in str(exc.value)


def test_a_group_whose_correspondence_failed_is_refused():
    manifest = _manifest()
    manifest["items"][3]["match"]["p95_m"] = 0.02
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    assert "p95" in str(exc.value)


def test_an_ambiguous_correspondence_is_refused():
    manifest = _manifest()
    manifest["items"][3]["match"]["min_ambiguity_margin"] = 1.5
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    assert "ambiguous_match" in str(exc.value)
    assert "margin 1.50 < 3.0" in str(exc.value)


def test_every_violation_is_reported_not_just_the_first():
    manifest = _manifest()
    manifest["items"][0]["context"][0]["capture_id"] = \
        manifest["items"][0]["target_capture_id"]
    # a genuinely distant microphone: the poses AND the record say 0.05 m
    manifest["items"][1]["context"][2]["rx_p"] = [0.1, 0.25, 1.0]
    manifest["items"][1]["context"][2]["rx_displacement_m"] = 0.05
    manifest["items"][3]["match"]["min_ambiguity_margin"] = 1.5
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    kinds = {v["kind"] for v in exc.value.report["violations"]}
    assert kinds == {"target_in_context", "mic_displacement", "ambiguous_match"}


def test_the_validator_is_importable_and_reports_json_safely():
    manifest = _manifest()
    manifest["items"][0]["context"][0]["capture_id"] = "000001"
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    json.dumps(exc.value.report, allow_nan=False)


# --------------------------------------------------------------------------- #
# item construction
# --------------------------------------------------------------------------- #
def _pose(key, capture_by_slot, xyz, rx_raw=None):
    # RAF is X front, Y up, Z left; RAF_TO_PIPELINE = (X, Z, Y). The RAW rows are
    # carried alongside the pipeline ones because the tracked height is a raw
    # quantity (r2 N8).
    raw = (rx_raw if rx_raw is not None
           else np.array([[0.1 * s, 1.0, 0.2] for s in range(36)]))
    return {"group_key": key, "tx_xyz": np.array(xyz, dtype=np.float64),
            "tx_xyz_p": np.array([xyz[0], xyz[2], xyz[1]], dtype=np.float64),
            "capture_ids": capture_by_slot,
            "rx_xyz": raw,
            "rx_xyz_p": np.array([[row[0], row[2], row[1]] for row in raw])}


def _placement_poses(n_poses=12, base=0):
    poses = []
    for i in range(n_poses):
        captures = [f"{base + i * 36 + s:06d}" for s in range(36)]
        poses.append(_pose(f"g{base + i}", captures, (float(i), 1.5, 2.0)))
    return poses


def test_build_items_produces_one_item_per_mic_slot():
    poses = _placement_poses()
    items = prep_a.build_items("EmptyRoom", "p000", poses,
                               assignment={p["group_key"]: list(range(36))
                                           for p in poses},
                               match={p["group_key"]: {"p95_m": 0.004, "max_m": 0.008,
                                                       "min_ambiguity_margin": 40.0}
                                      for p in poses}, k=8)
    assert len(items) == 36
    assert {i["mic_slot"] for i in items} == set(range(36))
    assert len({i["target_capture_id"] for i in items}) == 36
    for item in items:
        assert len(item["context"]) == 8
        assert item["target_capture_id"] not in [c["capture_id"] for c in item["context"]]
        assert all(c["xyz_key"] != item["target_xyz_key"] for c in item["context"])


def test_build_items_uses_one_hash_chosen_target_pose_for_the_placement():
    poses = _placement_poses()
    items = prep_a.build_items("EmptyRoom", "p000", poses,
                               assignment={p["group_key"]: list(range(36))
                                           for p in poses},
                               match={p["group_key"]: {"p95_m": 0.004, "max_m": 0.008,
                                                       "min_ambiguity_margin": 40.0}
                                      for p in poses}, k=8)
    assert len({i["target_group_key"] for i in items}) == 1
    assert items[0]["target_group_key"] == mac.select_target(
        "EmptyRoom", "p000", poses)["group_key"]


def test_build_items_validates_against_the_manifest_rules():
    poses = _placement_poses()
    items = prep_a.build_items("EmptyRoom", "p000", poses,
                               assignment={p["group_key"]: list(range(36))
                                           for p in poses},
                               match={p["group_key"]: {"p95_m": 0.004, "max_m": 0.008,
                                                       "min_ambiguity_margin": 40.0}
                                      for p in poses}, k=8)
    report = prep_a.validate_manifest({"items": items, "k": 8}, expected_items=36, k=8)
    assert report["passed"] is True


def test_build_items_needs_enough_source_distinct_poses():
    poses = _placement_poses(n_poses=5)
    with pytest.raises(ValueError):
        prep_a.build_items("EmptyRoom", "p000", poses,
                           assignment={p["group_key"]: list(range(36)) for p in poses},
                           match={p["group_key"]: {"p95_m": 0.004, "max_m": 0.008,
                                                   "min_ambiguity_margin": 40.0}
                                  for p in poses}, k=8)


# --------------------------------------------------------------------------- #
# cycle 10: the eval config
# --------------------------------------------------------------------------- #
def test_the_mappingA_eval_config_points_at_the_mappingA_surface():
    path = os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs", "RAF",
                        "eval", "raf_mappingA.json")
    with open(path) as f:
        config = json.load(f)
    entry = config["datasets"][0]
    assert entry["id"] == "RAF"
    assert entry["path"] == "RAF/mappingA"
    # derived, not spelled out: the config must follow the CLI's split root, which
    # is what actually decides where the manifest is written (N2)
    import prepare_mappingA as prep_a

    assert entry["json_file_path"] == (
        f"{prep_a.MAPPINGA_SPLIT_ROOT}/{prep_a.MANIFEST_NAME}")
    assert entry["custom_metadata_module"] == \
        "src/configs/dataset_configs/custom_metadata/RAF_A_md.py"
    assert entry["folder_name"] == "mono_rirs_22050Hz"
    assert config["modalities"]["acoustic_context"]["deterministic"] is True
    assert config["modalities"]["acoustic_context"]["max_context"] == prep_a.CANONICAL_K
    assert config["expected_items"] == prep_a.CANONICAL_N_ITEMS
    assert config["is_eval"] is True and config["drop_last"] is False


def test_the_mappingA_config_differs_from_the_mappingH_one_only_where_it_must():
    base = os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs", "RAF", "eval")
    with open(os.path.join(base, "raf_test.json")) as f:
        mapping_h = json.load(f)
    with open(os.path.join(base, "raf_mappingA.json")) as f:
        mapping_a = json.load(f)
    changed = {k for k in set(mapping_h) | set(mapping_a)
               if mapping_h.get(k) != mapping_a.get(k)}
    assert changed == {"datasets", "expected_items"}
    h_entry, a_entry = mapping_h["datasets"][0], mapping_a["datasets"][0]
    assert {k for k in h_entry if h_entry[k] != a_entry.get(k)} == {
        "path", "json_file_path", "custom_metadata_module"}
    assert mapping_h["modalities"] == mapping_a["modalities"]


def test_the_referenced_metadata_module_exists():
    assert os.path.isfile(os.path.join(
        _REPO_ROOT, "src", "configs", "dataset_configs", "custom_metadata",
        "RAF_A_md.py"))


# --------------------------------------------------------------------------- #
# r2 N6: the validator RECOMPUTES rather than reading the row's own claims
# --------------------------------------------------------------------------- #
def test_a_recorded_displacement_that_the_poses_contradict_is_refused():
    """The row's own number is not evidence of itself: a stale or doctored
    displacement would certify a "same microphone" the poses deny."""
    manifest = _manifest()
    manifest["items"][0]["context"][0]["rx_displacement_m"] = 0.001   # poses: 0.002
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    kinds = {v["kind"] for v in exc.value.report["violations"]}
    assert kinds == {"displacement_mismatch"}
    assert "0.002" in exc.value.report["violations"][0]["detail"]


def test_a_context_at_the_target_coordinates_is_refused_whatever_its_key_says():
    manifest = _manifest()
    item = manifest["items"][0]
    item["context"][3]["tx_p"] = list(item["tx_p"])          # same source position
    item["context"][3]["xyz_key"] = "not.the.target.key"     # but a different key
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    detail = exc.value.report["violations"][0]["detail"]
    assert "context_source_position" == exc.value.report["violations"][0]["kind"]
    assert "coordinates" in detail


@pytest.mark.parametrize("missing", ["match", "rx_target_row", "mic_slot", "tx_p"])
def test_an_item_missing_any_registered_field_is_refused(missing):
    """No defaults: match.get("p95_m", 0.0) read an ABSENT correspondence record as
    a perfect score, so an item carrying no evidence validated."""
    manifest = _manifest()
    del manifest["items"][2][missing]
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    kinds = {v["kind"] for v in exc.value.report["violations"]}
    assert kinds == {"schema"}
    assert any(missing in v["detail"] for v in exc.value.report["violations"])


@pytest.mark.parametrize("missing", ["match", "rx_row", "rx_p", "group_key"])
def test_a_context_missing_any_registered_field_is_refused(missing):
    manifest = _manifest()
    del manifest["items"][1]["context"][4][missing]
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    assert any(missing in v["detail"] for v in exc.value.report["violations"])


def test_a_context_group_whose_own_correspondence_failed_is_refused():
    """Each context is attested by ITS group's correspondence evidence, not by the
    target's: a context drawn from a group that failed the match is not the same
    microphone, however well the target matched."""
    manifest = _manifest()
    manifest["items"][0]["context"][1]["match"] = {
        "p95_m": 0.05, "max_m": 0.09, "min_ambiguity_margin": 30.0}
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    kinds = {v["kind"] for v in exc.value.report["violations"]}
    assert kinds == {"failed_match"}
    assert "000901" in exc.value.report["violations"][0]["detail"]


def test_a_context_from_the_targets_own_group_is_refused():
    manifest = _manifest()
    manifest["items"][0]["context"][0]["group_key"] = \
        manifest["items"][0]["target_group_key"]
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    assert "target_in_context" in {v["kind"] for v in exc.value.report["violations"]}


@pytest.mark.parametrize("field,value", [("mic_slot", 40), ("rx_target_row", 99)])
def test_a_slot_outside_the_array_is_refused(field, value):
    manifest = _manifest()
    item = manifest["items"][0]
    item[field] = value
    if field == "mic_slot":
        item["item_id"] = f"EmptyRoom/p000/slot{value:02d}"
        item["depth_file"] = f"EmptyRoom_p000_slot{value:02d}_depth_image.npy"
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    assert "mic_slot" in {v["kind"] for v in exc.value.report["violations"]}


def test_the_slot_the_id_and_the_depth_file_must_name_one_slot():
    manifest = _manifest()
    manifest["items"][0]["depth_file"] = "EmptyRoom_p000_slot07_depth_image.npy"
    with pytest.raises(prep_a.ManifestError) as exc:
        prep_a.validate_manifest(manifest, expected_items=4, k=8)
    assert "mic_slot" in {v["kind"] for v in exc.value.report["violations"]}


def test_the_registered_item_count_is_composed_of_registered_constants():
    import publish as raf_publish

    assert prep_a.CANONICAL_N_ITEMS == raf_publish.CANONICAL_MAPPINGA_PREPARE_PARAMS[
        "n_items"]
    assert prep_a.CANONICAL_N_ITEMS == (prep_a.CANONICAL_N_PLACEMENTS
                                        * prep_a.ARRAY_SIZE * 2) == 1152


# --------------------------------------------------------------------------- #
# r2 N6: the config's expected_items reaches the runtime stream gate
# --------------------------------------------------------------------------- #
def _eval_flac():
    import importlib

    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    return importlib.import_module("eval_FLAC")


def test_the_config_item_count_becomes_the_expected_stream_count():
    eval_FLAC = _eval_flac()
    path = os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs", "RAF",
                        "eval", "raf_mappingA.json")
    with open(path) as f:
        config = json.load(f)
    assert config["expected_items"] == prep_a.CANONICAL_N_ITEMS
    assert eval_FLAC.resolve_expected_stream_count(config) == 1152
    # an agreeing flag is fine; a contradicting one is a mistake, not an override
    assert eval_FLAC.resolve_expected_stream_count(config, 1152) == 1152
    with pytest.raises(ValueError) as exc:
        eval_FLAC.resolve_expected_stream_count(config, 6337)
    assert "6337" in str(exc.value) and "1152" in str(exc.value)


def test_a_config_without_an_item_count_leaves_the_flag_alone():
    """AR/HAA configs declare nothing, so their behaviour is byte-identical."""
    eval_FLAC = _eval_flac()
    assert eval_FLAC.resolve_expected_stream_count({}) is None
    assert eval_FLAC.resolve_expected_stream_count({}, 6337) == 6337
    ar = os.path.join(_REPO_ROOT, "src", "configs", "dataset_configs", "AR", "eval",
                      "acousticroom_unseeneval.json")
    with open(ar) as f:
        config = json.load(f)
    assert "expected_items" not in config
    assert eval_FLAC.resolve_expected_stream_count(config, 6337) == 6337


def test_a_non_positive_declared_count_is_refused():
    eval_FLAC = _eval_flac()
    with pytest.raises(ValueError):
        eval_FLAC.resolve_expected_stream_count({"expected_items": 0})


# --------------------------------------------------------------------------- #
# r2 N8: the tracked height is read from the RAW RAF row
# --------------------------------------------------------------------------- #
def _built_items(poses, k=8):
    return prep_a.build_items(
        "EmptyRoom", "p000", poses,
        assignment={p["group_key"]: list(range(36)) for p in poses},
        match={p["group_key"]: {"p95_m": 0.004, "max_m": 0.008,
                                "min_ambiguity_margin": 40.0} for p in poses}, k=k)


def test_the_tracked_height_is_the_raw_raf_up_axis():
    raw = np.array([[0.1 * s, 1.7, 0.2] for s in range(36)])
    items = _built_items(_placement_poses_with_raw(raw))
    assert prep_a.RAF_UP_AXIS == 1
    for item in items:
        assert item["rx_target_height_raf_m"] == 1.7


def test_a_wrong_gauge_cannot_change_the_tracked_height():
    """The height comes from the RAW row, so a pipeline coordinate that no longer
    holds it -- a mis-specified gauge -- cannot rewrite the number the depth
    renderer tracks. Reading rx_target_p[2] was right only by coincidence of the
    registered gauge."""
    raw = np.array([[0.1 * s, 1.7, 0.2] for s in range(36)])
    poses = _placement_poses_with_raw(raw)
    for pose in poses:                       # a deliberately wrong pipeline mapping
        pose["rx_xyz_p"] = np.array([[row[0], row[1], row[2]] for row in raw])
    items = _built_items(poses)
    for item in items:
        assert item["rx_target_height_raf_m"] == 1.7        # untouched raw height
        assert item["rx_target_p"][2] == 0.2                # the wrong gauge's z


def _placement_poses_with_raw(raw, n_poses=12):
    return [_pose(f"g{i}", [f"{i * 36 + s:06d}" for s in range(36)],
                  (float(i), 1.5, 2.0), rx_raw=raw)
            for i in range(n_poses)]
