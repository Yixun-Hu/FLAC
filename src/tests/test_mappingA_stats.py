"""Placement-level statistics (exp_21, contract E, cycle 10).

The clustering unit is the placement: 36 items of one placement share a room
position, an array, a target source and overlapping context, so item-i.i.d.
intervals would understate the uncertainty. Every oracle here is hand-computed.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAF_DIR = os.path.join(_REPO_ROOT, "data", "RAF")
if _RAF_DIR not in sys.path:
    sys.path.insert(0, _RAF_DIR)

import mappingA_stats as stats  # noqa: E402


def _records(values_by_placement, room="EmptyRoom", metric="C50"):
    records = []
    for placement, values in values_by_placement.items():
        for i, value in enumerate(values):
            records.append({"room": room, "placement_id": placement,
                            "item_id": f"{room}/{placement}/slot{i:02d}",
                            "metrics": {metric: value}})
    return records


def test_within_placement_aggregation_is_the_item_mean():
    records = _records({"p000": [1.0, 2.0, 3.0], "p001": [10.0, 20.0]})
    means, counts = stats.aggregate_within_placement(records, "C50")
    assert means == {("EmptyRoom", "p000"): 2.0, ("EmptyRoom", "p001"): 15.0}
    assert counts[("EmptyRoom", "p000")] == 3


def test_a_missing_metric_is_refused_rather_than_dropped():
    records = _records({"p000": [1.0, float("nan")]})
    with pytest.raises(ValueError) as exc:
        stats.aggregate_within_placement(records, "C50")
    assert "different estimand" in str(exc.value)


def test_the_macro_weights_rooms_equally_not_placements():
    """EmptyRoom contributes 3 placements averaging 2, FurnishedRoom 1 placement at
    10: the macro is (2 + 10)/2 = 6, not the placement mean 4."""
    means = {("EmptyRoom", "p000"): 1.0, ("EmptyRoom", "p001"): 2.0,
             ("EmptyRoom", "p002"): 3.0, ("FurnishedRoom", "p003"): 10.0}
    out = stats.macro_two_room(means)
    assert out["rooms"] == {"EmptyRoom": 2.0, "FurnishedRoom": 10.0}
    assert out["macro"] == 6.0
    assert out["n_placements"] == {"EmptyRoom": 3, "FurnishedRoom": 1}
    assert out["macro"] != np.mean([1.0, 2.0, 3.0, 10.0])


def test_the_bootstrap_collapses_when_every_placement_agrees():
    means = {("EmptyRoom", f"p{i:03d}"): 5.0 for i in range(4)}
    means.update({("FurnishedRoom", f"p1{i:02d}"): 7.0 for i in range(4)})
    out = stats.cluster_bootstrap(means, n_resamples=200)
    assert out["point"] == 6.0
    assert out["ci_low"] == out["ci_high"] == 6.0
    assert out["unit"] == "placement" and out["stratified_by"] == "room"


def test_the_bootstrap_interval_brackets_the_point_and_is_deterministic():
    rng = np.random.default_rng(0)
    means = {("EmptyRoom", f"p{i:03d}"): float(v)
             for i, v in enumerate(rng.normal(5.0, 1.0, 8))}
    means.update({("FurnishedRoom", f"q{i:03d}"): float(v)
                  for i, v in enumerate(rng.normal(7.0, 1.0, 8))})
    first = stats.cluster_bootstrap(means, n_resamples=500)
    second = stats.cluster_bootstrap(means, n_resamples=500)
    assert first == second                                  # stable seed
    assert first["ci_low"] < first["point"] < first["ci_high"]


def test_the_paired_test_is_exact_and_hand_checkable():
    """Four placements, every difference +1. Of the 2^4 = 16 sign assignments only
    all-plus and all-minus give |mean| >= 1, so the two-sided p is 2/16 = 0.125."""
    arm_a = {("EmptyRoom", f"p{i:03d}"): 1.0 for i in range(4)}
    arm_b = {("EmptyRoom", f"p{i:03d}"): 0.0 for i in range(4)}
    out = stats.paired_randomization(arm_a, arm_b)
    assert out["exact"] is True
    assert out["n_assignments"] == 16
    assert out["observed_difference"] == 1.0
    assert out["p_value"] == pytest.approx(2 / 16)


def test_the_paired_test_finds_no_effect_when_differences_cancel():
    arm_a = {("EmptyRoom", "p000"): 1.0, ("EmptyRoom", "p001"): 0.0}
    arm_b = {("EmptyRoom", "p000"): 0.0, ("EmptyRoom", "p001"): 1.0}
    out = stats.paired_randomization(arm_a, arm_b)
    assert out["observed_difference"] == 0.0
    assert out["p_value"] == 1.0


def test_the_paired_test_refuses_unpaired_placements():
    with pytest.raises(ValueError) as exc:
        stats.paired_randomization({("EmptyRoom", "p000"): 1.0},
                                   {("EmptyRoom", "p001"): 1.0})
    assert "not a paired test" in str(exc.value)


def test_a_large_design_falls_back_to_sampling_and_says_so():
    arm_a = {("EmptyRoom", f"p{i:03d}"): float(i) for i in range(32)}
    arm_b = {("EmptyRoom", f"p{i:03d}"): 0.0 for i in range(32)}
    out = stats.paired_randomization(arm_a, arm_b, n_resamples=500)
    assert out["exact"] is False
    assert out["n_assignments"] == 500
    assert out["n_placements"] == 32


def test_seed_variability_is_reported_separately_from_the_interval():
    out = stats.seed_variability({42: 1.0, 43: 1.2, 44: 0.8, 45: 1.1, 46: 0.9})
    assert out["n_seeds"] == 5
    assert out["mean"] == pytest.approx(1.0)
    assert out["sd"] == pytest.approx(np.std([1.0, 1.2, 0.8, 1.1, 0.9], ddof=1))
    assert "not an interval" in out["note"]


def test_the_full_pipeline_runs_from_item_records():
    records = (_records({f"p{i:03d}": [1.0 + 0.1 * i] * 36 for i in range(4)}) +
               _records({f"q{i:03d}": [2.0 + 0.1 * i] * 36 for i in range(4)},
                        room="FurnishedRoom"))
    means, counts = stats.aggregate_within_placement(records, "C50")
    assert sum(counts.values()) == 8 * 36
    macro = stats.macro_two_room(means)
    assert macro["macro"] == pytest.approx((1.15 + 2.15) / 2)
    interval = stats.cluster_bootstrap(means, n_resamples=200)
    assert interval["n_placements"] == {"EmptyRoom": 4, "FurnishedRoom": 4}


# --------------------------------------------------------------------------- #
# r2 N7: ingesting eval_FLAC's per-item sidecars, paired item x seed
# --------------------------------------------------------------------------- #
import json  # noqa: E402


def _provenance(seed, ckpt="a" * 64, **overrides):
    """A complete cell identity (r3 P3): weights by content, configs by digest, the
    corpus by its publication generation, the item stream by its hash."""
    provenance = {"seed": seed, "ckpt_path": "arm.ckpt", "ckpt_sha256": ckpt,
                  "ckpt_bytes": 1024, "cond_method": "vanilla",
                  "frame_avg_angles": None, "frame_avg_fwd_cap": None,
                  "orbit_execution": "n/a",
                  "model_config_sha256": "b" * 64,
                  "dataset_config_sha256": "c" * 64,
                  "publication_prepare_generation": "46a43f4ce82b",
                  "publication_depth_generation": "a44a723fce4c",
                  "expected_items": stats.REGISTERED_N_ITEMS,
                  "stream_input_hash": "d" * 64,
                  "stream_assignment_hash": "e" * 64,
                  "cond_autocast": "default", "batch_size": 64,
                  "source_sha": "0123456789ab",
                  "steps": 1, "cfg_scale": 1.0, "are_lambda": None,
                  "rotate_mode": "fixed", "rotate_deg": 0.0, "rotate_seed": None}
    provenance.update(overrides)
    return provenance


def _sidecar(tmp_path, name, seed, value_of, rooms=stats.REGISTERED_ROOMS,
             placements=stats.REGISTERED_PLACEMENTS_PER_ROOM,
             slots=stats.REGISTERED_SLOTS_PER_PLACEMENT, metric="C50",
             provenance=None, **overrides):
    """A registered-shape sidecar whose values come from value_of(room, p, slot)."""
    rows = []
    for room in rooms:
        for p in range(placements):
            placement = f"p{p:03d}"
            for slot in range(slots):
                rows.append({
                    "item_id": f"{room}/{placement}/slot{slot:02d}",
                    "room": room, "placement_id": placement, "mic_slot": slot,
                    "metrics": {metric: value_of(room, p, slot)}})
    payload = {"schema_version": stats.PER_ITEM_SCHEMA_VERSION, "n_items": len(rows),
               "metrics": [metric], "excluded_metrics": ["eval_FD", "eval_retrieval"],
               "provenance": provenance or _provenance(seed),
               "items": rows}
    payload.update(overrides)
    path = tmp_path / f"{name}_seed{seed}.per_item.json"
    path.write_text(json.dumps(payload))
    return str(path)


def _arm(tmp_path, label, offset, seeds=stats.REGISTERED_SEEDS, ckpt=None,
         registered=True, **provenance):
    paths = [_sidecar(tmp_path, label, seed,
                      lambda room, p, slot, offset=offset, seed=seed:
                      offset + p + 0.01 * slot + 0.001 * seed
                      + (100.0 if room == "FurnishedRoom" else 0.0),
                      provenance=_provenance(seed, ckpt=ckpt or (label[-1] * 64),
                                             **provenance))
             for seed in seeds]
    return stats.arm_from_sidecars(paths, "C50", label, registered=registered)


def test_a_sidecar_of_the_wrong_schema_is_refused(tmp_path):
    path = _sidecar(tmp_path, "arm", 0, lambda *a: 1.0, schema_version=99)
    with pytest.raises(ValueError) as exc:
        stats.load_per_item_sidecar(path)
    assert "schema version" in str(exc.value)


def test_the_registered_design_is_checked_not_inferred(tmp_path):
    path = _sidecar(tmp_path, "short", 0, lambda *a: 1.0, placements=8)
    rows = stats.load_per_item_sidecar(path)["rows"]
    with pytest.raises(ValueError) as exc:
        stats.assert_registered_design(rows)
    assert "8 placements" in str(exc.value) and "16" in str(exc.value)

    one_room = _sidecar(tmp_path, "one", 0, lambda *a: 1.0, rooms=("EmptyRoom",))
    with pytest.raises(ValueError) as exc:
        stats.assert_registered_design(stats.load_per_item_sidecar(one_room)["rows"])
    assert "rooms" in str(exc.value)


def test_a_full_registered_sidecar_passes_the_design_check(tmp_path):
    path = _sidecar(tmp_path, "full", 0, lambda room, p, slot: float(p))
    report = stats.assert_registered_design(stats.load_per_item_sidecar(path)["rows"])
    assert report["n_items"] == stats.REGISTERED_N_ITEMS == 1152
    assert report["n_placements"] == 32


def test_an_arm_refuses_a_repeated_seed(tmp_path):
    a = _sidecar(tmp_path, "armA", 0, lambda *a: 1.0)
    b = _sidecar(tmp_path, "armB", 0, lambda *a: 2.0)
    with pytest.raises(ValueError) as exc:
        stats.arm_from_sidecars([a, b], "C50", "arm", registered=False)
    assert "seed 0 appears" in str(exc.value)


def test_an_arm_refuses_seeds_that_evaluated_different_items(tmp_path):
    a = _sidecar(tmp_path, "arm", 0, lambda *a: 1.0)
    b = _sidecar(tmp_path, "arm", 1, lambda *a: 1.0)
    payload = json.loads(open(b).read())
    payload["items"][5]["item_id"] = "EmptyRoom/p999/slot05"
    open(b, "w").write(json.dumps(payload))
    with pytest.raises(ValueError) as exc:
        stats.arm_from_sidecars([a, b], "C50", "arm", registered=False)
    assert "identical items" in str(exc.value)


def test_two_arms_over_different_items_are_not_paired(tmp_path):
    arm_a = _arm(tmp_path, "armA", 0.0)
    arm_b = _arm(tmp_path, "armB", 1.0)
    arm_b["item_ids"] = arm_b["item_ids"][:-1]
    with pytest.raises(ValueError) as exc:
        stats.assert_paired(arm_a, arm_b)
    assert "not paired" in str(exc.value)


def test_two_arms_with_different_seed_sets_are_not_paired(tmp_path):
    arm_a = _arm(tmp_path, "armA", 0.0, seeds=(0, 1), registered=False)
    arm_b = _arm(tmp_path, "armB", 1.0, seeds=(0, 2), registered=False)
    with pytest.raises(ValueError) as exc:
        stats.assert_paired(arm_a, arm_b)
    assert "seeds differ" in str(exc.value)


def test_the_paired_difference_is_taken_item_by_item(tmp_path):
    """A constant offset must come back exactly, per placement -- the hand oracle
    for "difference first, average second"."""
    arm_a = _arm(tmp_path, "armA", 0.0)
    arm_b = _arm(tmp_path, "armB", -0.25)
    differences, counts = stats.paired_placement_differences(arm_a, arm_b)
    assert len(differences) == 32
    assert all(abs(value - 0.25) < 1e-12 for value in differences.values())
    assert set(counts.values()) == {36 * len(stats.REGISTERED_SEEDS)}


def test_the_contrast_report_is_paired_clustered_and_equal_room(tmp_path):
    arm_a = _arm(tmp_path, "armA", 0.0)
    arm_b = _arm(tmp_path, "armB", -0.25)
    report = stats.contrast_report(arm_a, arm_b, n_resamples=200)
    assert report["unit"] == "placement"
    assert abs(report["difference"]["macro"] - 0.25) < 1e-12
    assert report["difference"]["n_placements"] == {"EmptyRoom": 16,
                                                    "FurnishedRoom": 16}
    # a constant shift: every bootstrap resample gives the same difference
    assert abs(report["interval"]["ci_low"] - 0.25) < 1e-9
    assert abs(report["interval"]["ci_high"] - 0.25) < 1e-9
    # 32 placements is past the exact limit, so the p-value is sampled and says so
    assert report["randomization"]["exact"] is False
    assert report["randomization"]["p_value"] < 0.01
    # seed variability is BESIDE the interval, never inside it
    assert set(report["seed_variability"]) == {"armA", "armB"}
    assert "not part of the interval" in report["note"]
    json.dumps(report)


def test_a_contrast_over_the_wrong_number_of_placements_is_refused(tmp_path):
    def half(name, seed):
        return _sidecar(tmp_path, name, seed, lambda room, p, slot: float(p),
                        placements=8,
                        provenance=_provenance(seed, ckpt=name * 64))

    arm_a = stats.arm_from_sidecars([half("a", 0)], "C50", "a", enforce_design=False,
                                    registered=False)
    arm_b = stats.arm_from_sidecars([half("b", 0)], "C50", "b", enforce_design=False,
                                    registered=False)
    with pytest.raises(ValueError) as exc:
        stats.contrast_report(arm_a, arm_b, require_registered=False)
    assert "expected 32" in str(exc.value)


def test_a_non_finite_item_stops_the_contrast(tmp_path):
    arm_a = _arm(tmp_path, "armA", 0.0)
    arm_b = _arm(tmp_path, "armB", -0.25)
    first = arm_a["item_ids"][0]
    arm_a["by_seed"][arm_a["seeds"][0]][first]["value"] = None
    with pytest.raises(ValueError) as exc:
        stats.paired_placement_differences(arm_a, arm_b)
    assert "different estimand" in str(exc.value)


# --------------------------------------------------------------------------- #
# r2 N7: the producer side (eval_FLAC --record-per-item)
# --------------------------------------------------------------------------- #
def _eval_flac():
    import importlib

    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    return importlib.import_module("eval_FLAC")


class _StubCallback:
    """Records what it was updated with and returns one item's metrics."""

    def __init__(self):
        self.batches = []

    def update_metrics(self, stage, pred, ref, *args, **kwargs):
        self.batches.append((stage, tuple(pred.shape), tuple(ref.shape), args, kwargs))
        self.pending = float(pred.reshape(-1)[0])

    def compute_metrics(self, stage):
        return {"C50": self.pending, "T60": self.pending * 2,
                "by_scene": {"EmptyRoom": {}}}


def _md(slot=0, room="EmptyRoom", placement="p000", **overrides):
    md = {"scene": room, "placement_id": placement, "mic_slot": slot,
          "item_id": f"{room}/{placement}/slot{slot:02d}"}
    md.update(overrides)
    return md


def test_the_producer_and_the_reader_share_one_schema_version():
    assert _eval_flac().PER_ITEM_SCHEMA_VERSION == stats.PER_ITEM_SCHEMA_VERSION


def test_the_sidecar_path_sits_beside_the_metrics_record():
    eval_FLAC = _eval_flac()
    assert (eval_FLAC.per_item_sidecar_path("/o/x_metrics_a.json")
            == "/o/x_metrics_a.per_item.json")
    # ... and never collides with the assignment stream sidecar
    assert (eval_FLAC.per_item_sidecar_path("/o/x_metrics_a.json")
            != eval_FLAC.stream_sidecar_path("/o/x_metrics_a.json"))


def test_the_per_item_config_disables_the_distribution_level_metrics():
    eval_FLAC = _eval_flac()
    model_config = {"training": {"metrics": {"eval_FD": True, "eval_retrieval": True,
                                             "eval_C50": True}}}
    per_item = eval_FLAC.per_item_metric_config(model_config)
    assert per_item["training"]["metrics"] == {"eval_FD": False,
                                               "eval_retrieval": False,
                                               "eval_C50": True}
    # the caller's config is untouched -- the headline run still computes FD
    assert model_config["training"]["metrics"]["eval_FD"] is True


@pytest.mark.parametrize("missing", ["item_id", "scene", "placement_id", "mic_slot"])
def test_an_unidentifiable_item_stops_the_recording(missing):
    eval_FLAC = _eval_flac()
    md = _md()
    del md[missing]
    with pytest.raises(ValueError) as exc:
        eval_FLAC.per_item_identity(md)
    assert missing in str(exc.value)


def test_an_item_id_that_contradicts_its_slot_is_refused():
    eval_FLAC = _eval_flac()
    with pytest.raises(ValueError) as exc:
        eval_FLAC.per_item_identity(_md(slot=3, item_id="EmptyRoom/p000/slot07"))
    assert "does not match" in str(exc.value)


def test_each_row_is_computed_on_its_own_item():
    """One item in, one row out: the accumulators must not carry across items."""
    import torch

    eval_FLAC = _eval_flac()
    callback = _StubCallback()
    fakes = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    reals = torch.zeros_like(fakes)
    metadata = [_md(slot=i) for i in range(3)]
    rows = eval_FLAC.record_per_item_metrics(callback, fakes, reals, metadata)
    assert [row["item_id"] for row in rows] == [
        f"EmptyRoom/p000/slot{i:02d}" for i in range(3)]
    assert [row["metrics"]["C50"] for row in rows] == [1.0, 2.0, 3.0]
    assert all(shape[0] == 1 for _, shape, _, _, _ in callback.batches)
    assert len(callback.batches) == 3
    # the RAF equal-room macro cannot be computed per item, so by_scene is dropped
    assert all("by_scene" not in row["metrics"] for row in rows)


def test_a_non_finite_metric_is_recorded_as_null_not_as_a_number():
    import torch

    eval_FLAC = _eval_flac()
    callback = _StubCallback()
    fakes = torch.tensor([[[float("nan")]]])
    rows = eval_FLAC.record_per_item_metrics(callback, fakes,
                                             torch.zeros_like(fakes), [_md()])
    assert rows[0]["metrics"]["C50"] is None
    # ... and the reader refuses to average over it rather than dropping the item
    payload = {"schema_version": stats.PER_ITEM_SCHEMA_VERSION, "n_items": 1,
               "metrics": ["C50"], "provenance": {"seed": 0}, "items": rows}
    assert payload["items"][0]["metrics"]["C50"] is None


def test_the_sidecar_payload_names_what_it_excludes_and_why():
    eval_FLAC = _eval_flac()
    rows = [dict(_md(slot=i), metrics={"C50": float(i)}) for i in range(2)]
    record = eval_FLAC.build_per_item_record(rows, {"seed": 0})
    assert record["schema_version"] == stats.PER_ITEM_SCHEMA_VERSION
    assert record["n_items"] == 2 and record["metrics"] == ["C50"]
    assert record["excluded_metrics"] == ["eval_FD", "eval_retrieval"]
    assert "single item" in record["excluded_reason"]
    assert record["provenance"]["seed"] == 0
    json.dumps(record)


def test_a_produced_sidecar_reads_back_through_the_stats_loader(tmp_path):
    """The two halves of the contract, joined: what eval_FLAC writes is what
    mappingA_stats accepts."""
    eval_FLAC = _eval_flac()
    rows = []
    for room in stats.REGISTERED_ROOMS:
        for p in range(stats.REGISTERED_PLACEMENTS_PER_ROOM):
            for slot in range(stats.REGISTERED_SLOTS_PER_PLACEMENT):
                md = _md(slot=slot, room=room, placement=f"p{p:03d}")
                rows.append(dict(eval_FLAC.per_item_identity(md),
                                 metrics={"C50": float(p) + 0.01 * slot}))
    record = eval_FLAC.build_per_item_record(rows, _provenance(3))
    path = tmp_path / "run_metrics_a.per_item.json"
    path.write_text(json.dumps(record))

    arm = stats.arm_from_sidecars([str(path)], "C50", "armA", registered=False)
    assert arm["seeds"] == [3]
    assert len(arm["item_ids"]) == stats.REGISTERED_N_ITEMS
    means = stats.arm_placement_means(arm)[3]
    assert abs(means[("EmptyRoom", "p000")] - 0.175) < 1e-12


# --------------------------------------------------------------------------- #
# r3 P3: an arm is a cell, identified by its provenance
# --------------------------------------------------------------------------- #
def test_sidecars_from_two_checkpoints_cannot_be_pooled_as_one_arm(tmp_path):
    """The operator's file list used to be the only thing claiming these runs were
    one arm, so two checkpoints could be averaged together under one label."""
    a = _sidecar(tmp_path, "x", 42, lambda *a: 1.0,
                 provenance=_provenance(42, ckpt="a" * 64))
    b = _sidecar(tmp_path, "x", 43, lambda *a: 1.0,
                 provenance=_provenance(43, ckpt="f" * 64))
    with pytest.raises(ValueError) as exc:
        stats.arm_from_sidecars([a, b], "C50", "arm", registered=False)
    assert "not the same cell" in str(exc.value)
    assert "ckpt_sha256" in str(exc.value)


@pytest.mark.parametrize("field,value", [
    ("cond_method", "fa_invariant"),
    ("frame_avg_angles", [0.0, 90.0, 180.0, 270.0]),
    ("cond_autocast", "bf16"),
    ("batch_size", 32),
    ("publication_prepare_generation", "deadbeefdead"),
    ("publication_depth_generation", "deadbeefdead"),
    ("dataset_config_sha256", "9" * 64),
    ("steps", 8),
    ("are_lambda", 0.5),
])
def test_any_non_seed_protocol_field_must_be_constant_within_an_arm(tmp_path, field,
                                                                    value):
    a = _sidecar(tmp_path, "x", 42, lambda *a: 1.0, provenance=_provenance(42))
    b = _sidecar(tmp_path, "x", 43, lambda *a: 1.0,
                 provenance=_provenance(43, **{field: value}))
    with pytest.raises(ValueError) as exc:
        stats.arm_from_sidecars([a, b], "C50", "arm", registered=False)
    assert field in str(exc.value) and "Only the seed may vary" in str(exc.value)


def test_a_sidecar_without_the_identity_fields_is_refused(tmp_path):
    provenance = _provenance(42)
    del provenance["ckpt_sha256"]
    path = _sidecar(tmp_path, "x", 42, lambda *a: 1.0, provenance=provenance)
    with pytest.raises(ValueError) as exc:
        stats.arm_from_sidecars([path], "C50", "arm", registered=False)
    assert "ckpt_sha256" in str(exc.value)


def test_the_registered_report_requires_exactly_the_registered_seeds(tmp_path):
    with pytest.raises(ValueError) as exc:
        _arm(tmp_path, "armA", 0.0, seeds=(42,))
    assert "[42, 43, 44, 45, 46]" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        _arm(tmp_path, "armA", 0.0, seeds=(1, 2, 3, 4, 5))
    assert "different experiment" in str(exc.value)
    # the registered set passes, and the arm says so
    arm = _arm(tmp_path, "armA", 0.0)
    assert arm["seeds"] == list(stats.REGISTERED_SEEDS)
    assert arm["registered"] is True


def test_an_exploratory_arm_cannot_receive_the_registered_report(tmp_path):
    arm_a = _arm(tmp_path, "armA", 0.0, seeds=(0, 1), registered=False)
    arm_b = _arm(tmp_path, "armB", -0.25, seeds=(0, 1), registered=False)
    with pytest.raises(ValueError) as exc:
        stats.contrast_report(arm_a, arm_b)
    assert "registered arms" in str(exc.value)
    report = stats.contrast_report(arm_a, arm_b, n_resamples=50,
                                   require_registered=False)
    assert report["registered"] is False


def test_arms_that_scored_different_corpora_are_not_comparable(tmp_path):
    arm_a = _arm(tmp_path, "armA", 0.0)
    arm_b = _arm(tmp_path, "armB", -0.25,
                 publication_prepare_generation="deadbeefdead")
    with pytest.raises(ValueError) as exc:
        stats.assert_paired(arm_a, arm_b)
    assert "publication_prepare_generation differs" in str(exc.value)


def test_a_depth_republish_between_arms_is_caught(tmp_path):
    """The depth maps are published under their own generation and can move
    without the prepare generation changing -- so the corpus can differ while the
    old single-generation record said it had not (r4 Q2)."""
    arm_a = _arm(tmp_path, "armA", 0.0)
    arm_b = _arm(tmp_path, "armB", -0.25,
                 publication_depth_generation="feedfacefeed")
    with pytest.raises(ValueError) as exc:
        stats.assert_paired(arm_a, arm_b)
    assert "publication_depth_generation differs" in str(exc.value)


@pytest.mark.parametrize("field,value", [
    ("steps", 8),
    ("cfg_scale", 3.0),
    ("are_lambda", 0.25),
    ("cond_autocast", "bf16"),
    ("batch_size", 16),
    ("rotate_deg", 45.0),
    ("rotate_mode", "random"),
    ("source_sha", "ffffffffffff"),
])
def test_arms_that_held_a_control_at_different_values_are_not_comparable(
        tmp_path, field, value):
    """The reviewer's probe was 1-step vs 8-step: otherwise matched arms whose
    sampler budget differed compared as if only the checkpoint had changed."""
    arm_a = _arm(tmp_path, "armA", 0.0)
    arm_b = _arm(tmp_path, "armB", -0.25, **{field: value})
    with pytest.raises(ValueError) as exc:
        stats.assert_paired(arm_a, arm_b)
    assert f"{field} differs" in str(exc.value)


def test_arms_may_differ_in_what_the_experiment_varies(tmp_path):
    """A vanilla arm and a frame-averaged arm ARE the contrast: the conditioning
    protocol each checkpoint was trained for is what differs by design."""
    arm_a = _arm(tmp_path, "armA", 0.0)
    arm_b = _arm(tmp_path, "armB", -0.25, cond_method="fa_invariant",
                 frame_avg_angles=[0.0, 90.0, 180.0, 270.0],
                 frame_avg_fwd_cap=64, orbit_execution="batched",
                 model_config_sha256="9" * 64)
    assert stats.assert_paired(arm_a, arm_b)["n_items"] == stats.REGISTERED_N_ITEMS


def test_arms_that_evaluated_different_item_streams_are_not_comparable(tmp_path):
    """The stream hash is over the items and their conditioning, so it is identical
    across arms by construction -- unless they did not evaluate the same run of the
    same split."""
    arm_a = _arm(tmp_path, "armA", 0.0)
    arm_b = _arm(tmp_path, "armB", -0.25, stream_input_hash="0" * 64)
    with pytest.raises(ValueError) as exc:
        stats.assert_paired(arm_a, arm_b)
    assert "stream_input_hash differs" in str(exc.value)


def test_one_cell_compared_with_itself_is_not_a_contrast(tmp_path):
    arm_a = _arm(tmp_path, "armA", 0.0, ckpt="a" * 64)
    arm_b = _arm(tmp_path, "armB", 0.0, ckpt="a" * 64)
    assert arm_a["identity_sha256"] == arm_b["identity_sha256"]
    with pytest.raises(ValueError) as exc:
        stats.assert_paired(arm_a, arm_b)
    assert "one cell compared with itself" in str(exc.value)


def test_the_registered_contrast_names_both_arm_identities(tmp_path):
    arm_a = _arm(tmp_path, "armA", 0.0)
    arm_b = _arm(tmp_path, "armB", -0.25)
    report = stats.contrast_report(arm_a, arm_b, n_resamples=100)
    assert report["registered"] is True
    assert report["seeds"] == list(stats.REGISTERED_SEEDS)
    assert set(report["arm_identities"]) == {"armA", "armB"}
    assert (report["arm_identities"]["armA"]
            != report["arm_identities"]["armB"])
    assert abs(report["difference"]["macro"] - 0.25) < 1e-12
    json.dumps(report)


def test_both_publication_generations_must_be_attested_by_the_loader():
    eval_FLAC = _eval_flac()
    metadata = [dict(_md(slot=i), publication_prepare_generation="aaaa",
                     publication_depth_generation="bbbb") for i in range(3)]
    assert eval_FLAC.resolve_publication_generations(metadata) == {
        "publication_prepare_generation": "aaaa",
        "publication_depth_generation": "bbbb"}

    metadata[1]["publication_depth_generation"] = "cccc"
    with pytest.raises(ValueError) as exc:
        eval_FLAC.resolve_publication_generations(metadata)
    assert "publication_depth_generation" in str(exc.value)

    for md in metadata:
        md.pop("publication_depth_generation")
    with pytest.raises(ValueError) as exc:
        eval_FLAC.resolve_publication_generations(metadata)
    assert "attested no publication_depth_generation" in str(exc.value)


def test_the_producer_and_the_reader_share_the_identity_field_set():
    assert (_eval_flac().PER_ITEM_IDENTITY_FIELDS
            == stats.ARM_IDENTITY_FIELDS)
    assert set(stats.SHARED_IDENTITY_FIELDS) <= set(stats.ARM_IDENTITY_FIELDS)
