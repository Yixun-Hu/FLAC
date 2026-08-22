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


def _sidecar(tmp_path, name, seed, value_of, rooms=stats.REGISTERED_ROOMS,
             placements=stats.REGISTERED_PLACEMENTS_PER_ROOM,
             slots=stats.REGISTERED_SLOTS_PER_PLACEMENT, metric="C50", **overrides):
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
               "provenance": {"seed": seed, "ckpt_path": f"{name}.ckpt"},
               "items": rows}
    payload.update(overrides)
    path = tmp_path / f"{name}_seed{seed}.per_item.json"
    path.write_text(json.dumps(payload))
    return str(path)


def _arm(tmp_path, label, offset, seeds=(0, 1)):
    paths = [_sidecar(tmp_path, label, seed,
                      lambda room, p, slot, offset=offset, seed=seed:
                      offset + p + 0.01 * slot + 0.001 * seed
                      + (100.0 if room == "FurnishedRoom" else 0.0))
             for seed in seeds]
    return stats.arm_from_sidecars(paths, "C50", label)


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
        stats.arm_from_sidecars([a, b], "C50", "arm")
    assert "seed 0 appears" in str(exc.value)


def test_an_arm_refuses_seeds_that_evaluated_different_items(tmp_path):
    a = _sidecar(tmp_path, "arm", 0, lambda *a: 1.0)
    b = _sidecar(tmp_path, "arm", 1, lambda *a: 1.0)
    payload = json.loads(open(b).read())
    payload["items"][5]["item_id"] = "EmptyRoom/p999/slot05"
    open(b, "w").write(json.dumps(payload))
    with pytest.raises(ValueError) as exc:
        stats.arm_from_sidecars([a, b], "C50", "arm")
    assert "identical items" in str(exc.value)


def test_two_arms_over_different_items_are_not_paired(tmp_path):
    arm_a = _arm(tmp_path, "armA", 0.0)
    arm_b = _arm(tmp_path, "armB", 1.0)
    arm_b["item_ids"] = arm_b["item_ids"][:-1]
    with pytest.raises(ValueError) as exc:
        stats.assert_paired(arm_a, arm_b)
    assert "not paired" in str(exc.value)


def test_two_arms_with_different_seed_sets_are_not_paired(tmp_path):
    arm_a = _arm(tmp_path, "armA", 0.0, seeds=(0, 1))
    arm_b = _arm(tmp_path, "armB", 1.0, seeds=(0, 2))
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
    assert set(counts.values()) == {36 * 2}          # 36 slots x 2 seeds


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
                        placements=8)

    arm_a = stats.arm_from_sidecars([half("a", 0)], "C50", "a", enforce_design=False)
    arm_b = stats.arm_from_sidecars([half("b", 0)], "C50", "b", enforce_design=False)
    with pytest.raises(ValueError) as exc:
        stats.contrast_report(arm_a, arm_b)
    assert "expected 32" in str(exc.value)


def test_a_non_finite_item_stops_the_contrast(tmp_path):
    arm_a = _arm(tmp_path, "armA", 0.0)
    arm_b = _arm(tmp_path, "armB", -0.25)
    first = arm_a["item_ids"][0]
    arm_a["by_seed"][0][first]["value"] = None
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
    record = eval_FLAC.build_per_item_record(rows, {"seed": 3, "ckpt_path": "x.ckpt"})
    path = tmp_path / "run_metrics_a.per_item.json"
    path.write_text(json.dumps(record))

    arm = stats.arm_from_sidecars([str(path)], "C50", "armA")
    assert arm["seeds"] == [3]
    assert len(arm["item_ids"]) == stats.REGISTERED_N_ITEMS
    means = stats.arm_placement_means(arm)[3]
    assert abs(means[("EmptyRoom", "p000")] - 0.175) < 1e-12
