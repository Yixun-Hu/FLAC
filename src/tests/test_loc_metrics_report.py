"""exp_18 R4 offline aggregation (`src/localization/metrics_report.py`).

Every statistic is checked against a synthetic fixture whose answer is known by
construction, and the two conventions that already exist in the campaign -- the
reciprocal-rank tie-break and the nearest-context retrieval geometry -- are
pinned by equality against the code that defines them, not re-derived here.
"""
import json
import math
import os

import numpy as np
import pytest
import torch

from src.localization import metrics_report as mr
from src.localization import rir_metrics as rm
from src.localization.reaggregate import encode_sims

# --------------------------------------------------------------------------- #
# fixture geometry: world and camera frames DIFFER, so a routine that mixes them
# lands 10 m away and the test fails loudly.
# --------------------------------------------------------------------------- #
CAND_WORLD = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
CAND_CAM = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [13.0, 0.0, 0.0]]
CTX_CAM = [[11.2, 0.0, 0.0], [13.4, 0.0, 0.0]]          # nearest cand: 1, then 2
GT_INDEX = 0
CONTEXT_MEMBER = [False, True, False]                    # candidate 1 is in context


def _hex_vector(values):
    return encode_sims(torch.tensor([list(values)], dtype=torch.float32))[0]


def _family_block(distances, ctx_distances):
    matrix = torch.tensor(distances, dtype=torch.float32)
    aggregations = {how: _hex_vector(rm.aggregate_over_k(matrix, how).tolist())
                    for how in ("mean", "min", "median", "lme")}
    pred = int(rm.predict_from_distances(rm.aggregate_over_k(matrix, "mean")))
    return {"candidates_hex": encode_sims(matrix),
            "context_hex": _hex_vector(ctx_distances),
            "aggregations": aggregations, "pred_index": pred, "pred_node": pred,
            "correct": bool(pred == GT_INDEX)}


def _metrics_row(query_id, room_id, position, families, agree_pred=0,
                 context_member=None, m4=None, sensitivities=None):
    return {
        "query_id": query_id, "room_id": room_id, "position": position,
        "n_candidates": 3, "n_samples": 2, "n_context": 2,
        "candidate_nodes": [0, 1, 2], "gt_index": GT_INDEX, "gt_node": 0,
        "candidate_xyz_world": CAND_WORLD, "gt_xyz_world": CAND_WORLD[GT_INDEX],
        "context_member": list(CONTEXT_MEMBER if context_member is None else context_member),
        "agree_pred_index": int(agree_pred),
        "agree_e_loc": float(np.linalg.norm(np.asarray(CAND_WORLD[int(agree_pred)])
                                            - np.asarray(CAND_WORLD[GT_INDEX]))),
        "families": {name: _family_block(dist, ctx)
                     for name, (dist, ctx) in families.items()},
        "m4": m4, "sensitivities": sensitivities,
        "metric_config": {"delta_max": 8, "families": ["m1", "m2", "m3", "m4", "m5"]},
    }


def _replay_row(query_id, room_id, ctx_sims=(0.9, 0.1)):
    return {"query_id": query_id, "room_id": room_id,
            "candidate_xyz_cam": CAND_CAM, "candidate_xyz_world": CAND_WORLD,
            "context_xyz_cam": CTX_CAM, "context_sims_hex": _hex_vector(ctx_sims),
            "gt_index": GT_INDEX, "gt_xyz_world": CAND_WORLD[GT_INDEX],
            "context_member": list(CONTEXT_MEMBER)}


#: mean over K picks candidate 1 (1 m off); min over K would pick candidate 2 --
#: the aggregation choice is therefore observable in every derived number.
DIST_MEAN_PICKS_1 = [[0.90, 0.90], [0.40, 0.40], [0.00, 1.60]]
CTX_PREFERS_FIRST = [0.10, 0.50]
CTX_PREFERS_SECOND = [0.50, 0.10]


def _fixture(tmp_path, n_queries=4):
    """Four queries over two rooms; family "m1" is the one with known answers."""
    metrics_path = os.path.join(str(tmp_path), "metrics.jsonl")
    rows_path = os.path.join(str(tmp_path), "rows.jsonl")
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(metrics_path, "w") as mh, open(rows_path, "w") as rh:
        for q in range(n_queries):
            qid, room = f"q{q}", f"Room/R{q % 2}"
            families = {"m1": (DIST_MEAN_PICKS_1, CTX_PREFERS_FIRST)}
            mh.write(json.dumps(_metrics_row(qid, room, q, families)) + "\n")
            rh.write(json.dumps(_replay_row(qid, room)) + "\n")
    return metrics_path, rows_path


# --------------------------------------------------------------------------- #
# decoding and per-query records
# --------------------------------------------------------------------------- #
def test_decode_round_trips_the_exact_float32_payload():
    matrix = torch.tensor([[0.25, 0.5], [1.0, 2.0]], dtype=torch.float32)
    assert np.array_equal(mr.decode_matrix(encode_sims(matrix)), matrix.numpy())
    assert np.array_equal(mr.decode_vector(_hex_vector([0.5, 0.25])),
                          np.array([0.5, 0.25], dtype=np.float32))


@pytest.mark.parametrize("how", ["mean", "min", "median", "lme"])
def test_family_scores_are_the_recorded_aggregation(how):
    row = _metrics_row("q", "Room/R0", 0, {"m1": (DIST_MEAN_PICKS_1, CTX_PREFERS_FIRST)})
    scores = mr.family_scores(row, "m1", how)
    expected = rm.aggregate_over_k(torch.tensor(DIST_MEAN_PICKS_1), how).numpy()
    assert np.allclose(scores, expected, atol=0, rtol=0)


def test_family_record_uses_the_primary_mean_aggregation_and_world_geometry():
    row = _metrics_row("q", "Room/R0", 0, {"m1": (DIST_MEAN_PICKS_1, CTX_PREFERS_FIRST)})
    record = mr.family_record(row, "m1")
    assert record["pred_index"] == 1                    # mean, not min
    assert record["e_loc"] == pytest.approx(1.0)        # world frame, not camera
    assert record["top1"] == 0.0
    # mean over K: cand1 0.40 < cand2 0.80 < cand0 0.90, so the GT ranks third
    assert record["rr"] == pytest.approx(1.0 / 3)
    assert record["room_id"] == "Room/R0" and record["query_id"] == "q"
    assert record["context_member_pred"] is True        # candidate 1 is in context

    on_min = mr.family_record(row, "m1", aggregation="min")
    assert on_min["pred_index"] == 2 and on_min["e_loc"] == pytest.approx(3.0)


def test_reciprocal_rank_matches_the_registered_driver_convention():
    """The campaign's rank convention lives in the driver; this is the same
    function read for distances instead of similarities."""
    import eval_localization as el
    generator = np.random.default_rng(0)
    for _ in range(25):
        distances = generator.choice([0.1, 0.2, 0.3], size=6)
        for gt in range(6):
            assert mr.reciprocal_rank(distances, gt) == pytest.approx(
                el.gt_reciprocal_rank(torch.tensor(-distances), gt))


# --------------------------------------------------------------------------- #
# the metric-matched retrieval control: identical geometry to the registered one
# --------------------------------------------------------------------------- #
def test_retrieval_record_is_the_registered_nearest_context_geometry():
    from src.localization.scoring import nearest_context_baseline

    row = _metrics_row("q", "Room/R0", 0, {"m1": (DIST_MEAN_PICKS_1, CTX_PREFERS_SECOND)})
    replay = _replay_row("q", "Room/R0")
    raw = mr.retrieval_record(row, replay, "m1", masked=False)
    masked = mr.retrieval_record(row, replay, "m1", masked=True)

    expected_raw = nearest_context_baseline(CAND_CAM, CTX_CAM,
                                            -torch.tensor(CTX_PREFERS_SECOND))
    expected_masked = nearest_context_baseline(
        CAND_CAM, CTX_CAM, -torch.tensor(CTX_PREFERS_SECOND),
        eligible_mask=[not m for m in CONTEXT_MEMBER])
    assert raw["pred_index"] == expected_raw == 2
    assert masked["pred_index"] == expected_masked == 2
    assert raw["e_loc"] == pytest.approx(3.0)

    closer = _metrics_row("q", "Room/R0", 0, {"m1": (DIST_MEAN_PICKS_1, CTX_PREFERS_FIRST)})
    assert mr.retrieval_record(closer, replay, "m1", masked=False)["pred_index"] == 1
    # masking the context member moves the same control to an eligible candidate
    assert mr.retrieval_record(closer, replay, "m1", masked=True)["pred_index"] == 0


def test_agree_retrieval_record_reads_similarities_not_distances():
    from src.localization.scoring import nearest_context_baseline

    replay = _replay_row("q", "Room/R0", ctx_sims=(0.1, 0.9))     # second context wins
    record = mr.agree_retrieval_record(replay, masked=False)
    assert record["pred_index"] == nearest_context_baseline(
        CAND_CAM, CTX_CAM, torch.tensor([0.1, 0.9])) == 2
    assert mr.agree_retrieval_record(_replay_row("q", "Room/R0", ctx_sims=(0.9, 0.1)),
                                     masked=False)["pred_index"] == 1


def test_agree_record_reads_the_recorded_flac_prediction():
    row = _metrics_row("q", "Room/R0", 0, {"m1": (DIST_MEAN_PICKS_1, CTX_PREFERS_FIRST)},
                       agree_pred=2)
    record = mr.agree_record(row)
    assert record["pred_index"] == 2 and record["e_loc"] == pytest.approx(3.0)
    assert record["top1"] == 0.0 and record["context_member_pred"] is False


# --------------------------------------------------------------------------- #
# summaries and stream joining
# --------------------------------------------------------------------------- #
def test_summarize_records_matches_the_campaign_summary_block():
    records = [{"query_id": "a", "room_id": "R0", "e_loc": 0.0, "top1": 1.0, "rr": 1.0},
               {"query_id": "b", "room_id": "R0", "e_loc": 0.4, "top1": 0.0, "rr": 0.5},
               {"query_id": "c", "room_id": "R1", "e_loc": 0.8, "top1": 0.0, "rr": 0.5},
               {"query_id": "d", "room_id": "R1", "e_loc": 3.0, "top1": 0.0, "rr": 0.25}]
    summary = mr.summarize_records(records)
    assert summary["pooled"]["median_e_loc"] == pytest.approx(0.6)
    assert summary["pooled"]["mean_e_loc"] == pytest.approx(1.05)
    assert summary["pooled"]["success"][0.5] == pytest.approx(0.5)
    assert summary["pooled"]["success"][1.0] == pytest.approx(0.75)
    assert summary["pooled"]["top1"] == pytest.approx(0.25)
    assert summary["pooled"]["mrr"] == pytest.approx((1.0 + 0.5 + 0.5 + 0.25) / 4)
    assert summary["macro"]["n_rooms"] == 2
    assert summary["macro"]["mean_of_room_means"] == pytest.approx((0.2 + 1.9) / 2)
    assert set(summary["per_room"]) == {"R0", "R1"}


def test_scan_seed_joins_the_two_streams_and_refuses_a_mismatch(tmp_path):
    metrics_path, rows_path = _fixture(tmp_path)
    scan = mr.scan_seed(metrics_path, rows_path, families=("m1",))
    assert scan["n_queries"] == 4 and scan["n_rooms"] == 2
    assert len(scan["records"]["m1"]["mean"]) == 4
    assert scan["records"]["m1"]["mean"][0]["e_loc"] == pytest.approx(1.0)
    assert len(scan["retrieval"]["m1"]["masked"]) == 4
    assert len(scan["agree"]) == 4 and len(scan["agree_retrieval"]["masked"]) == 4

    bad = os.path.join(str(tmp_path), "bad_rows.jsonl")
    with open(rows_path) as handle:
        lines = handle.readlines()
    payload = json.loads(lines[2])
    payload["query_id"] = "not-the-same-query"
    lines[2] = json.dumps(payload) + "\n"
    with open(bad, "w") as handle:
        handle.writelines(lines)
    with pytest.raises(ValueError, match="position 2"):
        mr.scan_seed(metrics_path, bad, families=("m1",))


def test_scan_seed_refuses_streams_of_different_length(tmp_path):
    metrics_path, rows_path = _fixture(tmp_path)
    short = os.path.join(str(tmp_path), "short_rows.jsonl")
    with open(rows_path) as handle:
        lines = handle.readlines()
    with open(short, "w") as handle:
        handle.writelines(lines[:-1])
    with pytest.raises(ValueError, match="length"):
        mr.scan_seed(metrics_path, short, families=("m1",))


# --------------------------------------------------------------------------- #
# §4 comparisons: paired per-query differences, clustered CIs, Holm correction
# --------------------------------------------------------------------------- #
def _records(e_locs, top1s, rooms=("R0", "R0", "R1", "R1")):
    return [{"query_id": f"q{i}", "room_id": rooms[i], "e_loc": float(e),
             "top1": float(t), "rr": 1.0, "pred_index": 0, "context_member_pred": False}
            for i, (e, t) in enumerate(zip(e_locs, top1s))]


def test_compare_records_reports_the_paired_primary_and_the_top1_difference():
    better = _records([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0])
    worse = _records([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0])
    comparison = mr.compare_records(better, worse, "m1_vs_matched", n_boot=200, seed=0)

    assert comparison["label"] == "m1_vs_matched"
    assert comparison["e_loc"]["point"] == pytest.approx(-0.5)   # median of [-1,-1,0,0]
    assert comparison["e_loc"]["stat"] == "median_paired_difference"
    assert comparison["top1"]["point"] == pytest.approx(0.5)     # mean of [1,1,0,0]
    assert comparison["top1_a"] == pytest.approx(0.5)
    assert comparison["top1_b"] == pytest.approx(0.0)
    assert comparison["top1_delta"] == pytest.approx(0.5)
    assert comparison["quantity"] == {"e_loc": "e_loc_metres", "top1": "top1_indicator"}
    for block in (comparison["e_loc"], comparison["top1"]):
        assert 0.0 <= block["p_value"] <= 1.0 and block["n_clusters"] == 2


def test_compare_records_refuses_inputs_that_are_not_the_same_queries():
    left = _records([0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0])
    right = _records([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0])
    right[2]["query_id"] = "somewhere-else"
    with pytest.raises(ValueError):
        mr.compare_records(left, right, "bad", n_boot=50)


def test_holm_over_delegates_to_the_registered_helper():
    comparisons = [
        {"label": "a", "e_loc": {"p_value": 0.001}, "top1": {"p_value": 0.20}},
        {"label": "b", "e_loc": {"p_value": 0.04}, "top1": {"p_value": 0.30}},
        {"label": "c", "e_loc": {"p_value": 0.60}, "top1": {"p_value": 0.90}},
    ]
    holm = mr.holm_over(comparisons, quantity="e_loc")
    assert holm == rm.holm_bonferroni({"a": 0.001, "b": 0.04, "c": 0.60})
    # 0.04 x 2 = 0.08 > alpha, so only the smallest p survives the step-down
    assert [t["rejected"] for t in holm["tests"]] == [True, False, False]
    assert mr.holm_over(comparisons, quantity="top1")["n_tests"] == 3


# --------------------------------------------------------------------------- #
# §3 controls
# --------------------------------------------------------------------------- #
def test_power_by_family_is_the_registered_power_statistic():
    from src.localization.scoring import power_statistic
    row = _metrics_row("q", "Room/R0", 0, {"m1": (DIST_MEAN_PICKS_1, CTX_PREFERS_FIRST)})
    assert mr.power_by_family(row, "m1") == pytest.approx(
        power_statistic(torch.tensor(DIST_MEAN_PICKS_1)))


def test_power_by_family_returns_none_for_a_degenerate_family():
    """M4 empties its own validity mask on some queries, and the whole [M, K]
    block is then NaN: that query is a skip, never a zero."""
    row = _metrics_row("q", "Room/R0", 0, {"m1": (DIST_MEAN_PICKS_1, CTX_PREFERS_FIRST)})
    nan_matrix = torch.full((3, 2), float("nan"))
    row["families"]["m4"] = {"candidates_hex": encode_sims(nan_matrix),
                             "context_hex": _hex_vector([float("nan")] * 2),
                             "aggregations": {}, "pred_index": 0, "correct": False}
    assert mr.power_by_family(row, "m4") is None
    single = torch.tensor([[0.5], [0.7], [0.9]])
    row["families"]["m4"]["candidates_hex"] = encode_sims(single)
    assert mr.power_by_family(row, "m4") is None


def test_context_split_reports_both_buckets():
    records = _records([0.0, 1.0, 0.0, 2.0], [1.0, 0.0, 1.0, 0.0])
    records[0]["context_member_pred"] = True
    records[1]["context_member_pred"] = True
    split = mr.context_split(records)
    assert split["context"]["n_queries"] == 2 and split["non_context"]["n_queries"] == 2
    assert split["context"]["top1"] == pytest.approx(0.5)
    assert split["non_context"]["top1"] == pytest.approx(0.5)
    assert split["context_member_rate"] == pytest.approx(0.5)
    assert split["context"]["median_e_loc"] == pytest.approx(0.5)


def _m4_block(features, obs, mask=None, dropped=None):
    features = np.asarray(features, dtype=float)
    n_features = features.shape[-1]
    mask = [True] * n_features if mask is None else list(mask)
    return {"features": features.tolist(), "obs_features": list(obs), "mask": mask,
            "context_features": np.zeros((2, n_features)).tolist(),
            "dropped": dropped or {"n_features": n_features, "n_kept": int(sum(mask)),
                                   "n_dropped": int(n_features - sum(mask)),
                                   "dropped": [], "causes": {}}}


def test_m4_diagnostics_score_each_feature_and_count_the_drops():
    # feature 0 puts the GT (candidate 0) closest to the observation; feature 1
    # points at candidate 2 instead, so their single-feature top-1 differs.
    features = [[[0.0, 3.0], [0.0, 3.0]], [[5.0, 2.0], [5.0, 2.0]], [[9.0, 1.0], [11.0, 1.0]]]
    rows = [_metrics_row("q0", "Room/R0", 0, {"m1": (DIST_MEAN_PICKS_1, CTX_PREFERS_FIRST)},
                         m4=_m4_block(features, [0.1, 1.0]))]
    rows.append(_metrics_row("q1", "Room/R0", 1,
                             {"m1": (DIST_MEAN_PICKS_1, CTX_PREFERS_FIRST)},
                             m4=_m4_block(features, [0.1, 1.0], mask=[True, False],
                                          dropped={"n_features": 2, "n_kept": 1,
                                                   "n_dropped": 1, "dropped": ["f1"],
                                                   "causes": {"f1": {"obs_invalid": True}}})))
    accumulator = mr.M4Accumulator(names=("f0", "f1"))
    for row in rows:
        accumulator.add(row)
    diagnostics = accumulator.result()

    assert diagnostics["n_queries"] == 2
    assert diagnostics["dropped"]["n_queries_with_a_drop"] == 1
    assert diagnostics["dropped"]["total_dropped"] == 1
    assert diagnostics["dropped"]["per_feature"]["f1"] == 1
    per_feature = {entry["feature"]: entry for entry in diagnostics["per_feature"]}
    assert per_feature["f0"]["top1"] == pytest.approx(1.0)
    assert per_feature["f1"]["top1"] == pytest.approx(0.0)
    # population variance (ddof = 0), the convention the calibration diagnostics use
    assert per_feature["f0"]["within_var"] == pytest.approx(np.mean([0.0, 0.0, 1.0]))
    assert per_feature["f1"]["within_var"] == pytest.approx(0.0)
    assert per_feature["f1"]["power"] == float("inf") or math.isinf(per_feature["f1"]["power"])


def _battery(distances_by_variant):
    return {variant: {"m1": encode_sims(torch.tensor(dist, dtype=torch.float32))}
            for variant, dist in distances_by_variant.items()}


def test_sensitivity_summary_compares_every_variant_with_its_baseline():
    unchanged = DIST_MEAN_PICKS_1                       # same prediction as baseline
    moved = [[0.10, 0.10], [0.80, 0.80], [0.90, 0.90]]  # now predicts the GT
    rows = [_metrics_row("q0", "Room/R0", 0, {"m1": (DIST_MEAN_PICKS_1, CTX_PREFERS_FIRST)},
                         sensitivities=_battery({"gain_x2": unchanged,
                                                 "direct_crop_2p5ms": moved}))]
    summary = mr.sensitivity_summary(rows)
    assert summary["n_rows_with_battery"] == 1
    assert summary["baseline_top1"]["m1"] == pytest.approx(0.0)
    assert summary["variants"]["gain_x2"]["m1"]["top1"] == pytest.approx(0.0)
    assert summary["variants"]["gain_x2"]["m1"]["prediction_change_rate"] == pytest.approx(0.0)
    assert summary["variants"]["direct_crop_2p5ms"]["m1"]["top1"] == pytest.approx(1.0)
    assert summary["variants"]["direct_crop_2p5ms"]["m1"][
        "prediction_change_rate"] == pytest.approx(1.0)
    assert summary["variants"]["gain_x2"]["m1"]["mean_abs_score_change"] == pytest.approx(0.0)


def test_sensitivity_summary_is_explicit_when_no_row_carries_a_battery():
    rows = [_metrics_row("q0", "Room/R0", 0, {"m1": (DIST_MEAN_PICKS_1, CTX_PREFERS_FIRST)})]
    summary = mr.sensitivity_summary(rows)
    assert summary["n_rows_with_battery"] == 0 and summary["variants"] == {}
    assert "not" in summary["status"]


def test_scan_seed_accumulates_power_m4_and_the_battery(tmp_path):
    features = [[[0.0, 3.0], [0.0, 3.0]], [[5.0, 2.0], [5.0, 2.0]], [[9.0, 1.0], [11.0, 1.0]]]
    metrics_path = os.path.join(str(tmp_path), "m.jsonl")
    rows_path = os.path.join(str(tmp_path), "r.jsonl")
    with open(metrics_path, "w") as mh, open(rows_path, "w") as rh:
        for q in range(3):
            row = _metrics_row(f"q{q}", f"Room/R{q % 2}", q,
                               {"m1": (DIST_MEAN_PICKS_1, CTX_PREFERS_FIRST)},
                               m4=_m4_block(features, [0.1, 1.0]),
                               sensitivities=_battery({"gain_x2": DIST_MEAN_PICKS_1}))
            mh.write(json.dumps(row) + "\n")
            rh.write(json.dumps(_replay_row(f"q{q}", f"Room/R{q % 2}")) + "\n")
    scan = mr.scan_seed(metrics_path, rows_path, families=("m1",))
    assert len(scan["power"]["m1"]) == 3
    assert scan["m4"]["n_queries"] == 3
    assert scan["sensitivity"]["n_rows_with_battery"] == 3


def test_seen_vs_unseen_puts_the_two_splits_side_by_side():
    seen = {"m1": {"pooled": {"top1": 0.8, "median_e_loc": 0.0, "mean_e_loc": 0.4}}}
    unseen = {"m1": {"pooled": {"top1": 0.5, "median_e_loc": 0.5, "mean_e_loc": 1.0}},
              "m2_complex": {"pooled": {"top1": 0.4, "median_e_loc": 1.0, "mean_e_loc": 2.0}}}
    table = mr.seen_vs_unseen(seen, unseen)
    assert table["m1"]["seen_top1"] == pytest.approx(0.8)
    assert table["m1"]["unseen_top1"] == pytest.approx(0.5)
    assert table["m1"]["top1_gap"] == pytest.approx(0.3)
    assert table["m2_complex"]["seen_top1"] is None      # no seen counterpart
    assert table["m2_complex"]["status"].startswith("no seen")


# --------------------------------------------------------------------------- #
# report assembly, the six conclusion questions, and the markdown block
# --------------------------------------------------------------------------- #
def _seed_fixture(tmp_path, seed, family_distances, ctx_distances, agree_pred=1,
                  n_queries=6, ctx_sims=(0.9, 0.1)):
    metrics_path = os.path.join(str(tmp_path), f"m{seed}.jsonl")
    rows_path = os.path.join(str(tmp_path), f"r{seed}.jsonl")
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(metrics_path, "w") as mh, open(rows_path, "w") as rh:
        for q in range(n_queries):
            qid, room = f"q{q}", f"Room/R{q % 3}"
            row = _metrics_row(qid, room, q,
                               {"m1": (family_distances, ctx_distances)},
                               agree_pred=agree_pred)
            mh.write(json.dumps(row) + "\n")
            rh.write(json.dumps(_replay_row(qid, room, ctx_sims=ctx_sims)) + "\n")
    return metrics_path, rows_path


#: distances whose mean over K puts the GT (candidate 0) first
DIST_MEAN_PICKS_GT = [[0.10, 0.10], [0.40, 0.40], [0.90, 0.90]]


def test_build_seed_report_covers_the_families_and_the_primary_tests(tmp_path):
    paths = _seed_fixture(tmp_path, 42, DIST_MEAN_PICKS_GT, CTX_PREFERS_SECOND)
    scan = mr.scan_seed(*paths, families=("m1",))
    report = mr.build_seed_report(scan, seed=42, n_boot=100)

    assert report["seed"] == 42 and report["n_queries"] == 6 and report["n_rooms"] == 3
    block = report["families"]["m1"]
    assert block["primary"]["pooled"]["top1"] == pytest.approx(1.0)
    assert block["aggregations"]["min"]["pooled"]["top1"] == pytest.approx(1.0)
    assert block["retrieval"]["masked"]["pooled"]["top1"] is not None
    assert block["context_split"]["context_member_rate"] == pytest.approx(0.0)
    assert block["power"]["n_queries"] == 6
    labels = [c["label"] for c in report["comparisons"]]
    assert "m1_vs_agree_retrieval" in labels and "m1_vs_matched_retrieval" in labels
    assert report["holm"]["e_loc"]["n_tests"] == len(report["primary_comparisons"])
    assert report["agree"]["pooled"]["top1"] == pytest.approx(0.0)   # agree_pred = 1


def test_conclusions_answer_all_six_questions_from_computed_values(tmp_path):
    paths = _seed_fixture(tmp_path, 42, DIST_MEAN_PICKS_GT, CTX_PREFERS_SECOND)
    scan = mr.scan_seed(*paths, families=("m1",))
    seed_reports = [mr.build_seed_report(scan, seed=42, n_boot=100)]
    answers = mr.conclusions(seed_reports, families=("m1",))

    assert set(answers) == {"q1_exceeds_agree_retrieval", "q2_beats_own_matched_control",
                            "q3_seed_and_room_consistent", "q4_reduces_context_member_failure",
                            "q5_robustness_caveats", "q6_adds_information_vs_different_scorer"}
    q1 = answers["q1_exceeds_agree_retrieval"]["m1"]
    assert q1["top1_mean"] == pytest.approx(1.0)
    assert q1["reference"] == pytest.approx(mr.AGREE_RETRIEVAL_REFERENCE)
    assert q1["exceeds"] is True and q1["delta_vs_reference"] == pytest.approx(0.311)
    q4 = answers["q4_reduces_context_member_failure"]["m1"]
    assert q4["context_member_rate"] == pytest.approx(0.0)
    assert q4["agree_reference"] == pytest.approx(mr.AGREE_CONTEXT_MEMBER_RATE)
    assert q4["reduces"] is True
    q6 = answers["q6_adds_information_vs_different_scorer"]["m1"]
    assert q6["contingency"]["family_right_agree_wrong"] == 6
    assert q6["union_top1"] == pytest.approx(1.0)
    assert q6["rescue_rate"] == pytest.approx(1.0)
    for key, block in answers.items():
        assert "rule" in block, f"{key} states no decision rule"


def test_seed_table_reports_mean_and_sd(tmp_path):
    reports = []
    for seed, distances in ((42, DIST_MEAN_PICKS_GT), (43, DIST_MEAN_PICKS_1)):
        paths = _seed_fixture(tmp_path / str(seed), seed, distances, CTX_PREFERS_SECOND)
        scan = mr.scan_seed(*paths, families=("m1",))
        reports.append(mr.build_seed_report(scan, seed=seed, n_boot=100))
    table = mr.seed_table(reports, families=("m1",))
    assert table["m1"]["top1"]["per_seed"] == {"42": 1.0, "43": 0.0}
    assert table["m1"]["top1"]["mean"] == pytest.approx(0.5)
    assert table["m1"]["top1"]["sd"] == pytest.approx(0.5)     # population SD, n = 2
    assert table["m1"]["median_e_loc"]["per_seed"]["43"] == pytest.approx(1.0)


def test_render_markdown_has_the_required_tables(tmp_path):
    paths = _seed_fixture(tmp_path, 42, DIST_MEAN_PICKS_GT, CTX_PREFERS_SECOND)
    scan = mr.scan_seed(*paths, families=("m1",))
    report = mr.build_report([mr.build_seed_report(scan, seed=42, n_boot=100)],
                             families=("m1",))
    text = mr.render_markdown(report)
    assert "| family |" in text and "m1" in text
    assert "0.689" in text                        # the fixed reference is named
    assert "declared-secondary" in text or "primary" in text
    assert "PRELIMINARY" in text
    for question in ("q1", "q2", "q3", "q4", "q5", "q6"):
        assert question in text


def test_build_report_is_deterministic(tmp_path):
    paths = _seed_fixture(tmp_path, 42, DIST_MEAN_PICKS_GT, CTX_PREFERS_SECOND)
    scan = mr.scan_seed(*paths, families=("m1",))
    first = mr.build_report([mr.build_seed_report(scan, seed=42, n_boot=100)],
                            families=("m1",))
    second = mr.build_report([mr.build_seed_report(scan, seed=42, n_boot=100)],
                             families=("m1",))
    assert json.dumps(first, sort_keys=True, default=str) == \
        json.dumps(second, sort_keys=True, default=str)


# --------------------------------------------------------------------------- #
# round trip on a slice of the REAL published stream
# --------------------------------------------------------------------------- #
_REAL_METRICS = ("outputs_loc/exp18/exp18_R4_unseen_flac_ctl-none_vanilla_ac-default_"
                 "lme_tau0.02_K8_seed42_scorer-AGREE_AR_registered_replay_metrics.jsonl")
_REAL_ROWS = ("outputs_loc/exp18/exp18_R4_unseen_flac_ctl-none_vanilla_ac-default_"
              "lme_tau0.02_K8_seed42_scorer-AGREE_AR_registered_replay_rows.jsonl")


@pytest.mark.skipif(not os.path.exists(_REAL_METRICS), reason="R4 unseen pass not present")
def test_real_slice_agrees_with_what_the_driver_recorded(tmp_path):
    """The offline aggregation must reproduce the predictions the online pass
    wrote into every row -- that equality is what makes it a re-reading rather
    than a second, differently-behaved scorer."""
    slice_metrics = os.path.join(str(tmp_path), "slice_metrics.jsonl")
    slice_rows = os.path.join(str(tmp_path), "slice_rows.jsonl")
    with open(_REAL_METRICS) as src, open(slice_metrics, "w") as dst:
        for line, _ in zip(src, range(12)):
            dst.write(line)
    with open(_REAL_ROWS) as src, open(slice_rows, "w") as dst:
        for line, _ in zip(src, range(12)):
            dst.write(line)

    seen_families = set()
    for row in mr.iter_rows(slice_metrics):
        for family, block in row["families"].items():
            record = mr.family_record(row, family)
            assert record["pred_index"] == int(block["pred_index"]), family
            assert bool(record["top1"]) == bool(block["correct"]), family
            seen_families.add(family)
    assert set(mr.PRIMARY_FAMILIES) <= seen_families
    assert set(mr.SECONDARY_FAMILIES) <= seen_families

    scan = mr.scan_seed(slice_metrics, slice_rows)
    assert scan["n_queries"] == 12
    report = mr.build_seed_report(scan, seed=42, n_boot=50)
    assert report["families"]["m1"]["primary"]["n_queries"] == 12


# --------------------------------------------------------------------------- #
# the driver's thin --mode metrics-report
# --------------------------------------------------------------------------- #
def test_driver_metrics_report_mode_writes_the_report_and_the_markdown(tmp_path):
    import eval_localization as el

    metrics_path, rows_path = _seed_fixture(tmp_path, 42, DIST_MEAN_PICKS_GT,
                                            CTX_PREFERS_SECOND)
    out_dir = str(tmp_path / "out")
    args = el.validate_args(el.parse_args(
        ["--mode", "metrics-report", "--model-config", "m.json",
         "--dataset-config", "d.json", "--out-dir", out_dir, "--eval-name", "R4_report",
         "--report-input", f"42:{metrics_path}:{rows_path}",
         "--report-bootstrap", "50", "--report-families", "m1"]))
    result = el.run_metrics_report(args)

    assert os.path.exists(result["report_path"]) and os.path.exists(result["markdown_path"])
    with open(result["report_path"]) as handle:
        payload = json.load(handle)
    assert payload["provenance"]["seeds"] == [42]
    assert payload["seed_table"]["m1"]["top1"]["mean"] == pytest.approx(1.0)
    assert "PRELIMINARY" in open(result["markdown_path"]).read()


def test_driver_metrics_report_refuses_a_malformed_input_spec(tmp_path):
    import eval_localization as el

    with pytest.raises(SystemExit, match="--report-input"):
        el.validate_args(el.parse_args(
            ["--mode", "metrics-report", "--model-config", "m.json",
             "--dataset-config", "d.json", "--out-dir", str(tmp_path),
             "--eval-name", "bad", "--report-input", "42:only-one-path"]))
    with pytest.raises(SystemExit, match="--report-input"):
        el.validate_args(el.parse_args(
            ["--mode", "metrics-report", "--model-config", "m.json",
             "--dataset-config", "d.json", "--out-dir", str(tmp_path),
             "--eval-name", "bad"]))
