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
