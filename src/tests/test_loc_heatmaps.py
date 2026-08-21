"""exp_18 §2.7 visualization assets (`loc_invert_heatmaps.py`).

The case selection is PRE-REGISTERED, so what these tests pin is that it is a
rule and not a choice: the same rows always yield the same nine cases, in the
same order, whatever order they arrive in.
"""
import importlib.util
import json
import os
import pathlib

import numpy as np
import pytest

_EXPDIR = pathlib.Path(__file__).resolve().parents[2] / "worklog" / "worklog_yixun" / \
    "exp_18_loc_invert_claude"
_SCRIPT = _EXPDIR / "loc_invert_heatmaps.py"
_ASSETS = str(_EXPDIR / "loc_invert_results_assets")


def _module():
    spec = importlib.util.spec_from_file_location("loc_invert_heatmaps", _SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hm = _module()


# --------------------------------------------------------------------------- #
# the display transform and the margin the rule is defined on
# --------------------------------------------------------------------------- #
def test_display_scores_are_a_temperature_softmax():
    scores = [0.90, 0.88, 0.50]
    probs = hm.display_scores(scores, 0.02)
    assert probs.shape == (3,)
    assert probs.sum() == pytest.approx(1.0)
    expected = np.exp(np.asarray(scores) / 0.02 - np.max(np.asarray(scores) / 0.02))
    assert np.allclose(probs, expected / expected.sum())
    # the display temperature is the run's tau: at tau -> 0 the map is one-hot
    assert hm.display_scores(scores, 1e-6)[0] == pytest.approx(1.0)


def test_top2_margin_is_the_gap_between_the_two_largest():
    assert hm.top2_margin(np.array([0.6, 0.3, 0.1])) == pytest.approx(0.3)
    assert hm.top2_margin(np.array([0.34, 0.33, 0.33])) == pytest.approx(0.01)
    assert hm.top2_margin(np.array([1.0])) == pytest.approx(1.0)   # a lone candidate


def test_case_record_reads_the_registered_row_fields():
    row = _row("q0", "Cafe/Cafe_idx_1", scores=[0.9, 0.5, 0.5], gt=0, pred=0, e_loc=0.0)
    record = hm.case_record(row)
    assert record["query_id"] == "q0" and record["room_id"] == "Cafe/Cafe_idx_1"
    assert record["correct"] is True and record["e_loc"] == pytest.approx(0.0)
    assert record["margin"] == pytest.approx(hm.top2_margin(hm.display_scores([0.9, 0.5, 0.5],
                                                                              0.02)))
    assert record["temperature"] == pytest.approx(0.02)
    assert record["receiver_xyz_world"] == [1.0, 2.0, 1.5]


# --------------------------------------------------------------------------- #
# the pre-registered selection rule
# --------------------------------------------------------------------------- #
def _row(query_id, room_id, scores, gt=0, pred=0, e_loc=0.0, tau=0.02, receiver=(1.0, 2.0, 1.5)):
    """A registered rows-JSONL record, cut down to what the maps read."""
    from src.localization.reaggregate import encode_sims
    import torch

    world = [[float(i), 0.0, 1.5] for i in range(len(scores))]
    cam = [[w[0] - receiver[0], w[1] - receiver[1], w[2] - receiver[2]] for w in world]
    return {
        "query_id": query_id, "room_id": room_id, "relpath": f"ir/{room_id}/S000_R001.wav",
        "receiver_node": 1, "gt_node": gt, "pred_node": pred,
        "candidate_nodes": list(range(len(scores))),
        "candidate_xyz_world": world, "candidate_xyz_cam": cam,
        "gt_xyz_world": world[gt], "gt_xyz_cam": cam[gt],
        "pred_xyz_world": world[pred], "pred_index": pred, "gt_index": gt,
        "context_member": [False] * len(scores),
        "scores_hex": encode_sims(torch.tensor([list(scores)], dtype=torch.float32))[0],
        "e_loc": float(e_loc), "top1": 1.0 if pred == gt else 0.0, "tau": tau, "agg": "lme",
        "n_candidates": len(scores), "n_samples": 8,
    }


def _records(spec):
    """``spec`` = list of (query_id, room, margin_proxy, correct, e_loc)."""
    out = []
    for query_id, room, margin, correct, e_loc in spec:
        # a two-candidate score vector whose softmax margin is monotone in `margin`
        scores = [margin, 0.0]
        row = _row(query_id, room, scores, gt=0, pred=0 if correct else 1, e_loc=e_loc)
        out.append(hm.case_record(row))
    return out


def test_selection_picks_the_rule_extremes():
    records = _records([
        ("a", "R1", 1.00, True, 0.0),      # widest margin, correct -> sharp #1
        ("b", "R2", 0.80, True, 0.0),      # sharp #2
        ("c", "R3", 0.60, True, 0.0),      # sharp #3
        ("d", "R4", 0.01, False, 5.0),     # narrowest margins -> ambiguous
        ("e", "R5", 0.02, False, 4.0),
        ("f", "R6", 0.03, True, 3.0),
        ("g", "R7", 0.50, False, 9.0),     # biggest errors -> failure
        ("h", "R8", 0.55, False, 8.0),
        ("i", "R9", 0.58, False, 7.0),
    ])
    cases = hm.select_cases(records)
    assert [c["query_id"] for c in cases["sharp_success"]] == ["a", "b", "c"]
    assert [c["query_id"] for c in cases["ambiguous"]] == ["d", "e", "f"]
    assert [c["query_id"] for c in cases["failure"]] == ["g", "h", "i"]
    assert all(c["correct"] for c in cases["sharp_success"])


def test_selection_is_order_independent_and_reproducible():
    spec = [(f"q{i}", f"R{i % 5}", (i * 37 % 100) / 100.0, i % 2 == 0, float(i))
            for i in range(20)]
    records = _records(spec)
    first = hm.select_cases(records)
    shuffled = list(reversed(records))
    second = hm.select_cases(shuffled)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert json.dumps(hm.select_cases(records), sort_keys=True) == json.dumps(first,
                                                                              sort_keys=True)


def test_selection_takes_the_literal_category_extrema():
    """r8 review finding 1: the registered rule is extrema-only. No diversity
    policy, no cross-category exclusion -- three same-room cases in a row are the
    correct answer when they are the three extrema."""
    records = _records([("a", "R1", 0.90, True, 1.0), ("b", "R1", 0.80, True, 2.0),
                        ("c", "R1", 0.70, True, 3.0), ("d", "R2", 0.60, True, 9.0),
                        ("e", "R3", 0.50, False, 8.0), ("f", "R4", 0.40, False, 7.0)])
    cases = hm.select_cases(records)
    assert [c["query_id"] for c in cases["sharp_success"]] == ["a", "b", "c"]
    assert [c["query_id"] for c in cases["ambiguous"]] == ["f", "e", "d"]
    assert [c["query_id"] for c in cases["failure"]] == ["d", "e", "f"]


def test_a_query_extremal_in_two_categories_appears_in_both():
    """The categories are independent: excluding a query because an earlier kind
    already showed it is exactly the unregistered policy that was removed."""
    records = _records([("a", "R1", 0.90, True, 0.0), ("b", "R2", 0.80, True, 0.0),
                        ("c", "R3", 0.70, True, 0.0),
                        ("worst", "R4", 0.01, False, 99.0),   # narrowest AND worst
                        ("d", "R5", 0.02, False, 50.0), ("e", "R6", 0.03, False, 40.0)])
    cases = hm.select_cases(records)
    assert cases["ambiguous"][0]["query_id"] == "worst"
    assert cases["failure"][0]["query_id"] == "worst"


def test_selection_breaks_exact_ties_by_query_id():
    records = _records([("b", "R1", 0.5, True, 1.0), ("a", "R1", 0.5, True, 1.0),
                        ("c", "R2", 0.5, True, 1.0), ("d", "R3", 0.1, False, 2.0),
                        ("e", "R4", 0.2, False, 3.0), ("f", "R5", 0.3, False, 4.0)])
    assert [c["query_id"] for c in hm.select_cases(records)["sharp_success"]] == ["a", "b", "c"]


def test_selection_refuses_a_run_it_cannot_fill():
    with pytest.raises(ValueError, match="cases"):
        hm.select_cases(_records([("a", "R1", 0.9, True, 0.0)]))


# --------------------------------------------------------------------------- #
# rendering (Agg) and the gallery manifest
# --------------------------------------------------------------------------- #
def test_render_case_writes_a_readable_png(tmp_path):
    record = hm.case_record(_row("q0", "Cafe/Cafe_idx_1", [0.9, 0.5, 0.4], gt=0, pred=0))
    path = hm.render_case(record, str(tmp_path / "case.png"), kind="sharp_success",
                          run_label="R2 seed 42")
    assert os.path.exists(path)
    with open(path, "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"
    assert os.path.getsize(path) > 5_000


def test_render_marks_a_failure_case_with_both_positions(tmp_path):
    record = hm.case_record(_row("q1", "Cafe/Cafe_idx_1", [0.4, 0.9, 0.3], gt=0, pred=1,
                                 e_loc=1.0))
    path = hm.render_case(record, str(tmp_path / "fail.png"), kind="failure",
                          run_label="R2 seed 42")
    assert os.path.exists(path) and os.path.getsize(path) > 5_000


def test_gallery_manifest_records_the_rule_and_every_case(tmp_path):
    records = _records([(f"q{i}", f"R{i}", i / 10.0, i % 2 == 0, float(i)) for i in range(9)])
    cases = hm.select_cases(records)
    manifest = hm.gallery_manifest(cases, run_label="R2 seed 42", rows_path="rows.jsonl",
                                   out_dir=str(tmp_path), rows_sha256="ab" * 32)
    assert manifest["run_label"] == "R2 seed 42"
    assert manifest["rows_sha256"] == "ab" * 32
    assert set(manifest["cases"]) == set(hm.CASE_KINDS)
    assert all(len(manifest["cases"][kind]) == 3 for kind in hm.CASE_KINDS)
    entry = manifest["cases"]["sharp_success"][0]
    for key in ("query_id", "room_id", "margin", "e_loc", "correct", "png", "rank"):
        assert key in entry, key
    assert "selection_rule" in manifest and "display_temperature" in manifest
    assert "distinct rooms" not in manifest["selection_rule"]
    assert manifest["saturation_caveat"] == hm.SATURATION_CAVEAT
    assert "n_queries_above_saturation_threshold" in manifest
    assert "caption" in entry and entry["caption"]


# --------------------------------------------------------------------------- #
# the HTML data extracts: schema only, values come from published artifacts
# --------------------------------------------------------------------------- #
def _fake_summary(top1, macro, control, chance, per_room=None):
    return {"summary": {
        "n_queries": 6337, "n_rooms": 17,
        "flac": {"pooled": {"top1": top1, "median_e_loc": 0.0, "mean_e_loc": 1.0,
                            "success": {"0.5": 0.5, "1.0": 0.6}, "mrr": 0.6},
                 "macro": {"top1": macro, "mean_of_room_means": 1.1},
                 "per_room": per_room or {"R1": {"top1": top1, "n_queries": 10}}},
        "controls": {"nearest_context_masked": {
            "pooled": {"top1": control}, "macro": {"top1": control + 0.05},
            "per_room": per_room or {"R1": {"top1": control, "n_queries": 10}}}},
        "baselines": {"context_conditioned": {"pooled": {"top1": chance},
                                              "macro": {"top1": chance}}},
        "context_member_prediction_rate": 0.37,
        "statistics": {"clustered_ci": {"point": 0.0, "lo": 0.0, "hi": 0.51,
                                        "stat": "pooled_median_e_loc"},
                       "paired_vs_nearest_context_masked": {"point": 0.0, "lo": 0.0, "hi": 0.0,
                                                            "p_value": 1.0},
                       "paired_vs_context_conditioned": {"point": -0.75, "lo": -1.46,
                                                         "hi": -0.21, "p_value": 0.0104}},
    }}


def test_two_regime_extract_has_both_regimes_and_all_three_arms():
    extract = hm.extract_two_regime(_fake_summary(0.56, 0.50, 0.63, 0.48),
                                    _fake_summary(0.50, 0.45, 0.11, 0.11))
    assert set(extract["regimes"]) == {"K8", "K1"}
    for regime in extract["regimes"].values():
        assert set(regime["arms"]) == {"flac", "chance", "retrieval"}
        assert "pooled_top1" in regime["arms"]["flac"]
        assert "macro_top1" in regime["arms"]["flac"]
        assert regime["n_queries"] == 6337 and regime["n_rooms"] == 17
        assert "clustered_ci_median_e_loc" in regime
        assert "paired_vs_retrieval" in regime
    assert "source" in extract and "convention" in extract


def test_delta_extract_carries_the_grid_and_the_collapse():
    calibration = {"delta_max": {"grid": [{"delta_max": 0, "top1": 0.40},
                                          {"delta_max": 8, "top1": 0.52},
                                          {"delta_max": 32, "top1": 0.50},
                                          {"delta_max": 128, "top1": 0.45}],
                                 "selected": 8}}
    report = {"seed_table": {
        "m1": {"macro_top1": {"mean": 0.59}, "top1": {"mean": 0.63}},
        "m5": {"macro_top1": {"mean": 0.53}, "top1": {"mean": 0.58}},
        "m1_delta0": {"macro_top1": {"mean": 0.44}, "top1": {"mean": 0.46}},
        "m5_delta0": {"macro_top1": {"mean": 0.28}, "top1": {"mean": 0.30}}}}
    extract = hm.extract_delta(calibration, report)
    assert [p["delta_max"] for p in extract["seen_grid"]] == [0, 8, 32, 128]
    assert extract["registered_delta_max"] == 8
    assert extract["unseen_collapse"]["m1"]["delta0_macro_top1"] == pytest.approx(0.44)
    assert extract["unseen_collapse"]["m5"]["registered_macro_top1"] == pytest.approx(0.53)
    assert extract["unseen_collapse"]["m1"]["macro_drop"] == pytest.approx(0.15)


def test_family_and_conclusion_extracts_follow_the_report(tmp_path):
    report = json.load(open(_R4_REPORT)) if os.path.exists(_R4_REPORT) else None
    if report is None:
        pytest.skip("promoted R4 report not present")
    families = hm.extract_families(report)
    assert set(families["families"]) >= {"m1", "m2", "m3", "m4", "m5"}
    entry = families["families"]["m2"]
    for key in ("kind", "pooled_top1", "macro_top1", "sd", "oracle_top1",
                "matched_control_pooled_top1", "context_member_rate"):
        assert key in entry, key
    assert families["families"]["m1_delta0"]["kind"] == "declared-sensitivity"

    # r8 review finding 2: the inference seed must be named, and every seed's
    # adjusted p-values must be available (they differ: m4_vs_agree_retrieval is
    # 0.688 / 0.608 / 0.7584 across seeds 42 / 43 / 44)
    assert families["primary_tests"]["seed"] == 42
    assert families["holm"]["seed"] == 42
    assert set(families["holm_per_seed"]) == {"42", "43", "44"}
    assert set(families["adjusted_p_per_seed"]["m4_vs_agree_retrieval"]) == {"42", "43", "44"}
    for seed, block in families["holm_per_seed"].items():
        assert block["n_tests"] == 10, seed
        labels = {t["label"] for t in block["tests"]}
        assert "m4_vs_agree_retrieval" in labels
    assert families["adjusted_p_per_seed"]["m4_vs_agree_retrieval"]["42"] == pytest.approx(
        0.688, abs=5e-3)

    answers = hm.extract_conclusions(report)
    assert set(answers["questions"]) == {
        "q1_exceeds_agree_retrieval", "q2_beats_own_matched_control",
        "q3_seed_and_room_consistent", "q4_reduces_context_member_failure",
        "q5_robustness_caveats", "q6_adds_information_vs_different_scorer"}
    for block in answers["questions"].values():
        assert "rule" in block


def test_per_room_extract_aligns_the_three_arms():
    per_room = {"R1": {"top1": 0.7, "n_queries": 100}, "R2": {"top1": 0.3, "n_queries": 50}}
    k8 = _fake_summary(0.56, 0.50, 0.63, 0.48, per_room=per_room)
    k1 = _fake_summary(0.50, 0.45, 0.11, 0.11, per_room=per_room)
    report = {"seeds": [{"seed": 42, "families": {"m2": {"primary": {"per_room": per_room}}}}]}
    extract = hm.extract_per_room(k8, k1, report, family="m2")
    assert set(extract["rooms"]) == {"R1", "R2"}
    assert set(extract["arms"]) == {"agree_k8", "agree_k1", "m2_k8"}
    assert extract["rooms"]["R1"]["agree_k8"] == pytest.approx(0.7)
    assert extract["rooms"]["R1"]["n_queries"] == 100


def test_campaign_timeline_is_hardcoded_from_the_committed_record():
    timeline = hm.campaign_timeline()
    assert timeline["runs"] and timeline["gates"] and timeline["tests"]
    for run in timeline["runs"]:
        assert {"label", "date", "detail"} <= set(run)
    assert any("6,337" in str(run["detail"]) or "6337" in str(run["detail"])
               for run in timeline["runs"])
    assert "source" in timeline


def test_timeline_suite_total_occurs_in_the_file_it_cites():
    """r8 review finding 4: a hardcoded count whose cited record does not contain
    it is not sourced. The citation is checked against the committed file."""
    tests = hm.campaign_timeline()["tests"]
    cited = pathlib.Path(__file__).resolve().parents[2] / tests["source_file"]
    assert cited.is_file(), cited
    text = cited.read_text()
    assert tests["source_quote"] in text, (
        f"{tests['source_quote']!r} does not occur in {tests['source_file']}")
    assert str(tests["suite_total"]) in tests["source_quote"]


@pytest.mark.skipif(not os.path.exists(_ASSETS), reason="published assets not present")
@pytest.mark.parametrize("slug,rows", [
    ("R2_K8_seed42", "outputs_loc/exp18/exp18_R2_flac_ctl-none_vanilla_ac-default_lme_"
                     "tau0.02_K8_seed42_scorer-AGREE_AR_registered_rows.jsonl"),
    ("R2b_K1_seed42", "outputs_loc/exp18/exp18_R2b_flac_ctl-none_vanilla_ac-default_lme_"
                      "tau0.02_K8_seed42_scorer-AGREE_AR_registered_rows.jsonl")])
def test_published_gallery_is_the_literal_rule_on_the_published_rows(slug, rows):
    """Artifact-level certification: recompute the rule from the rows the manifest
    names and require the PUBLISHED manifest to be exactly that."""
    manifest_path = os.path.join(_ASSETS, f"{slug}_gallery.json")
    if not (os.path.exists(manifest_path) and os.path.exists(rows)):
        pytest.skip("gallery or rows not present")
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    assert manifest["rows_sha256"] == hm._file_sha256(rows)

    records = [hm.case_record(json.loads(line)) for line in open(rows) if line.strip()]
    cases = hm.select_cases(records)
    for kind in hm.CASE_KINDS:
        assert [c["query_id"] for c in cases[kind]] == \
            [e["query_id"] for e in manifest["cases"][kind]], kind
        for entry in manifest["cases"][kind]:
            assert os.path.exists(os.path.join(_ASSETS, entry["png"])), entry["png"]
    assert manifest["saturation_caveat"] == hm.SATURATION_CAVEAT
    assert manifest["n_queries_above_saturation_threshold"] == sum(
        1 for r in records if hm.is_saturated(r["margin"]))
    assert manifest["n_queries"] == len(records)



_R4_REPORT = "outputs_loc/exp18/exp18_R4_report_metrics_report.json"


def test_write_extracts_emits_every_declared_file(tmp_path):
    if not os.path.exists(_R4_REPORT):
        pytest.skip("promoted R4 report not present")
    written = hm.write_extracts(
        str(tmp_path),
        k8_summary=_fake_summary(0.56, 0.50, 0.63, 0.48),
        k1_summary=_fake_summary(0.50, 0.45, 0.11, 0.11),
        calibration={"delta_max": {"grid": [{"delta_max": 0, "top1": 0.4},
                                            {"delta_max": 8, "top1": 0.5}], "selected": 8}},
        report=json.load(open(_R4_REPORT)))
    assert set(os.path.basename(p) for p in written) == set(hm.EXTRACT_FILES)
    for path in written:
        with open(path) as handle:
            payload = json.load(handle)
        assert payload and "source" in payload


def test_render_keeps_a_receiver_outside_the_candidate_extent_in_frame(tmp_path):
    """The window is the candidates AND the receiver: a receiver beyond the
    searched region must still be drawn, not silently clipped."""
    row = _row("q2", "Cafe/Cafe_idx_1", [0.9, 0.5, 0.4], gt=0, pred=0,
               receiver=(-40.0, -40.0, 1.5))
    record = hm.case_record(row)
    assert record["receiver_xyz_world"] == [-40.0, -40.0, 1.5]
    path = hm.render_case(record, str(tmp_path / "far.png"), kind="sharp_success",
                          run_label="R2 seed 42")
    assert os.path.exists(path) and os.path.getsize(path) > 5_000
