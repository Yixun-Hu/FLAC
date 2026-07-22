"""exp-09 D-stage: threshold->verdict ADAPTER tests (CPU-only).

Covers the Codex D-tool review fixes:
  * F1 COVERAGE ENFORCEMENT: partial/over-complete inputs are REJECTED (exit 2), only the
    exact registered matrix is evaluated -- every review probe scenario => rejection.
  * F4 verbatim zero-sigma: sc==0 => n=inf => OUTSIDE (equal means => NOT equivalence).
  * F5 conditioning schema: the real A2b artifact (per-angle pooled_relerr/patch_relerr).
  * F7a: e2e/conditioning reduce over ALL cells (a LATER failing cell fails the gate).
  * F7b: the K=1 D1 arm and K=1 H-A3 arm are actually evaluated (a K=1-only failure fails).
  * plus the standing band math (gate_verdict.py verbatim), all-cells H-A3, D2 thresholds,
    matched-vs-contextual, atomic finite verdicts consumed by aggregate_gate.py, exit codes.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

_EXP09_DIR = Path(__file__).resolve().parents[1]
if str(_EXP09_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP09_DIR))

import gate_thresholds_to_verdicts as gtv  # noqa: E402
import aggregate_gate  # noqa: E402

# --------------------------------------------------------------------------------------- #
# registered matrix (mirrors what the D records will pin)
# --------------------------------------------------------------------------------------- #
_CTRL = {
    "8": {"T60": [8.609, 0.012], "C50": [0.9682, 0.0030], "EDT": [37.10, 0.07],
          "RIR_to_GT_RIR_R@1": [7.06, 0.10]},
    "1": {"T60": [9.969, 0.039], "C50": [1.0460, 0.0064], "EDT": [39.95, 0.37],
          "RIR_to_GT_RIR_R@1": [6.83, 0.22]},
}
_D1_EXPECT = {"K": [1, 8], "seeds": 5, "metrics": ["T60", "C50", "EDT"],
              "advisory_metrics": ["RIR_to_GT_RIR_R@1"]}
_COND_ANGLES = [45, 90, 180, 270]
_E2E_MATRIX = {"1": [0, 45, 90, 180, 270], "8": [0, 90]}
_FLAT_MATRIX = {"1": [45, 90, 180, 270], "8": [90]}
_FLAT_REF = {"1": {"T60": 9.969, "C50": 1.046, "EDT": 39.95},
             "8": {"T60": 8.609, "C50": 0.9682, "EDT": 37.10}}


# --------------------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------------------- #
def _d1cfg(mode="matched_control", **over):
    if mode == "contextual":
        cfg = {"mode": "contextual", "contextual_stats": _CTRL, "expect": dict(_D1_EXPECT),
               "note": "P1 pending"}
    else:
        cfg = {"mode": "matched_control", "control_name": "P1", "control_stats": _CTRL,
               "expect": dict(_D1_EXPECT)}
    cfg.update(over)
    return cfg


def _measured(overrides=None, seeds=5, ks=("1", "8")):
    overrides = overrides or {}
    out = {}
    for k in ks:
        out[k] = {}
        for metric, (mu, _sd) in _CTRL[k].items():
            out[k][metric] = [overrides.get((k, metric), mu)] * seeds
    return out


def _cond_artifact(overrides=None, angles=_COND_ANGLES):
    overrides = overrides or {}
    per = [{"angle": float(a),
            "pooled_relerr": overrides.get((a, "pooled"), 1e-5),
            "patch_relerr": overrides.get((a, "patch"), 3e-5)} for a in angles]
    return {"a2b": {"per_angle": per}}


def _e2e_index(overrides=None, matrix=_E2E_MATRIX):
    overrides = overrides or {}
    idx = {}
    for k, angs in matrix.items():
        idx[k] = {str(a): {"waveform_gap": {"mean_rel_l2": overrides.get((k, a), 0.002)}}
                  for a in angs}
    return idx


def _flat_index(overrides=None, matrix=_FLAT_MATRIX):
    overrides = overrides or {}
    idx = {}
    for k in matrix:
        idx[k] = {"0": dict(_FLAT_REF[k])}
        for a in matrix[k]:
            m = {metric: _FLAT_REF[k][metric] + overrides.get((k, a, metric), 0.0)
                 for metric in ("T60", "C50", "EDT")}
            idx[k][str(a)] = m
    return idx


def _write_eval_json(path, metrics):
    Path(path).write_text(json.dumps(
        {"metrics": dict(metrics), "ckpt_path": "x.ckpt", "rotate_deg": 0.0,
         "cond_method": "fa_invariant", "frame_avg_angles": [0.0], "cond_autocast": "bf16"}))
    return str(path)


def _run_cli(args):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, str(_EXP09_DIR / "gate_thresholds_to_verdicts.py"), *map(str, args)],
        capture_output=True, text=True, env=env, timeout=120)


def _write_refs(tmp_path, mode="matched_control"):
    if mode == "contextual":
        d1 = {"mode": "contextual", "contextual_stats": _CTRL, "expect": _D1_EXPECT, "note": "pending"}
    else:
        d1 = {"mode": "matched_control", "control_name": "P1", "control_stats": _CTRL,
              "expect": _D1_EXPECT}
    refs = {"d1": d1, "d2": {
        "conditioning": {"expect": {"angles": _COND_ANGLES}},
        "end_to_end": {"expect": {"matrix": _E2E_MATRIX}},
        "flatness": {"expect": {"matrix": _FLAT_MATRIX, "metrics": ["T60", "C50", "EDT"]}}}}
    p = tmp_path / "refs.json"; p.write_text(json.dumps(refs))
    return p


def _write_d1_measured(tmp_path, seeds=5, ks=("1", "8"), overrides=None):
    overrides = overrides or {}
    idx = {}
    for k in ks:
        paths = []
        for s in range(seeds):
            metrics = {m: overrides.get((k, m), mu) for m, (mu, _sd) in _CTRL[k].items()}
            paths.append(_write_eval_json(tmp_path / f"e_{k}_{s}.json", metrics))
        idx[k] = {"seeds": paths}
    p = tmp_path / f"d1_{'-'.join(ks)}_{seeds}.json"; p.write_text(json.dumps(idx))
    return p


# ======================================================================================= #
# seed_stats + band math (gate_verdict.py verbatim)
# ======================================================================================= #
def test_seed_stats_ddof1():
    mu, sd = gtv.seed_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mu == pytest.approx(3.0) and sd == pytest.approx(math.sqrt(2.5))


def test_band_equivalence_superior():
    b = gtv.sigma_band(8.59, 0.01, 8.60, 0.01, better_low=True)
    assert b["within_1sc"] and b["within_2sc"] and b["tier"] == "equiv<=1sc"
    assert b["direction"] == "superior"


def test_band_noninferiority_band():
    b = gtv.sigma_band(8.62, 0.01, 8.60, 0.01, better_low=True)  # n ~ +1.414
    assert not b["within_1sc"] and b["within_2sc"] and b["tier"] == "band<=2sc"


def test_band_outside_worse_fails():
    b = gtv.sigma_band(8.65, 0.01, 8.60, 0.01, better_low=True)
    assert not b["within_2sc"] and b["direction"] == "worse"


def test_band_higher_better_flips_direction():
    assert gtv.sigma_band(7.10, 0.10, 7.06, 0.10, better_low=False)["direction"] == "superior"
    assert gtv.sigma_band(6.90, 0.10, 7.06, 0.10, better_low=False)["direction"] == "worse"


def test_band_zero_sigma_verbatim_outside_not_equivalence():
    """F4: sc==0 => n=inf => OUTSIDE (fails 2sc), EVEN for equal means (gate_verdict.py:91).
    No non-finite value leaks into the emitted verdict."""
    b = gtv.sigma_band(8.60, 0.0, 8.60, 0.0, better_low=True)   # equal means, sc==0
    assert b["within_1sc"] is False and b["within_2sc"] is False
    assert b["tier"] == "outside>2sc"
    assert b["n_sigma"] is None and b["n_sigma_infinite"] is True
    assert b["direction"] == "tie"


# ======================================================================================= #
# D1 matched / contextual + F7b K=1 arm
# ======================================================================================= #
def test_d1_matched_all_within_passes():
    gate, adv = gtv.build_d1(_measured(), _d1cfg())
    assert adv is None and gate["gate"] == "d1_parity" and gate["pass"] is True
    assert gate["metrics"]["n_cells"] == 6  # K{1,8} x {T60,C50,EDT}
    assert any(c["metric"] == "RIR_to_GT_RIR_R@1" for c in gate["metrics"]["advisory"])


def test_d1_matched_k8_failure_fails():
    gate, _ = gtv.build_d1(_measured({("8", "T60"): 10.0}), _d1cfg())
    assert gate["pass"] is False


def test_d1_matched_k1_arm_failure_fails():
    """F7b: a K=1-only failure MUST fail the gate (the K=1 arm is actually evaluated)."""
    gate, _ = gtv.build_d1(_measured({("1", "T60"): 12.0}), _d1cfg())
    assert gate["pass"] is False
    bad = [c for c in gate["metrics"]["cells"] if c["K"] == "1" and c["metric"] == "T60"][0]
    assert not bad["within_2sc"]


def test_d1_contextual_emits_no_parity():
    gate, adv = gtv.build_d1(_measured(), _d1cfg(mode="contextual"))
    assert gate is None and adv["advisory"] is True and adv["gate"] != "d1_parity"
    assert "pass" not in adv


# ---- F1 coverage: partial D1 => rejection ----
def test_d1_reject_missing_seed():
    with pytest.raises(gtv.AdapterError):
        gtv.build_d1(_measured(seeds=1), _d1cfg())          # 1 seed, not 5


def test_d1_reject_missing_k():
    with pytest.raises(gtv.AdapterError):
        gtv.build_d1(_measured(ks=("8",)), _d1cfg())        # K=1 absent


def test_d1_reject_extra_k():
    m = _measured(); m["4"] = m["8"]                        # unexpected K=4
    with pytest.raises(gtv.AdapterError):
        gtv.build_d1(m, _d1cfg())


def test_d1_reject_missing_metric():
    m = _measured(); del m["8"]["EDT"]
    with pytest.raises(gtv.AdapterError):
        gtv.build_d1(m, _d1cfg())


# ======================================================================================= #
# D2 conditioning (A2b schema, F5) + coverage + F7a
# ======================================================================================= #
def test_cond_threshold_is_1e_4():
    assert gtv.D2_COND_THRESHOLD == 1e-4


def test_cond_from_a2b_artifact_passes():
    cells = gtv.conditioning_cells_from_artifact(_cond_artifact())
    v = gtv.build_conditioning_verdict(cells, _COND_ANGLES)
    assert v["gate"] == "d2_conditioning" and v["pass"] is True and v["metrics"]["n_angles"] == 4


def test_cond_later_cell_failure_fails_gate():
    """F7a: a LATER angle over threshold must fail (reduction is over ALL cells, not the first)."""
    cells = gtv.conditioning_cells_from_artifact(_cond_artifact({(270, "pooled"): 2e-4}))
    assert gtv.build_conditioning_verdict(cells, _COND_ANGLES)["pass"] is False


def test_cond_record_missing_a_channel_rejected():
    art = {"a2b": {"per_angle": [{"angle": 45.0, "pooled_relerr": 1e-5}]}}  # no patch_relerr
    with pytest.raises(gtv.AdapterError):
        gtv.conditioning_cells_from_artifact(art)


def test_cond_missing_angle_rejected():
    cells = gtv.conditioning_cells_from_artifact(_cond_artifact(angles=[45, 90, 180]))  # 270 missing
    with pytest.raises(gtv.AdapterError):
        gtv.build_conditioning_verdict(cells, _COND_ANGLES)


def test_cond_extra_angle_rejected():
    cells = gtv.conditioning_cells_from_artifact(_cond_artifact(angles=[45, 90, 135, 180, 270]))
    with pytest.raises(gtv.AdapterError):
        gtv.build_conditioning_verdict(cells, _COND_ANGLES)


# ======================================================================================= #
# D2 end-to-end + coverage + F7a
# ======================================================================================= #
def test_e2e_threshold_is_registered_bound():
    assert gtv.D2_E2E_REL_L2_THRESHOLD == 0.00931


def test_e2e_full_matrix_passes():
    cells = gtv.e2e_cells_from_index(_e2e_index())
    v = gtv.build_e2e_verdict(cells, _E2E_MATRIX)
    assert v["gate"] == "d2_end_to_end" and v["pass"] is True and v["metrics"]["n_cells"] == 7


def test_e2e_later_cell_failure_fails_gate():
    """F7a: a later (K,angle) over 0.00931 must fail."""
    cells = gtv.e2e_cells_from_index(_e2e_index({("8", 90): 0.010}))
    assert gtv.build_e2e_verdict(cells, _E2E_MATRIX)["pass"] is False


def test_e2e_missing_pair_rejected():
    idx = _e2e_index(); del idx["8"]["90"]
    with pytest.raises(gtv.AdapterError):
        gtv.build_e2e_verdict(gtv.e2e_cells_from_index(idx), _E2E_MATRIX)


def test_e2e_extra_pair_rejected():
    idx = _e2e_index(); idx["8"]["135"] = {"waveform_gap": {"mean_rel_l2": 0.001}}
    with pytest.raises(gtv.AdapterError):
        gtv.build_e2e_verdict(gtv.e2e_cells_from_index(idx), _E2E_MATRIX)


# ======================================================================================= #
# D2 H-A3 flatness: inlined constants, all-cells, coverage, F7b K=1 arm
# ======================================================================================= #
def test_h_a3_inlined_constants_exact():
    assert gtv.H_A3_THRESHOLDS[1] == {"T60": 0.080, "C50": 0.012, "EDT": 0.740}
    assert gtv.H_A3_THRESHOLDS[8] == {"T60": 0.024, "C50": 0.006, "EDT": 0.140}


def test_h_a3_full_matrix_passes():
    v = gtv.build_flatness_verdict(_flat_index(), _FLAT_MATRIX)
    assert v["gate"] == "d2_flatness" and v["pass"] is True
    assert v["metrics"]["n_cells"] == 15  # K1 x 4 angles x 3 + K8 x 1 angle x 3


def test_h_a3_one_cell_failure_fails_gate():
    """m3 all-cells: a single K=8 rot90 EDT over 0.140 fails the whole gate."""
    v = gtv.build_flatness_verdict(_flat_index({("8", 90, "EDT"): 0.2}), _FLAT_MATRIX)
    assert v["pass"] is False and v["metrics"]["n_failing"] == 1


def test_h_a3_k1_arm_failure_fails_gate():
    """F7b: a K=1-only failing cell MUST fail (the K=1 arm is evaluated)."""
    v = gtv.build_flatness_verdict(_flat_index({("1", 45, "T60"): 0.2}), _FLAT_MATRIX)
    assert v["pass"] is False


def test_h_a3_missing_k_rejected():
    idx = _flat_index(); del idx["1"]
    with pytest.raises(gtv.AdapterError):
        gtv.build_flatness_verdict(idx, _FLAT_MATRIX)


def test_h_a3_missing_rot0_rejected():
    idx = _flat_index(); del idx["8"]["0"]
    with pytest.raises(gtv.AdapterError):
        gtv.build_flatness_verdict(idx, _FLAT_MATRIX)


def test_h_a3_missing_rotated_angle_rejected():
    idx = _flat_index(); del idx["1"]["180"]
    with pytest.raises(gtv.AdapterError):
        gtv.build_flatness_verdict(idx, _FLAT_MATRIX)


def test_h_a3_extra_angle_rejected():
    idx = _flat_index(); idx["8"]["135"] = dict(_FLAT_REF["8"])
    with pytest.raises(gtv.AdapterError):
        gtv.build_flatness_verdict(idx, _FLAT_MATRIX)


# ======================================================================================= #
# aggregate_gate contract tie-in
# ======================================================================================= #
def test_emitted_verdict_is_valid_aggregate_gate_input(tmp_path):
    v = gtv.build_flatness_verdict(_flat_index(), _FLAT_MATRIX)
    p = tmp_path / "verdict_d2_flatness.json"
    aggregate_gate.write_verdict(str(p), v)
    assert aggregate_gate.aggregate([str(p)], ["d2_flatness"])["all_pass"] is True


# ======================================================================================= #
# CLI: F1 probe rejections, F3 fresh-dir, exit codes, atomic finite
# ======================================================================================= #
def test_cli_full_matrix_all_pass_exit_0(tmp_path):
    refs = _write_refs(tmp_path, "matched_control")
    d1 = _write_d1_measured(tmp_path)
    cond = tmp_path / "cond.json"; cond.write_text(json.dumps(_cond_artifact()))
    e2e = tmp_path / "e2e.json"; e2e.write_text(json.dumps(_e2e_index()))
    flat = tmp_path / "flat.json"; flat.write_text(json.dumps(_flat_index()))
    out = tmp_path / "verdicts"
    proc = _run_cli(["--references", refs, "--out-dir", out, "--d1", d1,
                     "--d2-cond", cond, "--d2-e2e", e2e, "--d2-flatness", flat])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for gate in ("d1_parity", "d2_conditioning", "d2_end_to_end", "d2_flatness"):
        rec = json.loads((out / f"verdict_{gate}.json").read_text())
        assert rec["gate"] == gate and isinstance(rec["pass"], bool)
    assert not list(out.glob(".*tmp*"))


def test_cli_failing_gate_exit_1(tmp_path):
    refs = _write_refs(tmp_path, "matched_control")
    d1 = _write_d1_measured(tmp_path, overrides={("8", "T60"): 12.0})
    out = tmp_path / "verdicts"
    proc = _run_cli(["--references", refs, "--out-dir", out, "--d1", d1])
    assert proc.returncode == 1, proc.stdout + proc.stderr


# ---- F1: each review probe scenario => REJECTION (exit 2), nothing written ----
def test_cli_reject_d1_partial_one_seed(tmp_path):
    refs = _write_refs(tmp_path, "matched_control")
    d1 = _write_d1_measured(tmp_path, seeds=1)              # review probe: 1 seed
    out = tmp_path / "verdicts"
    proc = _run_cli(["--references", refs, "--out-dir", out, "--d1", d1])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not (out.exists() and list(out.glob("*.json")))


def test_cli_reject_d1_one_k(tmp_path):
    refs = _write_refs(tmp_path, "matched_control")
    d1 = _write_d1_measured(tmp_path, ks=("8",))           # review probe: K=8 only
    proc = _run_cli(["--references", refs, "--out-dir", tmp_path / "v", "--d1", d1])
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_cli_reject_flatness_k8_only(tmp_path):
    refs = _write_refs(tmp_path, "contextual")
    idx = _flat_index(); del idx["1"]                      # review probe: K=8 rot0/90 only
    flat = tmp_path / "flat.json"; flat.write_text(json.dumps(idx))
    proc = _run_cli(["--references", refs, "--out-dir", tmp_path / "v", "--d2-flatness", flat])
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_cli_reject_conditioning_split(tmp_path):
    """The review's 'pooled@45 but patch@90' split: with the real per-angle-record schema,
    a record missing its patch channel is rejected outright."""
    refs = _write_refs(tmp_path, "contextual")
    art = {"a2b": {"per_angle": [{"angle": 45.0, "pooled_relerr": 1e-5},
                                 {"angle": 90.0, "patch_relerr": 3e-5}]}}
    cond = tmp_path / "cond.json"; cond.write_text(json.dumps(art))
    proc = _run_cli(["--references", refs, "--out-dir", tmp_path / "v", "--d2-cond", cond])
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_cli_reject_e2e_single_label(tmp_path):
    refs = _write_refs(tmp_path, "contextual")
    e2e = tmp_path / "e2e.json"
    e2e.write_text(json.dumps({"8": {"90": {"waveform_gap": {"mean_rel_l2": 0.002}}}}))  # one arb label
    proc = _run_cli(["--references", refs, "--out-dir", tmp_path / "v", "--d2-e2e", e2e])
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_cli_reject_missing_d2_expect_matrix(tmp_path):
    """No registered coverage matrix in the references => cannot enforce => reject."""
    refs = tmp_path / "refs.json"; refs.write_text(json.dumps({"d1": {"mode": "contextual"}}))
    flat = tmp_path / "flat.json"; flat.write_text(json.dumps(_flat_index()))
    proc = _run_cli(["--references", refs, "--out-dir", tmp_path / "v", "--d2-flatness", flat])
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_cli_fresh_dir_refuses_nonempty_out(tmp_path):
    """F3: a pre-existing verdict in the out dir => refusal, nothing new written."""
    refs = _write_refs(tmp_path, "matched_control")
    d1 = _write_d1_measured(tmp_path)
    out = tmp_path / "verdicts"; out.mkdir()
    stale = out / "verdict_d1_parity.json"; stale.write_text('{"stale": true}')
    proc = _run_cli(["--references", refs, "--out-dir", out, "--d1", d1])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert json.loads(stale.read_text()) == {"stale": True}  # untouched
    assert list(out.glob("*.json")) == [stale]               # nothing new written
