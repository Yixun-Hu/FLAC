"""exp-09 D-stage: threshold->verdict ADAPTER tests (CPU-only).

Covers the Codex D-tool r1 + r2 review fixes:
  * F1 COVERAGE: partial/over-complete inputs REJECTED (exit 2); only the exact matrix runs.
  * P1-1 DUPLICATES: explicit seed IDs; repeated artifacts (realpath collision) + duplicate
    seed IDs + duplicate (K,angle)/(angle) cells all rejected BEFORE any set/dict normalisation.
  * P1-2 ANGLE CONTRACT: the A2b converter JOINS j->degrees via a2_params (the REAL committed
    audit_convention.json is consumed read-only); cells keyed by degrees incl. 11.25 (j=1).
  * P2 EXIT CONTRACT: raw ValueError/KeyError/TypeError/json parse errors -> exit 2, no traceback.
  * F4 verbatim zero-sigma; F5 A2b schema; F7a later-cell reductions; F7b K=1 arms.
  * band math (gate_verdict.py verbatim), all-cells H-A3, matched-vs-contextual, aggregate_gate.
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

_REAL_AUDIT = _EXP09_DIR / "audit_convention.json"   # the committed Stage-A artifact (read-only)

# --------------------------------------------------------------------------------------- #
# registered matrix (mirrors what the D records will pin) — seed IDs, degree angles, rot0-free e2e
# --------------------------------------------------------------------------------------- #
_CTRL = {
    "8": {"T60": [8.609, 0.012], "C50": [0.9682, 0.0030], "EDT": [37.10, 0.07],
          "RIR_to_GT_RIR_R@1": [7.06, 0.10]},
    "1": {"T60": [9.969, 0.039], "C50": [1.0460, 0.0064], "EDT": [39.95, 0.37],
          "RIR_to_GT_RIR_R@1": [6.83, 0.22]},
}
_SEED_IDS = [42, 43, 44, 45, 46]
_D1_EXPECT = {"K": [1, 8], "seeds": _SEED_IDS, "metrics": ["T60", "C50", "EDT"],
              "advisory_metrics": ["RIR_to_GT_RIR_R@1"]}
_COND_ANGLES = [11.25, 45, 90, 180, 270]          # A2b j->degrees (j=1 -> 11.25)
_E2E_MATRIX = {"1": [45, 90, 180, 270], "8": [90]}   # rot0-FREE (r2 adjudication)
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


def _measured(overrides=None, seed_ids=None, ks=("1", "8")):
    overrides = overrides or {}
    seed_ids = list(_SEED_IDS if seed_ids is None else seed_ids)
    out = {}
    for k in ks:
        vals = {metric: [overrides.get((k, metric), mu)] * len(seed_ids)
                for metric, (mu, _sd) in _CTRL[k].items()}
        out[k] = {"seed_ids": [str(s) for s in seed_ids], "values": vals}
    return out


def _cond_artifact(overrides=None, angles=_COND_ANGLES):
    overrides = overrides or {}
    per = [{"angle": float(a),
            "pooled_relerr": overrides.get((a, "pooled"), 1e-5),
            "patch_relerr": overrides.get((a, "patch"), 3e-5)} for a in angles]
    return {"a2b": {"per_angle": per}}


def _e2e_index(overrides=None, matrix=_E2E_MATRIX):
    overrides = overrides or {}
    return {k: {str(a): {"waveform_gap": {"mean_rel_l2": overrides.get((k, a), 0.002)}}
                for a in angs} for k, angs in matrix.items()}


def _flat_index(overrides=None, matrix=_FLAT_MATRIX):
    overrides = overrides or {}
    idx = {}
    for k in matrix:
        idx[k] = {"0": dict(_FLAT_REF[k])}
        for a in matrix[k]:
            idx[k][str(a)] = {m: _FLAT_REF[k][m] + overrides.get((k, a, m), 0.0)
                              for m in ("T60", "C50", "EDT")}
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


def _write_d1_measured(tmp_path, seed_ids=None, ks=("1", "8"), overrides=None, shared_path=False):
    seed_ids = list(_SEED_IDS if seed_ids is None else seed_ids)
    overrides = overrides or {}
    idx = {}
    for k in ks:
        seeds, shared = {}, None
        for sid in seed_ids:
            metrics = {m: overrides.get((k, m), mu) for m, (mu, _sd) in _CTRL[k].items()}
            if shared_path:
                shared = shared or _write_eval_json(tmp_path / f"e_{k}.json", metrics)
                seeds[str(sid)] = shared
            else:
                seeds[str(sid)] = _write_eval_json(tmp_path / f"e_{k}_{sid}.json", metrics)
        idx[k] = {"seeds": seeds}
    p = tmp_path / f"d1_{'-'.join(ks)}.json"; p.write_text(json.dumps(idx))
    return p


# ======================================================================================= #
# band math (gate_verdict.py verbatim) incl. F4 zero-sigma
# ======================================================================================= #
def test_seed_stats_ddof1():
    mu, sd = gtv.seed_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mu == pytest.approx(3.0) and sd == pytest.approx(math.sqrt(2.5))


def test_band_equivalence_superior():
    b = gtv.sigma_band(8.59, 0.01, 8.60, 0.01, better_low=True)
    assert b["within_1sc"] and b["within_2sc"] and b["tier"] == "equiv<=1sc" and b["direction"] == "superior"


def test_band_noninferiority_band():
    b = gtv.sigma_band(8.62, 0.01, 8.60, 0.01, better_low=True)
    assert not b["within_1sc"] and b["within_2sc"] and b["tier"] == "band<=2sc"


def test_band_outside_worse_fails():
    b = gtv.sigma_band(8.65, 0.01, 8.60, 0.01, better_low=True)
    assert not b["within_2sc"] and b["direction"] == "worse"


def test_band_zero_sigma_verbatim_outside_not_equivalence():
    b = gtv.sigma_band(8.60, 0.0, 8.60, 0.0, better_low=True)
    assert b["within_2sc"] is False and b["tier"] == "outside>2sc"
    assert b["n_sigma"] is None and b["n_sigma_infinite"] is True and b["direction"] == "tie"


# ======================================================================================= #
# D1 matched / contextual + F7b + P1-1 seed IDs / dedupe
# ======================================================================================= #
def test_d1_matched_all_within_passes():
    gate, adv = gtv.build_d1(_measured(), _d1cfg())
    assert adv is None and gate["gate"] == "d1_parity" and gate["pass"] is True
    assert gate["metrics"]["n_cells"] == 6


def test_d1_matched_k8_failure_fails():
    gate, _ = gtv.build_d1(_measured({("8", "T60"): 10.0}), _d1cfg())
    assert gate["pass"] is False


def test_d1_matched_k1_arm_failure_fails():
    gate, _ = gtv.build_d1(_measured({("1", "T60"): 12.0}), _d1cfg())
    assert gate["pass"] is False


def test_d1_contextual_emits_no_parity():
    gate, adv = gtv.build_d1(_measured(), _d1cfg(mode="contextual"))
    assert gate is None and adv["advisory"] is True and adv["gate"] != "d1_parity" and "pass" not in adv


def test_d1_reject_missing_seed():
    with pytest.raises(gtv.AdapterError):
        gtv.build_d1(_measured(seed_ids=[42]), _d1cfg())


def test_d1_reject_duplicate_seed_id():
    """P1-1: five values but a repeated seed id (distinct-path) must be rejected."""
    with pytest.raises(gtv.AdapterError):
        gtv.build_d1(_measured(seed_ids=[42, 42, 43, 44, 45]), _d1cfg())


def test_d1_reject_wrong_seed_ids():
    """Seed-id SET must match the registered set exactly (not just count)."""
    with pytest.raises(gtv.AdapterError):
        gtv.build_d1(_measured(seed_ids=[42, 43, 44, 45, 99]), _d1cfg())


def test_d1_reject_missing_k():
    with pytest.raises(gtv.AdapterError):
        gtv.build_d1(_measured(ks=("8",)), _d1cfg())


def test_d1_reject_extra_k():
    m = _measured(); m["4"] = m["8"]
    with pytest.raises(gtv.AdapterError):
        gtv.build_d1(m, _d1cfg())


def test_d1_expect_seeds_must_be_id_list():
    cfg = _d1cfg(); cfg["expect"] = dict(cfg["expect"]); cfg["expect"]["seeds"] = 5  # old int form
    with pytest.raises(gtv.AdapterError):
        gtv.build_d1(_measured(), cfg)


# ---- CLI: realpath collision (repeated artifact) + inline-values rejection ----
def test_cli_reject_repeated_artifact_path(tmp_path):
    """P1-1: the same artifact reused for five seed ids => realpath collision => exit 2."""
    refs = _write_refs(tmp_path, "matched_control")
    d1 = _write_d1_measured(tmp_path, shared_path=True)
    out = tmp_path / "v"
    proc = _run_cli(["--references", refs, "--out-dir", out, "--d1", d1])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not (out.exists() and list(out.glob("*.json")))


def test_cli_rejects_inline_values_form(tmp_path):
    """r3: the inline {seed_ids, values} manifest carries no paths and would BYPASS the realpath
    uniqueness guard (correct IDs + five repeated values then pass as five distinct seeds). The
    CLI loader must REJECT it (exit 2, nothing published). The inline form stays available to unit
    tests ONLY via a DIRECT build_d1() call — see test_d1_matched_all_within_passes etc."""
    refs = _write_refs(tmp_path, "matched_control")
    inline = {k: {"seed_ids": _SEED_IDS,
                  "values": {m: [mu] * 5 for m, (mu, _sd) in _CTRL[k].items()}}
              for k in ("1", "8")}
    d1 = tmp_path / "d1_inline.json"; d1.write_text(json.dumps(inline))
    out = tmp_path / "v"
    proc = _run_cli(["--references", refs, "--out-dir", out, "--d1", d1])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "inline values are not accepted" in (proc.stdout + proc.stderr)
    assert not (out.exists() and list(out.glob("*.json")))


# ======================================================================================= #
# D2 conditioning: REAL A2b j->degrees join (P1-2), coverage, dedupe, F7a
# ======================================================================================= #
def test_cond_real_artifact_joins_j_to_degrees():
    """The converter consumes the REAL committed audit_convention.json (read-only) and yields
    exactly the five DEGREE cells {11.25,45,90,180,270}, each with both channels."""
    art = json.loads(_REAL_AUDIT.read_text())
    cells = gtv.conditioning_cells_from_artifact(art)
    assert sorted(c["angle"] for c in cells) == [11.25, 45.0, 90.0, 180.0, 270.0]
    assert all("pooled_relerr" in c and "patch_relerr" in c for c in cells)
    assert gtv.build_conditioning_verdict(cells, [11.25, 45, 90, 180, 270])["pass"] is True


def test_cond_real_artifact_missing_1125_rejects():
    cells = gtv.conditioning_cells_from_artifact(json.loads(_REAL_AUDIT.read_text()))
    with pytest.raises(gtv.AdapterError):
        gtv.build_conditioning_verdict(cells, [45, 90, 180, 270])       # missing 11.25


def test_cond_real_artifact_extra_angle_rejects():
    cells = gtv.conditioning_cells_from_artifact(json.loads(_REAL_AUDIT.read_text()))
    with pytest.raises(gtv.AdapterError):
        gtv.build_conditioning_verdict(cells, [11.25, 45, 90, 135, 180, 270])


def test_cond_j_without_a2params_rejects():
    """A record with only 'j' but no a2_params to join against must NOT be treated as degrees."""
    with pytest.raises(gtv.AdapterError):
        gtv.conditioning_cells_from_artifact({"a2b": {"per_angle": [
            {"j": 4, "pooled_relerr": 1e-5, "patch_relerr": 3e-5}]}})


def test_cond_degree_native_passes():
    v = gtv.build_conditioning_verdict(gtv.conditioning_cells_from_artifact(_cond_artifact()), _COND_ANGLES)
    assert v["gate"] == "d2_conditioning" and v["pass"] is True and v["metrics"]["n_angles"] == 5


def test_cond_later_cell_failure_fails_gate():
    cells = gtv.conditioning_cells_from_artifact(_cond_artifact({(270, "pooled"): 2e-4}))
    assert gtv.build_conditioning_verdict(cells, _COND_ANGLES)["pass"] is False


def test_cond_record_missing_channel_rejected():
    with pytest.raises(gtv.AdapterError):
        gtv.conditioning_cells_from_artifact({"a2b": {"per_angle": [{"angle": 45.0, "pooled_relerr": 1e-5}]}})


def test_cond_duplicate_angle_rejected():
    """P1-1: a duplicated angle in the raw per_angle list => reject (before set normalisation)."""
    art = _cond_artifact(angles=[11.25, 45, 90, 180, 270, 45])
    with pytest.raises(gtv.AdapterError):
        gtv.conditioning_cells_from_artifact(art)


def test_cond_missing_angle_rejected():
    cells = gtv.conditioning_cells_from_artifact(_cond_artifact(angles=[11.25, 45, 90, 180]))
    with pytest.raises(gtv.AdapterError):
        gtv.build_conditioning_verdict(cells, _COND_ANGLES)


# ======================================================================================= #
# D2 end-to-end (rot0-free) + coverage + dedupe + F7a
# ======================================================================================= #
def test_e2e_full_matrix_passes():
    v = gtv.build_e2e_verdict(gtv.e2e_cells_from_index(_e2e_index()), _E2E_MATRIX)
    assert v["gate"] == "d2_end_to_end" and v["pass"] is True and v["metrics"]["n_cells"] == 5


def test_e2e_later_cell_failure_fails_gate():
    cells = gtv.e2e_cells_from_index(_e2e_index({("8", 90): 0.010}))
    assert gtv.build_e2e_verdict(cells, _E2E_MATRIX)["pass"] is False


def test_e2e_missing_pair_rejected():
    idx = _e2e_index(); del idx["8"]["90"]
    with pytest.raises(gtv.AdapterError):
        gtv.build_e2e_verdict(gtv.e2e_cells_from_index(idx), _E2E_MATRIX)


def test_e2e_duplicate_angle_rejected():
    """P1-1: '90' and '90.0' collapse to one under float() — reject before normalisation."""
    idx = {"1": {"45": {"waveform_gap": {"mean_rel_l2": 0.002}}},
           "8": {"90": {"waveform_gap": {"mean_rel_l2": 0.002}},
                 "90.0": {"waveform_gap": {"mean_rel_l2": 0.003}}}}
    with pytest.raises(gtv.AdapterError):
        gtv.e2e_cells_from_index(idx)


# ======================================================================================= #
# D2 H-A3 flatness: inlined constants, all-cells, coverage, dedupe, F7b
# ======================================================================================= #
def test_h_a3_inlined_constants_exact():
    assert gtv.H_A3_THRESHOLDS[1] == {"T60": 0.080, "C50": 0.012, "EDT": 0.740}
    assert gtv.H_A3_THRESHOLDS[8] == {"T60": 0.024, "C50": 0.006, "EDT": 0.140}


def test_h_a3_full_matrix_passes():
    v = gtv.build_flatness_verdict(_flat_index(), _FLAT_MATRIX)
    assert v["gate"] == "d2_flatness" and v["pass"] is True and v["metrics"]["n_cells"] == 15


def test_h_a3_one_cell_failure_fails_gate():
    v = gtv.build_flatness_verdict(_flat_index({("8", 90, "EDT"): 0.2}), _FLAT_MATRIX)
    assert v["pass"] is False and v["metrics"]["n_failing"] == 1


def test_h_a3_k1_arm_failure_fails_gate():
    v = gtv.build_flatness_verdict(_flat_index({("1", 45, "T60"): 0.2}), _FLAT_MATRIX)
    assert v["pass"] is False


def test_h_a3_missing_rot0_rejected():
    idx = _flat_index(); del idx["8"]["0"]
    with pytest.raises(gtv.AdapterError):
        gtv.build_flatness_verdict(idx, _FLAT_MATRIX)


def test_h_a3_missing_rotated_angle_rejected():
    idx = _flat_index(); del idx["1"]["180"]
    with pytest.raises(gtv.AdapterError):
        gtv.build_flatness_verdict(idx, _FLAT_MATRIX)


def test_h_a3_duplicate_angle_rejected():
    """P1-1: '90' and '90.0' keys collapse silently in {float(a): m} — reject first."""
    idx = _flat_index()
    idx["8"]["90.0"] = dict(_FLAT_REF["8"])       # duplicate of "90"
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
# CLI: all-pass, fail, F1 rejections, F3 fresh-dir, P2 exit boundary
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
    proc = _run_cli(["--references", refs, "--out-dir", tmp_path / "v", "--d1", d1])
    assert proc.returncode == 1, proc.stdout + proc.stderr


def test_cli_reject_d1_partial_seeds(tmp_path):
    refs = _write_refs(tmp_path, "matched_control")
    d1 = _write_d1_measured(tmp_path, seed_ids=[42])
    out = tmp_path / "v"
    proc = _run_cli(["--references", refs, "--out-dir", out, "--d1", d1])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert not (out.exists() and list(out.glob("*.json")))


def test_cli_reject_flatness_k8_only(tmp_path):
    refs = _write_refs(tmp_path, "contextual")
    idx = _flat_index(); del idx["1"]
    flat = tmp_path / "flat.json"; flat.write_text(json.dumps(idx))
    proc = _run_cli(["--references", refs, "--out-dir", tmp_path / "v", "--d2-flatness", flat])
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_cli_reject_e2e_single_label(tmp_path):
    refs = _write_refs(tmp_path, "contextual")
    e2e = tmp_path / "e2e.json"
    e2e.write_text(json.dumps({"8": {"90": {"waveform_gap": {"mean_rel_l2": 0.002}}}}))
    proc = _run_cli(["--references", refs, "--out-dir", tmp_path / "v", "--d2-e2e", e2e])
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_cli_nonnumeric_e2e_angle_exit_2(tmp_path):
    """P2: a nonnumeric e2e angle raises ValueError deep in parsing -> exit 2, no traceback."""
    refs = _write_refs(tmp_path, "contextual")
    e2e = tmp_path / "e2e.json"
    e2e.write_text(json.dumps({"8": {"abc": {"waveform_gap": {"mean_rel_l2": 0.002}}}}))
    out = tmp_path / "v"
    proc = _run_cli(["--references", refs, "--out-dir", out, "--d2-e2e", e2e])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "REJECTED" in (proc.stdout + proc.stderr)
    assert "Traceback" not in (proc.stdout + proc.stderr)
    assert not (out.exists() and list(out.glob("*.json")))


def test_cli_fresh_dir_refuses_nonempty_out(tmp_path):
    refs = _write_refs(tmp_path, "matched_control")
    d1 = _write_d1_measured(tmp_path)
    out = tmp_path / "verdicts"; out.mkdir()
    stale = out / "verdict_d1_parity.json"; stale.write_text('{"stale": true}')
    proc = _run_cli(["--references", refs, "--out-dir", out, "--d1", d1])
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert json.loads(stale.read_text()) == {"stale": True}
    assert list(out.glob("*.json")) == [stale]
