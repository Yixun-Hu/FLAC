"""exp-09 D-stage: threshold->verdict ADAPTER tests (RED-first, CPU-only).

``gate_thresholds_to_verdicts.py`` is the thin adapter the B-code r1 review required to
land before any D acceptance run: it consumes raw metric JSONs (eval_FLAC for D1;
compare_predictions / equivariance for D2) plus a REFERENCES config, and emits the
``{gate, pass, metrics}`` per-gate verdicts consumed by ``aggregate_gate.py``.

Covered:
  * band math VERBATIM vs exp_07 gate_verdict.py (equivalence <=1sigma_c /
    non-inferiority <=2sigma_c, both directions; degenerate sc==0);
  * D1 matched-control mode emits a gating d1_parity verdict; contextual-only mode
    emits NO parity verdict (advisory only), even when numbers are present;
  * H-A3 flatness all-cells rule (one failing cell => the gate fails), inlined constants;
  * D2 conditioning (<=1e-4) and end-to-end waveform rel-L2 (<=0.00931) thresholds;
  * emitted verdicts are valid aggregate_gate.py inputs;
  * CLI exit codes (0 all-pass / 1 any-fail / nonzero rejection), atomic finite JSON.
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

import gate_thresholds_to_verdicts as gtv  # RED until the D-stage adapter lands
import aggregate_gate  # same dir; conftest put it on sys.path


# --------------------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------------------- #
def _write_eval_json(path, metrics, **extra):
    """An eval_FLAC-shaped metric JSON (record has a 'metrics' dict)."""
    rec = {"metrics": dict(metrics), "ckpt_path": "x.ckpt", "rotate_deg": 0.0,
           "cond_method": "fa_invariant", "frame_avg_angles": [0.0], "cond_autocast": "bf16"}
    rec.update(extra)
    Path(path).write_text(json.dumps(rec))
    return str(path)


def _run_cli(args, env_extra=None):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(_EXP09_DIR / "gate_thresholds_to_verdicts.py"), *map(str, args)],
        capture_output=True, text=True, env=env, timeout=120,
    )


# --------------------------------------------------------------------------------------- #
# seed_stats: 5-seed mean +/- sample std (ddof=1, exp_01/gate_verdict.py convention)
# --------------------------------------------------------------------------------------- #
def test_seed_stats_matches_ddof1_convention():
    mu, sd = gtv.seed_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mu == pytest.approx(3.0)
    assert sd == pytest.approx(math.sqrt(2.5))  # sqrt(10/4)


def test_seed_stats_single_value_zero_std():
    mu, sd = gtv.seed_stats([7.0])
    assert mu == pytest.approx(7.0) and sd == 0.0


def test_seed_stats_rejects_non_finite():
    with pytest.raises(gtv.AdapterError):
        gtv.seed_stats([1.0, float("nan"), 3.0])


# --------------------------------------------------------------------------------------- #
# sigma_band: the exact gate_verdict.py band math
#   sc = sqrt(sd^2 + rsd^2); n = (mu-rmu)/sc; |n|<=1 equiv, |n|<=2 band, else outside.
#   lower-better: d<0 SUPERIOR. within_2sc is THE gate decision (mutation m1 target).
# --------------------------------------------------------------------------------------- #
def test_band_equivalence_superior_lower_better():
    b = gtv.sigma_band(8.59, 0.01, 8.60, 0.01, better_low=True)
    assert b["within_1sc"] and b["within_2sc"]
    assert b["tier"] == "equiv<=1sc"
    assert b["direction"] == "superior"          # measured lower than ref -> better
    assert b["n_sigma"] == pytest.approx(-0.01 / math.sqrt(2 * 0.01 ** 2))


def test_band_noninferiority_worse_within_2sc():
    b = gtv.sigma_band(8.62, 0.01, 8.60, 0.01, better_low=True)  # n ~ +1.414
    assert not b["within_1sc"] and b["within_2sc"]
    assert b["tier"] == "band<=2sc"
    assert b["direction"] == "worse"


def test_band_outside_worse_fails():
    b = gtv.sigma_band(8.65, 0.01, 8.60, 0.01, better_low=True)  # n ~ +3.54
    assert not b["within_2sc"]
    assert b["tier"] == "outside>2sc" and b["direction"] == "worse"


def test_band_outside_superior_also_fails_the_2sc_band():
    """gate_verdict.py's gate is |n|<=2 (two-sided); a wildly-superior cell is still
    OUTSIDE the band -- the direction is reported, but within_2sc is False."""
    b = gtv.sigma_band(8.55, 0.01, 8.60, 0.01, better_low=True)  # n ~ -3.54
    assert not b["within_2sc"]
    assert b["direction"] == "superior"


def test_band_higher_better_direction_flips():
    # R@1 is higher-better: measured ABOVE ref is SUPERIOR.
    b = gtv.sigma_band(7.10, 0.10, 7.06, 0.10, better_low=False)
    assert b["direction"] == "superior"
    b2 = gtv.sigma_band(6.90, 0.10, 7.06, 0.10, better_low=False)
    assert b2["direction"] == "worse"


def test_band_degenerate_zero_sigma_no_nonfinite():
    b = gtv.sigma_band(8.60, 0.0, 8.60, 0.0, better_low=True)   # identical, sc==0
    assert b["within_2sc"] and b["direction"] == "tie"
    assert b["n_sigma"] is None                                  # never +/-inf in output
    b2 = gtv.sigma_band(8.61, 0.0, 8.60, 0.0, better_low=True)   # differ, sc==0
    assert not b2["within_2sc"]
    assert b2["n_sigma"] is None


# --------------------------------------------------------------------------------------- #
# D1 matched-control: gating d1_parity verdict
# --------------------------------------------------------------------------------------- #
_CTRL = {
    "8": {"T60": [8.609, 0.012], "C50": [0.9682, 0.0030], "EDT": [37.10, 0.07],
          "RIR_to_GT_RIR_R@1": [7.06, 0.10]},
    "1": {"T60": [9.969, 0.039], "C50": [1.0460, 0.0064], "EDT": [39.95, 0.37],
          "RIR_to_GT_RIR_R@1": [6.83, 0.22]},
}


def _measured_near(control, jitter=0.0):
    """Measured 5-seed lists whose means sit at each control mean (+ jitter)."""
    out = {}
    for k, mv in control.items():
        out[k] = {}
        for metric, (mu, _sd) in mv.items():
            base = mu + jitter
            out[k][metric] = [base - 0.0, base + 0.0, base, base, base]  # mean == base, sd 0
    return out


def test_d1_matched_all_within_band_passes():
    measured = _measured_near(_CTRL, jitter=0.0)  # exactly on the control means
    gate, advisory = gtv.build_d1(measured, {"mode": "matched_control", "control_name": "P1",
                                             "control_stats": _CTRL})
    assert advisory is None
    assert gate["gate"] == "d1_parity" and gate["pass"] is True
    # R@1 is advisory (reported, non-gating) -- present but not in the gate cells
    assert any(c["metric"] == "RIR_to_GT_RIR_R@1" for c in gate["metrics"]["advisory"])
    assert all(c["metric"] in ("T60", "C50", "EDT") for c in gate["metrics"]["cells"])


def test_d1_matched_one_metric_far_fails_the_gate():
    measured = _measured_near(_CTRL, jitter=0.0)
    measured["8"]["T60"] = [10.0] * 5   # ~117 sigma above 8.609 -> outside 2sc
    gate, _ = gtv.build_d1(measured, {"mode": "matched_control", "control_stats": _CTRL})
    assert gate["pass"] is False
    bad = [c for c in gate["metrics"]["cells"] if c["K"] == "8" and c["metric"] == "T60"][0]
    assert not bad["within_2sc"] and bad["direction"] == "worse"


# --------------------------------------------------------------------------------------- #
# D1 contextual-only: NO parity verdict (mutation m2 target)
# --------------------------------------------------------------------------------------- #
def test_d1_contextual_emits_no_parity_verdict():
    """Even when comparator numbers are present, contextual mode is authoritative:
    NO matched control => NO gating parity verdict (plan §4 D1). The adapter returns
    an advisory record instead, and it carries no boolean 'pass'."""
    measured = _measured_near(_CTRL, jitter=0.0)
    gate, advisory = gtv.build_d1(
        measured, {"mode": "contextual", "contextual_stats": _CTRL,
                   "note": "P1 pending"})
    assert gate is None, "contextual mode must NOT emit a parity gate verdict"
    assert advisory is not None
    assert advisory.get("advisory") is True
    assert advisory["gate"] != "d1_parity"
    assert "pass" not in advisory, "an advisory record must not carry a gating pass"


def test_d1_rejects_unknown_mode():
    with pytest.raises(gtv.AdapterError):
        gtv.build_d1(_measured_near(_CTRL), {"mode": "bogus"})


# --------------------------------------------------------------------------------------- #
# D2 conditioning-level: pooled invariance AND patch roll-equivariance, both <= 1e-4
# --------------------------------------------------------------------------------------- #
def test_d2_conditioning_threshold_is_1e_4():
    assert gtv.D2_COND_THRESHOLD == 1e-4


def test_d2_conditioning_pass_and_fail():
    ok = gtv.build_conditioning_verdict({"45": 1e-5, "90": 9e-5}, {"45": 3e-5, "90": 8e-5})
    assert ok["gate"] == "d2_conditioning" and ok["pass"] is True
    bad = gtv.build_conditioning_verdict({"45": 1e-5}, {"45": 2e-4})  # patch too big
    assert bad["pass"] is False


def test_d2_conditioning_requires_both_channels():
    with pytest.raises(gtv.AdapterError):
        gtv.build_conditioning_verdict({"45": 1e-5}, {})     # patch channel empty


# --------------------------------------------------------------------------------------- #
# D2 end-to-end waveform rel-L2 <= 0.00931 (consumes compare_predictions waveform_gap)
# --------------------------------------------------------------------------------------- #
def test_d2_e2e_threshold_is_registered_bound():
    assert gtv.D2_E2E_REL_L2_THRESHOLD == 0.00931


def test_d2_e2e_pass_and_fail_from_compare_predictions_shape():
    good = {"45": {"waveform_gap": {"mean_rel_l2": 0.0022, "mean_abs_diff": 1e-4}},
            "90": {"waveform_gap": {"mean_rel_l2": 0.0090}}}
    v = gtv.build_e2e_verdict(gtv.extract_rel_l2(good))
    assert v["gate"] == "d2_end_to_end" and v["pass"] is True
    bad = {"45": {"waveform_gap": {"mean_rel_l2": 0.010}}}   # over 0.00931
    v2 = gtv.build_e2e_verdict(gtv.extract_rel_l2(bad))
    assert v2["pass"] is False


# --------------------------------------------------------------------------------------- #
# D2 H-A3 flatness: inlined constants + ALL-cells rule (mutation m3 target)
# --------------------------------------------------------------------------------------- #
def test_h_a3_inlined_constants_exact():
    assert gtv.H_A3_THRESHOLDS[1] == {"T60": 0.080, "C50": 0.012, "EDT": 0.740}
    assert gtv.H_A3_THRESHOLDS[8] == {"T60": 0.024, "C50": 0.006, "EDT": 0.140}


def _flat_metrics(k1_edt90=37.10, k8_edt90=37.10):
    return {
        "1": {
            "0":  {"T60": 9.969, "C50": 1.046, "EDT": 39.950},
            "45": {"T60": 9.980, "C50": 1.050, "EDT": 40.000},   # deltas within K=1 band
        },
        "8": {
            "0":  {"T60": 8.609, "C50": 0.9682, "EDT": 37.10},
            "90": {"T60": 8.620, "C50": 0.9700, "EDT": k8_edt90},
        },
    }


def test_h_a3_all_cells_within_band_passes():
    v = gtv.build_flatness_verdict(_flat_metrics(k8_edt90=37.20))  # |dEDT|=0.10 <= 0.140
    assert v["gate"] == "d2_flatness" and v["pass"] is True
    assert v["metrics"]["n_failing"] == 0


def test_h_a3_one_failing_cell_fails_whole_gate():
    """ALL cells must pass: a single EDT cell over the K=8 threshold (0.140) fails the gate."""
    v = gtv.build_flatness_verdict(_flat_metrics(k8_edt90=37.30))  # |dEDT|=0.20 > 0.140
    assert v["pass"] is False
    assert v["metrics"]["n_failing"] == 1
    failing = [c for c in v["metrics"]["cells"] if not c["pass"]]
    assert failing[0]["metric"] == "EDT" and failing[0]["K"] == 8 and failing[0]["angle"] == 90.0


def test_h_a3_missing_rot0_reference_is_rejected():
    with pytest.raises(gtv.AdapterError):
        gtv.build_flatness_verdict({"8": {"90": {"T60": 8.6, "C50": 0.97, "EDT": 37.1}}})


# --------------------------------------------------------------------------------------- #
# emitted verdicts are valid aggregate_gate.py inputs (contract tie-in)
# --------------------------------------------------------------------------------------- #
def test_emitted_verdicts_are_valid_aggregate_gate_inputs(tmp_path):
    v = gtv.build_flatness_verdict(_flat_metrics(k8_edt90=37.20))
    p = tmp_path / "verdict_d2_flatness.json"
    aggregate_gate.write_verdict(str(p), v)
    loaded = aggregate_gate.load_verdict(str(p))      # would raise on non-bool/non-finite
    assert loaded["gate"] == "d2_flatness"
    agg = aggregate_gate.aggregate([str(p)], ["d2_flatness"])
    assert agg["all_pass"] is True


# --------------------------------------------------------------------------------------- #
# CLI: exit codes + atomic finite JSON (mutation m5 target = the 0/1 return)
# --------------------------------------------------------------------------------------- #
def _refs(tmp_path, mode="matched_control"):
    refs = {"d1": {"mode": mode, "control_name": "P1", "control_stats": _CTRL}}
    if mode == "contextual":
        refs["d1"] = {"mode": "contextual", "contextual_stats": _CTRL, "note": "P1 pending"}
    p = tmp_path / "references.json"
    p.write_text(json.dumps(refs))
    return p


def _d1_measured_file(tmp_path, control, jitter=0.0):
    """A d1-measured index that points at real eval_FLAC-shaped JSONs (5 'seeds')."""
    idx = {}
    for k, mv in control.items():
        seed_paths = []
        for s in range(5):
            metrics = {metric: (mu + jitter) for metric, (mu, _sd) in mv.items()}
            seed_paths.append(_write_eval_json(tmp_path / f"eval_K{k}_s{s}.json", metrics))
        idx[k] = {"seeds": seed_paths}
    p = tmp_path / "d1_measured.json"
    p.write_text(json.dumps(idx))
    return p


def test_cli_all_pass_exit_0(tmp_path):
    refs = _refs(tmp_path, "matched_control")
    d1 = _d1_measured_file(tmp_path, _CTRL, jitter=0.0)
    out = tmp_path / "verdicts"
    proc = _run_cli(["--references", refs, "--out-dir", out, "--d1", d1])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    written = list(out.glob("verdict_*.json"))
    assert any("d1_parity" in p.name for p in written)


def test_cli_failing_gate_exit_1(tmp_path):
    refs = _refs(tmp_path, "matched_control")
    d1 = _d1_measured_file(tmp_path, _CTRL, jitter=5.0)   # every mean way off -> fail
    out = tmp_path / "verdicts"
    proc = _run_cli(["--references", refs, "--out-dir", out, "--d1", d1])
    assert proc.returncode == 1, proc.stdout + proc.stderr


def test_cli_contextual_writes_no_parity_verdict(tmp_path):
    refs = _refs(tmp_path, "contextual")
    d1 = _d1_measured_file(tmp_path, _CTRL, jitter=0.0)
    out = tmp_path / "verdicts"
    proc = _run_cli(["--references", refs, "--out-dir", out, "--d1", d1])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not list(out.glob("verdict_d1_parity.json")), "contextual mode wrote a parity verdict"
    assert list(out.glob("advisory_*.json")), "contextual mode should write an advisory record"


def test_cli_rejects_missing_references(tmp_path):
    out = tmp_path / "verdicts"
    proc = _run_cli(["--references", tmp_path / "nope.json", "--out-dir", out,
                     "--d1", _d1_measured_file(tmp_path, _CTRL)])
    assert proc.returncode not in (0, 1), proc.stdout + proc.stderr  # hard rejection


def test_cli_d2_all_gates_atomic_finite(tmp_path):
    cond = tmp_path / "cond.json"; cond.write_text(json.dumps(
        {"pooled_invariance": {"45": 1e-5}, "patch_roll_equiv": {"45": 3e-5}}))
    e2e = tmp_path / "e2e.json"; e2e.write_text(json.dumps(
        {"45": {"waveform_gap": {"mean_rel_l2": 0.0022}}}))
    flat = tmp_path / "flat.json"
    flat_idx = {"8": {"0": _write_eval_json(tmp_path / "f0.json", {"T60": 8.6, "C50": 0.97, "EDT": 37.1}),
                      "90": _write_eval_json(tmp_path / "f90.json", {"T60": 8.61, "C50": 0.971, "EDT": 37.2})}}
    flat.write_text(json.dumps(flat_idx))
    refs = tmp_path / "refs.json"; refs.write_text(json.dumps({"d1": {"mode": "contextual"}}))
    out = tmp_path / "verdicts"
    proc = _run_cli(["--references", refs, "--out-dir", out,
                     "--d2-cond", cond, "--d2-e2e", e2e, "--d2-flatness", flat])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for gate in ("d2_conditioning", "d2_end_to_end", "d2_flatness"):
        vp = out / f"verdict_{gate}.json"
        assert vp.exists()
        rec = json.loads(vp.read_text())
        assert rec["gate"] == gate and isinstance(rec["pass"], bool)
        # no leftover atomic temp siblings
    assert not list(out.glob(".*tmp*"))
