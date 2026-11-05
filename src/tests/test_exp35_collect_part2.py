"""exp_35 round 5b (i): Tier 2-C4 and the Tier 3 B-F row (plan_orbitrir_yaw_evidence.md r2 §4, frozen before any
phase-(i) data exists, §11 R8). Tier 2-C4: the six (K1, K8) x (90, 180, 270) pairs at seed 42 against B-F's own
composition control (matched offset AND pooled, the 1e-2 cap, the 0.1 alarm), T-dup exact, the 45-degree and
random-yaw distances descriptive only. Tier 3: the registered rule for B-F against P1-pin's Delta_C4 read at full
precision from the hash-pinned exp_25 JSON (28 checks, FD included).

Synthetic exp_35 cells from the shared fixture (real evaluator functions, N = 8 items).

Written by the exp_35 Coder (Claude Opus 5.5, max effort) per plan_orbitrir_yaw_evidence.md r2/r3.
"""
import os

import pytest

import test_depth_canon_collect as T

C, MANIFEST, PIN = T.C, T.MANIFEST, T.PIN
PHASE_I = ["CTL", "BF-pin", "BF-dup"]
CONTROL_BF = {"d128": 2e-4, "d256": 3e-4, "d384": 4e-4, "pooled": 6e-4}


def _row(name):
    return next(c for c in MANIFEST["cells"] if c["cell"] == name)


def bf_campaign(tmp, metrics_fn=None, waves=None, skip=()):
    T.write_campaign(tmp, PHASE_I, metrics_fn=metrics_fn, waves_fn=(lambda r: (waves or {}).get(r["cell"])), skip=skip)


def collect35(tmp, blocks=PHASE_I):
    return C.collect_cells(MANIFEST, C.select_cells(MANIFEST, blocks=list(blocks)), farm_root=T.FARM,
                           dump_root=os.path.join(str(tmp), "dumps"), pin=PIN, canon_index=T.canon_index(),
                           base_dir=str(tmp), check_link=False, no_dump_blocks=(), dump_check="full")


def _scaled(name, eps, item=None):
    w = T.waveforms_for(_row(name)).clone()
    if item is None:
        return w * (1 + eps)
    w[item] *= (1 + eps)
    return w


def _t2(tmp, waves=None, control=None):
    bf_campaign(tmp, waves=waves)
    return C.tier2_c4(collect35(tmp), dict(control or CONTROL_BF))


# --------------------------------------------------------------------------- #
# Tier 2-C4
# --------------------------------------------------------------------------- #
def test_an_invariant_bf_campaign_passes_tier2_c4(tmp_path):
    t2 = _t2(tmp_path)
    assert t2["pass"] is True and len(t2["pairs"]) == 6 and all(p["relL2"]["max"] == 0 for p in t2["pairs"])
    assert sorted((p["K"], p["deg"], p["control_key"]) for p in t2["pairs"]) == sorted(
        (k, d, {90: "d128", 180: "d256", 270: "d384"}[d]) for k in (1, 8) for d in (90, 180, 270))
    assert t2["pooled"]["pass"] and t2["pooled"]["control_p99"] == CONTROL_BF["pooled"]
    assert t2["determinism"]["pass"] and len(t2["determinism"]["pairs"]) == 2 and not t2["alarm"]["suspended"]
    d = t2["descriptive"]
    assert set(d) == {"45", "R"} and len(d["45"]) == 2 and len(d["R"]) == 10
    assert all("pass" not in x and "relL2" in x and "relLinf" in x for x in d["45"] + d["R"])


def test_a_pair_beyond_three_times_its_matched_control_fails(tmp_path):
    name = "exp35_BF_C4_S40000_K1_s42_th90"
    t2 = _t2(tmp_path, waves={name: _scaled(name, 1e-3)})
    bad = next(p for p in t2["pairs"] if p["cell"] == name)
    assert abs(bad["relL2"]["p99"] - 1e-3) < 1e-6 and not bad["pass_matched"] and bad["pass_cap"]
    assert t2["pass"] is False and all(p["pass_matched"] for p in t2["pairs"] if p["cell"] != name)


def test_the_hard_cap_holds_whatever_the_control(tmp_path):
    name = "exp35_BF_C4_S40000_K8_s42_th180"
    t2 = _t2(tmp_path, waves={name: _scaled(name, 2e-2)}, control=dict(CONTROL_BF, d256=1.0, pooled=1.0))
    bad = next(p for p in t2["pairs"] if p["cell"] == name)
    assert bad["pass_matched"] and not bad["pass_cap"] and t2["pass"] is False


def test_the_pooled_rule_binds_on_its_own(tmp_path):
    waves = {f"exp35_BF_C4_S40000_K{k}_s42_th{d}": _scaled(f"exp35_BF_C4_S40000_K{k}_s42_th{d}", 5e-4)
             for k in (1, 8) for d in (90, 180, 270)}
    t2 = _t2(tmp_path, waves=waves, control=dict(CONTROL_BF, pooled=1e-4))
    assert all(p["pass_matched"] and p["pass_cap"] for p in t2["pairs"])
    assert t2["pooled"]["pass"] is False and t2["pass"] is False


def test_a_zero_control_demands_a_zero_p99(tmp_path):
    name = "exp35_BF_C4_S40000_K1_s42_th270"
    t2 = _t2(tmp_path, waves={name: _scaled(name, 1e-6)}, control=dict(CONTROL_BF, d384=0.0))   # > float32 eps
    bad = next(p for p in t2["pairs"] if p["cell"] == name)
    assert bad["control_p99"] == 0.0 and not bad["pass_matched"] and t2["pass"] is False


def test_the_alarm_names_the_offending_items(tmp_path):
    name = "exp35_BF_C4_S40000_K8_s42_th90"
    t2 = _t2(tmp_path, waves={name: _scaled(name, 0.5, item=3)})
    assert t2["alarm"]["suspended"] and t2["alarm"]["items"] == {name: [T.target_ids("unseen")[3]]}


def test_the_bf_duplicate_must_be_exactly_reproducible(tmp_path):
    name = "exp35_BF_T_S40000_K8_s42_dup"
    t2 = _t2(tmp_path, waves={name: _scaled(name, 1e-7)})
    d = next(p for p in t2["determinism"]["pairs"] if p["cell"] == name)
    assert not t2["determinism"]["pass"] and d["relL2_max"] > 0 and not d["dump_sha_equal"]
    assert t2["pass"] is True                                                 # a separate requirement


def test_off_orbit_distances_are_descriptive_only(tmp_path):
    names = ["exp35_BF_C4_S40000_K1_s42_th45", "exp35_BF_R_S40000_K8_s44"]
    t2 = _t2(tmp_path, waves={n: _scaled(n, 0.05) for n in names})
    assert t2["pass"] is True and not t2["alarm"]["suspended"]                # outside the C4 claim
    d = {x["cell"]: x for x in t2["descriptive"]["45"] + t2["descriptive"]["R"]}
    assert abs(d[names[0]]["relL2"]["p99"] - 0.05) < 1e-6 and abs(d[names[1]]["relL2"]["max"] - 0.05) < 1e-6


def test_tier2_c4_without_a_control_or_a_pair_is_not_a_pass(tmp_path):
    bf_campaign(tmp_path)
    cells = collect35(tmp_path)
    assert C.tier2_c4(cells, None)["pass"] is None
    cells["exp35_BF_C4_S40000_K8_s42_th270"]["stream"]["input_hash"] = "0" * 64
    t2 = C.tier2_c4(cells, dict(CONTROL_BF))
    bad = next(p for p in t2["pairs"] if p["cell"] == "exp35_BF_C4_S40000_K8_s42_th270")
    assert bad["computable"] is False and t2["pass"] is False


# --------------------------------------------------------------------------- #
# Tier 3, the B-F row
# --------------------------------------------------------------------------- #
SHIFT = {90: 0.0003, 180: -0.0001, 270: 0.0001, 45: 0.3}


def bf_metrics(r, r_shift=0.2, shift=SHIFT):
    v = 5.0 + 0.01 * (r["seed"] - 42) + (0.5 if r["K"] == 1 else 0.0)
    if r["block"] == "BF-dup":
        v += 7.0                                                              # never enters Tier 3
    elif r["rotate_mode"] == "random":
        v += r_shift
    elif r["rotate_deg"]:
        v += shift[r["rotate_deg"]]
    return {k: round(v + j, 6) for j, k in enumerate(T.EMITTED)}


@pytest.fixture(scope="module")
def exp25():
    doc, reasons = C.load_exp25_reference()
    assert reasons == []
    return doc


def test_tier3_bf_on_known_numbers(tmp_path, exp25):
    bf_campaign(tmp_path, metrics_fn=bf_metrics)
    t3 = C.tier3_bf(collect35(tmp_path), exp25)
    bf = t3["arms"]["BF"][1]["item"]["T60"]
    assert abs(bf["delta_c4"] - 0.0004) < 1e-9 and abs(bf["probe45"] - 0.3) < 1e-9
    assert abs(bf["delta_r"] - 0.2) < 1e-9 and bf["n_r_pairs"] == 5 and abs(bf["sd_z"] - 0.0158113883) < 1e-9
    assert len(t3["checks"]) == 14 and {c["metric"] for c in t3["checks"]} == set(C.TIER3_KEYS)   # x 2 rules = 28
    for c in t3["checks"]:                                                    # the full-precision exp_25 value
        assert c["p1_delta_c4"] == exp25["tier3"]["arms"]["P1"][str(c["K"])]["item"][c["metric"]]["delta_c4"]
    assert all(c["pass_c4"] for c in t3["checks"]) and not any(c["pass_r"] for c in t3["checks"])
    assert t3["pass"] is False                                                # E4: C4-invariant, not yaw-invariant


def test_tier3_bf_passes_when_both_rules_hold(tmp_path, exp25):
    bf_campaign(tmp_path, metrics_fn=lambda r: bf_metrics(r, r_shift=0.001))
    t3 = C.tier3_bf(collect35(tmp_path), exp25)
    assert t3["pass"] is True and all(c["pass_c4"] and c["pass_r"] for c in t3["checks"])


def test_the_c4_rule_is_ten_percent_of_p1_pin_per_metric(tmp_path, exp25):
    fd = exp25["tier3"]["arms"]["P1"]["1"]["item"]["FD"]["delta_c4"]        # the smallest yardstick (~0.0059)
    shift = {90: 0.11 * fd, 180: 0.0, 270: 0.0, 45: 0.0}
    bf_campaign(tmp_path, metrics_fn=lambda r: bf_metrics(r, r_shift=0.0, shift=shift))
    t3 = C.tier3_bf(collect35(tmp_path), exp25)
    fails = {(c["K"], c["metric"]) for c in t3["checks"] if not c["pass_c4"]}
    assert (1, "FD") in fails and (1, "T60") not in fails and t3["pass"] is False


def test_tier3_bf_with_a_missing_cell_is_not_computable(tmp_path, exp25):
    bf_campaign(tmp_path, metrics_fn=bf_metrics, skip={"exp35_BF_R_S40000_K8_s45"})
    t3 = C.tier3_bf(collect35(tmp_path), exp25)
    assert t3["pass"] is None and any(not c["computable"] for c in t3["checks"] if c["K"] == 8)
    assert all(c["computable"] for c in t3["checks"] if c["K"] == 1)


# --------------------------------------------------------------------------- #
# Codex C3a HIGH 2 / HIGH 3: tier2_c4 alone on synthetic entries (item counts the fixture's N = 8 cannot give)
# --------------------------------------------------------------------------- #
def _synthetic(monkeypatch, n, perturb=None):
    """Every BF-pin / BF-dup cell of the manifest with identical waveforms, item ``i`` of cell ``c`` scaled by
    ``1 + perturb[c][i]`` (its relL2 against an unperturbed reference); the real tier2_c4 helpers except the dump
    reader, which serves these arrays."""
    import numpy as np
    base = np.random.default_rng(0).standard_normal((n, 64)).astype(np.float32) + 2.0
    waves, cells = {}, {}
    for r in [c for c in MANIFEST["cells"] if c["block"] in ("BF-pin", "BF-dup")]:
        w = base.copy()
        for item, eps in (perturb or {}).get(r["cell"], {}).items():
            w[item] *= np.float32(1 + eps)
        waves[r["cell"]] = w
        cells[r["cell"]] = {"row": r, "valid": True, "dump": True,
                            "stream": {"input_hash": "h", "input_tuples": [[i, f"t{i}"] for i in range(n)]},
                            "receipt": {"dump_evidence": {"manifest": {"sha256": r["cell"] if r["cell"] in (
                                perturb or {}) else "same"}}}}
    monkeypatch.setattr(C, "_waveforms", lambda e: waves[e["row"]["cell"]])
    return cells


def test_the_pooled_rule_carries_no_cap_of_its_own(monkeypatch):
    """§4: 1e-2 binds each of the six PAIRS; the pooled p99 is judged by 3 x the pooled control only."""
    pert = {f"exp35_BF_C4_S40000_K{k}_s42_th{d}": {3: 0.0105} for k in (1, 8) for d in (90, 180, 270)}
    t2 = C.tier2_c4(_synthetic(monkeypatch, 8, pert), {"d128": 4e-3, "d256": 4e-3, "d384": 4e-3, "pooled": 4e-3})
    assert all(p["pass_cap"] and p["pass_matched"] and p["relL2"]["p99"] < 1e-2 for p in t2["pairs"])
    assert 1e-2 < t2["pooled"]["relL2"]["p99"] <= 3 * 4e-3 and t2["pooled"]["pass"] is True
    assert t2["pass"] is True and t2["claim"] == "PASS"


def test_a_rare_alarm_suspends_the_claim_while_the_p99_rules_pass(monkeypatch):
    """§4: any C4 item with relL2 > 0.1 suspends the claim -- here one item in 200 (< 1 %), so every p99 rule
    passes; the registered p99 verdict stays PASS, the claim is SUSPENDED with the item named."""
    cell = "exp35_BF_C4_S40000_K8_s42_th180"
    t2 = C.tier2_c4(_synthetic(monkeypatch, 200, {cell: {17: 0.5}}), dict(CONTROL_BF))
    assert t2["pass"] is True and all(p["pass_matched"] and p["pass_cap"] for p in t2["pairs"])
    assert t2["alarm"]["suspended"] and t2["alarm"]["items"] == {cell: ["t17"]} and t2["claim"] == "SUSPENDED"


def test_the_claim_status_otherwise_follows_the_registered_verdict(monkeypatch):
    clean = C.tier2_c4(_synthetic(monkeypatch, 8), dict(CONTROL_BF))
    bad = C.tier2_c4(_synthetic(monkeypatch, 8, {"exp35_BF_C4_S40000_K1_s42_th90": {i: 0.02 for i in range(8)}}),
                     dict(CONTROL_BF))
    assert (clean["claim"], bad["claim"], bad["pass"]) == ("PASS", "FAIL", False)
    assert C.tier2_c4(_synthetic(monkeypatch, 8), None)["claim"] == "NOT COMPUTABLE"
