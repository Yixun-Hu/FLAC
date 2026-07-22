#!/usr/bin/env python3
"""exp-09 Stage D threshold->verdict ADAPTER (plan §4; B-code r1 review requirement).

The B-code r1 review registered that "a thin threshold->verdict adapter must land before
any D acceptance run". ``eval_FLAC.py`` and ``compare_predictions.py`` write RAW metric
JSONs and NEVER decide pass/fail; ``aggregate_gate.py`` only aggregates already-decided
``{gate, pass, metrics}`` verdicts into an exit code. THIS adapter is the missing middle:
it applies the pre-registered D thresholds/bands to the raw metrics and emits the per-gate
verdicts ``aggregate_gate.py`` consumes.

COVERAGE ENFORCEMENT (Codex D-tool review F1) — the headline rule: the adapter takes the
REGISTERED evaluation matrix from the references config as EXPLICIT expectations and REFUSES
(exit 2, nothing written) any input that is not EXACTLY complete — a missing seed/K/angle/
cell OR an unexpected extra entry is a rejection, never a silent partial pass. Only a fully
complete matrix is evaluated into pass/fail.

References config (the D records pin it)::

    {"d1": {"mode": "matched_control"|"contextual", "control_name": "P1",
            "control_stats": {"<K>": {"T60":[mean,std], "C50":[...], "EDT":[...],
                                       "RIR_to_GT_RIR_R@1":[...]}, ...},
            "expect": {"K": [1,8], "seeds": [42,43,44,45,46],   # EXPLICIT eval seed IDs
                       "metrics": ["T60","C50","EDT"],
                       "advisory_metrics": ["RIR_to_GT_RIR_R@1"]}},
     "d2": {"conditioning": {"expect": {"angles": [11.25,45,90,180,270]}},  # A2b j->degrees
            "end_to_end":  {"expect": {"matrix": {"1":[45,90,180,270], "8":[90]}}},  # rot0-FREE
            "flatness":    {"expect": {"matrix": {"1":[45,90,180,270], "8":[90]},
                                        "metrics": ["T60","C50","EDT"]}}}}

The D records must supply a FRESH, UNIQUE ``--out-dir`` per run (records obligation; main()
refuses a non-empty dir). rot0 predictions are retained only as e2e comparator baselines, so
the e2e matrix is rot0-free; flatness still consumes rot-0 as its per-K reference.

Inputs (raw producer artifacts):
* ``--d1``   ``{"<K>": {"seeds": {"<seed_id>": <eval_FLAC json path>, ...}}}`` — seed IDs are
  pinned by ``references.d1.expect.seeds`` (a list of ``{"seed_id","path"}`` is also accepted).
  Duplicate seed IDs, OR a resolved (realpath) artifact reused ANYWHERE in the manifest, =>
  reject (exit 2): "the same artifact five times" can no longer masquerade as five seeds.
* ``--d2-cond``   the REAL A2b audit artifact: ``a2b.per_angle`` records carry ``j`` +
  ``pooled_relerr`` + ``patch_relerr``; ``a2_params.angles`` carries the j->degrees join
  (``degrees`` field, or ``alpha_rad`` -> round(deg, 2)). The converter JOINS j->degrees and
  keys cells by DEGREES (audit_convention.py:1036/1057). A duplicated angle => reject.
* ``--d2-e2e``   ``{"<K>": {"<angle>": <compare_predictions out with waveform_gap.mean_rel_l2>}}``;
* ``--d2-flatness``   ``{"<K>": {"<angle>": <eval_FLAC json | inline metrics>}}``.
  For d2-e2e / d2-flatness a duplicated (K, angle) entry (e.g. "90" and "90.0") => reject.

D1 band math is exp_07 ``gate_verdict.py`` VERBATIM (equivalence <=1 sigma_c / non-inferiority
<=2 sigma_c; sc==0 => n=inf => OUTSIDE). D2 thresholds are INLINED: conditioning <=1e-4,
end-to-end waveform rel-L2 <=0.00931, H-A3 per-(K,metric) exp-01 constants (all-cells).

Output: one ``{gate, pass, metrics}`` verdict JSON per gate under ``--out-dir``
(``verdict_<gate>.json``) + ``advisory_<gate>.json`` for non-gating records, each written
atomically (reusing ``aggregate_gate``'s writer). ``main()`` REFUSES a non-empty ``--out-dir``
(fresh-dir publish, F3). Exit: 0 all-pass / 1 any gate fails / 2 any rejection.
"""
import argparse
import json
import math
import os
import sys
import typing as tp

# Reuse the reviewed atomic writer + finiteness helper so the adapter and aggregator can
# never drift (plan §2: aggregate_gate mirrors the Stage-A audit_convention atomic writer).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_gate import atomic_write_json, collect_non_finite  # noqa: E402

# ============================================================================================
# Pre-registered constants (plan §4). D2 thresholds are INLINED here (not read from refs).
# ============================================================================================
PRIMARY = ("T60", "C50", "EDT")          # gate cells (all lower-better)
ADVISORY = ("RIR_to_GT_RIR_R@1",)        # reported, non-gating (R@1 is higher-better)

D2_COND_THRESHOLD = 1e-4                 # conditioning-level pooled + patch roll rel-err
D2_E2E_REL_L2_THRESHOLD = 0.00931        # end-to-end waveform Metric-1 rel-L2 (exp_08 bound)

# H-A3 flatness, "within 2x exp-01 single-eval noise", constants INLINED from the exp-01
# source (reproduce_flac_table1_results.md:39-40), per plan §4 D2 / r3 #1.
H_A3_THRESHOLDS = {
    1: {"T60": 0.080, "C50": 0.012, "EDT": 0.740},
    8: {"T60": 0.024, "C50": 0.006, "EDT": 0.140},
}

D1_GATE_NAME = "d1_parity"
D1_ADVISORY_NAME = "d1_task_advisory"


class AdapterError(Exception):
    """Any refusal to adapt: incomplete/over-complete coverage, malformed/missing/non-finite
    input, an unknown D1 mode, a missing matched control, a stale output dir. Explicit raise
    -> survives ``python -O`` (fail-closed)."""


# ============================================================================================
# Finiteness + small numeric helpers.
# ============================================================================================
def _finite(value: tp.Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AdapterError(f"non-finite/invalid value at {where}: {value!r}")
    return float(value)


def _maybe_int(k: tp.Any) -> tp.Optional[int]:
    try:
        return int(k)
    except (TypeError, ValueError):
        return None


def seed_stats(values: tp.Sequence[float]) -> tp.Tuple[float, float]:
    """5-seed mean +/- sample std (ddof=1) -- the exp_01 / gate_verdict.py convention.
    n<=1 yields std 0.0 (matches gate_verdict.py.stats)."""
    vals = [_finite(v, f"seed[{i}]") for i, v in enumerate(values)]
    if not vals:
        raise AdapterError("seed_stats: empty value list")
    n = len(vals)
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return mu, sd


# ============================================================================================
# The exp_07 gate_verdict.py band math -- reproduced VERBATIM (gate_verdict.py:91-104).
#   sc = sqrt(sd^2 + rsd^2); n = d/sc if sc>0 else +inf; |n|<=1 equivalence, |n|<=2
#   non-inferiority band (the gate), else outside. sc==0 => n=inf => OUTSIDE (fails 2sc),
#   EVEN for equal means (F4). The emitted verdict never carries a non-finite number
#   (aggregate_gate would reject it): an infinite n is serialised as null + a flag.
# ============================================================================================
def sigma_band(mu: float, sd: float, rmu: float, rsd: float, better_low: bool) -> tp.Dict[str, tp.Any]:
    d = mu - rmu
    sc = math.sqrt(sd * sd + rsd * rsd)
    if sc > 0.0:
        n = d / sc
        an = abs(n)
        within_1sc = an <= 1.0
        within_2sc = an <= 2.0                 # <-- THE gate band (mutation m1 target)
        tier = "equiv<=1sc" if within_1sc else "band<=2sc" if within_2sc else "outside>2sc"
        n_out: tp.Optional[float] = n
        n_inf = False
    else:  # sc==0 => n=inf => OUTSIDE (verbatim gate_verdict.py:91). Equal means still OUTSIDE.
        n_out = None
        n_inf = True
        within_1sc = within_2sc = False
        tier = "outside>2sc"
    if d == 0.0:
        direction = "tie"
    elif better_low:
        direction = "superior" if d < 0.0 else "worse"
    else:
        direction = "superior" if d > 0.0 else "worse"
    return {"d": d, "sc": sc, "n_sigma": n_out, "n_sigma_infinite": bool(n_inf),
            "within_1sc": bool(within_1sc), "within_2sc": bool(within_2sc),
            "tier": tier, "direction": direction}


# ============================================================================================
# Coverage helpers -- the registered matrix as EXPLICIT expectations (F1).
# ============================================================================================
def _k_str_set(seq: tp.Iterable[tp.Any]) -> tp.Set[str]:
    return {str(k) for k in seq}


def _angle_set(seq: tp.Iterable[tp.Any]) -> tp.Set[float]:
    return {float(a) for a in seq}


def _require_exact(got: tp.Set[tp.Any], want: tp.Set[tp.Any], what: str) -> None:
    missing = want - got
    extra = got - want
    if missing or extra:
        raise AdapterError(
            f"{what}: coverage mismatch — missing={sorted(map(str, missing))} "
            f"unexpected={sorted(map(str, extra))} (registered={sorted(map(str, want))})")


# ============================================================================================
# D1: matched-control parity gate vs contextual-only advisory, with coverage enforcement.
# ============================================================================================
def _ref_for_k(ref_stats: tp.Mapping[str, tp.Any], k: tp.Any) -> tp.Mapping[str, tp.Any]:
    for key in (str(k), k, _maybe_int(k)):
        if key is not None and key in ref_stats:
            return ref_stats[key]
    raise AdapterError(f"references have no control stats for K={k!r}")


def _enforce_d1_coverage(measured: tp.Mapping[str, tp.Any], expect: tp.Mapping[str, tp.Any],
                         all_metrics: tp.Sequence[str]) -> tp.List[str]:
    exp_ks = expect.get("K")
    reg_ids = expect.get("seeds")
    if not exp_ks or not isinstance(reg_ids, list) or not reg_ids:
        raise AdapterError(
            "references.d1.expect needs a non-empty 'K' list and a 'seeds' LIST of seed IDs")
    reg_ids_str = [str(s) for s in reg_ids]
    if len(set(reg_ids_str)) != len(reg_ids_str):
        raise AdapterError(f"references.d1.expect.seeds has duplicate seed IDs: {reg_ids}")
    reg_sorted = sorted(reg_ids_str)
    _require_exact(_k_str_set(measured), _k_str_set(exp_ks), "D1 K set")
    for k in measured:
        arm = measured[k]
        ids = [str(s) for s in arm.get("seed_ids", [])]
        # ONE authoritative seed-ID guard: the declared IDs must be EXACTLY the registered set,
        # each once — a sorted-sequence compare rejects repeats, missing, AND extras together.
        if sorted(ids) != reg_sorted:                  # <-- seed-ID exactness+no-repeats (mutation m1)
            raise AdapterError(
                f"D1 K={k} seed IDs {ids} != registered {reg_ids} EXACTLY (no repeats/missing/extra)")
        vals = arm.get("values", {})
        for metric in all_metrics:
            got = vals.get(metric)
            if not isinstance(got, list) or len(got) != len(reg_ids):
                raise AdapterError(
                    f"D1 K={k} metric {metric!r}: expected exactly {len(reg_ids)} seed values, "
                    f"got {0 if got is None else len(got)}")
    return [str(k) for k in sorted(_k_str_set(exp_ks), key=lambda x: int(x))]


def _band_cells(measured, control, ks, metrics, better_low):
    cells, all_within = [], True
    for k_key in ks:                                   # <-- D1 K-arm iteration (F7b target)
        ref_k = _ref_for_k(control, k_key)
        vals_k = (measured[k_key] if k_key in measured else measured[int(k_key)])["values"]
        for metric in metrics:
            mu, sd = seed_stats(vals_k[metric])
            if metric not in ref_k:
                raise AdapterError(f"control_stats K={k_key} missing metric {metric!r}")
            rmu = _finite(ref_k[metric][0], f"ref[{k_key}][{metric}].mean")
            rsd = _finite(ref_k[metric][1], f"ref[{k_key}][{metric}].std")
            band = sigma_band(mu, sd, rmu, rsd, better_low)
            cells.append({"K": k_key, "metric": metric, "mean": mu, "std": sd,
                          "ref_mean": rmu, "ref_std": rsd, **band})
            all_within = all_within and band["within_2sc"]
    return cells, all_within


def build_d1(measured: tp.Mapping[str, tp.Any],
             d1cfg: tp.Mapping[str, tp.Any]) -> tp.Tuple[tp.Optional[dict], tp.Optional[dict]]:
    """``(gate_verdict_or_None, advisory_or_None)``. Coverage is enforced in BOTH modes: the
    measured data must be EXACTLY the registered K x seeds x metrics matrix. matched_control ->
    a GATING d1_parity verdict (pass = all primary cells within 2 sigma_c). contextual -> None
    gate + a descriptive advisory record with NO gating ``pass`` (mode is authoritative: even
    with numbers present, no matched control means no parity verdict -- plan §4 D1)."""
    mode = d1cfg.get("mode")
    if mode not in ("matched_control", "contextual"):
        raise AdapterError(f"references.d1.mode must be 'matched_control' or 'contextual', got {mode!r}")
    expect = d1cfg.get("expect")
    if not isinstance(expect, dict):
        raise AdapterError("references.d1 requires an 'expect' coverage matrix")
    metrics = list(expect.get("metrics", PRIMARY))
    adv_metrics = list(expect.get("advisory_metrics", ADVISORY))
    ks = _enforce_d1_coverage(measured, expect, metrics + adv_metrics)

    # THE emit decision (mutation m2 target): only a matched control yields a parity gate.
    emit_parity_gate = (mode == "matched_control")

    if emit_parity_gate:
        control = d1cfg.get("control_stats")
        if not isinstance(control, dict) or not control:
            raise AdapterError("matched_control mode requires a non-empty 'control_stats'")
        _require_exact(_k_str_set(control) & _k_str_set(ks), _k_str_set(ks),
                       "D1 control_stats K set")
        cells, all_within = _band_cells(measured, control, ks, metrics, better_low=True)
        advisory_cells, _ = _band_cells(measured, control, ks, adv_metrics, better_low=False)
        verdict = {
            "gate": D1_GATE_NAME, "pass": bool(all_within), "mode": mode,
            "control_name": d1cfg.get("control_name"),
            "metrics": {
                "cells": cells, "advisory": advisory_cells, "n_cells": len(cells),
                "n_failing": sum(1 for c in cells if not c["within_2sc"]),
                "equivalence_sigma_c": 1.0, "noninferiority_sigma_c": 2.0,
            },
        }
        return verdict, None

    # contextual-only: descriptive record, NO parity verdict.
    desc: tp.Dict[str, tp.Any] = {"measured": _measured_report(measured, metrics + adv_metrics)}
    ctx = d1cfg.get("contextual_stats")
    if isinstance(ctx, dict) and ctx:
        ctx_cells, _ = _band_cells(measured, ctx, ks, metrics, better_low=True)
        desc["contextual_bands"] = ctx_cells          # tiers reported, NON-gating
    advisory = {
        "gate": D1_ADVISORY_NAME, "advisory": True, "mode": mode,
        "note": d1cfg.get("note", "no matched control pinned; parity verdict withheld (plan §4 D1)"),
        "metrics": desc,
    }
    return None, advisory


def _measured_report(measured, metrics):
    rep: tp.Dict[str, tp.Any] = {}
    for k, arm in measured.items():
        rep[str(k)] = {"seed_ids": list(arm.get("seed_ids", []))}
        for metric in metrics:
            vals = arm["values"][metric]
            mu, sd = seed_stats(vals)
            rep[str(k)][metric] = {"mean": mu, "std": sd, "n": len(vals)}
    return rep


# ============================================================================================
# D2 conditioning-level (A2b-harness schema) -- explicit converter + coverage (F5, F1).
# ============================================================================================
def _extract_cond_records(artifact: tp.Any) -> tp.List[tp.Any]:
    if isinstance(artifact, dict) and isinstance(artifact.get("a2b"), dict) \
            and "per_angle" in artifact["a2b"]:
        return list(artifact["a2b"]["per_angle"])
    if isinstance(artifact, dict) and "per_angle" in artifact:
        return list(artifact["per_angle"])
    if isinstance(artifact, list):
        return list(artifact)
    raise AdapterError("conditioning artifact must be the A2b shape "
                       "{'a2b': {'per_angle': [...]}} (or a bare per-angle list)")


def _build_j_to_degrees(artifact: tp.Any) -> tp.Optional[tp.Dict[int, float]]:
    """Build the ``j -> degrees`` map from ``a2_params.angles`` (audit_convention.py:1036).
    Prefers the explicit ``degrees`` field; else converts ``alpha_rad`` and rounds to 2 dp.
    Returns None when the artifact has no ``a2_params`` (a degree-native producer)."""
    a2p = artifact.get("a2_params") if isinstance(artifact, dict) else None
    if not isinstance(a2p, dict) or "angles" not in a2p:
        return None
    mapping: tp.Dict[int, float] = {}
    for e in a2p["angles"]:
        if not isinstance(e, dict) or "j" not in e:
            raise AdapterError("a2_params.angles entry missing 'j'")
        j = int(e["j"])
        if "degrees" in e:
            deg = _finite(e["degrees"], f"a2_params.j={j}.degrees")
        elif "alpha_rad" in e:
            deg = round(math.degrees(_finite(e["alpha_rad"], f"a2_params.j={j}.alpha_rad")), 2)
        else:
            raise AdapterError(f"a2_params.angles j={j} has neither 'degrees' nor 'alpha_rad'")
        mapping[j] = deg
    return mapping


def _resolve_cond_angle(rec: tp.Mapping[str, tp.Any], j2deg: tp.Optional[tp.Dict[int, float]],
                        i: int) -> float:
    """Resolve a conditioning record's angle in DEGREES. Real A2b records serialise only ``j``;
    they are JOINED to degrees via ``a2_params`` (never treated as the angle directly)."""
    if "j" in rec:
        if j2deg is None:
            raise AdapterError(
                f"conditioning record #{i} carries 'j' but the artifact has no a2_params.angles "
                "to join j->degrees")
        j = int(rec["j"])
        if j not in j2deg:
            raise AdapterError(f"conditioning record #{i}: j={j} not in a2_params.angles {sorted(j2deg)}")
        return j2deg[j]
    for key in ("angle", "rotate_deg"):   # degree-native producer fallback
        if key in rec:
            return float(rec[key])
    raise AdapterError(f"conditioning record #{i} has no 'j' (with a2_params) nor 'angle'/'rotate_deg'")


def conditioning_cells_from_artifact(artifact: tp.Any) -> tp.List[dict]:
    """Convert the raw A2b artifact into conditioning cells keyed by DEGREES. EACH per-angle
    record must carry BOTH ``pooled_relerr`` and ``patch_relerr`` (a record missing a channel
    is a rejection). Real A2b records serialise only ``j`` -> joined to degrees via a2_params.
    A duplicated angle (raw list) is rejected BEFORE any set normalisation (P1-1b)."""
    j2deg = _build_j_to_degrees(artifact)
    cells = []
    seen: tp.Set[float] = set()
    for i, rec in enumerate(_extract_cond_records(artifact)):
        if not isinstance(rec, dict):
            raise AdapterError(f"conditioning record #{i} is not an object")
        angle = _resolve_cond_angle(rec, j2deg, i)
        if angle in seen:                    # <-- duplicate angle in the RAW list (mutation dup)
            raise AdapterError(f"conditioning artifact has a DUPLICATE angle {angle} (deg)")
        seen.add(angle)
        if "pooled_relerr" not in rec or "patch_relerr" not in rec:
            raise AdapterError(
                f"conditioning record at angle {angle}: needs BOTH pooled_relerr AND patch_relerr")
        cells.append({"angle": angle,
                      "pooled_relerr": _finite(rec["pooled_relerr"], f"cond[{angle}].pooled_relerr"),
                      "patch_relerr": _finite(rec["patch_relerr"], f"cond[{angle}].patch_relerr")})
    return cells


def build_conditioning_verdict(cells: tp.Sequence[dict], expect_angles: tp.Sequence[tp.Any]) -> dict:
    if not cells:
        raise AdapterError("conditioning gate needs at least one per-angle record")
    _require_exact({c["angle"] for c in cells}, _angle_set(expect_angles), "conditioning angles")
    pooled_max = max(c["pooled_relerr"] for c in cells)   # reduce over ALL cells (F7a)
    patch_max = max(c["patch_relerr"] for c in cells)
    overall = max(pooled_max, patch_max)
    return {
        "gate": "d2_conditioning", "pass": bool(overall <= D2_COND_THRESHOLD),
        "metrics": {"cells": list(cells), "pooled_max": pooled_max, "patch_roll_max": patch_max,
                    "max_relerr": overall, "threshold": D2_COND_THRESHOLD, "n_angles": len(cells)},
    }


# ============================================================================================
# D2 end-to-end waveform rel-L2 (compare_predictions.waveform_gap.mean_rel_l2) + coverage.
# ============================================================================================
def _one_rel_l2(spec: tp.Any, where: str) -> float:
    if isinstance(spec, dict) and "waveform_gap" in spec:
        wg = spec["waveform_gap"]
        if not isinstance(wg, dict) or "mean_rel_l2" not in wg:
            raise AdapterError(f"{where}.waveform_gap missing 'mean_rel_l2'")
        return _finite(wg["mean_rel_l2"], f"{where}.waveform_gap.mean_rel_l2")
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return _finite(spec, where)
    raise AdapterError(f"unrecognised end-to-end entry at {where}: {type(spec).__name__}")


def e2e_cells_from_index(index: tp.Any) -> tp.List[dict]:
    if not isinstance(index, dict) or not index:
        raise AdapterError("end-to-end index must be a non-empty {K: {angle: compare_predictions}} map")
    cells = []
    for k, angle_map in index.items():
        if not isinstance(angle_map, dict):
            raise AdapterError(f"end-to-end K={k}: angle map must be an object")
        seen: tp.Set[float] = set()
        for angle, spec in angle_map.items():
            fa = float(angle)
            if fa in seen:                    # <-- duplicate (K, angle) BEFORE normalisation (P1-1b)
                raise AdapterError(f"end-to-end K={k}: DUPLICATE angle {angle} (== {fa} deg)")
            seen.add(fa)
            cells.append({"K": str(k), "angle": fa,
                          "rel_l2": _one_rel_l2(spec, f"e2e K{k} rot{angle}")})
    return cells


def build_e2e_verdict(cells: tp.Sequence[dict], expect_matrix: tp.Mapping[str, tp.Any]) -> dict:
    if not cells:
        raise AdapterError("end-to-end gate needs at least one waveform rel-L2 value")
    want = {(str(k), float(a)) for k, angs in expect_matrix.items() for a in angs}
    _require_exact({(c["K"], c["angle"]) for c in cells}, want, "end-to-end (K, angle) matrix")
    worst = max(c["rel_l2"] for c in cells)               # reduce over ALL cells (F7a)
    return {
        "gate": "d2_end_to_end", "pass": bool(worst <= D2_E2E_REL_L2_THRESHOLD),
        "metrics": {"cells": list(cells), "max_rel_l2": worst,
                    "threshold": D2_E2E_REL_L2_THRESHOLD, "n_cells": len(cells)},
    }


# ============================================================================================
# D2 H-A3 flatness: inlined constants + ALL-cells rule + coverage (F1, mutation m3 target).
# ============================================================================================
def build_flatness_verdict(per_k_angle_metrics: tp.Mapping[str, tp.Any],
                           expect_matrix: tp.Mapping[str, tp.Any],
                           metrics: tp.Sequence[str] = PRIMARY) -> dict:
    """H-A3: per (K, metric), for every registered rotated angle,
    ``|metric(rot a) - metric(rot 0)| <= H_A3_THRESHOLDS[K][metric]``; ALL cells pass. Coverage:
    input K set == registered; each K carries rot-0 + EXACTLY the registered rotated angles."""
    _require_exact(_k_str_set(per_k_angle_metrics), _k_str_set(expect_matrix), "H-A3 K set")
    cells: tp.List[dict] = []
    all_pass = True
    for k_key in sorted(_k_str_set(expect_matrix), key=lambda x: int(x)):  # <-- H-A3 K-arm (F7b)
        k = int(k_key)
        if k not in H_A3_THRESHOLDS:
            raise AdapterError(f"H-A3 has no thresholds for K={k} (known: {sorted(H_A3_THRESHOLDS)})")
        thr = H_A3_THRESHOLDS[k]
        angle_map = per_k_angle_metrics[k_key] if k_key in per_k_angle_metrics \
            else per_k_angle_metrics[int(k_key)]
        raw_angles = [float(a) for a in angle_map]        # RAW list BEFORE dict collapse (P1-1b)
        if len(set(raw_angles)) != len(raw_angles):       # <-- duplicate angle keys (e.g. 90/90.0)
            raise AdapterError(f"H-A3 K={k}: DUPLICATE angle keys {list(angle_map)} collapse silently")
        by_angle = {float(a): m for a, m in angle_map.items()}
        if 0.0 not in by_angle:
            raise AdapterError(f"H-A3 K={k}: missing the rot-0 reference angle")
        want_rot = _angle_set(expect_matrix[k_key]) if k_key in expect_matrix \
            else _angle_set(expect_matrix[int(k_key)])
        _require_exact(set(by_angle) - {0.0}, want_rot, f"H-A3 K={k} rotated angles")
        ref = by_angle[0.0]
        for angle in sorted(want_rot):
            m = by_angle[angle]
            for metric in metrics:                        # <-- all metrics/cells (mutation m3)
                cur = _finite(m[metric], f"K{k}.rot{angle}.{metric}")
                base = _finite(ref[metric], f"K{k}.rot0.{metric}")
                delta = abs(cur - base)
                threshold = thr[metric]
                cell_pass = delta <= threshold
                all_pass = all_pass and cell_pass
                cells.append({"K": k, "angle": angle, "metric": metric, "value": cur,
                              "ref_value": base, "delta": delta, "threshold": threshold,
                              "pass": bool(cell_pass)})
    if not cells:
        raise AdapterError("H-A3 produced no cells")
    return {
        "gate": "d2_flatness", "pass": bool(all_pass),
        "metrics": {"cells": cells, "n_cells": len(cells),
                    "n_failing": sum(1 for c in cells if not c["pass"])},
    }


# ============================================================================================
# CLI input loading (consumes the raw eval_FLAC / compare_predictions / A2b JSONs).
# ============================================================================================
def _load_json(path: str) -> tp.Any:
    if not os.path.isfile(path):
        raise AdapterError(f"input file not found: {path}")
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise AdapterError(f"could not read JSON {path}: {exc}") from exc


def _eval_metrics(path: str) -> tp.Dict[str, tp.Any]:
    rec = _load_json(path)
    if not isinstance(rec, dict) or not isinstance(rec.get("metrics"), dict):
        raise AdapterError(f"{path}: not an eval_FLAC metric JSON (no 'metrics' object)")
    return rec["metrics"]


def _seed_pairs(entry: tp.Mapping[str, tp.Any], k: tp.Any) -> tp.List[tp.Tuple[str, str]]:
    """Normalise a K arm's ``seeds`` into ordered ``(seed_id, path)`` pairs. Accepts a
    ``{seed_id: path}`` map OR a list of ``{"seed_id","path"}`` (the list form can express a
    duplicate seed id, which coverage then rejects)."""
    seeds = entry.get("seeds")
    if isinstance(seeds, dict):
        return [(str(sid), path) for sid, path in seeds.items()]
    if isinstance(seeds, list):
        pairs = []
        for j, item in enumerate(seeds):
            if not isinstance(item, dict) or "seed_id" not in item or "path" not in item:
                raise AdapterError(f"D1 K={k} seeds[{j}]: list form needs {{'seed_id','path'}}")
            pairs.append((str(item["seed_id"]), item["path"]))
        return pairs
    raise AdapterError(f"D1 K={k}: 'seeds' must be a {{seed_id: path}} map or a list of {{seed_id,path}}")


def _load_d1_measured(path: str) -> tp.Dict[str, tp.Any]:
    """Load the D1 measured index into ``{K: {"seed_ids":[...], "values": {metric: [..]}}}``.
    The ONLY accepted CLI form is a seed_id -> artifact-PATH manifest: each path is loaded and
    globally realpath-collision-guarded, so five copies of one file cannot masquerade as five
    seeds. The inline ``{"seed_ids":[...], "values": {..}}`` form is REJECTED here (Codex D-tool
    r3): it carries no paths and would bypass the uniqueness guard. That pre-grouped form stays
    available to unit tests ONLY via a DIRECT ``build_d1()`` call (never through the CLI/loader).
    Keeps ALL wanted metric lists so coverage can detect a missing metric/seed."""
    index = _load_json(path)
    if not isinstance(index, dict) or not index:
        raise AdapterError(f"{path}: D1 measured index must be a non-empty object")
    measured: tp.Dict[str, tp.Any] = {}
    wanted = list(PRIMARY) + list(ADVISORY)
    seen_paths: tp.Dict[str, str] = {}   # realpath -> "K=.. seed .." (collision reporting)
    for k, entry in index.items():
        if not isinstance(entry, dict):
            raise AdapterError(f"D1 measured K={k}: entry must be an object")
        if "values" in entry:            # <-- inline values bypass the realpath guard: REJECT (r3)
            raise AdapterError(
                "D1 manifest must map seed_id -> artifact path; inline values are not accepted "
                f"on the CLI (they bypass the realpath uniqueness guard) — K={k}")
        pairs = _seed_pairs(entry, k)
        values: tp.Dict[str, tp.List[float]] = {m: [] for m in wanted}
        for sid, spath in pairs:
            rp = os.path.realpath(spath)
            if rp in seen_paths:         # <-- realpath collision ANYWHERE (mutation m2)
                raise AdapterError(
                    f"D1 artifact path {rp} is reused ({seen_paths[rp]} AND K={k} seed {sid}) — "
                    "the same artifact cannot count as multiple seeds")
            seen_paths[rp] = f"K={k} seed {sid}"
            metrics = _eval_metrics(spath)
            for m in wanted:
                if m in metrics:
                    values[m].append(_finite(metrics[m], f"{spath}:{m}"))
        measured[str(k)] = {"seed_ids": [sid for sid, _ in pairs], "values": values}
    return measured


def _load_flatness(path: str) -> tp.Dict[str, tp.Any]:
    index = _load_json(path)
    if not isinstance(index, dict) or not index:
        raise AdapterError(f"{path}: flatness index must be a non-empty object")
    out: tp.Dict[str, tp.Any] = {}
    for k, angle_map in index.items():
        if not isinstance(angle_map, dict):
            raise AdapterError(f"flatness K={k}: angle map must be an object")
        out[str(k)] = {}
        for angle, spec in angle_map.items():
            if isinstance(spec, str):
                out[str(k)][angle] = _eval_metrics(spec)
            elif isinstance(spec, dict) and "metrics" in spec:
                out[str(k)][angle] = spec["metrics"]
            elif isinstance(spec, dict):
                out[str(k)][angle] = spec
            else:
                raise AdapterError(f"flatness K={k} angle={angle}: need a path or metrics object")
    return out


def _require_expect(d2ref: tp.Mapping[str, tp.Any], gate: str, key: str) -> tp.Mapping[str, tp.Any]:
    block = d2ref.get(gate, {})
    expect = block.get("expect") if isinstance(block, dict) else None
    if not isinstance(expect, dict) or key not in expect:
        raise AdapterError(
            f"references.d2.{gate}.expect.{key} is required (registered coverage matrix; F1)")
    return expect


# ============================================================================================
# Driver.
# ============================================================================================
def run(references: tp.Mapping[str, tp.Any], *, d1: tp.Optional[str] = None,
        d2_cond: tp.Optional[str] = None, d2_e2e: tp.Optional[str] = None,
        d2_flatness: tp.Optional[str] = None) -> tp.Tuple[tp.List[dict], tp.List[dict]]:
    """Build every requested verdict/advisory (raises AdapterError before writing anything)."""
    verdicts: tp.List[dict] = []
    advisories: tp.List[dict] = []
    d2ref = references.get("d2", {}) if isinstance(references.get("d2"), dict) else {}

    if d1 is not None:
        d1cfg = references.get("d1")
        if not isinstance(d1cfg, dict):
            raise AdapterError("references config has no 'd1' block but --d1 was given")
        gate, advisory = build_d1(_load_d1_measured(d1), d1cfg)
        if gate is not None:
            verdicts.append(gate)
        if advisory is not None:
            advisories.append(advisory)
    if d2_cond is not None:
        expect = _require_expect(d2ref, "conditioning", "angles")
        cells = conditioning_cells_from_artifact(_load_json(d2_cond))
        verdicts.append(build_conditioning_verdict(cells, expect["angles"]))
    if d2_e2e is not None:
        expect = _require_expect(d2ref, "end_to_end", "matrix")
        verdicts.append(build_e2e_verdict(e2e_cells_from_index(_load_json(d2_e2e)), expect["matrix"]))
    if d2_flatness is not None:
        expect = _require_expect(d2ref, "flatness", "matrix")
        metrics = list(expect.get("metrics", PRIMARY))
        verdicts.append(build_flatness_verdict(_load_flatness(d2_flatness), expect["matrix"], metrics))

    if not verdicts and not advisories:
        raise AdapterError("no D1/D2 inputs provided; nothing to adapt")
    for rec in verdicts + advisories:                 # finiteness backstop
        bad = collect_non_finite(rec)
        if bad:
            raise AdapterError(f"verdict {rec.get('gate')!r} has non-finite values at: {bad}")
    return verdicts, advisories


def main(argv: tp.Optional[tp.Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="exp-09 Stage D threshold->verdict adapter")
    parser.add_argument("--references", required=True, help="REFERENCES config JSON (modes + matrix)")
    parser.add_argument("--out-dir", required=True, help="FRESH directory for the per-gate verdicts")
    parser.add_argument("--d1", help="D1 measured index (eval_FLAC JSONs grouped by K)")
    parser.add_argument("--d2-cond", dest="d2_cond", help="D2 conditioning A2b artifact JSON")
    parser.add_argument("--d2-e2e", dest="d2_e2e", help="D2 end-to-end compare_predictions index JSON")
    parser.add_argument("--d2-flatness", dest="d2_flatness", help="D2 H-A3 per-(K, angle) metrics JSON")
    args = parser.parse_args(argv)

    try:
        references = _load_json(args.references)
        if not isinstance(references, dict):
            raise AdapterError("references config must be a JSON object")
        # F3 fresh-dir publish: refuse a pre-existing NON-EMPTY output dir (stale verdicts).
        if os.path.exists(args.out_dir) and not os.path.isdir(args.out_dir):
            raise AdapterError(f"--out-dir {args.out_dir!r} exists and is not a directory")
        if os.path.isdir(args.out_dir) and os.listdir(args.out_dir):
            raise AdapterError(
                f"--out-dir {args.out_dir!r} is not empty — each run needs a FRESH dir "
                "(stale-verdict guard, F3); the D records create per-run dirs")
        verdicts, advisories = run(
            references, d1=args.d1, d2_cond=args.d2_cond,
            d2_e2e=args.d2_e2e, d2_flatness=args.d2_flatness)
    # Exit-2 boundary (P2): AdapterError AND any raw input-parse error (bad float/key/type/
    # JSON) map to the documented rejection — never a raw traceback, never a partial write.
    except (AdapterError, ValueError, KeyError, TypeError) as exc:
        print(f"gate_thresholds_to_verdicts: REJECTED — {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2  # hard rejection: no verdict written

    os.makedirs(args.out_dir, exist_ok=True)
    for verdict in verdicts:
        atomic_write_json(os.path.join(args.out_dir, f"verdict_{verdict['gate']}.json"), verdict)
    for advisory in advisories:
        atomic_write_json(os.path.join(args.out_dir, f"advisory_{advisory['gate']}.json"), advisory)

    all_pass = all(v["pass"] for v in verdicts)
    failed = [v["gate"] for v in verdicts if not v["pass"]]
    print(f"gate_thresholds_to_verdicts: {'PASS' if all_pass else 'FAIL'} — "
          f"{len(verdicts)} gate verdict(s), {len(advisories)} advisory; failed={failed} -> {args.out_dir}")
    return 0 if all_pass else 1   # <-- pass/fail exit (mutation m5 target)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
