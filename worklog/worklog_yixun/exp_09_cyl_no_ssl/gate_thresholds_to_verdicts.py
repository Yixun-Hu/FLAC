#!/usr/bin/env python3
"""exp-09 Stage D threshold->verdict ADAPTER (plan §4; B-code r1 review requirement).

The B-code r1 review registered that "a thin threshold->verdict adapter must land before
any D acceptance run". ``eval_FLAC.py`` and ``compare_predictions.py`` write RAW metric
JSONs and NEVER decide pass/fail; ``aggregate_gate.py`` only aggregates already-decided
``{gate, pass, metrics}`` verdicts and turns them into an exit code. THIS adapter is the
missing middle: it applies the pre-registered D thresholds/bands to the raw metrics and
emits the per-gate verdicts ``aggregate_gate.py`` consumes.

Inputs
------
* ``--references <refs.json>`` (REQUIRED): pins the D1 comparator. Two modes (plan §4 D1):
    - ``matched_control``: a matched P1/B-F control with per-(K, metric) ``[mean, std]``;
      the adapter emits a GATING ``d1_parity`` verdict using exp_07 ``gate_verdict.py``'s
      band math VERBATIM (equivalence <= 1 sigma_c, non-inferiority <= 2 sigma_c).
    - ``contextual``: NO matched control pinned -> NO parity verdict is emitted; the
      adapter writes a descriptive ADVISORY record only (no gating ``pass``).
* ``--d1 <index.json>`` (optional): consumes eval_FLAC metric JSONs. Shape
  ``{K: {"seeds": [<eval_FLAC json paths>, ...]}}`` (5 seeds/K); primary T60/C50/EDT gate,
  R@1 advisory. (``{K: {"values": {metric: [..]}}}`` inline form also accepted for tests.)
* ``--d2-cond <cond.json>`` (optional): conditioning-level rel-errs. Shape
  ``{"pooled_invariance": <scalar|{angle: relerr}>, "patch_roll_equiv": <same>}``. Both
  channels required; gate = max(all) <= 1e-4.
* ``--d2-e2e <e2e.json>`` (optional): compare_predictions output(s). A single
  ``{"waveform_gap": {"mean_rel_l2": ..}}`` or a per-angle map of them; gate = max
  mean_rel_l2 <= 0.00931 (exp_08 registered bound).
* ``--d2-flatness <flat.json>`` (optional): H-A3. Shape ``{K: {angle: <eval_FLAC json
  path|inline metrics>}}``. Per (K, metric, angle!=0): ``|m(rot a) - m(rot 0)| <=`` the
  INLINED exp-01 constant; ALL cells must pass.

Output
------
One ``{gate, pass, metrics}`` verdict JSON per gate under ``--out-dir`` (``verdict_<gate>
.json``), plus ``advisory_<gate>.json`` for non-gating records -- each written atomically
(reusing ``aggregate_gate``'s writer, so the two never drift) with only finite numbers.
Exit code: 0 iff every EMITTED gate verdict passes; 1 if any fails; 2 on any input
rejection (malformed/missing/non-finite) -- fail-closed, no partial verdict on rejection.

Run:  python gate_thresholds_to_verdicts.py --references refs.json --out-dir D \
        --d1 d1.json --d2-cond cond.json --d2-e2e e2e.json --d2-flatness flat.json
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
    """Any refusal to adapt: malformed/missing/non-finite input, an unknown D1 mode, a
    missing matched control, or a missing rot-0 flatness reference. Explicit raise ->
    survives ``python -O`` (fail-closed)."""


# ============================================================================================
# Finiteness + small numeric helpers.
# ============================================================================================
def _finite(value: tp.Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AdapterError(f"non-finite/invalid value at {where}: {value!r}")
    return float(value)


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


def _as_values(obj: tp.Any, where: str) -> tp.List[float]:
    """Normalise a scalar / {angle: v} / [v, ..] rel-err spec into a finite float list."""
    if isinstance(obj, dict):
        return [_finite(v, f"{where}.{k}") for k, v in obj.items()]
    if isinstance(obj, (list, tuple)):
        return [_finite(v, f"{where}[{i}]") for i, v in enumerate(obj)]
    return [_finite(obj, where)]


# ============================================================================================
# The exp_07 gate_verdict.py band math -- reproduced VERBATIM.
#   sc = sqrt(sd^2 + rsd^2); n = (mu - rmu)/sc; |n|<=1 equivalence, |n|<=2 non-inferiority
#   band (the gate), else outside. Direction from `better_low`. Degenerate sc==0 handled so
#   the emitted verdict NEVER carries a non-finite number (aggregate_gate would reject it).
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
    else:  # both stds exactly zero: equal -> tie/pass, differ -> outside/fail (no inf leaks)
        n_out = None
        within_1sc = within_2sc = (d == 0.0)
        tier = "equiv<=1sc" if d == 0.0 else "outside>2sc"
    if d == 0.0:
        direction = "tie"
    elif better_low:
        direction = "superior" if d < 0.0 else "worse"
    else:
        direction = "superior" if d > 0.0 else "worse"
    return {"d": d, "sc": sc, "n_sigma": n_out, "within_1sc": bool(within_1sc),
            "within_2sc": bool(within_2sc), "tier": tier, "direction": direction}


# ============================================================================================
# D1: matched-control parity gate vs contextual-only advisory.
# ============================================================================================
def _ref_for_k(ref_stats: tp.Mapping[str, tp.Any], k: tp.Any) -> tp.Mapping[str, tp.Any]:
    for key in (k, str(k), _maybe_int(k)):
        if key is not None and key in ref_stats:
            return ref_stats[key]
    raise AdapterError(f"references have no stats for K={k!r}")


def _maybe_int(k: tp.Any) -> tp.Optional[int]:
    try:
        return int(k)
    except (TypeError, ValueError):
        return None


def _band_cells(measured: tp.Mapping[str, tp.Any], ref_stats: tp.Mapping[str, tp.Any],
                metrics: tp.Sequence[str], better_low: bool) -> tp.Tuple[tp.List[dict], bool]:
    cells: tp.List[dict] = []
    all_within = True
    for k, per_metric in measured.items():
        ref_k = _ref_for_k(ref_stats, k)
        for metric in metrics:
            if metric not in per_metric:
                raise AdapterError(f"measured K={k} missing metric {metric!r}")
            mu, sd = seed_stats(per_metric[metric])
            if metric not in ref_k:
                raise AdapterError(f"reference K={k} missing metric {metric!r}")
            rmu = _finite(ref_k[metric][0], f"ref[{k}][{metric}].mean")
            rsd = _finite(ref_k[metric][1], f"ref[{k}][{metric}].std")
            band = sigma_band(mu, sd, rmu, rsd, better_low)
            cell = {"K": k, "metric": metric, "mean": mu, "std": sd,
                    "ref_mean": rmu, "ref_std": rsd, **band}
            cells.append(cell)
            all_within = all_within and band["within_2sc"]
    return cells, all_within


def build_d1(measured: tp.Mapping[str, tp.Any],
             d1cfg: tp.Mapping[str, tp.Any]) -> tp.Tuple[tp.Optional[dict], tp.Optional[dict]]:
    """Return ``(gate_verdict_or_None, advisory_or_None)``.

    matched_control -> a GATING d1_parity verdict (pass = all primary cells within 2 sigma_c),
    R@1 reported as advisory. contextual -> None gate + a descriptive advisory record with
    NO gating ``pass`` (mode is authoritative: even if numbers are present, no matched
    control means no parity verdict -- plan §4 D1)."""
    mode = d1cfg.get("mode")
    if mode not in ("matched_control", "contextual"):
        raise AdapterError(f"references.d1.mode must be 'matched_control' or 'contextual', got {mode!r}")

    # THE emit decision (mutation m2 target): only a matched control yields a parity gate.
    emit_parity_gate = (mode == "matched_control")

    if emit_parity_gate:
        control = d1cfg.get("control_stats")
        if not isinstance(control, dict) or not control:
            raise AdapterError("matched_control mode requires a non-empty 'control_stats'")
        cells, all_within = _band_cells(measured, control, PRIMARY, better_low=True)
        advisory_cells, _ = _band_cells(measured, control, ADVISORY, better_low=False) \
            if _measured_has(measured, ADVISORY) else ([], True)
        verdict = {
            "gate": D1_GATE_NAME, "pass": bool(all_within), "mode": mode,
            "control_name": d1cfg.get("control_name"),
            "metrics": {
                "cells": cells, "advisory": advisory_cells,
                "n_cells": len(cells),
                "n_failing": sum(1 for c in cells if not c["within_2sc"]),
                "equivalence_sigma_c": 1.0, "noninferiority_sigma_c": 2.0,
            },
        }
        return verdict, None

    # contextual-only: descriptive record, NO parity verdict.
    desc: tp.Dict[str, tp.Any] = {"measured": _measured_report(measured)}
    ctx = d1cfg.get("contextual_stats")
    if isinstance(ctx, dict) and ctx:
        ctx_cells, _ = _band_cells(measured, ctx, PRIMARY, better_low=True)
        desc["contextual_bands"] = ctx_cells          # tiers reported, NON-gating
    advisory = {
        "gate": D1_ADVISORY_NAME, "advisory": True, "mode": mode,
        "note": d1cfg.get("note", "no matched control pinned; parity verdict withheld (plan §4 D1)"),
        "metrics": desc,
    }
    return None, advisory


def _measured_has(measured: tp.Mapping[str, tp.Any], metrics: tp.Sequence[str]) -> bool:
    return all(all(m in per for m in metrics) for per in measured.values())


def _measured_report(measured: tp.Mapping[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    rep: tp.Dict[str, tp.Any] = {}
    for k, per in measured.items():
        rep[str(k)] = {}
        for metric, vals in per.items():
            mu, sd = seed_stats(vals)
            rep[str(k)][metric] = {"mean": mu, "std": sd, "n": len(list(vals))}
    return rep


# ============================================================================================
# D2 conditioning-level, end-to-end, and H-A3 flatness verdicts.
# ============================================================================================
def build_conditioning_verdict(pooled: tp.Any, patch: tp.Any) -> dict:
    pooled_vals = _as_values(pooled, "pooled_invariance")
    patch_vals = _as_values(patch, "patch_roll_equiv")
    if not pooled_vals or not patch_vals:
        raise AdapterError("conditioning gate needs BOTH pooled invariance AND patch "
                           "roll-equivariance rel-errs (plan §4 D2)")
    pooled_max, patch_max = max(pooled_vals), max(patch_vals)
    overall = max(pooled_max, patch_max)
    return {
        "gate": "d2_conditioning", "pass": bool(overall <= D2_COND_THRESHOLD),
        "metrics": {"pooled_max": pooled_max, "patch_roll_max": patch_max,
                    "max_relerr": overall, "threshold": D2_COND_THRESHOLD},
    }


def extract_rel_l2(e2e: tp.Any) -> tp.List[float]:
    """Pull waveform rel-L2 value(s) out of compare_predictions output(s)."""
    def one(x: tp.Any, where: str) -> float:
        if isinstance(x, dict) and "waveform_gap" in x:
            wg = x["waveform_gap"]
            if not isinstance(wg, dict) or "mean_rel_l2" not in wg:
                raise AdapterError(f"{where}.waveform_gap missing 'mean_rel_l2'")
            return _finite(wg["mean_rel_l2"], f"{where}.waveform_gap.mean_rel_l2")
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return _finite(x, where)
        raise AdapterError(f"unrecognised end-to-end entry at {where}: {type(x).__name__}")

    if isinstance(e2e, dict) and "waveform_gap" in e2e:
        return [one(e2e, "e2e")]
    if isinstance(e2e, dict):
        return [one(v, f"e2e.{k}") for k, v in e2e.items()]
    if isinstance(e2e, (list, tuple)):
        return [one(v, f"e2e[{i}]") for i, v in enumerate(e2e)]
    raise AdapterError("end-to-end input must be a compare_predictions dict, a per-angle map, or a list")


def build_e2e_verdict(rel_l2_values: tp.Sequence[float]) -> dict:
    vals = [_finite(v, f"rel_l2[{i}]") for i, v in enumerate(rel_l2_values)]
    if not vals:
        raise AdapterError("end-to-end gate needs at least one waveform rel-L2 value")
    mx = max(vals)
    return {
        "gate": "d2_end_to_end", "pass": bool(mx <= D2_E2E_REL_L2_THRESHOLD),
        "metrics": {"max_rel_l2": mx, "threshold": D2_E2E_REL_L2_THRESHOLD, "n_angles": len(vals)},
    }


def build_flatness_verdict(per_k_angle_metrics: tp.Mapping[str, tp.Any]) -> dict:
    """H-A3: per (K, metric), for every evaluated angle != 0,
    ``|metric(rot a) - metric(rot 0)| <= H_A3_THRESHOLDS[K][metric]``; ALL cells pass."""
    cells: tp.List[dict] = []
    all_pass = True
    for k_raw, angle_map in per_k_angle_metrics.items():
        k = _maybe_int(k_raw)
        if k not in H_A3_THRESHOLDS:
            raise AdapterError(f"H-A3 has no thresholds for K={k_raw!r} (known: {sorted(H_A3_THRESHOLDS)})")
        thr = H_A3_THRESHOLDS[k]
        ref = None
        for a_raw, metrics in angle_map.items():
            if float(a_raw) == 0.0:
                ref = metrics
        if ref is None:
            raise AdapterError(f"H-A3 K={k}: missing the rot-0 reference angle")
        for a_raw, metrics in angle_map.items():
            angle = float(a_raw)
            if angle == 0.0:
                continue
            for metric in ("T60", "C50", "EDT"):   # <-- all three cells (mutation m3 target)
                cur = _finite(metrics[metric], f"K{k}.rot{angle}.{metric}")
                base = _finite(ref[metric], f"K{k}.rot0.{metric}")
                delta = abs(cur - base)
                threshold = thr[metric]
                cell_pass = delta <= threshold
                all_pass = all_pass and cell_pass
                cells.append({"K": k, "angle": angle, "metric": metric,
                              "value": cur, "ref_value": base, "delta": delta,
                              "threshold": threshold, "pass": bool(cell_pass)})
    if not cells:
        raise AdapterError("H-A3 produced no cells (need at least one rotated angle per K)")
    return {
        "gate": "d2_flatness", "pass": bool(all_pass),
        "metrics": {"cells": cells, "n_cells": len(cells),
                    "n_failing": sum(1 for c in cells if not c["pass"])},
    }


# ============================================================================================
# CLI input loading (consumes the raw eval_FLAC / compare_predictions JSONs).
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
    """Extract the ``metrics`` dict from an eval_FLAC-shaped JSON record."""
    rec = _load_json(path)
    if not isinstance(rec, dict) or not isinstance(rec.get("metrics"), dict):
        raise AdapterError(f"{path}: not an eval_FLAC metric JSON (no 'metrics' object)")
    return rec["metrics"]


def _load_d1_measured(path: str) -> tp.Dict[str, tp.Any]:
    """Load the D1 measured index into ``{K: {metric: [values across seeds]}}``.
    ``seeds`` = list of eval_FLAC JSON paths (primary path); ``values`` = inline lists."""
    index = _load_json(path)
    if not isinstance(index, dict) or not index:
        raise AdapterError(f"{path}: D1 measured index must be a non-empty object")
    measured: tp.Dict[str, tp.Any] = {}
    wanted = PRIMARY + ADVISORY
    for k, entry in index.items():
        if isinstance(entry, dict) and "seeds" in entry:
            per_metric: tp.Dict[str, tp.List[float]] = {m: [] for m in wanted}
            for spath in entry["seeds"]:
                metrics = _eval_metrics(spath)
                for m in wanted:
                    if m in metrics:
                        per_metric[m].append(_finite(metrics[m], f"{spath}:{m}"))
            per_metric = {m: v for m, v in per_metric.items() if v}
        elif isinstance(entry, dict) and "values" in entry:
            per_metric = {m: [_finite(x, f"K{k}:{m}") for x in vals]
                          for m, vals in entry["values"].items()}
        else:
            raise AdapterError(f"D1 measured K={k}: entry needs a 'seeds' list or 'values' map")
        if not any(m in per_metric for m in PRIMARY):
            raise AdapterError(f"D1 measured K={k}: no primary metric (T60/C50/EDT) found")
        measured[k] = per_metric
    return measured


def _load_flatness(path: str) -> tp.Dict[str, tp.Any]:
    """Load ``{K: {angle: <eval_FLAC json path | inline metrics>}}`` into nested metrics."""
    index = _load_json(path)
    if not isinstance(index, dict) or not index:
        raise AdapterError(f"{path}: flatness index must be a non-empty object")
    out: tp.Dict[str, tp.Any] = {}
    for k, angle_map in index.items():
        if not isinstance(angle_map, dict):
            raise AdapterError(f"flatness K={k}: angle map must be an object")
        out[k] = {}
        for angle, spec in angle_map.items():
            if isinstance(spec, str):
                out[k][angle] = _eval_metrics(spec)
            elif isinstance(spec, dict) and "metrics" in spec:
                out[k][angle] = spec["metrics"]
            elif isinstance(spec, dict):
                out[k][angle] = spec
            else:
                raise AdapterError(f"flatness K={k} angle={angle}: need a path or metrics object")
    return out


# ============================================================================================
# Driver.
# ============================================================================================
def run(references: tp.Mapping[str, tp.Any], out_dir: str, *, d1: tp.Optional[str] = None,
        d2_cond: tp.Optional[str] = None, d2_e2e: tp.Optional[str] = None,
        d2_flatness: tp.Optional[str] = None) -> tp.Tuple[tp.List[dict], tp.List[dict]]:
    """Build every requested verdict/advisory (raises AdapterError before writing anything)."""
    verdicts: tp.List[dict] = []
    advisories: tp.List[dict] = []

    if d1 is not None:
        if "d1" not in references:
            raise AdapterError("references config has no 'd1' block but --d1 was given")
        measured = _load_d1_measured(d1)
        gate, advisory = build_d1(measured, references["d1"])
        if gate is not None:
            verdicts.append(gate)
        if advisory is not None:
            advisories.append(advisory)
    if d2_cond is not None:
        cfg = _load_json(d2_cond)
        verdicts.append(build_conditioning_verdict(
            cfg.get("pooled_invariance"), cfg.get("patch_roll_equiv")))
    if d2_e2e is not None:
        verdicts.append(build_e2e_verdict(extract_rel_l2(_load_json(d2_e2e))))
    if d2_flatness is not None:
        verdicts.append(build_flatness_verdict(_load_flatness(d2_flatness)))

    if not verdicts and not advisories:
        raise AdapterError("no D1/D2 inputs provided; nothing to adapt")

    # Backstop: no emitted record may carry a non-finite number (aggregate_gate rejects them).
    for rec in verdicts + advisories:
        bad = collect_non_finite(rec)
        if bad:
            raise AdapterError(f"verdict {rec.get('gate')!r} has non-finite values at: {bad}")
    return verdicts, advisories


def main(argv: tp.Optional[tp.Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="exp-09 Stage D threshold->verdict adapter")
    parser.add_argument("--references", required=True, help="REFERENCES config JSON (D1 mode + comparator)")
    parser.add_argument("--out-dir", required=True, help="directory for the per-gate verdict JSONs")
    parser.add_argument("--d1", help="D1 measured index (eval_FLAC JSONs grouped by K)")
    parser.add_argument("--d2-cond", dest="d2_cond", help="D2 conditioning-level rel-errs JSON")
    parser.add_argument("--d2-e2e", dest="d2_e2e", help="D2 end-to-end compare_predictions JSON(s)")
    parser.add_argument("--d2-flatness", dest="d2_flatness", help="D2 H-A3 per-(K, angle) metrics JSON")
    args = parser.parse_args(argv)

    try:
        references = _load_json(args.references)
        if not isinstance(references, dict):
            raise AdapterError("references config must be a JSON object")
        verdicts, advisories = run(
            references, args.out_dir, d1=args.d1, d2_cond=args.d2_cond,
            d2_e2e=args.d2_e2e, d2_flatness=args.d2_flatness)
    except AdapterError as exc:
        print(f"gate_thresholds_to_verdicts: REJECTED — {exc}", file=sys.stderr)
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
