#!/usr/bin/env python3
"""exp_19 — reproduce every published table from the raw metric JSONs.

Single source of truth for the aggregation that produced:
  results_haa_ft_P1_YAW_rows.md, results_haa_ft_three_arm_table.md,
  results_haa_ft_four_arm_final.md, results_haa_ft_steps_curve.md

Paper convention: per-room mean -> cross-room mean; T60 excludes the dampened
room (FLAC paper D.1/D.2, implemented dataset-side by metric_callback.py's HAA
branch for the flat keys and reproduced here over the recorded by_scene block).
Pooled convention: the evaluator's flat keys (global sample-weighted mean).

Usage: python exp19_aggregate.py [--table {two,three,four,curve}] [--style {paper,pooled}]
Written by the main session seat (Claude Fable 5); part of the exp_19 closure.
"""
import argparse, json, glob, statistics as st

KEYS = ("T60","C50","EDT","RIR_to_GT_RIR_R@1","RIR_to_GT_RIR_R@5","RIR_to_GT_RIR_R@10","FD")
PREC = (4,4,3,3,3,3,4)
NAMES = {"P1":"Vanilla FLAC (P1→HAA)","BF":"Per-angle FA (B-F→HAA)",
         "YAW":"Yaw-Aug, aug ON in FT","YNA":"Yaw-Aug init, aug OFF in FT","BNA":"FA(B-F) init, vanilla FT",
         "CYL":"Cyl-DINOv3 no-SSL (AR-40k→HAA)","CYLSSL":"Cyl-DINOv3 SSL (AR-42.5k→HAA)"}
EXPECTED_CM = {"P1":"vanilla","YAW":"vanilla","YNA":"vanilla","BNA":"vanilla","BF":"fa_invariant",
               "CYL":"fa_invariant","CYLSSL":"fa_invariant"}

def records(arm, step, K, seeds):
    fs = sorted(f for f in glob.glob(
        f"outputs_FLAC/exp19_HAA_{arm}/**/*metrics*exp19_HAA_{arm}_S{step}_K{K}_s*.json",
        recursive=True) if ".stream." not in f
        and any(f"_s{s}." in f or f"_s{s}_" in f for s in seeds))
    if len(fs) != len(seeds):
        raise SystemExit(f"{arm} S{step} K{K}: want {len(seeds)} records, found {len(fs)}")
    out = []
    for f in fs:
        r = json.load(open(f))
        if r.get("cond_method") != EXPECTED_CM[arm] or r.get("cond_autocast") != "bf16":
            raise SystemExit(f"{f}: protocol mismatch ({r.get('cond_method')}/{r.get('cond_autocast')})")
        out.append(r)
    return out

def one(rec, style):
    if style == "pooled":
        return {k: rec["metrics"][k] for k in KEYS}
    bs = rec["metrics"].get("by_scene") or rec.get("by_scene")
    if not bs:
        raise SystemExit("per-scene block missing; cell was not recorded with --record-per-scene")
    return {k: st.mean(v[k] for room, v in bs.items()
                       if not (k == "T60" and "dampened" in room)) for k in KEYS}

def agg(arm, step, K, style, seeds=(42,43,44,45,46)):
    vals = {k: [] for k in KEYS}
    for r in records(arm, step, K, seeds):
        m = one(r, style)
        for k in KEYS: vals[k].append(m[k])
    return {k: ((st.mean(v), st.stdev(v)) if len(v) > 1 else (v[0], float("nan"))) for k, v in vals.items()}

def row(name, m):
    return f"| {name} | " + " | ".join(f"{m[k][0]:.{p}f} ± {m[k][1]:.{p}f}" for k, p in zip(KEYS, PREC)) + " |"

def print_endpoint(arms, style):
    for K in (8, 1):
        print(f"## K = {K} ({style} convention)\n")
        print("| Method | T60↓ | C50↓ | EDT↓ | R@1↑ | R@5↑ | R@10↑ | FD↓ |")
        print("|---|---|---|---|---|---|---|---|")
        for arm in arms:
            print(row(NAMES[arm], agg(arm, 1000, K, style)))
        print()

def print_curve(style):
    steps = [100,200,300,410,500,600,700,800,900,1000]
    for key, lab in (("T60","T60 (%) ↓"),("C50","C50 (dB) ↓"),("EDT","EDT (ms) ↓"),("RIR_to_GT_RIR_R@1","R@1 (%) ↑")):
        print(f"## {lab} (K=8, seed 42, {style})\n")
        print("| steps | " + " | ".join(map(str, steps)) + " |")
        print("|---" * (len(steps) + 1) + "|")
        for arm in ("P1","BF","YAW"):
            cells = []
            for s in steps:
                try:
                    m = agg(arm, s, 8, style, seeds=(42,))
                    cells.append(f"{m[key][0]:.2f}")
                except SystemExit:
                    cells.append("—")
            print(f"| {NAMES[arm].split(' (')[0].split(',')[0]} | " + " | ".join(cells) + " |")
        print()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=["two","three","four","five","seven","curve"], default="four")
    ap.add_argument("--style", choices=["paper","pooled"], default="paper")
    a = ap.parse_args()
    arms = {"two":("P1","YAW"), "three":("P1","BF","YAW"), "four":("P1","YNA","YAW","BF"), "five":("P1","YNA","BNA","YAW","BF"),
            "seven":("P1","YNA","BNA","YAW","BF","CYL","CYLSSL")}.get(a.table)
    if a.table == "curve": print_curve(a.style)
    else: print_endpoint(arms, a.style)
