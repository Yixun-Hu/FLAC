#!/usr/bin/env python3
"""exp_23 — P1 (vanilla FLAC) vs CYL (CylDINO no-SSL) vs CYLORI (CylDINO no-SSL + orientation
field) on HAA, reproduced from the raw metric JSONs with exp_19's aggregation conventions
(paper convention = per-room mean -> cross-room mean, T60 excludes the dampened room).

Usage: python exp23_aggregate.py [--step 1000] [--style paper|pooled] [--arms P1,CYL,CYLORI]
       python exp23_aggregate.py --curve          (K=8, seed 42, steps 100..1000)
       python exp23_aggregate.py --rooms          (per-room breakdown at --step, K=8)
"""
import argparse, json, glob, statistics as st
KEYS = ("T60","C50","EDT","RIR_to_GT_RIR_R@1","RIR_to_GT_RIR_R@5","RIR_to_GT_RIR_R@10","FD")
PREC = (4,4,3,3,3,3,4)
ARMS = {"P1":   ("outputs_FLAC/exp19_HAA_P1",     "exp19_HAA_P1",     "vanilla",      "Vanilla FLAC (P1@40k→HAA)"),
        "YNA":  ("outputs_FLAC/exp19_HAA_YNA",    "exp19_HAA_YNA",    "vanilla",      "Yaw-Aug init, aug OFF in FT"),
        "BF":   ("outputs_FLAC/exp19_HAA_BF",     "exp19_HAA_BF",     "fa_invariant", "Per-angle FA (B-F→HAA)"),
        "CYL":  ("outputs_FLAC/exp19_HAA_CYL",    "exp19_HAA_CYL",    "fa_invariant", "CylDINO no-SSL (AR-40k→HAA)"),
        "CYLSSL":("outputs_FLAC/exp19_HAA_CYLSSL","exp19_HAA_CYLSSL", "fa_invariant", "CylDINO SSL (AR-42.5k→HAA)"),
        "CYLORI":("outputs_FLAC/exp23_HAA_CYLORI","exp23_HAA_CYLORI", "fa_invariant", "CylDINO no-SSL + orientation field s=2.7 (AR-40k→HAA)"),
        "CYLORI27":("outputs_FLAC/exp23_HAA_CYLORI27","exp23_HAA_CYLORI27", "fa_invariant", "CylDINO no-SSL + orientation field s=27 (AR-40k→HAA)"),
        "CYLORI27_s43":("outputs_FLAC/exp23_HAA_CYLORI27_s43","exp23_HAA_CYLORI27_s43", "fa_invariant", "CylDINO no-SSL + orientation field s=27, FT seed 43"),
        "P1_s43":("outputs_FLAC/exp23_HAA_P1_s43","exp23_HAA_P1_s43", "vanilla", "Vanilla FLAC (P1@40k→HAA), FT seed 43")}
def records(arm, step, K, seeds):
    root, stem, cm, _ = ARMS[arm]
    fs = sorted(f for f in glob.glob(f"{root}/**/*metrics*{stem}_S{step}_K{K}_s*.json", recursive=True)
                if ".stream." not in f and any(f"_s{s}." in f or f"_s{s}_" in f for s in seeds))
    if len(fs) != len(seeds): raise SystemExit(f"{arm} S{step} K{K}: want {len(seeds)} records, found {len(fs)}")
    out = []
    for f in fs:
        r = json.load(open(f))
        if r.get("cond_method") != cm or r.get("cond_autocast") != "bf16": raise SystemExit(f"{f}: protocol mismatch")
        out.append(r)
    return out
def one(rec, style):
    if style == "pooled": return {k: rec["metrics"][k] for k in KEYS}
    bs = rec["metrics"].get("by_scene") or rec.get("by_scene")
    return {k: st.mean(v[k] for room, v in bs.items() if not (k == "T60" and "dampened" in room)) for k in KEYS}
def agg(arm, step, K, style, seeds=(42,43,44,45,46)):
    vals = {k: [] for k in KEYS}
    for r in records(arm, step, K, seeds):
        m = one(r, style)
        for k in KEYS: vals[k].append(m[k])
    return {k: ((st.mean(v), st.stdev(v)) if len(v) > 1 else (v[0], float("nan"))) for k, v in vals.items()}
def row(name, m): return f"| {name} | " + " | ".join(f"{m[k][0]:.{p}f} ± {m[k][1]:.{p}f}" for k, p in zip(KEYS, PREC)) + " |"
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--step", type=int, default=1000); ap.add_argument("--style", default="paper")
    ap.add_argument("--arms", default="P1,CYL,CYLORI"); ap.add_argument("--curve", action="store_true"); ap.add_argument("--rooms", action="store_true")
    a = ap.parse_args(); arms = a.arms.split(",")
    if a.curve:
        steps = [100,200,300,410,500,600,700,800,900,1000]
        for key, lab in (("T60","T60 (%) ↓"),("C50","C50 (dB) ↓"),("EDT","EDT (ms) ↓"),("RIR_to_GT_RIR_R@1","R@1 (%) ↑")):
            print(f"## {lab} (K=8, seed 42, {a.style})\n\n| steps | " + " | ".join(map(str, steps)) + " |\n" + "|---" * (len(steps)+1) + "|")
            for arm in arms:
                cells = []
                for s in steps:
                    try: cells.append(f"{agg(arm, s, 8, a.style, seeds=(42,))[key][0]:.2f}")
                    except SystemExit: cells.append("—")
                print(f"| {ARMS[arm][3]} | " + " | ".join(cells) + " |")
            print()
    elif a.rooms:
        for arm in arms:
            rooms = {}
            for r in records(arm, a.step, 8, (42,43,44,45,46)):
                for room, v in (r["metrics"].get("by_scene") or r["by_scene"]).items():
                    for k in ("T60","C50","EDT","RIR_to_GT_RIR_R@1"): rooms.setdefault(room, {}).setdefault(k, []).append(v[k])
            print(f"## {ARMS[arm][3]} (ckpt-{a.step}, K=8, 5 seeds)\n\n| room | T60 | C50 | EDT | R@1 |\n|---|---|---|---|---|")
            for room in sorted(rooms): d = rooms[room]; print(f"| {room} | {st.mean(d['T60']):.3f} | {st.mean(d['C50']):.3f} | {st.mean(d['EDT']):.2f} | {st.mean(d['RIR_to_GT_RIR_R@1']):.2f} |")
            print()
    else:
        for K in (8, 1):
            print(f"## K = {K} (ckpt-{a.step}, {a.style} convention, 5 eval seeds)\n\n| Method | T60↓ | C50↓ | EDT↓ | R@1↑ | R@5↑ | R@10↑ | FD↓ |\n|---|---|---|---|---|---|---|---|")
            for arm in arms: print(row(ARMS[arm][3], agg(arm, a.step, K, a.style)))
            print()
