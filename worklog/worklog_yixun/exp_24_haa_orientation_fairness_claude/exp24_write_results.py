#!/usr/bin/env python3
"""exp_24 results writer — HAA orientation-cue fairness ablation. Reuses exp_23's aggregation (paper convention =
per-room mean -> cross-room mean, T60 excludes the dampened room; pooled = evaluator flat keys) and extends its arm
table with the two stock-backbone cue arms. Writes results_fairness.md; arms whose records are missing are skipped."""
import sys, os, statistics as st
E = os.path.dirname(os.path.abspath(__file__)); E23 = os.path.join(os.path.dirname(E), "exp_23_haa_cyl_orientation_claude")
sys.path.insert(0, E23); import exp23_aggregate as X
X.ARMS.update({
 "P1ORI27":  ("outputs_FLAC/exp24_HAA_P1ORI27",  "exp24_HAA_P1ORI27",  "vanilla", "Vanilla FLAC + orientation cue s=27 (P1@40k→HAA)"),
 "YAW":      ("outputs_FLAC/exp19_HAA_YAW",      "exp19_HAA_YAW",      "vanilla", "Yaw-Aug FLAC, aug ON in FT (exp17@40k→HAA)"),
 "YAWORI27": ("outputs_FLAC/exp24_HAA_YAWORI27", "exp24_HAA_YAWORI27", "vanilla", "Yaw-Aug FLAC, aug ON in FT + orientation cue s=27"),
 "P1ZUP27":  ("outputs_FLAC/exp24_HAA_P1ZUP27",  "exp24_HAA_P1ZUP27",  "vanilla",      "CONTROL: Vanilla FLAC + constant UP field s=27 (no orientation info)"),
 "CYLZUP27": ("outputs_FLAC/exp24_HAA_CYLZUP27", "exp24_HAA_CYLZUP27", "fa_invariant", "CONTROL: CylDINO + constant UP field s=27 (no orientation info)"),
})
ORDER = ["P1", "P1ORI27", "P1ZUP27", "YAW", "YAWORI27", "YNA", "CYL", "CYLORI27", "CYLZUP27"]
def have(arm, step=1000, K=8, seeds=(42,43,44,45,46)):
    try: X.records(arm, step, K, seeds); return True
    except SystemExit: return False
arms = [a for a in ORDER if have(a) and have(a, K=1)]
md = ["# exp_24 results — HAA orientation-cue fairness ablation", "",
      "*Same HAA protocol for every arm (exp_19 registered recipe: AR-40k EMA inits, 1,000 steps, batch 16×accum 4, AdamW 5e-6, seed 42; "
      "eval ckpt-1000, test split, K∈{8,1}, 5 eval seeds, each arm on its own conditioning path, bf16, cfg 1.0, one step). "
      "Cue = the loudspeaker facing direction appended as a zero-initialised XYZ triple (scale 27) to the geometry-encoder input; on the "
      "stock (world-frame) backbone it is a constant field, on CylDINO the gauge makes it column-relative. P1/YAW/YNA/CYL rows are the "
      "committed exp_19 records, CYLORI27 is exp_23, P1ORI27 and YAWORI27 are this experiment. Paper convention unless stated.*", "",
      f"Arms with complete records: {', '.join(arms)}", ""]
def table(step, style):
    out = []
    for K in (8, 1):
        out += [f"## K = {K} (ckpt-{step}, {style} convention, 5 eval seeds)", "", "| Method | T60↓ | C50↓ | EDT↓ | R@1↑ | R@5↑ | R@10↑ | FD↓ |", "|---|---|---|---|---|---|---|---|"]
        for a in arms: out.append(X.row(X.ARMS[a][3], X.agg(a, step, K, style)))
        out.append("")
    return out
md += table(1000, "paper")
def rel(a, b, K=8, step=1000):
    ra, rb = X.agg(a, step, K, "paper"), X.agg(b, step, K, "paper")
    return " · ".join(f"{k.replace('RIR_to_GT_RIR_','')} {100*(ra[k][0]-rb[k][0])/rb[k][0]:+.1f}%" for k in ("T60","C50","EDT","RIR_to_GT_RIR_R@1","RIR_to_GT_RIR_R@10"))
md += ["## Controlled comparisons (ckpt-1000; first arm relative to second)", ""]
for a, b, why in (("CYLORI27", "P1ORI27", "KEY: CylDINO + cue vs FLAC + cue — isolates CERPA with the conditioning information fixed"),
                  ("P1ORI27", "P1", "does the cue alone help vanilla FLAC?"),
                  ("YAWORI27", "YAW", "does the cue restore what yaw augmentation removed?"),
                  ("CYLORI27", "P1", "CylDINO + cue vs plain FLAC (exp_23 headline)"),
                  ("CYLORI27", "CYL", "the cue's effect on CylDINO"),
                  ("P1ZUP27", "P1", "CONTROL: does a constant NON-orientation field (same scale) give vanilla FLAC the same gains as the facing cue?"),
                  ("P1ORI27", "P1ZUP27", "facing cue vs constant-up field on vanilla FLAC (information beyond the bias effect?)"),
                  ("CYLZUP27", "CYL", "CONTROL: constant NON-orientation field on CylDINO (bias effect alone)"),
                  ("CYLORI27", "CYLZUP27", "facing cue vs constant-up field on CylDINO (the orientation information itself)")):
    if a in arms and b in arms: md += [f"- **{a} vs {b}** ({why}): K=8 {rel(a,b)}; K=1 {rel(a,b,K=1)}"]
md += ["", "## Per-room (K=8, ckpt-1000, 5 seeds)", ""]
for a in arms:
    rooms = {}
    for r in X.records(a, 1000, 8, (42,43,44,45,46)):
        for room, v in (r["metrics"].get("by_scene") or r["by_scene"]).items():
            for k in ("T60","C50","EDT","RIR_to_GT_RIR_R@1"): rooms.setdefault(room, {}).setdefault(k, []).append(v[k])
    md += [f"### {X.ARMS[a][3]}", "", "| room | T60 | C50 | EDT | R@1 |", "|---|---|---|---|---|"]
    for room in sorted(rooms): d = rooms[room]; md.append(f"| {room} | {st.mean(d['T60']):.3f} | {st.mean(d['C50']):.3f} | {st.mean(d['EDT']):.2f} | {st.mean(d['RIR_to_GT_RIR_R@1']):.2f} |")
    md.append("")
steps = [100,200,300,400,500,600,700,800,900,1000]
md += ["## Steps curve (K=8, seed 42, paper convention)", ""]
for key, lab in (("T60","T60 (%) ↓"),("C50","C50 (dB) ↓"),("EDT","EDT (ms) ↓")):
    md += [f"### {lab}", "", "| steps | " + " | ".join(map(str, steps)) + " |", "|---" * (len(steps)+1) + "|"]
    for a in arms:
        cells = []
        for s in steps:
            try: cells.append(f"{X.agg(a, s, 8, 'paper', seeds=(42,))[key][0]:.2f}")
            except SystemExit: cells.append("—")
        md.append(f"| {X.ARMS[a][3]} | " + " | ".join(cells) + " |")
    md.append("")
md += table(1000, "pooled")
LOWER = {"T60", "C50", "EDT", "FD"}; LK = [("T60", 3), ("C50", 4), ("EDT", 3), ("RIR_to_GT_RIR_R@1", 3), ("RIR_to_GT_RIR_R@5", 3), ("RIR_to_GT_RIR_R@10", 3)]
LAB = {"P1": "\\FLAC{}", "P1ORI27": "\\FLAC{} (+facing)", "P1ZUP27": "\\FLAC{} (+constant field)", "CYLZUP27": "\\CylDINO{} (+constant field)", "YAW": "Yaw-aug \\FLAC{}", "YAWORI27": "Yaw-aug \\FLAC{} (+facing)", "YNA": "Yaw-aug init, stock FT", "CYL": "\\CylDINO{}", "CYLORI27": "\\CylDINO{} (+facing)"}
md += ["## LaTeX rows (\\bms = best in the K block over the listed arms)", "", "```latex"]
for K in (1, 8):
    rows = {a: X.agg(a, 1000, K, "paper") for a in arms}; best = {k: (min if k in LOWER else max)({a: rows[a][k][0] for a in arms}, key=lambda a: rows[a][k][0]) for k, _ in LK}
    for a in arms:
        cells = [f"$\\{'bms' if best[k] == a else 'ms'}{{{rows[a][k][0]:.{p}f}}}{{{rows[a][k][1]:.{p}f}}}$" for k, p in LK]
        md.append(f"{LAB[a]} & {K} & " + " & ".join(cells) + " \\\\")
    if K == 1: md.append("\\addlinespace[1.5pt]")
md += ["```", ""]
open(f"{E}/results_fairness.md", "w").write("\n".join(md)); print("\n".join(md))
