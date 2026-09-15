#!/usr/bin/env python3
"""exp_23 results writer: P1 vs CYL vs CYLORI tables (paper + pooled), per-room breakdown, steps
curve, and Yixun's LaTeX macro rows (\\bms = best within each K block per column). Writes
results_cylori.md. Run from the FLAC root after the eval chain finishes."""
import sys, os, subprocess, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp23_aggregate import agg, ARMS, records, KEYS
E = os.path.dirname(os.path.abspath(__file__))
def run(*a): return subprocess.run([sys.executable, f"{E}/exp23_aggregate.py", *a], capture_output=True, text=True).stdout
LOWER = {"T60", "C50", "EDT", "FD"}
LATEX_KEYS = [("T60", 3), ("C50", 4), ("EDT", 3), ("RIR_to_GT_RIR_R@1", 3), ("RIR_to_GT_RIR_R@5", 3), ("RIR_to_GT_RIR_R@10", 3)]
def latex_block(arms, labels, step=1000):
    out = []
    for K in (1, 8):
        rows = {a: agg(a, step, K, "paper") for a in arms}
        best = {}
        for k, _ in LATEX_KEYS:
            vals = {a: rows[a][k][0] for a in arms}
            best[k] = (min if k in LOWER else max)(vals, key=vals.get)
        for a in arms:
            cells = []
            for k, p in LATEX_KEYS:
                m, s = rows[a][k]; mac = "bms" if best[k] == a else "ms"
                cells.append(f"$\\{mac}{{{m:.{p}f}}}{{{s:.{p}f}}}$")
            out.append(f"{labels[a]} & {K}\n& " + " & ".join(cells[:3]) + "\n& " + " & ".join(cells[3:]) + " \\\\")
        if K == 1: out.append("\\addlinespace[1.5pt]")
    return "\n".join(out)
def rel(a, b, step=1000, K=8):
    ra, rb = agg(a, step, K, "paper"), agg(b, step, K, "paper")
    return " · ".join(f"{k.replace('RIR_to_GT_RIR_','')} {100*(ra[k][0]-rb[k][0])/rb[k][0]:+.0f}%" for k in ("T60","C50","EDT","RIR_to_GT_RIR_R@1","RIR_to_GT_RIR_R@10"))
md = ["# exp_23 results — CylDINO no-SSL + orientation field (CYLORI) vs vanilla FLAC on HAA", "",
      "*Registered exp_19 HAA recipe verbatim (AR-40k EMA inits, 1,000 steps, seed 42), ckpt-1000, 5 eval seeds, each arm under its own protocol. "
      "P1/CYL rows are the committed exp_19 records; CYLORI is this experiment. Paper convention = per-room mean → cross-room mean, T60 excludes the dampened room.*", ""]
ARMSET = "P1,CYL,CYLORI"
try: agg("CYLORI27", 1000, 8, "paper"); agg("CYLORI27", 1000, 1, "paper"); ARMSET += ",CYLORI27"
except SystemExit: pass
md += [run("--arms", ARMSET, "--step", "1000")]
if "CYLORI27" in ARMSET:
    md += ["### Relative to vanilla FLAC (K=8 / K=1, ckpt-1000) — CYLORI27", "", f"- K=8: {rel('CYLORI27','P1')}", f"- K=1: {rel('CYLORI27','P1',K=1)}", ""]
md += ["### Relative to vanilla FLAC (K=8, ckpt-1000)", "", f"- CYL: {rel('CYL','P1')}", f"- CYLORI: {rel('CYLORI','P1')}", ""]
md += ["### Relative to vanilla FLAC (K=1, ckpt-1000)", "", f"- CYL: {rel('CYL','P1',K=1)}", f"- CYLORI: {rel('CYLORI','P1',K=1)}", ""]
md += ["## Per-room (K=8, ckpt-1000, 5 seeds)", "", run("--arms", ARMSET, "--rooms", "--step", "1000")]
try: md += ["## ckpt-410 (second registered reading)", "", run("--arms", "P1,CYL,CYLORI", "--step", "410")]
except Exception as e: md += [f"(ckpt-410 unavailable: {e})"]
md += ["## Steps curve (K=8, seed 42, paper convention)", "", run("--arms", ("P1,CYLORI,CYLORI27" if "CYLORI27" in ARMSET else "P1,CYLORI"), "--curve")]
md += ["## Pooled convention (evaluator flat keys), ckpt-1000", "", run("--arms", ARMSET, "--step", "1000", "--style", "pooled")]
md += ["## LaTeX rows (Yixun's macro style; \\bms = best in the K block)", "", "```latex",
       latex_block(["P1", "CYLORI"], {"P1": "\\FLAC{}~\\citep{brunetto2026flac}", "CYLORI": "\\CylDINO{} (+facing)"}), "```", "",
       "Multi-row variant (adds the field-off control and, when present, the 10x-scale arm):", "", "```latex",
       latex_block(ARMSET.split(","), {"P1": "\\FLAC{}~\\citep{brunetto2026flac}", "CYL": "\\CylDINO{}", "CYLORI": "\\CylDINO{} (+facing, s=2.7)", "CYLORI27": "\\CylDINO{} (+facing, s=27)"}), "```", ""]
open(f"{E}/results_cylori.md", "w").write("\n".join(md))
print("\n".join(md))
