#!/usr/bin/env python3
"""
Aggregate exp_08 (fa_matched) metric JSONs into the exact numbers used by
fa_matched_results.md / fa_matched_analysis.md / the HTML page.

Single source of truth: reads the committed per-seed eval JSONs, computes
5-seed mean +/- sample-std (ddof=1, matching exp_01's convention), derives the
FA marginal (A-F minus A-V bf16 mirror) with combined sigma, and evaluates the
pre-registered H-A1 tiered bands and the M5 training-seed downgrade rule.

Prints a plain-text report; writes nothing. Re-runnable from repo root:
    python worklog/exp_08_fa_matched_claude/aggregate_results.py
"""
import glob
import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AF_DIR = os.path.join(REPO, "outputs_FLAC", "exp08_AF_ft")
AV_DIR = os.path.join(REPO, "outputs_FLAC", "exp05_V1p_freezebn_ft")
M5_AV = os.path.join(REPO, "outputs_FLAC", "exp08_AVs43_ft")
M5_AF = os.path.join(REPO, "outputs_FLAC", "exp08_AFs43_ft")

SEEDS = [42, 43, 44, 45, 46]
KEYS = ["T60", "C50", "EDT", "FD", "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@10"]
SHORT = {"T60": "T60", "C50": "C50", "EDT": "EDT", "FD": "FD",
         "RIR_to_GT_RIR_R@1": "R@1", "RIR_to_GT_RIR_R@10": "R@10"}


def load(path):
    with open(path) as f:
        return json.load(f)["metrics"]


def mean(xs):
    return sum(xs) / len(xs)


def sstd(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def group(dir_, template, k):
    """mean/std over SEEDS for one arm at fixed K; returns {key:(mean,std,[vals])}."""
    out = {}
    for key in KEYS:
        vals = []
        for s in SEEDS:
            p = os.path.join(dir_, template.format(k=k, s=s))
            matches = glob.glob(p)
            assert len(matches) == 1, f"expected 1 file for {p}, got {matches}"
            vals.append(load(matches[0])[key])
        out[key] = (mean(vals), sstd(vals), vals)
    return out


AF_TPL = "FLAC_exp08_AF_metrics_1_1.0_exp08_AF_K{k}_seed{s}_fa_invariant_a4.json"
AV_TPL = "FLAC_exp05_V1p_freezebn_metrics_1_1.0_exp08_AVmirror_K{k}_seed{s}.json"
V1P_TPL = "FLAC_exp05_V1p_freezebn_metrics_1_1.0_exp05_V1p_K{k}_seed{s}.json"


def fmt_row(label, g):
    cells = []
    for key in KEYS:
        m, sd, _ = g[key]
        cells.append(f"{SHORT[key]} {m:.4f}+-{sd:.4f}")
    return f"{label:<22} " + " | ".join(cells)


print("=" * 100)
print("exp_08 fa_matched — aggregated 5-seed metrics (mean +- sample std, ddof=1)")
print("=" * 100)

arms = {}
for k in (1, 8):
    arms[("AF", k)] = group(AF_DIR, AF_TPL, k)
    arms[("AVmirror", k)] = group(AV_DIR, AV_TPL, k)
    arms[("AV_fp16", k)] = group(AV_DIR, V1P_TPL, k)

for k in (1, 8):
    print(f"\n--- K={k} ---")
    print(fmt_row("A-V fp16 (exp05)", arms[("AV_fp16", k)]))
    print(fmt_row("A-V bf16 mirror", arms[("AVmirror", k)]))
    print(fmt_row("A-F fa_invariant", arms[("AF", k)]))

# ---- H-A1: FA marginal (A-F minus A-V bf16 mirror) with tiered bands ----
print("\n" + "=" * 100)
print("H-A1: FA marginal = A-F - A-V(bf16 mirror); combined sigma_c = sqrt(sd_AF^2 + sd_AV^2)")
print("Tiers: |d|<=1 sig_c equivalence | 1-2 sig_c non-inferiority | >2 sig_c regression; sign gives direction")
print("Lower is better for T60/C50/EDT (regression = positive delta); higher better for R@k")
print("=" * 100)
for k in (1, 8):
    print(f"\n--- K={k} ---")
    for key in ("T60", "C50", "EDT", "RIR_to_GT_RIR_R@1"):
        mf, sf, _ = arms[("AF", k)][key]
        mv, sv, _ = arms[("AVmirror", k)][key]
        d = mf - mv
        sc = math.sqrt(sf ** 2 + sv ** 2)
        n = d / sc if sc > 0 else float("inf")
        better_low = key in ("T60", "C50", "EDT")
        if better_low:
            direction = "SUPERIOR" if d < 0 else "regression"
        else:
            direction = "SUPERIOR" if d > 0 else "regression"
        tier = ("equivalence" if abs(n) <= 1 else
                "non-inferiority" if abs(n) <= 2 else "OUTSIDE-2sig")
        print(f"  {SHORT[key]:<5} A-F {mf:.4f}  A-V {mv:.4f}  d={d:+.4f}  "
              f"sig_c={sc:.4f}  d/sig_c={n:+.1f}  2sig_c=+-{2*sc:.4f}  [{tier}, {direction}]")

# ---- M5: training-seed sensitivity (K=8, eval-seed 42) ----
print("\n" + "=" * 100)
print("M5: training-seed sensitivity pair (K=8, eval-seed 42). Delta_seed = seed43 - seed42.")
print("Downgrade rule: if worst per-arm |Delta_seed| >= |FA effect|/2 the H-A1 cell is DOWNGRADED to indeterminate.")
print("=" * 100)
def load_one(dir_, pat):
    matches = glob.glob(os.path.join(dir_, pat))
    assert len(matches) == 1, f"expected 1 file for {pat} in {dir_}, got {matches}"
    print(f"  M5 file: {matches[0]}")
    return load(matches[0])


avs43 = load_one(M5_AV, "*AVs43_screen_K8*.json")
afs43 = load_one(M5_AF, "*AFs43_screen_K8*.json")
for key in ("T60", "C50", "EDT"):
    # seed-42 K=8 values from the 5-seed groups' seed-42 element (index 0)
    av42 = arms[("AVmirror", 8)][key][2][0]
    af42 = arms[("AF", 8)][key][2][0]
    fa_eff = af42 - av42                       # FA effect at seed 42
    d_av = avs43[key] - av42                   # A-V training-seed delta
    d_af = afs43[key] - af42                   # A-F training-seed delta
    worst = max(abs(d_av), abs(d_af))
    fa_s43 = afs43[key] - avs43[key]           # FA effect reproduced at seed 43
    verdict = "SURVIVES" if worst < abs(fa_eff) / 2 else "DOWNGRADE(indeterminate)"
    print(f"  {SHORT[key]:<5} AV42={av42:.4f} AVs43={avs43[key]:.4f} (d={d_av:+.4f}) | "
          f"AF42={af42:.4f} AFs43={afs43[key]:.4f} (d={d_af:+.4f})")
    print(f"        FA_eff(s42)={fa_eff:+.4f}  FA_eff(s43)={fa_s43:+.4f}  worst|d_seed|={worst:.4f}  "
          f"|FA_eff|/2={abs(fa_eff)/2:.4f}  -> {verdict}")

print("\ndone.")
