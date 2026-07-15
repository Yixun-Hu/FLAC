#!/usr/bin/env python3
"""exp_07 pre-registered B-V gate: B-V@67,500 vs released Table-1, tiered sigma_c bands.

Reads the gate-eval JSONs (exp07_BV_gate_K{8,1}_seed{42..46}) written by eval_FLAC.py,
computes 5-eval-seed mean +/- sample std (ddof=1, exp_01 convention), forms combined
sigma_c = sqrt(sd_rel^2 + sd_BV^2) against the exp_01 released-baseline 5-seed stats,
and prints the per-cell tier (|d|<=1 sig_c equivalence / <=2 non-inferiority-band /
>2 outside; sign = direction; lower-better for T60/C50/EDT, higher for R@1-advisory).
Also prints the selection-curve rows and the 291k corroborating row if present.

Gate rule (plan §2c + Yixun step 4/5): PASS cell = within 2 sigma_c. The launch
decision on a non-PASS profile belongs to Yixun (stop-and-ask).

Run from repo root: python worklog/worklog_yixun/exp_07_fa_scratch_claude/gate_verdict.py
"""
import glob
import json
import math
import os


def _repo_root(p):  # marker-walk (layout-proof)
    p = os.path.abspath(p)
    while not os.path.isdir(os.path.join(p, ".git")):
        parent = os.path.dirname(p)
        if parent == p:
            raise RuntimeError("repo root (.git) not found")
        p = parent
    return p


REPO = _repo_root(os.path.dirname(os.path.abspath(__file__)))
BV = os.path.join(REPO, "outputs_FLAC", "exp07_BV")

# exp_01 released-baseline stats (5 eval-seeds, full unseen split) — provenance:
# worklog_yixun/exp_01_reproduce_flac_table1_claude/reproduce_flac_table1_results.md
REL = {
    8: {"T60": (8.609, 0.012), "C50": (0.9682, 0.0030), "EDT": (37.10, 0.07),
        "RIR_to_GT_RIR_R@1": (7.06, 0.10)},
    1: {"T60": (9.969, 0.039), "C50": (1.0460, 0.0064), "EDT": (39.95, 0.37),
        "RIR_to_GT_RIR_R@1": (6.83, 0.22)},
}
# PRIMARY (gate cells) / ADVISORY (reported, non-gating; plan §3) defined with load() below
SEEDS = (42, 43, 44, 45, 46)


REQUIRED_KEYS = PRIMARY = ("T60", "C50", "EDT")
ADVISORY = ("RIR_to_GT_RIR_R@1",)


def load(path, require=PRIMARY):
    with open(path) as f:
        m = json.load(f)["metrics"]
    for k in require:  # fail-closed: explicit raise (survives -O), finite values only
        if k not in m:
            raise RuntimeError(f"{path}: missing metric {k!r}")
        if not (isinstance(m[k], (int, float)) and math.isfinite(m[k])):
            raise RuntimeError(f"{path}: non-finite {k!r} = {m[k]!r}")
    return m


def one(pattern):
    m = glob.glob(pattern)
    if len(m) != 1:  # explicit raise, not assert (python -O safe)
        raise RuntimeError(f"expected exactly 1 file for {pattern}, got {m}")
    return load(m[0])


def stats(vals):
    n = len(vals)
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return mu, sd


print("=" * 96)
print("exp_07 B-V GATE — B-V@67,500 (EMA) vs released Table-1; tiered sigma_c bands")
print("=" * 96)

verdicts = {}
for K in (8, 1):
    print(f"\n--- K={K} (5 eval-seeds, full unseen split, bf16) ---")
    per_seed = {k: [] for k in PRIMARY + ADVISORY}
    for s in SEEDS:
        m = one(os.path.join(BV, f"*step=67500_metrics_1_1.0_exp07_BV_gate_K{K}_seed{s}.json"))
        for k in per_seed:
            per_seed[k].append(m[k])
    for k in PRIMARY + ADVISORY:
        mu, sd = stats(per_seed[k])
        rmu, rsd = REL[K][k]
        d = mu - rmu
        sc = math.sqrt(sd ** 2 + rsd ** 2)
        n = d / sc if sc > 0 else float("inf")
        better_low = k in PRIMARY  # all three primary are lower-better; R@1 higher-better
        if d == 0:
            direction = "TIE"
        elif better_low:
            direction = "SUPERIOR" if d < 0 else "worse"
        else:
            direction = "SUPERIOR" if d > 0 else "worse"
        tier = ("PASS(equiv<=1sc)" if abs(n) <= 1 else
                "PASS(band<=2sc)" if abs(n) <= 2 else "OUTSIDE(>2sc)")
        tag = "advisory" if k in ADVISORY else "GATE"
        verdicts[(K, k)] = (tier, direction, d, n)
        print(f"  [{tag:8s}] {k:<20s} BV {mu:9.4f} ±{sd:.4f} | rel {rmu:9.4f} ±{rsd:.4f} | "
              f"d={d:+8.4f} d/sc={n:+7.1f} 2sc=±{2*sc:.4f} -> {tier} ({direction})")

gate_cells = [(K, k) for K in (8, 1) for k in PRIMARY]
n_pass = sum(1 for c in gate_cells if verdicts[c][0].startswith("PASS"))
n_sup = sum(1 for c in gate_cells if verdicts[c][0] == "OUTSIDE(>2sc)" and verdicts[c][1] == "SUPERIOR")
print(f"\nGATE SUMMARY: {n_pass}/6 primary cells within 2sigma_c "
      f"(+{n_sup} outside-band-but-SUPERIOR); strict pre-registered PASS requires 6/6.")
print("Decision on any non-6/6 profile belongs to Yixun (stop-and-ask, GO step 5).")

# ---- selection curve (EMA, K=8, seed 42): screens + extras + gate seed-42 point ----
# dict keyed by step: rejects conflicting duplicates, warns on missing expected points
print("\n--- T60/EDT selection curve (K=8, eval-seed 42, EMA) ---")
import re as _re

points = {}


def add_point(step, m, src):
    if step in points and points[step] != m:
        raise RuntimeError(f"conflicting duplicate selection-curve point at step {step} ({src})")
    points[step] = m


for path in glob.glob(os.path.join(BV, "*_metrics_1_1.0_exp07_BV_screen_S*_ema.json")):
    mt = _re.search(r"_screen_S(\d+)_ema\.json$", os.path.basename(path))
    if mt:
        add_point(int(mt.group(1)), load(path), path)
for path in glob.glob(os.path.join(BV, "*_metrics_1_1.0_exp07_BV_selcurve_S*.json")):
    mt = _re.search(r"_selcurve_S(\d+)\.json$", os.path.basename(path))
    if mt:
        add_point(int(mt.group(1)), load(path), path)
g42 = glob.glob(os.path.join(BV, "*step=67500_metrics_1_1.0_exp07_BV_gate_K8_seed42.json"))
if g42:
    add_point(67500, load(g42[0]), g42[0])
EXPECTED = {10000, 20000, 27500, 30000, 32500, 40000, 50000, 60000, 62500, 65000, 67500}
missing = EXPECTED - set(points)
if missing:
    print(f"  WARNING: expected curve points missing: {sorted(missing)}")
for step in sorted(points):
    m = points[step]
    print(f"  step {step:>6d}: T60 {m['T60']:7.4f}  C50 {m['C50']:6.4f}  EDT {m['EDT']:7.4f}  "
          f"FD {m.get('FD', float('nan')):6.4f}  R@1 {m.get('RIR_to_GT_RIR_R@1', float('nan')):5.2f}")
if points:
    t60s = [m["T60"] for s, m in points.items() if s >= 20000]
    print(f"  band (steps>=20k): T60 min {min(t60s):.4f} / max {max(t60s):.4f} "
          f"(released 8.609 {'INSIDE' if min(t60s) <= 8.609 <= max(t60s) else 'OUTSIDE'} the band)")

# ---- 291k corroborating row (context-only; exact path per review fix — no wildcard discovery) ----
CORROB = ("/home/yixunhu/codespace/rir2rir-oneroom/test/flac_codec/flac_arbitrary_rx_eval/"
          "outputs_291k_scratch_vanilla/epoch=14-step=67500_metrics_1_1.0_exp07_291k_corrob_K8_s42.json")
if os.path.exists(CORROB):
    m = load(CORROB)
    print(f"\n291k corroborating (THEIR ckpt, OUR protocol, K=8 s42; context-only — data-folder/micro-batch/seed provenance caveats):")
    print(f"  T60 {m['T60']:.4f}  C50 {m['C50']:.4f}  EDT {m['EDT']:.4f}  FD {m.get('FD', float('nan')):.4f}  "
          f"R@1 {m.get('RIR_to_GT_RIR_R@1', float('nan')):.2f}")
else:
    print(f"\n291k corroborating JSON not found at the exact expected path (screen may have failed — check the gate-block log):\n  {CORROB}")

print("\ndone.")
