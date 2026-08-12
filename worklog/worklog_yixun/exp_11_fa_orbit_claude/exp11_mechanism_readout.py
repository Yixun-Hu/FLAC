#!/usr/bin/env python3
"""exp_11 mechanism readout — every derived number in the R2/R3/q9 report,
regenerated from the committed metric JSONs.

Consolidates the ad-hoc delta computations of 2026-08-11/12 into one
reviewable driver (universal-review mandate). Reads ONLY validated cell
outputs; every glob must resolve to exactly one file (fail-closed), and the
q9 blocks require exactly the five eval seeds 42-46. No file is written.

Usage: python exp11_mechanism_readout.py [--repo-root PATH]
"""
import argparse
import glob
import json
import math
import os

KEYS = [("T60", "T60"), ("C50", "C50"), ("EDT", "EDT"),
        ("RIR_to_GT_RIR_R@1", "R@1"), ("RIR_to_GT_RIR_R@5", "R@5"),
        ("RIR_to_GT_RIR_R@10", "R@10")]
K4 = KEYS[:4]
ARMS = ("C4L", "C8", "C16", "C32")
ORBITS = (4, 8, 16, 32)
# 95% two-sided t quantile at df=4 (5 paired eval seeds)
T975_DF4 = 2.776445


def one(root, pat):
    fs = [f for f in glob.glob(os.path.join(root, pat), recursive=True)
          if not f.endswith(".screenmeta.json")]
    if len(fs) != 1:
        raise SystemExit(f"expected exactly one match for {pat}, got {fs}")
    with open(fs[0]) as fh:
        d = json.load(fh)
    return d.get("metrics", d)


def seeds(root, pat):
    out = {}
    for f in glob.glob(os.path.join(root, pat), recursive=True):
        if f.endswith(".screenmeta.json"):
            continue
        s = int(f.split("_s")[-1].split("_")[0].split(".")[0])
        if s in out:
            raise SystemExit(f"seed {s} appears twice for {pat}")
        with open(f) as fh:
            d = json.load(fh)
        out[s] = d.get("metrics", d)
    if sorted(out) != [42, 43, 44, 45, 46]:
        raise SystemExit(f"need seeds 42-46 for {pat}, got {sorted(out)}")
    return out


def paired(a, b, key):  # mean and 95% CI of a-b across the shared seeds
    ds = [a[s][key] - b[s][key] for s in sorted(a)]
    m = sum(ds) / len(ds)
    sd = math.sqrt(sum((d - m) ** 2 for d in ds) / (len(ds) - 1))
    return m, T975_DF4 * sd / math.sqrt(len(ds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "..", ".."))
    root = os.path.abspath(ap.parse_args().repo_root)

    own = {a: one(root, f"outputs_FLAC/exp11_{a}/**/*_r3_rot0_s42_K8_*.json")
           for a in ARMS}

    def cross(arm, a):
        return one(root, f"outputs_FLAC/exp11_{arm}/**/*_cross_a{a}_S40000_s42_K8_*.json")

    print("=== R3 yaw-flatness (K8 s42 @40k, delta vs own rot0) ===")
    for a in ARMS:
        n = 4 if a == "C4L" else int(a[1:])
        for rot in ("5p625", "11p25", "22p5", "45"):
            deg = float(rot.replace("p", "."))
            m = one(root, f"outputs_FLAC/exp11_{a}/**/*_r3_rot{rot}_s42_K8_*.json")
            grp = "IN-GROUP " if abs(deg * n / 360.0 - round(deg * n / 360.0)) < 1e-9 else "off-group"
            print(f"{a:4s} rot{deg:<7} {grp}: " +
                  "  ".join(f"{l} {m[k]-own[a][k]:+.3f}" for k, l in K4))

    print("\n=== R2 decomposition: Cn/an - C4L/a4 = eval + train + interaction (K8 s42 @40k) ===")
    for n in (8, 16, 32):
        for k, l in K4:
            tot = own[f"C{n}"][k] - own["C4L"][k]
            ev = cross("C4L", n)[k] - own["C4L"][k]
            tr = cross(f"C{n}", 4)[k] - own["C4L"][k]
            print(f"n={n:2d} {l:4s}: total {tot:+.3f} = eval {ev:+.3f} "
                  f"+ train {tr:+.3f} + ix {tot-ev-tr:+.3f}")

    print("\n=== full cross matrix (T60 delta vs own protocol) ===")
    for a in ARMS:
        na = 4 if a == "C4L" else int(a[1:])
        row = ["own" if o == na else f"a{o} {cross(a, o)['T60']-own[a]['T60']:+.3f}"
               for o in ORBITS]
        print(f"{a:4s}  " + "   ".join(row))

    print("\n=== legacy C4 (exp_07 B-F @40k) cross deltas vs batched-era own-a4 (cell 3684155) ===")
    leg = one(root, "outputs_FLAC/exp07_BF/**/*C4backfill_S40000_s42_K8_fa_invariant_a4.json")
    for n in (8, 16, 32):
        m = one(root, f"outputs_FLAC/exp07_BF/**/*C4backfill_cross_a{n}_S40000_s42_K8_*.json")
        print(f"  a{n}: " + "  ".join(f"{l} {m[k]-leg[k]:+.3f}" for k, l in K4))

    print("\n=== q9: fa(C4L) - vanilla(VANL), 5-seed paired, mean +- 95% CI (df=4) ===")
    for k_ctx in (8, 1):
        v = seeds(root, f"outputs_FLAC/exp11_VANL/**/*q9_S40000_s4[2-6]_K{k_ctx}.json")
        c = seeds(root, f"outputs_FLAC/exp11_C4L/**/*q9_S40000_s4[2-6]_K{k_ctx}_*.json")
        print(f"K{k_ctx}: " + "  ".join(
            "{} {:+.3f}±{:.3f}".format(l, *paired(c, v, k)) for k, l in KEYS))


if __name__ == "__main__":
    main()
