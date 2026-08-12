Reading additional input from stdin...
OpenAI Codex v0.146.0
--------
workdir: /n/fs/gatrdp/codespace/FLAC
model: gpt-5.6-sol
provider: openai
approval: on-request
sandbox: danger-full-access
reasoning effort: xhigh
reasoning summaries: none
session id: 019ff662-0235-76b0-853c-df33e7bd1cdb
--------
user
Read-only code review (do NOT install anything or modify environments/files; output = review text only). Repo /n/fs/gatrdp/codespace/FLAC. File under review: worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_mechanism_readout.py — a consolidated readout driver that regenerates every derived number in the exp_11 R2/R3/q9 mechanism report from committed metric JSONs (no writes). It replaces ad-hoc inline snippets per the repo's universal-review mandate. Its printed output has been verified to reproduce the previously reported numbers exactly.

Check: (1) correctness of the paired-t CI (df=4, t=2.776445, sample sd with n-1); (2) the seed-extraction parse ('_s' split) against the actual filename schemas in outputs_FLAC/exp11_VANL (vanilla schema: ..._q9_S40000_s42_K1.json) and exp11_C4L (fa schema: ..._q9_S40000_s42_K8_fa_invariant_a4.json) — any filename in those dirs that would mis-parse or collide; (3) fail-closed properties (exactly-one-file assertion, duplicate-seed, seed-set); (4) the in-group/off-group angle classification arithmetic (deg*n/360 integrality); (5) the R2 identity total=eval+train+ix is by construction — confirm no double-counting confusion in labels; (6) glob patterns cannot accidentally match cross/r3 files into the q9 blocks or vice versa. Verdict: APPROVED or REVISE with findings.
codex
I’ll inspect the driver and enumerate the actual metric filenames/glob matches read-only, then trace each requested invariant against the implementation.
exec
/usr/bin/bash -lc "sed -n '1,260p' worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_mechanism_readout.py
rg --files outputs_FLAC/exp11_VANL outputs_FLAC/exp11_C4L | sort" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
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
/usr/bin/bash: line 2: rg: command not found

codex
The CI formula itself uses the sample standard deviation denominator \(n-1\) and the requested df=4 critical value. `rg` is unavailable here, so I’m falling back to a read-only `find` enumeration for the filename and glob audit.
exec
/usr/bin/bash -lc 'find outputs_FLAC/exp11_VANL outputs_FLAC/exp11_C4L -type f -print | sort' in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=0-step=2500.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=0-step=2500_metrics_1_1.0_exp11_C4L_screen_S2500_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=0-step=2500_metrics_1_1.0_exp11_C4L_screen_S2500_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=0-step=2500_metrics_1_1.0_exp11_C4L_screen_S2500_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=0-step=2500_metrics_1_1.0_exp11_C4L_screen_S2500_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=1-step=5000.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=1-step=5000_metrics_1_1.0_exp11_C4L_screen_S5000_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=1-step=5000_metrics_1_1.0_exp11_C4L_screen_S5000_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=1-step=5000_metrics_1_1.0_exp11_C4L_screen_S5000_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=1-step=5000_metrics_1_1.0_exp11_C4L_screen_S5000_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=1-step=7500.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=1-step=7500_metrics_1_1.0_exp11_C4L_screen_S7500_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=1-step=7500_metrics_1_1.0_exp11_C4L_screen_S7500_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=1-step=7500_metrics_1_1.0_exp11_C4L_screen_S7500_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=1-step=7500_metrics_1_1.0_exp11_C4L_screen_S7500_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=2-step=10000.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=2-step=10000_metrics_1_1.0_exp11_C4L_screen_S10000_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=2-step=10000_metrics_1_1.0_exp11_C4L_screen_S10000_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=2-step=10000_metrics_1_1.0_exp11_C4L_screen_S10000_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=2-step=10000_metrics_1_1.0_exp11_C4L_screen_S10000_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=2-step=12500.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=2-step=12500_metrics_1_1.0_exp11_C4L_screen_S12500_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=2-step=12500_metrics_1_1.0_exp11_C4L_screen_S12500_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=2-step=12500_metrics_1_1.0_exp11_C4L_screen_S12500_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=2-step=12500_metrics_1_1.0_exp11_C4L_screen_S12500_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=3-step=15000.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=3-step=15000_metrics_1_1.0_exp11_C4L_screen_S15000_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=3-step=15000_metrics_1_1.0_exp11_C4L_screen_S15000_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=3-step=15000_metrics_1_1.0_exp11_C4L_screen_S15000_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=3-step=15000_metrics_1_1.0_exp11_C4L_screen_S15000_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=3-step=17500.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=3-step=17500_metrics_1_1.0_exp11_C4L_screen_S17500_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=3-step=17500_metrics_1_1.0_exp11_C4L_screen_S17500_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=3-step=17500_metrics_1_1.0_exp11_C4L_screen_S17500_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=3-step=17500_metrics_1_1.0_exp11_C4L_screen_S17500_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=4-step=20000.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=4-step=20000_metrics_1_1.0_exp11_C4L_screen_S20000_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=4-step=20000_metrics_1_1.0_exp11_C4L_screen_S20000_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=4-step=20000_metrics_1_1.0_exp11_C4L_screen_S20000_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=4-step=20000_metrics_1_1.0_exp11_C4L_screen_S20000_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=4-step=22500.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=4-step=22500_metrics_1_1.0_exp11_C4L_screen_S22500_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=4-step=22500_metrics_1_1.0_exp11_C4L_screen_S22500_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=4-step=22500_metrics_1_1.0_exp11_C4L_screen_S22500_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=4-step=22500_metrics_1_1.0_exp11_C4L_screen_S22500_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=5-step=25000.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=5-step=25000_metrics_1_1.0_exp11_C4L_screen_S25000_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=5-step=25000_metrics_1_1.0_exp11_C4L_screen_S25000_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=5-step=25000_metrics_1_1.0_exp11_C4L_screen_S25000_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=5-step=25000_metrics_1_1.0_exp11_C4L_screen_S25000_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=6-step=27500.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=6-step=27500_metrics_1_1.0_exp11_C4L_screen_S27500_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=6-step=27500_metrics_1_1.0_exp11_C4L_screen_S27500_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=6-step=27500_metrics_1_1.0_exp11_C4L_screen_S27500_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=6-step=27500_metrics_1_1.0_exp11_C4L_screen_S27500_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=6-step=30000.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=6-step=30000_metrics_1_1.0_exp11_C4L_screen_S30000_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=6-step=30000_metrics_1_1.0_exp11_C4L_screen_S30000_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=6-step=30000_metrics_1_1.0_exp11_C4L_screen_S30000_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=6-step=30000_metrics_1_1.0_exp11_C4L_screen_S30000_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=7-step=32500.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=7-step=32500_metrics_1_1.0_exp11_C4L_screen_S32500_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=7-step=32500_metrics_1_1.0_exp11_C4L_screen_S32500_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=7-step=32500_metrics_1_1.0_exp11_C4L_screen_S32500_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=7-step=32500_metrics_1_1.0_exp11_C4L_screen_S32500_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=7-step=35000.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=7-step=35000_metrics_1_1.0_exp11_C4L_screen_S35000_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=7-step=35000_metrics_1_1.0_exp11_C4L_screen_S35000_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=7-step=35000_metrics_1_1.0_exp11_C4L_screen_S35000_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=7-step=35000_metrics_1_1.0_exp11_C4L_screen_S35000_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=37500.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=37500_metrics_1_1.0_exp11_C4L_screen_S37500_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=37500_metrics_1_1.0_exp11_C4L_screen_S37500_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=37500_metrics_1_1.0_exp11_C4L_screen_S37500_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=37500_metrics_1_1.0_exp11_C4L_screen_S37500_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000.ckpt
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s43_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s43_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s43_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s43_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s44_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s44_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s44_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s44_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s45_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s45_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s45_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s45_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s46_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s46_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s46_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s46_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_cross_a16_S40000_s42_K8_fa_invariant_a16.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_cross_a16_S40000_s42_K8_fa_invariant_a16.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_cross_a32_S40000_s42_K8_fa_invariant_a32.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_cross_a32_S40000_s42_K8_fa_invariant_a32.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_cross_a8_S40000_s42_K8_fa_invariant_a8.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_cross_a8_S40000_s42_K8_fa_invariant_a8.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s43_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s43_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s43_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s43_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s44_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s44_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s44_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s44_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s45_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s45_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s45_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s45_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s46_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s46_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s46_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s46_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot0_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot0_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot11p25_s42_K8_fa_invariant_a4_rot11p25.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot11p25_s42_K8_fa_invariant_a4_rot11p25.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot22p5_s42_K8_fa_invariant_a4_rot22p5.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot22p5_s42_K8_fa_invariant_a4_rot22p5.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot45_s42_K8_fa_invariant_a4_rot45.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot45_s42_K8_fa_invariant_a4_rot45.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot5p625_s42_K8_fa_invariant_a4_rot5p625.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot5p625_s42_K8_fa_invariant_a4_rot5p625.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s42_K1_rotrand42_fa_invariant_a4_rotrand42.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s42_K1_rotrand42_fa_invariant_a4_rotrand42.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s42_K1_rotrand42_fa_invariant_a4_rotrand42.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s42_K8_rotrand42_fa_invariant_a4_rotrand42.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s42_K8_rotrand42_fa_invariant_a4_rotrand42.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s42_K8_rotrand42_fa_invariant_a4_rotrand42.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s43_K1_rotrand43_fa_invariant_a4_rotrand43.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s43_K1_rotrand43_fa_invariant_a4_rotrand43.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s43_K1_rotrand43_fa_invariant_a4_rotrand43.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s43_K8_rotrand43_fa_invariant_a4_rotrand43.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s43_K8_rotrand43_fa_invariant_a4_rotrand43.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s43_K8_rotrand43_fa_invariant_a4_rotrand43.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s44_K1_rotrand44_fa_invariant_a4_rotrand44.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s44_K1_rotrand44_fa_invariant_a4_rotrand44.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s44_K1_rotrand44_fa_invariant_a4_rotrand44.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s44_K8_rotrand44_fa_invariant_a4_rotrand44.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s44_K8_rotrand44_fa_invariant_a4_rotrand44.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s44_K8_rotrand44_fa_invariant_a4_rotrand44.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s45_K1_rotrand45_fa_invariant_a4_rotrand45.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s45_K1_rotrand45_fa_invariant_a4_rotrand45.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s45_K1_rotrand45_fa_invariant_a4_rotrand45.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s45_K8_rotrand45_fa_invariant_a4_rotrand45.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s45_K8_rotrand45_fa_invariant_a4_rotrand45.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s45_K8_rotrand45_fa_invariant_a4_rotrand45.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s46_K1_rotrand46_fa_invariant_a4_rotrand46.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s46_K1_rotrand46_fa_invariant_a4_rotrand46.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s46_K1_rotrand46_fa_invariant_a4_rotrand46.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s46_K8_rotrand46_fa_invariant_a4_rotrand46.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s46_K8_rotrand46_fa_invariant_a4_rotrand46.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_rgen_S40000_s46_K8_rotrand46_fa_invariant_a4_rotrand46.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_vctl_S40000_s42_K8_rot45_fa_invariant_a4_rot45.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_vctl_S40000_s42_K8_rot45_fa_invariant_a4_rot45.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_vctl_S40000_s42_K8_rot45_fa_invariant_a4_rot45.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_vctl_S40000_s42_K8_rot90_fa_invariant_a4_rot90.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_vctl_S40000_s42_K8_rot90_fa_invariant_a4_rot90.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_vctl_S40000_s42_K8_rot90_fa_invariant_a4_rot90.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s42_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s42_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s42_K1_fa_invariant_a4.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s42_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s42_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s42_K8_fa_invariant_a4.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s43_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s43_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s43_K1_fa_invariant_a4.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s43_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s43_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s43_K8_fa_invariant_a4.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s44_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s44_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s44_K1_fa_invariant_a4.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s44_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s44_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s44_K8_fa_invariant_a4.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s45_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s45_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s45_K1_fa_invariant_a4.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s45_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s45_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s45_K8_fa_invariant_a4.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s46_K1_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s46_K1_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s46_K1_fa_invariant_a4.stream.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s46_K8_fa_invariant_a4.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s46_K8_fa_invariant_a4.json.screenmeta.json
outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_C4L_zref_S40000_s46_K8_fa_invariant_a4.stream.json
outputs_FLAC/exp11_C4L/fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_manifest.txt
outputs_FLAC/exp11_C4L/fa_orbit_2026-08-06_20-36-00_C4L_8x8_jid3648694_train.log
outputs_FLAC/exp11_C4L/launch_manifest.txt
outputs_FLAC/exp11_C4L/preflight_2026-08-06_20-36-00_jid3648694.out
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=0-step=2500.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=0-step=2500_metrics_1_1.0_exp11_VANL_screen_S2500_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=0-step=2500_metrics_1_1.0_exp11_VANL_screen_S2500_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=0-step=2500_metrics_1_1.0_exp11_VANL_screen_S2500_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=0-step=2500_metrics_1_1.0_exp11_VANL_screen_S2500_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=5000.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=5000_metrics_1_1.0_exp11_VANL_screen_S5000_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=5000_metrics_1_1.0_exp11_VANL_screen_S5000_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=5000_metrics_1_1.0_exp11_VANL_screen_S5000_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=5000_metrics_1_1.0_exp11_VANL_screen_S5000_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=7500.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=7500_metrics_1_1.0_exp11_VANL_screen_S7500_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=7500_metrics_1_1.0_exp11_VANL_screen_S7500_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=7500_metrics_1_1.0_exp11_VANL_screen_S7500_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=1-step=7500_metrics_1_1.0_exp11_VANL_screen_S7500_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=10000.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=10000_metrics_1_1.0_exp11_VANL_screen_S10000_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=10000_metrics_1_1.0_exp11_VANL_screen_S10000_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=10000_metrics_1_1.0_exp11_VANL_screen_S10000_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=10000_metrics_1_1.0_exp11_VANL_screen_S10000_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=12500.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=12500_metrics_1_1.0_exp11_VANL_screen_S12500_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=12500_metrics_1_1.0_exp11_VANL_screen_S12500_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=12500_metrics_1_1.0_exp11_VANL_screen_S12500_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=2-step=12500_metrics_1_1.0_exp11_VANL_screen_S12500_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=15000.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=15000_metrics_1_1.0_exp11_VANL_screen_S15000_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=15000_metrics_1_1.0_exp11_VANL_screen_S15000_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=15000_metrics_1_1.0_exp11_VANL_screen_S15000_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=15000_metrics_1_1.0_exp11_VANL_screen_S15000_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=17500.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=17500_metrics_1_1.0_exp11_VANL_screen_S17500_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=17500_metrics_1_1.0_exp11_VANL_screen_S17500_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=17500_metrics_1_1.0_exp11_VANL_screen_S17500_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=3-step=17500_metrics_1_1.0_exp11_VANL_screen_S17500_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=20000.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=20000_metrics_1_1.0_exp11_VANL_screen_S20000_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=20000_metrics_1_1.0_exp11_VANL_screen_S20000_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=20000_metrics_1_1.0_exp11_VANL_screen_S20000_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=20000_metrics_1_1.0_exp11_VANL_screen_S20000_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=22500.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=22500_metrics_1_1.0_exp11_VANL_screen_S22500_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=22500_metrics_1_1.0_exp11_VANL_screen_S22500_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=22500_metrics_1_1.0_exp11_VANL_screen_S22500_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=4-step=22500_metrics_1_1.0_exp11_VANL_screen_S22500_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=5-step=25000.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=5-step=25000_metrics_1_1.0_exp11_VANL_screen_S25000_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=5-step=25000_metrics_1_1.0_exp11_VANL_screen_S25000_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=5-step=25000_metrics_1_1.0_exp11_VANL_screen_S25000_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=5-step=25000_metrics_1_1.0_exp11_VANL_screen_S25000_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=27500.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=27500_metrics_1_1.0_exp11_VANL_screen_S27500_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=27500_metrics_1_1.0_exp11_VANL_screen_S27500_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=27500_metrics_1_1.0_exp11_VANL_screen_S27500_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=27500_metrics_1_1.0_exp11_VANL_screen_S27500_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=30000.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=30000_metrics_1_1.0_exp11_VANL_screen_S30000_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=30000_metrics_1_1.0_exp11_VANL_screen_S30000_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=30000_metrics_1_1.0_exp11_VANL_screen_S30000_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=6-step=30000_metrics_1_1.0_exp11_VANL_screen_S30000_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=32500.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=32500_metrics_1_1.0_exp11_VANL_screen_S32500_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=32500_metrics_1_1.0_exp11_VANL_screen_S32500_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=32500_metrics_1_1.0_exp11_VANL_screen_S32500_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=32500_metrics_1_1.0_exp11_VANL_screen_S32500_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=35000.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=35000_metrics_1_1.0_exp11_VANL_screen_S35000_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=35000_metrics_1_1.0_exp11_VANL_screen_S35000_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=35000_metrics_1_1.0_exp11_VANL_screen_S35000_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=7-step=35000_metrics_1_1.0_exp11_VANL_screen_S35000_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=37500.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=37500_metrics_1_1.0_exp11_VANL_screen_S37500_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=37500_metrics_1_1.0_exp11_VANL_screen_S37500_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=37500_metrics_1_1.0_exp11_VANL_screen_S37500_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=37500_metrics_1_1.0_exp11_VANL_screen_S37500_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000.ckpt
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s43_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s43_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s43_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s43_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s44_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s44_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s44_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s44_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s45_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s45_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s45_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s45_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s46_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s46_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s46_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s46_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_screen_S40000_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_screen_S40000_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_screen_S40000_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp11_VANL_screen_S40000_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s42_K1_rotrand42_rotrand42.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s42_K1_rotrand42_rotrand42.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s42_K1_rotrand42_rotrand42.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s42_K8_rotrand42_rotrand42.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s42_K8_rotrand42_rotrand42.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s42_K8_rotrand42_rotrand42.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s43_K1_rotrand43_rotrand43.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s43_K1_rotrand43_rotrand43.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s43_K1_rotrand43_rotrand43.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s43_K8_rotrand43_rotrand43.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s43_K8_rotrand43_rotrand43.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s43_K8_rotrand43_rotrand43.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s44_K1_rotrand44_rotrand44.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s44_K1_rotrand44_rotrand44.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s44_K1_rotrand44_rotrand44.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s44_K8_rotrand44_rotrand44.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s44_K8_rotrand44_rotrand44.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s44_K8_rotrand44_rotrand44.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s45_K1_rotrand45_rotrand45.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s45_K1_rotrand45_rotrand45.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s45_K1_rotrand45_rotrand45.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s45_K8_rotrand45_rotrand45.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s45_K8_rotrand45_rotrand45.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s45_K8_rotrand45_rotrand45.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s46_K1_rotrand46_rotrand46.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s46_K1_rotrand46_rotrand46.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s46_K1_rotrand46_rotrand46.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s46_K8_rotrand46_rotrand46.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s46_K8_rotrand46_rotrand46.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_rgen_S40000_s46_K8_rotrand46_rotrand46.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_vctl_S40000_s42_K8_rot90_rot90.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_vctl_S40000_s42_K8_rot90_rot90.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_vctl_S40000_s42_K8_rot90_rot90.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s42_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s42_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s42_K1.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s42_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s42_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s42_K8.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s43_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s43_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s43_K1.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s43_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s43_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s43_K8.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s44_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s44_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s44_K1.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s44_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s44_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s44_K8.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s45_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s45_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s45_K1.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s45_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s45_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s45_K8.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s46_K1.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s46_K1.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s46_K1.stream.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s46_K8.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s46_K8.json.screenmeta.json
outputs_FLAC/exp11_VANL/FLAC_exp11_VANL/exp11_VANL/checkpoints/epoch=8-step=40000_metrics_1_1.0_exp14_VANL_zref_S40000_s46_K8.stream.json
outputs_FLAC/exp11_VANL/fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_manifest.txt
outputs_FLAC/exp11_VANL/fa_orbit_2026-08-09_12-06-39_VANL_8x8_jid3661520_train.log
outputs_FLAC/exp11_VANL/launch_manifest.txt

exec
/usr/bin/bash -lc "python -c '
import glob, os
root=os.path.abspath(\".\")
def ms(p): return [f for f in glob.glob(os.path.join(root,p),recursive=True) if not f.endswith(\".screenmeta.json\")]
patterns=[]
for arm in (\"C4L\",\"C8\",\"C16\",\"C32\"):
 patterns.append((f\"{arm} own\",f\"outputs_FLAC/exp11_{arm}/**/*_r3_rot0_s42_K8_*.json\"))
 for rot in (\"5p625\",\"11p25\",\"22p5\",\"45\"):
  patterns.append((f\"{arm} r3 {rot}\",f\"outputs_FLAC/exp11_{arm}/**/*_r3_rot{rot}_s42_K8_*.json\"))
for arm,n in ((\"C4L\",4),(\"C8\",8),(\"C16\",16),(\"C32\",32)):
 for a in (4,8,16,32):
  if a"'!=n: patterns.append((f"{arm} cross a{a}",f"outputs_FLAC/exp11_{arm}/**/*_cross_a{a}_S40000_s42_K8_*.json"))
patterns.append(("legacy own","outputs_FLAC/exp07_BF/**/*C4backfill_S40000_s42_K8_fa_invariant_a4.json"))
for n in (8,16,32): patterns.append((f"legacy cross a{n}",f"outputs_FLAC/exp07_BF/**/*C4backfill_cross_a{n}_S40000_s42_K8_*.json"))
for label,p in patterns:
 x=ms(p)
 print(label, len(x), *(os.path.basename(f) for f in x))
print("Q9")
qsets={}
for arm,tail in (("VANL","K{K}.json"),("C4L","K{K}_*.json")):
 for K in (1,8):
  p=f"outputs_FLAC/exp11_{arm}/**/*q9_S40000_s4[2-6]_"+tail.format(K=K)
  x=ms(p); qsets[(arm,K)]=set(x)
  parsed=[(os.path.basename(f),int(f.split("_s")[-1].split("_")[0].split(".")[0]),os.path.basename(f).count("_s")) for f in x]
  print(arm,K,len(x),parsed)
for arm in ("VANL","C4L"):
 allq=set(f for f in glob.glob(os.path.join(root,f"outputs_FLAC/exp11_{arm}/**/*q9*.json"),recursive=True) if not f.endswith(".screenmeta.json"))
 covered=qsets[(arm,1)]|qsets[(arm,8)]
 print(arm,"all_q9",len(allq),"uncovered",[os.path.basename(f) for f in sorted(allq-covered)])
 cats={name:set(f for f in glob.glob(os.path.join(root,f"outputs_FLAC/exp11_{arm}/**/*{token}*.json"),recursive=True) if not f.endswith(".screenmeta.json")) for name,token in (("q9","_q9_"),("cross","_cross_"),("r3","_r3_"))}
 print(arm,"category_intersections",{a+"/"+b:len(cats[a]&cats[b]) for a,b in (("q9","cross"),("q9","r3"),("cross","r3"))})
'"' " in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
C4L own 1 epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot0_s42_K8_fa_invariant_a4.json
C4L r3 5p625 1 epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot5p625_s42_K8_fa_invariant_a4_rot5p625.json
C4L r3 11p25 1 epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot11p25_s42_K8_fa_invariant_a4_rot11p25.json
C4L r3 22p5 1 epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot22p5_s42_K8_fa_invariant_a4_rot22p5.json
C4L r3 45 1 epoch=8-step=40000_metrics_1_1.0_exp11_C4L_r3_rot45_s42_K8_fa_invariant_a4_rot45.json
C8 own 1 epoch=8-step=40000_metrics_1_1.0_exp11_C8_r3_rot0_s42_K8_fa_invariant_a8.json
C8 r3 5p625 1 epoch=8-step=40000_metrics_1_1.0_exp11_C8_r3_rot5p625_s42_K8_fa_invariant_a8_rot5p625.json
C8 r3 11p25 1 epoch=8-step=40000_metrics_1_1.0_exp11_C8_r3_rot11p25_s42_K8_fa_invariant_a8_rot11p25.json
C8 r3 22p5 1 epoch=8-step=40000_metrics_1_1.0_exp11_C8_r3_rot22p5_s42_K8_fa_invariant_a8_rot22p5.json
C8 r3 45 1 epoch=8-step=40000_metrics_1_1.0_exp11_C8_r3_rot45_s42_K8_fa_invariant_a8_rot45.json
C16 own 1 epoch=8-step=40000_metrics_1_1.0_exp11_C16_r3_rot0_s42_K8_fa_invariant_a16.json
C16 r3 5p625 1 epoch=8-step=40000_metrics_1_1.0_exp11_C16_r3_rot5p625_s42_K8_fa_invariant_a16_rot5p625.json
C16 r3 11p25 1 epoch=8-step=40000_metrics_1_1.0_exp11_C16_r3_rot11p25_s42_K8_fa_invariant_a16_rot11p25.json
C16 r3 22p5 1 epoch=8-step=40000_metrics_1_1.0_exp11_C16_r3_rot22p5_s42_K8_fa_invariant_a16_rot22p5.json
C16 r3 45 1 epoch=8-step=40000_metrics_1_1.0_exp11_C16_r3_rot45_s42_K8_fa_invariant_a16_rot45.json
C32 own 1 epoch=8-step=40000_metrics_1_1.0_exp11_C32_r3_rot0_s42_K8_fa_invariant_a32.json
C32 r3 5p625 1 epoch=8-step=40000_metrics_1_1.0_exp11_C32_r3_rot5p625_s42_K8_fa_invariant_a32_rot5p625.json
C32 r3 11p25 1 epoch=8-step=40000_metrics_1_1.0_exp11_C32_r3_rot11p25_s42_K8_fa_invariant_a32_rot11p25.json
C32 r3 22p5 1 epoch=8-step=40000_metrics_1_1.0_exp11_C32_r3_rot22p5_s42_K8_fa_invariant_a32_rot22p5.json
C32 r3 45 1 epoch=8-step=40000_metrics_1_1.0_exp11_C32_r3_rot45_s42_K8_fa_invariant_a32_rot45.json
C4L cross a8 1 epoch=8-step=40000_metrics_1_1.0_exp11_C4L_cross_a8_S40000_s42_K8_fa_invariant_a8.json
C4L cross a16 1 epoch=8-step=40000_metrics_1_1.0_exp11_C4L_cross_a16_S40000_s42_K8_fa_invariant_a16.json
C4L cross a32 1 epoch=8-step=40000_metrics_1_1.0_exp11_C4L_cross_a32_S40000_s42_K8_fa_invariant_a32.json
C8 cross a4 1 epoch=8-step=40000_metrics_1_1.0_exp11_C8_cross_a4_S40000_s42_K8_fa_invariant_a4.json
C8 cross a16 1 epoch=8-step=40000_metrics_1_1.0_exp11_C8_cross_a16_S40000_s42_K8_fa_invariant_a16.json
C8 cross a32 1 epoch=8-step=40000_metrics_1_1.0_exp11_C8_cross_a32_S40000_s42_K8_fa_invariant_a32.json
C16 cross a4 1 epoch=8-step=40000_metrics_1_1.0_exp11_C16_cross_a4_S40000_s42_K8_fa_invariant_a4.json
C16 cross a8 1 epoch=8-step=40000_metrics_1_1.0_exp11_C16_cross_a8_S40000_s42_K8_fa_invariant_a8.json
C16 cross a32 1 epoch=8-step=40000_metrics_1_1.0_exp11_C16_cross_a32_S40000_s42_K8_fa_invariant_a32.json
C32 cross a4 1 epoch=8-step=40000_metrics_1_1.0_exp11_C32_cross_a4_S40000_s42_K8_fa_invariant_a4.json
C32 cross a8 1 epoch=8-step=40000_metrics_1_1.0_exp11_C32_cross_a8_S40000_s42_K8_fa_invariant_a8.json
C32 cross a16 1 epoch=8-step=40000_metrics_1_1.0_exp11_C32_cross_a16_S40000_s42_K8_fa_invariant_a16.json
legacy own 1 epoch=8-step=40000_metrics_1_1.0_exp11_C4backfill_S40000_s42_K8_fa_invariant_a4.json
legacy cross a8 1 epoch=8-step=40000_metrics_1_1.0_exp11_C4backfill_cross_a8_S40000_s42_K8_fa_invariant_a8.json
legacy cross a16 1 epoch=8-step=40000_metrics_1_1.0_exp11_C4backfill_cross_a16_S40000_s42_K8_fa_invariant_a16.json
legacy cross a32 1 epoch=8-step=40000_metrics_1_1.0_exp11_C4backfill_cross_a32_S40000_s42_K8_fa_invariant_a32.json
Q9
VANL 1 5 [('epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s42_K1.json', 42, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s45_K1.json', 45, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s44_K1.json', 44, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s46_K1.json', 46, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s43_K1.json', 43, 1)]
VANL 8 5 [('epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s46_K8.json', 46, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s44_K8.json', 44, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s43_K8.json', 43, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s42_K8.json', 42, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_VANL_q9_S40000_s45_K8.json', 45, 1)]
C4L 1 5 [('epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s43_K1_fa_invariant_a4.json', 43, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s46_K1_fa_invariant_a4.json', 46, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s44_K1_fa_invariant_a4.json', 44, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s45_K1_fa_invariant_a4.json', 45, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s42_K1_fa_invariant_a4.json', 42, 1)]
C4L 8 5 [('epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s45_K8_fa_invariant_a4.json', 45, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s42_K8_fa_invariant_a4.json', 42, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s43_K8_fa_invariant_a4.json', 43, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s46_K8_fa_invariant_a4.json', 46, 1), ('epoch=8-step=40000_metrics_1_1.0_exp11_C4L_q9_S40000_s44_K8_fa_invariant_a4.json', 44, 1)]
VANL all_q9 10 uncovered []
VANL category_intersections {'q9/cross': 0, 'q9/r3': 0, 'cross/r3': 0}
C4L all_q9 10 uncovered []
C4L category_intersections {'q9/cross': 0, 'q9/r3': 0, 'cross/r3': 0}

2026-08-12T14:32:05.499692Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
exec
/usr/bin/bash -lc "nl -ba worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_mechanism_readout.py | sed -n '1,220p'
git status --short -- worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_mechanism_readout.py
git diff -- worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_mechanism_readout.py" in /n/fs/gatrdp/codespace/FLAC
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""exp_11 mechanism readout — every derived number in the R2/R3/q9 report,
     3	regenerated from the committed metric JSONs.
     4	
     5	Consolidates the ad-hoc delta computations of 2026-08-11/12 into one
     6	reviewable driver (universal-review mandate). Reads ONLY validated cell
     7	outputs; every glob must resolve to exactly one file (fail-closed), and the
     8	q9 blocks require exactly the five eval seeds 42-46. No file is written.
     9	
    10	Usage: python exp11_mechanism_readout.py [--repo-root PATH]
    11	"""
    12	import argparse
    13	import glob
    14	import json
    15	import math
    16	import os
    17	
    18	KEYS = [("T60", "T60"), ("C50", "C50"), ("EDT", "EDT"),
    19	        ("RIR_to_GT_RIR_R@1", "R@1"), ("RIR_to_GT_RIR_R@5", "R@5"),
    20	        ("RIR_to_GT_RIR_R@10", "R@10")]
    21	K4 = KEYS[:4]
    22	ARMS = ("C4L", "C8", "C16", "C32")
    23	ORBITS = (4, 8, 16, 32)
    24	# 95% two-sided t quantile at df=4 (5 paired eval seeds)
    25	T975_DF4 = 2.776445
    26	
    27	
    28	def one(root, pat):
    29	    fs = [f for f in glob.glob(os.path.join(root, pat), recursive=True)
    30	          if not f.endswith(".screenmeta.json")]
    31	    if len(fs) != 1:
    32	        raise SystemExit(f"expected exactly one match for {pat}, got {fs}")
    33	    with open(fs[0]) as fh:
    34	        d = json.load(fh)
    35	    return d.get("metrics", d)
    36	
    37	
    38	def seeds(root, pat):
    39	    out = {}
    40	    for f in glob.glob(os.path.join(root, pat), recursive=True):
    41	        if f.endswith(".screenmeta.json"):
    42	            continue
    43	        s = int(f.split("_s")[-1].split("_")[0].split(".")[0])
    44	        if s in out:
    45	            raise SystemExit(f"seed {s} appears twice for {pat}")
    46	        with open(f) as fh:
    47	            d = json.load(fh)
    48	        out[s] = d.get("metrics", d)
    49	    if sorted(out) != [42, 43, 44, 45, 46]:
    50	        raise SystemExit(f"need seeds 42-46 for {pat}, got {sorted(out)}")
    51	    return out
    52	
    53	
    54	def paired(a, b, key):  # mean and 95% CI of a-b across the shared seeds
    55	    ds = [a[s][key] - b[s][key] for s in sorted(a)]
    56	    m = sum(ds) / len(ds)
    57	    sd = math.sqrt(sum((d - m) ** 2 for d in ds) / (len(ds) - 1))
    58	    return m, T975_DF4 * sd / math.sqrt(len(ds))
    59	
    60	
    61	def main():
    62	    ap = argparse.ArgumentParser()
    63	    ap.add_argument("--repo-root", default=os.path.join(os.path.dirname(
    64	        os.path.abspath(__file__)), "..", "..", ".."))
    65	    root = os.path.abspath(ap.parse_args().repo_root)
    66	
    67	    own = {a: one(root, f"outputs_FLAC/exp11_{a}/**/*_r3_rot0_s42_K8_*.json")
    68	           for a in ARMS}
    69	
    70	    def cross(arm, a):
    71	        return one(root, f"outputs_FLAC/exp11_{arm}/**/*_cross_a{a}_S40000_s42_K8_*.json")
    72	
    73	    print("=== R3 yaw-flatness (K8 s42 @40k, delta vs own rot0) ===")
    74	    for a in ARMS:
    75	        n = 4 if a == "C4L" else int(a[1:])
    76	        for rot in ("5p625", "11p25", "22p5", "45"):
    77	            deg = float(rot.replace("p", "."))
    78	            m = one(root, f"outputs_FLAC/exp11_{a}/**/*_r3_rot{rot}_s42_K8_*.json")
    79	            grp = "IN-GROUP " if abs(deg * n / 360.0 - round(deg * n / 360.0)) < 1e-9 else "off-group"
    80	            print(f"{a:4s} rot{deg:<7} {grp}: " +
    81	                  "  ".join(f"{l} {m[k]-own[a][k]:+.3f}" for k, l in K4))
    82	
    83	    print("\n=== R2 decomposition: Cn/an - C4L/a4 = eval + train + interaction (K8 s42 @40k) ===")
    84	    for n in (8, 16, 32):
    85	        for k, l in K4:
    86	            tot = own[f"C{n}"][k] - own["C4L"][k]
    87	            ev = cross("C4L", n)[k] - own["C4L"][k]
    88	            tr = cross(f"C{n}", 4)[k] - own["C4L"][k]
    89	            print(f"n={n:2d} {l:4s}: total {tot:+.3f} = eval {ev:+.3f} "
    90	                  f"+ train {tr:+.3f} + ix {tot-ev-tr:+.3f}")
    91	
    92	    print("\n=== full cross matrix (T60 delta vs own protocol) ===")
    93	    for a in ARMS:
    94	        na = 4 if a == "C4L" else int(a[1:])
    95	        row = ["own" if o == na else f"a{o} {cross(a, o)['T60']-own[a]['T60']:+.3f}"
    96	               for o in ORBITS]
    97	        print(f"{a:4s}  " + "   ".join(row))
    98	
    99	    print("\n=== legacy C4 (exp_07 B-F @40k) cross deltas vs batched-era own-a4 (cell 3684155) ===")
   100	    leg = one(root, "outputs_FLAC/exp07_BF/**/*C4backfill_S40000_s42_K8_fa_invariant_a4.json")
   101	    for n in (8, 16, 32):
   102	        m = one(root, f"outputs_FLAC/exp07_BF/**/*C4backfill_cross_a{n}_S40000_s42_K8_*.json")
   103	        print(f"  a{n}: " + "  ".join(f"{l} {m[k]-leg[k]:+.3f}" for k, l in K4))
   104	
   105	    print("\n=== q9: fa(C4L) - vanilla(VANL), 5-seed paired, mean +- 95% CI (df=4) ===")
   106	    for k_ctx in (8, 1):
   107	        v = seeds(root, f"outputs_FLAC/exp11_VANL/**/*q9_S40000_s4[2-6]_K{k_ctx}.json")
   108	        c = seeds(root, f"outputs_FLAC/exp11_C4L/**/*q9_S40000_s4[2-6]_K{k_ctx}_*.json")
   109	        print(f"K{k_ctx}: " + "  ".join(
   110	            "{} {:+.3f}±{:.3f}".format(l, *paired(c, v, k)) for k, l in KEYS))
   111	
   112	
   113	if __name__ == "__main__":
   114	    main()
?? worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_mechanism_readout.py

2026-08-12T14:32:13.612432Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `base_instructions` at line 94 column 5
codex
APPROVED

No correctness findings in [exp11_mechanism_readout.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_mechanism_readout.py).

- Paired CI is correct: five paired differences, sample SD with denominator \(n-1=4\), and margin \(2.776445\,s/\sqrt{5}\).
- All matched q9 filenames contain exactly one `_s<seed>` segment and parse uniquely to seeds 42–46. No actual filename mis-parses or collides.
- `one()` enforces exactly one non-screenmeta match. q9 loading rejects duplicate seeds and any seed set other than exactly 42–46.
- `deg*n/360` correctly classifies the tested rotations: none for C4; 45° for C8; 22.5°/45° for C16; 11.25°/22.5°/45° for C32. These angles are exactly representable here.
- R2 labels are consistent: with \(B=M_{4,4}\), eval \(=M_{4,n}-B\), train \(=M_{n,4}-B\), and interaction \(=M_{n,n}-M_{4,n}-M_{n,4}+B\). Their sum is total \(M_{n,n}-B\), without double-counting.
- Actual R3, cross, legacy, and q9 patterns resolve to their intended files. The q9/cross/R3 file sets have zero intersections; screenmeta companions are excluded.
tokens used
40,199
APPROVED

No correctness findings in [exp11_mechanism_readout.py](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_mechanism_readout.py).

- Paired CI is correct: five paired differences, sample SD with denominator \(n-1=4\), and margin \(2.776445\,s/\sqrt{5}\).
- All matched q9 filenames contain exactly one `_s<seed>` segment and parse uniquely to seeds 42–46. No actual filename mis-parses or collides.
- `one()` enforces exactly one non-screenmeta match. q9 loading rejects duplicate seeds and any seed set other than exactly 42–46.
- `deg*n/360` correctly classifies the tested rotations: none for C4; 45° for C8; 22.5°/45° for C16; 11.25°/22.5°/45° for C32. These angles are exactly representable here.
- R2 labels are consistent: with \(B=M_{4,4}\), eval \(=M_{4,n}-B\), train \(=M_{n,4}-B\), and interaction \(=M_{n,n}-M_{4,n}-M_{n,4}+B\). Their sum is total \(M_{n,n}-B\), without double-counting.
- Actual R3, cross, legacy, and q9 patterns resolve to their intended files. The q9/cross/R3 file sets have zero intersections; screenmeta companions are excluded.
