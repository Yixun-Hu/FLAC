"""
Tabulate the 12-cell arbitrary-RIR eval matrix
(plan/eval_arbRIR_v0_vs_baseline_K1_K8.md, task #31).

Headline = per-scene mean (CLAUDE.md: paper numbers average per-scene
results). Requires eval_FLAC.py run with per_scene=True so each metrics JSON
carries `metrics['by_scene']`. The all-samples aggregate is also reported for
transparency.

Rows: Ablation V3 | Baseline-A (s_i-r_q) | Baseline-B (s_i-r_i)
Cols: K=8 | K=1   ; one table block per split (seen, unseen).
"""
import glob
import json
import os

STEP = 145000
CKPT = f"epoch=15-step={STEP}"
ABL_DIR = "outputs_FLAC/FLAC_arbRIR_v0/FLAC_arbRIR_v0_training/checkpoints"
BASE_DIR = "outputs_FLAC/FLAC_AR_baseline_short/FLAC_AR_baseline_short_training/checkpoints"

# (row label, ckpt dir, eval-name prefix)
ROWS = [
    ("Ablation V3",            ABL_DIR,  "arbRIR_v0"),
    ("Baseline-A (s_i-r_q)",   BASE_DIR, "baseline_A"),
    ("Baseline-B (s_i-r_i)",   BASE_DIR, "baseline_B"),
]
METRICS = ["T60", "C50", "EDT", "FD",
           "RIR_to_GT_RIR_R@1", "RIR_to_GT_RIR_R@5",
           "RIR_to_geom_R@1", "RIR_to_geom_R@5"]


def jpath(ckpt_dir, prefix, split, K):
    return os.path.join(ckpt_dir, f"{CKPT}_metrics_1_1.0_{prefix}_{split}_K{K}.json")


def per_scene_mean(metrics, key):
    bs = metrics.get("by_scene")
    if not bs:
        return None
    vals = [s[key] for s in bs.values() if key in s and s[key] is not None]
    return sum(vals) / len(vals) if vals else None


def cell(ckpt_dir, prefix, split, K):
    p = jpath(ckpt_dir, prefix, split, K)
    if not os.path.exists(p):
        return None, f"MISSING ({os.path.basename(p)})"
    m = json.load(open(p))["metrics"]
    return m, None


def fmt(v):
    return "  n/a" if v is None else f"{v:8.4f}"


def main():
    lines = []
    found = sorted(glob.glob(os.path.join(ABL_DIR, f"{CKPT}_metrics_1_1.0_arbRIR_v0_*.json"))) + \
        sorted(glob.glob(os.path.join(BASE_DIR, f"{CKPT}_metrics_1_1.0_baseline_*.json")))
    lines.append(f"# arbitrary-RIR eval matrix @ step={STEP}  ({len(found)}/12 cells present)\n")

    for split in ("seen", "unseen"):
        lines.append(f"\n## Split: {split}  — per-scene mean (headline)\n")
        header = f"| {'Model':22} | {'K':>2} | " + " | ".join(f"{x:>10}" for x in METRICS) + " |"
        lines.append(header)
        lines.append("|" + "-" * (len(header) - 2) + "|")
        for label, cdir, prefix in ROWS:
            for K in (8, 1):
                m, err = cell(cdir, prefix, split, K)
                if err:
                    lines.append(f"| {label:22} | {K:>2} | {err}")
                    continue
                psm = {k: per_scene_mean(m, k) for k in METRICS}
                row = f"| {label:22} | {K:>2} | " + " | ".join(fmt(psm[k]) for k in METRICS) + " |"
                lines.append(row)

        lines.append(f"\n### Split: {split}  — all-samples aggregate (reference)\n")
        lines.append(header)
        lines.append("|" + "-" * (len(header) - 2) + "|")
        for label, cdir, prefix in ROWS:
            for K in (8, 1):
                m, err = cell(cdir, prefix, split, K)
                if err:
                    lines.append(f"| {label:22} | {K:>2} | {err}")
                    continue
                row = f"| {label:22} | {K:>2} | " + " | ".join(fmt(m.get(k)) for k in METRICS) + " |"
                lines.append(row)

    out = "\n".join(lines)
    print(out)
    dst = "outputs_FLAC/arbRIR_eval_logs/RESULTS.md"
    with open(dst, "w") as f:
        f.write(out + "\n")
    print(f"\n[written] {dst}")


if __name__ == "__main__":
    main()
