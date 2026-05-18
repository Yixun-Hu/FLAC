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
# (json_key, short display header) — short headers so the table previews cleanly.
METRICS = [
    ("T60", "T60&darr; (%)"),
    ("C50", "C50&darr; (dB)"),
    ("EDT", "EDT&darr; (ms)"),
    ("FD", "FD&darr;"),
    ("RIR_to_GT_RIR_R@1", "GT R@1&uarr;"),
    ("RIR_to_GT_RIR_R@5", "GT R@5&uarr;"),
    ("RIR_to_geom_R@1", "geom R@1&uarr;"),
    ("RIR_to_geom_R@5", "geom R@5&uarr;"),
    ("RIR_to_geom_R@10", "geom R@10&uarr;"),
]
KEYS = [k for k, _ in METRICS]


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
    return "n/a" if v is None else f"{v:.4f}"


HEADER = "| Model | K | " + " | ".join(d for _, d in METRICS) + " |"
# GFM separator: one cell per column (Model left-aligned, the rest right).
SEP = "| :--- | ---: | " + " | ".join(["---:"] * len(METRICS)) + " |"


def emit(lines, split, kind):
    lines.append(HEADER)
    lines.append(SEP)
    for label, cdir, prefix in ROWS:
        for K in (8, 1):
            m, err = cell(cdir, prefix, split, K)
            if err:
                cells = [err] + [""] * (len(METRICS) - 1)
            else:
                src = (lambda k: per_scene_mean(m, k)) if kind == "ps" else m.get
                cells = [fmt(src(k)) for k in KEYS]
            lines.append(f"| {label} | {K} | " + " | ".join(cells) + " |")


def main():
    lines = []
    found = sorted(glob.glob(os.path.join(ABL_DIR, f"{CKPT}_metrics_1_1.0_arbRIR_v0_*.json"))) + \
        sorted(glob.glob(os.path.join(BASE_DIR, f"{CKPT}_metrics_1_1.0_baseline_*.json")))
    lines.append(f"# Arbitrary-RIR eval matrix @ step={STEP}")
    lines.append("")
    lines.append(f"{len(found)}/12 cells present. **Per-scene mean** is the headline "
                 "(CLAUDE.md protocol); all-samples aggregate is reference only. "
                 "T60 / C50 / EDT / FD: lower is better. R@k: higher is better.")

    for split in ("seen", "unseen"):
        lines.append("")
        lines.append(f"## {split} — per-scene mean (headline)")
        lines.append("")
        emit(lines, split, "ps")
        lines.append("")
        lines.append(f"### {split} — all-samples aggregate (reference)")
        lines.append("")
        emit(lines, split, "all")

    out = "\n".join(lines) + "\n"
    print(out)
    dst = "outputs_FLAC/arbRIR_eval_logs/RESULTS.md"
    with open(dst, "w") as f:
        f.write(out)
    print(f"[written] {dst}")


if __name__ == "__main__":
    main()
