#!/usr/bin/env python3
"""gen_model_comparison.py - regenerates worklog/worklog_yixun/model_comparison.md.

Single source of truth for the cross-experiment model table (Yixun directive
2026-08-03: log the table to model_comparison.md and commit+push on every model
-results update). Rows are registered below as (label, K, [glob patterns]); every
cell is aggregated fresh from the raw per-seed metric JSONs on disk — numbers
never live in this script. Rows whose JSON count is below MIN_SEEDS are rendered
as pending. To add a model: append a row spec, rerun, commit.
"""
import json, glob, os, statistics as st, datetime

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "worklog", "worklog_yixun", "model_comparison.md")
KEYS = [("T60", "T60"), ("C50", "C50"), ("EDT", "EDT"),
        ("R@1", "RIR_to_GT_RIR_R@1"), ("R@5", "RIR_to_GT_RIR_R@5"), ("R@10", "RIR_to_GT_RIR_R@10")]
MIN_SEEDS = 5

# (model label, eval-protocol note, K, [glob patterns relative to repo root])
ROWS = [
 ("released FLAC_EMA (exp_01 repro)", "vanilla eval", 1, ["weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K1_seed4[2-6].json"]),
 ("released FLAC_EMA (exp_01 repro)", "vanilla eval", 8, ["weights/FLAC/FLAC_EMA_metrics_1_1.0_exp01_unseen_K8_seed4[2-6].json"]),
 ("B-V 8x8 vanilla @40k", "vanilla eval", 1, ["outputs_FLAC/exp07_BV/*exp07_BV40_K1_s4[2-6]*.json"]),
 ("B-V 8x8 vanilla @40k", "vanilla eval", 8, ["outputs_FLAC/exp07_BV/*exp07_BV40_K8_s4[3-6]*.json",
                                              "outputs_FLAC/exp07_BV/*step=40000_metrics_*exp07_BV_screen_S40000_ema.json"]),
 ("B-V 8x8 vanilla @67.5k (endpoint)", "vanilla eval", 1, ["outputs_FLAC/exp07_BV/*step=67500_metrics_*_exp07_BV_gate_K1_seed4[2-6].json"]),
 ("B-V 8x8 vanilla @67.5k (endpoint)", "vanilla eval", 8, ["outputs_FLAC/exp07_BV/*step=67500_metrics_*_exp07_BV_gate_K8_seed4[2-6].json"]),
 ("B-V 8x8 @92.5k (extend best)", "vanilla eval", 1, ["outputs_FLAC/exp07_BVextend/**/*exp07_BVext92500_K1_s4[2-6]*.json"]),
 ("B-V 8x8 @92.5k (extend best)", "vanilla eval", 8, ["outputs_FLAC/exp07_BVextend/**/*exp07_BVext_S92500_K8_seed4[3-6].json",
                                                      "outputs_FLAC/exp07_BVextend/**/*exp07_BVext_selcurve_S92500*.json"]),
 ("P1 vanilla @40k (SyncBN-64)", "vanilla eval", 1, ["outputs_FLAC/exp07_P1/**/*exp07_P140_K1_s4[2-6]*.json"]),
 ("P1 vanilla @40k (SyncBN-64)", "vanilla eval", 8, ["outputs_FLAC/exp07_P1/**/*exp07_P140_K8_s4[3-6]*.json",
                                                     "outputs_FLAC/exp07_P1/**/*exp07_P1_screen_S40000_ema*.json"]),
 ("P1 vanilla @57.5k (SyncBN-64)", "vanilla eval", 1, ["outputs_FLAC/exp07_P1/**/*exp07_P1_gate57_K1_seed4[2-6]*.json"]),
 ("P1 vanilla @57.5k (SyncBN-64)", "vanilla eval", 8, ["outputs_FLAC/exp07_P1/**/*exp07_P1_gate57_K8_seed4[3-6]*.json",
                                                       "outputs_FLAC/exp07_P1/**/*exp07_P1_selcurve_S57500*.json"]),
 ("P1 vanilla @67.5k (endpoint)", "vanilla eval", 1, ["outputs_FLAC/exp07_P1/**/*exp07_P1_gate67_K1_seed4[2-6]*.json"]),
 ("P1 vanilla @67.5k (endpoint)", "vanilla eval", 8, ["outputs_FLAC/exp07_P1/**/*exp07_P1_gate67_K8_seed4[3-6]*.json",
                                                      "outputs_FLAC/exp07_P1/**/*exp07_P1_selcurve_S67500*.json"]),
 ("P1 vanilla @87.5k - ANCHOR", "vanilla eval", 1, ["outputs_FLAC/exp07_P1/**/*exp07_P1ext_confirm87_K1_seed4[2-6]*.json"]),
 ("P1 vanilla @87.5k - ANCHOR", "vanilla eval", 8, ["outputs_FLAC/exp07_P1/**/*exp07_P1ext_confirm87_K8_seed4[3-6]*.json",
                                                    "outputs_FLAC/exp07_P1/**/*exp07_P1ext_screen_S87500.json"]),
 ("fa fine-tune Fw @95k (equivariant)", "fa eval", 1, ["outputs_FLAC/exp09_Fw/**/*exp09_Fw95_fae_K1_s4[2-6]*.json"]),
 ("fa fine-tune Fw @95k (equivariant)", "fa eval", 8, ["outputs_FLAC/exp09_Fw/**/*exp09_Fw95_fae_K8_s4[3-6]*.json",
                                                       "outputs_FLAC/exp09_Fw/**/*exp09_Fw_fae_S95000*.json"]),
 ("fa scratch B-F @40k (equivariant, 59% budget)", "fa eval", 1, ["outputs_FLAC/exp07_BF/**/*exp10_BF40_K1_s4[2-6]*.json"]),
 ("fa scratch B-F @40k (equivariant, 59% budget)", "fa eval", 8, ["outputs_FLAC/exp07_BF/**/*exp10_BF40_K8_s4[3-6]*.json",
                                                                  "outputs_FLAC/exp07_BF/**/*BF40_fae_rot0*.json"]),
 ("P1 vanilla @40k + fa-eval (decomposition)", "fa eval", 1, ["outputs_FLAC/exp07_P1/**/*exp10_P140fae_K1_s4[2-6]*.json"]),
 ("P1 vanilla @40k + fa-eval (decomposition)", "fa eval", 8, ["outputs_FLAC/exp07_P1/**/*exp10_P140fae_K8_s4[2-6]*.json"]),
 # exp_10 endpoint rows: registered in advance; render pending until the gate JSONs land
 ("fa scratch @67.5k (exp_10, pending gates)", "fa eval", 1, ["outputs_FLAC/exp10_BF/**/*exp10_BF67_K1_s4[2-6]*.json"]),
 ("fa scratch @67.5k (exp_10, pending gates)", "fa eval", 8, ["outputs_FLAC/exp10_BF/**/*exp10_BF67_K8_s4[3-6]*.json"]),
]

def agg(patterns):
    files = sorted(set(sum((glob.glob(os.path.join(REPO, p), recursive=True) for p in patterns), [])))
    if not files:
        return None, 0
    vals = {k: [] for k, _ in KEYS}
    for f in files:
        d = json.load(open(f))["metrics"]
        for k, kk in KEYS:
            vals[k].append(d[kk])
    n = len(files)
    return {k: (st.mean(v), st.stdev(v) if n > 1 else 0.0) for k, v in vals.items()}, n

lines = [
    "# Model comparison — cross-experiment results table",
    "",
    f"Auto-generated by `gen_model_comparison.py` on {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} — "
    "do not edit numbers by hand; every cell is aggregated from the raw per-seed metric JSONs on disk.",
    "All rows: full published unseen split (6,337 items / 17 rooms), EMA weights, cfg 1.0, bf16 cond-autocast; "
    "mean ± std over eval seeds (target 5). Equivariant arms are evaluated under their own fa protocol "
    "(`--cond-method fa_invariant`); vanilla arms under vanilla eval. Single-seed screens never enter this table.",
    "",
    "| Model | eval | K | n | T60 ↓ | C50 ↓ | EDT ↓ | R@1 ↑ | R@5 ↑ | R@10 ↑ |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for label, proto, K, pats in ROWS:
    r, n = agg(pats)
    if r is None or n < MIN_SEEDS:
        lines.append(f"| {label} | {proto} | {K} | {n} | *pending ({n}/{MIN_SEEDS} seeds on disk)* | | | | | |")
        continue
    cells = " | ".join(f"{m:.3f} ± {s:.3f}" if k != "C50" else f"{m:.4f} ± {s:.4f}" for k, (m, s) in r.items())
    lines.append(f"| {label} | {proto} | {K} | {n} | {cells} |")
lines += ["", "Provenance: row specs (glob patterns) live in `gen_model_comparison.py`; "
          "checkpoint paths and eval commands in each experiment's `_command.md`."]
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
