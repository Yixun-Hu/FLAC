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
 ("fa fine-tune Fw @95k (equivariant)", "fa eval (legacy-loop)", 1, ["outputs_FLAC/exp09_Fw/**/*exp09_Fw95_fae_K1_s4[2-6]*.json"]),
 ("fa fine-tune Fw @95k (equivariant)", "fa eval (legacy-loop)", 8, ["outputs_FLAC/exp09_Fw/**/*exp09_Fw95_fae_K8_s4[3-6]*.json",
                                                       "outputs_FLAC/exp09_Fw/**/*exp09_Fw_fae_S95000*.json"]),
 ("fa scratch B-F @40k (equivariant, 59% budget)", "fa eval (legacy-loop)", 1, ["outputs_FLAC/exp07_BF/**/*exp10_BF40_K1_s4[2-6]*.json"]),
 ("fa scratch B-F @40k (equivariant, 59% budget)", "fa eval (legacy-loop)", 8, ["outputs_FLAC/exp07_BF/**/*exp10_BF40_K8_s4[3-6]*.json",
                                                                  "outputs_FLAC/exp07_BF/**/*BF40_fae_rot0*.json"]),
 ("P1 vanilla @40k + fa-eval (decomposition)", "fa eval (legacy-loop)", 1, ["outputs_FLAC/exp07_P1/**/*exp10_P140fae_K1_s4[2-6]*.json"]),
 ("P1 vanilla @40k + fa-eval (decomposition)", "fa eval (legacy-loop)", 8, ["outputs_FLAC/exp07_P1/**/*exp10_P140fae_K8_s4[2-6]*.json"]),
 ("fa scratch B-F @40k + vanilla-eval (2x2 off-diagonal)", "vanilla eval", 1, ["outputs_FLAC/exp07_BF/**/*exp10_BF40van_K1_s4[2-6]*.json"]),
 ("fa scratch B-F @40k + vanilla-eval (2x2 off-diagonal)", "vanilla eval", 8, ["outputs_FLAC/exp07_BF/**/*exp10_BF40van_K8_s4[2-6]*.json"]),
 # exp_10 endpoint rows: registered in advance; render pending until the gate JSONs land
 ("fa scratch @67.5k (exp_10, pending gates)", "fa eval (legacy-loop)", 1, ["outputs_FLAC/exp10_BF/**/*exp10_BF67_K1_s4[2-6]*.json"]),
 ("fa scratch @67.5k (exp_10, pending gates)", "fa eval (legacy-loop)", 8, ["outputs_FLAC/exp10_BF/**/*exp10_BF67_K8_s4[3-6]*.json"]),
]

EXP11_VALIDATOR = os.path.join(REPO, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude",
                               "exp11_validate_rows.py")


def is_exp11_row(patterns):
    """Does this row's evidence come from the exp_11 orbit sweep?"""
    return any("exp11_" in p for p in patterns)


def _load_validator():
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp11_validate_rows", EXP11_VALIDATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_exp11_cell(files):
    """Gate an exp_11 table cell (round-4 review B5).

    The validator used to be advisory: this script globbed raw JSONs and averaged
    whatever it found, so an unprovable row could become a published number. Now
    a cell must pass the TABLE contract — five conf rows, one checkpoint/config/
    evaluator identity, the batched orbit, EMA, the full split — or the row is
    rendered blocked instead of numeric. An EMPTY cell is not a failure: the
    generator already renders those as pending.
    """
    if not files:
        return True, []
    try:
        V = _load_validator()
    except Exception as exc:                       # the gate must not fail open
        return False, [f"cannot load the exp_11 validator ({exc})"]
    try:
        first, probs = V.validate_row(files[0])
        if not first:
            return False, probs
        rows, problems = V.validate_cell(files, arm=first["arm"], step=first["step"],
                                         k=first["K"], contract="table")
    except Exception as exc:
        return False, [f"validation raised {type(exc).__name__}: {exc}"]
    return (not problems), problems


def render_row(label, proto, K, files):
    """``(markdown_line, blocked)`` for one row, gating exp_11 evidence."""
    if is_exp11_row(files) or any("exp11_" in os.path.basename(f) for f in files):
        ok, problems = validate_exp11_cell(files)
        if not ok:
            detail = problems[0] if problems else "unspecified"
            return (f"| {label} | {proto} | {K} | {len(files)} | "
                    f"**BLOCKED — row validation failed:** {detail} | | | | | |"), True
    r, n = agg_files(files)
    if r is None or n < MIN_SEEDS:
        return (f"| {label} | {proto} | {K} | {n} | "
                f"*pending ({n}/{MIN_SEEDS} seeds on disk)* | | | | | |"), False
    cells = " | ".join(f"{m:.3f} ± {s:.3f}" if k != "C50" else f"{m:.4f} ± {s:.4f}"
                       for k, (m, s) in r.items())
    return f"| {label} | {proto} | {K} | {n} | {cells} |", False


def agg_files(files):
    if not files:
        return None, 0
    vals = {k: [] for k, _ in KEYS}
    for f in files:
        d = json.load(open(f))["metrics"]
        for k, kk in KEYS:
            vals[k].append(d[kk])
    n = len(files)
    return {k: (st.mean(v), st.stdev(v) if n > 1 else 0.0) for k, v in vals.items()}, n


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

def main():
    """Regenerate the table. Behind main() ON PURPOSE (round-4 re-review item 6):
    this module is imported by its tests, and at module scope the write below
    would rewrite model_comparison.md merely by running pytest — which would
    defeat the deliberate freeze while the exp_10 evidence is missing."""
    lines = [
        "# Model comparison — cross-experiment results table",
        "",
        f"Auto-generated by `gen_model_comparison.py` on {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} — "
        "do not edit numbers by hand; every cell is aggregated from the raw per-seed metric JSONs on disk.",
        "All rows: full published unseen split (6,337 items / 17 rooms), EMA weights, cfg 1.0, bf16 cond-autocast; "
        "mean ± std over eval seeds (target 5). Equivariant arms are evaluated under their own fa protocol "
        "(`--cond-method fa_invariant`); vanilla arms under vanilla eval. Single-seed screens never enter this table.",
        "",
        "**Orbit execution (exp_11 disclosure).** `fa eval (legacy-loop)` rows were produced by the "
        "per-angle frame-average loop; `fa eval (batched)` rows by the batched-orbit implementation, which "
        "shares one train-mode RoPE draw per chunk and regroups the split's tail batch. The two are NOT "
        "interchangeable: inference is reserved for the contemporaneous C4L bridge and the other exp_11 arms, "
        "and legacy-loop rows are background, never a substitute for C4L. exp_11 rows are emitted only after "
        "`exp11_validate_rows.py` passes the table contract for that cell; a cell that fails renders as "
        "**BLOCKED**.",
        "",
        "| Model | eval | K | n | T60 ↓ | C50 ↓ | EDT ↓ | R@1 ↑ | R@5 ↑ | R@10 ↑ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    blocked_rows = []
    for label, proto, K, pats in ROWS:
        files = sorted(set(sum((glob.glob(os.path.join(REPO, p), recursive=True) for p in pats), [])))
        line, blocked = render_row(label, proto, K, files)
        lines.append(line)
        if blocked:
            blocked_rows.append(f"{label} (K={K})")
    lines += ["", "Provenance: row specs (glob patterns) live in `gen_model_comparison.py`; "
              "checkpoint paths and eval commands in each experiment's `_command.md`."]
    if blocked_rows:
        lines += ["", f"**{len(blocked_rows)} row(s) BLOCKED by row validation:** "
                  + ", ".join(blocked_rows) + ". Fix the evidence, not the table."]
    open(OUT, "w").write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
