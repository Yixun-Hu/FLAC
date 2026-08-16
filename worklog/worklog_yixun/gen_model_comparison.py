#!/usr/bin/env python3
"""gen_model_comparison.py - regenerates worklog/worklog_yixun/model_comparison.md.

Single source of truth for the cross-experiment model table (Yixun directive
2026-08-03: log the table to model_comparison.md and commit+push on every model
-results update). Rows are registered below as (label, K, [glob patterns]); every
cell is aggregated fresh from the raw per-seed metric JSONs on disk — numbers
never live in this script. Rows whose JSON count is below MIN_SEEDS are rendered
as pending. To add a model: append a row spec, rerun, commit.
"""
import argparse, json, glob, math, os, re, statistics as st, sys, datetime

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "worklog", "worklog_yixun", "model_comparison.md")
KEYS = [("T60", "T60"), ("C50", "C50"), ("EDT", "EDT"),
        ("R@1", "RIR_to_GT_RIR_R@1"), ("R@5", "RIR_to_GT_RIR_R@5"), ("R@10", "RIR_to_GT_RIR_R@10")]
MIN_SEEDS = 5

# (model label, eval-protocol note, K, [glob patterns relative to repo root])
ROWS = [
 # --- neuronic line (2026-08-07): exp_02 P1 verification + exp_03 max_mlp ablation;
 # raws imported under outputs_FLAC/{exp02_neuronic_import,exp03n_maxpoolmlp_import}
 # (sha256 manifest in exp02_neuronic_import/IMPORT_SHA256SUMS.txt) ---
 ("P1 @87.5k re-eval (neuronic, original ckpt)", "vanilla eval", 1, ["outputs_FLAC/exp02_neuronic_import/*orig87500_K1_s4[2-6].json"]),
 ("P1 @87.5k re-eval (neuronic, original ckpt)", "vanilla eval", 8, ["outputs_FLAC/exp02_neuronic_import/*orig87500_K8_s4[2-6].json"]),
 ("P1 rerun @87.5k (neuronic from-scratch)", "vanilla eval", 1, ["outputs_FLAC/exp02_neuronic_import/*new87500_K1_s4[2-6].json"]),
 ("P1 rerun @87.5k (neuronic from-scratch)", "vanilla eval", 8, ["outputs_FLAC/exp02_neuronic_import/*new87500_K8_s4[2-6].json"]),
 ("cyl no-SSL max_mlp @40k (exp_03 neuronic)", "fa eval, online", 1, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_40000_online_K1_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_mlp @40k (exp_03 neuronic)", "fa eval, online", 8, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_40000_online_K8_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_mlp @40k (exp_03 neuronic)", "fa eval, EMA", 1, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_40000_ema_K1_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_mlp @40k (exp_03 neuronic)", "fa eval, EMA", 8, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_40000_ema_K8_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_mlp @67.5k (exp_03 neuronic)", "fa eval, online", 1, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_67500_online_K1_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_mlp @67.5k (exp_03 neuronic)", "fa eval, online", 8, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_67500_online_K8_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_mlp @67.5k (exp_03 neuronic)", "fa eval, EMA", 1, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_67500_ema_K1_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_mlp @67.5k (exp_03 neuronic)", "fa eval, EMA", 8, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_67500_ema_K8_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL mean_mlp @40k (exp_04 neuronic)", "fa eval, online", 1, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_40000_online_K1_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL mean_mlp @40k (exp_04 neuronic)", "fa eval, online", 8, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_40000_online_K8_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL mean_mlp @40k (exp_04 neuronic)", "fa eval, EMA", 1, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_40000_ema_K1_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL mean_mlp @40k (exp_04 neuronic)", "fa eval, EMA", 8, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_40000_ema_K8_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL mean_mlp @67.5k (exp_04 neuronic)", "fa eval, online", 1, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_67500_online_K1_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL mean_mlp @67.5k (exp_04 neuronic)", "fa eval, online", 8, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_67500_online_K8_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL mean_mlp @67.5k (exp_04 neuronic)", "fa eval, EMA", 1, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_67500_ema_K1_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL mean_mlp @67.5k (exp_04 neuronic)", "fa eval, EMA", 8, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_67500_ema_K8_s4[2-6]_fa_invariant_a1.json"]),
 # --- exp_06 neuronic: max pooling + bare Linear head (2x2 completion); raws
 # imported under outputs_FLAC/exp06n_maxpoollinear_import (sha manifest there) ---
 ("cyl no-SSL max_linear @40k (exp_06 neuronic)", "fa eval, online", 1, ["outputs_FLAC/exp06n_maxpoollinear_import/*exp06n_40000_online_K1_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_linear @40k (exp_06 neuronic)", "fa eval, online", 8, ["outputs_FLAC/exp06n_maxpoollinear_import/*exp06n_40000_online_K8_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_linear @40k (exp_06 neuronic)", "fa eval, EMA", 1, ["outputs_FLAC/exp06n_maxpoollinear_import/*exp06n_40000_ema_K1_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_linear @40k (exp_06 neuronic)", "fa eval, EMA", 8, ["outputs_FLAC/exp06n_maxpoollinear_import/*exp06n_40000_ema_K8_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_linear @67.5k (exp_06 neuronic)", "fa eval, online", 1, ["outputs_FLAC/exp06n_maxpoollinear_import/*exp06n_67500_online_K1_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_linear @67.5k (exp_06 neuronic)", "fa eval, online", 8, ["outputs_FLAC/exp06n_maxpoollinear_import/*exp06n_67500_online_K8_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_linear @67.5k (exp_06 neuronic)", "fa eval, EMA", 1, ["outputs_FLAC/exp06n_maxpoollinear_import/*exp06n_67500_ema_K1_s4[2-6]_fa_invariant_a1.json"]),
 ("cyl no-SSL max_linear @67.5k (exp_06 neuronic)", "fa eval, EMA", 8, ["outputs_FLAC/exp06n_maxpoollinear_import/*exp06n_67500_ema_K8_s4[2-6]_fa_invariant_a1.json"]),
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
 ("fa scratch B-F @40k + vanilla-eval (2x2 off-diagonal)", "vanilla eval", 1, ["outputs_FLAC/exp07_BF/**/*exp10_BF40van_K1_s4[2-6]*.json"]),
 ("fa scratch B-F @40k + vanilla-eval (2x2 off-diagonal)", "vanilla eval", 8, ["outputs_FLAC/exp07_BF/**/*exp10_BF40van_K8_s4[2-6]*.json"]),
 ("fa scratch @62.5k (band-typical best, exploratory)", "fa eval", 1, ["outputs_FLAC/exp10_BF/**/*exp10_BF62_K1_s4[2-6]*.json"]),
 ("fa scratch @62.5k (band-typical best, exploratory)", "fa eval", 8, ["outputs_FLAC/exp10_BF/**/*exp10_BF62_K8_s4[3-6]*.json", "outputs_FLAC/exp10_BF/**/*exp10_BF_screen_S62500*.json"]),
 ("fa scratch @25k (matched-compute vs anchor)", "fa eval", 1, ["outputs_FLAC/exp07_BF/**/*exp10_BFmc_K1_s4[2-6]*.json"]),
 ("fa scratch @25k (matched-compute vs anchor)", "fa eval", 8, ["outputs_FLAC/exp07_BF/**/*exp10_BFmc_K8_s4[2-6]*.json"]),
 ("decay-tail S93750 (C50/retrieval flavor, exploratory)", "vanilla eval", 1, ["outputs_FLAC/exp13_DT/**/*exp13_DT93750_K1_s4[2-6]*.json"]),
 ("decay-tail S93750 (C50/retrieval flavor, exploratory)", "vanilla eval", 8, ["outputs_FLAC/exp13_DT/**/*exp13_DT93750_K8_s4[3-6]*.json", "outputs_FLAC/exp13_DT/**/*exp13_DT_screen_S93750*.json"]),
 # exp_11 VANL (Q9): the vanilla arm of THIS recipe. Labelled distinctly from the
 # legacy vanilla rows on purpose — it is vanilla conditioning in the batched era,
 # so VANL vs C4L is frame averaging alone, while VANL vs a legacy vanilla row
 # would still carry the whole recipe/environment shift the C4L bridge measured.
 # NOTE the EXACT metric suffix. A trailing "*.json" also matches each row's
 # "<name>.json.screenmeta.json" sidecar, which would hand the validator ten
 # files for a five-seed cell and render the row BLOCKED for a glob bug.
 # The Q9 namespace (cell q9) keeps this round's evidence separate from the
 # original campaign's conf cells, so C4L's 0c6e9ff rows are preserved.
 ("fa-recipe vanilla VANL @40k (exp_11 baseline)", "vanilla eval (batched-era)", 1,
  ["outputs_FLAC/exp11_VANL/**/*exp11_VANL_q9_S40000_s4[2-6]_K1.json"], "q9"),
 ("fa-recipe vanilla VANL @40k (exp_11 baseline)", "vanilla eval (batched-era)", 8,
  ["outputs_FLAC/exp11_VANL/**/*exp11_VANL_q9_S40000_s4[2-6]_K8.json"], "q9"),
 ("C4L @40k re-measured at Q (Q9 fa side)", "fa eval (batched)", 1,
  ["outputs_FLAC/exp11_C4L/**/*exp11_C4L_q9_S40000_s4[2-6]_K1_fa_invariant_a4.json"], "q9"),
 ("C4L @40k re-measured at Q (Q9 fa side)", "fa eval (batched)", 8,
  ["outputs_FLAC/exp11_C4L/**/*exp11_C4L_q9_S40000_s4[2-6]_K8_fa_invariant_a4.json"], "q9"),
 # exp_10 endpoint rows: registered in advance; render pending until the gate JSONs land
 ("fa scratch @67.5k (exp_10, pending gates)", "fa eval", 1, ["outputs_FLAC/exp10_BF/**/*exp10_BF67_K1_s4[2-6]*.json"]),
 ("fa scratch @67.5k (exp_10, pending gates)", "fa eval", 8, ["outputs_FLAC/exp10_BF/**/*exp10_BF67_K8_s4[3-6]*.json"]),
 ("fa orbit C4L @40k (exp_11 bridge)", "fa eval", 1, ["outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/*step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s4[2-6]_K1_fa_invariant_a4.json"]),
 ("fa orbit C4L @40k (exp_11 bridge)", "fa eval", 8, ["outputs_FLAC/exp11_C4L/FLAC_exp11_C4L/exp11_C4L/checkpoints/*step=40000_metrics_1_1.0_exp11_C4L_conf_S40000_s4[2-6]_K8_fa_invariant_a4.json"]),
 ("fa orbit C8 @40k (exp_11)", "fa eval", 1, ["outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/*step=40000_metrics_1_1.0_exp11_C8_conf_S40000_s4[2-6]_K1_fa_invariant_a8.json"]),
 ("fa orbit C8 @40k (exp_11)", "fa eval", 8, ["outputs_FLAC/exp11_C8/FLAC_exp11_C8/exp11_C8/checkpoints/*step=40000_metrics_1_1.0_exp11_C8_conf_S40000_s4[2-6]_K8_fa_invariant_a8.json"]),
 ("fa orbit C16 @40k (exp_11)", "fa eval", 1, ["outputs_FLAC/exp11_C16/FLAC_exp11_C16/exp11_C16/checkpoints/*step=40000_metrics_1_1.0_exp11_C16_conf_S40000_s4[2-6]_K1_fa_invariant_a16.json"]),
 ("fa orbit C16 @40k (exp_11)", "fa eval", 8, ["outputs_FLAC/exp11_C16/FLAC_exp11_C16/exp11_C16/checkpoints/*step=40000_metrics_1_1.0_exp11_C16_conf_S40000_s4[2-6]_K8_fa_invariant_a16.json"]),
 ("fa orbit C32 @40k (exp_11)", "fa eval", 1, ["outputs_FLAC/exp11_C32/FLAC_exp11_C32/exp11_C32/checkpoints/*step=40000_metrics_1_1.0_exp11_C32_conf_S40000_s4[2-6]_K1_fa_invariant_a32.json"]),
 ("fa orbit C32 @40k (exp_11)", "fa eval", 8, ["outputs_FLAC/exp11_C32/FLAC_exp11_C32/exp11_C32/checkpoints/*step=40000_metrics_1_1.0_exp11_C32_conf_S40000_s4[2-6]_K8_fa_invariant_a32.json"]),
 # --- exp_14 Z: all five arms re-measured at ONE campaign pin -----------------
 # Plan §5.7 as AMENDED (worklog 2026-08-11 12:39): exp_11's Q9 already populated
 # the VANL row, so exp_14 no longer replaces any exp_11 spec — it ADDS its own
 # rows and exp_11's specs, rows and validator are untouched. What these rows are
 # for: the θ=0 reference block of the yaw-generalization campaign, where all five
 # arms were measured at one evaluator pin, so VANL-vs-Cn here is frame averaging
 # alone with no cross-pin difference folded in.
 # They live UNDER outputs_FLAC/exp11_<ARM>/ (the arm's own run directory), so the
 # "exp11_ appears in the pattern" heuristic would hand them to a validator that
 # only understands exp_11 eval names — hence the explicit exp14z contract, which
 # routes them to exp_14's own per-cell validator instead.
] + [
 (f"{arm} @40k — exp_14 Z (one-pin re-measurement)",
  "vanilla eval (batched-era)" if arm == "VANL" else "fa eval (batched)", k,
  [f"outputs_FLAC/exp11_{arm}/FLAC_exp11_{arm}/exp11_{arm}/checkpoints/"
   f"*step=40000_metrics_1_1.0_exp14_{arm}_zref_S40000_s4[2-6]_K{k}"
   + ("" if arm == "VANL" else f"_fa_invariant_a{orbit}") + ".json"], "exp14z")
 for arm, orbit in (("VANL", 0), ("C4L", 4), ("C8", 8), ("C16", 16), ("C32", 32))
 for k in (1, 8)
] + [
 # --- exp_15 YAWAUG (plan §6.9): the augmentation arm's Table-1 row ------------
 # ADDITIVE, appended as its OWN list so no existing spec or comprehension is
 # touched. exp_15 adds NOTHING for VANL — that row is exp_14's §5.7 contract, and
 # exp_15's own VANL cells stay a results-local reproduction block (plan §6.9).
 # contract="exp15" routes these through exp_15's validator AND through §13's
 # ratified aggregation: T60/C50/EDT are the mean over the ten AR room families,
 # retrieval stays split-level. The flat averaging every other row uses would
 # publish a split-level T60 under a label that promises a scene mean.
 # Registered in advance; renders *pending* until the five T cells land.
 ("yaw-aug vanilla YAWAUG @40k (exp_15)", "vanilla eval, theta=0 (scene-routed)", 1,
  ["outputs_FLAC/exp15_YAWAUG/**/*exp15_YAWAUG_tbl_S40000_s4[2-6]_K1.json"], "exp15"),
 ("yaw-aug vanilla YAWAUG @40k (exp_15)", "vanilla eval, theta=0 (scene-routed)", 8,
  ["outputs_FLAC/exp15_YAWAUG/**/*exp15_YAWAUG_tbl_S40000_s4[2-6]_K8.json"], "exp15"),
]

EXP11_VALIDATOR = os.path.join(REPO, "worklog", "worklog_yixun", "exp_11_fa_orbit_claude",
                               "exp11_validate_rows.py")


def repo_paths(repo_root=None):
    """Resolve every path this generator touches against ONE explicit root.

    Measurement now runs from pinned worktrees, so the cwd is routinely NOT the
    main checkout. Globbing evidence relative to an ambient cwd would silently
    find nothing (rendering published rows as *pending*) or, worse, find a
    worktree's stale copy. Everything — evidence, validator, output — hangs off
    this one root, which defaults to the main tree this file lives in."""
    root = os.path.realpath(repo_root or REPO)
    return {
        "root": root,
        "out": os.path.join(root, "worklog", "worklog_yixun", "model_comparison.md"),
        "validator": os.path.join(root, "worklog", "worklog_yixun",
                                  "exp_11_fa_orbit_claude", "exp11_validate_rows.py"),
    }


def is_exp14_row(patterns):
    """Does this row's evidence come from the exp_14 yaw campaign's Z block?"""
    return any("exp14_" in p for p in patterns)


def is_exp11_row(patterns):
    """Does this row's evidence come from the exp_11 orbit sweep?

    exp_14's cells live under ``outputs_FLAC/exp11_<ARM>/`` — the ARM's run
    directory, which exp_11 created — so the substring alone would claim them and
    send them to a validator that only understands exp_11 eval names. Excluding
    them here changes nothing for any exp_11 row (none names an exp14 artifact)
    and keeps exp_11's gate exactly as it was.
    """
    return any("exp11_" in p for p in patterns) and not is_exp14_row(patterns)


def is_batched_orbit_row(patterns):
    """Was this row produced by the BATCHED orbit implementation?

    True for exp_11 and for exp_14, which evaluates through the same code path.
    The legacy-loop label migration must not claim either of them.
    """
    return is_exp11_row(patterns) or is_exp14_row(patterns)


def _load_validator(validator_path=None):
    """Import the validator module from an explicit path.

    It puts its own repo root on sys.path to reach ``src.*``; make sure that
    root is importable from here too, because a generator run from a pinned
    worktree (or any foreign cwd) has no ``src`` on its path at all — the
    failure mode is an unloadable validator, which fails the gate CLOSED and
    would render every real row BLOCKED for the wrong reason."""
    import importlib.util
    path = os.path.abspath(validator_path or EXP11_VALIDATOR)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(path))))
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location("exp11_validate_rows", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_exp11_cell(files, repo_root=None, validator_path=None, contract="table"):
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
        # NOTE: ``repo_root`` redirects where EVIDENCE is resolved; it does not
        # decide which validator runs. main() passes an explicit validator_path
        # for its root; everyone else gets this checkout's validator.
        V = _load_validator(validator_path)
        if repo_root:                              # tests resolve the canonical arm
            V.REPO = repo_root                     # roots inside their fixture tree
            V.OUTPUT_ROOT_BASE = repo_root         # (outputs live under the main tree
                                                   #  in production; unchanged there)
    except Exception as exc:                       # the gate must not fail open
        return False, [f"cannot load the exp_11 validator ({exc})"]
    try:
        first, probs = V.validate_row(files[0])
        if not first:
            return False, probs
        # The contract is a property of the ROW, not a constant: q9 rows live in
        # their own registered cell and are inadmissible under `table`, so a
        # hard-coded "table" renders every populated q9 row BLOCKED.
        rows, problems = V.validate_cell(files, arm=first["arm"], step=first["step"],
                                         k=first["K"], contract=contract,
                                         verify_hashes=True)   # item 3: never trust the sidecar
    except Exception as exc:
        return False, [f"validation raised {type(exc).__name__}: {exc}"]
    return (not problems), problems


# --- the exp_14 Z contract (plan §5.7, amended) ------------------------------
EXP14_DIR = ("worklog", "worklog_yixun", "exp_14_yaw_gen_claude")
EXP14_EXPECTED_COUNT = 6337          # the full published unseen split
EXP14_LABEL = "exp_14 Z (one-pin re-measurement)"


def exp14_validator_path(repo_root=None):
    root = repo_paths(repo_root)["root"]
    return os.path.join(root, *EXP14_DIR, "exp14_validate_cell.py")


def _load_exp14_validator(repo_root=None):
    """Import exp_14's per-cell validator — the SAME predicate the screen driver,
    the wave submitter and the collector run."""
    import importlib.util
    path = exp14_validator_path(repo_root)
    spec = importlib.util.spec_from_file_location("exp14_validate_cell", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the exp_14 validator from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _exp14_cell_reasons(V14, path, cell, pin, ckpt_sha, count):
    """Named reasons one exp_14 row is not the cell it claims to be.

    This is the IMPORTED top-level predicate — ``exp14_validate_cell.validate_cell``
    — not a re-implementation that calls some of its parts. The stream audit is
    therefore REQUIRED, exactly as it is for the screen driver and the collector
    (round-3 review B3): the kit passes --record-stream for every cell, so a
    missing ``.stream.json`` is a missing artifact, not a scope decision, and
    without it the split order, substitutions, tuple counts and recomputed hashes
    are all unproved while a numeric row publishes anyway.
    """
    reasons = list(V14.validate_cell(path, cell, pin=pin, ckpt_sha=ckpt_sha,
                                     expected_count=count))
    # ...and the payload this TABLE will print: present, numeric, finite (B5).
    # A cell can satisfy every provenance rule and still carry a NaN, which would
    # propagate silently into a published mean.
    try:
        with open(path) as fh:
            metrics = (json.load(fh) or {}).get("metrics")
    except (ValueError, OSError) as exc:
        return reasons + [f"unreadable while checking the reported metrics ({exc})"]
    if not isinstance(metrics, dict):
        return reasons + ["metrics block is missing or is not an object"]
    for _label, key in KEYS:
        if key not in metrics:
            reasons.append(f"metrics does not report {key}, which this table prints")
        elif not _finite_number(metrics[key]):
            reasons.append(f"metrics[{key!r}] = {metrics[key]!r} is not a finite number")
    return reasons


def validate_exp14_cell(files, repo_root=None, expected_count=None,
                        expected_arm=None, expected_k=None):
    """Gate one exp_14 table cell: five unrotated seeds, one pin, one checkpoint.

    ``expected_arm``/``expected_k`` come from the ROW CONTRACT, and binding them
    is the point (round-3 review B2): a homogeneous five-seed C16 payload renamed
    to match a C8 glob otherwise validates under the C8 label, and the table
    publishes one arm's numbers beside another arm's name.

    Empty is not a failure — the generator already renders an empty cell as
    pending."""
    if not files:
        return True, []
    count = int(expected_count or EXP14_EXPECTED_COUNT)
    try:
        V14 = _load_exp14_validator(repo_root)
    except Exception as exc:                       # the gate must not fail open
        return False, [f"cannot load the exp_14 validator ({exc})"]
    records, problems = {}, []
    for path in sorted(files):
        try:
            with open(path) as fh:
                records[path] = json.load(fh)
        except (ValueError, OSError) as exc:
            problems.append(f"{os.path.basename(path)}: unreadable ({exc})")
    if problems:
        return False, problems
    shas = sorted({str(r.get("source_sha")) for r in records.values()})
    if len(shas) != 1:
        return False, [f"the cell spans MORE THAN ONE evaluator pin {shas}: exp_14 is a "
                       "one-pin campaign, so a row mixing pins is not the cell it claims"]
    pin = shas[0]
    try:
        expect = V14.load_ckpt_expect(os.path.join(
            repo_paths(repo_root)["root"], *EXP14_DIR, "exp14_ckpt_expect.json"))
    except ValueError as exc:
        return False, [str(exc)]
    seeds, arms, ks = [], set(), set()
    for path, rec in sorted(records.items()):
        try:
            cell = V14.parse_eval_name(str(rec.get("eval_name") or ""))
        except ValueError as exc:
            problems.append(f"{os.path.basename(path)}: {exc}")
            continue
        if cell.cell != "zref":
            problems.append(f"{os.path.basename(path)}: only the UNROTATED (zref) block "
                            f"is a comparable table row; this is a {cell.cell} cell")
            continue
        seeds.append(int(cell.seed))
        arms.add(cell.arm)
        ks.add(int(cell.k))
        problems += [f"{os.path.basename(path)}: {r}" for r in
                     _exp14_cell_reasons(V14, path, cell, pin, expect[cell.arm], count)]
    if len(arms) > 1 or len(ks) > 1:
        problems.append(f"one cell is one (arm, K); this one spans arms {sorted(arms)} "
                        f"and K {sorted(ks)}")
    # the evidence must be the DECLARED row's, not merely self-consistent
    if expected_arm is not None and arms and arms != {expected_arm}:
        problems.append(f"the row declares arm {expected_arm} but its evidence is "
                        f"{sorted(arms)}: a renamed payload is not a re-measurement")
    if expected_k is not None and ks and ks != {int(expected_k)}:
        problems.append(f"the row declares K={int(expected_k)} but its evidence is "
                        f"K {sorted(ks)}")
    # exactly one record per registered seed: "no seed missing" also holds for six
    # files with a duplicate, which agg_files would then average as six
    if sorted(seeds) != list(V14.SEEDS):
        counts = {s: seeds.count(s) for s in sorted(set(seeds))}
        dupes = sorted(s for s, n in counts.items() if n > 1)
        problems.append(
            f"the cell must be exactly {len(V14.SEEDS)} records, one per seed "
            f"{list(V14.SEEDS)}; got {len(seeds)} with seeds {sorted(seeds)}"
            + (f" (duplicated: {dupes})" if dupes else ""))
    return (not problems), problems


def validate_exp15_cell(files, repo_root=None, expected_k=None):
    """``(ok, problems)`` for one exp_15 row, via exp_15's own validator.

    ADDITIVE: no existing row reaches this. It refuses to publish a row whose
    cells are not the registered exp_15 T cells at the pinned endpoint, are not
    protocol-conformant, or are not five distinct seeds.
    """
    problems = []
    try:
        V = _load_exp15_validator()
    except Exception as exc:                              # noqa: BLE001
        return False, [f"could not load exp_15's validator: {exc}"]
    # EXACTLY five files, not "five distinct seeds among however many matched":
    # a glob that also caught a sidecar or a stale duplicate would otherwise
    # publish a mean over six payloads (re-review finding 3).
    if len(files) != MIN_SEEDS:
        problems.append(f"{len(files)} files matched, need exactly {MIN_SEEDS}")
    seeds = set()
    pins = set()
    for f in sorted(files):
        try:
            cell = V.parse_eval_name(
                os.path.basename(f).split("_metrics_", 1)[1]
                .rsplit(".json", 1)[0].split("_", 2)[2])
        except Exception:                                  # noqa: BLE001
            # fall back to the authoritative matcher: reconstruct each registered
            # basename and compare, exactly as the collector does
            cell = None
            stem = os.path.basename(f).split("_metrics_", 1)[0]
            fake = os.path.join(os.path.dirname(f), stem + ".ckpt")
            for candidate in V.expected_grid():
                if os.path.basename(V.metrics_path(fake, candidate)) == os.path.basename(f):
                    cell = candidate
                    break
        if cell is None:
            problems.append(f"{os.path.basename(f)}: not a registered exp_15 cell")
            continue
        if cell.cell != "tbl":
            problems.append(f"{os.path.basename(f)}: only the T (theta=0) block is "
                            "publishable as a model row")
        if expected_k is not None and int(cell.k) != int(expected_k):
            problems.append(f"{os.path.basename(f)}: K={cell.k}, row says K={expected_k}")
        seeds.add(int(cell.seed))
        try:
            record = json.load(open(f))
        except Exception as exc:                           # noqa: BLE001
            problems.append(f"{os.path.basename(f)}: unreadable ({exc})")
            continue
        # The screenmeta sidecar EXISTS and records the campaign pin; validating
        # the metrics record alone left the published row unable to say which
        # commit measured it (re-review finding 3).
        meta_path = f + ".screenmeta.json"
        pin = None
        if not os.path.isfile(meta_path):
            problems.append(f"{os.path.basename(f)}: no .screenmeta.json sidecar")
        else:
            try:
                meta = json.load(open(meta_path))
            except Exception as exc:                       # noqa: BLE001
                problems.append(f"{os.path.basename(meta_path)}: unreadable ({exc})")
                meta = None
            if isinstance(meta, dict):
                pin = meta.get("commit")
                if not (isinstance(pin, str) and len(pin) == 40):
                    problems.append(f"{os.path.basename(meta_path)}: records no "
                                    "40-hex campaign pin")
                    pin = None
                pins.add(pin)
                reasons = V.validate_screenmeta(meta, cell, pin=pin,
                                                ckpt_sha=meta.get("ckpt_sha256"))
                if reasons:
                    problems.append(f"{os.path.basename(meta_path)}: {reasons[0]}")
        reasons = V.validate_metrics_record(record, cell, pin=pin)
        if reasons:
            problems.append(f"{os.path.basename(f)}: {reasons[0]}")
    if len(seeds) != MIN_SEEDS:
        problems.append(f"{len(seeds)} distinct seeds, need {MIN_SEEDS}")
    # ONE campaign pin across the row: five cells measured at different commits
    # are not a row, they are five measurements wearing one label.
    real_pins = {p for p in pins if p}
    if len(real_pins) > 1:
        problems.append(f"cells span {len(real_pins)} campaign pins: "
                        + ", ".join(sorted(p[:12] for p in real_pins)))
    return (not problems), problems


def exp14_arm_of(label):
    """The arm a registered exp_14 row label names (its first token)."""
    return str(label).split(" ", 1)[0]


def check_exp14_round(exp14_cells, exp14_status):
    """Which exp_14 arms may publish, and why the others may not.

    Returns ``(withheld_labels, problems)``.

    Two rules, deliberately at different scopes (round-3 review B4):

    * **Per ARM, a two-K transaction.** An arm publishes when BOTH its K rows
      carry a full five-seed block AND BOTH VALIDATED, and not before — a lone K
      would invite a K=8-only comparison against rows that carry both. It does
      NOT wait for the other arms: §5.7 sequences the regeneration on VANL's own
      block completing, so a campaign-wide gate would withhold exactly the row
      that triggered it.

      ``exp14_status`` maps ``(label, K) -> bool`` and is REQUIRED: readiness
      used to be declared from file counts alone, never seeing the validation
      already computed while rendering, so five valid K=1 files beside five K=8
      files carrying one NaN published a numeric K=1 row next to a BLOCKED K=8
      one — the transaction violated in exactly the direction it exists to
      prevent (round-3 closure A4/B2).
    * **Across ARMS, one campaign pin.** exp_14 is a one-pin campaign and its
      whole point is that VANL-vs-Cn carries no cross-pin difference. If the
      published arms disagree on ``source_sha``, none of them publishes.
    """
    present = {key: files for key, files in exp14_cells.items() if files}
    if not present:
        return set(), []                           # nothing landed yet: not an update
    problems, ready, shas = [], set(), {}
    arms = {exp14_arm_of(label) for label, _k in exp14_cells}
    for arm in sorted(arms):
        rows = {k: exp14_cells.get((label, k))
                for (label, k) in exp14_cells if exp14_arm_of(label) == arm}
        if not any(rows.values()):
            continue                               # this arm has not started
        short = {k: (0 if f is None else len(f)) for k, f in rows.items()
                 if f is None or len(f) < MIN_SEEDS}
        if short:
            problems.append(f"{arm}: publishes only as a complete two-K pair; "
                            + ", ".join(f"K={k} has {n}/{MIN_SEEDS} seeds"
                                        for k, n in sorted(short.items())))
            continue
        # counting files is not validating them: a five-file block can still be
        # refused (a NaN, a wrong checkpoint digest, a missing stream audit), and
        # its partner must not publish beside it.
        invalid = sorted(k for (label, k) in exp14_cells
                         if exp14_arm_of(label) == arm and exp14_cells[(label, k)]
                         and not exp14_status.get((label, k), False))
        if invalid:
            problems.append(f"{arm}: K={', K='.join(str(k) for k in invalid)} did not "
                            "validate — both K must be VALID, not merely present, or "
                            "the pair publishes a number beside a refusal")
            continue
        ready.add(arm)
    for (label, k), files in sorted(present.items()):
        if exp14_arm_of(label) not in ready:
            continue                               # an unready arm cannot fix the pin
        for path in files:
            try:
                with open(path) as fh:
                    sha = json.load(fh).get("source_sha")
            except Exception as exc:
                problems.append(f"{os.path.basename(path)}: unreadable while checking "
                                f"the exp_14 pin ({exc})")
                continue
            shas.setdefault(str(sha), []).append(f"{label} K={k}")
    if len(shas) > 1:
        detail = "; ".join(f"{sha[:12]}: {sorted(set(cells))}"
                           for sha, cells in sorted(shas.items()))
        problems.append("the published exp_14 arms span MORE THAN ONE evaluator pin, so "
                        f"VANL-vs-Cn is not a within-pin comparison — {detail}")
        ready = set()                              # nobody publishes across pins
    withheld = {label for label, _k in exp14_cells
                if exp14_arm_of(label) not in ready}
    return withheld, problems


Q9_REQUIRED_CELLS = 4        # VANL x {K1,K8} and C4L x {K1,K8}


def check_q9_round(q9_cells):
    """The Q9 estimand is a WITHIN-PIN delta, so the round is one transaction.

    Per-cell validation proves each arm x K block internally; it cannot see that
    the four blocks belong together. Without this, one arm could publish without
    its comparator, or the arms could be measured at different evaluator pins and
    still validate individually — and the frame-averaging delta would then be
    confounded with whatever moved between those pins, which is the one thing the
    same-pin design exists to exclude."""
    present = {key: files for key, files in q9_cells.items() if files}
    if not present:
        return []                                   # nothing landed yet: not an update
    problems = []
    if len(present) != Q9_REQUIRED_CELLS:
        missing = [f"{lbl} (K={k})" for (lbl, k), f in sorted(q9_cells.items()) if not f]
        problems.append(f"the Q9 round publishes as ONE transaction: {len(present)}/"
                        f"{Q9_REQUIRED_CELLS} cells have evidence, missing "
                        + (", ".join(missing) or "<unregistered cells>"))
    shas = {}
    for key, files in sorted(present.items()):
        for f in files:
            try:
                sha = json.load(open(f)).get("source_sha")
            except Exception as exc:
                problems.append(f"{os.path.basename(f)}: unreadable while checking the Q9 pin ({exc})")
                continue
            shas.setdefault(str(sha), []).append(f"{key[0]} K={key[1]}")
    if len(shas) > 1:
        detail = "; ".join(f"{sha[:12]}: {sorted(set(cells))}" for sha, cells in sorted(shas.items()))
        problems.append("the Q9 round spans MORE THAN ONE evaluator pin, so the VANL-vs-C4L "
                        f"delta is not a within-pin comparison — {detail}")
    return problems


def check_two_k_coverage(status_by_row):
    """Item 4b: an exp_11 table update is a TWO-K transaction.

    ``status_by_row`` maps ``(label, K) -> ok``. A row that lands only one K, or
    whose other K is blocked, is not a complete update: reporting half of it
    would invite a K=8-only comparison against rows that have both."""
    labels = {label for (label, _k) in status_by_row}
    problems = []
    for label in sorted(labels):
        for k in (1, 8):
            if (label, k) not in status_by_row:
                problems.append(f"{label}: K={k} cell is absent — a table update must carry both K")
            elif not status_by_row[(label, k)]:
                problems.append(f"{label}: K={k} cell did not validate — both K must be valid")
    return problems


# --- item 7: the label migration is DEFERRED until the exp_10 evidence returns -
# Relabelling means regenerating, and regenerating right now would replace the
# published exp_10 @67.5k numbers with "pending" because their per-seed JSONs are
# on another machine. So the protocol labels migrate only once that evidence is
# back; until then the table keeps its current labels and carries a loud note.
EXP10_ENDPOINT_GLOBS = ["outputs_FLAC/exp10_BF/**/*exp10_BF67_K1_s4[2-6]*.json",
                        "outputs_FLAC/exp10_BF/**/*exp10_BF67_K8_s4[3-6]*.json"]


def exp10_evidence_present(repo_root=None):
    """Are the exp_10 endpoint JSONs back on this machine?"""
    root = repo_paths(repo_root)["root"]
    return all(glob.glob(os.path.join(root, g), recursive=True) for g in EXP10_ENDPOINT_GLOBS)


def protocol_label(base, exp11, evidence_ready):
    """The row's protocol string. exp_11 rows always disclose 'batched'; the
    historical 'legacy-loop' migration waits for the deferred regeneration."""
    if "fa" not in base:
        return base
    if exp11:
        return base if "batched" in base else f"{base} (batched)"
    if not evidence_ready:
        return base
    return base if "legacy-loop" in base else f"{base} (legacy-loop)"


def build_header(evidence_ready):
    head = [
        "**Orbit execution (exp_11 disclosure).** `fa eval (legacy-loop)` rows were produced by the "
        "per-angle frame-average loop; `fa eval (batched)` rows by the batched-orbit implementation, "
        "which shares one train-mode RoPE draw per chunk and regroups the split's tail batch. The two "
        "are NOT interchangeable: inference is reserved for the contemporaneous C4L bridge and the "
        "other exp_11 arms, and legacy-loop rows are background, never a substitute for C4L. exp_11 "
        "rows are emitted only after `exp11_validate_rows.py` passes the table contract for that "
        "cell (hashes recomputed); a cell that fails renders as **BLOCKED**.",
        "",
        "**exp_14 Z rows (one-pin re-measurement).** Rows labelled `exp_14 Z` are the "
        "θ=0 reference block of the yaw-generalization campaign: all five arms measured "
        "at ONE evaluator pin, so VANL-vs-Cn among them is frame averaging alone. They "
        "are gated by `exp14_validate_cell.py` (the same predicate the screen driver and "
        "the wave submitter run) and publish only as a complete two-K, five-seed, "
        "single-pin transaction, and each arm's pair publishes independently while all "
        "published arms must share ONE campaign pin. The gate is the imported "
        "`validate_cell` predicate in full: metrics record, submission manifest AND the "
        "per-position `.stream.json` assignment audit, which must therefore travel with "
        "the evidence — an unproved split order is not a publishable row.",
    ]
    if not evidence_ready:
        head += [
            "",
            "> **LABEL MIGRATION DEFERRED.** Every `fa eval` row below WITHOUT an explicit "
            "`(batched)` tag was produced by the legacy per-angle loop and must be read as "
            "`fa eval (legacy-loop)`. The labels are not yet written into the rows because "
            "regenerating this table today would replace the published exp_10 @67.5k numbers with "
            "*pending* — those per-seed JSONs are being recovered from another machine. The "
            "generator migrates the labels automatically on the first regeneration after that "
            "evidence is back; until then the numbers here are preserved as published.",
        ]
    return head


def render_row(label, proto, K, files, repo_root=None, validator_path=None, contract="table"):
    """``(markdown_line, blocked)`` for one row, gating exp_11 and exp_14 evidence."""
    # exp_14 FIRST and exclusively: its cells sit inside outputs_FLAC/exp11_<ARM>/,
    # so the exp_11 branch below would otherwise claim them and block every one on
    # an eval name its validator was never written to parse.
    if contract == "exp14z" or is_exp14_row([os.path.basename(f) for f in files]):
        ok, problems = validate_exp14_cell(files, repo_root=repo_root,
                                           expected_arm=exp14_arm_of(label),
                                           expected_k=K)
        if not ok:
            detail = problems[0] if problems else "unspecified"
            return (f"| {label} | {proto} | {K} | {len(files)} | "
                    f"**BLOCKED — row validation failed:** {detail} | | | | | |"), True
    elif is_exp11_row(files) or any("exp11_" in os.path.basename(f) for f in files):
        ok, problems = validate_exp11_cell(files, repo_root=repo_root,
                                           validator_path=validator_path, contract=contract)
        if not ok:
            detail = problems[0] if problems else "unspecified"
            return (f"| {label} | {proto} | {K} | {len(files)} | "
                    f"**BLOCKED — row validation failed:** {detail} | | | | | |"), True
    if contract == "exp15":
        ok, problems = validate_exp15_cell(files, repo_root=repo_root, expected_k=K)
        if not ok:
            detail = problems[0] if problems else "unspecified"
            return (f"| {label} | {proto} | {K} | {len(files)} | "
                    f"**BLOCKED — row validation failed:** {detail} | | | | | |"), True
        r, n = agg_files_exp15(files)
    else:
        r, n = agg_files(files)
    if r is None or n < MIN_SEEDS:
        return (f"| {label} | {proto} | {K} | {n} | "
                f"*pending ({n}/{MIN_SEEDS} seeds on disk)* | | | | | |"), False
    cells = " | ".join(f"{m:.3f} ± {s:.3f}" if k != "C50" else f"{m:.4f} ± {s:.4f}"
                       for k, (m, s) in r.items())
    return f"| {label} | {proto} | {K} | {n} | {cells} |", False


def _finite_number(value):
    """A published cell is a finite number, never a NaN wearing one's clothes."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def agg_files(files):
    """Aggregate one cell's per-seed JSONs. RAISES on a payload it cannot print.

    Round-3 review B5: a missing key raised KeyError deep inside a regeneration,
    and a non-finite value propagated into a published mean as `nan`. Both are
    now one named ValueError, which render_row turns into a BLOCKED row.
    """
    if not files:
        return None, 0
    vals = {k: [] for k, _ in KEYS}
    for f in files:
        d = json.load(open(f))
        d = d.get("metrics") if isinstance(d, dict) else None
        if not isinstance(d, dict):
            raise ValueError(f"{os.path.basename(f)}: no metrics object to aggregate")
        for k, kk in KEYS:
            if kk not in d:
                raise ValueError(f"{os.path.basename(f)}: does not report {kk}")
            if not _finite_number(d[kk]):
                raise ValueError(f"{os.path.basename(f)}: {kk} = {d[kk]!r} is not a "
                                 "finite number")
            vals[k].append(d[kk])
    n = len(files)
    return {k: (st.mean(v), st.stdev(v) if n > 1 else 0.0) for k, v in vals.items()}, n


EXP15_VALIDATOR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "exp_15_yaw_aug_claude",
    "exp15_validate_cell.py")


def _load_exp15_validator(path=None):
    """exp_15's own validator, imported by path (same shape as exp_11/exp_14)."""
    import importlib.util
    target = os.path.abspath(path or EXP15_VALIDATOR)
    spec = importlib.util.spec_from_file_location("exp15_validate_cell", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def agg_files_exp15(files):
    """Aggregate exp_15 cells under plan §13's RATIFIED routing.

    ADDITIVE — no existing row reaches this function, and ``agg_files`` is
    untouched. It exists because §13 routes the acoustic family (T60, C50, EDT)
    to the mean over the ten AR room-family groups, which this generator's flat
    ``metrics``-block averaging cannot express: reusing it would publish a
    split-level T60 under a label that promises a scene mean. Retrieval stays
    split-level, which is what the flat path already did.
    """
    if not files:
        return None, 0
    V = _load_exp15_validator()
    vals = {k: [] for k, _ in KEYS}
    for f in files:
        d = json.load(open(f))
        flat = d.get("metrics") if isinstance(d, dict) else None
        by_scene = d.get("by_scene") if isinstance(d, dict) else None
        if not isinstance(flat, dict):
            raise ValueError(f"{os.path.basename(f)}: no metrics object to aggregate")
        if not isinstance(by_scene, dict) or len(by_scene) != len(V.EXPECTED_SCENE_KEYS):
            raise ValueError(f"{os.path.basename(f)}: needs a by_scene block over the "
                             f"{len(V.EXPECTED_SCENE_KEYS)} room families for §13 routing")
        for k, kk in KEYS:
            if kk in ("T60", "C50", "EDT"):          # acoustic -> ten-group mean
                col = []
                for scene in V.EXPECTED_SCENE_KEYS:
                    payload = by_scene.get(scene)
                    if not isinstance(payload, dict) or kk not in payload:
                        raise ValueError(f"{os.path.basename(f)}: by_scene[{scene!r}] "
                                         f"does not report {kk}")
                    if not _finite_number(payload[kk]):
                        raise ValueError(f"{os.path.basename(f)}: by_scene[{scene!r}]"
                                         f".{kk} = {payload[kk]!r} is not finite")
                    col.append(float(payload[kk]))
                vals[k].append(st.mean(col))
                continue
            if kk not in flat:                        # retrieval -> split level
                raise ValueError(f"{os.path.basename(f)}: does not report {kk}")
            if not _finite_number(flat[kk]):
                raise ValueError(f"{os.path.basename(f)}: {kk} = {flat[kk]!r} is not a "
                                 "finite number")
            vals[k].append(float(flat[kk]))
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

# --- the published table is EVIDENCE, and regeneration must not destroy it ----
# A regeneration run from the wrong root, or with evidence temporarily off-disk,
# silently rewrites rows that carry published numbers into "pending (0/5 seeds on
# disk)". That is not a table update; it is data loss with a timestamp on it.
# (Observed for real: a generator run with an argv bug replaced the exp_10 @67.5k
# rows with 0/5 pending.) Any such rewrite now aborts the whole write.
_NUMERIC_CELL = re.compile(r"^-?\d")


def parse_table_rows(text):
    """``{(label, K): {"n": int, "numeric": bool}}`` for every data row."""
    rows = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5 or cells[0] == "Model":
            continue
        try:
            k, n = int(cells[2]), int(cells[3])
        except ValueError:
            continue                      # header / footnote lines
        rows[(cells[0], k)] = {"n": n, "numeric": bool(_NUMERIC_CELL.match(cells[4]))}
    return rows


def detect_row_regressions(old_rows, new_rows):
    """What this regeneration would DESTROY. Empty means it only adds."""
    problems = []
    for key, was in sorted(old_rows.items()):
        now = new_rows.get(key)
        label, k = key
        if now is None:
            problems.append(f"{label} (K={k}): the row is in the published table but this "
                            "run would drop it entirely")
            continue
        if was["numeric"] and not now["numeric"]:
            problems.append(f"{label} (K={k}): published NUMBERS would become non-numeric "
                            f"(n {was['n']} -> {now['n']})")
        elif now["n"] < was["n"]:
            problems.append(f"{label} (K={k}): evidence count would regress "
                            f"{was['n']} -> {now['n']} seeds")
    return problems


def withheld_row(label, proto, K, n, why):
    """A row deliberately not published (an incomplete two-K transaction)."""
    return (f"| {label} | {proto} | {K} | {n} | **WITHHELD — {why}** | | | | | |")


def main(argv=None):
    """Regenerate the table. Behind main() ON PURPOSE (round-4 re-review item 6):
    this module is imported by its tests, and at module scope the write below
    would rewrite model_comparison.md merely by running pytest — which would
    defeat the deliberate freeze while the exp_10 evidence is missing."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=None,
                    help="main-tree root for ALL evidence, the validator and the output "
                         "(default: the tree this file lives in). Never the cwd: screens "
                         "run from pinned worktrees.")
    ap.add_argument("--allow-row-regression", action="store_true",
                    help="permit a regeneration that turns published numbers into pending, "
                         "drops a row, or lowers a seed count. Audited: the affected rows are "
                         "printed either way. Use only for a deliberate retraction.")
    ap.add_argument("--allow-partial-exp11", action="store_true",
                    help="write the table even when an exp_11 update covers only one K; the "
                         "affected rows still render WITHHELD, never as numbers.")
    # argv=None means "the real command line" (argparse reads sys.argv[1:]).
    # Passing [] here made --repo-root unreachable from the shell: the flag
    # parsed, then main() threw the arguments away and used the default.
    args = ap.parse_args(argv)
    paths = repo_paths(args.repo_root)
    root, out_path, validator = paths["root"], paths["out"], paths["validator"]

    evidence_ready = exp10_evidence_present(root)
    lines = [
        "# Model comparison — cross-experiment results table",
        "",
        f"Auto-generated by `gen_model_comparison.py` on {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} — "
        "do not edit numbers by hand; every cell is aggregated from the raw per-seed metric JSONs on disk.",
        "All rows: full published unseen split (6,337 items / 17 rooms), EMA weights, cfg 1.0, bf16 cond-autocast; "
        "mean ± std over eval seeds (target 5). Equivariant arms are evaluated under their own fa protocol "
        "(`--cond-method fa_invariant`); vanilla arms under vanilla eval. Single-seed screens never enter this table.",
        "",
    ]
    # The disclosure header (and, while the exp_10 evidence is away, the LABEL
    # MIGRATION DEFERRED note) is built in exactly one place. It used to be
    # duplicated inline here, so build_header() was dead code and the deferral
    # note — the thing that tells a reader which rows are legacy-loop — never
    # reached the generated file.
    lines += build_header(evidence_ready)
    lines += [
        "",
        "| Model | eval | K | n | T60 ↓ | C50 ↓ | EDT ↓ | R@1 ↑ | R@5 ↑ | R@10 ↑ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    rendered, blocked_rows, exp11_status, q9_cells = [], [], {}, {}
    exp14_cells, exp14_status = {}, {}
    for spec in ROWS:
        label, proto, K, pats = spec[:4]
        contract = spec[4] if len(spec) > 4 else "table"
        files = sorted(set(sum((glob.glob(os.path.join(root, p), recursive=True) for p in pats), [])))
        proto = protocol_label(proto, is_batched_orbit_row(pats), evidence_ready)
        line, blocked = render_row(label, proto, K, files, repo_root=root,
                                   validator_path=validator, contract=contract)
        rendered.append({"label": label, "proto": proto, "K": K, "n": len(files),
                         "line": line, "exp11": is_exp11_row(pats)})
        if contract == "q9":
            q9_cells[(label, K)] = files
        if contract == "exp14z":
            exp14_cells[(label, K)] = files
            # the validation result is computed RIGHT HERE while rendering; the
            # transaction gate consumes it rather than re-deriving readiness from
            # how many files happen to exist (round-3 closure A4/B2)
            exp14_status[(label, K)] = bool(files) and not blocked
        if blocked:
            blocked_rows.append(f"{label} (K={K})")
        if is_exp11_row(pats) and files:
            exp11_status[(label, K)] = not blocked

    # --- the two-K gate is TRANSACTIONAL, not advisory -----------------------
    # An exp_11 label lands as a PAIR or not at all: a lone K=8 row invites a
    # comparison against rows that carry both. On failure the affected label's
    # rows are rewritten as WITHHELD (no number can leak) and the write is
    # refused outright unless the operator explicitly accepts a partial table.
    q9_problems = check_q9_round(q9_cells)
    if q9_problems:
        print("Q9 ROUND is not publishable as one transaction:", file=sys.stderr)
        for problem in q9_problems:
            print("  -", problem, file=sys.stderr)
        for r in rendered:
            if (r["label"], r["K"]) in q9_cells:
                r["line"] = withheld_row(r["label"], r["proto"], r["K"], r["n"],
                                         "the Q9 round publishes only as a complete, single-pin "
                                         "set of four cells")
        lines_tail = ["", "**Q9 round WITHHELD:** " + "; ".join(q9_problems)]
    else:
        lines_tail = []
    # The exp_14 Z block is the same kind of transaction, one experiment later.
    exp14_withheld, exp14_problems = check_exp14_round(exp14_cells, exp14_status)
    if exp14_problems:
        print("exp_14 Z rows not publishable:", file=sys.stderr)
        for problem in exp14_problems:
            print("  -", problem, file=sys.stderr)
    for r in rendered:
        if (r["label"], r["K"]) in exp14_cells and r["n"] and r["label"] in exp14_withheld:
            r["line"] = withheld_row(r["label"], r["proto"], r["K"], r["n"],
                                     "this arm publishes only as a complete two-K, "
                                     "five-seed block at the one campaign pin")
    if exp14_problems:
        lines_tail += ["", "**exp_14 Z rows WITHHELD:** " + "; ".join(exp14_problems)]
    two_k = check_two_k_coverage(exp11_status)
    if two_k:
        bad_labels = {p.split(":")[0] for p in two_k}
        for r in rendered:
            if r["exp11"] and r["label"] in bad_labels:
                r["line"] = withheld_row(r["label"], r["proto"], r["K"], r["n"],
                                         "incomplete two-K update: this row publishes only "
                                         "with its K=1 and K=8 partner")
    lines += [r["line"] for r in rendered]
    lines += lines_tail
    lines += ["", "Provenance: row specs (glob patterns) live in `gen_model_comparison.py`; "
              "checkpoint paths and eval commands in each experiment's `_command.md`."]
    if blocked_rows:
        lines += ["", f"**{len(blocked_rows)} row(s) BLOCKED by row validation:** "
                  + ", ".join(blocked_rows) + ". Fix the evidence, not the table."]
    # --- would this run destroy published evidence? --------------------------
    existing = ""
    if os.path.isfile(out_path):
        existing = open(out_path).read()
    regressions = detect_row_regressions(parse_table_rows(existing),
                                         parse_table_rows("\n".join(lines)))
    if regressions:
        head = ("REGRESSION in the published table — "
                f"{len(regressions)} row(s) would lose evidence:")
        print(head, file=sys.stderr)
        for problem in regressions:
            print("  -", problem, file=sys.stderr)
        if not args.allow_row_regression:
            print(f"  (nothing was written to {out_path}; restore the missing evidence, or "
                  "rerun with --allow-row-regression if this retraction is deliberate)",
                  file=sys.stderr)
            return 4
        print("  (proceeding: --allow-row-regression was given)", file=sys.stderr)
        lines += ["", f"**{len(regressions)} row(s) REGRESSED in this regeneration "
                  "(--allow-row-regression):** " + "; ".join(regressions)]
    if two_k:
        lines += ["", "**Incomplete exp_11 table update (both K required):** " + "; ".join(two_k)]
        if not args.allow_partial_exp11:
            print("ABORTED without writing — incomplete exp_11 update (both K required):",
                  file=sys.stderr)
            for problem in two_k:
                print("  -", problem, file=sys.stderr)
            print(f"  (nothing was written to {out_path}; land the missing K, or rerun with "
                  "--allow-partial-exp11 to publish the affected rows as WITHHELD)",
                  file=sys.stderr)
            return 3
    open(out_path, "w").write("\n".join(lines) + "\n")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
