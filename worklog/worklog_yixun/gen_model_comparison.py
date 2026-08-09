#!/usr/bin/env python3
"""gen_model_comparison.py - regenerates worklog/worklog_yixun/model_comparison.md.

Single source of truth for the cross-experiment model table (Yixun directive
2026-08-03: log the table to model_comparison.md and commit+push on every model
-results update). Rows are registered below as (label, K, [glob patterns]); every
cell is aggregated fresh from the raw per-seed metric JSONs on disk — numbers
never live in this script. Rows whose JSON count is below MIN_SEEDS are rendered
as pending. To add a model: append a row spec, rerun, commit.
"""
import argparse, json, glob, os, re, statistics as st, sys, datetime

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
 ("cyl no-SSL max_mlp @40k (exp_03 neuronic)", "fa eval, online", 1, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_40000_online_K1_s4[2-6].json"]),
 ("cyl no-SSL max_mlp @40k (exp_03 neuronic)", "fa eval, online", 8, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_40000_online_K8_s4[2-6].json"]),
 ("cyl no-SSL max_mlp @40k (exp_03 neuronic)", "fa eval, EMA", 1, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_40000_ema_K1_s4[2-6].json"]),
 ("cyl no-SSL max_mlp @40k (exp_03 neuronic)", "fa eval, EMA", 8, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_40000_ema_K8_s4[2-6].json"]),
 ("cyl no-SSL max_mlp @67.5k (exp_03 neuronic)", "fa eval, online", 1, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_67500_online_K1_s4[2-6].json"]),
 ("cyl no-SSL max_mlp @67.5k (exp_03 neuronic)", "fa eval, online", 8, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_67500_online_K8_s4[2-6].json"]),
 ("cyl no-SSL max_mlp @67.5k (exp_03 neuronic)", "fa eval, EMA", 1, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_67500_ema_K1_s4[2-6].json"]),
 ("cyl no-SSL max_mlp @67.5k (exp_03 neuronic)", "fa eval, EMA", 8, ["outputs_FLAC/exp03n_maxpoolmlp_import/*exp03n_67500_ema_K8_s4[2-6].json"]),
 ("cyl no-SSL mean_mlp @40k (exp_04 neuronic)", "fa eval, online", 1, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_40000_online_K1_s4[2-6].json"]),
 ("cyl no-SSL mean_mlp @40k (exp_04 neuronic)", "fa eval, online", 8, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_40000_online_K8_s4[2-6].json"]),
 ("cyl no-SSL mean_mlp @40k (exp_04 neuronic)", "fa eval, EMA", 1, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_40000_ema_K1_s4[2-6].json"]),
 ("cyl no-SSL mean_mlp @40k (exp_04 neuronic)", "fa eval, EMA", 8, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_40000_ema_K8_s4[2-6].json"]),
 ("cyl no-SSL mean_mlp @67.5k (exp_04 neuronic)", "fa eval, online", 1, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_67500_online_K1_s4[2-6].json"]),
 ("cyl no-SSL mean_mlp @67.5k (exp_04 neuronic)", "fa eval, online", 8, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_67500_online_K8_s4[2-6].json"]),
 ("cyl no-SSL mean_mlp @67.5k (exp_04 neuronic)", "fa eval, EMA", 1, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_67500_ema_K1_s4[2-6].json"]),
 ("cyl no-SSL mean_mlp @67.5k (exp_04 neuronic)", "fa eval, EMA", 8, ["outputs_FLAC/exp04n_meanpoolmlp_import/*exp04n_67500_ema_K8_s4[2-6].json"]),
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
  ["outputs_FLAC/exp11_VANL/**/*exp11_VANL_q9_S40000_s4[2-6]_K1.json"]),
 ("fa-recipe vanilla VANL @40k (exp_11 baseline)", "vanilla eval (batched-era)", 8,
  ["outputs_FLAC/exp11_VANL/**/*exp11_VANL_q9_S40000_s4[2-6]_K8.json"]),
 ("C4L @40k re-measured at Q (Q9 fa side)", "fa eval (batched)", 1,
  ["outputs_FLAC/exp11_C4L/**/*exp11_C4L_q9_S40000_s4[2-6]_K1_fa_invariant_a4.json"]),
 ("C4L @40k re-measured at Q (Q9 fa side)", "fa eval (batched)", 8,
  ["outputs_FLAC/exp11_C4L/**/*exp11_C4L_q9_S40000_s4[2-6]_K8_fa_invariant_a4.json"]),
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


def is_exp11_row(patterns):
    """Does this row's evidence come from the exp_11 orbit sweep?"""
    return any("exp11_" in p for p in patterns)


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


def validate_exp11_cell(files, repo_root=None, validator_path=None):
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
        rows, problems = V.validate_cell(files, arm=first["arm"], step=first["step"],
                                         k=first["K"], contract="table",
                                         verify_hashes=True)   # item 3: never trust the sidecar
    except Exception as exc:
        return False, [f"validation raised {type(exc).__name__}: {exc}"]
    return (not problems), problems


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


def render_row(label, proto, K, files, repo_root=None, validator_path=None):
    """``(markdown_line, blocked)`` for one row, gating exp_11 evidence."""
    if is_exp11_row(files) or any("exp11_" in os.path.basename(f) for f in files):
        ok, problems = validate_exp11_cell(files, repo_root=repo_root,
                                           validator_path=validator_path)
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

    rendered, blocked_rows, exp11_status = [], [], {}
    for label, proto, K, pats in ROWS:
        files = sorted(set(sum((glob.glob(os.path.join(root, p), recursive=True) for p in pats), [])))
        proto = protocol_label(proto, is_exp11_row(pats), evidence_ready)
        line, blocked = render_row(label, proto, K, files, repo_root=root,
                                   validator_path=validator)
        rendered.append({"label": label, "proto": proto, "K": K, "n": len(files),
                         "line": line, "exp11": is_exp11_row(pats)})
        if blocked:
            blocked_rows.append(f"{label} (K={K})")
        if is_exp11_row(pats) and files:
            exp11_status[(label, K)] = not blocked

    # --- the two-K gate is TRANSACTIONAL, not advisory -----------------------
    # An exp_11 label lands as a PAIR or not at all: a lone K=8 row invites a
    # comparison against rows that carry both. On failure the affected label's
    # rows are rewritten as WITHHELD (no number can leak) and the write is
    # refused outright unless the operator explicitly accepts a partial table.
    two_k = check_two_k_coverage(exp11_status)
    if two_k:
        bad_labels = {p.split(":")[0] for p in two_k}
        for r in rendered:
            if r["exp11"] and r["label"] in bad_labels:
                r["line"] = withheld_row(r["label"], r["proto"], r["K"], r["n"],
                                         "incomplete two-K update: this row publishes only "
                                         "with its K=1 and K=8 partner")
    lines += [r["line"] for r in rendered]
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
