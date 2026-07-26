# exp_06 ADDENDUM — matched D1 comparison vs P1 (records frozen 2026-07-26, PRE-P1)

**Authorization:** Yixun, 2026-07-26, verbatim: "go for the matched D1 comparison after P1
completes." Trigger: P1 (BVp1) final ckpt `epoch=*-step=67500.ckpt` in the FLAC checkout
(completion monitor armed). exp_06 remains CLOSED; this fulfills its registered PENDING item.

## What P1 is
FLAC-owned BVp1 (`/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_07_fa_scratch_claude/
FLAC_AR_BVp1.json`): the vanilla-DINOv3 baseline arm under the SAME matched protocol as the
no-SSL arm (67,500 steps, DDP+SyncBN, eff-batch 64, seed 42; FLAC-side p1-kit review: semantic
BV copy, only grad-ckpt deltas vs BV; use_ema true; no cond_method → native vanilla path).

## Import protocol (ZERO writes into the FLAC checkout)
On trigger: COPY the final ckpt + FLAC_AR_BVp1.json into `p1_matched_d1/p1_import/`; record
sha256 of BOTH at copy time in `p1_import_pins.txt` (format: `<sha256>  <filename>`); evals run
entirely inside this worktree; outputs land next to the COPIED ckpt (worktree, gitignored).
Precondition (driver-enforced): the P1 training process (`pgrep -f exp07_P1`) must be GONE.

## Eval protocol — exp_01 convention VERBATIM (P1's native path)
`p1_eval_driver.sh <K> <SEED> <GPU>`: eval_FLAC.py on the copied ckpt/config with
`--cond-method vanilla` (explicit; = eval_FLAC default = exp_01's released-repro convention;
the fa_invariant mandatory-flags rule applies ONLY to the cylindrical fa_invariant-trained
arm), default `--cond-autocast` (exp_01 had no flag), `--steps 1 --cfg-scale 1.0`,
dataset configs unseeneval[_1].json, seeds 42–46 × K∈{1,8} = 10 runs, eval names
`exp09_P1D1_K<k>_s<seed>`. Driver gates: frozen MIN_FREE_MB (exact-match to
c1_frozen_min_free.txt), free-VRAM check, per-invocation sha256 re-verification of the copied
ckpt+config against `p1_import_pins.txt`, refuse-while-P1-training-alive, external log dir,
DRY_RUN mode. The exp-09 pin gate does NOT apply (BVp1 is intentionally not an exp-09
registered config); its role is replaced by the sha256 import pins. Code provenance: this
worktree's eval code (exp-09 B-code-reviewed; vanilla path additive-untouched per those
reviews) — the eval log records the worktree HEAD.

## REGISTERED convention asymmetry (caveat)
Each arm is evaluated under its OWN registered convention: no-SSL = fa_invariant[0.0] + bf16
cond-autocast (its training/eval convention, mandatory per Stage-B); P1 = vanilla + default
autocast (its native convention = exp_01's, which also produced the contextual stats). The
comparison is arm-protocol-faithful, not autocast-matched; registered here pre-review.

## Verdict path
`build_p1_matched.py` (after the 10 evals): computes 5-seed mean/std per (K, metric incl.
R@1) FROM THE RAW P1 eval JSONs (never by hand) → writes `references_matched.json` = the
committed template with `control_stats` filled + `control_name: "P1"`,
`mode: "matched_control"`, the IDENTICAL expect block, and the d2 blocks copied verbatim
(required by the adapter's schema; only --d1 is passed, so no d2 input is consumed) →
adapter `--references references_matched.json --d1 ../d_records/d1_manifest.json` (the SAME
committed no-SSL measured manifest — unchanged data) → fresh `verdicts_matched_<ts>/` →
GATING `d1_parity` verdict (exp_07 verbatim: equivalence ≤1σ_c, non-inferiority ≤2σ_c,
sc==0 ⇒ OUTSIDE) → `aggregate_gate --require d1_parity` → Codex matched-review →
exp_06 worklog/tracker ADDENDUM (D3 SSL verdict resolved per plan §4: non-inferior ⇒
"no-SSL sufficient at this protocol (this β=0 initialization)"; fail ⇒ gap size reported,
SSL/distill rung stays a motivated hypothesis).
