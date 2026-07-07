# Plan — exp_08_fa_matched (Route A: matched fine-tune comparison, fa_invariant vs vanilla)

**Author:** Fable 5 (Planner) · **Coder:** none expected (config/flags only; TDD round only if review finds a gap) · **Reviewer:** Codex gpt-5.5 xhigh · **Date:** 2026-07-07
**Status:** AWAITING plan review + Yixun approval.

## 0. Question

At matched recipe and matched, fully-characterized fine-tune regression (exp_03–06), what is frame averaging's *marginal* effect — and does the fine-tuned fa_invariant model pass the cylindrical sanity check exactly (the project's minimum goal, on a trained model)?

## 1. Arms

| Arm | Status | Recipe |
|---|---|---|
| **A-V** (control) | **EXISTS = exp_05 V1′** (`outputs_FLAC/exp05_V1p_freezebn_ft/FLAC_exp05_V1p_freezebn.ckpt`) + its 10 gate evals | vanilla, `--freeze-bn`, lr 5e-6 const, batch 4×32 (eff. 128), 625 opt steps, seed 42, bf16-mixed, clip 0.0, use_ema off |
| **A-F** (method) | **TO RUN** on GPU 1 | identical + `--cond-method fa_invariant` |

Validity of reusing V1′ as the control (review-verified, wording corrected): V1′ trained at code state `51b7486` (post `5d1c64c` freeze-bn round); `git diff 5d1c64c..HEAD -- finetune_cond.py` shows only `--lr-schedule`, the warmup/schedule guard, and echo changes — **behavioral no-ops for a constant-lr, warmup-0, freeze-bn run** (flag-gated, pinned by the recipe tests). Claim: **recipe-equivalent reuse** (not bit-identical rerun — CUDA/dataloader nondeterminism precludes that). Reused artifact, exp_05 command, and this code-diff proof are recorded per the review's SOP-legality conditions.

**Recipe identity over hardware identity:** A-F runs batch 4 × accum 32 exactly like V1′ (recipe identity beats exploiting GPU 1's larger free memory; BN is frozen in both arms so micro-batch BN effects are moot, but padding-mask micro-averaging and data order must match).

## 2. Runs (GPU 1; commands to `_command.md` at launch; ~15 h total)

| # | Run | Est. | Gate/readout |
|---|---|---|---|
| M0 | EMA-on N/A — probe: fa_invariant fit/throughput 10 steps on GPU 1 (batch 4×32) | 10 min | ≥5 steps, finite loss, throughput anchor for ETA |
| M1 | **A-F fine-tune**: 625 opt steps | ~7.5 h (0.3× vanilla rate) | loss finite; clean export |
| M1.5 | **A-V bf16 eval mirror** (review Medium fix): rerun the 10 gate evals on the EXISTING V1′ ckpt with `--cond-autocast bf16` — removes the eval-precision confound from the marginal comparison at zero training cost | ~2.2 h | this row (not the exp_05 fp16-default row) is the H-A1 comparator |
| M2 | A-F gate evals: K∈{1,8} × seeds 42–46, `--cond-method fa_invariant --cond-autocast bf16`, full split | ~4 h (M0 probe updates ETA; K=8 fa evals may run long — 36 ViT forwards/batch) | **H-A1 (non-inferiority):** A-F within 2× combined 5-seed σ of the M1.5 mirror per T60/C50/EDT at both K (R@k advisory); superiority reported descriptively unless coherent across cells |
| M3 | bf16 Metric-1 floor re-registration on the A-F ckpt (rung-b-style, C₄, K=1&8, fixed noise) | ~20 min | floor logged in notebook BEFORE M4 is read |
| M4 | Rotation sweep K=1: α ∈ {0, 90, 180, 270, 45} + `--store_predictions` + comparator | ~1.5 h | **H-A2 (=H1):** Metric-1 rot0 ≡ 0.0 exactly; C₄ ≤ registered floor; 45° reported |
| M4b | K=8 spot α ∈ {0, 90} + comparator | ~1 h | H-A2 at K=8 |
| M5 | **Training-seed sensitivity pair (Yixun point 1, option b-screen):** retrain BOTH arms at seed 43 (A-V-s43 ~3 h, A-F-s43 ~7.5 h) and screen each (K=8, eval-seed 42, full split, bf16 mirror protocol) | ~11.5 h | measures training-seed Δ per arm directly; pre-registered use: if per-arm |Δ_train-seed| on T60/EDT is comparable to or larger than the seed-42 |A-F − A-V| difference, the H-A1 verdict is DOWNGRADED to "indeterminate at single-seed resolution" regardless of band; if ≪, the single-seed caveat is discharged with evidence |
| — | **H-A3 (=H2):** Metric-2 flatness across α ∈ C₄ from M4/M4b per-angle JSONs | free | flat within 2× exp_01 single-eval noise floor |

Context rows in results: released baseline (exp_01), zero-shot fa (exp_03 R0), A-V under both eval precisions (exp_05 fp16-default row for continuity; M1.5 bf16 mirror as the H-A1 comparator). All full-split (announcement 01). **Total ≈ 28.5 h including M5** (17 h without it; M5 runs last so all primary verdicts land first).

## 3. Pre-registered interpretations

- **H-A1 pass + H-A2 pass:** minimum project goal achieved on a fine-tuned model: exact C₄ invariance at zero accuracy cost relative to a matched control. exp_07 then becomes about the *maximum* goal only (Yixun decides post-exp_08, per the standing instruction).
- **H-A1 fail (A-F materially worse than A-V):** FA's information loss is real at this scale — the 4-view average costs accuracy; exp_07's from-scratch question changes character (can training-from-scratch absorb the averaging?), and the cylindrical-sanity claim stands on H-A2 alone with the cost quantified.
- **H-A2 fail:** implementation-level regression (would contradict exp_03's proofs) → stop, bisect before anything else.

## 4. Risks

- fa_invariant throughput uncertainty on GPU 1 (M0 probe anchors it; wall-clock table updated at launch).
- V1′-as-control code-state assumption — review-verified; if the review rejects it, A-V reruns (~3 h + 2.2 h, still cheap).
- Eval-time fa cost (4× conditioner) makes M2 the longest eval block (~4 h); accepted, no protocol shortcuts (announcement 01).
