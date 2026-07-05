# Plan — exp_04_warmup_unblock (Adam-transient test → pipeline resumption)

**Author:** Fable 5 (Planner) · **Coder:** Opus 4.8 max (TDD) · **Reviewer:** Codex gpt-5.5 xhigh · **Date:** 2026-07-05
**Status:** AWAITING plan review + Yixun approval.

## 0. Hypothesis chain (from exp_03)

Fine-tuning the released `FLAC_EMA.ckpt` is destructive to T60/EDT/C50 (6–62σ) while preserving retrieval, across two audited control recipes. Remaining prime suspect: **fresh Adam second-moment transient** (no optimizer state released; early bias-corrected steps are outsized at any constant lr). Test: identical R1b recipe + **linear lr warmup 0 → 5e-6 over 200 optimizer steps**, then constant. One variable changed vs R1b (β₂ untouched — one variable at a time).

## 1. Code to write (TDD; one small cycle)

### `finetune_cond.py` (+~35 lines)
- `warmup_lr_factor(step, warmup_steps) -> float` — pure: `min(1.0, (step + 1) / max(1, warmup_steps))`.
- `WarmupLR(pl.Callback)` — `on_train_batch_start`: sets every param-group lr to `target_lr * warmup_lr_factor(trainer.global_step, warmup_steps)`; after warmup it is a no-op writing the constant target (idempotent; coexists with the removed-scheduler design — no scheduler object involved).
- `--warmup-steps` (int, default 0 = exactly current behavior); threaded through `finetune()`/`build_trainer_kwargs` (callback appended when > 0); recorded in the recipe echo line.

### `src/tests/test_finetune_cond.py` (+~45 lines, RED first)
| Test | Pins |
|---|---|
| `test_warmup_lr_factor` | 0→1/200 at step 0; 0.5 at step 99; 1.0 at ≥199; warmup_steps=0 ⇒ 1.0 always |
| `test_warmup_callback_sets_lr` | fake trainer/optimizer: lr follows target×factor at steps {0, 100, 200, 5000}; multiple param groups |
| `test_warmup_accumulation_semantics` *(plan-review finding 2)* | 32 repeated `on_train_batch_start` calls at `global_step==0` keep lr = target×1/200 (no per-micro-batch advance); 32 more at `global_step==1` give target×2/200 — kills the micro-batch-counter bug class under accum 32 |
| `test_warmup_default_off` | parser default 0; build_trainer_kwargs/callback list unchanged when 0 (R1b behavior byte-identical) |
| `test_warmup_recorded` | recipe echo / run config includes warmup_steps |

Per-round Codex review (marker `warmup`) after the cycle; round closes before any launch.

## 2. Runs (staged, each gated; all full-split per announcement 01; commands land in `_command.md` at launch)

| # | Run | Recipe | Gate |
|---|---|---|---|
| W1 | **warmup control** — vanilla, R1b recipe + `--warmup-steps 200` (batch 4×32=128, 625 steps, lr 5e-6, seed 42) | ~3 h | eval K∈{1,8} × seeds 42–46, exp_01 protocol → **pre-registered 2σ gate (same numbers as exp_03)** |
| W1-probe | 10-step fit probe first (same as exp_03 C4 discipline) | minutes | ≥5 steps, finite loss |
| **W0 (conditional null control, plan-review finding 1)** — runs ONLY if W1 FAILS: R1b recipe, `--lr 0`, 625 steps (BN running buffers still mutate in train mode) | ~3 h | W0 PASS ⇒ train-loop/BN/export alone are innocent → transient/lr attribution stands but warmup was insufficient; W0 FAIL ⇒ BN-buffer mutation (or loop/export) is destructive by itself → entirely different root cause, also retroactively contaminating R1/R1b |
| W2 | **fa_invariant fine-tune** — identical W1 recipe incl. warmup | ~9.6 h | launched ONLY if W1 passes |
| W3 | W2 evals — K∈{1,8} × 5 seeds, `--cond-method fa_invariant --cond-autocast bf16` | ~2.2 h | **H3**: within 2σ of exp_01 at both K |
| W4 | rotation sweep on W2 @ K=1: α ∈ {0, 90, 180, 270, 45} + `--store_predictions` + comparator | ~1.5 h | **H1**: rot0 ≡ 0.0; C₄ ≤ bf16 floor (re-measured & re-registered on W2 BEFORE reading W4); 45° reported |
| W4b | K=8 spot check α ∈ {0, 90} + comparator | ~30 min | H1 at K=8 |
| — | **H2**: Metric-2 flatness across α ∈ C₄ from W4/W4b per-angle JSONs | free | flat within 2× exp_01 single-eval noise floor |

**Stop rules (revised per plan review):** W1 FAIL → run W0 null control for attribution, then stop (no W2) and analyze. **Marginal pass defined precisely:** all primary metrics ≤2σ with at least one ≥1.5σ ⇒ PAUSE for Yixun's decision; clear pass (all <1.5σ) ⇒ auto-launch W2. Interpretation language: W1 PASS demonstrates "warmup/lower-early-lr repairs the control", not uniquely "Adam second-moment transient proven" (integrated-lr confound: ~525.5 full-lr-equivalent steps vs 625 — accepted for a repair recipe; a matched-area constant-lr discriminator is deferred unless mechanism attribution becomes load-bearing). Any launch only after the warmup round closes. GPU 1 untouched; cotenant job respected (~26 GB envelope, batch 4 resident).

## 3. Acceptance criteria

Pre-registered in the notebook before W1 launch; identical gate numbers to exp_03 (exp_01 means/stds). exp_04 succeeds as an *experiment* on either W1 outcome — the hypothesis is falsifiable; the *project* advances to H3/H2 completion only on PASS.

## 4. Risks

- Warmup may be necessary-but-insufficient (transient + something else): partial recovery ⇒ documented, still a stop after W1 (no recipe fishing beyond the one declared variable).
- 200 steps is ⅓ of the 625-step budget at reduced lr: effective sample exposure at full lr shrinks ~16%; accepted (budget parity maintained on total steps; noted for interpretation).
- If W1 passes only marginally (2–3σ), Planner flags for Yixun rather than auto-launching W2.
