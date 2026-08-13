# Plan — exp_14 fa_drawshare (does the chunk plan change FA training?)

**Author:** main session (Opus 5, max effort) · **Rev 3**, 2026-08-13 — every finding of `fa_drawshare_codex_plan_review.md` applied · **Status:** re-review → **Yixun's final go before launch**.
**Yixun's decisions:** "做，要把因果钉死" → after the review showed 12 days cannot support a causal claim, he chose **option B**: run the scaled-down version and scope the claim accordingly. Sequencing: **顺序跑**.

## 1. What this experiment can and cannot say (fixed up front)

**One paired training seed (42) only.** Five eval seeds estimate generation noise for two fixed checkpoints; they say nothing about training-seed variability. Therefore the registered claim is:

> **the effect of the operational chunk plan on the seed-42 training trajectory** — suggestive evidence, NOT a general causal result.

A general causal claim would need ≥3 paired training seeds (~36 d at our rate) and is explicitly out of scope. **A null result is likewise scoped:** without an equivalence CI inside a pre-registered margin, "no effect observed for this seed pair" ≠ "no effect". And because changing the cap also changes forward count, GEMM shapes and global RNG consumption, a positive result identifies **the chunk plan as an operational policy**, not RoPE draw covariance in isolation (that would need the fixed-shape control with prescribed draw mappings — out of scope here).

## 2. Design — one delta, and it now hits the implicated regime

| arm | `frame_avg_max_fwd_samples` | angles/chunk at micro-32 (C4) | corresponds to |
|---|---|---|---|
| **DS-PA** | 32 | **1** | per-angle draws = exp_07/exp_10's B-F (July path) |
| **DS-CS3** | 96 | **3** | fully shared = **exp_11's C4L regime** (`96 // 32 = 3`) |

Review correction applied: the earlier plan wrongly said micro-32 tops out at 2/3. Cap 96 reaches 3/3 at the same cost, so the two arms bracket the whole range. **DS-CS3 reproduces exp_11's draw-sharing TOPOLOGY (one chunk holding all three non-identity angles), not its configuration**: exp_11's micro-8/cap-64 issues a 24-row GEMM while micro-32/cap-96 issues a 96-row one, so walltime, memory and numerics differ.

Everything else pinned to the exp_07 B-F recipe: `FLAC_AR_BF.json` (C4 `fa_invariant`), DDP micro-32 × 2 GPUs × accum 1 (eff 64), **SyncBN (BN=64)**, ViT grad-ckpt on, seed 42, bf16, ckpt/2,500, wandb, env `flac`, **40,000 steps**, from scratch. Eval flags declared explicitly per announcement 05 (`--cond-method fa_invariant --frame-avg-angles 0,90,180,270 --cond-autocast bf16 --rotate-deg 0`), chunk plan declared per announcement 06.

## 3. Implementation

1. **Config key, not an environment override** (review H4 — an ambient env var is invisible to the checkpoint and leaks across sessions). Add validated `training.frame_avg_max_fwd_samples` (default **64**, so every existing recipe is byte-identical), carried through the real handoff at **`src/training/factory.py:152`** into the wrapper, read at the call site `src/training/diffusion.py:469`, and threaded as a new keyword on `invariant_conditioning` — **no global mutation**.
   **Resume is NOT automatically safe (re-review 4).** `train.py:21` embeds `model_config`, but on resume the wrapper is rebuilt from the *current* JSON (`train.py:160`) before PL loads the checkpoint (`:230`) — so a changed cap would silently take effect. The launcher therefore performs a **fail-closed comparison of the embedded config against the current one**, aborting on any mismatch of the chunk plan.
   **Evaluation cap is a separate, pinned quantity (re-review 4).** `eval_FLAC.py:1005` calls `invariant_conditioning` with no cap and records the module default (`:590`). **Both arms are evaluated under ONE common evaluation cap, pinned to `64` (the module default — identical to every historical eval in the record)**, passed explicitly as `--frame-avg-max-fwd-samples 64` on every eval command and recorded in every metrics record and the params file, separately from each arm's training cap; the comparison is between training policies, never between eval policies.
   **TDD (`src/tests/`):** default → 64, behaviour unchanged; explicit value honoured; non-integer / bool / <1 / cap < micro-batch → fail closed; partitions at micro-32 for caps 32/64/96 → 1/2/3 angles; propagation through `factory.py` **and** direct-wrapper construction, with no module-global mutation; **resume mismatch rejected**; **common-cap evaluation enforced**; applied-cap provenance present in outputs; full existing suite green.
   ⚠️ **Shared-code caution:** this edits `src/data/yaw_rotation.py`, `src/training/diffusion.py`, `src/training/factory.py` and `eval_FLAC.py`, which the cluster session also runs. Default-preserving by construction, but it must land at a **safe boundary — no cluster job mid-flight against these paths — or be developed in an isolated pinned worktree and merged deliberately**; announcing is not sufficient.
2. **Fit probe before committing 12 days** (review B1): cap 96 puts 96 samples in one ViT chunk vs 64 today (~1.5× activation memory). 15-step real DDP fit at micro-32 × 2 with grad-ckpt, VRAM sampled. **If cap 96 does not fit, stop and report — do not silently fall back to cap 64**, which would no longer test exp_11's topology. Record steady-state cap-96 throughput during the probe and **rebase the ETD from it before launch** (the 0.079 steps/s figure is the cap-64 rate).
3. **From-scratch launcher** based on `bf_scratch_launch.sh` (review H6 — `f_arm_launch.sh` is resume-required and wrong for this), with distinct identities `exp14_DSPA` / `exp14_DSCS3`, the chunk plan echoed into the launch log and params file, plus the standard gates (env/PL, config contract, VRAM floor, wandb identity, DINOv3 pin, disk floor) and a resume path that re-asserts the arm's cap.

## 4. Readouts

- **DS1 (primary):** DS-CS3 − DS-PA at 40,000 steps, 5 eval seeds, both K. Report per-metric deltas with σ_c **and** the explicit one-training-seed caveat. Reference for magnitude: exp_11's reported reversal (T60 +0.366, EDT +4.180). If |DS1| is a small fraction of that, the chunk plan is not a sufficient explanation and the rung/topology remains the open suspect.
- **DS1b (trajectory, registered statistic):** mean of the 2,500-cadence screens over the last 10k steps (30,000–40,000 inclusive, 5 points) per arm — a band statistic, since single endpoints have misled this program three times.
- **DS2 (cross-era replication check, NOT a reproducibility floor — review H5):** DS-PA@40k vs exp_07 B-F@40k. Bundles source drift, RNG/data order, environment and calendar, so it is contextual only; a July-launch-commit parity audit accompanies it.
- **DS3 (contextual):** both arms vs vanilla P1@40k (5-seed on record) — does A4's conclusion hold under each chunk plan?
- **Screens:** every 2,500 steps, EMA/K=8/s42/full split, both arms.
- **Tiers:** EFFECT-OBSERVED (DS-CS3 worse beyond 2σ_c on T60 or EDT, seed-42 trajectory) / NO-EFFECT-OBSERVED (all six within 2σ_c; explicitly not an equivalence claim) / MIXED.

## 5. Sequencing, aborts, budget

DS-PA first (it also serves DS2), then DS-CS3 — sequential per Yixun; 2 GPUs cannot host both at BN=64. **Gate between the arms (re-review 6): DS-CS3 does not start until DS-PA passes a 40k admission/parity audit** — provenance, config identity, applied chunk plan, world size, BN, and the July-commit parity check. If parity fails, STOP and report rather than spend the second six days. *Metric* divergence from July is contextual and is NOT an abort. ~40,000 steps at ~0.079 opt-steps/s co-tenant ≈ **5.9 d/arm**, **~12 d total plus ~1.5 d for probes, screens, gates and reviews → ETD ≈ 2026-08-26**. Start after the GPUs are free of our own eval work. **Hard aborts only:** wrong SHA/config/cap/world-size/BN, OOM, non-finite loss, disk floor. **No metric-driven stopping** — the futility discipline that stopped B-F is deliberately not used here, because a "bad-looking" DS-CS3 curve is the hypothesis, not a failure.

## 6. Artifacts (SOP enumeration)

`plan_fa_drawshare.md` (this) · `fa_drawshare_yixun_query.md` · `fa_drawshare_worklog.md` · `fa_drawshare_codex_plan_review.md` (+ re-review) · `src/data/yaw_rotation.py`, `src/training/diffusion.py`, **`src/training/factory.py`** & `eval_FLAC.py` (cap threading + eval-cap pin) · `src/tests/test_frame_avg_cap_config.py` · `FLAC_AR_BF_DSPA.json` / `FLAC_AR_BF_DSCS3.json` · `dsarm_launch.sh` + `dsarm_launch_guardtests.sh` · `fa_drawshare_params_set_up.md` · `fa_drawshare_command.md` · per-round code reviews (`fa_drawshare_codex_code_r<N>_review.md`) + final `fa_drawshare_codex_code_full_review.md` · timestamped run/screen logs `fa_drawshare_<ts>_*.log` · `fa_drawshare_results.md` · `fa_drawshare_analysis.md` · `fa_drawshare_01_results.html` (+ `fa_drawshare_results_assets/` if any) · `fa_drawshare_codex_closure_review.md` · `commits_fa_drawshare.md`.
