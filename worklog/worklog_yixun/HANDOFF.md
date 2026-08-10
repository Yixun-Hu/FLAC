# HANDOFF.md — working-memory contract for the next session

Assume the reader has NO memory beyond the repo + `master_experiment_tracker.md` + `issue_report.md` + this file. Updated at every handoff trigger (CLAUDE.md protocol): "handoff"/"new session"/"wrap up"/`/compact` **or a model change / model-limit swap**.

**Last updated:** 2026-08-10 ~18:00 EDT, authored by **Opus 5 (1M, max effort)** — Yixun `/model`-switched Fable 5 → Opus 5 max, which is this refresh's trigger. **Nothing of ours is running.** Working tree clean; branch `check-equivariance-necessity` synced (pull-rebased onto the cluster session's exp_11 commits at `96aca57`).

> ⚠️ **The main session ALTERNATES models mid-session** (established 2026-07-16; `issue_report.md` §8). The harness fails over between turns on its own, not only on `/model`, and **an incoming model has NO memory of the other model's turns in the same session.** Treat these four docs as the live intra-session channel: write state the moment it changes, and re-verify anything time-sensitive (`pgrep -af train.py`, `nvidia-smi`, newest ckpt mtime) before quoting an ETA. A stale in-flight entry WILL produce a wrong wait-time report.

> ⚠️ **This branch has CONCURRENT WRITERS.** A second Claude session on the cluster works exp_11 in this same repo/branch and pushes frequently. **Always `git pull --rebase origin check-equivariance-necessity` before committing**, and never rewrite a file it owns (its exp_11 folder, its rows in `gen_model_comparison.py`). Two rebase conflicts have already occurred and were resolved by regenerating rather than choosing a side.

## Role map (current)
- **Main session model:** **Opus 5 (1M, max effort)** as of 2026-08-10; previously Fable 5 / Opus 4.8 alternating. Plans, analyzes, drives runs. Role-attribution rule stands: whenever a non-Fable model fills a Fable seat (esp. analysis), flag the model in the artifact by-line.
- **Coder subagent:** Opus 5, max effort (per Yixun 2026-07-25).
- **Reviewer:** OpenAI Codex `gpt-5.6-sol`, xhigh — `~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh --output-last-message <file> "<prompt>" < /dev/null` (stdin MUST close with `< /dev/null`). Auth broke 2026-08-08 (401) and was restored by Yixun the same morning; **declared fallback = Claude Opus 5 max**, which reviewed all of exp_13 and caught its decisive bug. `-s read-only` is NOT an environment guarantee (see issue_report §10) — always forbid installs explicitly in the prompt.

## Program state — exp_01…exp_13 CLOSED here; exp_11 ACTIVE on the cluster

Both project goals are **achieved and closed**. Three checkpoints are confirmed (5 eval seeds, full published unseen split 6,337/17, per-scene mean, EMA):

| Flavor | Checkpoint | Eval protocol | K=8 (T60/C50/EDT/R@1) | Standing |
|---|---|---|---|---|
| **Anchor** (all-round best) | `outputs_FLAC/exp07_P1/FLAC_exp07_P1/exp07_P1/checkpoints/epoch=19-step=87500.ckpt` | vanilla | 8.293 / 0.9660 / 35.951 / 6.959 | **8/8 cells SUPERIOR-or-EQUIV vs released Table-1** — the maximum project goal |
| **Equivariant** | `outputs_FLAC/exp09_Fw/FLAC_exp09_Fw/exp09_Fw/checkpoints/epoch=20-step=95000.ckpt` | **`--cond-method fa_invariant`** | 8.465 / 0.9582 / 37.497 / 6.924 | exact C₄ conditioning; 4 SUP + 1 EQUIV + 2 NONINF + 1 OUT vs released (registered tier NEGATIVE on the strict anchor-preservation gate) |
| **C50/retrieval flavor** | `outputs_FLAC/exp13_DT/…/epoch=…-step=93750.ckpt` | vanilla | 9.026 / 0.9288 / 37.171 / 6.950 | 6/8 SUP-or-EQUIV; K=1 C50 0.9951 (first sub-1.0 in the lineage); T60 the concession |

Everything else is per-experiment folders under `worklog/worklog_yixun/exp_*/` (query → plan → reviews → params → command → results → analysis → HTML → commits log). Superseded narrative lives there; it is deliberately NOT inlined here any more.

**exp_11 (fa_orbit) was finished by another agent (Yixun, 2026-08-10).** Cluster-owned work — this box neither tracks nor drives it; read its folder if you need its numbers.

## In flight right now
- **Ours: NOTHING.** No training, no evals, no monitors.
- **Yixun's own runs, co-tenant on both GPUs — DO NOT TOUCH:** `exp12A_c3c4` and `exp12C_ray12`, launched 2026-08-08 17:15, single-GPU each, `--batch-size 32 --accum-batches 2 --max-steps 67500`, wandb. **They run from a SEPARATE CHECKOUT** `~/codespace/exp-12-arms`, not this worktree — that is why their `worklog/worklog_yixun/exp_12_arms/` configs do not exist here. **Numbering collision:** in *this* repo `exp_12` = the cluster's mem_probe (closed); those arms are a different experiment line in a different clone. Sibling checkouts also exist for exp-08-cylvit-pe-cnn, exp-09-cyl-dinov3-no-ssl, exp-10-cyl-distill. **Before assuming any `train.py` is ours, run `readlink /proc/<pid>/cwd`.**
- GPU state at handoff: GPU0 5.9 GB used / 42.7 free, GPU1 14.9 / 33.6, both ~100% util from those runs. Plenty of room to co-tenant a 2-GPU DDP job (~16 GB/rank with grad-ckpt) if Yixun asks.

## Awaiting Yixun (nothing blocks autonomous work)
1. **Write-up target unclear.** "Paper columns" was the assistant's framing, not a stated goal — no paper artifact exists in this repo or the siblings. Concretely the open question is which of the three confirmed flavors and which comparison rows to feature *if* this program is written up. Yixun to say whether there is a target at all.
2. **~~Metrics consolidation~~ → delegated** (2026-08-10): a cluster agent will commit the model JSONs; afterwards just re-run `gen_model_comparison.py` here.
3. **~~exp_11~~ → closed by another agent** (2026-08-10). **Cluster work is not this box's concern** — do not track or drive it.

## Load-bearing technical facts (each of these cost real time to learn)
- **Eval-protocol flags are part of the experiment, not a default.** `eval_FLAC.py --cond-method {vanilla,fa_invariant}` must match how the checkpoint was trained. A mismatch produces plausible-looking but catastrophic numbers: the fa-trained B-F@40k reads 8.202/0.978/38.79/R5.39 under fa eval and **10.652/2.082/80.86/R0.68** under vanilla eval. This caused exp_09's protocol error and a retracted exp_07 conclusion. Put the flag in every launch/screen manifest. Companions: `--rotate-deg` (C₄ sweeps + 45° control), `--frame-avg-angles`, `--cond-autocast bf16`.
- **PL checkpoints serialize LR-scheduler hyperparameters, and they CLOBBER the config on warm resume.** `InverseLR` stores `inv_gamma/power/warmup/final_lr` as instance attributes → `LRScheduler.state_dict()` captures them → `load_state_dict` is `self.__dict__.update`. So **changing a schedule in the model config does nothing on resume** (measured: the intended 1.28e-5 tail silently stayed at 4.77e-5). To change a schedule mid-run you must rewrite the checkpoint: `src/tools/retune_lr_state.py` (copy-only, re-derives the target lr from the ckpt's own state and refuses on mismatch). Symmetric tool: `src/tools/strip_optimizer_state.py` for a fresh-Adam resume — **keep the optimizer entry and clear only `state`**; an absent `optimizer_states` key raises `KeyError` on restore, and an *empty list* silently runs the first step at the step-0 warmup lr (5e-7, ~96× under schedule).
- **Late checkpoints are draws from an oscillating band, not points on a curve.** InverseLR holds lr ≈ 4.8e-5 forever, so adjacent checkpoints swing ~±0.5 T60. Three program conclusions were distorted by treating single draws as trajectories (B-V's band-max 67.5k endpoint; fa-scratch's band-best 40k spike; fa-scratch's band-worst 67.5k endpoint). Always select over a window with a pre-registered rule and confirm on held-out eval seeds. exp_13 proved a decaying tail halves the band width but converges to a *different metric trade point* — it does not reproduce a wide-band best draw.
- **Launcher family** (all reviewed, guard-tested): `exp_07…/p1_ddp_launch.sh` → `exp_09…/f_arm_launch.sh` → `exp_10…/bf_resume_launch.sh` → `exp_13…/dtail_launch.sh`. Shared gates: conda+PL-version asserts, parsed-object config contract, sha-pinned resume lineage with INITIAL/RESTART modes, per-GPU free-VRAM floor, wandb identity gate, DINOv3 pin, df floor. Copy the newest one when starting an arm; each ships a `*_guardtests.sh`.
- **Living results table:** `worklog/worklog_yixun/model_comparison.md`, regenerated ONLY by `gen_model_comparison.py` (glob-spec rows aggregated from raw per-seed JSONs; single-seed screens structurally excluded). **Announcement 04 mandates regenerate + commit + push on every model-results update.**

## `CLAUDE.md` is now TRACKED (2026-08-10, Yixun's call, `02dbb4c`)

It used to be gitignored and machine-local; the ignore line was removed and the file committed, so guidance propagates. ⚠️ **The other checkout will hit a one-time pull failure** (*"untracked working tree file would be overwritten by merge: CLAUDE.md"*) because it holds an untracked copy. Remedy there: `mv CLAUDE.md CLAUDE.md.local && git pull --rebase`, fold local-only guidance into the tracked file, delete the backup. Standing rules that must bind every machine still belong in `worklog/worklog_yixun/announcement/*.md` (e.g. `05_eval_protocol_flags.md`).

## Environment
- conda env **`flac` for everything** (`source ~/miniconda3/etc/profile.d/conda.sh && conda activate flac`) — torch 2.7.0+cu126, PL 2.1.0, flash-attn active in the DiT. Env-bridge check (2026-07-20) proved eval results identical to 4 decimals across the old rir2rir env, so historical labels are provenance only.
- wandb key for **yh4742@princeton.edu** sits in `~/.bashrc` BELOW the interactive guard → non-interactive shells must `eval "$(grep -E '^\s*export\s+WANDB_API_KEY=' ~/.bashrc | tail -1)"`; plain `source ~/.bashrc` silently keeps the wrong key. Launch scripts self-extract.
- 2×A6000 48 GiB (paper used H100 80 GiB). Released recipe: eff-batch 64, accum 1, AdamW 5e-5/(0.9,0.999)/wd 1e-3, InverseLR(1e6,0.5,0.99), EMA on, bf16, 67,500 steps. Our parity recipe adds DDP 32/GPU×2×accum1 + **SyncBN** (BN batch 64 — accumulation never feeds BN stats) + ViT gradient checkpointing (≈16 GB/rank).
- `CUDA_VISIBLE_DEVICES=1` renumbers to cuda:0 inside the process — tracebacks naming "GPU 0" may mean GPU 1.

## Standing constraints
- Full published eval configs only (unseen = all 6,337 items / 17 rooms); never subsample or create new eval configs.
- TDD for new code (`src/tests/`); universal review loop before any round closes — including one-off scripts.
- Commit + push before long runs; never edit a running script; stop-and-ask on gate failure; wait-time reporting on every response with in-flight runs.

## Baselines (exp_01, full split — the parity targets)
- **K=8:** T60 8.609±0.012 / C50 0.9682±0.0030 / EDT 37.10±0.07 / R@1 7.06±0.10 / R@5 19.45±0.16 / R@10 27.43±0.22
- **K=1:** T60 9.969±0.039 / C50 1.0460±0.0064 / EDT 39.95±0.37 / R@1 6.83±0.22 / R@5 19.08±0.12 / R@10 26.98±0.17

## Deliverables a fresh session should know exist
`worklog/worklog_yixun/`: `model_comparison.md` (living table) · `trajectories_all_arms.{png,pdf,html}` (K=8, six metrics, five arms) · `trajectories_all_arms_K1K8.{png,pdf}` (12-panel, both K) · `A6000_METRICS_SHA256SUMS.txt` (raw-JSON manifest) · `announcement/04_model_comparison_table.md`.

## Gotchas
- Never `git amend` a commit that records its own SHA.
- `worklog/**/*_train.log` > 100 MB is gitignored — commit a filtered copy.
- `unwrap_model.py` still imports `stable_audio_tools` (upstream); adapt before use.
- The handoff hook (`.claude/hooks/model_change_handoff.py`) is the **detector/archiver only** — the live model still authors this refresh.
