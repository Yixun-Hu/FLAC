# HANDOFF.md — working-memory contract for the next session

Assume the reader has NO memory beyond the repo + `master_experiment_tracker.md` + `issue_report.md` + this file. Updated at every handoff trigger (CLAUDE.md protocol): "handoff"/"new session"/"wrap up"/`/compact` **or a model change / model-limit swap**.

**Last updated:** 2026-07-15 (trigger: model change).
**This handoff's trigger:** Fable 5 (previous main session) reached its usage limit → **Opus 4.8 (1M context), max effort** took over as main session via `/model`.

## Role map (current)
- **Main session model:** Opus 4.8 (1M context), max effort — plans, analyzes, drives runs. *(Was Fable 5; Fable hit its limit.)* Role-attribution rule: work previously done by Fable is now done by Opus — in experiment **analysis** files, flag "authored by Opus, not Fable."
- **Coder subagent:** Opus 4.8, max effort (writes experiment/production code).
- **Reviewer:** OpenAI Codex `gpt-5.6-sol`, xhigh, codex-cli 0.144.1 — `~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh --output-last-message <file> "<prompt>" < /dev/null` (stdin MUST close with `< /dev/null`).

## Where we are — exp_07 phase 2 (B-V parity, Yixun Q5)
**Mandate (Q5):** "The B-V should at least get the same results as FLAC. Please achieve this."

- **P0 (DONE):** checkpoint selection alone CANNOT reach released parity (21-pt ≥20k K=8 seed-42 EMA curve). Best observed: EDT **38.29@60k** (vs released 37.10), R@1 **6.22@65k** (vs 7.06); T60/C50 reachable in-band. → systematic factor(s) remain.
- **P1 (APPROVED 2026-07-15 — "I approve P1"):** micro-parity B-V rerun. Plan `plan_bv_parity.md` review-clean (Codex gpt-5.6-sol REQUEST-CHANGES → all findings applied), committed `67b8fce`.
  - **P1a fit probe (~20 min):** vanilla-only ladder **64×1 → 32×2 → 16×4**, 15 opt steps, EMA on, 1-s VRAM sampler, record steady-state samples/s (re-anchors ETA). Pick largest fitting rung. Review estimate: 64×1 likely OOMs on 48 GiB; 32×2 likely fits.
  - **P1b train (~3.4 d, re-anchored by probe):** `FLAC_AR_BV.json` (byte-copy), largest fitting rung, `--max-steps 67500`, seed 42, EMA on, ckpt every 2500, `HF_HUB_OFFLINE=1`, DINOv3 pin gate pre-launch. Then same 10k screens (EMA+online) + ≥20k selection curve + 5-seed gate.
  - **Success tiers (pre-registered, plan §1):** PARITY (composite-rule ckpt confirmed on held-out eval seeds 43–46, **R@1 REQUIRED**) / STRONG (≥50% late-curve gap closure on BOTH EDT+R@1: **EDT ≤38.59, R@1 ≥6.51**) / DIRECTIONAL / NULL. Late-curve statistic = mean over S∈{55k,57.5k,60k,62.5k,65k,67.5k}. Baseline (8×8, same statistic): **EDT 40.087, R@1 5.960**.
  - **Abort discipline:** hard aborts only (OOM/NaN/divergence); numerical futility check no earlier than 50k.
  - **Control rule:** 8×8 B-V stays the ONLY B-F control (incomplete factorial). Never compare B-F-8×8 causally against B-V-at-larger-micro.

## In flight right now
- **Nothing running as of this write.** P1a probe about to launch on GPU 1.
- **GPU 1** held for the P1 sequential window (~3 weeks). Verify idle via `nvidia-smi` before every launch.
- **B-F on hold** per the parity mandate — does NOT launch without a fresh explicit Yixun go, AND only after the P1 parity outcome.

## Do-not-touch (other sessions' jobs)
- The **`FLAC_vanilla291k`** run and the **rir2rir** jobs belong to other sessions — leave untouched. 291k is a corroborating row only (data folder `single_channel_ir` ≠ `_1`, micro 16×4; not B-V-certifiable).

## Standing constraints
- Full published eval config only (unseen = all 6337 items / 17 rooms; never subsample or create new eval configs).
- TDD (tests in `src/tests/`); universal Codex review (every executable → review loop before its round closes); commit+push before remote/long runs; never edit a running script.
- Stop-and-ask on gate fail. Wait-time reporting on every response with in-flight runs.

## Environment
- conda env `rir2rir`: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate rir2rir`.
- GPU 1 via `CUDA_VISIBLE_DEVICES=1` (renumbers to cuda:0 inside the process — tracebacks saying "GPU 0" are that renumbering, not GPU 0).
- A6000 48 GiB (authors used H100 80 GiB) — B-F OOMs above micro 8; common pair micro 8 × accum 8 (eff 64).
- Released recipe (from FLAC.ckpt's own counters): eff-batch **64**, accum **1**, single H100, AdamW 5e-5/(0.9,0.999)/wd 1e-3, InverseLR(1e6,0.5,0.99), EMA on, bf16, **67,500 steps**. README's 32×2×2 (eff 128) example is ckpt-INCOMPATIBLE.

## Key files (exp_07 phase 2)
- `worklog/worklog_yixun/exp_07_fa_scratch_claude/plan_bv_parity.md` — approved P1 plan (branch-and-estimand table, tiers).
- `.../gate_verdict.py` — 5-seed mean±std(ddof=1), σ_c, tiers, 6/6 rule (fail-closed).
- `.../assert_arm_configs.py` — DINOv3 pin gate (rev `114c1379…`, sha256 `4610ad75…`) + seeded init-identity check.
- `.../probe_released_ckpt.py` — proves released eff-batch 64 / accum 1.
- Model configs: `src/configs/model_configs/FLAC/AR/FLAC_AR_BV.json` (byte-copy of `FLAC_AR.json`), `FLAC_AR_BF.json` (+`cond_method:fa_invariant`, `frame_avg_angles:[0,90,180,270]`), `FLAC_AR_BV_online_eval.json` (`use_ema:false`).
- `train.py` — `--max-steps` flag (defaults.ini `max_steps=1000000`); tests `src/tests/test_train_max_steps.py`.

## Baselines (exp_01, full split — the parity targets)
- **K=8:** T60 8.609±0.012 / C50 0.9682±0.0030 / EDT 37.10±0.07 / R@1 7.06±0.10.
- **K=1:** T60 9.969±0.039 / C50 1.0460±0.0064 / EDT 39.95±0.37 / R@1 6.83±0.22.

## Recent commits
- `67b8fce` P1 plan (review-clean) · `cb85fd0` worklog namespace move · `a3e8cf5` exp_08 closure.

## Gotchas
- Never `git amend` a commit that records its own SHA (self-reference changes on amend) — use a follow-up commit.
- Training logs > 100 MB are rejected by GitHub — `worklog/worklog_yixun/**/*_train.log` is gitignored; commit a filtered compact version.
- Codex model change is standing: `gpt-5.6-sol` xhigh (NOT the old gpt-5.5).

## Automation (this handoff)
A Claude Code hook auto-detects model changes and archives the four handoff docs + injects a reminder into the incoming model's context. The hook is the **detector/archiver only** — the live model still authors the full refresh of the four docs. See `.claude/hooks/` and the CLAUDE.md handoff protocol.
