# HANDOFF.md — working-memory contract for the next session

Assume the reader has NO memory beyond the repo + `master_experiment_tracker.md` + `issue_report.md` + this file. Updated at every handoff trigger (CLAUDE.md protocol): "handoff"/"new session"/"wrap up"/`/compact` **or a model change / model-limit swap**.

**Last updated:** 2026-07-16 ~17:40 EDT (trigger: model change `fable -> opus` detected by the hook at 17:30:47; refresh authored by **Fable 5**, which is what the harness actually served that turn — see the alternation note below).
**This handoff's trigger:** the hook fired on a real family flip. Prior trigger (00:55): Yixun `/model`-switched back to **Fable 5** (Opus 4.8 1M/max had covered while Fable was at its limit; Opus authored the 2026-07-15 handoff, the hook, both launch scripts, and launched the extend).

> ⚠️ **The main session ALTERNATES models mid-session** (established 2026-07-16 ~17:35; `issue_report.md` §8). This session's transcript holds **38 fable + 36 opus** non-sidechain assistant records — the harness fails over between turns on its own, not just on `/model`. **An incoming model has NO memory of the other model's turns in the same session.** Concretely: the Fable turn at ~17:30 knew nothing of the Opus turns that stopped the extend (13:35), wired SyncBN, and resumed the second leg (16:39) — it recovered them only from these docs + `ps`. **Treat these four docs as the live intra-session channel, not just a cross-session one: write state the moment it changes, and re-verify anything time-sensitive (`ps`, `nvidia-smi`, newest ckpt mtime) before quoting an ETA — a stale in-flight entry WILL produce a wrong wait-time report.**

## Role map (current)
- **Main session model:** Fable 5 **or** Opus 4.8 — assume either; do not assume continuity of your own prior turns. Plans, analyzes, drives runs. Role-attribution rule stands: whenever a non-Fable model fills a Fable seat (esp. analysis), flag the model in the artifact by-line.
- **Coder subagent:** Opus 4.8, max effort (writes experiment/production code).
- **Reviewer:** OpenAI Codex `gpt-5.6-sol`, xhigh, codex-cli 0.144.1 — `~/.local/bin/codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort=xhigh --output-last-message <file> "<prompt>" < /dev/null` (stdin MUST close with `< /dev/null`).

## Where we are — exp_07 phase 2 (B-V parity, Yixun Q5)
**Mandate (Q5):** "The B-V should at least get the same results as FLAC. Please achieve this."

- **P0 (DONE):** checkpoint selection alone CANNOT reach released parity (21-pt ≥20k K=8 seed-42 EMA curve). Best observed: EDT **38.29@60k** (vs released 37.10), R@1 **6.22@65k** (vs 7.06); T60/C50 reachable in-band. → systematic factor(s) remain.
- **NEW (2026-07-15, Yixun):** **B-V EXTEND** — "continue our previous train on B-V@67.5k to check what is the best ckpt we have." **RUNNING** (see In-flight). First point S70000: ALL metrics improved vs the 67.5k endpoint (T60 9.107, C50 0.9368 — now below released, EDT 40.538, R@1 **6.486 = lineage max**).
- **NEW (2026-07-16 ~04:50, Yixun — GPU-1 QUEUE ORDER):** **extend → B-F → P1.** Yixun's slot pick counts as the **explicit B-F go** for the post-extend slot. Consequences: extend STOPS at 100k regardless of trend (the adaptive continue-to-135k decision is MOOT; further B-V extension deferred, resumable anytime from any ckpt); **B-F from-scratch launches Jul 17 ~16:00** via pre-staged `bf_scratch_launch.sh` (8×8 eff-64, seed 42, 67.5k, mirrors B-V manifest; ~9.6 d → verdict ~Jul 28); **P1 moves AFTER B-F** (~Jul 27 start → verdict ~Aug 1). Scientific basis: P1 cannot change B-F's design (B-F only fits 8×8; the 8×8 B-V is its sole pre-registered control).
- **NEW (2026-07-15, Yixun):** FLAC runs should log to **wandb yh4742@princeton.edu** — BLOCKED: current `WANDB_API_KEY` = yixunhu21@gmail.com. Launch scripts default `LOGGER=none` + fail-closed identity gate. Yixun must export yh4742's key (env var overrides `wandb login`).
- **P1 (APPROVED 2026-07-15 — "I approve P1"):** micro-parity B-V rerun, **now queued behind B-F** (2026-07-16 reorder). Plan `plan_bv_parity.md` review-clean, committed `67b8fce`. Launch scripts committed `c40908c` (post-review fixes). Note for P1's write-up: the extend's late-gain evidence (S70000+) must be folded into the under-training-vs-micro-batch attribution.
  - **P1a fit probe (~20 min):** vanilla-only ladder **64×1 → 32×2 → 16×4**, 15 opt steps, EMA on, 1-s VRAM sampler, record steady-state samples/s (re-anchors ETA). Pick largest fitting rung. Review estimate: 64×1 likely OOMs on 48 GiB; 32×2 likely fits.
  - **P1b train (~3.4 d, re-anchored by probe):** `FLAC_AR_BV.json` (byte-copy), largest fitting rung, `--max-steps 67500`, seed 42, EMA on, ckpt every 2500, `HF_HUB_OFFLINE=1`, DINOv3 pin gate pre-launch. Then same 10k screens (EMA+online) + ≥20k selection curve + 5-seed gate.
  - **Success tiers (pre-registered, plan §1):** PARITY (composite-rule ckpt confirmed on held-out eval seeds 43–46, **R@1 REQUIRED**) / STRONG (≥50% late-curve gap closure on BOTH EDT+R@1: **EDT ≤38.59, R@1 ≥6.51**) / DIRECTIONAL / NULL. Late-curve statistic = mean over S∈{55k,57.5k,60k,62.5k,65k,67.5k}. Baseline (8×8, same statistic): **EDT 40.087, R@1 5.960**.
  - **Abort discipline:** hard aborts only (OOM/NaN/divergence); numerical futility check no earlier than 50k.
  - **Control rule:** 8×8 B-V stays the ONLY B-F control (incomplete factorial). Never compare B-F-8×8 causally against B-V-at-larger-micro.

## In flight right now
- **SyncBN wiring DONE (commit `f362673`):** `--sync-batchnorm` → PL `Trainer(sync_batchnorm=True)`, fail-closed <2 GPUs, 40 tests green, review+reverify SHIP. Review key finding: **accumulation never feeds BN stats** → only rung 32/GPU×2×accum1 satisfies BN=64; probe is single-rung; launch pins `MB=32 ACC=1` literally. CFG dropout verified SyncBN-safe (`dit.py:302`).
- **M1 probe watcher ARMED** (bg task): fires when BOTH GPUs are free (extend ends ~Jul 17 ~19:40 on GPU 1; aug291k ~Jul 18 ~02:00 on GPU 0) → runs `m1_ddp_fit_probe.sh` → **REPORT RUNG TO YIXUN AND HOLD** (his instruction: no training without his post-probe go).
- **B-V EXTEND — SECOND LEG (GPU 1, wandb run `exp07_BVextend`/ypp5, resumed 2026-07-16 16:39 from ckpt 77500):** → 100k, ETA ~Jul 17 ~19:40 EDT; screens 72.5/75/77.5k ran co-located; 80k/90k/100k screened on arrival. *(First leg PID 3737059 was STOPPED 13:35 at ckpt 77500 for the B-F reprioritization — see `bv_extend_stop_restart.md`.)*
- *(superseded first-leg entry below kept for provenance)* **B-V EXTEND (GPU 1, PID 3737059, launched 2026-07-16 00:47 EDT):** `LOGGER=none bash worklog/worklog_yixun/exp_07_fa_scratch_claude/bv_extend_launch.sh 100000` — resume `outputs_FLAC/exp07_BV/epoch=14-step=67500.ckpt` → step 100,000, seed 42, 8×8 eff-64, ckpt every 2500 into `outputs_FLAC/exp07_BVextend/` (logger none → ckpts directly under save-dir). Log: `worklog/worklog_yixun/exp_07_fa_scratch_claude/fa_scratch_2026-07-16_00-47-25_BVextend_train.log` (gitignored raw; commit a compact filtered copy at close). Resume verified: "Restored all states", lr 4.84e-5 ✓, loss ~0.32–0.6, 10.4 GiB. **ETA ~Jul 17 ~14:45 EDT** (32.5k steps at phase-1 854 steps/h). First new ckpt (70k) ~03:40 Jul 16. **When done:** screen 70k–100k EMA K=8 seed-42 full split (mirror `exp07_BV_selcurve_*` protocol), report best-ckpt verdict + continue-to-135k recommendation, then launch P1a. **Known quirks:** PL mid-epoch-resume warning (no dataloader fast-forward — epoch-14 remainder drawn from a fresh epoch-14 iterator; acceptable, documented in script header); tqdm it/s inflated on resume (display artifact).
- **Codex review round CLOSED** (`p1_scripts_codex_reverify.md`, `..._reverify2.md`): **extend = SHIP** (running launch clean), **probe = SHIP**; hook's last residual (non-dict `message` type guard in the transcript reader) fixed post-reverify2. **Hook proven in production 2026-07-16 00:56:** detected `opus → claude-fable-5` on UserPromptSubmit (one prompt late, by documented design), archived 4/4 docs to `handoff_snapshots/2026-07-16_00-56-03__opus__to__fable/`, injected the reminder.
- **GPU 1** held for the extend→B-F→P1 sequential window (~2.5 weeks). GPU 0 is another session's job (PID 1284685) — do not touch.
- **B-F: GO GIVEN (2026-07-16) for the post-extend slot; wandb ON.** Pre-staged kit: `bf_scratch_launch.sh` (GPU-free guard + wandb identity gate + pin gate), `bf_screen.sh` (per-10k-ckpt EMA+online screens, recursive ckpt find), `FLAC_AR_BF_online_eval.json` (use_ema-only flip, diff-verified). Launch sequence when the extend completes: run 90k/100k screens → **`LOGGER=wandb bash worklog/worklog_yixun/exp_07_fa_scratch_claude/bf_scratch_launch.sh`** → arm ckpt monitor (recursive find — wandb nests ckpts under `outputs_FLAC/exp07_BF/<project>/<run>/checkpoints/`) + screens at 10k cadence. wandb: project `FLAC_exp07_BF`, run `exp07_BF`, account **yh4742@princeton.edu** (key verified 2026-07-16; script self-extracts it from `~/.bashrc` because the interactive guard blocks non-interactive sourcing). Logger delta vs B-V phase 1 (none vs wandb) is observation-only (no RNG consumption) — note it in the closing analysis.

## Parallel workstream — `cylindrical-dinov3` repo (NEW 2026-07-16, Fable 5)

Set up today at Yixun's request; **no GPU use, does not touch the exp_07 queue**.

- **Repo:** `~/codespace/cylindrical-dinov3` → GitHub `Yixun-Hu/cylindrical-dinov3` (**private**, default branch `main`, SSH remote, pushed). Commits: `9431737` scaffold → `3f6b82c` context export + SOP → `fcfc193` vanilla reference.
- **Why a sibling repo, not a FLAC subfolder:** `pip install -e` both into one env; FLAC and the SSL training scripts then import the same `cylindrical_dinov3` package, and weights cross via `save_pretrained()`/`from_pretrained()`. Explicitly NOT a Transformers fork/PR and NOT an edit to site-packages.
- **`gh` CLI installed** at `~/.local/bin/gh` (v2.96.0), authenticated as **Yixun-Hu** (ssh, scopes `gist, read:org, repo`) — it did not exist on this box before; GitHub ops now work non-interactively from any session.
- **`vanilla_dinov3/`** — read-only vendored `transformers==4.57.0` `dinov3_vit`, all 5 `.py` **sha256-verified against `transformers-4.57.0.dist-info/RECORD`** (all matched; `modeling_dinov3_vit.py`'s later mtime was a touch, not an edit). Verified in-file: `DINOv3ViTModel` uses **`self.layer`** (`:493`, iterated `:521`) — Transformers `main` renamed this to `self.model`, so **port against v4.57.0, never against `main`**. RoPE = `DINOv3ViTRopePositionEmbedding` (`:133`), attention = `DINOv3ViTAttention` (`:253`). Not importable standalone (package-relative imports); never add it to `sys.path`.
- **Design source of truth:** `ai_conversations/claude_context_dinov3_cylindrical_conversation_from_codex.md` (verbatim Codex transcript). Key load-bearing decisions in it: keep 16×16 patch (→ 16×32 token grid at 256×512 → strict **C₃₂**, 16-px-multiple rolls only); integer azimuth harmonics `m=0..15` (non-integer ⇒ seam breaks); **never** latitude-scale the azimuth phase; CLS/register must be kept as params (for `strict=True` load) but held OUT of the transformer, with patch **mean-pool** as `pooler_output` (FLAC's `Linear(384,256)` then needs no change); eager attention first; XYZ gauge alignment (per-column `Rz(-θ)`) only AFTER pure-roll tests pass. SSL verdict: **no from-scratch DINOv3 pretraining** — official weights → short continued SSL/distillation → FLAC fine-tune.
- **Yixun's task (received ~17:35, IN PROGRESS):** implement the cylindrical ViT into `src/cylindrical-dinov3/`, adapting the vanilla `modeling_dinov3_vit.py`, following the SOP pipeline — **plan → review → revise until approved → code → code review → revise until approved**.
- **Open:** that repo has no `CLAUDE.md` and an empty `announcement/` (see `issue_report.md` §9).

## Do-not-touch (other sessions' jobs)
- The **`FLAC_vanilla291k`** run and the **rir2rir** jobs belong to other sessions — leave untouched. 291k is a corroborating row only (data folder `single_channel_ir` ≠ `_1`, micro 16×4; not B-V-certifiable).

## Standing constraints
- Full published eval config only (unseen = all 6337 items / 17 rooms; never subsample or create new eval configs).
- TDD (tests in `src/tests/`); universal Codex review (every executable → review loop before its round closes); commit+push before remote/long runs; never edit a running script.
- Stop-and-ask on gate fail. Wait-time reporting on every response with in-flight runs.

## Environment
- conda env `rir2rir`: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate rir2rir`. **All exp_07 runs stay on `rir2rir`** — Yixun mentioned `conda activate flac` (env exists) but switching envs mid-experiment would break manifest identity with B-V phase 1 (pip-freeze recorded in its launch log); flagged to Yixun 2026-07-16, `flac` env reserved for future fresh experiments.
- wandb: key for **yh4742@princeton.edu** lives in `~/.bashrc` (line ~141) BELOW the interactive guard → non-interactive shells must `eval "$(grep -E '^\s*export\s+WANDB_API_KEY=' ~/.bashrc | tail -1)"` (plain `source ~/.bashrc` silently keeps the OLD yixunhu21 key from the harness env). Launch scripts self-extract.
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
