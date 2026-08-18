# exp_19 — HAA finetuning of P1-vanilla@40k and B-F FA@40k (plan, Rev 1)

**Seat:** planned by Claude Fable 5 (main session). Coder seat per SOP: Opus 5 max.
**Status: APPROVED by Yixun 2026-08-17 (~12:40 EDT)** — B1: 1,000 steps (ckpt/10; step-410 exists as a byproduct, endpoint = 1000). B2: HAA-YAW third arm INCLUDED. B3: single-GPU recipe, two arms parallel one-per-card, third follows. B4: HAA_md.py untouched (released behavior).

**R1 gauge analysis (planner seat, pre-implementation):** the yaw machinery is
gauge-CONSISTENT on HAA — the HAA panorama's column→azimuth map is the standard
CCW convention ((cosφcosθ, cosφsinθ, −sinφ), θ linear in column), the same
orientation `rotate_scene_metadata` assumes, and that function rotates the
STORED per-pixel 3D vectors themselves (not just rolling columns), so
consistency depends only on the (x,y)→θ orientation. The deliberate minus lives
in the vertical/translation components and does not enter yaw pairing. FA's C4
orbit is additionally closed under sign flip regardless. The empirical R1 probe
remains a launch gate to catch anything this reading missed.

## 0. The recipe, verified against the sources

The released recipe (README "Finetuning on HAA", consistent with FLAC_pdf.md
B.1's global training setup) is:

| Item | Value | Source |
|---|---|---|
| Steps | **1,000** (`--max-steps 1000`; our TDD flag replaces the old hardcode) | README |
| Batch | 16 × accum 4 = eff **64** (small dataset ⇒ accumulation) | README |
| LR / sched | 5e-6 AdamW, InverseLR(1e6, 0.5, 0.99) | `FLAC_HAA_finetune.json` |
| Init | `--pretrained-ckpt-path` = **EMA weights** (paper used `FLAC_EMA.ckpt`) — weights-only load into a fresh training wrapper | README + `train.py:139` |
| VAE | `--pretransform-ckpt-path weights/FLAC/VAE.safetensors`, never finetuned | README + paper §"do not fine-tune FLAC's VAE" |
| Data | `haa_train.json` / val `haa_val.json`; test `haa_test{_1,}.json` (K=8 / K=1) | configs on disk |
| Cadence | `--val-every 10 --checkpoint-every 10` | README |
| Convention | HAA panorama at SOURCE position; `HAA_md.py:70` minus sign kept (released behavior); the "remove the minus" performance tip recorded as an optional ablation, NOT default | README + CLAUDE.md |

⚠️ **"410 steps" does not appear in FLAC_pdf.md** (searched). Resolution that
serves both readings: train the full 1,000 steps with checkpoints every 10 —
the **step-410 checkpoint exists either way**, and endpoint selection becomes
an eval-time decision (B1 below), not a training-time commitment.

**Why `--pretrained-ckpt-path` matters:** it is a WEIGHTS-ONLY load into a
fresh wrapper (train.py:139–147) — a fresh optimizer and a fresh 5e-6 schedule.
This sidesteps the repo's known scheduler-clobber trap (a warm RESUME would
silently keep the AR run's 4.77e-5 schedule; see CLAUDE.md checkpoint surgery).

## 1. Arms

| Arm | Init (EMA weights @40k) | Training protocol | Eval protocol |
|---|---|---|---|
| **HAA-P1** | `outputs_FLAC/exp07_P1/…/epoch=8-step=40000.ckpt` | vanilla (stock `FLAC_HAA_finetune.json`) | `--cond-method vanilla` |
| **HAA-BF** | `outputs_FLAC/exp07_BF/…/epoch=8-step=40000.ckpt` | fa (config + B-F's two training deltas: `cond_method: fa_invariant`, `frame_avg_angles: 4`) | `--cond-method fa_invariant` |
| *(B2)* HAA-YAW | foreign worktree Yaw-Aug @40k | vanilla + `training.yaw_aug` block | `--cond-method vanilla` |

Announcement-05 discipline throughout: each arm evaluated ONLY under its own
protocol, flags explicit in every manifest.

## 2. Planned files (all TDD, all through the review loop)

1. **`src/tools/extract_ema_weights.py`** (+tests) — copy-only: read a PL
   training checkpoint, emit the EMA model state dict in the format
   `--pretrained-ckpt-path` consumes (verified by loading it through
   `load_ckpt_state_dict` + the train.py:142 prefix-strip and asserting exact
   key-set match against `create_model_from_config(config)`). Needed because
   the released flow inits from `FLAC_EMA.ckpt` (plain EMA weights) while our
   40k artifacts are wrapped PL checkpoints, and `unwrap_model.py` is broken
   upstream (`stable_audio_tools` import).
   *Init equivalence check:* state_dict sha256 of loaded-model-weights must
   equal the EMA entry of the source checkpoint (fail-closed gate in the
   launcher).
2. **`worklog/…/exp_19…/FLAC_HAA_finetune_BF.json`** — byte-copy of the stock
   HAA finetune config + exactly B-F's two training deltas (the same
   single-delta-contract discipline as exp_17; contract test asserts it).
3. **`worklog/…/exp_19…/haa_ft_launch.sh`** (+guardtests) — per-arm launcher
   copying the exp_17-lineage gates: config contract, source pins
   (train.py/defaults.ini/HAA_md.py/HAA configs/VAE + the init weights sha),
   wandb identity, per-GPU VRAM floor, endpoint-reached check
   (`max_steps=1000` marker), and **the FA-on-HAA invariance probe as a launch
   gate for HAA-BF** (see risk R1).
4. **`worklog/…/exp_19…/haa_ft_eval.sh`** — eval queue: for each arm ×
   {step-410, step-1000 or val-selected} × {K=8 `haa_test.json`, K=1
   `haa_test_1.json`} × 5 eval seeds (42–46), `--cond-method` per arm,
   `--cond-autocast bf16`, cfg 1.0, steps 1. Metric JSONs into per-arm exp_19
   namespaces (symlink-farm pattern where an init lives outside this repo).
   Paper-style reporting = per-scene average (README "Paper Metrics") — the
   4-room HAA per-scene map comes from `--record-per-scene`; we report BOTH
   global and per-scene means, labeled.

## 3. Risks / pre-registered readings

- **R1 (FA on HAA is not yet validated).** The C4 machinery was built and
  proven on AR (listener-position panoramas). HAA reverses to source-position
  with a sign convention (`HAA_md.py:70`). Before any HAA-BF training:
  run the exp_03-style invariance probe — rotate the HAA panorama+poses by C4,
  assert conditioning invariance at ~1e-7 — as a HARD GATE. If it fails, the
  sign convention interacts with rotation; STOP and report to Yixun (do not
  silently "fix" the convention).
- **R2 (finetune-damage precedent).** exp_03–06 established that AR finetuning
  from a converged checkpoint converges to the new objective's optimum (can
  regress elsewhere). 1,000 steps at 5e-6 is mild; still, both endpoint AND
  best-val checkpoints are recorded so a damaged endpoint is visible.
- **R3 (five training seeds are NOT in scope).** One finetune run per arm
  (seed 42), 5 EVAL seeds — matching how all 40k rows were produced. Training
  -seed variance at 1,000 steps is unquantified; disclosed as a caveat.
- **R4 (paper numbers are not directly comparable).** Paper HAA rows finetune
  from the RELEASED 67.5k-lineage EMA; ours start from 40k-budget checkpoints.
  The comparison that IS valid: HAA-P1 vs HAA-BF (vs HAA-YAW), same budget,
  same recipe — the transfer question, not absolute parity with Table 3.

## 4. Open decisions (B*)

- **B1 — endpoint convention:** train 1,000 + ckpt/10 and report BOTH step-410
  and step-1000 (recommended; zero extra cost), or literal 410-step run?
  Also: where did 410 come from? If it is from another source (arXiv version /
  wandb), point me to it and I will pin that instead.
- **B2 — include HAA-YAW as a third arm?** Recommended: yes (the exp_17
  narrative's transfer test; +~1 h total). Yixun named only P1 and B-F, so it
  is his call.
- **B3 — GPU plan:** recipe is single-GPU (batch 16 × accum 4). Run two arms
  concurrently (one per A6000), third follows. No SyncBN question at
  single-GPU (BN sees micro-16 exactly as the released recipe did).
- **B4 — HAA_md sign tip:** default = released behavior (keep the minus).
  Optional ablation later if wanted.

## 5. Execution order (after approval)

1. TDD `extract_ema_weights.py` → Codex review → extract 2–3 EMA inits + shas.
2. HAA-BF config + contract tests; R1 invariance probe (gate).
3. Launchers + guardtests → Codex review → SMOKE (20 steps, own namespace).
4. FULL: 2 arms in parallel (~1–2 h each at single-GPU eff-64), third after.
5. Eval queue (both K, 5 seeds, both endpoints) → per-scene + global tables →
   results/analysis mds → closure review round.

**Wall-clock estimate:** build+review ~3–4 h; training ~2–3 h total; eval
~1.5 h; closure ~1 h. One working day end-to-end.
