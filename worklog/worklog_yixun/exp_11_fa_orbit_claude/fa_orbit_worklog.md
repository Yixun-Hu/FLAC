# Lab notebook — exp_11_fa_orbit

## 2026-08-05T15:40:00-04:00 — scaffold + environment reconnaissance

- **Goal** — scaffold exp_11 (orbit-size sweep C8/C16/C32 for fa_invariant conditioning, commissioned by Yixun 2026-08-05) and establish the ground truth of the machine/repo state before planning.
- **Version Control** — branch `check-equivariance-necessity`, base commit `b9e38ce` (model_comparison: fa-scratch@40k vanilla-eval rows). Working tree clean except untracked `AcousticRooms` symlink.
- **Result** — `passed` (scaffold). Reconnaissance findings, all verified from disk:
  - **Plumbing already exists end-to-end**: `training.frame_avg_angles` is read by `src/training/factory.py:76` → `LightningDiffusionModel` (train side), and `eval_FLAC.py --frame-avg-angles` (eval side, records `a{n}` in metric-JSON filenames). `invariant_conditioning` (src/data/yaw_rotation.py:221) accepts arbitrary angle tuples (angles[0] must be 0.0); the orbit loop is sequential (one `only_ids` ViT pass per non-identity angle, accumulate + average) so wall-clock scales ~linearly in |orbit| while non-ViT conditioners run once.
  - **Exactness geometry**: `rotate_scene_metadata` quantises to integer panorama-column rolls (W=512). C8=64 px, C16=32 px, C32=16 px — all exact AND all aligned to the DINOv3 16-px patch grid (C32 is the finest patch-aligned subgroup; C64 would be sub-patch).
  - **Measured C4 fa step rate (exp_10, DDP 2×L40, MB32)**: ckpt mtimes 55k→65k = 2,500 steps per ~7.19–7.55 h ⇒ **~0.095 steps/s** (slower than the 0.14 planning figure; node cotenancy suspected).
  - **exp_07 B-F ckpts 2.5k–40k all on disk** (`outputs_FLAC/exp07_BF/.../checkpoints/`) — a C4 fa-protocol screen curve below 40k can be backfilled cheaply (the exp_07 S10000–S30000 screens were the retracted mismatched-protocol evals and are NOT usable as fa comparators).
  - **Storage**: /n/fs/gatrdp at 48% (859 GB free); 3 arms × 16 ckpts × ~724 MB ≈ 35 GB — no quota risk.
- **Analysis** — the experiment is config-only on the training path (new model-config JSONs + a parameterized launch script); no model code changes needed. The dominant planning question is GPU cost: with ViT passes ~5/6 of the C4 fa step cost, projected per-arm cost to 40k steps is ~6.5–9.4 d (C8), ~13–18.5 d (C16), ~26–37 d (C32) on 2 GPUs each — a staging/budget decision that belongs to Yixun.
- **Next** — write `plan_fa_orbit.md`; Codex plan review; present for approval. No implementation before approval (SOP gate).

## 2026-08-05T15:40:00-04:00 — cross-experiment state found during reconnaissance (context, not exp_11 actions)

- **exp_10 B-F resume is STALLED at step 65,000/67,500**: last ckpt written 2026-08-05T09:33; no training process alive; all 8 GPUs idle at 15:27. Logged in exp_10's worklog; restart decision deferred to Yixun.
- **Untracked-file wipe ~2026-08-04T23:17**: every tracked file's mtime is 23:17 and exp_10's tee'd train log plus all pre-Aug-5 `wandb/run-*` dirs are gone (checkpoints under `outputs_FLAC/` survived). The exp_10 process itself survived the wipe and died later, cause not diagnosable without its log.
- **Cotenant session on this node**: a second Claude session (checkout `/n/fs/gatrdp/codespace/cylindrical-dinov3`, working in this FLAC repo) launched `FLAC_exp02_P1rerun` probes and a committed 2-GPU 87.5k-step run at 14:49–14:54 today (wandb dirs in this repo). Its GPUs read idle at 15:27 — treat 2 GPUs as presumptively claimed; launch-time VRAM gates (bf-launch family) will verify reality.

## 2026-08-05T16:25:00-04:00 — plan Rev 1 → Codex review (REJECT, 11 BLOCKING + 3 NIT) → Rev 2

- **Goal** — SOP plan-review loop before surfacing for approval.
- **Command / Validation** — `codex exec`, gpt-5.6-sol @ xhigh, full reviewer briefing (SOP + announcements + exp_07/10 context + implementation surfaces). NOTE: bwrap sandbox unavailable on this host (`max_user_namespaces=0`) → `--sandbox danger-full-access` with read-only instruction; tree verified clean post-review. Review saved: `fa_orbit_codex_plan_review.md`.
- **Result** — `passed` (loop closed at plan level): verdict REJECT; all 14 findings addressed in Rev 2 (per-finding changelog = plan §9). Substantive design changes: contemporaneous C4L bridge control arm (comparator switch), seed-paired statistics with K8 T60/R@1 co-primaries, pre-registered per-arm 2×2 mechanism cells, numeric trend estimates with NOT-ESTIMABLE rules, economic futility gates (FUTILITY-STOPPED excluded from inference), fail-closed row-provenance validator, D3 bf16-invariance-floor probe, D1 demoted to fully-specified cost-prior, R3 redesign, B-V-extend compute claim withdrawn (Planner error, acknowledged), R4 exploratory compute-frame cell, corrected cost algebra prose, Slurm-based launches (environment change discovered mid-planning: this is a 32-node Slurm cluster; org policy + current practice require the scheduler).
- **Analysis** — the review materially strengthened the design; the C4L bridge in particular converts a historical cross-run comparison into a same-environment controlled sweep at modest cost (~160–235 GPU-h).
- **Next** — present Rev 2 + staging options to Yixun for approval. Implementation (TDD rounds) only after sign-off.

## 2026-08-05T17:30:00-04:00 — plan APPROVED (Yixun Q2) with fast-recipe amendment → Rev 3 → implementation begins

- **Goal** — record approval + amendment; open Coder round 1.
- **Change** — plan §10 (Rev 3): recipe = DDP N×L40 WITHOUT ViT grad-ckpt, micro×N=64 + SyncBN (preserves eff-64 AND BN-64; rungs 8×8/16×4/32×2), one P0-selected rung for all four arms; P0 profiling stage (throughput/fit matrix + differential component attribution + conditional deep profile) replaces M2 and answers Yixun's "what blocks the training time"; staging = Option B (C4L+C8+C16 after P0; C32 = Yixun go/no-go on profiled numbers). exp_10 restart dropped (other machine). Query file Q2 appended (verbatim + recipe-semantics note).
- **Version Control** — plan/scaffold committed `f8d18d3`; exp_12 probe (informs the 32×2 rung prior) committed `4c095ae`, Slurm job 3637984 RUNNING.
- **Result** — `launched` (implementation): Coder round 1 (Opus seat, background) = four arm configs (TWO allowed leaf groups vs exp_07 `FLAC_AR_BF.json`: orbit angles + ViT grad-ckpt→false) + TDD tests (`test_exp11_orbit_configs.py`, parametrized invariance tests). Codex review before round 2 (P0 profiling kit).
- **Next** — round 1 review → P0 kit round → P0 jobs → rung selection + bottleneck report → arm launches.

## 2026-08-05T18:20:00-04:00 — round 1 CLOSED (code b1c1198 → Codex REJECT 2B+2N → fixes 91cfc0e → verified green)

- **Goal** — close the round-1 loop per SOP.
- **Version Control** — code `b1c1198` (4 arm configs + TDD tests; RED 13-fail → GREEN 26-pass); review `fa_orbit_codex_code_r1_review.md` (REJECT: loose tuple equality would accept gc=0; averaging tested only at C8; +2 nits); fixes `91cfc0e` (strict `is True`/`is False` leaf assertions + in-memory falsy regression; `test_cn_average_correctness` parametrized n∈{8,16,32} — mutation-verified to catch an always-÷8 bug at C16/C32; duplicate-key + NaN/Infinity-rejecting JSON loader; orbit-test runtime 175 s→14.4 s, root cause 52 default intra-op torch threads, pinned to 1 in-fixture).
- **Command / Validation** — Planner re-verification: strict assertions + parametrize + loader hooks confirmed in the 91cfc0e diff; `test_exp11_orbit_configs.py` re-run green (14 passed). Coder confirmed no pre-existing test modified (git diff audit).
- **Result** — `passed`; round CLOSED. Nothing launched from this round.
- **Next** — Coder round 2: P0 profiling kit (parameterized 30/10-step sbatch pair per cell, UUID-bound VRAM poll, matrix submitter + TDD'd collector) → Codex review → submit P0 matrix.

## 2026-08-05T20:15:00-04:00 — round 2 CLOSED (P0 kit: e566513 → REJECT 7B+2N → ec0250d → re-review REJECT 6B+2N → 60764c1 → verified)

- **Goal** — close the P0-kit review loop.
- **Version Control** — code `e566513` (paired-job kit) → review `fa_orbit_codex_code_r2_review.md` (REJECT; critical: PL 2.1.0 under sbatch ntasks=1 elects SLURMEnvironment and never spawns DDP ranks → every multi-GPU cell would have been world-size-1 with BN=micro; also paired-job timing invalid, manifest binding absent) → fixes `ec0250d` (torchrun --standalone + SLURM-env neutralization + pinned world-size/completion literals from installed PL source; in-fit rank0 synchronized t10/t30 via new `p0_runner.py`; manifest-bound collector; cell-derived configs; util/power poller; 2 real bugs found by the Coder's own testing: awk array/scalar fatal in tick validation, workers-mode manifest duplicate) → re-review `fa_orbit_codex_code_r2_reverify.md` (core CLOSED — launch/timing/runner parity verified against installed PL; 6 peripheral blockers incl. the methodological FA1 catch: VAN→C4L confounds pose-path change with ViT passes) → fixes v2 `60764c1` (FA1 control config `[0.0]` — empirically verified single-pass bit-identical; manifest modes + exact provenance binding incl. maxsteps/mb/ngpu/workers + pollcsv sha; mandatory finite util/power; per-cell time limits up to 4 h for C32_32x2 spot; no-clobber RUNID publish).
- **Command / Validation** — 75 tests green across the four exp_11 test files + invariance file; bash -n / py_compile clean; matrix = 13 single 30-step jobs (VAN/FA1/C4L/C8 × 3 rungs + CKPT4_32x2). Planner verification of v2 against each re-review finding: PASS. Remaining live validation per re-review NIT 8: the 2-GPU smoke (next entry) before the matrix.
- **Result** — `passed`; round CLOSED pending smoke evidence.
- **Next** — 2-GPU smoke (C4L_32x2, manual single job) → matrix submission → collect → rung selection + bottleneck report.
