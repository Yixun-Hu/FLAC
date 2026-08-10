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

## 2026-08-05T18:35:00-04:00 — P0 smoke 1 (C4L_32x2, job 3638618): machinery PROVEN live + first feasibility result

- **Goal** — re-review NIT-8 live validation of the P0 kit at commit `8d53691` (EXPECT_SHA-bound).
- **Result** — `passed` (as validation) + real finding:
  - **Machinery live-proof:** torchrun spawned 2 ranks (PL literal "All distributed processes registered. Starting with 2 processes"); both allocated UUIDs actively polled; complete provenance P0RESULT emitted with `valid=1`; OOM classified via the flushed log; job exit FAILED 3:0 as designed. wall_fit 33.2 s (died in first forward).
  - **Feasibility result: C4L at rung 32×2 WITHOUT grad-ckpt OOMs** — peak 45,455/46,068 MiB on BOTH GPUs; per-rank 44.38 GiB in use (35.81 torch-allocated, 7.87 reserved-unallocated) at a 14 MiB request. Symmetric across ranks ⇒ genuine capacity limit. Implies C8/C16/C32 at 32×2 no-ckpt also infeasible (strictly more retained per-pass output); 32×2 survives only for VAN/FA1 (fewer passes) and CKPT4 (ckpt ON).
- **Analysis** — the fast-recipe rung race is effectively between **16×4 and 8×8**. P0STEP timing path still unproven live (OOM before step 10) → smoke 2 = FA1_32x2 (job 3638630) to witness t10/t30 before the matrix spends queue time.
- **Next** — smoke 2 → matrix (13 jobs).

## 2026-08-05T18:45:00-04:00 — P0 smoke 2 (FA1_32x2, job 3638630): timing path PROVEN; kit fully live-validated

- **Result** — `passed`: rc=0, COMPLETED 0:0 in 60 s. Both P0STEP marks emitted (t10_mono 1674261.179 → t30_mono 1674280.978; Δ=19.799 s / 20 steps ⇒ **1.010 steps/s**, FA1 32×2 no-ckpt). Peak 40,425/46,068 MiB (87.7%) both UUIDs — FA1 fits at 32×2 but tight; corroborates smoke-1: each additional retained orbit pass pushes past capacity (C4L OOM confirmed). valid=1, full provenance.
- **Analysis** — P0 kit fully live-validated (spawn, world-size gate, timing, poller, provenance, OOM path, success path). NIT-8 requirement satisfied. First rate datum: FA1-no-ckpt at 2 GPUs already 1.01 steps/s vs the old C4-ckpt 2-GPU 0.095 — the fast-recipe direction is strongly confirmed.
- **Next** — submit the 13-job matrix.

## 2026-08-05T19:05:00-04:00 — P0 MATRIX LAUNCHED (13 cells, run aa4bc18-1785968431124626318-df9602ea)

- **Goal** — the Rev-3 §10 throughput/fit matrix + attribution cells.
- **Command / Validation** — `p0_submit_matrix.sh matrix` @ commit `aa4bc18`; jobs 3638637–3638649 (manifest `p0_manifest_aa4bc18-1785968431124626318-df9602ea.txt`); commands in `fa_orbit_command.md`. At submit+2 min: VAN_32x2 + VAN_8x8 RUNNING, 11 PENDING.
- **Acceptance criteria (pre-launch)** — every cell: commit-bound rc per class (0 success with both P0STEP marks and valid=1 poller evidence; 3 OOM with valid measurement; anything else = distinct failure); collection admits rows only via the manifest; matrix-mode attribution fit {FA1,C4L,C8} per rung where all three valid; no contact with jobs 3637217/3638486.
- **Result** — `launched`; ID-bound background waiter armed (a first name-filter waiter false-fired on the `-w6` suffix and was replaced — logged for honesty).
- **Next** — collect on completion → rung selection + bottleneck report; Coder round 3 (arm launcher) opened in parallel.

## 2026-08-05T20:10:00-04:00 — P0 attempt-2 COMPLETE (13/13 cells): no-ckpt recipe INFEASIBLE for C8+; recipe decision escalated to Yixun

- **Goal / Result** — full feasibility + throughput map, run 72a8114-1785969226421855487-c8d5b51f (attempt-1 aa4bc18 was partially killed by a mid-queue HEAD move — my operational error, now governed by a hard commit freeze while bound jobs are queued):
  - **No-ckpt OOM map (46 GB L40):** C4L: OOM @32×2, OOM @16×4, FITS @8×8 (37,255 MiB, 0.635 steps/s). **C8: OOM at ALL rungs incl. 8×8 (45,457 MiB)** ⇒ C16/C32 a fortiori. VAN/FA1 fit everywhere (peaks 40.4/21.5/11.1 GB at 32×2/16×4/8×8).
  - **Throughput:** VAN 1.010/1.502/1.581 steps/s (32×2/16×4/8×8) — 4→8 GPU gain ≈ 5% (comm/input-bound at micro-8; sweet spot 16×4). FA1 ≈ VAN (1.010→1.458→1.609): fa dispatch+cylindrical ≈ free; ViT-pass slope dominates. CKPT4_32x2 (grad-ckpt ON) 0.293 steps/s clean-node (3.1× exp_10's cotenancy-throttled 0.095).
  - **Anomalies:** attempt-2 FA1_32x2 = hardware ECC on neu322 (excluded henceforth via SBATCH_EXCLUDE); 8-GPU successes flagged valid=0 by an over-strict poller validator (76 complete 8-UUID ticks on inspection; steps/s marks unaffected) — validator tolerance fix queued into the round-3 worktree round.
- **Analysis** — Yixun's Q2 "fastest recipe = multi-card without gradient checkpointing" is **physically infeasible for the orbit arms** at BN-preserving rungs on this hardware (micro-4 would require 16 GPUs / 2 nodes — a much larger recipe departure). Decision escalated with options + measured numbers (uniform ckpt at best rung is the Planner recommendation). Supplemental CKPT4_{16x4,8x8} cells submitted to price it before Yixun decides.
- **Next** — supplemental ckpt cells → decision report to Yixun; coder worktree round (launcher hardening + validator tolerance) resumes at session-limit reset.

## 2026-08-05T21:25:00-04:00 — Yixun Q3 (uniform grad-ckpt, fastest rung) → consolidated round applied (d4f3977)

- **Goal** — record the recipe decision; land the consolidated worktree round.
- **Change** — patch `r3fix.patch` applied to main (12 files, +1003/−366; coder worktree commit daecd25 on 09d41ca): five configs → gc TRUE (C4L now byte-identical to exp_07 BF.json; others angles-only deltas — tests enforce), P0 poller precision fix (**root cause of the 8-GPU valid=0: awk OFMT %.6g snapped epoch timestamps to 6 significant digits — liveness compared rounding noise; replay-proven on the three real CSVs, true gaps 0.16–0.55 s**), launcher round-3 fixes (all 8 blockers + 2 nits: pin block with TO-PIN-AFTER-P0 refusal, fa_orbit_submit.sh flag derivation, torch.load restart preflight as tested pure functions, atomic run locks, wrong-world-size watcher replacing the scancel timer, FIFO dual-tee with exit precedence 6>3>4>7>raw, env/VAE/PL gates, wandb scrub + resume=must, SMOKE mode, temp-root guard suite).
- **Command / Validation** — worktree: 52 guard tests + 76 pytest green; main tree after apply: 61 kit tests green in 9.13 s. Supplemental CKPT4_8x8 (3639146) cancelled as redundant — the official matrix measures that cell.
- **Result** — `fix_ready`; round awaits Codex re-review (next), then the official ckpt-recipe matrix.
- **Next** — Codex review → official matrix (all cells now ckpt semantics) → collector report → pin commit → SMOKE → launches.

## 2026-08-05T21:45:00-04:00 — Q4: C32 GO (Yixun) — full four-arm commission authorized

- **Result** — staging fully resolved: C4L+C8+C16+C32 all launch post-(review ✓ + official matrix + pin commit + SMOKE). Projected C32 cost under ckpt ≈ 4–10 d (fitted slope from the official matrix will firm this up; vs 25–37 d under the original plan). Remaining gates are mechanical, no open user decisions.

## 2026-08-05T22:55:00-04:00 — final fix round landed (abbff5a..d1477c1): launch preconditions 1–2 satisfied → OFFICIAL MATRIX

- **Change** — NEW-1: `FLAC_AR_VANCKPT.json` (canonical + gc:true ×2, parsed-delta-tested); CKPT4 family fully retired; matrix = 12 all-ckpt cells. NEW-2: OUTPUT_ROOT pinned to production literal under Slurm (both scripts + submitters). B2/B3/B5/B7 residuals closed (fail-closed commit binding, flock ownership, checked pip-freeze/dual-copy/transcript with class-7, WANDB_ENTITY export + post-run run-identity verification). NEW-3 intent-before-sbatch with scancel-on-failure; NEW-4 FIFO hygiene; NEW-5 comment corrected + 2 s liveness bound restored.
- **Command / Validation** — fresh committed evidence: 72 guard cases + 78 pytest green (`fa_orbit_2026-08-05_21-45-11_guardtests.log`, `fa_orbit_2026-08-05_21-45-57_pytest.log`). Coder incident disclosed and repaired in-round (two tracked P0 manifests deleted by a cleanup, restored via git checkout, zero deletions at commit — caught by the suite's tracked-state snapshot).
- **Acceptance criteria (official matrix, pre-launch)** — 12/12 cells provenance-valid rows under one manifest at the pushed HEAD; OOM rows legal for orbit cells only if any; attribution fit {FA1,C4L,C8} per rung; VAN cells must RUN (VANCKPT semantics); COMMIT FREEZE from submission until the queue clears.
- **Next** — matrix → collect → report → pin commit → combined reviewer sign-off (pins) → SMOKE → sign-off → launch all four arms.

## 2026-08-06T (early) — official matrix run 86a752b: 7/12 cells killed by neu322 (uncorrectable ECC) → rerun with node excluded

- **Result** — `partial` (infrastructure, not code): all 32×2/16×4 cells except C8_32x2 landed on neu322 and died at CUDA init (identical 551 MiB peaks, `uncorrectable ECC error`; the node attracts small jobs because crashes keep its GPUs free). Survivors (all provenance-valid): VAN_8x8 1.0443 · FA1_8x8 1.0434 · C4L_8x8 0.3639 (5.9 GB) · C8_8x8 0.1876 (9.4 GB) · C8_32x2 0.1523 steps/s. Early read: 8×8 fastest for C8 (+23% vs 32×2); C4-family 16×4 ≈ 8×8 (supplemental CKPT4_16x4 0.366 ≈ C4L_8x8 0.364); orbit slope at 8×8 ≈ 0.60–0.65 s/step per extra ViT pass (near-linear) ⇒ projected C16 ≈ 0.097 steps/s (~4.8 d to 40k), C32 ≈ 0.049 (~9.5 d). Report + manifest committed as the failed-run record (`p0_report_86a752b.md`).
- **Analysis** — infra classification per SOP; no code change. Rerun binds `SBATCH_EXCLUDE=neu322` (sbatch honors the env default). neu322 should be reported to the cluster admins (Yixun's call — flagged in the next status).
- **Next** — official matrix rerun (attempt 2) → collect → pins → sign-off → SMOKE → launch.

## 2026-08-06 — OFFICIAL P0 REPORT SIGNED (run 1334933, 12/12 OK) → rung analysis → spot cells at 8×8

- **Result** — `passed`: steps/s (32×2/16×4/8×8): VAN 0.808/0.946/1.060 · FA1 0.802/0.941/1.030 · C4L 0.295/0.361/0.360 · C8 0.153/0.191/0.186. Peaks: C4L 15.9/9.5/5.9 GB · C8 28.6/15.9/9.4 GB.
- **Analysis** — orbit families: 16×4 ≈ 8×8 (within ~3%), both > 32×2. Per-pass retained memory ≈ 3.16/1.62/0.88 GB at micro 32/16/8 ⇒ C32 projects ~51 GB at 16×4 (OOM) vs ~30 GB at 8×8 (fits). Uniform-rung mandate (Q3) + C32 feasibility ⇒ **candidate pin = 8×8**, pending the pre-registered C16/C32 spot verification at that rung. Projections at 8×8 (slope ~0.60–0.63 s/step/pass): C16 ≈ 0.098 steps/s (~4.7 d to 40k), C32 ≈ 0.050 (~9.2 d).
- **Next** — `spot 8x8` (C16, C32; 30 steps) → pin commit → reviewer sign-off → SMOKE @8×8 → launch.

## 2026-08-06 — PIN COMMIT (launch precondition 5): rung 8×8, all values from the signed P0 evidence

- **Pins** — RUNG 8x8 / MB 8 / NGPU 8 (only rung where all four arms fit: per-pass retained memory 0.88 GB at micro-8 ⇒ C32 measured 30,817 MiB; 16×4 projects C32 ~51 GB OOM); MIN_FREE_MB 35,500 (max-across-arms floor, reviewer-allowed form); MAXSTEPS 40,000 / ckpt 2,500 (unchanged); time limits C4L 42 h · C8 80 h · C16 150 h · C32 167 h (partition cap) with **C32 pre-registered as a two-segment run** (wall-stop ~step 30–31k → reviewed RESTART mode to 40k, wandb resume=must); P0 manifest sha b2aeaf9c…208e (official run 1334933), spot manifest sha recorded in the commit message.
- **Justification trail** — official 12/12 report `p0_report_1334933.md` + spot `p0_report_spot_9bf1936.md` (C16 0.0982 steps/s / 16.6 GB; C32 0.0518 / 30.8 GB — linear-model predictions confirmed within 4%). Projected training: C4L ~31 h · C8 ~60 h · C16 ~113 h · C32 ~215 h (2 segments).
- **Next** — SMOKE=1 at the pinned rung → combined Codex sign-off (pins + smoke evidence) → launch.

## 2026-08-06 — batched-orbit code COMPLETE (reviews closed) → measurement sequence

- **Version Control** — batching rounds: 1479304/d4164e8 (impl+probe) → review (REJECT 7B+2N; headline: train-mode DINOv3 stochastic RoPE rescale makes batching a recipe change, not reordering) → Yixun Q6 PROCEED (disclosed recipe change; C4L sole inferential comparator; historical rows legacy-loop) → fixes 8094d60..10c41e1 → re-review (contract CLOSED; 3B+3N periphery) → d40f125 (PIPESTATUS array-clobber root-caused + wrapper block extracted-and-tested 10/10; CUDA fail-closed; exact record ids; real draw topology pinned; bf16 in summary; vanilla provenance n/a; comparator guard keys). 183 pytest + 10 wrapper guards green.
- **Next (measurement, per the preconditions)** — equivalence probe (gates the batched path) → FULL batched P0 matrix + 8×8 spot (revalidates the rung choice under the new execution profile — the optimum may shift) → re-pin → SMOKE → final sign-off → launch.

## 2026-08-06 — EQUIVALENCE PROBE PASS (attempt 6, job 3646653): batched path QUALIFIED

- **Evidence chain (6 attempts, all preserved)** — 3646612/3646615: no-GPU allocations (missing --gres directive, root-caused, file now self-sufficient); 3646616: train-C32 dual-graph OOM (cell moved to the 8×8 spot qualification, teardown hygiene added); 3646626: verdict=FAIL exposing the TF32 defect (the 'fp32' gate ran under matmul-precision 'medium'; B=1 GEMV vs GEMM compared different precisions — band 3.5–5.4e-4 = TF32 2⁻¹¹); 3646634: mm=highest, defect gone, measured fp32 shape-reordering envelope 0–1.979e-6 vs the too-tight 1e-6 pre-registration; 4077b45: bound adjusted-after-measurement to 5e-6 (pinned to the independently derived √384·2⁻²⁴ = 1.168e-6 noise scale; 2.53× above envelope, 70× below the caught defect, ~5 orders below a semantic slice error — flagged for final sign-off). **3646653: verdict=PASS, cells=13/13, gate_matmul=highest.**
- **Next** — batched P0: full matrix + spot 8x8 → re-pin → SMOKE → final sign-off → LAUNCH. Commit freeze while bound jobs queue.

## 2026-08-06 — SMOKE GREEN (attempt 3, job 3648568) — all launch preconditions except final sign-off satisfied

- **Batched P0 (signed, 14/14 OK)** — 8×8 batched: C4L 0.6598 (6.0 GB) · C8 0.4351 (9.5 GB) · C16 0.2454 (19.6 GB) · C32 0.1308 steps/s (32.1 GB) — speedups 1.83–2.53× over loop; 8×8 unambiguous fastest rung; C32 single-segment 3.54 d. Reports: p0_report_batched_matrix.md / p0_report_batched_spot.md.
- **Re-pin** — `ea94995` (floors/limits/manifest sha from batched evidence). **Launcher fixes** `71054cf` (wandb readback by run-id glob — PL save_dir overrides WANDB_DIR, train.py:165; scontrol-derived transcript path fail-closed; separator hygiene) + 15 readback unit tests + 6 guard cases (suite: 208 pytest + 16 guards green).
- **SMOKE evidence (jid 3648568 + prior 3646734's training-side)** — COMPLETED 0:0; torchrun/tee/wandb-identity/classification all rc=0; 8 ranks registered (PL literal); 30 steps; dual durable logs byte-identical; wandb run identity verified against manifest; checkpoint preflight PASS: global_step=30, fa_invariant C4 orbit embedded, optimizer FULL (449), scheduler last_epoch=30, EMA 212 entries, sha 5ad2053b…92f2. First-smoke artifacts preserved at outputs_FLAC/exp11_smoke/C4L_run1_jid3646734 (duplicate-run guard correctly refused reuse; dir renamed, not deleted).
- **Next** — final combined Codex sign-off (preconditions 5–8: re-pin vs batched reports, adjusted equivalence bound, smoke evidence) → LAUNCH C4L/C8/C16/C32.

## 2026-08-06 — LAUNCH-APPROVED (final Codex sign-off) → ALL FOUR ARMS SUBMITTED

- **Sign-off** — `fa_orbit_codex_signoff.md`: LAUNCH-APPROVED; pins vs batched P0 verified; adjusted equivalence bound judged pinned-to-predicted-noise (not fitted-to-pass); smoke evidence (runs 3646734 training-side + 3648568 fully green) satisfies precondition 7; acceptance-record contents enumerated and satisfied (see fa_orbit_command.md).
- **Result** — `launched`: C4L job 3648665 (24 h limit) · C8 3648666 (35 h) · C16 3648667 (60 h) · C32 3648668 (112 h); each 8× L40, MB 8, 40k steps, ckpt/2500, batched fa_invariant, commit-bound 4884abc-era HEAD (see intent manifests). Expected completions from batched P0 rates: ~17 h / ~26 h / ~45 h / ~85 h.
- **Next** — first-ckpt health checks (~63/96/170/320 min); round 4 kit (fa-protocol screens at matched cadence, C4 backfill, D1/D2/D3 probes, R1–R4 readouts) developed while arms train.

## 2026-08-07 — ALL FOUR ARMS RUNNING (3648694–97); attempt-1 gate-kill logged

- **Result** — `launched` (confirmed): C4L/C8/C16/C32 running on four full nodes, 8-rank registration verified on each. Attempt-1 (3648665–68) was killed by the commit-freeze trap (my post-submission record commit — third occurrence; rule hardened: no tracked changes between submission and all-jobs-started; records now written post-start). Expected completions from batched rates: C4L ~17 h, C8 ~26 h, C16 ~45 h, C32 ~85 h from start.
- **Next** — first-ckpt health checks (63/96/170/320 min per arm); round-4 kit (screens/backfill/probes/readouts) development begins now so screens are ready before C4L@40k.

## 2026-08-07 — MEASUREMENT CAMPAIGN LIVE: first 13 screens complete (jobs 3649915–27, all validated)

- **Operating conditions (verbatim per the final GO, `fa_orbit_codex_measurement_GO.md`)** — C1 single serialized submitter; C2 campaign freeze continuously engaged (engaged 2026-08-07T04:13:33); C3 pushed-HEAD submissions via the locked submitter; guards not run mid-campaign.
- **First orbit-comparison numbers (single-seed s42 K8 screens — trajectory context only, NOT confirmatory):**
  - @5000 (three-way matched step): R@1 monotone in orbit size — C16 1.199 > C8 1.120 > C4L 1.026; T60/C50/EDT within screen noise (10.22–10.26 / 1.504–1.519 / 55.0–55.6).
  - @2500: C8 R@1 0.710 > C4L 0.663. @7500: C8 1.767 > C4L 1.673. @10000: C4L edges C8 on all four (9.64/1.248/46.9/2.178 vs 10.14/1.301/50.4/2.146).
  - C4 backfill (historical curve for the 20k/30k gates): @20000 9.216/1.0742/43.45/R 3.219 · @30000 8.859/1.0592/39.46/R 4.134.
- **Analysis** — retrieval shows a consistent (if small) early finer-orbit edge, cleanest as the three-way monotone ordering at 5k; acoustic-parameter metrics are within the known screen wobble. No gate is triggerable yet (arms below 20k). Everything defers to the 20k/30k gates and the 40k R1 paired readout.
- **Next** — rolling screens as checkpoints land (C4L 17.5k/20k, C8 12.5k, C16 7.5k/10k, C32 first); C4L conf cells at 40k tonight.

## 2026-08-07 — C4L@40k CONFIRMED (5-seed × 2K, validated+hash-verified) → first exp_11 table row; BRIDGE EFFECT MEASURED

- **Row (fa eval batched)** — K8: 8.414 ± 0.006 / 1.0095 ± 0.0009 / 41.499 ± 0.048 / R@1 5.119 ± 0.126 · K1: 9.761 ± 0.051 / 1.0822 ± 0.0054 / 44.026 ± 0.249 / 4.952 ± 0.134. All four arm row specs added to the generator (C8/C16/C32 render pending until their 40k).
- **Bridge effect (the finding)** — C4L vs historical legacy-loop B-F@40k at K8: T60 +0.212, C50 +0.0317, EDT +2.71, R@1 −0.268 — the recipe/environment delta (8×8 rung + batched orbit + chunk-shared RoPE + L40 + torchrun DDP) shifts the absolute level measurably. **This is precisely why the review-mandated C4L bridge exists: orbit inferences (R1/trend) run against C4L, same recipe, and are untouched; cross-era comparisons vs historical rows would conflate this shift with the orbit effect and are excluded by the table's non-interchangeability rule.**
- **Gate note** — the pre-registered futility comparator is the C4 backfill (historical recipe); given the measured bridge shift, gate verdicts will be reported with BOTH the registered backfill comparison and the C4L-trajectory context; no arm is near a kill (all tracking C4L's curve).
- **Next** — C8@40k tonight → conf block; C4L trajectory analysis (17.5k–37.5k screens landed); R3 rotations + D-probes scheduling; C16 Aug 8, C32 Aug 10 → R1.

## 2026-08-08 (overnight) — C8 conf block attempt 1 refused (10× exit 2): mid-round intermediate commit; operating rule added

- **Result** — `partial` (infra/process, zero GPU cost): all 10 C8 conf jobs bound their worktree to 064e8e0b — an intermediate commit of the in-flight r3/cross coder round where fa_orbit_screen.sbatch references eval_FLAC.rot_token before eval_FLAC exports it; every job died at the render gate in seconds. C8@40k ckpt safe; C8 COMPLETED 0:0 in 27h54m.
- **Operating rule (C4, appended to the campaign conditions)** — measurement submissions ONLY at round-closed SHAs: no submission while a coder round is open on measurement surfaces; verify suite-green HEAD before submitting.
- **Next** — r3/cross round closes + review → resubmit C8 conf block at the closed SHA.

## 2026-08-08 ~04:45 — overnight measurement HALTED (stop-loss); training arms unaffected

- **Triage (all infra, zero wrong numbers):** (1) cluster ECC wave — uncorrectable-ECC failures across ≥8 nodes (neu301/303/305/306/317/319/322/332), small jobs funneling into sick GPUs; (2) **SBATCH_EXCLUDE env is dropped by the submitter's --with-lock re-exec** — proven: third-batch jobs landed on excluded neu303/332 — so no batch tonight actually had node exclusion; (3) two aux-LOG path mistakes of mine (tee into the pinned worktree; then node-local /tmp scratchpad paths breaking class-7 log provenance). Survivors: C8 trajectory screens 27.5k–37.5k + C16@20k gate screen (validated).
- **Daylight fix list (small, reviewable):** submitter accepts EXCLUDE= argument → explicit sbatch --exclude (no env reliance); driver default LOG names gain cell/rot/orbit qualifiers (removes any need for LOG overrides); sick-node list to cluster admins (Yixun).
- **Scorecard:** C4L row PUBLISHED ✓ · C8 trained+COMPLETED ✓ (row pending eval infra) · C16 ~28k ✓ · C32 ~17k ✓ · r3/cross kit built+reviewed+GO ✓ · campaign conditions held throughout (freeze on, fail-closed everywhere).

## 2026-08-08 ~05:00 — C16 COMPLETE-in-substance (class-7 postmortem); third arm banked

- **Result** — C16 job 3648696 exited class 7 after 1d22h, but the substance is verified intact: ckpt global_step 40000 / epoch 8 / fa_invariant n_angles 16 / optimizer 449 entries / EMA 210 entries. The failure was the end-of-run dual-log verification; probable cause: my git rebase --autostash cycles (remote coordination with the other-machine session) stash-cycled the TRACKED live train log while the job appended — clobbering the tee'd copy mid-run. Classification honest: infra postmortem, training valid, row proceeds after conf evals.
- **Lessons → fix round:** (1) submitter gains an explicit EXCLUDE= argument passed as sbatch --exclude (env proven dropped by the with-lock re-exec); (2) driver default LOG names gain cell/rot/orbit qualifiers; (3) live arm transcripts become untracked during runs (git rm --cached at launch, committed at completion) so remote coordination can never touch appending files.
- **Standing:** THREE arms banked (C4L row published; C8+C16 ckpts verified, rows pending eval infra). C32 sole trainee (~20k). Submissions remain halted until the fix round closes.

## 2026-08-08 (daylight) — FIX ROUND: exclusion root cause CORRECTED, log naming, live transcripts untracked

- **Root cause (correction to the ~04:45 triage).** The exclusion failure is **not** an env drop through the
  `--with-lock` re-exec. Measured directly: `SBATCH_EXCLUDE` survives that chain intact (plain child, and
  after the helper's `exec`, both report the value). The real cause is that **`SBATCH_EXCLUDE` does not
  exist**: `man sbatch` (Slurm 25.11.6) documents 58 input environment variables and there is no `--exclude`
  equivalent among them — the lookalike is `SBATCH_EXCLUSIVE`, which is `--exclusive`, a different option.
  sbatch therefore ignored the variable in silence, which means **no batch of the campaign ever had node
  exclusion in effect**, not merely the third. Earlier batches avoided the sick nodes by luck.
  Fix: `EXCLUDE=<nodelist>` argument → explicit `sbatch --exclude=<nodelist>`; and a *set* `SBATCH_EXCLUDE`
  is now refused with a pointer to the argument, so the silent-ignore path cannot recur.
- **C16 clobber CONFIRMED, with the mechanism refined.** Not "reset to an old committed blob": the on-disk
  transcript (8,084,545 B) is far larger than the only committed blob for that path (970,913 B, `d960990`).
  The mechanism is **descriptor detachment**. All four arm transcripts have the identical frozen mtime
  `2026-08-08 02:04:07`, spanning the `rebase (start)…rebase (finish)` cycle at 01:23 and the 02:04 commit
  (`a9af0d6`); each git working-tree write unlinks the path and creates a new inode, while the running job's
  stdout descriptor keeps pointing at the old, now-nameless one. C16's visible transcript stops at Epoch 5
  (1:44:21 in) — ≈28 h into a run that reached step 40000 — while its untracked tee'd copy holds the whole
  run (Epoch 8, 13,336,561 B). **Live proof**: C32 is still running; its tee'd log is at Epoch 4 and growing
  (mtime 19:28), while its tracked `.out` froze at Epoch 2 at 02:04:07. Substance was never at risk — the
  checkpoints and the untracked tee'd copies are intact.
- **Actions in this commit** — (1) `EXCLUDE=` argument → explicit flag (+ cell parameters exported with the
  job); (2) default screen-log names carry CELL, the rotation or eval-orbit token and the job id, so no
  `LOG=` override is ever needed (the three overnight path incidents all came from overrides) — proven: six
  same-second cells produce six distinct names, and without the tokens two R3 rotations collide exactly;
  (3) the arm launcher `git rm --cached`s its own Slurm transcript at launch (tolerating absence, recording
  the outcome in the manifest, closure documented in the header), and **C32's live transcript is untracked
  in this commit** so remote coordination can no longer touch it.
- **Validation** — 114/114 screen guard cases (deletion/thaw cases skipped: campaign freeze engaged), 200 pytest.

## 2026-08-09 — R1 (partial, C4L/C8/C16): finer orbits DEGRADE acoustic metrics monotonically; retrieval flat

- **Rows published (5-seed, both K, campaign-pinned SHA 0c6e9ff, all validated+hash-verified).** Seed-paired deltas vs C4L (mean ± 95% paired-t CI, df=4), K8: C8: T60 +0.299±0.005, C50 −0.005±0.002, EDT +1.461±0.042, R@1 +0.063±0.175 (n.s.) · C16: T60 +0.929±0.023, C50 +0.014±0.003, EDT +3.854±0.093, R@1 +0.069±0.197 (n.s.). K1 mirrors K8.
- **Co-primary verdicts (pre-registered: K8 T60 + K8 R@1, Holm):** C8 and C16 both SIGNIFICANTLY WORSE on T60 (p≪0.001), EQUIV on R@1 ⇒ per §4 rules both arms verdict DEGRADED. Trend C4L→C8→C16: monotone worsening in T60/EDT (≈ linear in orbit size); R@1 flat (the early 5k monotone-improving retrieval ordering did NOT persist to 40k).
- **Reading (conditional on training seed 42, matched STEPS not compute):** refining the frame-average past C4 costs acoustic-parameter accuracy roughly linearly and buys nothing measurable in retrieval — while consuming 2–4× the compute. C8/C16 are strictly dominated at this budget. C32 (due ~Aug 10) completes the trend; R2/R3 cells will show whether yaw-flatness improved even as headline metrics degraded (mechanistically informative either way).

## 2026-08-09 — Q' = bb9de07 APPROVED; campaign endgame armed

- **State** — all measurement code signed off. Sequence on arm completion: C32 conf block @ campaign pin 0c6e9ff → C32 row → --pin-campaign bb9de07 → q9 blocks (VANL+C4L × K1/K8, seeds 42–46; one-pin gate enforced) → VANL row + within-pin frame-averaging delta → figure/table regeneration → closing package (R1 full + R2/R3 + results/analysis/HTML). VANL ~10 h out; C32 ~12 h out.
- **Acceptance criteria (endgame, pre-registered):** C32 conf = 10/10 validated cells at 0c6e9ff; q9 = 20/20 validated at bb9de07 with check_q9_round PASS (single source_sha); every row through the generator's contract-correct gates; regression guard green on each regeneration.

## 2026-08-09 — K=1 trajectory program + paired-K figures

- **Data** — screen contract extended to K∈{1,8} (f893037; gates pinned K8-only via gate_K). 45 K=1 screens submitted; 44 evals completed and wrote valid records; every job self-refused SCREENRESULT because the PINNED worktree (0c6e9ff) carries the pre-K1 validator — **post-hoc validation under the current contract: 44/44 VALIDATED** (futility contract, hashes skipped for these figure-only cells; sidecar cross-checks intact). Deviation disclosed: figure-only cells validated out-of-band rather than in-job; no gate/table use. The 45th (backfill K1@40000) was gate-refused at the pin (manifest entry postdates it) and is omitted with disclosure.
- **Figures** — fa_orbit_trajectories_K8.png + fa_orbit_trajectories_K1.png (+ per-metric K-named PNGs; _all.png = K8 alias); HTML carries both K blocks with per-K provenance notes (P1 legacy has no sub-40k K1 raws; backfill K1 = 20k/30k only).

## 2026-08-10 — VANL COMPLETE-in-substance (class-7 = legacy tee casualty); C32 ~12h out; restart legs queued

- VANL job 3661520: 11h17m, 40k ckpt verified (vanilla dispatch, no angles key, full opt/EMA); class-7 = `tee exited 1` on the tracked transcript copy whose inode was detached by pre-gitignore rebases — same mechanism as C16, primary tee log intact. Row comes via the q9 block at the new pin per the Q′ prescription.
- Queue note: restart legs PENDING behind the other session's exp05 program; C32 on schedule (~midday).

## 2026-08-10T02:10:00-04:00 — RE-PIN REVIEW FIX ROUND 2/2: items 1, 2, 3 closed (06362f4, ac98e11, 60d4e31, 9469849)

- **Goal** — close the re-pin review's remaining Required-fix items (1 restart time pin + extension
  preflight, 2 per-checkpoint producer binding, 3 recorder hardening). Items 4 and 5-validator were
  closed by the previous round (1c18920); 5-figures and 6 (C32 anchor + recorded legs in the final
  pin) remain open and are NOT this round's scope.
- **Change** —
  - `fa_orbit_train.sbatch` (**gate-only**, no launch-path semantics): the wall pin follows the LEG
    (`EXPECTED_STEP > 0` → `PINNED_TIME_LIMIT_RESTART_<ARM>`, matching the submitter), gate H
    enforces the SELECTED pin by name, the lineage line prints it, and a real (non-smoke) restart
    now runs the preflight with `--extension --launch-registry`.
  - `fa_orbit_ckpt_preflight.py`: `check_extension_binding()` — the 40k→100k contract. Identity is
    bound to the COMMITTED registry (manifest bytes, arm/job/uuid/LAUNCH commit/rung/config sha/
    save-dir/seed 42, resume ckpt == the audited `final_ckpt_sha256` at `final_step`, in the
    canonical run directory); the budget may only rise; the INITIAL budget/commit are NOT required
    to equal the extension's — the review's exact wording.
  - `fa_orbit_producer_manifest.py` (new): append-only per-leg `step → sha256 → path`, plus
    `validate_leg()` / `verify_chain()`.
  - `fa_orbit_record_restart.py` (rewritten): resume file must exist, must be canonical, always
    re-hashed; all identity fields checked against the INITIAL row and the launcher's own Q10 pins;
    atomic tmp+rename under an exclusive lock on the registry directory; duplicates refused; the
    leg's producer manifest published/extended in the same transaction.
  - `fa_orbit_screen.sbatch` >40k branch: re-hashes the checkpoint and admits it only on an exact
    step/sha256/path match from a fully re-validated leg (and the leg's own restart manifest is
    re-hashed too).
- **Design choice (fix 2)** — the review allowed either the leg's JOB hashing checkpoints as it
  saves, or the RECORDER capturing the inventory into an append-only per-leg file. Chose the
  recorder: the first requires editing `fa_orbit_train.sbatch`'s training path while jobs
  3662828-30 are queued against it (forbidden this round) and would put sustained multi-GB reads
  beside a live training job on the shared filesystem. The recorder-side file is still immutable
  evidence in the sense that matters here: it is tracked, screens read it from the PINNED worktree,
  so a row can only become evidence by being committed into the campaign pin.
- **Command / Validation** —
  - RED: `fa_orbit_2026-08-10_01-24-26_guardtests.log` (time pin + extension: 12 new cases failing),
    `fa_orbit_2026-08-10_01-37-34_failopen_repro.log` (HEAD recorder records a leg for a resume file
    that does not exist, rc=0), `fa_orbit_2026-08-10_01-44-34_screen_guardtests.log` (a same-config
    checkpoint from a WRONG restart **accepted**, `want rc=2 …, got rc=0`, eval argv built).
  - GREEN: `fa_orbit_2026-08-10_01-57-49_guardtests.log` 87/87 · screen 172/172
    (`fa_orbit_2026-08-10_01-49-16_screen_guardtests.log` = 171/171 before the last clause) ·
    `test_exp11_restart_record.py` 46/46 · full suite 572 passed / 8 skipped
    (`fa_orbit_2026-08-10_01-53-27_pytest_fix2gate.log`).
  - Real-data rung: the queued C4L leg's extension preflight passes against the REAL registry and
    the real 40k checkpoint — launch job 3648694, launch commit `2b78f99`, running commit `1c18920`,
    `CKPT_SHA256 ed9d7a869ecded98…` == the registry anchor.
- **Result** — `fix_ready`. Items 1/2/3 closed at `9469849`. Nothing submitted; no pushes; the
  campaign freeze and the queue were not touched (3662828-30 still PENDING, C32 still RUNNING).
- **Analysis** — all three were *real bugs* in provenance machinery, not infrastructure. Note that
  the queued legs still carry `EXPECT_SHA=c85bc61` and will abort on the commit gate: they must be
  cancelled and resubmitted at the final reviewed SHA, which is the review's own instruction and is
  the operator's call, not this round's.
- **Next** — Codex review of this round; then items 5-figures and 6 (C32@40k anchor, record the
  legs with the hardened recorder, include those records in the single-pin candidate) before any
  q9/traj/VANL submission.
