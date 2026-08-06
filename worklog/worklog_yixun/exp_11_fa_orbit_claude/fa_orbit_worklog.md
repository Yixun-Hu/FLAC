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
