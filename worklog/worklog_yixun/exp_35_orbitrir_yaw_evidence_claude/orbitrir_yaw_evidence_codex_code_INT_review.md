**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** code INT (exp_35 integrative: the whole branch 682df5db → 5b065801e9d1c1844461ec84cd0558702c714326 + the Planner's phase-(i) launch script)

**Verdict:** REQUEST-CHANGES — the composed H35 branch is statically SHIP-ready, conditional on C3e, but the current Planner launch script does not cryptographically or procedurally bind `launch` to a successful, current `capture`. A failed or stale capture can still furnish inputs that `launch` accepts.

## Ranked findings

### BLOCKING — `launch` is not bound to a successful, current capture

Evidence: `exp35_phase_i_launch.sh:183-207,210-221,244-250`.

`capture` creates the tuple, its main-checkout copy, dry-run transcript, and printed launch command before testing `FAIL`. Consequently, a capture containing failed CHECKs still leaves a valid tuple and usable launch line. `launch` accepts any regular tuple whose bytes and live identity validate; it does not require:

- `CAPTURE OK`,
- a capture-log hash or success seal,
- the canonical tuple path under `RUNX`,
- the byte-identical main-checkout copy,
- the reviewed launch-script hash,
- or the external state captured earlier.

It reruns only `common_checks`, tuple identity and supervisor dry run. Mutable inputs checked only by `capture`—VAE/AGREE bytes, DINO cache, NAS/data availability, environment, external exp_30 helper pins and GPU-memory margin—can change between capture and launch without refusal. H35 does not digest all those external inputs.

Concrete fix:

1. Do not create or print a launchable tuple/command while `FAIL != 0`.
2. After all checks and the dry run pass, create a noclobber capture-seal file containing H35, expected launch-script SHA-256, canonical tuple path/SHA-256, main-copy SHA-256, campaign digest, external asset/environment pins, and capture-log SHA-256.
3. Pass the seal path and SHA-256 to `launch`; require a canonical `RUNX` tuple and byte-identical `MAINX` copy.
4. Recheck mutable external inputs immediately before detachment.
5. Hard-pin the reviewed script SHA `4adeb160af795fb5f1a43fcb7340c885dce6a66e1bc72a9e9c4ce85544b9d12e`.

### HIGH — several launch-entry prerequisites are recorded, not enforced

Evidence: `exp35_phase_i_launch.sh:118-119,150-155,164-179`.

- The offline DINO snapshot is only `rec`; an absent or incomplete cache does not refuse capture.
- NAS manifest matches are only `rec`.
- The interpreter/`torch` import may fail without affecting `FAIL`; only `pip freeze` is judged.
- The exp_22 watcher-file check ends with `true` and is only recorded. Only a currently running watcher is rejected.
- The script does not hard-record the final ladder hashes/status or assert the expected campaign digest `da82d012…`.
- Only one A6 CSV is hard-pinned at `:132`; all approved external A6 inputs need launch-entry pins.

Concrete fix: convert required availability/import/manifest/watcher checks to `need`, assert the exact DINO snapshot and all §8 external hashes, independently calculate the 116-file digest, and require the final ladder evidence—including the eventual Ladder 4 passing summary—before producing the capture seal.

### NIT — “read-only except new evidence files” is not fully guaranteed

Evidence: `exp35_phase_i_launch.sh:154-155`.

Two Python invocations omit `-B`/`PYTHONDONTWRITEBYTECODE=1`. They could create bytecode in an otherwise writable environment. Use `"$PY" -B ...` consistently and include `PYTHONDONTWRITEBYTECODE=1` in `LAUNCH_ENV`.

## I1 — PASS: end-to-end phase (i)

The composed sequence is correct:

| Chunk | Trace and terminal gate |
|---|---|
| CTL K1, chunk 1 | `chunk_plan` → one-cell runner with `EXPECT_HEAD`, `--only`, lane and shared tuple → no-flash launcher/overlay → vanilla P1 evaluator → receipt v2 → `--verify-cell` → E1 K1 byte/receipt equality → final guard → `gates_verified`. |
| B-F T s42 K8, chunk 4 | Same launch chain with `fa_invariant`/four-angle averaging → receipt verification → E2. E2 PASS/FAIL selects the 45° source; refusal is the hard stop. |
| Probe, chunk 5 | Supervisor directly invokes the pinned no-flash launcher with the registered BF probe argv → finalization → `PROBE_CONTRACT_BF` → sealed probe ledger → final guard. |
| Ordinary B-F cell | One-cell runner → pinned checkpoint/config/backend → receipt and `--verify-cell` → final guard. |
| GPU 0 hand-over | Two zero-process observations 60 seconds apart → O_EXCL GPU 0 reservation → state commit/epoch increment → authenticated removal of the GPU 1 reservation. |

Relevant code: `exp35_supervisor.py:53-63,493-506,588-711,713-820,822-884`; `depth_canon_eval_run.sh:128-207,366-418,495-521`; `exp35_no_flash_launch.py:169-199`; `depth_canon_collect.py:583-667`; `eval_FLAC.py:2268-2299`.

State advances `planned → in_flight → runner_verified → gates_verified`. The final-guard record is mandatory for a `gates_verified` chunk (`exp35_supervisor.py:185-201`). E1 mismatch, probe/finalization failure, receipt/backend/tree/checkpoint failure, guard failure, identity/reservation failure and nonzero runner exit all latch before the loop can spawn another chunk.

A violation detected while a child is already running latches but deliberately lets that child finish. Neither the supervisor nor launch script signals the runner or evaluator (`exp35_supervisor.py:713-738`). That matches the plan.

## I2 — PASS: one campaign digest covers phases (i) and (ii)

`depth_canon_eval_run.sh:128-154` selects the following 116 current files for every exp_35 selection. The digest independently recomputes to:

`da82d0121413b792e2f42b94026b884d371d995c17e2dcc111c7edd58f8650c1`

Complete expanded set:

- `data/AR`: `all_data.json`, `seen_eval.json`, `train.json`, `unseen_eval.json`
- Root/overlay: `eval_FLAC.py`, `overlays/no_flash/exp35_no_flash_launch.py`, `overlays/no_flash/flash_attn.py`
- `src`: `__init__.py`
- `src/configs/dataset_configs/AR/eval`: `acousticroom_seeneval.json`, `acousticroom_seeneval_1.json`, `acousticroom_seeneval_4.json`, `acousticroom_unseeneval.json`, `acousticroom_unseeneval_1.json`, `acousticroom_unseeneval_4.json`
- `src/configs/dataset_configs/AR/train`: `acousticroom_train.json`, `acousticroom_train_VAECtxt.json`
- `src/configs/dataset_configs/HAA/eval`: `haa_test.json`, `haa_test_1.json`, `haa_val.json`
- `src/configs/dataset_configs/HAA/train`: `haa_train.json`
- `src/configs/dataset_configs/RAF/eval`: `raf_diagnostic.json`, `raf_test.json`, `raf_val.json`
- `src/configs/dataset_configs/RAF/train`: `raf_train.json`
- `src/configs/dataset_configs/custom_metadata`: `AR_md.py`, `HAA_md.py`, `RAF_md.py`
- `src/configs/model_configs/FLAC/AR`: `FLAC_AR.json`, `FLAC_AR_AllCA.json`, `FLAC_AR_CylViT.json`, `FLAC_AR_CylViT_PE_CNN.json`, `FLAC_AR_CylViT_PE_Linear.json`, `FLAC_AR_InContext.json`, `FLAC_AR_S.json`, `FLAC_AR_SimpleViT.json`, `FLAC_AR_VAECtxt.json`, `FLAC_AR_noGeom.json`
- `src/configs/model_configs/FLAC/HAA`: `FLAC_HAA_finetune.json`
- `src/configs/model_configs/FLAC/RAF`: `FLAC_RAF_finetune.json`
- `src/configs/model_configs/VAE`: `VAE.json`
- `src/data`: `__init__.py`, `are_anchor.py`, `dataset.py`, `utils.py`, `yaw_rotation.py`
- `src/inference`: `__init__.py`, `generation.py`, `sampling.py`, `utils.py`
- `src/interface`: `aeiou.py`
- `src/localization`: `__init__.py`, `agree_embed.py`, `candidates.py`, `crossarm.py`, `geometry.py`, `meshgrid_crossarm.py`, `meshgrid_drift_measurement.py`, `meshgrid_engine.py`, `meshgrid_fa_parity.py`, `meshgrid_geometry.py`, `meshgrid_offgrid_probe.py`, `meshgrid_queries.py`, `meshgrid_report.py`, `meshgrid_retrieval_control.py`, `meshgrid_segments.py`, `metrics_report.py`, `reaggregate.py`, `rir_metrics.py`, `scoring.py`
- `src/metrics`: `metric_callback.py`
- `src/metrics/modules`: `C50.py`, `EDT.py`, `Env.py`, `FD.py`, `RT60.py`, `Retrieval.py`, `l1_stft.py`, `l1_stft_multires.py`
- `src/models`: `__init__.py`, `autoencoders.py`, `blocks.py`, `bottleneck.py`, `conditioners.py`, `cyl_vit.py`, `diffusion.py`, `discriminators.py`, `dit.py`, `encodec.py`, `factory.py`, `pretrained.py`, `pretransforms.py`, `simplevit.py`, `transformer.py`, `utils.py`
- `src/tools`: `__init__.py`, `depth_canon_stats.py`, `retune_lr_state.py`, `strip_optimizer_state.py`
- `src/training`: `__init__.py`, `autoencoders.py`, `diffusion.py`, `factory.py`, `utils.py`
- `src/training/losses`: `__init__.py`, `auraloss.py`, `losses.py`
- `worklog/worklog_yixun/exp_07_fa_scratch_claude`: `FLAC_AR_BF.json`, `FLAC_AR_BVp1.json`
- `worklog/worklog_yixun/exp_15_yaw_aug_claude`: `FLAC_AR_YAWAUG.json`
- `worklog/worklog_yixun/exp_25_depth_canon_claude`: `FLAC_AR_CANON.json`, `depth_canon_audit_gate.py`, `depth_canon_cells.json`, `depth_canon_collect.py`, `depth_canon_eval_run.sh`, `depth_canon_make_cells.py`
- `worklog/worklog_yixun/exp_35_orbitrir_yaw_evidence_claude`: `exp35_gpu_identity.py`

The manifest already contains OFF-P1, OFF-BF, OFF-CANON, CANON-45 and E2-FAIL. E2-FAIL is default-excluded but available under explicit selection (`depth_canon_collect.py:670-684`). The existing collector already implements artifact completeness, per-`(method,K)` E2-FAIL fallback, tables (a)/(b)/(c), Tier 2/C4 and Tier 3 (`depth_canon_collect.py:2007-2192,2207-2279`). No collector or results-code change is needed after H35.

`analysis/yaw_robustness/plot_yaw_curves.py` is outside `RELEVANT`. The future E5 script must likewise be created outside these pathspecs—preferably under `analysis/yaw_robustness/` or the exp_35 worklog directory—and its exact path/hash should be pre-registered. Phase (ii) must separately re-pin the external CANON manifest/checkpoint.

## I3 — PASS: exp_25 mode and evaluator closure

The requested closure diff:

```text
git diff d6ca307a 5b065801... -- eval_FLAC.py src ':(exclude)src/tests' data/AR
```

contains only:

```text
src/localization/meshgrid_crossarm.py
src/localization/meshgrid_engine.py
src/localization/meshgrid_fa_parity.py
src/localization/meshgrid_offgrid_probe.py
src/localization/meshgrid_report.py
src/localization/meshgrid_segments.py
```

That is exactly the exp_22 localization merge; `eval_FLAC.py`, non-localization `src`, and `data/AR` are unchanged.

Ladder 1’s supplied SHA-256 matched before reading. Its equivalent-environment guard reports the exp_25 results JSON at `ec3fcd08…` and the Markdown byte-identical. C3d C7 therefore still holds at H35.

## I4 — PASS: phase-(i) scientific contract

No plan drift found.

- Manifest population: two CTL cells; BF T/R seeds 42–46 at K1/K8; T duplicate K1/K8; C4 90/180/270/45 at K1/K8, seed 42.
- BF argv: `--cond-method fa_invariant --frame-avg-angles 0,90,180,270 --frame-avg-max-fwd-samples 64`.
- CTL argv remains vanilla.
- E1 checks byte/receipt equality against the force-added exp_25 references.
- E2 has all 14 tolerances and determines only the 45° source.
- Probe registration is BF, n=64, batch=64, seed=42, CUDA, registered BF checkpoint/config and unseen K8 dataset (`orbitrir_bf_cond_probe.py:487-491,638-729`).
- Required Tier 2 dumps and pooled/pair/duplicate/C4 summaries are present; Tier 3 includes all 28 cells and FD.
- Guard thresholds are exactly W500 `9.712`, mean `8.521`, W100 `9.950`, with strict greater-than refusal (`exp35_training_guard.py:34-105`).
- Predictor is exactly `log_mtime + (40000-step)×3.5s > now + 2×planning + 1800s` (`exp35_supervisor.py:73-81`).

## I5 — FAIL: Planner launch script

Positive parts:

- `env -i` is sufficient statically: absolute conda Python, system tools in `PATH`, correct `HOME` for Hugging Face, no inherited `CUDA_VISIBLE_DEVICES`, and the runner/probe set lane, offline and overlay variables explicitly.
- Detachment is correct: the background subshell `exec`s `setsid nohup`, stdin is `/dev/null`, output goes directly to the supervisor log, and `$!` is the supervisor PID (`exp35_phase_i_launch.sh:222-241`).
- Nothing retains the outer `tee` pipe.
- The script contains no signal/kill operation and does not edit or attach to the exp_30 processes.
- Two concurrent launch attempts could both briefly fork, but the supervisor’s exclusive lock prevents two active supervisors (`exp35_supervisor.py:835-848`).
- Process matching compares argv-element basenames and avoids the usual `pgrep` self-match.

The FAIL is the BLOCKING capture/launch binding defect and incomplete hard admission checks described above.

## I6 — PASS with required launch-entry disclosures

Pre-register or disclose:

- Runner/collector imports may recreate ignored `__pycache__`/`.pyc` under `src/`; Ladder 3b demonstrated this. These files are ignored and outside the digest, but use `PYTHONDONTWRITEBYTECODE=1` globally or declare the behavior and cleanup.
- Shared-card memory is not isolation: a co-tenant allocation spike can OOM either job despite the 10 GiB admission margin.
- NAS/CIFS traffic or stalls can slow an in-flight cell. The supervisor deliberately does not kill it; a latch only prevents the next spawn.
- Guard and identity checks are sampled at up to five-minute intervals, so very short-lived interference can be missed.
- The predictor carries the approved faster-training/long-cell forecast risk.
- GPU 0 can acquire a new foreign process after the two free checks; the active watch will latch, but will not terminate the current cell.
- Record C3e’s verdict, all four ladder hashes, the eventual Ladder 4 summary, the reviewed launch-script hash, tuple/seal hashes and campaign digest.
- Pre-register the future E5 path/hash and require it to remain outside `RELEVANT`.

## I7 — FAIL for the combined launch package

`5b065801e9d1c1844461ec84cd0558702c714326` itself is ready to serve as H35 without another H35 code commit, conditional on C3e accepting `955ac38c`/`d3381fed`. The composed phase-(i) code is internally consistent and does not require a post-H35 collector change for phase (ii).

The current launch package is not ready because “capture passed” is not an invariant enforced by `launch`. Repair and review the Planner script, then require the sealed capture to pass. The Ladder 4 transcript was still two lines long when read—only its start banner and file list—so there is no pytest summary line to report; the Planner’s separately owned final-suite gate remains outstanding.

## What I verified

Read-only probes performed:

- Confirmed branch, HEAD, merge base and 38-commit range.
- Read the binding plan and prior C1–C3d reviews; did not duplicate C3e’s detailed fix-commit review.
- Inspected committed content with `git show`, full-range `git diff`, targeted diffs and line-addressed source reads.
- Verified all supplied launch/Ladder 1/2/3/3b SHA-256 values.
- Verified the launch script is byte-identical to main-checkout commit `8c71e258`.
- Expanded the runner pathspec to exactly 116 files and independently recomputed digest `da82d012…`.
- Checked manifest counts and arms with `jq`.
- Verified no relevant-tree dirty paths.
- Ran `git diff --check` narrowly over code/config files: clean. The unrestricted check reports trailing spaces only in committed historical transcript logs.
- Read Ladder 2/3/3b evidence and checked Ladder 4’s current tail/summary scan.
- Did not run tests, runner, supervisor, guard, probe, collector, launch script, evaluator, training, or any GPU command; did not signal or attach to a process.
