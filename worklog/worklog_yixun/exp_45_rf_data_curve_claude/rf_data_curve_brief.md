# exp_45 rf_data_curve — brief for the neuronic Claude session

Written 2026-11-14 ≈ 19:15 EST by the A6000 rebuttal session (Claude Opus 5.5, `mae-cab-lab-server`), at
Yixun's request (`rf_data_curve_yixun_query.md`, this folder). **This is a brief, not a plan.** Per
`worklog/experiment_SOP.md` the neuronic session writes `plan_rf_data_curve.md`, gets Yixun's approval, and only
then implements (TDD, Opus 5.5 subagents for all code, Codex `gpt-5.6-sol` xhigh ≤ 3 plan + 3 code rounds) and
runs. Everything below marked **decided** is Yixun's decision and should not be asked again. Everything marked
**open** goes into the plan's decision table, with a recommendation.

The A6000 session has no neuronic access. It reads this branch and `exp-38-data-efficiency` from GitHub only.

## 0. Read first

1. `worklog/experiment_SOP.md` and every file in `worklog/worklog_yixun/announcement/`. The relevant ones are
   01 (full 6,337-item unseen eval), 04 (`model_comparison.md` regeneration), 05 (eval flags match training),
   07 (checkpoints under the NAS `checkpoints/`) and 08 (save predicted waveforms).
2. exp_43's plan and kit, on this branch (base `4a7e3b3a`): `worklog/worklog_yixun/exp_43_std_aug_claude/`.
   That is the RF arm: `plan_std_aug.md`, `FLAC_AR_YAWAUG_FLIP.json` and the frame-audit records.
3. exp_38's plan, neuronic kit and command sheet, on `origin/exp-38-data-efficiency` (`3593f34c`):
   - `worklog/worklog_yixun/exp_38_data_efficiency_claude/plan_data_efficiency.md`
   - `data_efficiency_neuronic_commands.md`
   - `exp38_registry.py`, `exp38_train.sbatch`, `exp38_submit_neuronic.py`, `exp38_supervise.py`,
     `exp38_subset_audit.py`, `exp38_content_manifest.py`
4. Your own exp_38 records on neuronic: the results commit that is not pushed yet, where the fraction cells were
   evaluated, and the measured 2 × L40 rate.

## 1. Goal and estimand

**Question.** How does the PanoFormer-style standard yaw augmentation (exp_43 arm `RF`: full-circle roll + mirror
flip, p = ½) behave at 25 / 50 / 75 % of AR? Its curve sits next to exp_38's YAW-Aug and OrbitRIR (B-F) fraction
curves, for the OrbitRIR rebuttal (Submission 22312).

**Estimand** (the same as exp_38 / exp_42 / exp_43 Stage A). For each fraction f ∈ {25, 50, 75} and each K ∈ {8, 1}:
- the item-weighted mean over the 6,337 unseen-eval items of EDT, C50, T60, R@1, R@5 and R@10 (plus FD and
  the geometry retrievals, recorded but not headlined);
- at θ = 0, with 1 Euler step and cfg 1.0;
- for the 40,000-step EMA checkpoint of one training run (training seed 42);
- reported as mean ± sample SD over eval seeds 42–46.

Per-scene means are recorded and not headlined.

No verdict is registered. This is a descriptive curve.

## 2. Decisions already taken (Yixun, 2026-11-14 — decided)

| # | Decision |
|---|---|
| D8 | The 25 / 50 / 75 % rows use **RF**, not C16. exp_42 (the C16 fractions) is on hold, not cancelled. |
| D1 | Train on **neuronic, 2 × L40 × 32/GPU**, the recipe of the exp_38 fraction runs (40,000 steps, seed 42, exp_38's subsets). |
| — | A new experiment **exp_45** (folder `exp_45_rf_data_curve_claude/`, branch `exp-45-rf-data-curve`). The neuronic session plans, implements and runs it. |
| — | Preparation starts now. The FULL trainings start after the RF 100 % θ = 0 result is confirmed. **Status: confirmed 2026-11-14 19:05 EST** (§7), so this gate is met. Plan approval by Yixun is still required before the first FULL. |
| D7-A | Push exp_38's results commit (per-seed metric JSONs, stream sidecars, receipts, table evidence) to `origin/exp-38-data-efficiency`. The YAW / OrbitRIR fraction rows are then quoted from a hash-pinned commit, not from the table pasted on Nov 12. |

## 3. Sources

| What | Where | Notes |
|---|---|---|
| RF implementation | `origin/exp-43-std-aug` at `4a7e3b3a948388434bc102b8bd7e7aba94648d71` (= this branch's base, the exp_43 training pin) | `src/data/yaw_mirror.py` (mirror operator), `src/data/yaw_offset_step.py`, `src/training/diffusion.py`, `src/training/factory.py` (`training.yaw_aug.flip`; wrapper keyword-only `yaw_aug_flip`). `src/data/yaw_rotation.py` must stay byte-identical. Tests: `src/tests/test_exp43_*.py`. |
| RF model config | `worklog/worklog_yixun/exp_43_std_aug_claude/FLAC_AR_YAWAUG_FLIP.json`, sha256 `d6a02ddebdc37c3844038898cc4888b4562e1d9d6669dd52ef021e221ac5086d` | Differs from exp_38's `de_yaw` config (`exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json`, `b14caae2…`) in exactly one key, `training.yaw_aug.flip = true` (checked 2026-11-14). Training banner on every rank: `yaw_aug ENABLED img_w=512 seed=42 flip=0.5`. Use the file in place, unchanged. |
| Fraction data side | `origin/exp-38-data-efficiency` at `3593f34c` (campaign pin `df849937`) | The 11 paths of §4.1, byte-exact. |
| Neuronic launcher | same branch | `exp38_train.sbatch` → `exp38_supervise.py --platform neuronic`, `exp38_submit_neuronic.py` (owner records, `reconcile`, `abandon`), plus the evidence callback `src/training/run_evidence.py` with `--evidence-json` in `train.py` / `defaults.ini`. |
| Merge base | `2b42468c` | Outside `worklog/`, the two branches overlap in **one** file: `src/tests/test_gen_model_comparison_depth_canon.py`. Inside `worklog/`, the overlap is `gen_model_comparison.py`: exp_43's lineage carries the exp_40 contract and exp_38 adds its own. A merged generator needs both contracts plus an exp_45 contract, with a regression test that every existing contract still resolves. |

Recommended code basis (**open**, plan decision): this branch (base `4a7e3b3a`), then
`git merge origin/exp-38-data-efficiency` once exp_38's results commit is pushed (D7-A). Then fork the exp_38
neuronic kit into `exp45_*` files: copy verbatim first, then make changes in separate commits, so that the review
object is the difference from reviewed code. Leave the exp_38 and exp_43 folders byte-frozen. A test should check
`git diff --quiet` on both folders.

### 3.1 Data-side port (from exp_42 plan §3, verified there by Codex plan round 1)

Exactly these paths, from `origin/exp-38-data-efficiency`:

- `data/AR/train_frac{025,050,075}_s2026.json`, `data/AR/train_frac_manifest_s2026.json` (`553534ea…`),
  `data/AR/train_frac_manifest_s2026.sha256`
- `src/configs/dataset_configs/AR/train/acousticroom_train_frac{025,050,075}.json`. Each is the base config plus
  its own split path plus `acoustic_context.restrict_to_split: true`.
- `src/configs/dataset_configs/custom_metadata/AR_md.py`. `restrict_to_split` filters the same-receiver context
  pool to the split's basenames, and an empty pool raises an error.
- `src/data/dataset.py`. `DatasetContractError` is re-raised instead of resampled.
- `src/tests/test_ar_md_restrict_context.py` (26 tests)

**Known side effect (exp_38 D8).** On a ported tree exactly three exp_25 tie-audit tests fail, because they bind
the bytes of `dataset.py` / `AR_md.py`. Put them in an exact allowlist; any other regression failure blocks.
- `test_depth_canon_audit_gate.py::test_the_real_full_audit_passes_the_registered_contract_without_rehash`
- `test_depth_canon_audit_gate.py::test_a_report_audited_in_another_checkout_is_accepted`
- `test_depth_canon_collect.py::test_the_overlap_derived_from_the_real_validated_audit`

## 4. Treatment

Three training runs. Each differs from exp_43's `exp43_RF_a6000_2x32` only in the dataset config, the platform /
launcher, and the names.

| arm | run id (suggested; must not contain `exp14_`) | dataset config (sha256) | split (sha256) | items | steps / epoch | final checkpoint |
|---|---|---|---|---|---|---|
| RFf025 | `exp45_RFf025_l40_2x32` | `acousticroom_train_frac025.json` (`7ac8243f…`) | `train_frac025_s2026.json` (`ce33f34f…`) | 73,473 | 1,148 | `epoch=34-step=40000.ckpt` |
| RFf050 | `exp45_RFf050_l40_2x32` | `acousticroom_train_frac050.json` (`b74687f7…`) | `train_frac050_s2026.json` (`bf330846…`) | 145,665 | 2,276 | `epoch=17-step=40000.ckpt` |
| RFf075 | `exp45_RFf075_l40_2x32` | `acousticroom_train_frac075.json` (`cc0e5fb8…`) | `train_frac075_s2026.json` (`57388ce8…`) | 218,463 | 3,413 | `epoch=11-step=40000.ckpt` |

- Steps per epoch = ⌊⌈items / 2⌉ / 32⌋ (DistributedSampler over 2 ranks, micro-batch 32, `drop_last=True`).
  These checkpoint names match exp_38's `de_yaw_f0xx` / `de_bf_f0xx` and exp_14's `dc_van_f0xx`.
- Effective item fractions: 25.2302 % / 50.0206 % / 75.0191 %.
- Fixed 40,000 updates, not matched epochs: ≈ 34.84 / 17.57 / 11.72 epochs, against 8.79 at 100 %.

### 4.1 Training argv: exp_43's FULL run, as launched (`std_aug_2026-11-13_19-11-21_FULL_launch.json`, A6000)

```
--model-config worklog/worklog_yixun/exp_43_std_aug_claude/FLAC_AR_YAWAUG_FLIP.json
--dataset-config src/configs/dataset_configs/AR/train/acousticroom_train.json
--pretransform-ckpt-path weights/FLAC/VAE.safetensors
--max-steps 40000 --batch-size 32 --accum-batches 1 --num-workers 6 --seed 42
--num-gpus 2 --num-nodes 1 --strategy ddp_find_unused_parameters_true --sync-batchnorm true
--precision bf16-mixed --val-every -1 --val-dataset-config "" --gradient-clip-val 0.0
--logger wandb --checkpoint-every 2500
--name exp43_std_aug --experiment-name exp43_RF_a6000_2x32 --save-dir <run root>
```

`WANDB_MODE=offline`. Training backend: flash-attn 2.7.4.post1. The exp_38 neuronic kit asserts that version
(`--flash-version`). Record the backend actually used, as exp_38's evidence callback does.

**Rule for the plan:** the exp_45 argv must equal exp_38's `de_yaw_f0xx` neuronic argv with exactly two
registered differences, `--model-config` and the names / roots. Diff it against exp_43's argv above and list
every remaining difference with a reason (for example `--logger` and `--evidence-json`, or `--num-workers` if the
neuronic kit sets another value).

**Loss logging — open, recommendation: keep the per-step training loss.** exp_44 (training efficiency) wants
loss curves that include neuronic L40 runs, and exp_38 trained with `--logger none`. So use W&B offline, and
preserve the run directory **before** any cleanup. PL's `WandbLogger(save_dir=".")` overrides `WANDB_DIR`, so an
exit trap that removes `$TMPDIR` destroys the run; this is how exp_40 lost its loss log. Copy the exp_43
pattern: `exp43_kit.py wandb-preserve` / `wandb-archive`. A CSV logger is an acceptable alternative. Add no
validation during training (`--val-every -1` stays).

**Frame audit.** No new frame audit is needed. The RF operator is byte-identical to exp_43's, whose frame audit
passed under gate A2. A2 was adopted by Yixun after the originally registered gate failed, and this is disclosed
(§9).

**Crash policy (exp_38's, keep it).** A crashed run is archived and re-run from step 0. A resumed continuation
never replaces a primary cell.

## 5. Evaluation protocol (decided: exactly exp_43 Stage A)

Per run: 10 cells (K ∈ {8, 1} × eval seeds 42–46), 30 in total, plus the bridge cells of §5.1. These are the
argv of exp_43's cell `exp43_RF_T_S40000_K8_s42` (`exp43_cells.json`); K = 1 uses
`acousticroom_unseeneval_1.json`:

```
eval_FLAC.py --model-config worklog/worklog_yixun/exp_43_std_aug_claude/FLAC_AR_YAWAUG_FLIP.json
  --dataset-config src/configs/dataset_configs/AR/eval/acousticroom_unseeneval.json   (sha256 063c66c2…)
  --cond-method vanilla --cond-autocast bf16 --cfg-scale 1.0 --steps 1 --batch-size 64 --device cuda
  --record-per-scene --record-stream --expected-stream-count 6337
  --seed <42..46> --rotate-mode fixed --rotate-deg 0 --eval-name <cell id> --ckpt-path <40k checkpoint>
  --dump-scored-waveforms <dump root>          (announcement 08)
```

- EMA weights, and the full 6,337-item unseen split (announcement 01).
- Attention backend: SDPA through the reviewed no-flash launcher
  (`overlays/no_flash/exp35_no_flash_launch.py eval_FLAC.py …`), as for every cell of exp_25 / 35 / 38 / 40 / 43.
- Every cell's input hash must equal the pinned exp_25 value for its (K, seed). exp_38's evaluator pins all ten.
  For example, K8 s42 is `4c950c1e2ed021a352f5c454b67e3b21234cdbd52060debc1c58919c9e3645ac`.

**Where to evaluate — open; establish the facts first.** exp_38's plan put evaluation on the A6000 box. Yixun
said on 2026-11-12 that the fraction cells' per-seed evidence exists only on neuronic. Take from your exp_38
records where and how `de_yaw_f0xx` / `de_bf_f0xx` were evaluated (host, GPU, evaluator pin, backend).
**Recommendation:** evaluate exp_45 the same way, on the same evaluator lineage, so that RF and YAW / OrbitRIR
share an evaluator at every fraction.

### 5.1 Bridge to the 100 % point (recommended)

The RF 100 % row was evaluated on the A6000 box at the exp_43 evaluation pin `5f5b493a8ac1…`.

**Recommendation:** re-evaluate the RF 100 % checkpoint on the exp_45 evaluator, with all 10 θ = 0 cells
(≈ 2 GPU-h), so that the whole RF curve 25 → 100 % shares one evaluator. Then compare cell by cell with the
A6000 values. Exact equality means the A6000 row and the neuronic curve are interchangeable. Any difference is
reported as a measured platform / evaluator difference, never hidden.

- Checkpoint: NAS `/media/diskstation/yixunhu/FLAC/checkpoints/exp43_std_aug/exp43_RF_a6000_2x32/epoch=8-step=40000.ckpt`.
  sha256 `ae62977e6a446c696f4b8f34e9393d335406bc8cd6404e7cc9caceaf55d1b6f6`, 723,922,731 B.
- Neuronic does not see the NAS, so Yixun copies it from a machine that sees both. Verify the sha256 after the
  copy.
- A6000 reference for cell K8 s42:
  - input hash `4c950c1e…45ac`; assignment hash `4894d0e119f71ed6f5d566aa01b69d8fc3f8a6cb88fb66f27f121ba59f0a8442`;
    stream sha256 `8d8018cabfd803be559b4224ef7c8833fe0af7b96542cbce65cd0a3e1d1d4af6`
  - metrics: T60 8.4182, C50 1.0393, EDT 41.8123, FD 0.3312, GT R@1 5.3338, R@5 16.2853, R@10 24.207
  - metrics file sha256 `cc004cd8…f4a1`

## 6. Before any production FULL (suggested ladder; the plan fixes it)

1. Port tests (§3.1) and the fork's own tests.
2. The repo CPU regression with the exact allowlist of §3.1.
3. The exp_38 subset audit `--verify`, and the AR content-manifest verify on the neuronic AcousticRooms copy.
4. Neuronic dry runs of every launch path.
5. One production-topology SMOKE per run (2 × L40, 30 steps). Check:
   - the RF banner on both ranks;
   - the observed loader length (1,148 / 2,276 / 3,413);
   - finite loss;
   - the flash backend;
   - the measured s/step, which sets `--time` for FULL.
6. The evaluator's E0. It must reproduce the pinned reference cells, and preferably the RF K8 s42 bridge cell of
   §5.1, before any production cell is trusted.

## 7. Reference values (100 % points, θ = 0, K = 8, item-weighted, mean ± SD over seeds 42–46)

From exp_43's rebuttal table (`std_aug_rebuttal_table.md`, generated 2026-11-14 19:05 EST at evaluation pin
`5f5b493a8ac1…`; table evidence `exp43_table_evidence_A.json` sha256 `e4c9e881…`, VALID). These files are in the
A6000 launch worktree and are not pushed yet.

| Model | EDT (ms) ↓ | C50 (dB) ↓ | T60 (%) ↓ | R@1 ↑ | R@5 ↑ | R@10 ↑ |
|---|---|---|---|---|---|---|
| FLAC (P1 @40k) | 40.650 ± 0.103 | 1.0093 ± 0.0035 | 8.993 ± 0.011 | 5.208 ± 0.105 | 15.471 ± 0.196 | 23.450 ± 0.059 |
| Yaw-Aug | 39.923 ± 0.058 | 1.0132 ± 0.0016 | 7.965 ± 0.015 | 5.356 ± 0.036 | 16.528 ± 0.221 | 23.961 ± 0.133 |
| C16 Yaw-Aug | 39.553 ± 0.102 | 1.0094 ± 0.0030 | 8.249 ± 0.007 | 5.577 ± 0.071 | 16.538 ± 0.146 | 24.355 ± 0.190 |
| **Roll + Flip (RF)** | 41.780 ± 0.089 | 1.0406 ± 0.0022 | 8.418 ± 0.013 | 5.410 ± 0.072 | 16.203 ± 0.153 | 24.081 ± 0.149 |
| OrbitRIR (B-F @40k, fa_invariant; legacy files, not re-validated by exp_43) | 38.793 ± 0.074 | 0.9778 ± 0.0015 | 8.202 ± 0.017 | 5.387 ± 0.075 | 16.456 ± 0.038 | 24.198 ± 0.164 |

RF at K = 1: EDT 43.998 ± 0.417, C50 1.1170 ± 0.0062, T60 9.943 ± 0.048, R@1 5.148 ± 0.190.

## 8. Acceptance criteria

**Per run:**
- exit 0;
- 16 cadence checkpoints with the fraction's names;
- end audit PASS: global step 40,000, the fraction's epoch, embedded model config = `d6a02dde…`, scheduled lr,
  optimizer state and EMA present;
- evidence on both ranks: the RF banner, `restrict_to_split` active, the observed dataset and loader lengths
  equal to §4, the argv equal to the registered projection;
- `CHECKPOINTS.sha256`;
- published to the NAS `/media/diskstation/yixunhu/FLAC/checkpoints/exp45_rf_data_curve/<run id>/`
  (announcement 07) through exp_38's staged, verified rsync;
- the training-loss record preserved (§4.1);
- the admission written by the kit, never by hand.

**Per cell:**
- 6,337-item stream, with the input hash equal to the pin;
- assignment hash;
- the full dump (`[6337, 10240]` float32 + manifest), announcement 08;
- a receipt bound to the run's admitted checkpoint (sha256 + bytes) and to the evaluator pin.

**Experiment:**
- E0 PASS;
- 30 / 30 cells verified, plus the bridge cells;
- both-K rows complete for all three fractions;
- the table builder refuses any missing (method × fraction × K) cell, and refuses any value not equal to its
  source.

A failed run or cell is reported with its log. It is never silently replaced, and never by a resumed run.

## 9. Disclosures to print under every exp_45 table

1. One training seed and one subset draw per point. The SD is evaluation randomness only, so no significance
   claim is made; paired per-seed deltas are descriptive only.
2. Nominal fractions; the effective item fractions are 25.2302 / 50.0206 / 75.0191 %.
3. Fixed 40,000 updates, not matched epochs (§4).
4. RF 25 / 50 / 75 % were trained on 2 × L40 (neuronic). RF 100 % was trained on 2 × A6000 with the unrestricted
   100 % config. exp_38's subset audit found no out-of-split candidate at 100 % (removed = 0, outside = 0), but
   the curve is cross-platform at 100 %. Show the bridge result (§5.1) if it was run.
5. The RF frame audit passed under gate A2, which was adopted after the originally registered gate failed.
6. Item-weighted aggregation, not the paper's per-scene protocol.
7. OrbitRIR rows use their own `fa_invariant` inference protocol, so the comparison is descriptive, not an
   augmentation-only controlled comparison.

## 10. Open decisions for the plan (each with a recommendation; Yixun decides)

| # | Decision | Recommendation |
|---|---|---|
| N1 | Code basis | This branch, plus a merge of `exp-38-data-efficiency` after D7-A, plus a forked exp_38 neuronic kit (§3). |
| N2 | Concurrency / order | All three FULLs at once if the partition allows; otherwise 25 → 50 → 75. One monolithic job per run, as exp_38. |
| N3 | Loss logging | W&B offline, preserved (§4.1). |
| N4 | Evaluation location | The same place and evaluator lineage as exp_38's fraction cells (§5). |
| N5 | 100 % bridge | Yes: all 10 θ = 0 cells of the RF 100 % checkpoint on the exp_45 evaluator (§5.1). |
| N6 | Publication | exp_45 stays on its own branch. Rows merge later with a merged generator (§3). The `model_comparison.md` regeneration is Yixun's (announcement 04). |

Rebuttal deadline: not known to the A6000 session. Ask Yixun if it constrains N2.

## 11. Standing rules and traps (from `CLAUDE.md` and the memory of the A6000 session)

- Never modify `main`. Do not put any account e-mail address in repo files.
- Never edit a bash script while it runs. Never stop or signal a job you did not start: check the owner / cwd.
- `flac` env: `WANDB_MODE=offline`; never upgrade the env.
- Codex: at most 3 plan + 3 code rounds. Hand over all code of a round at once and fix every finding. Every
  prompt must forbid installing or modifying environments (`-s read-only` does not protect them).
- Never pass `--allow-row-regression` / `--allow-partial-exp11` to `gen_model_comparison.py` without Yixun's
  explicit order.
- Neuronic:
  - there is no `SBATCH_EXCLUDE`; node exclusion is only an explicit `--exclude`;
  - ECC-flaky nodes (Aug 2026): neu301/303/305/306/317/319/322/332;
  - `sacct` truncates `State`, so request `State%-40` and normalise by prefix;
  - scheduler-query failure is fatal, never "no job found".
- Eval flags are part of the experiment (announcement 05). RF is `vanilla`; OrbitRIR is `fa_invariant`.

## 12. What to hand back

1. `origin/exp-38-data-efficiency`: exp_38's results commit (D7-A), pushed.
2. `origin/exp-45-rf-data-curve`, with:
   - the plan and its Codex artifacts, and the commits ledger;
   - per run: the run receipts, evidence and loss record;
   - per cell: metric JSON, stream sidecar and receipt;
   - the table evidence;
   - the results table: RF, YAW-Aug and OrbitRIR at 25 / 50 / 75 %, both K, with the disclosures of §9 and the
     bridge result.

   Push only on Yixun's go.

The A6000 session then quotes these rows, by commit and file sha256, in the rebuttal's data-efficiency table,
next to the 100 % rows of §7.
