**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-12 · **Round:** exp_43 plan review r1 (plan_std_aug.md r0). Everything below this line is the reviewer's output, verbatim.

REVISE

1. **BLOCKING — The registered frame-alignment gate is not valid as written.**  
   **Evidence:** The proposed point-cloud requirement is “identity median ≤ 0.10 m in every room” and the best mirrored hypothesis ≥3× worse ([plan_std_aug.md:127–131](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_43_std_aug_claude/plan_std_aug.md:127)). AcousticRooms does provide the required `src_loc`, `rec_loc`, and `depth_map/<category>/<room>/<receiver>.npy` layout ([AR_md.py:47](/home/yixunhu/codespace/exp-43-std-aug/src/configs/dataset_configs/custom_metadata/AR_md.py:47), [AR_md.py:68](/home/yixunhu/codespace/exp-43-std-aug/src/configs/dataset_configs/custom_metadata/AR_md.py:68)), but point clouds from different receivers do not correspond pointwise: visibility, occlusion, sampling density, and room symmetry differ. A read-only audit of three `Cafe_idx_0` receivers produced pairwise symmetric NN medians `[12.0138, 1.5257, 1.6476]` m, mean `5.0624` m for the intended identity frame; a coarser all-hypothesis search selected `(I, β=0)` but the best-reflection ratio was only `1.54`, not `3`.  
   **Change:** Replace the absolute threshold with a deterministic two-score identification gate:

   - Enumerate a frozen, hashed set of rooms, receivers, and sources.
   - Search all `(S, β)` hypotheses over both handedness choices and all 512 headings.
   - Primary score: transform each source vector into the candidate depth frame, sample scalar depth along that ray, and aggregate signed clearance/containment. This directly tests pose-to-depth agreement.
   - Secondary score: multi-receiver point-cloud NN agreement, used only as corroborative ranking.
   - Require `(I,0)` to be the unique aggregate optimum under both scores and stable under leave-one-room/category-out checks. Persist the complete score curves and selected inputs. Do not require every room to clear an arbitrary absolute distance or 3× ratio.
   - Freeze any numerical margin before model results; calibrate it using synthetic transforms, not outcome-dependent real-data tuning.

2. **MAJOR — F6 is mathematically correct but not bit-for-bit correct in its conversion-equivalence claim.**  
   **Evidence:** The grid satisfies the real-arithmetic identity \(\theta_{W-1-j}=-\theta_j\), so column reversal plus negating channel 1 implements \(M=\mathrm{diag}(1,-1,1)\), i.e. \(y\mapsto-y\) ([AR_md.py:58–66](/home/yixunhu/codespace/exp-43-std-aug/src/configs/dataset_configs/custom_metadata/AR_md.py:58)). The tensor operator itself is a bitwise involution for finite tensors. However, the stored floating grid is not bit-symmetric: a CPU check at `W=512` found `theta_pair_exact=False`; converting a reversed scalar depth map versus reversing vector columns and negating `y` gave `torch.equal=False`, maximum absolute difference `2.3841858e-07`, with 50,921 unequal float32 entries. This conflicts with the exact wording in F6 ([plan_std_aug.md:16](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_43_std_aug_claude/plan_std_aug.md:16)).  
   **Change:** State separately:

   - `mirror(mirror(x))` is bitwise identical for finite input tensors.
   - Pose-distance preservation is exact up to normal floating arithmetic.
   - Equality with re-rendering a reversed scalar depth map is numerical, with the pre-registered `≤1e-6` tolerance—not bitwise.

3. **MINOR — The conditioning-field transform is otherwise complete and physically consistent, subject to the frame gate.**  
   **Evidence:** AR metadata exposes only `source`, `source_vit`, `context_poses`, `context_poses_vit`, `context_audio`, and `depth` as conditioning-relevant values ([AR_md.py:8–54](/home/yixunhu/codespace/exp-43-std-aug/src/configs/dataset_configs/custom_metadata/AR_md.py:8)). Negating component 1 of all four pose-vector fields is the matching application of \(M\). `GeometryConditioner` consumes the already-reflected vector panorama ([conditioners.py:268–303](/home/yixunhu/codespace/exp-43-std-aug/src/models/conditioners.py:268)); its DINO positional encoding is applied at the new image positions, which is correct. `DistEmbedder` is coordinatewise and has no cross product or handedness-sensitive construction ([conditioners.py:305–350](/home/yixunhu/codespace/exp-43-std-aug/src/models/conditioners.py:305)). The arm’s configured metadata keys contain no additional ray or position representation ([FLAC_AR_YAWAUG.json:45](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_15_yaw_aug_claude/FLAC_AR_YAWAUG.json:45)). Dataset collation passes metadata dictionaries through without deriving column-index geometry ([dataset.py:379–392](/home/yixunhu/codespace/exp-43-std-aug/src/data/dataset.py:379)).  
   **Change:** Add this conditioning trace to §2.1. Explicitly state that no stored orientation matrix, ray encoding, surface normal, or handedness-sensitive cross product exists in this arm. Keep `context_audio` and the mono target unchanged.

4. **MAJOR — The repository contains no independent proof of AR horizontal handedness.**  
   **Evidence:** `AR_md.py` constructs poses by simple `src_loc-rec_loc` subtraction and supplies no camera orientation metadata ([AR_md.py:68–75](/home/yixunhu/codespace/exp-43-std-aug/src/configs/dataset_configs/custom_metadata/AR_md.py:68)). The HAA “remove minus” note ([README.md:165](/home/yixunhu/codespace/exp-43-std-aug/README.md:165)) concerns source/listener placement, not proof of AcousticRooms’ horizontal axis convention. No dataset document in the repository establishes camera yaw or handedness.  
   **Change:** Do not cite the HAA note as validation. Treat common frame plus handedness as an experimentally gated precondition and record any external AcousticRooms convention document in the evidence archive.

5. **MAJOR — The random-stream design is sound, but the evidence and wording need tightening.**  
   **Evidence:** `_yaw_aug_step_seed` depends on seed, global optimizer step, and rank ([diffusion.py:86–144](/home/yixunhu/codespace/exp-43-std-aug/src/training/diffusion.py:86)). A CPU check confirmed that drawing `randint(0,512,n)` first gives identical offsets whether or not a subsequent `randint(0,2,n)` is performed. Thus the YAW offset substream is preserved and the new bits are resume-exact for the same global step/rank/dataloader state. The registered exp_40 recipe uses accumulation 1 ([exp40_kit.py:280–294](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:280)), so repeated seeds across accumulated microbatches are not an issue. Each DDP rank receives a distinct stream.  
   **Change:** Clarify that this preserves the *reference offset draw stream*, not samplewise pairing with the already-trained historical YAW run. Require the evidence driver to capture both the actual flip-bit draw and the actual sample indices passed to `mirror_scene_metadata`; recomputation from the counter seed alone cannot prove that flips were applied. Replace the probabilistic “1e6 draws within 4σ” test with a fixed-seed exact fixture plus deterministic balance sanity check to avoid a flaky test.

6. **MAJOR — Make `yaw_aug_flip` keyword-only and strengthen the new module’s input contract.**  
   **Evidence:** The current constructor ends with `yaw_aug_offset_step` ([diffusion.py:186–213](/home/yixunhu/codespace/exp-43-std-aug/src/training/diffusion.py:186)), and [test_yaw_aug_c16.py:311–314](/home/yixunhu/codespace/exp-43-std-aug/src/tests/test_yaw_aug_c16.py:311) asserts that it is the last parameter. Appending another positional parameter and weakening that test to `[-2:]` preserves current calls but loses the API’s positional boundary. The proposed `int(img_w)` also accepts booleans, floats, and numeric strings, while direct public use of `mirror_scene_metadata` would lack the wrapper’s deeper checks.  
   **Change:** Use:

   ```python
   ..., yaw_aug_offset_step=None, *, yaw_aug_flip=False
   ```

   Update the test to assert that `yaw_aug_offset_step` is the last positional-or-keyword parameter and `yaw_aug_flip` is keyword-only. Reject non-literal booleans for the flag and non-literal positive integers for width. Validate depth tensor shape/type/finite values and each present pose tensor’s trailing dimension/type. Add aliasing/no-mutation tests where `source` and `source_vit` share storage.

7. **MAJOR — Preserve the parser’s disabled-mode compatibility explicitly.**  
   **Evidence:** The factory rejects unknown keys before checking `enabled`, but currently ignores optional-value validation when disabled ([factory.py:35–49](/home/yixunhu/codespace/exp-43-std-aug/src/training/factory.py:35)); [test_yaw_aug_c16.py:261–280](/home/yixunhu/codespace/exp-43-std-aug/src/tests/test_yaw_aug_c16.py:261) codifies that behavior for `offset_step`. Existing tests also require exact legacy kwargs when the new key is absent ([test_yaw_aug_training.py:504–526](/home/yixunhu/codespace/exp-43-std-aug/src/tests/test_yaw_aug_training.py:504)).  
   **Change:** Specify and test the same rule for `flip`: unknown keys are always rejected; when `enabled=false`, optional values are ignored and the parser returns `{}`; when enabled, `flip` must be a literal bool. Direct constructor calls with `yaw_aug_flip=True` and yaw disabled should still be rejected. Emit no new kwarg when `flip` is absent or false.

8. **MAJOR — The constrained-test and historical-pin inventory is incomplete.**  
   **Evidence:** The definite current-test break is the constructor-order assertion above. Other compatibility constraints that must remain green include:

   - Exact old banners: [test_yaw_aug_c16.py:317–328](/home/yixunhu/codespace/exp-43-std-aug/src/tests/test_yaw_aug_c16.py:317), [test_yaw_aug_training.py:638–659](/home/yixunhu/codespace/exp-43-std-aug/src/tests/test_yaw_aug_training.py:638).
   - Unknown-key and exact factory kwargs: [test_yaw_aug_training.py:504–548](/home/yixunhu/codespace/exp-43-std-aug/src/tests/test_yaw_aug_training.py:504).
   - Raw constructor guards and global RNG isolation: [test_yaw_aug_training.py:674–762](/home/yixunhu/codespace/exp-43-std-aug/src/tests/test_yaw_aug_training.py:674).
   - Exact training transform and step integration: [test_yaw_aug_training.py:765–850](/home/yixunhu/codespace/exp-43-std-aug/src/tests/test_yaw_aug_training.py:765).
   - Malformed metadata and no validation/test augmentation: [test_yaw_aug_training.py:853–964](/home/yixunhu/codespace/exp-43-std-aug/src/tests/test_yaw_aug_training.py:853).
   - Base/C16 config-byte and kwargs pins: [test_yaw_aug_c16_config.py:19–90](/home/yixunhu/codespace/exp-43-std-aug/src/tests/test_yaw_aug_c16_config.py:19).
   - Depth-canon imports of exact pre-change kwargs: [test_depth_canon_training.py:115–138](/home/yixunhu/codespace/exp-43-std-aug/src/tests/test_depth_canon_training.py:115).
   - The exp_25 historical manifest pins both `diffusion.py` and `factory.py`, not merely `yaw_rotation.py` ([depth_canon_pins.sha256:34–35](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_pins.sha256:34)); its launch guard enforces them ([depth_canon_launch.sh:197–209](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:197)).
   - `yaw_rotation.py` remains byte-pinned by [test_yaw_aug_c16.py:476–482](/home/yixunhu/codespace/exp-43-std-aug/src/tests/test_yaw_aug_c16.py:476).

   No equivalent exp_15/exp_17 source-byte manifest was found.  
   **Change:** Add this inventory to §6. Never update the exp_25 historical manifest to bless new bytes; instead state that the historical launcher is intentionally incompatible with the exp_43 tree and validate exp_43 with its own closure. Old banner, factory, disabled-golden, and base/C16 fixtures must pass unmodified.

9. **BLOCKING — The exp_40 fork-binding table is far from complete, and the proposed four-file copy cannot pass its own closure.**  
   **Evidence:** Important missed bindings include:

   - `exp40_kit.py`: experiment/config/fixture/pins ([49–61](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:49)); AR manifest, shared handshake/card locks, run-tree manifest ([69–74](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:69)); archive and closure lists ([77–87](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:77)); neuronic platforms and grammars ([88–98](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:88)); project/NAS/run ID ([115–119](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:115)); config and evidence schemas ([886–1026](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:886)); offset-only stream snippet and histogram reconstruction ([1444–1569](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:1444)); admission names/kinds/arm ([2408–2507](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:2408)); and extensive neuronic CLI paths through [exp40_kit.py:2621](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:2621).
   - The closure explicitly includes `exp40_train.sbatch` and `exp40_submit_neuronic.sh`, which the plan does not copy ([exp40_kit.py:77–81](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:77)).
   - `exp40_evidence_driver.py`: exp40 environment/error names ([2–46](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_evidence_driver.py:2)); offset-only evidence schema and 16-bin histogram ([50–124](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_evidence_driver.py:50)); rotation-only spy ([262–283](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_evidence_driver.py:262)); treatment block ([310–312](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_evidence_driver.py:310)).
   - `exp40_train_local.sh`: base, logs, checkpoint/NAS/HF/W&B/runtime roots, lock, kit/driver names, environment namespace, and admission file names throughout [31–92](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_train_local.sh:31), [147–233](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_train_local.sh:147).
   - `exp40_eval.py`: kit loader, arms, admissions, cells, comparator/E0/table/figure names, farm and dump roots ([68–108](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:68)); C16 cell routing ([279–302](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:279)); receipt/admission routing ([580–643](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:580)); comparator schema ([1434–1640](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:1434)); labels/table builders ([1655–1806](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:1655)); output and figure contracts ([1832–2015](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:1832)); C16 CLI branches ([2106–2207](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:2106)).

   **Change:** Replace the small binding table with an exhaustive executable-semantics checklist. Either remove all neuronic paths and closure entries or copy and deliberately disable them; a single identity check is insufficient. Add a test/inspection gate that no unintended `exp40`, `C16`, `offset_step`, `32`, `16-bin`, or `neuronic` binding remains in executable semantics.

10. **BLOCKING — “Reuse exp_40’s 82 C16 cells and 92 comparators with the validators as written” is factually wrong.**  
    **Evidence:** Reading `exp40_cells.json` gives 85 total cells: three E0 and 82 non-E0. Those 82 comprise **66 C16 + 8 YAW FILL + 8 P1 FILL**, not 82 C16. The comparator registry contains 92 older rows—56 exp_25 and 36 exp_36—not exp_40 cells. Its schema hard-codes those sources ([exp40_eval.py:1434–1467](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:1434)), and the dedicated authoritative validation path is not an exp_40-cell validator ([exp40_eval.py:1484–1506](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:1484)). The frozen exp_40 table evidence contains only the ten canonical C16 table cells plus E0, not all 82 ([exp40_eval.py:1946–2015](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:1946)).  
    **Change:** Correct the counts and create two distinct sources:

    - Reuse the existing 92 comparator registry only under its original source validators and exact E0-reference binding.
    - Add a pinned exp_40 source adapter that validates exp_40 admission, E0, receipts, cell hashes, and artifacts at the exp_40 code/results pin before importing its 82 non-E0 rows.

    Do not let copied CSV/table rows count as validated live receipts. The resulting historical comparison set is 174 rows: 92 old comparators plus 66 C16, 8 YAW FILL, and 8 P1 FILL.

11. **MAJOR — Runtime isolation needs a precise shared-versus-private policy.**  
    **Evidence:** The exp_40 kit uses a shared handshake directory and per-GPU UUID locks ([exp40_kit.py:69–74](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:69)), an exp40-named card lock, experiment-local receipts, admission records, runtime trees, W&B directories, checkpoint project/NAS paths, eval farms, waveform dumps, and E0/table artifacts. Reusing the handshake/UUID lock is useful mutual exclusion; reusing any evidence or output root can allow stale exp_40 artifacts to satisfy or corrupt exp_43 gates. The sealed hub is below the local base root ([exp40_train_local.sh:52–73](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_train_local.sh:52)).  
    **Change:** Explicitly retain only the shared GPU handshake/per-UUID locking, and either retain the old card lock solely as documented cross-experiment serialization or introduce a neutral common lock. Make all receipts, admissions, runtime/W&B roots, checkpoint/NAS project, farm/dump paths, E0, cells, tables, figures, and loss artifacts exp_43-specific. Copy/pin the AR manifest but issue fresh 48-hour receipts.

12. **MAJOR — The baseline allowlist and E0 procedure need stricter ordering and signatures.**  
    **Evidence:** The plan says the baseline can run “while code is written” ([plan_std_aug.md:133–139](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_43_std_aug_claude/plan_std_aug.md:133)). That can no longer prove the baseline came from exact `6a51f741`. A nodeid-only allowlist can also hide a newly introduced failure at an already-failing nodeid. The comparator registry binds to an exact E0-reference hash ([exp40_eval.py:1623–1624](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:1623)).  
    **Change:** Finish and archive the full baseline before any source edit, recording commit, collection count, nodeids, normalized error signatures, skips, and xfails. Post-change acceptance should allow only identical known signatures—no pass→fail, new skip/xfail, or collection loss. Byte-copy exp_40’s E0 reference if equality to exp_40 is intended, or regenerate the comparator registry binding; do not mix a new reference with the old registry. E0 validates inference compatibility, not the new training flip path.

13. **MAJOR — The REQUEST/sequencer design references an unavailable contract and does not cover crash recovery.**  
    **Evidence:** No exp_42 §5.3 schema exists in this worktree, so the plan’s reference is not implementable from the registered repository. Shell traps cannot remove a request after SIGKILL, host failure, or power loss. A second Stage-B invocation also needs the same admission and GPU ownership protections.  
    **Change:** Put the complete schema and state machine in the exp_43 plan and tests. Use an atomic, locked lease containing experiment, pin, host, PID, `/proc` start time, boot ID, timestamps/heartbeat, and explicit stale-reclamation rules. REQUEST is advisory—not launch evidence. Preserve the exp_40 card-lock, per-UUID lock, and occupancy recheck immediately before spawning work. Test concurrent sequencers, foreign/stale leases, every preflight failure, SIGTERM cleanup, idempotent resume, and Stage-B reacquisition. Acquire the lease only after CPU gates pass, so failed preflight cannot leave a live request.

14. **BLOCKING — The plan does not currently preserve what exp_44 needs, and F10’s “per-step loss” assertion is false.**  
    **Evidence:** The plan claims per-step training/loss preservation ([plan_std_aug.md:20](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_43_std_aug_claude/plan_std_aug.md:20)), but `train.py` sets `log_every_n_steps=100` ([train.py:77](/home/yixunhu/codespace/exp-43-std-aug/train.py:77)). Although the module calls logging each step ([diffusion.py:966–975](/home/yixunhu/codespace/exp-43-std-aug/src/training/diffusion.py:966)), exp_40’s evidence driver explicitly records no per-step loss digest, only sampled steps and finiteness ([exp40_evidence_driver.py:50–124](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_evidence_driver.py:50)). Offline W&B state is local to the runtime tree ([exp40_train_local.sh:67–73](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_train_local.sh:67)) and is not presently named as an admitted exp_44 artifact.  
    **Change:** Before training:

    - Define the exact loss cadence honestly—per-step or every 100 optimizer steps.
    - Produce a canonical admitted sidecar containing global step, rank-aggregated loss, learning rate, effective sample count, and non-finite status. If per-step collection is required, buffer it to avoid a device synchronization every step.
    - Publish/hash/admit the loss sidecar and/or offline W&B history to NAS.
    - Preserve all 16 intermediate full-state checkpoints and their optimizer/scheduler/global-step/config hashes.
    - Pre-register the exp_44 post-hoc validation protocol: fixed item manifest, split, batches, diffusion timesteps/noise, seeds, context selection, train/eval mode, precision, online versus EMA weights, preprocessing hashes, and aggregation.

    Without these additions, exp_44 may require rerunning RF.

15. **MAJOR — The estimand and comparison language overstate causal evidence.**  
    **Evidence:** The experiment uses one RF training run and compares it with historical arms from different runs—and, for some rows, different cluster histories. Five evaluation seeds measure generation/evaluation variability, not training-run uncertainty. Yet the question asks how much RF “recovers” over YAW and says the flip is the only change ([plan_std_aug.md:25–31](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_43_std_aug_claude/plan_std_aug.md:25)).  
    **Change:** Define the result as a descriptive cross-run comparison. Report paired per-evaluation-seed deltas where input/assignment hashes match, plus each arm’s mean and eval-seed SD, but state that these are not confidence intervals over training. Do not attribute the observed RF–YAW difference solely to flipping unless a contemporaneous same-pin YAW control is trained; multiple training seeds would be needed for a robust causal estimate. Clearly label P1/YAW/C16/B-F provenance and hardware in the table.

16. **MAJOR — Omitting mirrored evaluation leaves the stated mechanism untested.**  
    **Evidence:** The plan explicitly has no mirrored-evaluation cell ([plan_std_aug.md:59–62](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_43_std_aug_claude/plan_std_aug.md:59)). Yaw-robustness cells test rotational behavior, not reflection robustness or invariance.  
    **Change:** Add a small, pre-registered full-6337 diagnostic tier: mirrored conditioning for RF and YAW at `K∈{1,8}`, seed 42—four cells total—with the same tested mirror operator and strict input/stream hashes. Keep it descriptive. If this is not run, the paper must make no claim that RF learned mirror robustness; it can claim only endpoint performance after flip augmentation.

17. **MINOR — “Standard UniFuse / PanoFormer augmentation” needs more precise naming.**  
    **Evidence:** UniFuse performs full-width roll followed by flip, but also applies image color jitter in the same augmentation path ([UniFuse loader](https://github.com/alibaba/UniFuse-Unidirectional-Fusion/blob/main/UniFuse/datasets/matterport3d.py#L85-L97)). PanoFormer uses a quarter-width roll range before flipping ([PanoFormer loader](https://github.com/zhijieshen-bjtu/PanoFormer/blob/main/PanoFormer/dataload.py#L82-L94)). The plan correctly discloses the 512-versus-513 and `W`-versus-`W/4` differences, but the high-level name still suggests literal reproduction.  
    **Change:** Name the arm “full-circle roll + horizontal flip (UniFuse-style; PanoFormer uses a quarter-roll variant).” State that photometric color jitter is inapplicable to FLAC’s depth-vector conditioning and was not copied.

18. **MAJOR — The TDD and acceptance ladders omit several failure modes.**  
    **Evidence:** The current per-file plan and acceptance criteria ([plan_std_aug.md:177–218](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_43_std_aug_claude/plan_std_aug.md:177)) do not require proof of actual flip application, historical-source validation, request-lease recovery, loss-artifact admission, or the corrected frame gate.  
    **Change:** Add red-before-green tests for:

    - Keyword-only constructor semantics and disabled parser behavior.
    - Malformed mirror metadata, aliasing, no mutation, and exact involution.
    - Independent roll-then-flip oracle at training-step level.
    - Actual drawn bits versus actual applied sample indices in single-rank, DDP smoke, and full evidence.
    - Mutations of every exp_40 source admission/E0/receipt/artifact hash.
    - Removal/refusal of neuronic closure entries and unintended legacy bindings.
    - Lease concurrency, stale recovery, signals, and failure injection.
    - Exp_44 loss/history artifact publication and admission.
    - Exact comparator/E0-reference binding.
    - Mirrored full-split cells, if accepted.

    Acceptance must explicitly include the revised frame audit, applied-flip evidence, source-row validation, full 6337-item counts, admitted loss history, and a clean registered checkout.

19. **MINOR — The preflight estimate and commit-size plan are optimistic.**  
    **Evidence:** Exp_40’s full training took about 21.6 hours, E0 about 26 minutes, and the 82-cell two-lane evaluation about 8.8 wall-clock hours ([yaw_aug_c16_worklog.md:160–201](/home/yixunhu/codespace/exp-43-std-aug/worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/yaw_aug_c16_worklog.md:160)). The planned Stage-A/Stage-B runtime is plausible if both cards remain free. However, exp_43 adds a new transform, frame audit, source adapter, sequencer, loss preservation, three review rounds, and extensive fork cleanup; 6–9 hours of preparation is unlikely. The evaluator/kit fork changes cannot realistically remain near 200 lines per commit.  
    **Change:** Budget roughly 12–18 engineering hours or 1–2 workdays before launch, excluding review latency. Split large fork changes by behavior—identity/closure, training evidence, admission, cell manifest, historical-source adapter, table/figure, sequencer—rather than forcing artificial 200-line commits. Verbatim copies can remain a documented exception.

20. **MINOR — The acoustic invariance assumption should be narrowed.**  
    **Evidence:** The target is mono and the augmentation is a coordinate-gauge reflection, so leaving the target RIR and `context_audio` unchanged is consistent for scalar omnidirectional acoustics. The repository does not independently verify absence of source/receiver directivity or handed material orientation.  
    **Change:** State that target invariance assumes scalar mono simulation, omnidirectional source/receiver behavior, and no handed directional/material metadata. This is a scope assumption, not a measured result.

Current status: exp_43 plan r0 requires revision; no files, environment, git state, GPU jobs, training, or evaluation were changed or run.

Questions only the human owner can answer:

- Is there external AcousticRooms documentation or calibration data establishing the depth-camera heading and handedness that is not present in this repository?
- Are the historical YAW and OrbitRIR intermediate checkpoints and offline loss histories still available, and where are their immutable hashes recorded for exp_44?
- Is the scientific goal a causal flip-versus-YAW estimate? If so, is there budget for a contemporaneous same-pin YAW control and preferably additional training seeds?
- Should the four full-split mirrored diagnostic cells be added, or should the report explicitly avoid any mirror-robustness claim?
- Which owner choices should be frozen now for GPU order, Stage-B execution, deadline handling, and unattended launch authorization?