# std_aug — Codex code review, round 1

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-13 02:48–02:56 EST · **Round:** code 1 of 3 · **Reviewed:** `6a51f741..08c8936e` (training path; amendment A1) · **Prompt:** `std_aug_codex_code_r1_prompt.txt`

---

VERDICT: REVISE

1. **BLOCKING — `worklog/worklog_yixun/exp_43_std_aug_claude/exp43_frame_audit.py:975,1090,1163`; `src/tests/test_exp43_frame_audit.py:837` — the frozen audit inputs are not actually bound.**  
   Evidence: `load_inputs` trusts a self-consistent `selection_sha256`; `--split` is optional. A reduced/replaced manifest with recomputed hashes can therefore run and potentially PASS. The forgery test only refuses it when `--split` is supplied. Additionally, `_aggregates` at line 820 derives categories from surviving rooms, so an omitted/empty category gets no leave-one-category-out check.  
   Required change: require `--split`, bind the committed input SHA-256 `5124c23b504130378ab73385dc4b46023ec2ab2d09807092dd674e91e580af84` and/or selection SHA-256 `035f620f6ce858ac7023f6e885c63a7a3f12ac15990eb818b94dfd61ecbb07e6`, and refuse missing/empty registered categories. Add tests for a self-consistent forged subset without `--split`, changed split/tree, and an empty category.

2. **MINOR — `worklog/worklog_yixun/exp_43_std_aug_claude/frame_audit_sim/sim_gate.py:5`; `frame_audit_sim/design_sources.py:6,79` — the archived evidence generators are checkout- and scratchpad-specific.**  
   Evidence: both import the test module through `/home/yixunhu/codespace/exp-43-std-aug/...`; `design_sources.py` writes to a vanished Claude scratch directory. The committed JSON remains readable and the formal tests cover the main geometry, but the evidence supporting A2 cannot be reproduced from another clone.  
   Required change: resolve the repository root relative to `__file__`, emit to stdout or a required `--out`, and add a lightweight import/CLI test.

## Answers

1. **Mirror correctness:** Yes. For AR’s grid, column reversal maps \(\theta_j\) to \(-\theta_j\), and negating depth channel 1 gives exactly \(M=\mathrm{diag}(1,-1,1)\). All geometry consumed by the RF conditioners is covered: `depth`, `source`, `source_vit`, `context_poses`, and `context_poses_vit`. `context_audio`, masks, scene IDs, and the mono target carry no reflected geometry.

   The roll is applied first and the mirror second. Aliased pose inputs are cloned separately, preserving their values while avoiding write-through; dtype and device are preserved. Each DDP rank gets a private generator seeded by seed, optimizer step, and global rank, so streams are rank-distinct and resume-exact.

   Leaving the target and context RIRs unchanged is correct under the registered assumption: scalar mono responses from omnidirectional source/receiver simulation are invariant to reflection of the whole scene.

   Source deviations:

   - Relative render tolerance \(10^{-6}D\): **accepted**; it is the appropriate scale-aware numerical claim.
   - 102,892 unequal fp32 entries rather than 50,921: **accepted**; the invariant is non-bitwise equivalence plus the error bound, not that incidental count.
   - Quarter-turn bit-exactness only in fp32: **accepted**; AR supplies fp32 and other dtypes retain the numerical contract.
   - Contract check before drawing bits: **accepted**; malformed-input refusal is correctly independent of random outcomes.
   - No pose-finiteness check: **accepted**. This matches the registered pose contract and existing yaw operator; AR produces finite coordinates. A finite-pose test would be defensive hardening, not a launch requirement.

2. **Training integration:** With `flip` absent or false, the factory adds no keyword and `_apply_yaw_aug` returns immediately after the existing roll. There is no additional RNG draw or mirror-contract check. The new constructor parameter is keyword-only; F12 was correctly narrowed to “last positional-or-keyword parameter.”

   With flip enabled, offsets are drawn first, then bits from the same generator. Step/rank seeding, resume behavior, global-RNG isolation, refusal rules, and the `flip=0.5` banner are correct. Only `training_step` invokes augmentation; validation, test, and evaluator behavior are unchanged.

   `fa_invariant` and canonicalization are refused with yaw augmentation. RF declares no `are_lambda`; a hypothetical yaw+ARE combination is not explicitly refused, but is outside this arm. CFG dropout happens after transformed conditioning is computed, so it does not alter augmentation semantics.

3. **Frame audit:**

   a. The geometry is correct. The tool evaluates \(v_\text{world}=R(2\pi k/512)Sv_\text{depth}\); its inverse mapping, AR pixel-center convention, Score A containment test, and Score B receiver-baseline reprojection agree with `AR_md.py`.

   b. The registered full/LOCO/shared-bootstrap gate is implemented as documented, including tie-set containment and the 990/1000 threshold.

   c. **A2 is a sound substitute and I recommend approving it.** The archived true-frame simulations show that Score A’s precise maximum is unstable while Score B is exact. Retaining Score B’s ±2-column rule preserves the precise heading and handedness check; Score A still independently rejects the wrong handedness and large rotations.

   It protects against:

   - A mirrored frame: yes; the winning \(S=M\) hypotheses are rejected.
   - A rotation offset greater than two columns: yes, through Score B.
   - A wrong azimuth origin hidden by roll but exposed by flip: yes, if its error exceeds the intentionally allowed ±2 columns. Errors of one or two columns are deliberately accepted.

   d. NaN/Inf depths are refused; zero depth is invalid rather than silently treated as a surface, and all-zero/undefined scores fail. Empty overall input fails. However, the current frozen-input and empty-category holes can allow a PASS on the wrong selection; finding 1 must be fixed.

4. **Kit:** The operational bindings are correctly retargeted to exp_43: RF arm/config and digest, banner, project, run ID, A6000 2×32 identity, NAS paths, manifest, evidence schemas, admissions, and FULL/SMOKE/DDP_SMOKE expectations. The remaining exp_40 references are intentional—the roll-only golden and shared card lock—not executable C16/`offset_step` identity.

   W&B preservation is correct for normal completion, failure, and trapped INT/TERM paths: it searches all plausible PL locations, requires exactly one candidate, preserves before temporary-directory deletion, records hashes, archives to `<RUN_ID>__wandb`, and re-verifies the NAS copy. Zero or multiple candidates fail closed. A failed preservation retains the temporary directory. Refusing a stale `.partial` is appropriate.

   The applied-flip evidence observes the actual objects returned by rotation, requires mirror calls to receive those exact rotated objects, requires the wrapper output to use the mirrored result at precisely the expected indices, and independently recomputes offsets followed by bits. Combined with the pinned mirror source and unit tests, this proves that real-run flips occurred after roll with the exact deterministic \(p=1/2\) stream.

   All seven disclosed kit deviations are **accepted**: moving `ARM_CONFIG` into 6b; double project nesting; preserving the exp_40 golden without snapping; schema break to `stream_digest`; private recomputation; sibling W&B archive/manual `.partial` cleanup; and fresh stamped history directories.

   A W&B telemetry gap is plausible in an otherwise healthy optimizer/checkpoint run—for example, writer or final-flush failure. Therefore the strict 400-row requirement can wrongly block admission of a scientifically valid model. I nevertheless accept the no-override policy as a conservative evidence gate: the checkpoint and run evidence remain recoverable, and any later exception should require a reviewed policy change rather than a silent override.

5. **Tests:** Concrete gaps are:

   - No test pins the exact committed frame-input file/selection hash.
   - No test proves a self-consistent forged manifest is refused when `--split` is omitted.
   - No test requires every registered category to remain nonempty and receive a LOCO aggregate.
   - No test proves the archived simulation scripts run from another checkout.
   - Actual multi-process DDP/resume and real W&B preservation remain ladder tests at L4, not unit tests.
   - The real `.wandb` extraction test is opt-in, so the normal suite proves the parser mainly with synthetic files.

   Disabled-path equality, unchanged-RIR, and “validation/test never flip” tests would individually pass if the feature were removed, but the independent-oracle training test, applied-index checks, mirror spy, and flip golden make the suite as a whole non-vacuous.

6. **Regression/history tools:** `compare` cannot PASS if a previously passing node fails, errors, becomes skipped/xfail, or disappears. It also validates the JUnit and terminal summaries against each other.

   The history extractor cannot silently drop or duplicate a qualifying `train/loss` step: missing keys are refused, duplicate/off-cadence/nonmonotonic steps fail cadence, and missing expected steps become gaps. History records without `train/loss` are intentionally omitted and counted in the sidecar.

7. **Rule compliance:** The commit partition and disclosed 6b size deviation are acceptable. `src/data/yaw_rotation.py` is byte-identical to base (`69205176…`). The training path uses the required vanilla protocol and NAS `checkpoints/` tree; announcements 01, 05, and 08 apply to the deferred evaluation kit and are not violated here. No training or other running script was edited in place.

   No account address appears in an exp_43 file or newly added diff line. The repository already contained account addresses in `HANDOFF.md`/`issue_report.md` at the base commit, so a literal whole-repository interpretation was already false; this branch introduces none.

8. **40k-run value:** After findings 1–2 are fixed, A2 is explicitly approved or rejected, and L1–L5 pass, I found no training-path defect that would make the 40k run worthless or irreproducible. Do not launch before the frozen-input binding is fixed. Keeping the registered gate instead of A2 will likely stop at L1 on a valid frame, but it will not launch a bad run.

## Verified correct

- Mirror implementation and all conditioning-field coverage.
- Roll-then-mirror RNG order, rank/resume isolation, and train-only dispatch.
- Exact RF config delta and keyword-only integration.
- W&B preservation/archive/admission bindings and applied-flip evidence.
- Regression comparator and history cadence logic.
- `src/data/yaw_rotation.py` base hash.
- `src/tests/test_yaw_mirror.py`: **108 passed** on CPU, read-only.

## Status

- exp_43: **awaiting revision**; no GPU run launched and no ETA until fixes, A2 decision, and L1–L5.
- exp_40: **complete**.
- exp_42: **on hold** pending Yixun’s follow-up.
- exp_44: **planning**, awaiting its registered decisions.
- Other historical experiments: closed or deferred per the living handoff; no process was launched by this review.

**Earliest intervention needed:** now/before L1—decide whether to adopt proposed A2. My recommendation is **approve A2 after finding 1 is fixed**.