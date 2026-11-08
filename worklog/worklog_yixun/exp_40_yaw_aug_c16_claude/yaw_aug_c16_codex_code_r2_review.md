**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-08 · **Round:** exp_40 code review 2 of 3 (HEAD e5954933)

**Verdict:** REQUEST-CHANGES

The treatment is correct, but the experiment is not launch-ready. The remaining blockers are concentrated in provenance and acceptance gates: neuronic exclusivity can still be split across control directories, the claim probe is not fully bound to the submission, training evidence validators remain shallower than their producers, and evaluation receipts do not prove their declared runtime-input identity.

## Q1 — Round-1 findings

- **Only partially fixed:** B1, B2, B3, B5, B9, B10, H2, M1, M2.
- **B6:** the all-16 NAS rehash and derived admission are fixed, but admission now introduces the live-checkout provenance defect in HIGH 2.
- **Fixed:** B4, B7, B8; H1, H3–H13; M3–M7.
- The prior NITs were updated, although some new comments overstate validator strictness.

Specifically:

- B1: capture is now self-exporting and per-pin evidence exists, but assembly never validates `source_tree_digest`.
- B2: durable shared state/flock is implemented, but “one” control directory is still an unrestricted environment override.
- B3: real barriers and atomic receipts are implemented, but source SHA, mount validity, and the current mount are not validated.
- B5: exact top-level FULL schema is implemented, but the smoke/PILOT validator is shallow and FULL nested records remain non-exact.
- B9/B10: the committed E0 reference and broad receipt machinery exist, but runtime-input and E0 receipt fields are not actually bound by the validators.
- H2: killable publication and a real-NAS probe exist, but the probe validator does not validate the exact checks or mount, and the manifested-file path omits the final source rehash.
- M1: flash state is restored and validation executes a negative test, but the test-step test never calls `test_step`.
- M2: most original test gaps were filled, but several new tests named “strict” or “every identity mutation” omit the fields that currently fail open.

## Q2 — Round-2 defects

Yes. The principal defects are BLOCKING 1–4 and HIGH 2–3 below. They include a cross-control-directory exclusivity race, stale or foreign claim-probe acceptance, shallow evidence schemas, unbound evaluation runtime manifests, live-tree admission, and incomplete CIFS/NAS receipt validation.

I did not find an additional defect in the intended PL 2.1 save-hook barrier/broadcast sequence, torch 2.7 treatment path, allocated-GPU UUID mapping, or bounded bash process-group cleanup.

## Q3 — Treatment and absent-key equivalence

The production treatment still matches plan §2:

- Draw the original uniform column offset, then apply `32 * floor(d / 32)`.
- The result is exactly uniform over the 16 offsets.
- Quantization consumes no RNG and occurs after the original draw.
- `offset_step` is optional, strictly validated, and the constructor argument remains last.
- With the key absent, the factory passes no new keyword, so exp_15/exp_17 use the old path.
- `offset_step=1` preserves the old offset values.
- `src/data/yaw_rotation.py` has no diff from `2b42468c`; the new helpers are isolated in `src/data/yaw_offset_step.py`.
- Validation and test code do not dispatch yaw augmentation.

Relevant code: `src/data/yaw_offset_step.py:10-52`, `src/training/factory.py:11-89`, `src/training/diffusion.py:205-212`, `:275-290`, `:717-744`, `:825-834`.

## Q4 — Evaluation

The core protocol is substantially correct:

- The 85 new cells and their block counts are correct.
- Argv derivation preserves the exp_25 YAW protocol except for the declared substitutions.
- E0 references are committed and hash-pinned.
- The evaluator uses SDPA, full unseen split, EMA, strict metrics/stream/dump checks, checkpoint caching, card/per-UUID locks, and final selected-receipt proof.
- The 92-cell comparator registry goes through the source validators.
- Collection includes the promised tables, P1 off-grid `d`, paper-import CSV, frozen CSV figure path, figure contract, and exact five-file table contract.

It is not yet acceptable because the durable receipt validators do not prove the runtime-input manifest or all E0 identity fields, and later verification does not establish that the recorded AR receipt was valid/fresh when the cell ran.

## Q5 — Records and command sheets

Not fully mechanical yet:

- The A6000 step labelled “every gate” is a FULL dry run before the required AR/NAS/smoke receipts and is expected to fail.
- A fresh neuronic worktree lacks the untracked `receipts/` directory, so its first `ar-verify --out .../receipts/...` fails before the submitter can create it.
- Evaluation dry-run returns success even when its gates report errors.
- Admission runs through the live checkout rather than the immutable training source.

The remaining copy/publish/admit and evaluation ordering is otherwise coherent.

## Q6 — Other

The permitted pytest command could not start in the review interpreter:

```text
/home/yixunhu/miniconda3/bin/python: No module named pytest
```

I did not install anything. The committed log reports `719 passed`, but that result was not independently reproduced in this review.

The worktree also had pre-existing state outside HEAD: modified `yaw_aug_c16_command.md`, untracked `AcousticRooms`, and untracked `receipts/`. I reviewed the requested committed diffs and did not modify these paths.

# Findings

## BLOCKING

### 1. The neuronic at-most-one guarantee can still be split across multiple control namespaces

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_submit_neuronic.sh:43-49`  
`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:2209-2221`, `:2576-2582`  
`src/tests/test_exp40_kit.py:442`

`EXP40_CONTROL` may be any absolute canonical path. Two operators or checkouts can therefore submit the same FULL identity with different values, obtaining separate flocks, states, claim files, and probe receipts. Both jobs can pass their respective execution claims. The concurrency test proves only that two submitters using the same directory serialize.

**Required change:** pin neuronic operation to exactly one non-overridable control path, or resolve a single centrally registered namespace tied to a validated mount identity. Reject alternate `EXP40_CONTROL` values. Add a test proving two attempted namespaces cannot admit the same execution identity.

### 2. The claim-probe receipt is not bound to the submitted source or current shared mount

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:2380-2418`, `:2834-2838`  
`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_claim_probe.sbatch:28-36`  
`src/tests/test_exp40_kit.py:1039-1064`

Any 40-hex `source_sha` passes; it is never compared with `EXPECT_SHA`. The nested mount objects need only produce one identical `(source, fstype)` tuple, so two empty or failed mount records can pass. `findmnt_rc`, non-empty source/type, nested schema, and the control directory’s current mount are not checked.

Thus a fresh-looking receipt from another source revision—or from a mount that has since changed—can authorize an irreversible FULL identity.

**Required change:** pass `EXPECT_SHA` into `probe_receipt_reasons` and require exact equality; validate the exact nested mount schema, non-empty source/type and `findmnt_rc == 0`; compare the current control mount with the probed mount; add valid-but-wrong SHA and empty/failed/current-mount mutation tests.

### 3. Smoke/PILOT and FULL evidence validators accept records weaker than the producer contract

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:827-840`, `:917-926`, `:1724-1834`, `:2704-2714`  
`src/tests/test_exp40_kit.py:1257-1286`

`receipt_reasons` has no exact schema and checks only selected header fields, evidence verdict, and two exit codes. It does not validate mode, run ID, stamp, argv, publisher status, exact ranks, summaries, offsets, SyncBatchNorm, GPU identities, or finite/positive PILOT timing. The synthetic reference receipt in the tests omits many producer fields and passes.

The FULL validator has an exact top level but not exact nested schemas. It accepts `fit_start` records without inner rank, GPU UUID/name, SyncBatchNorm count, or batchnorm count, and summaries without rank or exact producer fields. It then uses the shallow reference validator for initialization parity.

**Required change:** define separate exact DDP_SMOKE and PILOT schemas and validate every produced acceptance field before FULL or timing use. Require a finite positive `sec_per_step`. Make FULL `fit_start`, counters, summaries, signatures, rank identities, GPU/backend/SyncBN fields, and reference receipt exact and type-strict. Mutation-test every producer field and every omission.

### 4. Evaluation receipts do not prove their declared runtime-input or E0 identity

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:916-979`, `:1013-1062`, `:1596-1625`, `:1992-2018`  
`src/tests/test_exp40_eval.py:513-537`, `:841-859`

`runtime_inputs_sha256` is merely checked for hex syntax; it is never recomputed or compared with identity supplied to the validator. Nested runtime-input entries do not need exact `{path, sha256}` schemas, absolute paths, or existing bytes. Later `verify` and `collect` do not validate the stored AR receipt or prove it was fresh at the cell’s timestamp.

The E0 gate ignores its `runtime_inputs_sha256`, `ar_receipt_sha256`, `dataset_config_sha256`, and timestamp. Consequently, an exact-top-level but internally inconsistent E0 receipt can authorize all non-E0 cells. The tests called “strict” and “every identity mutation” do not mutate these fields.

**Required change:** carry an exact runtime-input identity into every validator, preferably using a canonical manifest digest plus exact entries; validate stored paths, hashes, admission, and the AR receipt relative to cell time. Derive and compare E0 runtime/AR/config identities from the three bound cell receipts. Use the same validator in run, verify, collect, figure, and table publication, with mutation tests for every field.

## HIGH

### 1. Golden assembly records source-tree provenance but never validates it

`src/tests/exp40_yaw_aug_golden_capture.py:291-314`, `:322-356`  
`src/tests/test_yaw_aug_c16.py:92-116`

The capture command correctly exports the named SHA itself, but `assemble_reasons` never inspects `source_tree_digest`. A digest can be changed or exchanged between otherwise valid captures and assembly still succeeds; the tests only prove that capture JSON and fixture repeat the same value.

**Required change:** give assembly the repository, require an exact capture schema, recompute `git ls-tree -r <sha> -- src` for every declared pin, and compare its digest. Add altered and swapped source-tree-digest tests.

### 2. Admission reinterprets training evidence with the current live checkout

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:1796-1802`, `:1918-1935`, `:1944-1982`, `:2753-2766`  
`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/yaw_aug_c16_neuronic_commands.md:44-46`, `:121-124`

The run-result records `runtree`, but `resolve_run_result` neither verifies nor uses it. Admission recomputes offsets through hard-coded `ROOT`, reads the model config from `tree=ROOT`, and builds the strict-load wrapper from that live source. The CLI always passes its own checkout’s `ROOT`.

If the checkout advances between training and admission, a valid run may be rejected—or interpreted under code other than `train_sha`. The neuronic path makes this more pronounced because the recorded remote run-tree path is unavailable on the A6000 host.

**Required change:** rebuild or carry an immutable tree for `rr["sha"]`, verify it against the evidence run-tree digest and closure, and perform recomputation, config loading, and strict wrapper construction exclusively there. Reject current HEAD/closure drift instead of silently using it.

### 3. The NAS capability and manifested-checkpoint paths remain incompletely validated

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:1491-1525`, `:1528-1616`  
`src/tests/test_exp40_kit.py:1219-1237`

The capability validator accepts any five true check names and ignores the recorded mount entirely. It does not require the exact primitive set or verify that the current destination still has the probed CIFS identity.

For an already-manifested checkpoint, `publish_pass` rehashes only the NAS destination. It does not rehash the source against its `.saved` marker, so a same-size source mutation can be acknowledged and training can continue until the much later final audit.

**Required change:** require the exact five named probe checks, exact mount schema, successful `findmnt`, and equality with the current destination mount. On every acknowledgement path—including already manifested files—rehash the source against the save marker immediately. Extend real-receipt and same-size source-mutation tests.

## MEDIUM

### 1. The launch sheet is not runnable from a fresh checkout in its stated order

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/yaw_aug_c16_neuronic_commands.md:23-32`, `:69-80`  
`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_kit.py:185-201`

The A6000 step labelled “every gate” invokes FULL before the receipts it requires, so it is expected to refuse rather than pass all gates. On neuronic, the first `ar-verify` writes into `$K/receipts`, but that directory is untracked and is not created until the submitter runs later. `atomic_write_json` does not create its parent.

**Required change:** create `$K/receipts` explicitly in the prerequisites; move the FULL dry run after AR, NAS, and DDP-smoke receipts exist, or label the initial command as an expected refusal and provide a later all-gates dry run.

### 2. Evaluation dry-run reports gate failures but exits successfully

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_eval.py:1913-1940`  
`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/yaw_aug_c16_eval_commands.md:80-85`

When `--dry-run` is set, accumulated worktree, AR, environment, or E0 gate errors are printed and then the function returns `0`. This makes the documented preflight unsuitable for mechanical scripting.

**Required change:** dry-run should skip only locks and evaluator spawning. It must return nonzero through the same named refusal when any required gate fails.

### 3. Negative tests omit the remaining acceptance holes

`src/tests/test_exp40_kit.py:442`, `:1039-1064`, `:1219-1231`, `:1257-1328`  
`src/tests/test_exp40_eval.py:513-537`, `:841-859`

Missing cases include:

- the same execution identity under two control directories;
- a valid but wrong claim-probe source SHA;
- empty/failed or changed mount identities;
- renamed five-entry NAS check maps;
- omitted nested FULL evidence fields;
- altered runtime-manifest SHA or nested paths/hashes;
- altered E0 AR/runtime/config fields.

**Required change:** add these exact mutations and assert named refusals, not merely a truthy reason list.

### 4. The test-step “no augmentation” test does not run `test_step`

`src/tests/test_yaw_aug_c16.py:412-427`

Validation is exercised with `_apply_yaw_aug` replaced by a raising stub. The test-step case only inspects `co_names` and an attribute, so indirect dispatch or a future helper call would pass.

**Required change:** execute `test_step` with `_apply_yaw_aug` replaced by a raising stub, using a minimal viable test configuration or a narrowly stubbed sampler.

## NIT

### 1. Local FULL output uses “PASS” before experiment admission

`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/exp40_train_local.sh:20-22`, `:210-227`  
`worklog/worklog_yixun/exp_40_yaw_aug_c16_claude/yaw_aug_c16_neuronic_commands.md:38-46`

The sheet does direct the operator to admission, but the launcher’s final banner and run-result verdict say PASS before the experiment-level admission exists.

**Required change:** label this state `TRAINING_AUDIT_COMPLETE_PENDING_ADMISSION` or equivalent; reserve experiment “PASS” wording for successful admission.

## What I verified

- Read the requested SOP, all announcements, final plan including binding §14, round-1 review, round-2 worklog mapping, and commit ledger.
- Reviewed both `15f73ff5..e5954933` and the complete `2b42468c..e5954933` experiment history.
- Verified the treatment mathematics, factory/constructor behavior, absent-key path, config delta, dispatch sites, and base-identical `yaw_rotation.py`.
- Traced evidence → audit → run result → admission → runtime manifest → cell/E0 receipt → collect/figure/table.
- Checked the 85-cell plan, exp_25 argv derivation, E0/reference construction, 92-cell comparator registry, and collection/table contracts.
- Attempted exactly the permitted pytest command; it failed before collection because `pytest` is not installed in the selected interpreter.
- Ran no training, evaluation, GPU, Slurm, checkpoint-loading, process-control, package-management, or write command, and modified no files.