**Reviewer:** OpenAI Codex (`gpt-5.6-sol`, xhigh, read-only sandbox) · **Date:** 2026-08-15 · **Round:** r3 (verification of the r2 fixes)

Reviewed at HEAD `6da5ca1` on `exp17-yawaug-scratch`. Static review only.

⚠️ **Context added after the review:** on 2026-08-15 Yixun decided to stand down this launcher and rely on the FULL run already training from the `exp-17-yawaug-a6000` worktree. The launcher and guard-suite findings below therefore do NOT block a running experiment — but they DO block any reuse of these two scripts, and the test-file finding is a live correction to a scientific claim that remains in the repo.

Note that r3's grad-checkpointing finding *supports* that decision: the running arm has checkpointing ON, matching P1 exactly, so it carries none of the numerical confound this file's OFF setting would have introduced.

---

# Verdict: REQUEST-CHANGES

Static review only at current HEAD `6da5ca1` on `exp17-yawaug-scratch`. The branch advanced during review, but none of the four reviewed files changed. No tests, launchers, training, guard suite, or GPU commands were run.

| File | Verdict |
|---|---|
| `FLAC_AR_YAWAUG_A6000.json` | **SHIP** |
| `yaw_aug_a6000_launch.sh` | **REQUEST-CHANGES** |
| `yaw_aug_a6000_guardtests.sh` | **REQUEST-CHANGES** |
| `test_yaw_aug_a6000_arm_config.py` | **REQUEST-CHANGES** |

## R2 closure

- **Resolved:** All nine pin hashes match current bytes; `defaults.ini` and `data/AR/train.json` also match P1 commit `e50d098`. [launcher:71](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:71)
- **Resolved:** `SMOKE_STEPS=25` is pinned and range-checked below cadence. [launcher:52](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:52)
- **Resolved:** Uncomputable R3 projections fail. [launcher:357](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:357)
- **Resolved:** The backticked termination-marker patterns are shell-correct and match the recorded Lightning output. [launcher:217](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:217)
- **Resolved:** FULL requires endpoint-marker evidence and a `*step=40000.ckpt`. [launcher:311](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:311)
- **Resolved:** Config validation now precedes every FULL-only prerequisite. [launcher:132](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:132)
- **Partially resolved:** Completed-smoke checks are much stronger, but evidence is not bound to the current revision/pin set.
- **Partially resolved:** The grad-checkpointing evidence is now described factually, but the conclusion still overstates what it establishes.
- **Not resolved:** Guard sections H/I still duplicate or inspect historical logic rather than executing the launcher’s post-run branches.
- **Not resolved:** Production-log hiding/cleanup remains concurrency-unsafe.

## Per-file findings

### `FLAC_AR_YAWAUG_A6000.json` — SHIP

No finding. The only changes from P1 are:

- `gradient_checkpointing: true → false` at [line 87](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/FLAC_AR_YAWAUG_A6000.json:87) and [line 103](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/FLAC_AR_YAWAUG_A6000.json:103).
- The registered `training.yaw_aug` block at [lines 195–199](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/FLAC_AR_YAWAUG_A6000.json:195).

### `yaw_aug_a6000_launch.sh` — REQUEST-CHANGES

- **Blocking — reviewed-source binding remains incomplete.** The two named pin gaps are fixed, but HEAD/reviewed-tree binding is still absent. The launcher accepts any clean commit at [lines 124–130](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:124). Clean committed changes to unpinned model, dataset, loss, or conditioner code pass. The executed DINO gate script at [lines 264–267](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:264) is also under the exempt `worklog` tree and is not content-pinned.

- **Blocking — smoke evidence is not bound to the current nine-pin revision.** [Line 221](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:221) accepts the generic substring `source pins OK`; an older seven-pin smoke can qualify. Neither current HEAD, the current pin manifest, nor `config contract OK` is required. The negative endpoint diagnostic at [line 317](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:317) also reproduces the positive marker searched at line 217, so an appended/composite log can counterfeit that fact.

- **Blocking — `pipefail` makes post-run log checks unreliable.** With `set -o pipefail` at [line 32](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:32), the `tr | grep -q` pipelines at [lines 304, 316, and 344](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:304) can return nonzero after a successful early match when `grep -q` closes the pipe and `tr` receives SIGPIPE. That can falsely reject the early banner; more seriously, it can miss an early NaN/Inf and allow a smoke PASS.

- **Blocking — FULL can still exit zero with non-finite training.** NaN/Inf checking is SMOKE-only at [lines 342–371](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:342). A FULL run that becomes non-finite later can reach 40,000, write a checkpoint, and satisfy every post-run gate.

- **Non-blocking — live-log race.** Output is written through asynchronous process-substitution `tee` at [line 109](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_launch.sh:109), then immediately reread after training. Tail marker data can briefly remain undrained, causing a false endpoint failure.

### `yaw_aug_a6000_guardtests.sh` — REQUEST-CHANGES

- **Blocking — H remains deletion-vacuous.** H3/H4 reimplement `tr | grep` locally at [lines 217–223](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_guardtests.sh:217); H5 reads a historical smoke at [lines 224–227](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_guardtests.sh:224). Deleting the current launcher banner gate leaves all H cases green.

- **Blocking — I still does not drive production verdict paths.** `r3_verdict` is a copied, hard-coded implementation at [lines 237–245](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_guardtests.sh:237); I4/I5 test literal strings at [lines 246–255](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_guardtests.sh:246). Deleting or breaking the launcher’s R3, endpoint, or smoke-checkpoint branches remains undetected.

- **Blocking — cleanup can destroy concurrent production evidence.** [Lines 56–59](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_guardtests.sh:56) delete every new `*_train.log` absent from the initial snapshot, including legitimate concurrent smoke/FULL logs. `hide_all_smoke` and `unhide_all_smoke` at [lines 39–42](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_guardtests.sh:39) broadly move production files with `mv -f`, and unhide every unrelated `*.guardhidden`.

- **Blocking — synthetic files still enter the production evidence directory.** The decoy at [lines 196–200](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_guardtests.sh:196) is deliberately nonqualifying, so the original concurrent false-accept defect is fixed. Nevertheless, the stated “no synthetic evidence in production” property is false; the fixed path can overwrite and cleanup can delete a pre-existing file.

- **Blocking — coverage remains incomplete.** A1–A4 lack `DRY_RUN=1` at [lines 104–107](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_guardtests.sh:104). No case independently reaches the clean-tree gate. I2 omits `SMOKE_STEPS`, so restoring environment overrideability would leave all cases green. [guard:230–233](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_guardtests.sh:230)

- **Non-blocking — `argval()` fixes value substrings but not flag-token boundaries.** Exact equality now rejects `400000`, `25000`, etc. However [lines 168–170](/home/yixunhu/codespace/FLAC/worklog/worklog_yixun/exp_17_yaw_aug_a6000_claude/yaw_aug_a6000_guardtests.sh:168) also match `--not-max-steps 40000`; the flag itself is not extracted as a whole token.

G1 now has nominal EXIT restoration at lines 31/53, but the fixed `.guardbak` and unchecked `mv` remain unsafe. The static case count is correctly **52**.

### `test_yaw_aug_a6000_arm_config.py` — REQUEST-CHANGES

- **Blocking — factual restatement accurate, scientific conclusion still too strong.** [Lines 22–31](/home/yixunhu/codespace/FLAC/src/tests/test_yaw_aug_a6000_arm_config.py:22) correctly state the actual evidence: fp32 CPU, at least 100 tensors, `allclose(atol=1e-6, rtol=1e-5)`, and 210/0.0 only as an exp_07 observation. That part is accurate.

  However, [lines 16–18](/home/yixunhu/codespace/FLAC/src/tests/test_yaw_aug_a6000_arm_config.py:16) still say checkpointing “cannot move the trajectory,” while [lines 33–36](/home/yixunhu/codespace/FLAC/src/tests/test_yaw_aug_a6000_arm_config.py:33) say exactness is expected and retain the augmentation-only causal claim. A one-step CPU allclose probe does not establish equality of a 40k-step bf16 CUDA trajectory; sub-tolerance differences can compound.

The right strength is: checkpointing is intended to preserve the mathematical computation; CPU gradients were observed allclose; bf16 CUDA trajectory equivalence is not established, so checkpointing remains a disclosed implementation/numerical confound.

Remaining invalid-success paths are clean committed source drift, stale/composite smoke evidence, NaN/Inf masked by the `pipefail` pipeline, later FULL divergence to non-finite loss, concurrent/foreign endpoint artifacts, and augmentation-only attribution despite the unestablished CUDA checkpointing equivalence.
