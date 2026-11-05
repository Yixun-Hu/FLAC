**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-11-05 · **Round:** plan r4 (exp_35 orbitrir_yaw_evidence: plan r4 at e2166326)

**Verdict:** APPROVE

## r3 finding status

| Finding | Status | Assessment |
|---|---|---|
| **BLOCKER — GPU-1 FIFO exclusion** | **RESOLVED** | §13 takes option 3 exactly: it withdraws the plan-authored absolute table-job exclusion, restates Yixun’s decision 6 as sharing GPU 1 with s43 training, and explicitly accepts and discloses the residual possibility of one-cell overlap with the subsequent table job. The stricter 3.5 s/step predictor, `2 × planning duration + 30 min` margin, durable s44-completion evidence, and new-process stop latch are appropriate mitigations, not claims of mechanical exclusion. |
| **NIT 1 — state durability** | **RESOLVED** | §13 requires state-directory `fsync` after rename and a parsed-offset/last-4-KiB content anchor with hard-stop behavior. |
| **NIT 2 — chunk-5 ledger fields** | **RESOLVED** | §13 explicitly defines probe rc, JSON SHA, sidecar SHA, final-log SHA, and `PROBE_CONTRACT_BF` result. |
| **NIT 3 — frozen external evidence** | **RESOLVED** | §13 requires the launch entry to record the queue, evaluator, and admission-helper SHAs plus the s44 completion evidence. |

## Remaining BLOCKING finding

**None.** Once the code faithfully implements §11–§13 and passes the already-required reviews and validation ladder, no plan-level issue blocks phase (i).

## NIT

§13’s “ONLY way” wording is slightly too strong: overlap could also result from an unusually long cell exceeding `2 × planning duration + 30 min`, or a stale step observation—not solely training running faster than 3.5 s/step. This is non-blocking because §13 broadly accepts the resulting one-in-flight-cell overlap risk and mandates the stop latch, but the disclosure would be more exact if it named forecast failure generally.

## What I verified

- FLAC is on `localization-exp` at `e216632660eed0d4d1eb617f6ffb4e4fae6ccbc1`.
- Its parent is the reviewed r3 commit `2cf630199494748a7715be88d5428a707cfd1fe1`; the parent plan SHA-256 is the previously reviewed `f59a37c…`.
- The plan diff is the r4 status-header update plus the appended §13. Current plan SHA-256 is `d17fea82b86e782fb2fb056be5057b68d10817a35bd6d1db26f5f2035ebe1623`.
- The cited FIFO evidence is consistent and durable:
  - `.jobs` enqueues `ACEattn 44` at the final 40,000-step checkpoint.
  - `.status` records exit `0` for that same checkpoint.
  - `.log` records `EVAL_DONE ACEattn s44 cells=40 2026-11-04T16:01:40-05:00`, followed by verified evaluator-group cleanup.
  - The queue implementation writes status only after waiting for the evaluator and capturing its exit code.
- Current external SHAs match §13:
  - Queue: `9394ca903931de7778f9665e36b355d9cfab1d61e693703a29e02d43c505caaa`
  - Evaluator: `94fcc9e17da578049f0c4a3a05798d00f7d312426300b1f20786367b7b863d6c`
  - Admission helper: `8572bb0305bcd1f1f245a9d2b31ca9760ba7ed036bccf5bf612d2b2208b33907`
- The queue’s only checkout diff is the `TRAIN_JOBS` array.
- I did not independently reproduce the historical “no eval process at 11:58” observation; the persisted completion/cleanup evidence suffices, and §13 requires launch-time re-verification.
- I ran no GPU/process command, evaluator, runner, collector, probe, test, package operation, or side-effecting script, and modified no files.
