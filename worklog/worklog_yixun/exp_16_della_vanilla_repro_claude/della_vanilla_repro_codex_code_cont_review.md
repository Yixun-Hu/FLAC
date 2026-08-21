**Reviewer:** OpenAI Codex (gpt-5, codex-cli 0.147.0, `codex exec`, GitHub-connector read-only — local sandbox unavailable on della) · **Date:** 2026-08-21
**Round:** code round F (CONT mode for the chunking A/B) · **Verdict:** APPROVE

### BLOCKING

### NON-BLOCKING

- The claim that both completed chain and probe trees report “nothing to submit” is not true for a real probe invocation. Probe selects the clean-tree gate and exits rc=10 on its checkpoint before reaching the rc=2 completed-tree check. Only `DRYRUN=1` continues past that gate and reaches “nothing to submit.” This is safe and pre-existing, but the stated behavior should be corrected or the checks reordered. `della_chain_submit.sh:228-243`, `della_chain_submit.sh:261-263`

- “Empty save-dir” is slightly stronger than the implementation. `newest_step` only recognizes `epoch=*-step=*.ckpt` and uses zero both for “none” and a step-zero checkpoint; both wrapper and driver accept zero. No silent resume follows—the driver adds `--ckpt-path` only when `S > 0`—but literal “any pre-existing checkpoint refuses” is not guaranteed. `della_chain_submit.sh:116-130`, `della_chain_submit.sh:236-241`, `della_chain.sbatch:127-153`, `della_chain.sbatch:319-330`, `della_chain.sbatch:393-416`

- The stamp is not wholly inert in CONT. A positive-step checkpoint is guaranteed to hit rc=9 before stamp processing, but an attempt that fails before its first checkpoint can be resubmitted: the stamp advances at `S=0`, and a third such start can emit `CHAINHALT`. This does not invalidate a later successful arm because every attempt remains a fresh, seed-42 process. Its cancellation call is inert because CONT clears `MANIFEST_LATER`. `della_chain.sbatch:194-212`, `della_chain.sbatch:319-330`, `della_chain.sbatch:362-390`

- The exact 67,500-step invariant resides in the required wrapper; the driver independently enforces only `CHAIN_TOTAL == CHAIN_CHUNK`. A hand-written direct submission could therefore run another equal total/chunk pair. `della_chain_submit.sh:153-167`, `della_chain.sbatch:241-242`

Overall, the authorized wrapper path preserves the A/B: one held `exp16-cont` job, `TOTAL=CHUNK=67500`, separate save-dir, identical training argv except `--save-dir` and `--max-steps`, seed exactly 42, no resume argument, dependency, or manifest. CONT/PROBE exclusivity and both-sided job-name interlocks are sound. The driver cannot scancel in CONT; wrapper `scancel` calls remain correctly active only to roll back its own held job on transaction failure. `CONTFAIL`/`CONTRESULT` and `_cont`/`_chainprobe` log suffixes are sensible and unambiguous.

> Invocation note (Planner): GitHub-connector review at HEAD 9d23cfb. Verdict APPROVE; the four nits are recorded as accepted limitations of the authorized wrapper path (probe rc-ordering message, step-0-ckpt edge, S=0 stamp retries, wrapper-side 67500 pin).
