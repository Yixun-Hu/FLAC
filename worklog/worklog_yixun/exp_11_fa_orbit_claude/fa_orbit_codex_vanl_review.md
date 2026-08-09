# VANL launch review — Q9 fifth arm (f30802b, 9ce96a3)

**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex-cli 0.146.0, `codex exec`) · danger-full-access; read-only · **Date:** 2026-08-09

# exp_11 VANL launch review — `f30802b` + `9ce96a3`

## Verdict

**GO for launching VANL training now at exact SHA `9ce96a3369f1539d1d4e1081c114c26e28780c75`.**

**NO-GO for VANL evaluations or table publication from this SHA.** The remaining defects are downstream-only and can be repaired while VANL trains.

## Training judgment

- The reviewed launch surfaces are clean; local `HEAD` equals the pushed branch.
- Fresh initialization verification passed: VANL and C4L have identical 753-tensor, 64.50M-parameter state dicts under seed 42, SHA-256 prefix `44a2f6aadd7d2180`.
- `FLAC_AR_VANCKPT.json` differs from C4L only by absence of `training.cond_method` and `training.frame_avg_angles`; both ViT checkpointing leaves are literal `true`.
- The `DEFAULT_FRAME_ANGLES` fallback is inert under the verified vanilla dispatch at [diffusion.py:215](/n/fs/gatrdp/codespace/FLAC/src/training/diffusion.py:215). Checking the dispatch plus the orbit-free serialized config is the correct gate.
- The exact dry run passed at 8×8, SyncBN-64, seed 42, 40,000 steps, checkpoint cadence 2,500, and 14 hours. The limit is supported by the measured VAN 8×8 rate of 1.0722 steps/s.
- Training-relevant code and both configs are unchanged from the original arm-launch lineage; later changes are evaluation/infrastructure additions. Training at HEAD does not compromise the single-delta claim.

Sanctioned command:

```bash
SMOKE=0 DRYRUN=0 bash worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_submit.sh VANL
```

The three training-guard reds do **not** block this launch. Two assert obsolete `TO-PIN-AFTER-P0` states; the third searches the launcher itself for `wandb-metadata.json`, although the launcher invokes the tested `fa_orbit_wandb_readback.py` at [fa_orbit_train.sbatch:605](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_train.sbatch:605). They belong to the cleanup round.

## Required evaluation fixes

1. **The sanctioned screen submitter currently rejects VANL.** Its ARM whitelist omits VANL at [fa_orbit_screen_submit.sh:90](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_submit.sh:90). The driver supports VANL, but no production evaluation can reach it through the required wrapper.

2. **The VANL table globs also match sidecars.** The `K1*.json`/`K8*.json` patterns at [gen_model_comparison.py:86](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/gen_model_comparison.py:86) match both each metrics JSON and its `.json.screenmeta.json`. The generator will therefore pass ten files per five-seed cell to the validator and render the row BLOCKED. Make the metric suffix exact.

3. **The claimed mandatory vanilla schema is slightly fail-open.** The validator accepts a missing `frame_avg_fwd_cap` and missing/empty `frame_avg_angles` because protocol fields bypass the mandatory loop and are checked through `.get()`/truthiness at [exp11_validate_rows.py:383](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/exp11_validate_rows.py:383). Require both record keys to exist and be exactly `None`; require the sidecar angle field to be exactly `None`.

4. **Pin parking needs interruption safety.** The guard suite clears the live pin at [fa_orbit_screen_guardtests.sh:589](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:589) and restores it only on the normal path at [line 893](/n/fs/gatrdp/codespace/FLAC/worklog/worklog_yixun/exp_11_fa_orbit_claude/fa_orbit_screen_guardtests.sh:893). Add trap-backed restoration and do not run this suite during active campaign submissions. The current pin was verified restored to `0c6e9ff`.

The absence-proving checkpoint gate, vanilla output schema, `orbit=None`, `orbit_execution="n/a"`, and cross refusal are otherwise correct. Refusing VANL R3 is acceptable as the current registered scope, although yaw-offset sensitivity is scientifically meaningful for vanilla; it should be described as unregistered, not meaningless.

## Evaluation-pin prescription

Keep the original campaign pinned to `0c6e9ff` until C32’s entire conf block has completed and validated.

Then create a **separate Q9 measurement round** pinned to one new reviewed SHA, `Q`, containing:

- the actual VANL launch-registry entry;
- the submitter whitelist fix;
- exact table globs;
- strict vanilla-schema validation;
- corresponding guards.

Do **not** pin VANL evaluations to `9ce96a3`: it lacks the launch-registry entry and the fixes above. The registry entry should be made only after the INITIAL job has started and published `outputs_FLAC/exp11_VANL/launch_manifest.txt`; `sbatch` return alone is insufficient.

For the primary fa-vs-vanilla claim, evaluate **both VANL and C4L at `Q`**, using a registered collision-free Q9 conf namespace so the existing `0c6e9ff` C4L evidence is preserved rather than overwritten. Seeds 42–46 and both K values must share `Q`. This keeps the new comparison genuinely same-pin and single-delta without retroactively moving or mixing the original four-arm campaign.
