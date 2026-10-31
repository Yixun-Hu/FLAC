**Reviewer:** OpenAI Codex (gpt-5.6-sol, xhigh, codex exec, read-only sandbox) · **Date:** 2026-10-24 · **Round:** code rA (r3b-F1 re-verification + round 3d auto-launch watcher)

**Verdict — PART 1:** B1 **NOT RESOLVED**; B2 **PARTIAL**; H3 **RESOLVED**; H4 **RESOLVED**; H5 **PARTIAL**; H6 **RESOLVED**; H7 **PARTIAL**; NITs 8–10 **RESOLVED**.

**Verdict — PART 2:** **REQUEST-CHANGES**

**Verdict — PART 3:** TDD/commit/static evidence is substantially compliant, but launch remains blocked by the findings below.

## Findings

### BLOCKING

1. **B1 remains bypassable: the audit report is self-authenticating, unpinned, and structurally under-validated.**

   The launcher reads `depth_canon_tie_audit.json` and its sidecar at [depth_canon_launch.sh:236](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:236). It requires:

   - exact sidecar text;
   - schema 1;
   - gate `PASS`, rc 0, zero survivors/errors;
   - exact three-split/full-coverage declaration;
   - registered item counts;
   - current dataset-config, split-JSON, `AR_md.py`, operator, dataset-loader and audit-script hashes;
   - a 9,650-element `rows` list with no truthy `error` or `survivor`.

   However, [depth_canon_launch.sh:285](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:285) does not validate row paths, uniqueness, recorded depth-map hashes, counts, `img_w`, tie classes, or the dtype/shape contract. Nor does it re-hash current depth-map bytes. A depth map changed after the audit therefore remains accepted.

   More decisively, the guard suite constructs a completely synthetic report with arbitrary `m0…m9649` rows at [depth_canon_guardtests.sh:59](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:59), recomputes its sidecar, and demonstrates that it qualifies at [depth_canon_guardtests.sh:375](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_guardtests.sh:375). The audit has no harness marker. Moving the same shape of synthetic artifact to the production path would pass.

   The audit and sidecar are also absent from the 46-file pin definition at [depth_canon_make_pins.sh:23](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_make_pins.sh:23). During review, the newly generated full-audit artifacts appeared as untracked files; the launcher would not reject their modification if the sidecar were recomputed.

   **Fix:** after the full audit, commit and HEAD-bind the exact report and sidecar—or add their reviewed digest to a committed pin. Require production/non-harness provenance, type-strict fields, 9,650 unique expected map paths, row hashes tied to those paths, and either revalidate current map hashes or explicitly pin the audited data snapshot. Add cases where duplicated/arbitrary rows and a harness-generated audit with a recomputed sidecar are rejected.

### HIGH

2. **H5: the DINO closure still omits a numerically active file.**

   The two missing BV/BF configs are now pinned and the manifest correctly contains 46 entries. But `assert_vit_pin()` hashes only `model.safetensors` at [assert_arm_configs.py:70](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_07_fa_scratch_claude/assert_arm_configs.py:70), while production constructs DINO with `AutoModel.from_pretrained` at [conditioners.py:458](/home/yixunhu/codespace/exp-25-depth-canon/src/models/conditioners.py:458), which also consumes the snapshot’s mutable `config.json`.

   A compatible edit such as dropout or layer-norm epsilon can preserve parameter shapes and weight loading while changing the run. The current cached `config.json` SHA-256 is `9481247be9f95a134a5599402b4bfc838eecdf9a7fffbf4debd1c70ec213898b`.

   **Fix:** hash-pin `config.json`—preferably every regular file consumed from the sole snapshot—in `assert_vit_pin()`, with a synthetic-cache regression test.

3. **H7/W2 do not strictly implement “any `nvidia-smi` failure fails closed.”**

   The normal paths correctly map UUIDs and detect processes on GPUs 0/1 without signaling them. But the launcher masks memory and UUID-query exit statuses at [depth_canon_launch.sh:385](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:385) and [depth_canon_launch.sh:394](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:394). A command that emits plausible partial output and then exits nonzero can pass. Malformed nonempty compute rows with an absent/unknown UUID are ignored at [depth_canon_launch.sh:404](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:404).

   The watcher similarly masks the memory-query pipeline status at [depth_canon_autolaunch.sh:161](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:161). Its `fail` test fails the earlier UUID query, so this memory-specific branch is not covered.

   **Fix:** capture status and output separately, reject any nonzero status even if stdout is numeric, require exactly one numeric memory row per queried GPU, and reject malformed compute-app rows. Add partial-output-plus-nonzero tests.

4. **W5b: inherited `MB`/`ACC` can turn the unattended launch window into a terminal failure.**

   `run_launcher` pins only `MODE`, `EXPECT_HEAD`, and `DRY_RUN` at [depth_canon_autolaunch.sh:167](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:167). An inherited `MB` or `ACC` cannot produce a wrong run—the launcher refuses it—but that rc 2 is classified as a final non-GPU SMOKE failure. After approximately 30 hours of waiting, a stray shell variable can therefore lose the launch window.

   This also contradicts the now-prescriptive v10 text at [plan_depth_canon.md:105](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/plan_depth_canon.md:105).

   **Fix:** invoke with literal `MB=32 ACC=1 DRY_RUN=0`; in production also clear harness-only state. Make the fake launcher record/assert MB and ACC.

### NIT

5. **B2 latency excludes part of the function it claims to measure.**

   `_canonicalize_batch` unwraps the carrier and validates every metadata entry at [diffusion.py:642](/home/yixunhu/codespace/exp-25-depth-canon/src/training/diffusion.py:642), but starts `perf_counter()` only afterward at [diffusion.py:646](/home/yixunhu/codespace/exp-25-depth-canon/src/training/diffusion.py:646). Thus the measurement covers the per-sample operator loop, not the whole `_canonicalize_batch` treatment overhead.

   **Fix:** start the timer before carrier unwrapping and schema validation. The present measurement is still monotonic, per-rank, and otherwise usable.

6. **Watcher hardening/test gaps.**

   - Its interpreter check is a textual glob, `*/envs/flac/bin/python*`, at [depth_canon_autolaunch.sh:75](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:75), rather than the launcher’s exact resolved path.
   - The token accepts blank lines and validates `issued` only by a date prefix. These cannot admit a wrong head or manifest, but the claimed format is slightly looser than “exact four lines.”
   - Production override tests cover three of five names; `AUTOLAUNCH_STABLE_POLLS` and `AUTOLAUNCH_MAX_POLLS` lack individual refusal tests.
   - There is no watcher-side clean-tree/self check. The launcher protects training correctness, but a dirty pinned path becomes a terminal failure instead of `WAITING_TOKEN`.
   - The transient classification duplicates launcher text at [depth_canon_autolaunch.sh:44](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_autolaunch.sh:44). It is byte-compatible today, but not a cross-file single source of truth. A dedicated transient rc or machine-readable marker would be safer.

## PART 1 adjudication

- **B1 — NOT RESOLVED.** Genuine complete reports are checked extensively, but stale depth bytes, synthetic rows, and harness/forged reports with recomputed sidecars can qualify.
- **B2 — PARTIAL.** The loss regex is `(?:^|[\s,\[])train/loss=([^,\]\s]+)` and only counts lines also matching `\| (\d+)/\d+ \[` with step ≥1 at [depth_canon_launch.sh:593](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:593). Absent, nonnumeric, NaN, or Inf losses cannot pass. Exactly one valid latency line is required at [depth_canon_launch.sh:601](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:601), and FULL requires a positive finite JSON float at [depth_canon_launch.sh:323](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:323). Disabled execution remains unaffected and the reported golden suites passed. Only the timing scope is incomplete.
- **H3 — RESOLVED.** Raw float64 `(256,512)`, CPU `torch.float64 (3,256,512)` and depth modality are enforced at [depth_canon_tie_audit.py:58](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_tie_audit.py:58); violations become row errors and `INCOMPLETE` rc 3.
- **H4 — RESOLVED.** Registered arguments and post-load EMA/CANON binding are enforced at [depth_canon_cond_probe.py:386](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_cond_probe.py:386). Diagnostics write `registered:false` and never exit 0. The collector independently requires `registered is True`, a valid sidecar, and the exact CANON@40k checkpoint SHA at [depth_canon_collect.py:660](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_collect.py:660).
- **H5 — PARTIAL.** BV/BF are pinned and all 46 manifest hashes verify; the HF `config.json` omission remains.
- **H6 — RESOLVED for the launcher.** Production requires `/home/yixunhu/miniconda3/envs/flac/bin/python`, requires PATH’s `python` to resolve to it, and uses `$PY` for all Python gates and training at [depth_canon_launch.sh:110](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:110). Ordinary aliases, wrappers, and symlinks to another interpreter fail.
- **H7 — PARTIAL.** Core occupancy behavior, UUID mapping, and no-signal policy are correct; strict partial-output failure handling needs strengthening.
- **NIT 8 — RESOLVED.** `INCOMPLETE > FAIL > REFUSED_SUBSET > PASS` at [depth_canon_tie_audit.py:343](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_tie_audit.py:343).
- **NIT 9 — RESOLVED within the declared repository closure.** Evidence binds manifest, launcher, pin generator, arm/control contract and audit digest at [depth_canon_launch.sh:295](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:295). Record-only commits can pass after updating `EXPECT_HEAD`; pinned-source changes cannot masquerade as record-only changes. The unpinned DINO config is the H5 exception.
- **NIT 10 — RESOLVED.** The `wandb/` extra write target is documented at [depth_canon_launch.sh:29](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_launch.sh:29).

## PART 2 watcher conclusions

- **W1:** Head and manifest are checked every poll and before SMOKE/FULL. Duplicate keys effectively fail because four nonempty entries plus every required field are needed; CRLF, surrounding whitespace, missing `by`, or wrong head/manifest fail. Uncommitted pinned changes do not invalidate the token itself, but the launcher refuses them. For unattended liveness, move that check into `token_ok` as well.
- **W2:** UUID mapping, both-card occupancy, ≥23,040 MiB, three consecutive polls and dirty-reset logic are correct. The current xRIR processes cannot satisfy the normal condition. A job starting after the final watcher poll but before the launcher gate is caught and retried. The partial-output failure hole is finding 3.
- **W3:** Current launcher strings match and current non-GPU failures do not print them. SMOKE’s pipeline waits for its capture `tee`, so classification sees the complete capture. FULL output is duplicated into both launcher and watcher logs for the entire run; this is unnecessary but likely modest for 40k steps. The outer watcher `tee` has no explicit status/drain check, so it is secondary evidence rather than a guaranteed durable log.
- **W4:** No `kill`/`pkill`; advisory `flock` releases automatically after a crash; direct watcher writes are confined to EXPDIR; state replacement is same-directory `tmp + mv`; JSON fields and heartbeat are present. Production overrides are refused. `setsid nohup … </dev/null &`, as recorded in the notebook, survives the Planner shell and SIGHUP; the script does not detach itself.
- **W5:** The freeze rule is sufficient and operationally necessary for one uninterrupted watcher run because the token and `EXPECT_HEAD` remain HEAD-strict. It is stricter than the content-scoped smoke binding: a record-only commit is not logically unsafe, but invalidates the token and forces reissue/re-SMOKE.
- **W5b:** Pin `MB=32 ACC=1`. Strictly rejecting an integer `canonicalization_latency_ms` is appropriate because the genuine writer serializes a float. The per-attempt SMOKE capture in EXPDIR is acceptable and intentionally untracked.
- **W6:** The 27 subprocess cases cover the named (a)–(o) branches and are generally load-bearing. Empty compute-app output and UUID CSV formatting are realistic. Missing cases are memory-query nonzero with numeric stdout, near-matching fake interpreter paths, the two untested production overrides, dirty pinned paths, and a fake launcher that independently rejects a post-check HEAD race. Runtime is acceptable; the entire 25-file receipt was 595 seconds.

## PART 3 — SOP and guard suite

- RED/GREEN evidence is recorded for every F1 fix and the watcher cycles at [depth_canon_worklog.md:929](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_worklog.md:929) and [depth_canon_worklog.md:977](/home/yixunhu/codespace/exp-25-depth-canon/worklog/worklog_yixun/exp_25_depth_canon_claude/depth_canon_worklog.md:977).
- Every scoped F1/3d commit is below 200 changed lines; the largest are 191 (`827548d`) and 189 (`929576d`).
- No environment/package modification is recorded or was performed by this review.
- Gate additions (8)–(12) have load-bearing guard sections `U`, `I`, `T`, `L`, and `B`. Section `T` also exposes the B1 weakness by accepting its synthetic report.
- The `MB`/`ACC` deviation is not acceptable for an unattended watcher and should be fixed as above.
- Process NIT: r3c began after r3b returned REQUEST-CHANGES but before r3b-F1 closed it, contrary to the no-new-round rule at [experiment_SOP.md:35](/home/yixunhu/codespace/exp-25-depth-canon/worklog/experiment_SOP.md:35). This is historical and does not itself invalidate the current code, but should be recorded as an SOP deviation.

## Verification performed

Read-only only:

- `git diff --check 84d7403 944636c` on the scoped files.
- `bash -n` on launcher, watcher, guard suite, and pin generator.
- Python `ast.parse` under `/home/yixunhu/miniconda3/envs/flac/bin/python -B`.
- Regenerated the 46-entry manifest to stdout, compared byte-for-byte, and verified all hashes.
- Checked scoped commit sizes and RED/GREEN history.
- Confirmed scoped implementation bytes remained identical from `944636c` after the branch advanced through record-only commits during review.
- Read the HF snapshot file set and confirmed `config.json` is consumed but not hash-gated.

I did not run pytest, the watcher, launcher, guard suite, audit, training/evaluation, `nvidia-smi`, NAS writes, or process-management commands. The full audit that appeared during this review was launched by another session and was untouched.

**Status:** exp_25 is awaiting fixes and re-review; SMOKE/FULL is not authorized. The xRIR jobs on GPUs 0/1 are not ours and were untouched; their recorded ETD is approximately 2026-10-25 09:00 EDT. No Yixun decision is needed before the blocking/high fixes return; A3 remains the next user decision before the evaluation campaign.